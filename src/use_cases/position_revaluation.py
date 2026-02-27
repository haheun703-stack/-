"""보유종목 동적 목표가 재판정 엔진.

7축 조정(리포트/뉴스/수급/MACD/RSI/볼린저/DART)으로 base_target을
동적으로 보정하여 HOLD/ADD/PARTIAL_SELL/FULL_SELL 판정.

의존: entities.position_models (도메인 모델만)
외부 데이터: JSON + parquet + SQLite (모두 로컬 — API 호출 없음)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.entities.position_models import (
    AdjustmentBreakdown,
    MonitorAction,
    MonitorResult,
    PositionTarget,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "jarvis_archive.db"
CFG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("JSON 로드 실패 (%s): %s", filename, e)
        return {}


class PositionRevaluationEngine:
    """보유종목 동적 목표가 재산출 엔진."""

    def __init__(self, config_path: str | Path | None = None):
        self._load_config(config_path)
        self._load_data_sources()
        self._ensure_db_tables()

    # ──────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────

    def _load_config(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else CFG_PATH
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg.get("position_monitor", {})

    def _load_data_sources(self) -> None:
        """JSON 데이터 소스 한 번에 로드."""
        # 기관 목표가
        it = _load_json("institutional_targets.json")
        self._targets = it.get("targets", {})

        # 증권사 리포트
        mr = _load_json("morning_reports.json")
        self._report_boost_map = mr.get("report_boost_map", {})
        self._reports = mr.get("reports", [])

        # 시장 인텔리전스
        mi = _load_json("market_intelligence.json")
        self._beneficiary = set(mi.get("beneficiary_stocks", []))
        self._risk_stocks = set(mi.get("risk_stocks", []))
        self._sector_boost = mi.get("sector_boost", {})

        # DART 이벤트
        dart = _load_json("dart_event_signals.json")
        self._dart_signals = {}
        for s in dart.get("signals", []):
            self._dart_signals.setdefault(s["ticker"], []).append(s)
        self._dart_avoid = {
            item["ticker"] if isinstance(item, dict) else item
            for item in dart.get("avoid_list", [])
        }

        # 시장 뉴스
        mn = _load_json("market_news.json")
        self._news_articles = mn.get("articles", [])

        # 추천 이력 (보유 중 목표가 참조)
        ph = _load_json("picks_history.json")
        self._picks_map = {}
        for r in ph.get("records", []):
            if r.get("status") == "holding":
                self._picks_map[r["ticker"]] = r

    def _ensure_db_tables(self) -> None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_targets (
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    base_target REAL,
                    report_adj REAL,
                    news_adj REAL,
                    supply_adj REAL,
                    tech_adj REAL,
                    dart_adj REAL,
                    final_target REAL,
                    action TEXT,
                    reason TEXT,
                    confidence REAL,
                    pnl_pct REAL,
                    current_price REAL,
                    PRIMARY KEY (date, ticker)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reason TEXT,
                    adj_pct REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("DB 테이블 생성 실패: %s", e)

    # ──────────────────────────────────────────
    # 메인 실행
    # ──────────────────────────────────────────

    def run(self, holdings: list[dict]) -> MonitorResult:
        now = datetime.now()
        result = MonitorResult(
            date=now.strftime("%Y-%m-%d"),
            generated_at=now.strftime("%Y-%m-%d %H:%M"),
            total_holdings=len(holdings),
        )
        summary: dict[str, int] = {}
        for h in holdings:
            try:
                pt = self.run_single(h, result.date)
                result.positions.append(pt)
                key = pt.action.value
                summary[key] = summary.get(key, 0) + 1
                result.processed += 1
            except Exception as e:
                ticker = h.get("ticker", "?")
                logger.warning("재판정 실패 (%s): %s", ticker, e)
                result.errors.append(f"{ticker}: {e}")

        result.actions_summary = summary
        self._save_to_db(result)
        return result

    def run_single(self, holding: dict, date_str: str = "") -> PositionTarget:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        ticker = holding["ticker"]
        name = holding.get("name", "")
        current_price = float(holding.get("current_price", 0))
        pnl_pct = float(holding.get("pnl_pct", 0))

        # parquet 로드
        df = self._load_parquet(ticker)

        # 1. base_target
        base_target = self._get_base_target(ticker, current_price)

        # 2. RSI (override 체크 우선)
        rsi_adj, rsi_reasons, rsi_override = self._calc_rsi_adj(df, current_price)
        if rsi_override is not None:
            base_target = rsi_override

        # 3~8. 나머지 6축 조정
        report_adj, report_reasons = self._calc_report_adj(ticker, name)
        news_adj, news_reasons = self._calc_news_adj(ticker, name)
        supply_adj, supply_reasons = self._calc_supply_adj(df)
        macd_adj, macd_reasons = self._calc_macd_adj(df)
        bb_adj, bb_reasons = self._calc_bb_adj(df)
        dart_adj, dart_reasons = self._calc_dart_adj(ticker)

        adj = AdjustmentBreakdown(
            report_adj=report_adj,
            news_adj=news_adj,
            supply_adj=supply_adj,
            macd_adj=macd_adj,
            rsi_adj=rsi_adj,
            bb_adj=bb_adj,
            dart_adj=dart_adj,
        )

        # 9. 총 조정 클램핑
        total = max(
            self.cfg.get("total_adj_min", -0.15),
            min(self.cfg.get("total_adj_max", 0.15), adj.total),
        )

        # 10. final_target
        final_target = round(base_target * (1 + total))

        # 11. 판정
        all_reasons = (
            report_reasons + news_reasons + supply_reasons
            + macd_reasons + rsi_reasons + bb_reasons + dart_reasons
        )
        # SAR 트레일링 스톱용 데이터
        sar_value, sar_trend, atr = 0.0, 0, 0.0
        if df is not None and not df.empty:
            last = df.iloc[-1]
            sar_value = float(last.get("sar", 0) or 0)
            sar_trend = int(last.get("sar_trend", 0) or 0)
            atr = float(last.get("atr_14", 0) or 0)

        action = self._determine_action(
            final_target, current_price, pnl_pct,
            sar_value=sar_value, sar_trend=sar_trend, atr=atr,
        )

        # confidence: 기관목표가 신뢰도 × 데이터 충분도
        inst_conf = self._targets.get(ticker, {}).get("confidence", 0.5)
        data_count = sum(1 for v in [report_adj, news_adj, supply_adj,
                                      macd_adj, rsi_adj, bb_adj, dart_adj] if v != 0)
        data_conf = min(data_count / 4, 1.0)
        confidence = round(inst_conf * 0.6 + data_conf * 0.4, 2)

        ratio = round(final_target / current_price, 4) if current_price > 0 else 0

        return PositionTarget(
            date=date_str,
            ticker=ticker,
            name=name,
            quantity=int(holding.get("quantity", 0)),
            avg_price=float(holding.get("avg_price", 0)),
            current_price=current_price,
            pnl_pct=pnl_pct,
            base_target=base_target,
            adjustment=adj,
            final_target=final_target,
            action=action,
            reasons=all_reasons,
            confidence=confidence,
            ratio_to_current=ratio,
        )

    # ──────────────────────────────────────────
    # 데이터 로딩
    # ──────────────────────────────────────────

    def _load_parquet(self, ticker: str) -> pd.DataFrame | None:
        path = DATA_DIR / "processed" / f"{ticker}.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    # ──────────────────────────────────────────
    # 1. 기준 목표가
    # ──────────────────────────────────────────

    def _get_base_target(self, ticker: str, current_price: float) -> float:
        # 1순위: 기관 추정 목표가
        inst = self._targets.get(ticker, {})
        if inst.get("estimated_target") and inst.get("confidence", 0) >= 0.4:
            return float(inst["estimated_target"])

        # 2순위: 추천 이력 목표가
        pick = self._picks_map.get(ticker, {})
        if pick.get("target_price"):
            return float(pick["target_price"])

        # 3순위: 현재가 × 1.10 폴백
        return round(current_price * 1.10)

    # ──────────────────────────────────────────
    # 2. RSI 조정
    # ──────────────────────────────────────────

    def _calc_rsi_adj(
        self, df: pd.DataFrame | None, current_price: float,
    ) -> tuple[float, list[str], float | None]:
        """Returns (adj_pct, reasons, override_target_or_None)."""
        if df is None or df.empty:
            return 0.0, [], None

        rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0
        cfg = self.cfg.get("rsi", {})
        extreme = cfg.get("extreme_overbought", 75)
        mult = cfg.get("override_multiplier", 1.03)

        if rsi >= extreme:
            override = round(current_price * mult)
            return 0.0, [f"RSI {rsi:.0f} 극과열→목표가 강제하향"], override
        if rsi >= cfg.get("overbought", 70):
            return cfg.get("overbought_adj", -0.02), [f"RSI {rsi:.0f} 과매수"], None
        if rsi <= cfg.get("oversold", 30):
            return cfg.get("oversold_adj", 0.03), [f"RSI {rsi:.0f} 과매도 반등기대"], None
        if rsi <= cfg.get("near_oversold", 40):
            return cfg.get("near_oversold_adj", 0.015), [f"RSI {rsi:.0f} 저가대"], None
        return 0.0, [], None

    # ──────────────────────────────────────────
    # 3. 증권사 리포트 조정
    # ──────────────────────────────────────────

    def _calc_report_adj(self, ticker: str, name: str) -> tuple[float, list[str]]:
        cfg = self.cfg.get("report", {})
        boost_info = self._report_boost_map.get(ticker, {})
        if not boost_info:
            return 0.0, []

        boost = boost_info.get("boost", 0)
        tag = boost_info.get("tag", "")
        broker = boost_info.get("broker", "")

        if boost >= 5:
            # boost 8 = 목표가상향+매수, boost 5 = 매수
            if boost >= 8:
                adj = cfg.get("strong_buy_adj", 0.04)
            else:
                adj = cfg.get("buy_adj", 0.03)
            return adj, [f"{broker} {tag} +{adj*100:.0f}%"]
        elif boost <= -5:
            adj = cfg.get("reduce_adj", -0.05)
            return adj, [f"{broker} {tag} {adj*100:.0f}%"]

        return 0.0, []

    # ──────────────────────────────────────────
    # 4. 뉴스 임팩트 조정
    # ──────────────────────────────────────────

    def _calc_news_adj(self, ticker: str, name: str) -> tuple[float, list[str]]:
        cfg = self.cfg.get("news", {})
        adj = 0.0
        reasons: list[str] = []

        # 뉴스 기사 종목명 매칭 (3글자 이상만)
        if name and len(name) >= 3:
            pos_kw = cfg.get("positive_keywords", [])
            neg_kw = cfg.get("negative_keywords", [])

            for art in self._news_articles:
                title = art.get("title", "")
                if name not in title:
                    continue
                impact = art.get("impact", "low")

                is_pos = any(kw in title for kw in pos_kw)
                is_neg = any(kw in title for kw in neg_kw)

                if impact == "high":
                    if is_pos:
                        adj += cfg.get("high_positive_adj", 0.025)
                        reasons.append(f"뉴스호재: {title[:20]}")
                    elif is_neg:
                        adj += cfg.get("high_negative_adj", -0.04)
                        reasons.append(f"뉴스악재: {title[:20]}")
                elif impact == "medium":
                    scale = cfg.get("medium_scale", 0.6)
                    if is_pos:
                        adj += cfg.get("high_positive_adj", 0.025) * scale
                    elif is_neg:
                        adj += cfg.get("high_negative_adj", -0.04) * scale
                break  # 첫 매칭만

        # market_intelligence 수혜/위험
        if name in self._beneficiary:
            adj += cfg.get("beneficiary_adj", 0.02)
            reasons.append("인텔리전스 수혜종목")
        if name in self._risk_stocks:
            adj += cfg.get("risk_adj", -0.03)
            reasons.append("인텔리전스 주의종목")

        return round(adj, 4), reasons

    # ──────────────────────────────────────────
    # 5. 수급 조정
    # ──────────────────────────────────────────

    def _calc_supply_adj(self, df: pd.DataFrame | None) -> tuple[float, list[str]]:
        if df is None or df.empty:
            return 0.0, []

        cfg = self.cfg.get("supply", {})
        adj = 0.0
        reasons: list[str] = []

        # 외인 연속 매수 (parquet 컬럼)
        f_streak = 0
        if "foreign_consecutive_buy" in df.columns:
            f_streak = int(df["foreign_consecutive_buy"].iloc[-1])

        # 기관 연속 매수 (inst_net_streak 또는 기관합계에서 계산)
        i_streak = 0
        if "inst_net_streak" in df.columns:
            i_streak = int(df["inst_net_streak"].iloc[-1])
        elif "기관합계" in df.columns:
            inst = df["기관합계"].tail(5).values
            i_streak = self._calc_streak(inst)

        if f_streak >= 3 and i_streak >= 3:
            adj = cfg.get("dual_buy_3d_adj", 0.02)
            reasons.append(f"외인{f_streak}일+기관{i_streak}일 동시순매수")
        elif f_streak >= 3:
            adj = cfg.get("foreign_only_3d_adj", 0.015)
            reasons.append(f"외인 {f_streak}일 연속순매수")
        elif i_streak >= 3:
            adj = cfg.get("inst_only_3d_adj", 0.01)
            reasons.append(f"기관 {i_streak}일 연속순매수")
        elif f_streak <= -3 and i_streak <= -3:
            adj = cfg.get("dual_sell_3d_adj", -0.02)
            reasons.append(f"외인{f_streak}일+기관{i_streak}일 동시순매도")
        elif f_streak <= -3:
            adj = cfg.get("foreign_sell_3d_adj", -0.015)
            reasons.append(f"외인 {abs(f_streak)}일 연속순매도")

        return round(adj, 4), reasons

    @staticmethod
    def _calc_streak(values: np.ndarray) -> int:
        """최근부터 연속 양/음 일수 계산."""
        if len(values) == 0:
            return 0
        streak = 0
        sign = 1 if values[-1] > 0 else -1
        for v in reversed(values):
            if (v > 0 and sign > 0) or (v < 0 and sign < 0):
                streak += sign
            else:
                break
        return streak

    # ──────────────────────────────────────────
    # 6. MACD 조정
    # ──────────────────────────────────────────

    def _calc_macd_adj(self, df: pd.DataFrame | None) -> tuple[float, list[str]]:
        if df is None or df.empty:
            return 0.0, []

        cfg = self.cfg.get("macd", {})
        adj = 0.0
        reasons: list[str] = []

        if "macd_histogram" not in df.columns:
            return 0.0, []

        hist_now = float(df["macd_histogram"].iloc[-1])
        hist_prev = float(df["macd_histogram_prev"].iloc[-1]) if "macd_histogram_prev" in df.columns else float(df["macd_histogram"].iloc[-2]) if len(df) > 1 else 0

        # 골든크로스: 전일 음 → 당일 양
        if hist_prev < 0 and hist_now > 0:
            adj += cfg.get("golden_cross_adj", 0.02)
            reasons.append("MACD 골든크로스")
        # 데드크로스: 전일 양 → 당일 음
        elif hist_prev > 0 and hist_now < 0:
            adj += cfg.get("dead_cross_adj", -0.03)
            reasons.append("MACD 데드크로스")

        # 다이버전스 감지 (20일)
        lookback = cfg.get("divergence_lookback", 20)
        if len(df) >= lookback and "close" in df.columns:
            recent = df.tail(lookback)
            closes = recent["close"].values
            macds = recent["macd_histogram"].values

            price_peak = int(np.argmax(closes))
            macd_peak = int(np.argmax(macds))

            # Bearish: 가격 고점이 MACD 고점보다 최근 + MACD 하강 중
            if price_peak > macd_peak and macds[-1] < macds[-2] and hist_now > 0:
                adj += cfg.get("bearish_divergence_adj", -0.05)
                reasons.append("MACD 약세 다이버전스")

            # Bullish: 가격 저점이 MACD 저점보다 최근 + MACD 상승 중
            price_trough = int(np.argmin(closes))
            macd_trough = int(np.argmin(macds))
            if price_trough > macd_trough and macds[-1] > macds[-2] and hist_now < 0:
                adj += cfg.get("bullish_divergence_adj", 0.02)
                reasons.append("MACD 강세 다이버전스")

        return round(adj, 4), reasons

    # ──────────────────────────────────────────
    # 7. 볼린저밴드 조정
    # ──────────────────────────────────────────

    def _calc_bb_adj(self, df: pd.DataFrame | None) -> tuple[float, list[str]]:
        if df is None or df.empty or "bb_position" not in df.columns:
            return 0.0, []

        cfg = self.cfg.get("bollinger", {})
        bb_pos = float(df["bb_position"].iloc[-1])

        if bb_pos > cfg.get("upper_breach", 100):
            return cfg.get("upper_breach_adj", 0.01), [f"볼린저 상단돌파 ({bb_pos:.0f}%)"]
        if bb_pos < cfg.get("lower_breach", 5):
            return cfg.get("lower_breach_adj", -0.05), [f"볼린저 하단이탈 ({bb_pos:.0f}%)"]
        return 0.0, []

    # ──────────────────────────────────────────
    # 8. DART 이벤트 조정
    # ──────────────────────────────────────────

    def _calc_dart_adj(self, ticker: str) -> tuple[float, list[str]]:
        cfg = self.cfg.get("dart", {})

        # avoid_list 최우선
        if ticker in self._dart_avoid:
            return cfg.get("avoid_adj", -0.10), ["DART AVOID 종목"]

        signals = self._dart_signals.get(ticker, [])
        if not signals:
            return 0.0, []

        adj = 0.0
        reasons: list[str] = []
        for s in signals:
            tier = s.get("tier", "")
            action = s.get("action", "")
            event = s.get("event", "")
            if tier == "tier1_즉시" and action == "BUY":
                adj += cfg.get("tier1_buy_adj", 0.03)
                reasons.append(f"DART {event}")
            elif "tier2" in tier:
                adj += cfg.get("tier2_adj", 0.01)
            break  # 첫 매칭만

        return round(adj, 4), reasons

    # ──────────────────────────────────────────
    # 판정
    # ──────────────────────────────────────────

    def _determine_action(
        self, final_target: float, current_price: float, pnl_pct: float,
        sar_value: float = 0, sar_trend: int = 0, atr: float = 0,
    ) -> MonitorAction:
        # 보조 안전장치 (최우선)
        hard_stop = self.cfg.get("hard_stop_loss_pct", -8.0)
        profit_take = self.cfg.get("profit_take_pct", 12.0)

        if pnl_pct <= hard_stop:
            return MonitorAction.FULL_SELL

        # SAR 트레일링 스톱 (SAR 하향 전환 시)
        sar_cfg = self.cfg.get("sar_trailing", {})
        if sar_cfg.get("enabled", False) and sar_trend == -1 and sar_value > 0 and current_price > 0:
            buffer = atr * sar_cfg.get("buffer_atr_mult", 0.3)
            if current_price < sar_value - buffer:
                return MonitorAction.FULL_SELL
            elif current_price < sar_value:
                return MonitorAction.PARTIAL_SELL

        if current_price <= 0:
            return MonitorAction.HOLD

        ratio = final_target / current_price

        if pnl_pct >= profit_take:
            # 이미 충분히 올랐으면 최소 PARTIAL_SELL
            if ratio >= self.cfg.get("add_threshold", 1.08):
                return MonitorAction.HOLD  # 업사이드 많으면 홀딩
            return MonitorAction.PARTIAL_SELL

        if ratio >= self.cfg.get("add_threshold", 1.08):
            return MonitorAction.ADD
        if ratio >= self.cfg.get("hold_threshold", 1.03):
            return MonitorAction.HOLD
        if ratio >= self.cfg.get("partial_sell_threshold", 1.01):
            return MonitorAction.PARTIAL_SELL
        return MonitorAction.FULL_SELL

    # ──────────────────────────────────────────
    # SQLite 저장
    # ──────────────────────────────────────────

    def _save_to_db(self, result: MonitorResult) -> None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            for pt in result.positions:
                adj = pt.adjustment
                tech_adj = round(adj.macd_adj + adj.rsi_adj + adj.bb_adj, 4)
                reason_str = " | ".join(pt.reasons[:5])
                conn.execute(
                    """INSERT OR REPLACE INTO position_targets
                       (date, ticker, name, base_target, report_adj, news_adj,
                        supply_adj, tech_adj, dart_adj, final_target,
                        action, reason, confidence, pnl_pct, current_price)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        pt.date, pt.ticker, pt.name, pt.base_target,
                        adj.report_adj, adj.news_adj, adj.supply_adj,
                        tech_adj, adj.dart_adj, pt.final_target,
                        pt.action.value, reason_str, pt.confidence,
                        pt.pnl_pct, pt.current_price,
                    ),
                )
                # 개별 조정 이력
                for source, val in [
                    ("report", adj.report_adj), ("news", adj.news_adj),
                    ("supply", adj.supply_adj), ("macd", adj.macd_adj),
                    ("rsi", adj.rsi_adj), ("bb", adj.bb_adj),
                    ("dart", adj.dart_adj),
                ]:
                    if val != 0:
                        conn.execute(
                            """INSERT INTO position_adjustments
                               (date, ticker, source, reason, adj_pct)
                               VALUES (?,?,?,?,?)""",
                            (pt.date, pt.ticker, source, "", val),
                        )
            conn.commit()
            conn.close()
            logger.info("DB 저장 완료: %d건", len(result.positions))
        except Exception as e:
            logger.warning("DB 저장 실패: %s", e)
