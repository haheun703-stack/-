"""FLOWX Supabase 업로드 어댑터.

퀀트봇 데이터를 FLOWX PRO 대시보드용 Supabase DB에 업로드.
담당 테이블: etf_signals, foreign_flow, paper_trades, short_signals

Usage:
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()
    uploader.upload_etf_signals(rows)
    uploader.upload_foreign_flow(rows)
    uploader.upload_ai_picks(rows)
    uploader.upload_paper_trade(trade)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class FlowxUploader:
    """FLOWX Supabase 업로드 클라이언트."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

        if not url or not key:
            logger.warning("[FLOWX] SUPABASE_URL/KEY 미설정 — 업로드 비활성")
            self.client = None
            return

        try:
            from supabase import create_client
            self.client = create_client(url, key)
            logger.info("[FLOWX] Supabase 연결 완료")
        except Exception as e:
            logger.error("[FLOWX] Supabase 연결 실패: %s", e)
            self.client = None

    @property
    def is_active(self) -> bool:
        return self.client is not None

    # ── ETF 시그널 ─────────────────────────────────

    def upload_etf_signals(self, rows: list[dict]) -> bool:
        """ETF 시그널 업로드 (UPSERT on date+code)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("etf_signals").upsert(
                rows, on_conflict="date,code"
            ).execute()
            logger.info("[FLOWX] ETF 시그널 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] ETF 시그널 업로드 실패: %s", e)
            return False

    # ── 외국인 자금 흐름 ─────────────────────────────

    def upload_foreign_flow(self, rows: list[dict]) -> bool:
        """외국인 자금 흐름 업로드 (UPSERT on date+code).

        Supabase 테이블명: china_flow (기존 테이블 재활용, 스키마 동일).
        """
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("china_flow").upsert(
                rows, on_conflict="date,code"
            ).execute()
            logger.info("[FLOWX] 외국인 자금 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 외국인 자금 업로드 실패: %s", e)
            return False

    # ── AI 추천 (short_signals) ─────────────────────

    def upload_ai_picks(self, rows: list[dict]) -> bool:
        """AI 추천 종목 업로드 (UPSERT on date+code)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("short_signals").upsert(
                rows, on_conflict="date,code"
            ).execute()
            logger.info("[FLOWX] AI 추천 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] AI 추천 업로드 실패: %s", e)
            return False

    # ── 모닝 브리핑 ──────────────────────────────────

    def upload_morning_briefing(self, briefing: dict) -> bool:
        """모닝 브리핑 업로드 (UPSERT on date)."""
        if not self.is_active or not briefing:
            return False
        try:
            import json as _json
            row = {
                "date": briefing["date"],
                "market_status": briefing.get("market_status", "NEUTRAL"),
                "us_summary": briefing.get("us_summary", ""),
                "kr_summary": briefing.get("kr_summary", ""),
                "news_picks": _json.loads(_json.dumps(briefing.get("news_picks", []))),
                "sector_focus": _json.loads(_json.dumps(briefing.get("sector_focus", []))),
            }
            result = self.client.table("morning_briefings").upsert(
                row, on_conflict="date"
            ).execute()
            logger.info("[FLOWX] 모닝 브리핑 업로드: %s", briefing["date"])
            return True
        except Exception as e:
            logger.error("[FLOWX] 모닝 브리핑 업로드 실패: %s", e)
            return False

    # ── 시그널 로깅 ──────────────────────────────────

    def insert_signal(self, signal: dict) -> bool:
        """시그널 INSERT (STEP 2/3용)."""
        if not self.is_active or not signal:
            return False
        try:
            result = self.client.table("signals").insert(signal).execute()
            logger.info("[FLOWX] 시그널 기록: %s %s %s",
                        signal.get("bot_type", "?"),
                        signal.get("signal_type", "?"),
                        signal.get("ticker_name", "?"))
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 기록 실패: %s", e)
            return False

    def update_signal_performance(self, signal_id: str, updates: dict) -> bool:
        """시그널 수익률/상태 업데이트 (STEP 2용)."""
        if not self.is_active or not signal_id:
            return False
        try:
            result = self.client.table("signals").update(updates).eq(
                "id", signal_id
            ).execute()
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 업데이트 실패: %s", e)
            return False

    def upsert_scoreboard(self, rows: list[dict]) -> bool:
        """성적표 집계 UPSERT (STEP 2용)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("scoreboard").upsert(
                rows, on_conflict="bot_type,period"
            ).execute()
            logger.info("[FLOWX] 성적표 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 성적표 업로드 실패: %s", e)
            return False

    # ── 시그널 조회 ──────────────────────────────────

    def fetch_open_signals(self, bot_type: str = "") -> list[dict]:
        """OPEN 상태 시그널 조회."""
        if not self.is_active:
            return []
        try:
            q = self.client.table("signals").select("*").eq("status", "OPEN")
            if bot_type:
                q = q.eq("bot_type", bot_type)
            result = q.execute()
            return result.data or []
        except Exception as e:
            logger.error("[FLOWX] 시그널 조회 실패: %s", e)
            return []

    def fetch_signals_by_period(
        self, bot_type: str, status_list: list[str], from_date: str
    ) -> list[dict]:
        """기간별 시그널 조회 (성적표 집계용)."""
        if not self.is_active:
            return []
        try:
            q = (
                self.client.table("signals")
                .select("*")
                .in_("status", status_list)
                .gte("signal_date", from_date)
            )
            if bot_type and bot_type != "ALL":
                q = q.eq("bot_type", bot_type)
            result = q.execute()
            return result.data or []
        except Exception as e:
            logger.error("[FLOWX] 기간별 시그널 조회 실패: %s", e)
            return []

    def close_signal(self, signal_id: str, close_data: dict) -> bool:
        """시그널 종료 (CLOSED/STOPPED)."""
        if not self.is_active or not signal_id:
            return False
        try:
            self.client.table("signals").update(close_data).eq(
                "id", signal_id
            ).execute()
            logger.info("[FLOWX] 시그널 종료: %s → %s", signal_id[:8], close_data.get("status", "?"))
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 종료 실패: %s", e)
            return False

    # ── 시나리오 대시보드 (FLOWX /quant) ─────────────

    def upload_quant_scenarios(self, scenario_data: dict, date_str: str) -> bool:
        """시나리오 대시보드 데이터 업로드 (UPSERT on date)."""
        if not self.is_active or not scenario_data:
            return False
        try:
            row = {
                "date": date_str,
                "data": scenario_data,
            }
            result = self.client.table("quant_scenario_dashboard").upsert(
                row, on_conflict="date"
            ).execute()
            sc_count = len(scenario_data.get("active_scenarios", []))
            logger.info("[FLOWX] 시나리오 대시보드 업로드: %s (%d개 시나리오)", date_str, sc_count)
            return True
        except Exception as e:
            logger.error("[FLOWX] 시나리오 대시보드 업로드 실패: %s", e)
            return False

    # ── 자비스 컨트롤타워 ─────────────────────────────

    def upload_jarvis_data(self, jarvis_data: dict, date_str: str) -> bool:
        """자비스 컨트롤타워 데이터 업로드 (UPSERT on date).

        jarvis_data 구조:
          picks: {target_date_label, mode_label, total_candidates, stats, picks[]}
          accuracy: {pullback_scan: {hit_rate, total}, ...}
          brain: {regime, vix, cash_ratio}
          shield: {status, max_drawdown}
        """
        if not self.is_active or not jarvis_data:
            return False
        try:
            row = {
                "date": date_str,
                "data": jarvis_data,
            }
            result = self.client.table("quant_jarvis").upsert(
                row, on_conflict="date"
            ).execute()
            n_picks = len(jarvis_data.get("picks", {}).get("picks", []))
            logger.info("[FLOWX] 자비스 컨트롤타워 업로드: %s (%d종목)", date_str, n_picks)
            return True
        except Exception as e:
            logger.error("[FLOWX] 자비스 업로드 실패: %s", e)
            return False

    # ── 페이퍼 트레이딩 ──────────────────────────────

    def upload_paper_trade(self, trade: dict) -> bool:
        """페이퍼 매매 기록 업로드 (INSERT, UPSERT 아님)."""
        if not self.is_active or not trade:
            return False
        try:
            result = self.client.table("paper_trades").insert(trade).execute()
            side = trade.get("side", "?")
            name = trade.get("name", "?")
            price = trade.get("price", 0)
            logger.info("[FLOWX] 매매기록: %s %s @ %s", side, name, f"{price:,}")
            return True
        except Exception as e:
            logger.error("[FLOWX] 매매기록 업로드 실패: %s", e)
            return False


# ── 데이터 변환 함수 ─────────────────────────────

def build_etf_signal_rows(date_str: str = "") -> list[dict]:
    """기존 JSON → etf_signals 테이블 포맷 변환.

    소스:
      - data/sector_rotation/sector_momentum.json (섹터 순위 + RSI)
      - data/etf_rotation_result.json (order_queue → BUY/SELL)
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    rows = []

    # 1) 섹터 모멘텀 → 20개 섹터 ETF
    sm_path = DATA_DIR / "sector_rotation" / "sector_momentum.json"
    if sm_path.exists():
        with open(sm_path, encoding="utf-8") as f:
            sm = json.load(f)

        for sec in sm.get("sectors", []):
            rows.append({
                "date": date_str,
                "code": sec.get("etf_code", ""),
                "name": sec.get("sector", ""),
                "signal": "HOLD",  # 기본값, 아래에서 order_queue로 오버라이드
                "score": round(sec.get("momentum_score", 0), 1),
                "change_1d": round(sec.get("ret_5", 0) / 5, 2) if sec.get("ret_5") else 0,
                "change_5d": round(sec.get("ret_5", 0), 2),
                "rsi": round(sec.get("rsi_14", 0), 1),
                "sector_rotation_rank": sec.get("rank", 0),
                "group_tag": None,
            })

    # 2) order_queue에서 BUY/SELL 오버라이드
    etf_path = DATA_DIR / "etf_rotation_result.json"
    if etf_path.exists():
        with open(etf_path, encoding="utf-8") as f:
            etf = json.load(f)

        code_signal_map = {}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            action = order.get("action", "").upper()
            if code and action in ("BUY", "SELL"):
                code_signal_map[code] = action

        for row in rows:
            if row["code"] in code_signal_map:
                row["signal"] = code_signal_map[row["code"]]

        # order_queue에 있지만 sector_momentum에 없는 ETF (인버스, 레버리지 등)
        existing_codes = {r["code"] for r in rows}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            if code and code not in existing_codes:
                action = order.get("action", "").upper()
                if action in ("BUY", "SELL"):
                    rows.append({
                        "date": date_str,
                        "code": code,
                        "name": order.get("name", ""),
                        "signal": action,
                        "score": 0,
                        "change_1d": 0,
                        "change_5d": 0,
                        "rsi": 0,
                        "sector_rotation_rank": 0,
                        "group_tag": None,
                    })

    return rows


def build_foreign_flow_rows(date_str: str = "") -> list[dict]:
    """국적별 외국인 수급 → foreign_flow 테이블 포맷 변환.

    소스: data/krx_nationality/nationality_signal.json
    시그널 매핑:
      - foreign_direction=="BUY" → INFLOW
      - signal=="SELL" or foreign_direction=="SELL" → OUTFLOW
      - 나머지 → NEUTRAL
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    sig_path = DATA_DIR / "krx_nationality" / "nationality_signal.json"
    if not sig_path.exists():
        return []

    with open(sig_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for sig in data.get("signals", []):
        # 시그널 매핑: 외국인 순매수/순매도 방향 기준
        foreign_dir = sig.get("foreign_direction", "NEUTRAL")
        raw_signal = sig.get("signal", "NEUTRAL")

        if foreign_dir == "BUY":
            mapped = "INFLOW"
        elif foreign_dir == "SELL" or raw_signal == "SELL":
            mapped = "OUTFLOW"
        else:
            mapped = "NEUTRAL"

        rows.append({
            "date": date_str,
            "code": sig.get("ticker", ""),
            "name": sig.get("name", ""),
            "signal": mapped,
            "score": sig.get("score", 0),
            "z_score": round(sig.get("inst_zscore", 0), 2),
            "change_5d_pct": round(sig.get("price_change_pct", 0), 2),
        })

    return rows


def build_ai_pick_rows(date_str: str = "") -> list[dict]:
    """AI 추천 종목 → short_signals 테이블 포맷 변환.

    소스: data/tomorrow_picks.json
      - ai_largecap: AI 두뇌 대형주 추천 (confidence 기반)
      - picks: 전략 종합 추천 (적극매수/매수 등급만)

    short_signals 스키마:
        date, code, name, grade, total_score, foreign_detail,
        inst_support, entry_price, stop_loss, target_price,
        holding_days, signal_type, volume_ratio, momentum_regime
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # FLOWX 모드 JSON 우선 로드 (없으면 기본 tomorrow_picks.json 폴백)
    flowx_path = DATA_DIR / "tomorrow_picks_flowx.json"
    picks_path = flowx_path if flowx_path.exists() else DATA_DIR / "tomorrow_picks.json"
    if not picks_path.exists():
        return []
    is_flowx = picks_path == flowx_path

    with open(picks_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    seen_codes: set[str] = set()

    # v6: AI 대형주 독립 섹션 제거 — 시그널 기반 picks만 사용
    # (ai_largecap은 이제 항상 빈 리스트)

    # picks → 적극매수/매수 (FLOWX: 관심매수도 포함)
    allowed_grades = {"적극매수", "매수", "관심매수"} if is_flowx else {"적극매수", "매수"}
    for pick in data.get("picks", []):
        ticker = pick.get("ticker", "")
        grade_kr = pick.get("grade", "")
        if not ticker or ticker in seen_codes:
            continue
        if grade_kr not in allowed_grades:
            continue
        seen_codes.add(ticker)

        grade_map = {"적극매수": "AA", "매수": "A", "관심매수": "B"}
        grade = grade_map.get(grade_kr, "B")
        close = pick.get("close", 0) or _get_close(ticker)
        if close <= 0:
            continue  # 종가 없으면 스킵 (stop_loss=0 방지)

        # 시나리오 필드 (scan_tomorrow_picks.py에서 생성)
        sc_tag = pick.get("scenario_tag", "")
        sc_narrative = pick.get("scenario_narrative", "")
        sc_rr = pick.get("scenario_risk_reward", {})

        row = {
            "date": date_str,
            "code": ticker,
            "name": pick.get("name", ""),
            "grade": grade,
            "total_score": round(pick.get("total_score", 0), 1),
            "foreign_detail": None,
            "inst_support": bool(pick.get("inst_5d", 0) > 0),
            "entry_price": pick.get("entry_price", close),
            "stop_loss": pick.get("stop_loss", int(close * 0.92)),
            "target_price": pick.get("target_price", int(close * 1.1)),
            "holding_days": 5,
            "signal_type": "BUY",
            "volume_ratio": round(pick.get("ret_5d", 0) / 5, 1) if pick.get("ret_5d") else 1.0,
            "momentum_regime": "QUANT",
        }
        # 시나리오 데이터가 있으면 추가 (Supabase 컬럼 있을 때만 유효)
        if sc_tag:
            row["scenario_tag"] = sc_tag
            row["narrative"] = sc_narrative[:200] if sc_narrative else ""
            rr_parts = []
            if sc_rr.get("commodity"):
                rr_parts.append(f"{sc_rr['commodity']}")
            if sc_rr.get("cost_gap_pct") is not None:
                rr_parts.append(f"갭{sc_rr['cost_gap_pct']:.0f}%")
            if sc_rr.get("zone"):
                rr_parts.append(sc_rr["zone"])
            row["risk_reward"] = " ".join(rr_parts) if rr_parts else ""
        rows.append(row)

    return rows


def build_jarvis_payload() -> dict:
    """자비스 컨트롤타워 데이터 통합 빌드.

    data/tomorrow_picks.json + brain_decision.json + shield_report.json +
    market_learning/signal_accuracy.json 통합.
    """
    def _load(name: str) -> dict:
        p = DATA_DIR / name
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    picks_raw = _load("tomorrow_picks.json")
    brain = _load("brain_decision.json")
    shield = _load("shield_report.json")

    # signal_accuracy: 최신 학습 데이터에서
    acc_raw = {}
    learn_idx = _load("market_learning/_index.json")
    latest_learn = learn_idx.get("latest", "")
    if latest_learn:
        learn = _load(f"market_learning/{latest_learn}.json")
        acc_raw = learn.get("signal_accuracy", {})

    # picks 요약 (전체 picks는 너무 크므로 관찰 이상만)
    all_picks = picks_raw.get("picks", [])
    buyable_grades = {"적극매수", "매수", "관심매수", "관찰"}
    buyable = [p for p in all_picks if p.get("grade") in buyable_grades]
    buyable.sort(key=lambda x: -x.get("total_score", 0))

    # 상위 20개만 전송 (Supabase JSONB 크기 제한)
    top_picks = buyable[:20]

    return {
        "picks": {
            "target_date_label": picks_raw.get("target_date_label", ""),
            "mode_label": picks_raw.get("mode_label", ""),
            "total_candidates": picks_raw.get("total_candidates", 0),
            "stats": picks_raw.get("stats", {}),
            "picks": top_picks,
        },
        "accuracy": acc_raw,
        "brain": {
            "regime": brain.get("regime", brain.get("direction", "NEUTRAL")),
            "vix": brain.get("vix", 0),
            "cash_ratio": brain.get("cash_ratio", brain.get("cash_pct", 0)),
            "recommendation": brain.get("recommendation", ""),
        },
        "shield": {
            "status": shield.get("status", "YELLOW"),
            "sector_concentration": shield.get("sector_concentration", 0),
            "max_drawdown": shield.get("max_drawdown", 0),
        },
    }


def _get_close(ticker: str) -> int:
    """parquet에서 최신 종가 조회."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            if len(df) > 0:
                return int(df.iloc[-1]["close"])
        except Exception:
            pass
    return 0


def build_paper_trade(
    code: str,
    name: str,
    side: str,
    price: float,
    quantity: int,
    strategy: str,
    pnl_pct: float | None = None,
    memo: str = "",
    stats: dict | None = None,
) -> dict:
    """페이퍼 매매 기록 dict 생성.

    Args:
        stats: paper_portfolio.json의 stats dict (PF/MDD/승률).
    """
    s = stats or {}
    total = s.get("total_trades", 0)
    wins = s.get("wins", 0)

    return {
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "code": code,
        "name": name,
        "side": side.upper(),
        "price": price,
        "quantity": quantity,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "strategy": strategy,
        "cumulative_pf": round(s.get("pf", 0), 2) if s.get("pf") else None,
        "cumulative_mdd": round(s.get("mdd", 0), 2) if s.get("mdd") else None,
        "win_rate": round(wins / total * 100, 1) if total > 0 else None,
        "memo": memo,
    }
