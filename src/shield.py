"""SHIELD -- 포트폴리오 방어 시스템

BRAIN 전에 실행되어 전체 포트폴리오 상태를 점검하고,
결과를 shield_report.json으로 저장한다.
BRAIN은 이 리포트를 읽어 ARM 비중 추가 보정에 반영한다.

3대 방어축:
  S1: 섹터 오버랩 감지 (스윙+ETF 섹터 중복)
  S2: 포트폴리오 MDD 관리 (실시간 drawdown + 킬스위치)
  S3: 종목별 손절 통합 (개별 경고 + 시스템적 리스크)

Usage:
    from src.shield import Shield
    shield = Shield()
    report = shield.check()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SHIELD_OUTPUT_PATH = DATA_DIR / "shield_report.json"


# ================================================================
# 데이터 모델
# ================================================================

@dataclass
class SectorOverlap:
    """섹터 오버랩 감지 결과."""
    sector: str
    swing_holdings: list[str]       # 해당 섹터의 개별주 보유 (이름(코드))
    etf_holdings: list[str]         # 해당 섹터의 ETF 보유
    total_exposure_pct: float       # 전체 포트폴리오 대비 노출도 (%)
    limit_pct: float                # 허용 한도 (%)
    severity: str                   # OK / WARNING / DANGER

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "swing_holdings": self.swing_holdings,
            "etf_holdings": self.etf_holdings,
            "total_exposure_pct": self.total_exposure_pct,
            "limit_pct": self.limit_pct,
            "severity": self.severity,
        }


@dataclass
class MddStatus:
    """MDD 추적 상태."""
    peak_equity: float
    current_equity: float
    current_mdd_pct: float          # 현재 drawdown (음수)
    max_mdd_pct: float              # 역대 최대 drawdown
    max_mdd_date: str
    killswitch_level: str           # NONE / LEVEL_1 / LEVEL_2 / LEVEL_3
    killswitch_action: str          # 해당 레벨의 조치

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_mdd_pct": self.current_mdd_pct,
            "max_mdd_pct": self.max_mdd_pct,
            "max_mdd_date": self.max_mdd_date,
            "killswitch_level": self.killswitch_level,
            "killswitch_action": self.killswitch_action,
        }


@dataclass
class StockAlert:
    """개별 종목 손절 경고."""
    ticker: str
    name: str
    pnl_pct: float
    alert_type: str                 # DEEP_LOSS
    message: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "pnl_pct": self.pnl_pct,
            "alert_type": self.alert_type,
            "message": self.message,
        }


@dataclass
class SystemicRisk:
    """시스템적 리스크 감지."""
    simultaneous_decline_count: int     # 동시 하락 종목 수
    decline_tickers: list[str]          # 하락 종목 목록
    avg_decline_pct: float              # 평균 하락폭
    severity: str                       # OK / WARNING / DANGER
    message: str

    def to_dict(self) -> dict:
        return {
            "simultaneous_decline_count": self.simultaneous_decline_count,
            "decline_tickers": self.decline_tickers,
            "avg_decline_pct": self.avg_decline_pct,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class CorrelationBreakdown:
    """교차자산 상관관계 붕괴 감지 (S4)."""
    pair_name: str               # "gold_spy", "dollar_spy" 등
    current_corr: float          # 현재 60일 상관계수
    normal_sign: str             # "negative" 또는 "positive"
    breakdown_threshold: float   # 붕괴 판정 기준
    is_breakdown: bool           # 붕괴 여부
    severity: str                # OK / WARNING / DANGER

    def to_dict(self) -> dict:
        return {
            "pair_name": self.pair_name,
            "current_corr": self.current_corr,
            "normal_sign": self.normal_sign,
            "breakdown_threshold": self.breakdown_threshold,
            "is_breakdown": self.is_breakdown,
            "severity": self.severity,
        }


@dataclass
class ShieldReport:
    """SHIELD 전체 리포트."""
    timestamp: str
    overall_level: str                  # GREEN / YELLOW / ORANGE / RED
    sector_overlaps: list[SectorOverlap]
    mdd_status: MddStatus
    stock_alerts: list[StockAlert]
    systemic_risk: SystemicRisk
    correlation_breakdowns: list[CorrelationBreakdown]
    brain_overrides: dict               # BRAIN에 전달할 보정 지시
    warnings: list[str]
    telegram_message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_level": self.overall_level,
            "sector_overlaps": [s.to_dict() for s in self.sector_overlaps],
            "mdd_status": self.mdd_status.to_dict(),
            "stock_alerts": [a.to_dict() for a in self.stock_alerts],
            "systemic_risk": self.systemic_risk.to_dict(),
            "correlation_breakdowns": [c.to_dict() for c in self.correlation_breakdowns],
            "brain_overrides": self.brain_overrides,
            "warnings": self.warnings,
        }


# ================================================================
# SHIELD 메인 엔진
# ================================================================

# 종목 → 섹터 하드코딩 매핑 (빈번히 보유하는 종목)
KNOWN_STOCK_SECTORS: dict[str, str] = {
    "003570": "방산",          # SNT다이내믹스
    "005380": "자동차",        # 현대차
    "005930": "반도체",        # 삼성전자
    "000660": "반도체",        # SK하이닉스
    "010140": "조선",          # 삼성중공업
    "010950": "에너지화학",    # S-Oil
    "024060": "에너지화학",    # 흥구석유
    "064350": "방산",          # 현대로템
    "068270": "바이오",        # 셀트리온
    "323410": "금융",          # 카카오뱅크
    "009540": "조선",          # 한국조선해양
    "329180": "조선",          # HD현대중공업
    "042660": "반도체",        # 대우조선해양→한화오션(코드주의)
    "006800": "에너지화학",    # 미래에셋증권→(대한유화)
    "009830": "조선",          # 한화솔루션→(한화오션)
    "042700": "방산",          # 한화에어로스페이스
    "012450": "조선",          # 한화에너지→(한화해양)
    "207940": "바이오",        # 삼성바이오로직스
    "373220": "배터리",        # LG에너지솔루션
    "034730": "IT",            # SK
    "055550": "금융",          # 신한지주
    "105560": "금융",          # KB금융
}

# 키워드 기반 섹터 매핑 (종목명에 포함된 단어)
SECTOR_KEYWORDS: dict[str, str] = {
    "반도체": "반도체",
    "하이닉스": "반도체",
    "자동차": "자동차",
    "현대차": "자동차",
    "기아": "자동차",
    "조선": "조선",
    "해양": "조선",
    "중공업": "조선",
    "방산": "방산",
    "에어로": "방산",
    "한화시스템": "방산",
    "바이오": "바이오",
    "셀트리온": "바이오",
    "제약": "바이오",
    "은행": "금융",
    "금융": "금융",
    "증권": "금융",
    "보험": "금융",
    "석유": "에너지화학",
    "에너지": "에너지화학",
    "화학": "에너지화학",
    "정유": "에너지화학",
    "건설": "건설",
    "철강": "철강",
    "포스코": "철강",
    "배터리": "배터리",
    "2차전지": "배터리",
    "게임": "게임",
    "엔씨": "게임",
    "크래프톤": "게임",
    "소프트": "IT",
    "카카오": "IT",
    "네이버": "IT",
}


class Shield:
    """포트폴리오 방어 시스템.

    BRAIN.compute() 전에 실행되어 포트폴리오 전체 상태를 점검.
    결과를 shield_report.json으로 저장하고, BRAIN이 읽어 보정에 반영.
    """

    def __init__(self, settings: dict | None = None):
        if settings is None:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
        self.settings = settings
        self.shield_cfg = settings.get("shield", {})
        self.risk_cfg = settings.get("etf_rotation", {}).get("risk", {})

    # ────────────────────────────────────────
    # 메인 엔트리포인트
    # ────────────────────────────────────────
    def check(self) -> ShieldReport:
        """전체 SHIELD 점검 실행."""
        warnings: list[str] = []

        # ── 1. 데이터 로드 ──
        kis_balance = self._load_json(DATA_DIR / "kis_balance.json")
        etf_result = self._load_json(DATA_DIR / "etf_rotation_result.json")
        equity_tracker = self._load_json(DATA_DIR / "equity_tracker.json")

        holdings = kis_balance.get("holdings", [])
        total_eval = kis_balance.get("total_eval", 0)
        available_cash = kis_balance.get("available_cash", 0)
        portfolio_value = total_eval + available_cash

        if portfolio_value <= 0:
            warnings.append("포트폴리오 가치 0 이하 -- 잔고 확인 필요")
            return self._empty_report(warnings)

        # ── 2. S1: 섹터 오버랩 감지 ──
        overlaps = self._check_sector_overlap(holdings, etf_result, portfolio_value)

        # ── 3. S2: MDD 관리 ──
        mdd_status = self._check_mdd(portfolio_value, equity_tracker)

        # ── 3.5. 킬스위치 자동 회복 ──
        recovery = self._check_recovery(mdd_status)
        if recovery["triggered"]:
            mdd_status.killswitch_level = "NONE"
            mdd_status.killswitch_action = ""
            warnings.append(recovery["message"])

        # ── 4. S3: 종목별 손절 통합 ──
        stock_alerts = self._check_stock_alerts(holdings)
        systemic_risk = self._check_systemic_risk(holdings)

        # ── 4.5. S4: 교차자산 상관관계 붕괴 감지 ──
        correlation_breakdowns = self._check_correlation_breakdown()

        # ── 5. 종합 위험 등급 ──
        overall_level = self._determine_overall_level(
            overlaps, mdd_status, stock_alerts, systemic_risk,
            correlation_breakdowns
        )

        # 킬스위치 회복 시 최소 ORANGE 보장 (GREEN 직행 방지)
        if recovery["triggered"] and overall_level == "GREEN":
            overall_level = "ORANGE"

        # ── 6. BRAIN 보정 지시 생성 ──
        brain_overrides = self._build_brain_overrides(
            overlaps, mdd_status, stock_alerts, systemic_risk,
            correlation_breakdowns, overall_level
        )

        # 회복 트리거 정보를 brain_overrides에 기록
        if recovery["triggered"]:
            brain_overrides["recovery_triggered"] = True
            brain_overrides["recovery_reason"] = recovery["message"]

        # ── 7. 텔레그램 메시지 ──
        telegram_msg = self._build_telegram_message(
            overall_level, overlaps, mdd_status, stock_alerts, systemic_risk,
            correlation_breakdowns, portfolio_value
        )
        if recovery["triggered"]:
            telegram_msg = f"⚡ SHIELD 완화: LEVEL_3 → ORANGE (반등 신호 감지)\n\n{telegram_msg}"

        report = ShieldReport(
            timestamp=datetime.now().isoformat(),
            overall_level=overall_level,
            sector_overlaps=overlaps,
            mdd_status=mdd_status,
            stock_alerts=stock_alerts,
            systemic_risk=systemic_risk,
            correlation_breakdowns=correlation_breakdowns,
            brain_overrides=brain_overrides,
            warnings=warnings,
            telegram_message=telegram_msg,
        )

        # ── 8. 저장 ──
        self._save_report(report)

        # ── 9. equity_tracker 업데이트 ──
        if self.shield_cfg.get("equity_tracker_auto_update", True):
            self._update_equity_tracker(portfolio_value, equity_tracker)

        return report

    # ────────────────────────────────────────
    # S1: 섹터 오버랩 감지
    # ────────────────────────────────────────
    def _classify_stock_sector(self, ticker: str, name: str) -> str | None:
        """개별종목 → 섹터 매핑."""
        ticker = str(ticker).zfill(6)
        if ticker in KNOWN_STOCK_SECTORS:
            return KNOWN_STOCK_SECTORS[ticker]

        for keyword, sector in SECTOR_KEYWORDS.items():
            if keyword in name:
                return sector

        return None

    def _get_etf_sectors_held(self, etf_result: dict) -> dict[str, float]:
        """ETF 로테이션 결과에서 보유 중인 섹터 ETF 비중 추출."""
        sectors: dict[str, float] = {}
        sector_result = etf_result.get("sector_result", {})

        for pos in sector_result.get("current_positions", []):
            sector_name = pos.get("sector", "")
            weight = pos.get("weight_pct", 0)
            if sector_name:
                sectors[sector_name] = sectors.get(sector_name, 0) + weight

        return sectors

    def _check_sector_overlap(
        self,
        holdings: list[dict],
        etf_result: dict,
        portfolio_value: float,
    ) -> list[SectorOverlap]:
        """스윙(개별주) + ETF 섹터 오버랩 감지."""
        max_sector_pct = self.shield_cfg.get("max_sector_exposure_pct", 30)

        # 개별주 → 섹터별 그룹핑
        swing_sectors: dict[str, list[dict]] = {}
        for h in holdings:
            sector = self._classify_stock_sector(h.get("ticker", ""), h.get("name", ""))
            if not sector:
                continue
            exposure_pct = (h.get("eval_amount", 0) / portfolio_value * 100) if portfolio_value > 0 else 0
            swing_sectors.setdefault(sector, []).append({
                "ticker": h["ticker"],
                "name": h.get("name", h["ticker"]),
                "exposure_pct": round(exposure_pct, 1),
            })

        # ETF 섹터 보유 현황
        etf_sectors = self._get_etf_sectors_held(etf_result)

        # 오버랩 감지
        overlaps: list[SectorOverlap] = []
        all_sectors = set(swing_sectors.keys()) | set(etf_sectors.keys())

        for sector in sorted(all_sectors):
            swing_items = swing_sectors.get(sector, [])
            swing_total = sum(item["exposure_pct"] for item in swing_items)
            etf_total = etf_sectors.get(sector, 0)
            total = swing_total + etf_total

            # 한도 초과이거나, 스윙+ETF 양쪽에 동시 보유(중복)인 경우
            is_overlap = len(swing_items) > 0 and etf_total > 0

            if total > max_sector_pct or is_overlap:
                if total > max_sector_pct * 1.5:
                    severity = "DANGER"
                else:
                    severity = "WARNING"

                overlaps.append(SectorOverlap(
                    sector=sector,
                    swing_holdings=[f"{s['name']}({s['ticker']}, {s['exposure_pct']:.1f}%)" for s in swing_items],
                    etf_holdings=[f"ETF {sector} {etf_total:.1f}%"] if etf_total > 0 else [],
                    total_exposure_pct=round(total, 1),
                    limit_pct=max_sector_pct,
                    severity=severity,
                ))

            # 동일 섹터 개별주 2개+ (ETF 없어도 집중도 경고)
            elif len(swing_items) >= 2 and swing_total > max_sector_pct * 0.6:
                overlaps.append(SectorOverlap(
                    sector=sector,
                    swing_holdings=[f"{s['name']}({s['ticker']}, {s['exposure_pct']:.1f}%)" for s in swing_items],
                    etf_holdings=[],
                    total_exposure_pct=round(swing_total, 1),
                    limit_pct=max_sector_pct,
                    severity="WARNING",
                ))

        return overlaps

    # ────────────────────────────────────────
    # S2: MDD 관리
    # ────────────────────────────────────────
    def _check_mdd(self, portfolio_value: float, equity_tracker: dict) -> MddStatus:
        """실시간 MDD 계산 + 킬스위치 레벨 판정."""
        peak = equity_tracker.get("peak_equity", portfolio_value)

        # peak 갱신
        if portfolio_value > peak:
            peak = portfolio_value

        # drawdown 계산
        current_mdd_pct = ((portfolio_value - peak) / peak * 100) if peak > 0 else 0.0

        # 역대 최대 MDD
        max_mdd_pct = equity_tracker.get("max_mdd_pct", 0.0)
        max_mdd_date = equity_tracker.get("max_mdd_date", "")
        if current_mdd_pct < max_mdd_pct:
            max_mdd_pct = current_mdd_pct
            max_mdd_date = datetime.now().strftime("%Y-%m-%d")

        # 킬스위치 레벨 판정
        killswitch_levels = self.risk_cfg.get("killswitch_levels", [])
        ks_level = "NONE"
        ks_action = ""

        for i, level in enumerate(killswitch_levels):
            threshold = level.get("drawdown_pct", -999)
            if current_mdd_pct <= threshold:
                ks_level = f"LEVEL_{i + 1}"
                ks_action = level.get("desc", level.get("action", ""))

        return MddStatus(
            peak_equity=round(peak),
            current_equity=round(portfolio_value),
            current_mdd_pct=round(current_mdd_pct, 2),
            max_mdd_pct=round(max_mdd_pct, 2),
            max_mdd_date=max_mdd_date,
            killswitch_level=ks_level,
            killswitch_action=ks_action,
        )

    def _update_equity_tracker(self, portfolio_value: float, equity_tracker: dict):
        """equity_tracker.json 실시간 업데이트."""
        today = datetime.now().strftime("%Y-%m-%d")

        peak = max(equity_tracker.get("peak_equity", 0), portfolio_value)
        mdd_pct = ((portfolio_value - peak) / peak * 100) if peak > 0 else 0.0

        max_mdd = equity_tracker.get("max_mdd_pct", 0.0)
        max_mdd_date = equity_tracker.get("max_mdd_date", "")
        if mdd_pct < max_mdd:
            max_mdd = mdd_pct
            max_mdd_date = today

        # daily_log upsert
        daily_log = equity_tracker.get("daily_log", [])
        entry = {"date": today, "equity": round(portfolio_value), "mdd_pct": round(mdd_pct, 2)}
        if daily_log and daily_log[-1].get("date") == today:
            daily_log[-1] = entry
        else:
            daily_log.append(entry)
        daily_log = daily_log[-180:]  # 180일 보관

        updated = {
            "initial_capital": equity_tracker.get("initial_capital", round(portfolio_value)),
            "peak_equity": round(peak),
            "peak_date": today if portfolio_value >= peak else equity_tracker.get("peak_date", today),
            "current_equity": round(portfolio_value),
            "current_mdd_pct": round(mdd_pct, 2),
            "max_mdd_pct": round(max_mdd, 2),
            "max_mdd_date": max_mdd_date,
            "daily_log": daily_log,
            "alert_history": equity_tracker.get("alert_history", []),
        }

        tracker_path = DATA_DIR / "equity_tracker.json"
        tracker_path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("equity_tracker 업데이트: MDD %.2f%%", mdd_pct)

    # ────────────────────────────────────────
    # S3: 종목별 손절 통합
    # ────────────────────────────────────────
    def _check_stock_alerts(self, holdings: list[dict]) -> list[StockAlert]:
        """개별 종목 손절 경고."""
        deep_loss_threshold = self.shield_cfg.get("deep_loss_pct", -15.0)
        alerts: list[StockAlert] = []

        for h in holdings:
            pnl_pct = h.get("pnl_pct", 0)
            if pnl_pct <= deep_loss_threshold:
                alerts.append(StockAlert(
                    ticker=h.get("ticker", ""),
                    name=h.get("name", ""),
                    pnl_pct=pnl_pct,
                    alert_type="DEEP_LOSS",
                    message=f"{h.get('name', '')}({h.get('ticker', '')}) {pnl_pct:+.1f}%",
                ))

        return alerts

    def _check_systemic_risk(self, holdings: list[dict]) -> SystemicRisk:
        """동시 하락 감지."""
        decline_threshold = self.shield_cfg.get("simultaneous_decline_pct", -5.0)
        min_count = self.shield_cfg.get("systemic_min_count", 3)

        declining: list[str] = []
        pnls: list[float] = []
        for h in holdings:
            pnl_pct = h.get("pnl_pct", 0)
            if pnl_pct <= decline_threshold:
                declining.append(f"{h.get('name', '')}({pnl_pct:+.1f}%)")
                pnls.append(pnl_pct)

        count = len(declining)
        avg_decline = sum(pnls) / len(pnls) if pnls else 0

        if count >= min_count:
            severity = "DANGER"
            message = f"{count}개 종목 동시 {decline_threshold}%+ 하락"
        elif count >= 2:
            severity = "WARNING"
            message = f"{count}개 종목 동시 하락 중"
        else:
            severity = "OK"
            message = "동시 하락 없음"

        return SystemicRisk(
            simultaneous_decline_count=count,
            decline_tickers=declining,
            avg_decline_pct=round(avg_decline, 2),
            severity=severity,
            message=message,
        )

    # ────────────────────────────────────────
    # S4: 교차자산 상관관계 붕괴 감지
    # ────────────────────────────────────────
    def _check_correlation_breakdown(self) -> list[CorrelationBreakdown]:
        """us_daily.parquet에서 60일 롤링 상관관계 읽어 붕괴 감지."""
        corr_cfg = self.shield_cfg.get("correlation", {})
        if not corr_cfg.get("enabled", False):
            return []

        parquet_path = PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet"
        if pd is None or not parquet_path.exists():
            return []

        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            logger.warning("S4: parquet 읽기 실패")
            return []

        if df.empty:
            return []

        latest = df.iloc[-1]
        pairs_cfg = corr_cfg.get("pairs", {})

        # 컬럼 매핑: pair_name → parquet 컬럼
        corr_columns = {
            "gold_spy": "corr_gold_spy_60d",
            "dollar_spy": "corr_dollar_spy_60d",
            "bond_spy": "corr_bond_spy_60d",
            "oil_spy": "corr_oil_spy_60d",
        }

        breakdowns: list[CorrelationBreakdown] = []

        for pair_name, col_name in corr_columns.items():
            pair_cfg = pairs_cfg.get(pair_name, {})
            if not pair_cfg:
                continue

            corr_val = None
            try:
                v = latest.get(col_name) if hasattr(latest, "get") else latest[col_name]
                if v is not None:
                    import math
                    f = float(v)
                    if not math.isnan(f) and not math.isinf(f):
                        corr_val = f
            except (KeyError, IndexError, ValueError, TypeError):
                pass

            if corr_val is None:
                continue

            normal_sign = pair_cfg.get("normal_sign", "negative")
            threshold = pair_cfg.get("breakdown_threshold", 0.3)

            # 붕괴 판정
            if normal_sign == "negative":
                # 정상: 음의 상관 → 양으로 전환되면 붕괴
                is_breakdown = corr_val > threshold
            else:
                # 정상: 양의 상관 → 음으로 전환되면 붕괴
                is_breakdown = corr_val < threshold

            if is_breakdown:
                severity = "DANGER"
            elif normal_sign == "negative" and corr_val > threshold * 0.7:
                severity = "WARNING"
            elif normal_sign == "positive" and corr_val < threshold * 0.7:
                severity = "WARNING"
            else:
                severity = "OK"

            breakdowns.append(CorrelationBreakdown(
                pair_name=pair_name,
                current_corr=round(corr_val, 4),
                normal_sign=normal_sign,
                breakdown_threshold=threshold,
                is_breakdown=is_breakdown,
                severity=severity,
            ))

        breakdown_count = sum(1 for b in breakdowns if b.is_breakdown)
        if breakdown_count > 0:
            logger.warning("S4: %d개 상관관계 붕괴 감지", breakdown_count)

        return breakdowns

    # ────────────────────────────────────────
    # S5: 킬스위치 자동 회복
    # ────────────────────────────────────────
    def _check_recovery(self, mdd: MddStatus) -> dict:
        """킬스위치 LEVEL_3 상태에서 반등 신호 감지 시 자동 해제.

        조건 (2개 동시 충족):
          1. KOSPI 3일 누적 수익률 >= +3%
          2. VIX 전일 대비 -10% 이상 하락

        Returns:
            {"triggered": bool, "message": str, "kospi_3d": float, "vix_chg": float}
        """
        result = {"triggered": False, "message": "", "kospi_3d": 0.0, "vix_chg": 0.0}

        # 킬스위치가 LEVEL_3이 아니면 해당 없음
        if mdd.killswitch_level != "LEVEL_3":
            return result

        recovery_cfg = self.shield_cfg.get("recovery", {})
        kospi_threshold = recovery_cfg.get("kospi_3d_pct", 3.0)
        vix_threshold = recovery_cfg.get("vix_drop_pct", -10.0)

        # ── 조건1: KOSPI 3일 누적 수익률 ──
        kospi_3d = self._get_kospi_3d_return()
        result["kospi_3d"] = kospi_3d

        # ── 조건2: VIX 전일비 변화율 ──
        vix_chg = self._get_vix_daily_change()
        result["vix_chg"] = vix_chg

        cond1 = kospi_3d >= kospi_threshold
        cond2 = vix_chg <= vix_threshold

        if cond1 and cond2:
            result["triggered"] = True
            result["message"] = (
                f"킬스위치 회복: KOSPI 3일 {kospi_3d:+.1f}% (≥{kospi_threshold}%) "
                f"+ VIX {vix_chg:+.1f}% (≤{vix_threshold}%)"
            )
            logger.info("SHIELD 킬스위치 회복 트리거: %s", result["message"])
        else:
            logger.info(
                "SHIELD 회복 미충족: KOSPI 3일 %+.1f%% (%s), VIX %+.1f%% (%s)",
                kospi_3d, "OK" if cond1 else "미달",
                vix_chg, "OK" if cond2 else "미달",
            )

        return result

    def _get_kospi_3d_return(self) -> float:
        """KOSPI 최근 3일 누적 수익률 (%)."""
        try:
            if pd is None:
                return 0.0
            csv_path = DATA_DIR / "kospi_index.csv"
            if not csv_path.exists():
                return 0.0
            df = pd.read_csv(csv_path)
            if len(df) < 4:
                return 0.0
            closes = df["close"].dropna()
            if len(closes) < 4:
                return 0.0
            return float((closes.iloc[-1] / closes.iloc[-4] - 1) * 100)
        except Exception as e:
            logger.warning("KOSPI 3일 수익률 계산 실패: %s", e)
            return 0.0

    def _get_vix_daily_change(self) -> float:
        """VIX 전일 대비 변화율 (%)."""
        try:
            if pd is None:
                return 0.0
            parquet_path = DATA_DIR / "us_market" / "us_daily.parquet"
            if not parquet_path.exists():
                return 0.0
            df = pd.read_parquet(parquet_path, columns=["vix_close"])
            vix = df["vix_close"].dropna()
            if len(vix) < 2:
                return 0.0
            return float((vix.iloc[-1] / vix.iloc[-2] - 1) * 100)
        except Exception as e:
            logger.warning("VIX 변화율 계산 실패: %s", e)
            return 0.0

    # ────────────────────────────────────────
    # 종합 판정
    # ────────────────────────────────────────
    def _determine_overall_level(
        self,
        overlaps: list[SectorOverlap],
        mdd: MddStatus,
        alerts: list[StockAlert],
        systemic: SystemicRisk,
        corr_breakdowns: list[CorrelationBreakdown] | None = None,
    ) -> str:
        """GREEN / YELLOW / ORANGE / RED 결정."""
        danger_count = 0
        warning_count = 0

        for o in overlaps:
            if o.severity == "DANGER":
                danger_count += 1
            elif o.severity == "WARNING":
                warning_count += 1

        if mdd.killswitch_level != "NONE":
            return "RED"

        if systemic.severity == "DANGER":
            danger_count += 1
        elif systemic.severity == "WARNING":
            warning_count += 1

        warning_count += len(alerts)

        # S4: 상관관계 붕괴
        danger_threshold = self.shield_cfg.get("correlation", {}).get("danger_count", 3)
        if corr_breakdowns:
            bd_count = sum(1 for b in corr_breakdowns if b.is_breakdown)
            if bd_count >= danger_threshold:
                danger_count += 1  # 유동성 위기
            elif bd_count >= 2:
                warning_count += 1

        if danger_count >= 2:
            return "RED"
        if danger_count >= 1:
            return "ORANGE"
        if warning_count >= 3:
            return "ORANGE"
        if warning_count >= 1:
            return "YELLOW"
        return "GREEN"

    def _build_brain_overrides(
        self,
        overlaps: list[SectorOverlap],
        mdd: MddStatus,
        alerts: list[StockAlert],
        systemic: SystemicRisk,
        corr_breakdowns: list[CorrelationBreakdown] | None = None,
        overall_level: str = "GREEN",
    ) -> dict:
        """BRAIN에 전달할 보정 지시."""
        overrides: dict = {
            "shield_level": overall_level,
            "arm_adjustments": {},
            "frozen_sectors": [],
            "messages": [],
        }

        # S1: 오버랩 섹터 → ETF 섹터 ARM 축소
        for o in overlaps:
            if o.severity in ("WARNING", "DANGER"):
                overrides["frozen_sectors"].append(o.sector)
                excess = max(0, o.total_exposure_pct - o.limit_pct)
                if excess > 0:
                    adj = overrides["arm_adjustments"]
                    adj["etf_sector"] = adj.get("etf_sector", 0) - excess
                    adj["cash"] = adj.get("cash", 0) + excess
                overrides["messages"].append(
                    f"S1: {o.sector} {o.total_exposure_pct:.1f}% (한도 {o.limit_pct}%)"
                )

        # S2: MDD 킬스위치
        if mdd.killswitch_level == "LEVEL_3":
            overrides["arm_adjustments"] = {
                "swing": -100, "etf_sector": -100,
                "etf_leverage": -100, "etf_index": -100, "cash": 100,
            }
            overrides["messages"].append(
                f"S2: LEVEL_3 (MDD {mdd.current_mdd_pct:+.1f}%) 전량 현금화"
            )
        elif mdd.killswitch_level == "LEVEL_2":
            adj = overrides["arm_adjustments"]
            adj["etf_sector"] = adj.get("etf_sector", 0) - 100
            adj["etf_leverage"] = adj.get("etf_leverage", 0) - 100
            adj["cash"] = adj.get("cash", 0) + 30
            overrides["messages"].append(
                f"S2: LEVEL_2 (MDD {mdd.current_mdd_pct:+.1f}%) ETF 청산"
            )
        elif mdd.killswitch_level == "LEVEL_1":
            adj = overrides["arm_adjustments"]
            adj["etf_leverage"] = adj.get("etf_leverage", 0) - 100
            adj["cash"] = adj.get("cash", 0) + 15
            overrides["messages"].append(
                f"S2: LEVEL_1 (MDD {mdd.current_mdd_pct:+.1f}%) 레버리지 청산"
            )

        # S3: 시스템적 리스크
        if systemic.severity == "DANGER":
            adj = overrides["arm_adjustments"]
            adj["swing"] = adj.get("swing", 0) - 10
            adj["etf_sector"] = adj.get("etf_sector", 0) - 5
            adj["cash"] = adj.get("cash", 0) + 15
            overrides["messages"].append(
                f"S3: {systemic.simultaneous_decline_count}개 동시하락 (avg {systemic.avg_decline_pct:+.1f}%)"
            )

        # S4: 상관관계 붕괴
        if corr_breakdowns:
            danger_threshold = self.shield_cfg.get("correlation", {}).get("danger_count", 3)
            bd_count = sum(1 for b in corr_breakdowns if b.is_breakdown)
            bd_pairs = [b.pair_name for b in corr_breakdowns if b.is_breakdown]
            if bd_count >= danger_threshold:
                adj = overrides["arm_adjustments"]
                adj["etf_leverage"] = adj.get("etf_leverage", 0) - 100
                adj["etf_sector"] = adj.get("etf_sector", 0) - 50
                adj["cash"] = adj.get("cash", 0) + 50
                overrides["messages"].append(
                    f"S4: 유동성 위기 ({bd_count}개 상관붕괴: {', '.join(bd_pairs)})"
                )
            elif bd_count >= 2:
                overrides["messages"].append(
                    f"S4: 상관붕괴 경고 ({bd_count}개: {', '.join(bd_pairs)})"
                )

        return overrides

    def _build_telegram_message(
        self,
        overall_level: str,
        overlaps: list[SectorOverlap],
        mdd: MddStatus,
        alerts: list[StockAlert],
        systemic: SystemicRisk,
        corr_breakdowns: list[CorrelationBreakdown] | None = None,
        portfolio_value: float = 0,
    ) -> str:
        """텔레그램 경보 메시지."""
        emoji_map = {"GREEN": "G", "YELLOW": "Y", "ORANGE": "O", "RED": "R"}
        level_tag = emoji_map.get(overall_level, "?")

        lines = [f"[SHIELD {level_tag}] {overall_level}"]
        lines.append(f"Portfolio: {portfolio_value/1e4:,.0f}만원")
        lines.append("")

        # MDD
        lines.append(f"MDD: {mdd.current_mdd_pct:+.2f}% (peak {mdd.peak_equity/1e4:,.0f}만)")
        if mdd.killswitch_level != "NONE":
            lines.append(f"  KILLSWITCH {mdd.killswitch_level}: {mdd.killswitch_action}")
        lines.append("")

        # 섹터 오버랩
        if overlaps:
            lines.append("Sector Overlap:")
            for o in overlaps:
                lines.append(f"  {o.sector}: {o.total_exposure_pct:.1f}%/{o.limit_pct}% [{o.severity}]")
                for sh in o.swing_holdings:
                    lines.append(f"    {sh}")
            lines.append("")

        # 종목 경고
        if alerts:
            lines.append("Stock Alerts:")
            for a in alerts:
                lines.append(f"  {a.message}")
            lines.append("")

        # 시스템적 리스크
        if systemic.severity != "OK":
            lines.append(f"Systemic: {systemic.message}")
            for t in systemic.decline_tickers:
                lines.append(f"  {t}")
            lines.append("")

        # 상관관계 붕괴
        if corr_breakdowns:
            bd_items = [b for b in corr_breakdowns if b.is_breakdown]
            if bd_items:
                lines.append(f"Correlation Breakdown ({len(bd_items)}):")
                for b in bd_items:
                    lines.append(f"  {b.pair_name}: {b.current_corr:+.3f} (정상: {b.normal_sign})")

        return "\n".join(lines)

    # ────────────────────────────────────────
    # 유틸리티
    # ────────────────────────────────────────
    def _save_report(self, report: ShieldReport):
        """shield_report.json 저장."""
        SHIELD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SHIELD_OUTPUT_PATH.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("SHIELD 리포트 저장: %s", SHIELD_OUTPUT_PATH)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _empty_report(self, warnings: list[str]) -> ShieldReport:
        """데이터 부재 시 기본 리포트."""
        return ShieldReport(
            timestamp=datetime.now().isoformat(),
            overall_level="YELLOW",
            sector_overlaps=[],
            mdd_status=MddStatus(0, 0, 0, 0, "", "NONE", ""),
            stock_alerts=[],
            systemic_risk=SystemicRisk(0, [], 0, "OK", "데이터 부족"),
            correlation_breakdowns=[],
            brain_overrides={
                "shield_level": "YELLOW",
                "arm_adjustments": {},
                "frozen_sectors": [],
                "messages": ["데이터 부족"],
            },
            warnings=warnings,
            telegram_message="[SHIELD Y] 데이터 부족",
        )
