"""KRX 외국인 수급 어댑터 — 차이나머니 감지용

Phase 1: KIS API 외국인 합계 + EWY 프록시 분석
Phase 2: KRX SMILE 국적별 보유 (로그인 자동화 후)

핵심 로직:
  - KIS API → 종목별 외국인 순매수 30일치
  - EWY 수익률과 외국인 순매수 디커플링 → 비미국 외국인(중국/홍콩 프록시)
  - 섹터별 외국인 집중도 분석
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.adapters.kis_investor_adapter import (
    fetch_investor_by_ticker,
    _issue_token,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHINA_MONEY_DIR = DATA_DIR / "china_money"
CHINA_MONEY_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════

@dataclass
class ForeignFlowSnapshot:
    """종목별 외국인 수급 스냅샷"""
    ticker: str
    name: str
    date: str

    # 외국인 순매수 (수량 기준)
    foreign_net_qty_1d: int = 0       # 당일
    foreign_net_qty_5d: int = 0       # 5일 누적
    foreign_net_qty_20d: int = 0      # 20일 누적

    # 외국인 순매수 (금액, 백만원)
    foreign_net_amt_1d: int = 0
    foreign_net_amt_5d: int = 0

    # 기관 순매수 (비교용)
    inst_net_qty_1d: int = 0
    inst_net_qty_5d: int = 0

    # 통계
    foreign_avg_20d: float = 0.0      # 20일 평균 순매수
    foreign_std_20d: float = 0.0      # 20일 표준편차
    foreign_zscore: float = 0.0       # (5일누적 - 평균) / 표준편차

    # 주가 정보
    close: int = 0
    pct_change_5d: float = 0.0        # 5일 수익률

    # 연속 매수/매도 일수
    consecutive_buy_days: int = 0
    consecutive_sell_days: int = 0


@dataclass
class ChinaMoneySignal:
    """차이나머니 유입 시그널"""
    date: str
    ticker: str
    name: str

    signal: str = "NORMAL"            # SURGE / INFLOW / SECTOR_FOCUS / STEALTH / NORMAL
    score: int = 0                    # 0~100
    reasons: list = field(default_factory=list)

    # 핵심 데이터
    foreign_net_5d: int = 0           # 외국인 5일 누적 순매수
    foreign_zscore: float = 0.0       # 외국인 z-score
    ewy_decouple: bool = False        # EWY 디커플링 여부
    consecutive_days: int = 0
    pct_change_5d: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════
# 어댑터 클래스
# ═══════════════════════════════════════════════════

class KRXForeignAdapter:
    """KRX/KIS 외국인 수급 어댑터 — 차이나머니 감지"""

    # 시총 상위 대형주 (84종목 유니버스에 없을 수 있는 것들)
    LARGE_CAP_TICKERS = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "005380": "현대차",
        "373220": "LG에너지솔루션",
        "005490": "POSCO홀딩스",
        "035420": "NAVER",
        "000270": "기아",
        "068270": "셀트리온",
        "051910": "LG화학",
        "006400": "삼성SDI",
        "003670": "포스코퓨처엠",
        "207940": "삼성바이오로직스",
        "034730": "SK",
        "055550": "신한지주",
        "105560": "KB금융",
        "012330": "현대모비스",
        "066570": "LG전자",
        "032830": "삼성생명",
        "316140": "우리금융지주",
        "086790": "하나금융지주",
    }

    # 중형주 30종목 (섹터 대표)
    MID_CAP_TICKERS = {
        "009150": "삼성전기",
        "028260": "삼성물산",
        "018260": "삼성에스디에스",
        "010130": "고려아연",
        "011200": "HMM",
        "009540": "한국조선해양",
        "042700": "한미반도체",
        "267260": "HD현대일렉트릭",
        "329180": "HD현대중공업",
        "010140": "삼성중공업",
        "352820": "하이브",
        "035720": "카카오",
        "030200": "KT",
        "017670": "SK텔레콤",
        "090430": "아모레퍼시픽",
        "051900": "LG생활건강",
        "047810": "한국항공우주",
        "012450": "한화에어로스페이스",
        "000880": "한화",
        "064350": "현대로템",
        "010950": "S-Oil",
        "036570": "엔씨소프트",
        "259960": "크래프톤",
        "000100": "유한양행",
        "128940": "한미약품",
        "003490": "대한항공",
        "180640": "한진칼",
        "034020": "두산에너빌리티",
        "009830": "한화솔루션",
        "003550": "LG",
    }

    # 섹터 분류 (차이나머니 섹터 집중 감지용)
    SECTOR_MAP = {
        "반도체": ["005930", "000660", "006400", "042700", "009150"],
        "2차전지": ["373220", "051910", "003670", "009830"],
        "자동차": ["005380", "000270", "012330", "064350"],
        "바이오": ["068270", "207940", "000100", "128940"],
        "금융": ["055550", "105560", "316140", "086790"],
        "IT": ["035420", "035720", "066570"],
        "조선/방산": ["009540", "329180", "010140", "047810", "012450"],
        "에너지/화학": ["010950", "051910", "034020"],
    }

    def __init__(self, delay: float = 0.12):
        """
        Args:
            delay: API 호출 간 대기 (초). KIS rate limit 대응.
        """
        self.delay = delay

    def fetch_universe_flow(
        self,
        tickers: dict[str, str],
    ) -> list[ForeignFlowSnapshot]:
        """유니버스 전체 외국인 수급 수집.

        Args:
            tickers: {종목코드: 종목명} dict

        Returns:
            ForeignFlowSnapshot 리스트
        """
        # 토큰 미리 발급
        _issue_token()

        results = []
        total = len(tickers)

        for i, (ticker, name) in enumerate(tickers.items()):
            try:
                snap = self._fetch_single(ticker, name)
                if snap:
                    results.append(snap)
            except Exception as e:
                logger.warning("[%s] %s 수급 실패: %s", ticker, name, e)

            if i > 0 and self.delay > 0:
                time.sleep(self.delay)

            if (i + 1) % 20 == 0:
                logger.info("외국인 수급 수집: %d/%d", i + 1, total)

        logger.info("외국인 수급 수집 완료: %d/%d 성공", len(results), total)
        return results

    def _fetch_single(self, ticker: str, name: str) -> ForeignFlowSnapshot | None:
        """단일 종목 외국인 수급 스냅샷 생성."""
        df = fetch_investor_by_ticker(ticker)
        if df is None or df.empty:
            return None

        today_str = datetime.today().strftime("%Y-%m-%d")
        snap = ForeignFlowSnapshot(
            ticker=ticker,
            name=name,
            date=today_str,
        )

        # 수량 기준 순매수
        if "foreign_net_qty" in df.columns:
            fq = df["foreign_net_qty"]
            snap.foreign_net_qty_1d = int(fq.iloc[-1]) if len(fq) >= 1 else 0
            snap.foreign_net_qty_5d = int(fq.tail(5).sum()) if len(fq) >= 5 else int(fq.sum())
            snap.foreign_net_qty_20d = int(fq.tail(20).sum()) if len(fq) >= 20 else int(fq.sum())

            # 통계
            if len(fq) >= 20:
                snap.foreign_avg_20d = float(fq.tail(20).mean())
                snap.foreign_std_20d = float(fq.tail(20).std())
                if snap.foreign_std_20d > 0:
                    snap.foreign_zscore = (snap.foreign_net_qty_5d - snap.foreign_avg_20d * 5) / (snap.foreign_std_20d * np.sqrt(5))
            elif len(fq) >= 5:
                snap.foreign_avg_20d = float(fq.mean())
                snap.foreign_std_20d = float(fq.std())
                if snap.foreign_std_20d > 0:
                    snap.foreign_zscore = (snap.foreign_net_qty_5d - snap.foreign_avg_20d * min(5, len(fq))) / (snap.foreign_std_20d * np.sqrt(min(5, len(fq))))

            # 연속 매수/매도 일수
            consec_buy = 0
            consec_sell = 0
            for val in reversed(fq.values):
                if val > 0:
                    consec_buy += 1
                else:
                    break
            for val in reversed(fq.values):
                if val < 0:
                    consec_sell += 1
                else:
                    break
            snap.consecutive_buy_days = consec_buy
            snap.consecutive_sell_days = consec_sell

        # 기관 순매수
        if "inst_net_qty" in df.columns:
            iq = df["inst_net_qty"]
            snap.inst_net_qty_1d = int(iq.iloc[-1]) if len(iq) >= 1 else 0
            snap.inst_net_qty_5d = int(iq.tail(5).sum()) if len(iq) >= 5 else int(iq.sum())

        # 금액 기준 외국인
        if "외국인합계" in df.columns:
            fa = df["외국인합계"]
            snap.foreign_net_amt_1d = int(fa.iloc[-1] / 1_000_000) if len(fa) >= 1 else 0
            snap.foreign_net_amt_5d = int(fa.tail(5).sum() / 1_000_000) if len(fa) >= 5 else 0

        # 주가
        if "close" in df.columns and len(df) >= 1:
            snap.close = int(df["close"].iloc[-1])
            if len(df) >= 6:
                c5 = df["close"].iloc[-6]
                if c5 > 0:
                    snap.pct_change_5d = round((snap.close - c5) / c5 * 100, 2)

        return snap


# ═══════════════════════════════════════════════════
# 시그널 분석기
# ═══════════════════════════════════════════════════

class ChinaMoneyAnalyzer:
    """차이나머니 유입 시그널 분석기

    Phase 1 룰 (KIS API 기반):
      R1. FOREIGN_SURGE — 외국인 5일 누적 z-score > 2.0
      R2. EWY_DECOUPLE — EWY 하락 but 외국인 순매수↑ (비미국 자금)
      R3. SECTOR_FOCUS — 반도체/2차전지 섹터 외국인 집중
      R4. STEALTH_BUY — 주가 하락 + 외국인 연속 매수 (저점 매집)
      R5. DUAL_BUYING — 외국인+기관 동시 순매수
    """

    def __init__(self):
        self.ewy_data = self._load_ewy()

    def _load_ewy(self) -> dict:
        """US Overnight에서 EWY 데이터 로드."""
        path = DATA_DIR / "us_market" / "overnight_signal.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("index_direction", {}).get("EWY", {})
        except Exception:
            return {}

    def analyze(
        self,
        snapshots: list[ForeignFlowSnapshot],
    ) -> list[ChinaMoneySignal]:
        """전체 유니버스 시그널 판정."""
        signals = []
        ewy_ret_5d = self.ewy_data.get("ret_5d", 0)

        for snap in snapshots:
            sig = self._analyze_single(snap, ewy_ret_5d)
            signals.append(sig)

        # 섹터 집중도 분석
        self._check_sector_focus(signals, snapshots)

        # 점수 기준 내림차순 정렬
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def _analyze_single(
        self,
        snap: ForeignFlowSnapshot,
        ewy_ret_5d: float,
    ) -> ChinaMoneySignal:
        """단일 종목 시그널 판정."""
        sig = ChinaMoneySignal(
            date=snap.date,
            ticker=snap.ticker,
            name=snap.name,
            foreign_net_5d=snap.foreign_net_qty_5d,
            foreign_zscore=round(snap.foreign_zscore, 2),
            consecutive_days=snap.consecutive_buy_days,
            pct_change_5d=snap.pct_change_5d,
        )
        score = 0
        reasons = []

        # R1: FOREIGN_SURGE — z-score 기반
        if snap.foreign_zscore > 2.0:
            score += 30
            reasons.append(f"외국인 폭풍매수 (z={snap.foreign_zscore:.1f})")
        elif snap.foreign_zscore > 1.0:
            score += 15
            reasons.append(f"외국인 강매수 (z={snap.foreign_zscore:.1f})")

        # R2: EWY_DECOUPLE — EWY 하락인데 외국인 순매수
        if ewy_ret_5d < -3.0 and snap.foreign_net_qty_5d > 0:
            score += 20
            reasons.append(f"EWY 디커플링 (EWY {ewy_ret_5d:+.1f}% but 외국인↑)")
            sig.ewy_decouple = True
        elif ewy_ret_5d < -1.0 and snap.foreign_net_qty_5d > 0 and snap.foreign_zscore > 1.0:
            score += 10
            reasons.append(f"EWY 약디커플링 (EWY {ewy_ret_5d:+.1f}%)")
            sig.ewy_decouple = True

        # R3: 연속 순매수
        if snap.consecutive_buy_days >= 5:
            score += 15
            reasons.append(f"외국인 {snap.consecutive_buy_days}일 연속 순매수")
        elif snap.consecutive_buy_days >= 3:
            score += 8
            reasons.append(f"외국인 {snap.consecutive_buy_days}일 연속 순매수")

        # R4: STEALTH_BUY — 주가 하락 + 외국인 매수
        if snap.pct_change_5d < -3.0 and snap.foreign_net_qty_5d > 0:
            score += 10
            reasons.append(f"스텔스 매수 (주가 {snap.pct_change_5d:+.1f}% but 외국인↑)")

        # R5: DUAL_BUYING — 외국인+기관 동시
        if snap.foreign_net_qty_5d > 0 and snap.inst_net_qty_5d > 0:
            score += 10
            reasons.append("외국인+기관 동시 순매수 (5일)")

        # 시그널 등급
        if score >= 70:
            sig.signal = "SURGE"
        elif score >= 50:
            sig.signal = "INFLOW"
        elif score >= 30:
            sig.signal = "SECTOR_FOCUS"
        elif score >= 15:
            sig.signal = "WATCH"
        else:
            sig.signal = "NORMAL"

        sig.score = score
        sig.reasons = reasons
        return sig

    def _check_sector_focus(
        self,
        signals: list[ChinaMoneySignal],
        snapshots: list[ForeignFlowSnapshot],
    ):
        """섹터별 외국인 집중도 → SECTOR_FOCUS 보너스."""
        # 섹터별 외국인 5일 누적
        sector_flow = {}
        snap_map = {s.ticker: s for s in snapshots}

        for sector, tickers in KRXForeignAdapter.SECTOR_MAP.items():
            total = 0
            for t in tickers:
                if t in snap_map:
                    total += snap_map[t].foreign_net_qty_5d
            sector_flow[sector] = total

        # 반도체/2차전지 집중 매수 보너스
        focus_sectors = ["반도체", "2차전지"]
        for sector in focus_sectors:
            if sector_flow.get(sector, 0) > 0:
                for t in KRXForeignAdapter.SECTOR_MAP.get(sector, []):
                    for sig in signals:
                        if sig.ticker == t and sig.foreign_net_5d > 0:
                            sig.score += 5
                            sig.reasons.append(f"{sector} 섹터 외국인 집중")
                            if sig.signal == "NORMAL" and sig.score >= 15:
                                sig.signal = "WATCH"


# ═══════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════

def load_universe_tickers(mode: str = "large") -> dict[str, str]:
    """차이나머니 감지 유니버스 로드.

    Args:
        mode: "large" = 대형주 TOP50 (기본, ~3분)
              "full" = 전체 1000+ 종목 (~2시간, 비권장)

    Returns:
        {종목코드: 종목명}
    """
    if mode == "full":
        tickers = {}
        parquet_dir = DATA_DIR / "processed"
        csv_dir = PROJECT_ROOT / "stock_data_daily"
        for pf in parquet_dir.glob("*.parquet"):
            ticker = pf.stem
            csv_path = csv_dir / f"{ticker}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    if "종목명" in df.columns:
                        tickers[ticker] = str(df["종목명"].iloc[0])
                    else:
                        tickers[ticker] = ticker
                except Exception:
                    tickers[ticker] = ticker
            else:
                tickers[ticker] = ticker
        return tickers

    # mode == "large": 시총 상위 대형주 50종목 (차이나머니 핵심 타깃)
    # 대형주 TOP20 + 추가 중형주 30종목
    tickers = dict(KRXForeignAdapter.LARGE_CAP_TICKERS)
    tickers.update(KRXForeignAdapter.MID_CAP_TICKERS)
    return tickers


def save_signals(signals: list[ChinaMoneySignal], date_str: str | None = None):
    """시그널 저장.

    - china_money_signal.json: 최신 (오늘)
    - history/YYYY-MM-DD.json: 일별 누적
    """
    date_str = date_str or datetime.today().strftime("%Y-%m-%d")
    data = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_stocks": len(signals),
        "summary": {
            "SURGE": sum(1 for s in signals if s.signal == "SURGE"),
            "INFLOW": sum(1 for s in signals if s.signal == "INFLOW"),
            "SECTOR_FOCUS": sum(1 for s in signals if s.signal == "SECTOR_FOCUS"),
            "WATCH": sum(1 for s in signals if s.signal == "WATCH"),
            "NORMAL": sum(1 for s in signals if s.signal == "NORMAL"),
        },
        "signals": [s.to_dict() for s in signals if s.signal != "NORMAL"],
        "top_foreign_buyers": [
            s.to_dict() for s in sorted(signals, key=lambda x: x.foreign_net_5d, reverse=True)[:10]
        ],
    }

    # 최신 파일
    signal_path = CHINA_MONEY_DIR / "china_money_signal.json"
    signal_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("차이나머니 시그널 저장: %s", signal_path)

    # 히스토리
    hist_dir = CHINA_MONEY_DIR / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"{date_str}.json"
    hist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return data
