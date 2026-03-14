"""추천 종목 딥다이브 분석 — 수급 + 급등패턴 + 이격 + 종합판정.

tomorrow_picks TOP5에 대해 자동 실행되어 텔레그램/HTML로 전달.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ─── 결과 데이터 ─────────────────────────────────

@dataclass
class SupplyDemand:
    """수급 정밀 분석 결과."""
    foreign_5d: float = 0.0           # 외인 5일 순매수 (원)
    foreign_20d: float = 0.0          # 외인 20일 순매수
    inst_5d: float = 0.0              # 기관 5일 순매수
    inst_20d: float = 0.0             # 기관 20일 순매수
    foreign_streak: int = 0           # 외인 연속매수 일수
    inst_streak: int = 0              # 기관 연속매수 일수
    foreign_vol_confirm: bool = False  # 외인 거래량 확인
    inst_vol_confirm: bool = False     # 기관 거래량 확인
    obv_5d_pct: float = 0.0           # OBV 5일 변화율
    obv_10d_pct: float = 0.0          # OBV 10일 변화율
    vol_ratio: float = 0.0            # 거래량비 (5일/20일)
    pension_5d: float = 0.0           # 연금 5일 순매수
    grade: str = ""                   # ★ 등급

    def to_dict(self) -> dict:
        return {
            "foreign_5d": self.foreign_5d,
            "foreign_20d": self.foreign_20d,
            "inst_5d": self.inst_5d,
            "inst_20d": self.inst_20d,
            "foreign_streak": self.foreign_streak,
            "inst_streak": self.inst_streak,
            "foreign_vol_confirm": self.foreign_vol_confirm,
            "inst_vol_confirm": self.inst_vol_confirm,
            "obv_5d_pct": self.obv_5d_pct,
            "obv_10d_pct": self.obv_10d_pct,
            "vol_ratio": self.vol_ratio,
            "pension_5d": self.pension_5d,
            "grade": self.grade,
        }


@dataclass
class SurgePattern:
    """급등 후 패턴 분석 결과."""
    total_surges: int = 0              # 총 급등 횟수
    continue_count: int = 0            # D+5 추가상승 횟수
    fade_count: int = 0                # D+5 하락 (날고끝) 횟수
    continue_rate: float = 0.0         # 추세지속 비율
    avg_d5_return: float = 0.0         # D+5 평균 수익률
    latest_surge_date: str = ""        # 최근 급등일
    latest_surge_d5: float = 0.0       # 최근 급등 후 D+5 수익률
    verdict: str = ""                  # "추세지속형" / "날고끝형" / "혼합"

    def to_dict(self) -> dict:
        return {
            "total_surges": self.total_surges,
            "continue_count": self.continue_count,
            "fade_count": self.fade_count,
            "continue_rate": self.continue_rate,
            "avg_d5_return": self.avg_d5_return,
            "latest_surge_date": self.latest_surge_date,
            "latest_surge_d5": self.latest_surge_d5,
            "verdict": self.verdict,
        }


@dataclass
class DeepDiveResult:
    """종목 딥다이브 종합 결과."""
    ticker: str = ""
    name: str = ""
    close: int = 0
    supply: SupplyDemand = field(default_factory=SupplyDemand)
    surge: SurgePattern = field(default_factory=SurgePattern)
    ma5_gap_pct: float = 0.0          # MA5 대비 이격도
    ma7_gap_pct: float = 0.0          # MA7 대비 이격도
    entry_timing: str = ""            # "진입적기" / "눌림대기" / "이격과대"
    verdict: str = ""                 # 종합판정
    score: int = 0                    # 딥다이브 점수 (0~100)
    safety: dict = field(default_factory=dict)  # 안전마진 판정

    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "name": self.name,
            "close": self.close,
            "supply": self.supply.to_dict(),
            "surge": self.surge.to_dict(),
            "ma5_gap_pct": self.ma5_gap_pct,
            "ma7_gap_pct": self.ma7_gap_pct,
            "entry_timing": self.entry_timing,
            "verdict": self.verdict,
            "score": self.score,
            "safety": self.safety,
        }
        return d


# ─── 분석 함수 ─────────────────────────────────

def analyze_supply_demand(df: pd.DataFrame) -> SupplyDemand:
    """수급 정밀 분석."""
    sd = SupplyDemand()
    if len(df) < 5:
        return sd

    latest = df.iloc[-1]
    last5 = df.tail(5)
    last10 = df.tail(10)
    last20 = df.tail(20)

    # 외인/기관 누적
    sd.foreign_5d = float(latest.get("foreign_net_5d", 0) or 0)
    sd.foreign_20d = float(latest.get("foreign_net_20d", 0) or 0)
    sd.inst_5d = float(latest.get("inst_net_5d", 0) or 0)
    sd.inst_20d = float(latest.get("inst_net_20d", 0) or 0)

    # 연속매수
    sd.foreign_streak = int(latest.get("foreign_consecutive_buy", 0) or 0)
    sd.inst_streak = int(latest.get("inst_consecutive_buy", 0) or 0)

    # 거래량 확인
    sd.foreign_vol_confirm = bool(latest.get("foreign_vol_confirm", False))
    sd.inst_vol_confirm = bool(latest.get("inst_vol_confirm", False))

    # 연금
    sd.pension_5d = float(latest.get("pension_net_5d", 0) or 0)

    # OBV 추세
    if "obv" in df.columns:
        obv_5 = last5["obv"].values
        if abs(obv_5[0]) > 0:
            sd.obv_5d_pct = (obv_5[-1] - obv_5[0]) / abs(obv_5[0]) * 100
        obv_10 = last10["obv"].values
        if abs(obv_10[0]) > 0:
            sd.obv_10d_pct = (obv_10[-1] - obv_10[0]) / abs(obv_10[0]) * 100

    # 거래량비
    if "volume" in df.columns:
        vol_5 = last5["volume"].mean()
        vol_20 = last20["volume"].mean()
        if vol_20 > 0:
            sd.vol_ratio = vol_5 / vol_20

    # 등급 산정
    stars = 0
    if sd.foreign_streak >= 3:
        stars += 1
    if sd.inst_streak >= 3:
        stars += 1
    if sd.foreign_5d > 0 and sd.inst_5d > 0:
        stars += 1  # 동시매수
    if sd.foreign_vol_confirm or sd.inst_vol_confirm:
        stars += 1
    if sd.obv_5d_pct > 2.0:
        stars += 1

    sd.grade = "★" * max(1, min(5, stars))
    return sd


def analyze_surge_pattern(df: pd.DataFrame, threshold: float = 5.0) -> SurgePattern:
    """급등(+5% 이상) 후 D+5 패턴 분석. 최근 6개월 데이터 사용."""
    sp = SurgePattern()

    # 최근 6개월만 (≈120거래일)
    recent = df.tail(130)
    if len(recent) < 20:
        return sp

    daily_ret = recent["close"].pct_change() * 100
    surge_mask = daily_ret >= threshold
    surge_indices = [i for i, v in enumerate(surge_mask.values) if v]

    if not surge_indices:
        sp.verdict = "급등없음"
        return sp

    d5_returns = []
    for si in surge_indices:
        actual_pos = recent.index.get_loc(recent.index[si]) if si < len(recent) else si
        # D+5 위치 계산 (원본 df 기준)
        df_pos = df.index.get_loc(recent.index[si])
        if df_pos + 5 >= len(df):
            continue

        surge_close = df.iloc[df_pos]["close"]
        d5_close = df.iloc[df_pos + 5]["close"]
        d5_ret = (d5_close / surge_close - 1) * 100
        d5_returns.append({
            "date": str(recent.index[si].date()),
            "d5_ret": d5_ret,
        })

    sp.total_surges = len(surge_indices)
    if not d5_returns:
        sp.verdict = "데이터부족"
        return sp

    sp.continue_count = sum(1 for d in d5_returns if d["d5_ret"] > 0)
    sp.fade_count = sum(1 for d in d5_returns if d["d5_ret"] <= 0)
    sp.continue_rate = sp.continue_count / len(d5_returns) * 100
    sp.avg_d5_return = np.mean([d["d5_ret"] for d in d5_returns])

    # 최근 급등
    sp.latest_surge_date = d5_returns[-1]["date"]
    sp.latest_surge_d5 = d5_returns[-1]["d5_ret"]

    # 판정
    if sp.continue_rate >= 60:
        sp.verdict = "추세지속형"
    elif sp.continue_rate <= 40:
        sp.verdict = "날고끝형"
    else:
        sp.verdict = "혼합"

    return sp


def analyze_entry_timing(df: pd.DataFrame) -> tuple[float, float, str]:
    """MA5/7 이격도 + 진입 타이밍 판정."""
    if len(df) < 20:
        return 0.0, 0.0, "데이터부족"

    latest = df.iloc[-1]
    close = latest["close"]

    ma5 = df["close"].tail(5).mean()
    ma7 = df["close"].tail(7).mean()

    gap5 = (close / ma5 - 1) * 100
    gap7 = (close / ma7 - 1) * 100

    if abs(gap5) <= 2.0:
        timing = "진입적기"
    elif gap5 > 4.0:
        timing = "이격과대"
    elif gap5 < -3.0:
        timing = "하향이탈"
    else:
        timing = "눌림대기"

    return gap5, gap7, timing


# ─── 메인 딥다이브 ─────────────────────────────────

def deep_dive(ticker: str, name: str = "") -> DeepDiveResult | None:
    """단일 종목 딥다이브 분석."""
    parquet_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        logger.warning("[딥다이브] %s parquet 없음", ticker)
        return None

    df = pd.read_parquet(parquet_path)
    df = df.sort_index()

    if len(df) < 20:
        return None

    result = DeepDiveResult()
    result.ticker = ticker
    result.name = name
    result.close = int(df.iloc[-1]["close"])

    # 1. 수급 정밀
    result.supply = analyze_supply_demand(df)

    # 2. 급등 패턴
    result.surge = analyze_surge_pattern(df)

    # 3. 이격 + 타이밍
    result.ma5_gap_pct, result.ma7_gap_pct, result.entry_timing = \
        analyze_entry_timing(df)

    # 4. 종합 점수 (100점)
    score = 0

    # 수급 (40점)
    supply_stars = len(result.supply.grade)
    score += supply_stars * 8  # 최대 40

    # 급등패턴 (20점)
    if result.surge.verdict == "추세지속형":
        score += 20
    elif result.surge.verdict == "혼합":
        score += 10
    elif result.surge.verdict == "날고끝형":
        score += 0
    elif result.surge.verdict == "급등없음":
        score += 10  # 급등 없는 안정적 상승

    # 이격 타이밍 (20점)
    if result.entry_timing == "진입적기":
        score += 20
    elif result.entry_timing == "눌림대기":
        score += 12
    elif result.entry_timing == "이격과대":
        score += 5
    elif result.entry_timing == "하향이탈":
        score += 8  # 바닥잡이 가능성

    # OBV 추세 (10점)
    if result.supply.obv_5d_pct > 3.0:
        score += 10
    elif result.supply.obv_5d_pct > 1.0:
        score += 6
    elif result.supply.obv_5d_pct > 0:
        score += 3

    # 외인+기관 동시매수 보너스 (10점)
    if result.supply.foreign_5d > 0 and result.supply.inst_5d > 0:
        score += 5
    if result.supply.foreign_streak >= 2 and result.supply.inst_streak >= 2:
        score += 5

    result.score = min(100, score)

    # 5. 종합판정
    verdicts = []
    if score >= 75:
        verdicts.append("진입 OK")
    elif score >= 50:
        verdicts.append("조건부 진입")
    else:
        verdicts.append("대기")

    if result.entry_timing == "이격과대":
        verdicts.append("눌림 후 재진입")
    if result.surge.verdict == "날고끝형":
        verdicts.append("패턴 불안")
    if result.supply.foreign_streak >= 4:
        verdicts.append("외인 매집 강력")
    if result.supply.inst_streak >= 4:
        verdicts.append("기관 매집 강력")

    result.verdict = " / ".join(verdicts)

    # 6. 안전마진 플래그
    try:
        from src.safety_margin import calc_safety_margin
        sm = calc_safety_margin(ticker, name, result.close)
        result.safety = sm.to_dict()
        # verdict에 안전마진 반영
        if sm.signal == "GREEN":
            verdicts.append("안전마진OK")
        elif sm.signal == "RED":
            verdicts.append("안전마진주의")
        result.verdict = " / ".join(verdicts)
    except Exception as e:
        logger.debug("[딥다이브] 안전마진 실패: %s", e)

    return result


def deep_dive_batch(picks: list[dict], top_n: int = 5) -> list[DeepDiveResult]:
    """추천 종목 배치 딥다이브. picks = tomorrow_picks['picks'] 형태."""
    results = []
    for pick in picks[:top_n]:
        ticker = pick.get("ticker", "")
        name = pick.get("name", "")
        if not ticker:
            continue
        r = deep_dive(ticker, name)
        if r:
            results.append(r)

    # 점수 내림차순 정렬
    results.sort(key=lambda x: x.score, reverse=True)
    return results


# ─── 텔레그램 메시지 포맷 ─────────────────────────

def _억(val: float) -> str:
    """원 → 억 변환."""
    if abs(val) >= 1e8:
        return f"{val/1e8:+,.0f}억"
    elif abs(val) >= 1e4:
        return f"{val/1e4:+,.0f}만"
    return f"{val:+,.0f}"


def format_deep_dive_telegram(results: list[DeepDiveResult]) -> str:
    """딥다이브 결과를 텔레그램 메시지로 포맷."""
    if not results:
        return ""

    lines = ["🔬 [딥다이브 분석] TOP5", "━" * 24]

    for i, r in enumerate(results, 1):
        lines.append(f"\n{'━'*24}")
        lines.append(f"#{i} {r.name} ({r.ticker}) {r.close:,}원")
        lines.append(f"📊 딥다이브 점수: {r.score}/100 — {r.verdict}")

        # 수급
        s = r.supply
        lines.append(f"\n💰 수급 {s.grade}")
        lines.append(f"  외인 5일:{_억(s.foreign_5d)} 20일:{_억(s.foreign_20d)}")
        lines.append(f"  기관 5일:{_억(s.inst_5d)} 20일:{_억(s.inst_20d)}")
        streak_parts = []
        if s.foreign_streak > 0:
            streak_parts.append(f"외인{s.foreign_streak}일연속")
        if s.inst_streak > 0:
            streak_parts.append(f"기관{s.inst_streak}일연속")
        if streak_parts:
            lines.append(f"  {' + '.join(streak_parts)}")
        confirm_parts = []
        if s.foreign_vol_confirm:
            confirm_parts.append("외인거래량✅")
        if s.inst_vol_confirm:
            confirm_parts.append("기관거래량✅")
        if confirm_parts:
            lines.append(f"  {' '.join(confirm_parts)}")
        lines.append(f"  OBV 5일:{s.obv_5d_pct:+.1f}% | 거래량비:{s.vol_ratio:.2f}x")

        # 급등패턴
        g = r.surge
        if g.total_surges > 0:
            lines.append(f"\n📈 급등패턴 ({g.verdict})")
            lines.append(f"  +5%↑ {g.total_surges}회 → "
                         f"D+5 상승:{g.continue_count} 하락:{g.fade_count} "
                         f"({g.continue_rate:.0f}%)")
            lines.append(f"  D+5 평균: {g.avg_d5_return:+.1f}%")
            if g.latest_surge_date:
                lines.append(f"  최근: {g.latest_surge_date} → "
                             f"D+5 {g.latest_surge_d5:+.1f}%")
        else:
            lines.append(f"\n📈 급등패턴: {g.verdict}")

        # 이격
        lines.append(f"\n📐 이격: MA5 {r.ma5_gap_pct:+.1f}% / "
                     f"MA7 {r.ma7_gap_pct:+.1f}% → {r.entry_timing}")

    lines.append(f"\n{'━'*24}")
    return "\n".join(lines)
