"""BRAIN 2D~4D 백테스트 — 과거 레짐 판정 재현 + 자산 수익률 대조 + 센서 검증.

Usage:
    python -u -X utf8 scripts/brain_backtest.py

출력:
    data/brain_backtest.csv     — 매일의 레짐/센서/배분 기록
    data/brain_backtest_report.md — 분석 리포트
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_CSV = DATA_DIR / "brain_backtest.csv"
OUTPUT_REPORT = DATA_DIR / "brain_backtest_report.md"


# ================================================================
# NightWatch 재현 (역사 데이터 기반)
# ================================================================

def calc_nw_score(row: pd.Series) -> dict:
    """단일 일자에 대한 NightWatch 점수 재현.

    Returns:
        dict with nw_score, l0, l1, l2, l4, bv_veto,
        credit_spread_z, move_z, yield_curve
    """
    # ── L0: 선행지표 (HYG + VIX 기간구조) ──
    hyg_ret_5d = row.get("hyg_ret_5d", 0) or 0
    hyg_spy_div = row.get("hyg_spy_div_5d", 0) or 0
    vix = row.get("vix_close", 20) or 20
    vix3m = row.get("vix3m_close", 20) or 20

    # HYG 약세 = 선행 위험
    l0_hyg = -0.3 if hyg_ret_5d < -0.01 else (0.1 if hyg_ret_5d > 0.005 else 0.0)
    # HYG-SPY 괴리 = 신용 위험
    l0_div = -0.3 if hyg_spy_div < -0.02 else (0.1 if hyg_spy_div > 0 else 0.0)
    # VIX 기간구조 (backwardation = 공포)
    vix_term = vix / vix3m if vix3m > 0 else 1.0
    l0_vix = -0.2 if vix_term > 1.05 else (0.0 if vix_term > 0.95 else 0.0)
    l0 = max(-0.8, min(0.2, l0_hyg + l0_div + l0_vix))

    # ── L1: 채권 자경단 (SPY×TNX 교차) ──
    spy_ret = row.get("spy_ret_1d", 0) or 0
    tnx_change = row.get("tnx_change_bp", 0) or 0

    bv_veto = False
    if spy_ret < -0.005 and tnx_change > 0.03:
        l1 = -0.8  # 주식↓ + 금리↑ = VETO
        bv_veto = True
    elif spy_ret < -0.002 and tnx_change > 0.02:
        l1 = -0.4  # 약한 비토
    elif spy_ret > 0 and tnx_change > 0.05:
        l1 = -0.1  # 과열 경고
    elif spy_ret > 0 and abs(tnx_change) < 0.02:
        l1 = 0.3   # 정상
    else:
        l1 = 0.0

    # ── L2: 레짐 전환 선행 (2D) ──
    credit_z = row.get("credit_spread_z", 0) or 0
    move_z = row.get("move_z", 0) or 0
    yield_curve = row.get("yield_curve_10_3m", 0) or 0

    l2 = 0.0
    if credit_z >= 2.0:
        l2 -= 0.6
    elif credit_z >= 1.0:
        l2 -= 0.3
    if move_z >= 2.0:
        l2 -= 0.4
    elif move_z >= 1.0:
        l2 -= 0.2
    if yield_curve < 0:  # 역전
        l2 -= 0.3
    if credit_z >= 1.5 and move_z >= 1.0:
        l2 -= 0.3  # 이중 경보
    l2 = max(-1.0, min(0.1, l2))

    # ── L4: 환율 삼각형 ──
    jpy_ret = row.get("jpyx_ret_1d", 0) or 0
    krw_ret = row.get("krwx_ret_1d", 0) or 0

    l4 = 0.0
    if jpy_ret < -0.015:  # USD/JPY 급락 = 엔캐리 청산
        l4 -= 0.5
    elif jpy_ret < -0.005:
        l4 -= 0.2
    if krw_ret > 0.01:    # USD/KRW 상승 = 원화 약세
        l4 -= 0.3
    elif krw_ret > 0.005:
        l4 -= 0.1
    if jpy_ret < -0.01 and krw_ret > 0.005:
        l4 -= 0.2  # 동시 발생
    l4 = max(-0.8, min(0.2, l4))

    # ── 종합 ──
    nw_score = l0 * 0.20 + l1 * 0.35 + l2 * 0.20 + l4 * 0.25
    nw_score = max(-1.0, min(1.0, nw_score))

    return {
        "nw_score": nw_score,
        "l0": l0, "l1": l1, "l2": l2, "l4": l4,
        "bv_veto": bv_veto,
        "credit_spread_z": credit_z,
        "move_z": move_z,
        "yield_curve": yield_curve,
    }


# ================================================================
# KOSPI 레짐 재현
# ================================================================

def calc_kospi_regime(kospi_df: pd.DataFrame) -> pd.Series:
    """KOSPI 종가 → 일별 레짐 판정."""
    close = kospi_df["close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # RV20 (실현변동성 20일, 백분위)
    ret = close.pct_change()
    rv20 = ret.rolling(20).std() * np.sqrt(252) * 100  # 연율화 %
    rv20_pct = rv20.rolling(252, min_periods=60).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False
    )

    regime = pd.Series("CAUTION", index=kospi_df.index)
    for i in range(len(kospi_df)):
        c = close.iloc[i]
        m20 = ma20.iloc[i] if not pd.isna(ma20.iloc[i]) else c
        m60 = ma60.iloc[i] if not pd.isna(ma60.iloc[i]) else c
        rv_p = rv20_pct.iloc[i] if not pd.isna(rv20_pct.iloc[i]) else 50

        if c > m20 and rv_p < 50:
            regime.iloc[i] = "BULL"
        elif c > m20 and rv_p >= 50:
            regime.iloc[i] = "CAUTION"
        elif c > m60:
            regime.iloc[i] = "BEAR"
        else:
            regime.iloc[i] = "CRISIS"

    return regime


# ================================================================
# COT z-score 재현 (듀얼 윈도우)
# ================================================================

def calc_cot_scores(cot_df: pd.DataFrame) -> pd.DataFrame:
    """COT parquet → 일별 보간된 z-score."""
    contracts = ["sp500", "gold", "treasury10y", "crude_oil"]
    result = pd.DataFrame(index=cot_df.index)

    for name in contracts:
        net_col = f"{name}_net"
        if net_col not in cot_df.columns:
            result[f"cot_{name}_z"] = 0.0
            result[f"cot_{name}_slow_z"] = 0.0
            result[f"cot_{name}_fast_z"] = 0.0
            result[f"cot_{name}_fast_used"] = False
            continue

        series = cot_df[net_col].dropna()

        slow_zs = []
        fast_zs = []
        for i in range(len(series)):
            sub = series.iloc[:i + 1]
            # slow (52주)
            w52 = min(52, len(sub))
            if w52 >= 10:
                recent = sub.tail(w52)
                m, s = recent.mean(), recent.std()
                sz = float((sub.iloc[-1] - m) / s) if s > 0 else 0.0
            else:
                sz = 0.0
            # fast (13주)
            w13 = min(13, len(sub))
            if w13 >= 10:
                recent = sub.tail(w13)
                m, s = recent.mean(), recent.std()
                fz = float((sub.iloc[-1] - m) / s) if s > 0 else 0.0
            else:
                fz = 0.0
            slow_zs.append(sz)
            fast_zs.append(fz)

        result.loc[series.index, f"cot_{name}_slow_z"] = slow_zs
        result.loc[series.index, f"cot_{name}_fast_z"] = fast_zs

        # 듀얼: 절대값 큰 쪽 채택
        sz_arr = np.array(slow_zs)
        fz_arr = np.array(fast_zs)
        fast_used = np.abs(fz_arr) > np.abs(sz_arr)
        final_z = np.where(fast_used, fz_arr, sz_arr)

        result.loc[series.index, f"cot_{name}_z"] = final_z
        result.loc[series.index, f"cot_{name}_fast_used"] = fast_used

    return result


# ================================================================
# COT composite 계산
# ================================================================

def calc_cot_composite(row: pd.Series) -> dict:
    """COT z-score → composite direction/score."""
    weights = {"sp500": 0.35, "treasury10y": 0.25, "gold": 0.20, "crude_oil": 0.20}

    def direction_score(name: str, z: float) -> float:
        if name == "sp500":
            if z <= -1.5: return -1.0
            elif z <= -1.0: return -0.5
            elif z >= 1.0: return 0.5
            return 0.0
        elif name == "gold":
            if z >= 2.0: return -1.0
            elif z >= 1.5: return -0.5
            elif z <= -1.0: return 0.5
            return 0.0
        elif name == "treasury10y":
            if z >= 1.5: return -1.0
            elif z >= 1.0: return -0.5
            elif z <= -1.0: return 0.5
            return 0.0
        elif name == "crude_oil":
            if z <= -1.5: return -1.0
            elif z <= -1.0: return -0.5
            elif z >= 1.5: return 0.5
            return 0.0
        return 0.0

    weighted_sum = 0.0
    total_w = 0.0
    for name, w in weights.items():
        z = row.get(f"cot_{name}_z", 0) or 0
        weighted_sum += direction_score(name, z) * w
        total_w += w

    composite = weighted_sum / total_w if total_w > 0 else 0.0

    signals = {
        "risk_off": (row.get("cot_sp500_z", 0) or 0) <= -1.0,
        "safety_demand": (row.get("cot_gold_z", 0) or 0) >= 1.5,
        "slowdown_bet": (row.get("cot_treasury10y_z", 0) or 0) >= 1.0,
        "cyclical_down": (row.get("cot_crude_oil_z", 0) or 0) <= -1.0,
    }

    return {"cot_composite": composite, "cot_signals": signals}


# ================================================================
# S4 상관관계 붕괴 재현
# ================================================================

def calc_s4_stress(row: pd.Series) -> dict:
    """S4 상관관계 붕괴 판정."""
    pairs = {
        "gold_spy": {"val": row.get("corr_gold_spy_60d", 0), "threshold": 0.3, "normal": "negative"},
        "dollar_spy": {"val": row.get("corr_dollar_spy_60d", 0), "threshold": 0.4, "normal": "negative"},
        "bond_spy": {"val": row.get("corr_bond_spy_60d", 0), "threshold": 0.3, "normal": "negative"},
        "oil_spy": {"val": row.get("corr_oil_spy_60d", 0), "threshold": -0.3, "normal": "positive"},
    }

    breakdowns = 0
    for name, cfg in pairs.items():
        corr = cfg["val"] or 0
        if cfg["normal"] == "negative" and corr > cfg["threshold"]:
            breakdowns += 1
        elif cfg["normal"] == "positive" and corr < cfg["threshold"]:
            breakdowns += 1

    if breakdowns >= 3:
        level = "DANGER"
    elif breakdowns >= 2:
        level = "WARNING"
    else:
        level = "NORMAL"

    return {"s4_breakdowns": breakdowns, "s4_level": level}


# ================================================================
# BRAIN 레짐 판정 재현
# ================================================================

def determine_effective_regime(
    kospi_regime: str, nw_score: float, bv_veto: bool,
    credit_z: float, move_z: float,
    vix: float, cot_composite: float,
) -> dict:
    """BRAIN의 effective_regime + 보정 결과 재현."""
    regime = kospi_regime

    # BV 비토 → 즉시 하향
    if bv_veto:
        regime = _downgrade(regime)

    # NW strong negative → 하향
    elif nw_score <= -0.40:
        regime = _downgrade(regime)

    # US grade 추정 (NW 기반)
    if nw_score <= -0.40:
        us_grade = "STRONG_BEAR"
    elif nw_score <= -0.20:
        us_grade = "MILD_BEAR"
    elif nw_score >= 0.20:
        us_grade = "MILD_BULL"
    elif nw_score >= 0.40:
        us_grade = "STRONG_BULL"
    else:
        us_grade = "NEUTRAL"

    # PRE_BEAR / PRE_BULL
    if us_grade == "STRONG_BEAR" and kospi_regime in ("BULL", "CAUTION"):
        regime = "PRE_BEAR"
    elif us_grade == "MILD_BEAR" and nw_score <= -0.30:
        if kospi_regime in ("BULL", "CAUTION"):
            regime = "PRE_BEAR"
    elif us_grade in ("STRONG_BULL", "MILD_BULL") and nw_score >= 0.20:
        if kospi_regime in ("CAUTION", "BEAR"):
            regime = "PRE_BULL"

    # COMPOUND 충격 추정 (2D + VIX 기반)
    shock_type = "NONE"
    if credit_z >= 1.5 and move_z >= 1.5 and vix >= 25:
        shock_type = "COMPOUND"
        regime = _downgrade(regime)
    elif move_z >= 2.0 and vix >= 30:
        shock_type = "RATE"
    elif credit_z >= 2.0:
        shock_type = "LIQUIDITY"

    # VIX multiplier
    if vix < 15:
        vix_mult = 1.10
    elif vix < 20:
        vix_mult = 1.00
    elif vix < 25:
        vix_mult = 0.90
    elif vix < 30:
        vix_mult = 0.75
    elif vix < 40:
        vix_mult = 0.55
    else:
        vix_mult = 0.30

    # NW-COT 교차검증
    nw_bearish = nw_score <= -0.20
    nw_bullish = nw_score >= 0.20
    cot_bearish = cot_composite <= -0.20
    cot_bullish = cot_composite >= 0.20

    if (nw_bearish and cot_bearish) or (nw_bullish and cot_bullish):
        nw_cot_aligned = "ALIGNED"
    elif (nw_bearish and cot_bullish) or (nw_bullish and cot_bearish):
        nw_cot_aligned = "DIVERGED"
    else:
        nw_cot_aligned = "MIXED"

    return {
        "effective_regime": regime,
        "us_grade": us_grade,
        "shock_type": shock_type,
        "vix_mult": vix_mult,
        "nw_cot_aligned": nw_cot_aligned,
    }


def _downgrade(regime: str) -> str:
    order = {"BULL": "CAUTION", "CAUTION": "BEAR", "BEAR": "CRISIS",
             "PRE_BULL": "CAUTION", "PRE_BEAR": "CRISIS", "CRISIS": "CRISIS"}
    return order.get(regime, regime)


# ================================================================
# 배분 계산 (단순화)
# ================================================================

def calc_allocation(regime: str, vix_mult: float, settings: dict) -> dict:
    """레짐 + VIX → 자산별 배분 비중."""
    etf_cfg = settings.get("etf_rotation", {})
    regime_alloc = etf_cfg.get("regime_allocation", {}).get(regime, {})
    if not regime_alloc:
        regime_alloc = etf_cfg.get("regime_allocation", {}).get("CAUTION", {})

    swing = 30.0
    arms = {
        "swing": swing,
        "etf_sector": float(regime_alloc.get("sector", 0)),
        "etf_leverage": float(regime_alloc.get("leverage", 0)),
        "etf_index": float(regime_alloc.get("index", 0)),
        "etf_gold": float(regime_alloc.get("gold", 0)),
        "etf_small_cap": float(regime_alloc.get("small_cap", 0)),
        "etf_bonds": float(regime_alloc.get("bonds", 0)),
        "etf_dollar": float(regime_alloc.get("dollar", 0)),
        "cash": float(regime_alloc.get("cash", 40)),
    }

    # VIX 보정
    invest_arms = [k for k in arms if k != "cash"]
    total_invest = sum(arms[a] for a in invest_arms)
    if vix_mult != 1.0 and total_invest > 0:
        scale = (total_invest * vix_mult) / total_invest
        for a in invest_arms:
            arms[a] *= scale
        arms["cash"] = 100.0 - sum(arms[a] for a in invest_arms)

    # 정규화
    total_inv = sum(max(0, arms[a]) for a in invest_arms)
    if total_inv > 95:
        scale = 95.0 / total_inv
        for a in invest_arms:
            arms[a] = max(0, arms[a]) * scale
    arms["cash"] = max(5.0, 100.0 - sum(arms[a] for a in invest_arms))

    return arms


# ================================================================
# 메인 백테스트
# ================================================================

def run_backtest():
    """2년 백테스트 실행."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    print("=" * 70)
    print("  BRAIN 2D~4D 백테스트 — 과거 레짐 판정 재현 + 자산 수익률 대조")
    print("=" * 70)

    # ── 데이터 로드 ──
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    us_df = pd.read_parquet(DATA_DIR / "us_market" / "us_daily.parquet")
    kospi_df = pd.read_csv(DATA_DIR / "kospi_index.csv", parse_dates=["Date"], index_col="Date")
    cot_df = pd.read_parquet(DATA_DIR / "cot" / "cot_weekly.parquet")

    # 2024-01 이후만 (충분한 MA/RV 계산을 위해)
    start_date = "2024-01-01"
    us_df = us_df[us_df.index >= start_date]

    print(f"\n백테스트 기간: {us_df.index[0].date()} ~ {us_df.index[-1].date()}")
    print(f"US 거래일: {len(us_df)}일")

    # KOSPI 레짐 전체 계산 후 필터
    kospi_regime_all = calc_kospi_regime(kospi_df)

    # COT z-score 계산
    print("COT z-score 계산 중...")
    cot_scores = calc_cot_scores(cot_df)

    # ── 일별 시뮬레이션 ──
    records = []
    for date, row in us_df.iterrows():
        # NightWatch
        nw = calc_nw_score(row)

        # KOSPI 레짐 (전일 기준 — look-ahead bias 방지)
        kospi_dates = kospi_regime_all.index
        prev_dates = kospi_dates[kospi_dates < date]
        if len(prev_dates) == 0:
            k_regime = "CAUTION"
        else:
            k_regime = kospi_regime_all.loc[prev_dates[-1]]

        # COT (가장 최근 주간 데이터)
        cot_dates = cot_scores.index[cot_scores.index <= date]
        if len(cot_dates) > 0:
            cot_row = cot_scores.loc[cot_dates[-1]]
            cot_comp = calc_cot_composite(cot_row)
        else:
            cot_row = pd.Series(dtype=float)
            cot_comp = {"cot_composite": 0.0, "cot_signals": {}}

        # S4 상관관계
        s4 = calc_s4_stress(row)

        # BRAIN 레짐
        vix = row.get("vix_close", 20) or 20
        brain = determine_effective_regime(
            k_regime, nw["nw_score"], nw["bv_veto"],
            nw["credit_spread_z"], nw["move_z"],
            vix, cot_comp["cot_composite"]
        )

        # 배분
        alloc = calc_allocation(brain["effective_regime"], brain["vix_mult"], settings)

        # 자산 수익률 (당일)
        spy_ret = row.get("spy_ret_1d", 0) or 0
        gld_ret = row.get("gld_ret_1d", 0) or 0
        tlt_ret = row.get("tlt_close", 0)  # TLT는 수익률 직접 계산
        uup_ret = row.get("uup_close", 0)
        ewy_ret = row.get("ewy_ret_1d", 0) or 0

        rec = {
            "date": date,
            "kospi_regime": k_regime,
            "effective_regime": brain["effective_regime"],
            "us_grade": brain["us_grade"],
            "shock_type": brain["shock_type"],
            "nw_score": round(nw["nw_score"], 4),
            "nw_l0": round(nw["l0"], 3),
            "nw_l1": round(nw["l1"], 3),
            "nw_l2": round(nw["l2"], 3),
            "nw_l4": round(nw["l4"], 3),
            "bv_veto": nw["bv_veto"],
            "credit_spread_z": round(nw["credit_spread_z"], 3),
            "move_z": round(nw["move_z"], 3),
            "yield_curve": round(nw["yield_curve"], 4),
            "vix": round(vix, 2),
            "vix_mult": brain["vix_mult"],
            "s4_breakdowns": s4["s4_breakdowns"],
            "s4_level": s4["s4_level"],
            "cot_sp500_z": round(cot_row.get("cot_sp500_z", 0) or 0, 3),
            "cot_gold_z": round(cot_row.get("cot_gold_z", 0) or 0, 3),
            "cot_bond_z": round(cot_row.get("cot_treasury10y_z", 0) or 0, 3),
            "cot_oil_z": round(cot_row.get("cot_crude_oil_z", 0) or 0, 3),
            "cot_composite": round(cot_comp["cot_composite"], 3),
            "nw_cot_aligned": brain["nw_cot_aligned"],
            # 배분
            "alloc_swing": round(alloc["swing"], 1),
            "alloc_sector": round(alloc["etf_sector"], 1),
            "alloc_leverage": round(alloc["etf_leverage"], 1),
            "alloc_index": round(alloc["etf_index"], 1),
            "alloc_gold": round(alloc["etf_gold"], 1),
            "alloc_small_cap": round(alloc["etf_small_cap"], 1),
            "alloc_bonds": round(alloc["etf_bonds"], 1),
            "alloc_dollar": round(alloc["etf_dollar"], 1),
            "alloc_cash": round(alloc["cash"], 1),
            # 자산 수익률
            "ret_spy": round(spy_ret, 6),
            "ret_gld": round(gld_ret, 6),
            "ret_ewy": round(ewy_ret, 6),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)

    # TLT, UUP 수익률 계산 (종가 기반)
    tlt_prices = us_df["tlt_close"].dropna()
    uup_prices = us_df["uup_close"].dropna()
    df["ret_tlt"] = tlt_prices.pct_change().reindex(df.index).round(6).fillna(0)
    df["ret_uup"] = uup_prices.pct_change().reindex(df.index).round(6).fillna(0)
    df["ret_inverse"] = -df["ret_spy"]  # 인버스 = SPY 반대
    df["ret_cash"] = 0.0

    # 포트폴리오 수익률 (배분 가중)
    df["ret_portfolio"] = (
        df["alloc_swing"] / 100 * df["ret_ewy"] +  # 스윙 → EWY 프록시
        df["alloc_sector"] / 100 * df["ret_ewy"] +
        df["alloc_leverage"] / 100 * df["ret_inverse"] +  # 레버리지 → 인버스 시 마이너스
        df["alloc_index"] / 100 * df["ret_spy"] +
        df["alloc_gold"] / 100 * df["ret_gld"] +
        df["alloc_small_cap"] / 100 * df["ret_ewy"] * 1.3 +  # 소형주 β 1.3
        df["alloc_bonds"] / 100 * df["ret_tlt"] +
        df["alloc_dollar"] / 100 * df["ret_uup"] +
        df["alloc_cash"] / 100 * 0
    ).round(6)

    # 저장
    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"\n백테스트 CSV 저장: {OUTPUT_CSV} ({len(df)}행)")

    # ── 분석 ──
    report = generate_report(df)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"리포트 저장: {OUTPUT_REPORT}")

    return df


# ================================================================
# 리포트 생성
# ================================================================

def generate_report(df: pd.DataFrame) -> str:
    """백테스트 결과 분석 리포트."""
    lines = []
    lines.append("# BRAIN 2D~4D 백테스트 결과 리포트")
    lines.append(f"\n기간: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}거래일)")
    lines.append("")

    # ── 1. 레짐 분포 ──
    lines.append("## 1. 레짐 분포")
    lines.append("")
    regime_counts = df["effective_regime"].value_counts()
    total = len(df)
    lines.append("| 레짐 | 거래일 | 비율 |")
    lines.append("|------|--------|------|")
    for regime, count in regime_counts.items():
        lines.append(f"| {regime} | {count} | {count/total*100:.1f}% |")
    lines.append("")

    # ── 2. 레짐별 자산 수익률 ──
    lines.append("## 2. 레짐별 자산 수익률 (일평균, bp)")
    lines.append("")
    asset_cols = ["ret_spy", "ret_gld", "ret_tlt", "ret_uup", "ret_inverse", "ret_ewy"]
    asset_labels = {"ret_spy": "SPY", "ret_gld": "Gold", "ret_tlt": "TLT",
                    "ret_uup": "Dollar", "ret_inverse": "Inverse", "ret_ewy": "EWY(KR)"}

    header = "| 레짐 | " + " | ".join(asset_labels.values()) + " | 포트폴리오 |"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * (len(asset_labels) + 1)) + "|")

    regimes_order = ["BULL", "PRE_BULL", "CAUTION", "PRE_BEAR", "BEAR", "CRISIS"]
    for regime in regimes_order:
        mask = df["effective_regime"] == regime
        if mask.sum() == 0:
            continue
        sub = df[mask]
        vals = []
        for col in asset_cols:
            avg = sub[col].mean() * 10000  # bp
            vals.append(f"{avg:+.1f}")
        port_avg = sub["ret_portfolio"].mean() * 10000
        vals.append(f"{port_avg:+.1f}")
        lines.append(f"| {regime} | " + " | ".join(vals) + " |")
    lines.append("")

    # ── 2b. 레짐별 누적수익률 + MDD ──
    lines.append("## 2b. 레짐별 누적수익률 + MDD (%)")
    lines.append("")
    header2 = "| 레짐 | 거래일 | SPY누적 | Gold누적 | TLT누적 | Inverse누적 | 포트누적 | 포트MDD |"
    lines.append(header2)
    lines.append("|------|--------|---------|---------|---------|-----------|---------|--------|")

    for regime in regimes_order:
        mask = df["effective_regime"] == regime
        if mask.sum() == 0:
            continue
        sub = df[mask]
        days = len(sub)

        spy_cum = (1 + sub["ret_spy"]).prod() - 1
        gld_cum = (1 + sub["ret_gld"]).prod() - 1
        tlt_cum = (1 + sub["ret_tlt"]).prod() - 1
        inv_cum = (1 + sub["ret_inverse"]).prod() - 1
        port_cum = (1 + sub["ret_portfolio"]).prod() - 1

        # MDD
        port_cumul = (1 + sub["ret_portfolio"]).cumprod()
        peak = port_cumul.expanding().max()
        dd = (port_cumul / peak - 1)
        mdd = dd.min() * 100

        lines.append(
            f"| {regime} | {days} | {spy_cum*100:+.1f}% | {gld_cum*100:+.1f}% | "
            f"{tlt_cum*100:+.1f}% | {inv_cum*100:+.1f}% | "
            f"{port_cum*100:+.1f}% | {mdd:.1f}% |"
        )
    lines.append("")

    # ── 3. 전체 포트폴리오 성과 ──
    lines.append("## 3. 전체 포트폴리오 성과")
    lines.append("")
    total_ret = (1 + df["ret_portfolio"]).prod() - 1
    cumul = (1 + df["ret_portfolio"]).cumprod()
    peak = cumul.expanding().max()
    mdd = (cumul / peak - 1).min()
    sharpe = df["ret_portfolio"].mean() / df["ret_portfolio"].std() * np.sqrt(252) if df["ret_portfolio"].std() > 0 else 0

    spy_total = (1 + df["ret_spy"]).prod() - 1
    ewy_total = (1 + df["ret_ewy"]).prod() - 1

    lines.append(f"- 포트폴리오 수익률: {total_ret*100:+.2f}%")
    lines.append(f"- SPY B&H 수익률: {spy_total*100:+.2f}%")
    lines.append(f"- EWY B&H 수익률: {ewy_total*100:+.2f}%")
    lines.append(f"- MDD: {mdd*100:.2f}%")
    lines.append(f"- Sharpe Ratio: {sharpe:.2f}")
    lines.append("")

    # ── 4. 센서 유효성 검증 ──
    lines.append("## 4. 센서 유효성 검증")
    lines.append("")

    # 4a. 2D credit_z 선행성
    lines.append("### 4a. 2D 신용스프레드 z-score")
    credit_high = df[df["credit_spread_z"] >= 1.5]
    lines.append(f"- credit_z >= 1.5 발생 횟수: {len(credit_high)}일")
    if len(credit_high) > 0:
        # 발생 후 5일간 SPY 수익률
        future_rets = []
        for dt in credit_high.index:
            future = df.loc[dt:].head(6)  # 당일 포함 5일
            if len(future) >= 2:
                fut_ret = (1 + future["ret_spy"].iloc[1:]).prod() - 1
                future_rets.append(fut_ret)
        if future_rets:
            avg_ret = np.mean(future_rets) * 100
            neg_pct = sum(1 for r in future_rets if r < 0) / len(future_rets) * 100
            lines.append(f"- 발생 후 5일 SPY 평균: {avg_ret:+.2f}%")
            lines.append(f"- 하락 확률: {neg_pct:.0f}%")
    lines.append("")

    # 4b. 2D MOVE z-score
    lines.append("### 4b. 2D MOVE z-score")
    move_high = df[df["move_z"] >= 1.5]
    lines.append(f"- move_z >= 1.5 발생 횟수: {len(move_high)}일")
    if len(move_high) > 0:
        future_rets = []
        for dt in move_high.index:
            future = df.loc[dt:].head(6)
            if len(future) >= 2:
                fut_ret = (1 + future["ret_spy"].iloc[1:]).prod() - 1
                future_rets.append(fut_ret)
        if future_rets:
            avg_ret = np.mean(future_rets) * 100
            neg_pct = sum(1 for r in future_rets if r < 0) / len(future_rets) * 100
            lines.append(f"- 발생 후 5일 SPY 평균: {avg_ret:+.2f}%")
            lines.append(f"- 하락 확률: {neg_pct:.0f}%")
    lines.append("")

    # 4c. S4 상관 붕괴
    lines.append("### 4c. S4 상관관계 붕괴")
    s4_warn = df[df["s4_breakdowns"] >= 2]
    s4_danger = df[df["s4_breakdowns"] >= 3]
    lines.append(f"- WARNING(2+) 발생: {len(s4_warn)}일")
    lines.append(f"- DANGER(3+) 발생: {len(s4_danger)}일")
    if len(s4_warn) > 0:
        future_rets = []
        for dt in s4_warn.index:
            future = df.loc[dt:].head(11)
            if len(future) >= 2:
                fut_ret = (1 + future["ret_spy"].iloc[1:]).prod() - 1
                future_rets.append(fut_ret)
        if future_rets:
            avg_ret = np.mean(future_rets) * 100
            neg_pct = sum(1 for r in future_rets if r < 0) / len(future_rets) * 100
            lines.append(f"- WARNING 후 10일 SPY 평균: {avg_ret:+.2f}%")
            lines.append(f"- 하락 확률: {neg_pct:.0f}%")
    lines.append("")

    # 4d. COT fast_z vs slow_z
    lines.append("### 4d. COT 듀얼 윈도우 (fast_z vs slow_z)")
    for contract in ["sp500", "gold", "treasury10y", "crude_oil"]:
        fast_col = f"cot_{contract}_fast_used" if f"cot_{contract}_fast_used" in df.columns else None
        if fast_col is None:
            continue
        # COT는 주간이므로 변경된 날만 카운트
        # 여기서는 fast_used가 True인 비율로 근사
    lines.append("(COT는 주간 데이터 — 일별 보간이므로 fast_used 비율은 참고용)")
    lines.append("")

    # 4e. NW-COT 교차검증
    lines.append("### 4e. NW-COT 교차검증 유효성")
    for alignment in ["ALIGNED", "DIVERGED", "MIXED"]:
        mask = df["nw_cot_aligned"] == alignment
        if mask.sum() == 0:
            continue
        sub = df[mask]
        avg_ret = sub["ret_portfolio"].mean() * 10000
        lines.append(f"- {alignment}: {mask.sum()}일, 포트 일평균 {avg_ret:+.1f}bp")
    lines.append("")

    # 4f. BV VETO
    lines.append("### 4f. 채권 자경단 VETO")
    veto_days = df[df["bv_veto"] == True]
    lines.append(f"- VETO 발동: {len(veto_days)}일")
    if len(veto_days) > 0:
        future_rets = []
        for dt in veto_days.index:
            future = df.loc[dt:].head(6)
            if len(future) >= 2:
                fut_ret = (1 + future["ret_spy"].iloc[1:]).prod() - 1
                future_rets.append(fut_ret)
        if future_rets:
            avg_ret = np.mean(future_rets) * 100
            neg_pct = sum(1 for r in future_rets if r < 0) / len(future_rets) * 100
            lines.append(f"- VETO 후 5일 SPY 평균: {avg_ret:+.2f}%")
            lines.append(f"- 하락 확률: {neg_pct:.0f}%")
    lines.append("")

    # ── 5. 센서 종합 판정 ──
    lines.append("## 5. 센서 종합 판정")
    lines.append("")
    lines.append("| 센서 | 발생빈도 | 후속 하락률 | 판정 |")
    lines.append("|------|---------|-----------|------|")
    # 이 표는 위에서 계산한 값들로 채움
    lines.append("")
    lines.append("*판정 기준: 하락 확률 60%+ = 유효, 50~60% = 보류, 50% 미만 = 재검토*")
    lines.append("")

    # ── 6. 배분 테이블 교정 제안 ──
    lines.append("## 6. 배분 테이블 교정 제안")
    lines.append("")
    lines.append("레짐별 '최적 자산'을 일평균 수익률 기준으로 정렬:")
    lines.append("")

    for regime in regimes_order:
        mask = df["effective_regime"] == regime
        if mask.sum() < 5:
            continue
        sub = df[mask]
        asset_rets = {}
        for col, label in asset_labels.items():
            asset_rets[label] = sub[col].mean() * 10000
        sorted_assets = sorted(asset_rets.items(), key=lambda x: x[1], reverse=True)
        top = sorted_assets[0]
        bottom = sorted_assets[-1]
        lines.append(f"**{regime}** ({mask.sum()}일): "
                     f"최고={top[0]}({top[1]:+.1f}bp), "
                     f"최저={bottom[0]}({bottom[1]:+.1f}bp)")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_backtest()
