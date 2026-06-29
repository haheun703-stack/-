"""주도주 사이클 진단 엔진 — 한규범 "주도주 사이클 절대 법칙" 구현.

방법론 요약(예측 아님, "현재 사이클 위치 + 확률적 잔여수명" 진단):
- 주도주의 기세는 산업이 아니라 "기간"으로 규정. 국내 평균 ~2년, 미국 ~2.5년 안에 95% 종료.
- 사이클 시작점 = 주봉 정배열 성립 시점(정배열 전 ~7개월 준비기는 카운트 제외).
- 델타 = 영업이익 성장률의 변화(2차 미분). 둔화(음전환)가 2년을 넘기 힘든 근본 원인.
- 매도 트리거는 델타 음전환이 아니라 "주봉 추세 붕괴"(MA20 종가 이탈 + 정배열 붕괴).
- 주도주는 시장(매크로)과 무관 — 비주도주엔 가설 적용 금지.

설계 원칙:
- 순수 함수: 주봉 DataFrame in → dict(JSON 직렬화 가능) out. 파일/DB/네트워크 I/O 없음.
- 데이터 소스 독립: KR(parquet 리샘플)·US(yfinance) 동일 함수로 진단 → 백테스트 캘리브레이션 일관.
- 모든 판정은 주봉 종가 기준(일봉 노이즈 금지).

사용 예:
    from src.use_cases.leader_cycle_diagnosis import diagnose_leader_cycle
    res = diagnose_leader_cycle(weekly_df, market="KR", op_growth=growth_series,
                                ticker="086520", name="에코프로")
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ─── 조정 가능 파라미터 (지시서 §6) ──────────────────────────
DEFAULT_PARAMS: dict = {
    "ma": (5, 20, 60),          # 주봉 이평선 (민감 버전: 5/10/20)
    "tolerance_weeks": 6,       # 사이클 유지 중 정배열 깨짐 허용 주수(잔파동 무시).
                                # 백테스트 캘리브레이션(§8): 6=한화에어로 방산 26m(+532%) 정확 재현,
                                # 일시 조정마다 사이클을 끊어 age를 리셋하는 오류 방지(후반→초반 오인 차단).
    "mdd_threshold": -0.20,     # 주봉 종가 기준 고점 대비 정상 조정 한계
    "us_stretch": 1.25,         # 미국 종목 생존율 곡선 ×1.25 (더 길게)
    "min_weeks": 65,            # MA60 + 역추적 최소 주차
    # 생존율 곡선 분기점 (개월) — KR 기준, US는 ×us_stretch
    "clock_noon": 10.0,         # 오전 → 정오
    "clock_afternoon": 12.0,    # 정오 → 오후
    "clock_close": 24.0,        # 오후 → 마감
    "buy_window_months": 10.0,  # 매수적기 상한 (정배열 성립 ~ age 10개월)
}

# 지시서 §7 — 봇이 결과에 반드시 명시할 한계
LIMITATIONS = [
    "생존편향: 사후 통계다. 정배열은 필요조건이지 충분조건 아님 → 정배열=주도주 아님. "
    "매수적기 신호도 별도 펀더멘털·수급 확인 필요.",
    "'2년'은 중앙값. 개별 분포는 7~24개월로 매우 넓다 → 확정적 만기로 쓰지 말 것.",
    "시작점(정배열 시점)은 이평선 설정에 민감 → cycle_start는 ±1~2개월 오차 가정.",
    "실적 무뒷받침(기대주·바이오)은 곡선보다 짧게 끝나는 경향 → delta 데이터 없으면 신뢰도 하향.",
    "V자 급반등주(준비기 없이 바닥에서 급등)는 MA60 우상향 전환이 늦어 정배열 성립이 지연 → "
    "사이클 시작을 늦게 잡거나 폭등 대부분을 놓친다(예: 에코프로비엠 2023). 이런 종목은 진단 신뢰도 하향.",
]


def _normalize_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """주봉 DataFrame 정규화: DatetimeIndex + open/high/low/close/volume 소문자."""
    w = weekly.copy()
    if not isinstance(w.index, pd.DatetimeIndex):
        # 'date' 컬럼이 있으면 인덱스로, 없으면 인덱스 캐스팅
        if "date" in w.columns:
            w = w.set_index(pd.to_datetime(w["date"]))
        else:
            w.index = pd.to_datetime(w.index)
    w = w.rename(columns={c: c.lower() for c in w.columns})
    w = w.loc[:, ~w.columns.duplicated()]      # 동명(대/소문자) 컬럼 중복 제거 → close 2차원 방지
    if "close" in w.columns:
        w = w[w["close"].notna()]              # NaN 종가 행 제거(유효 데이터만)
    w = w.sort_index()
    return w


def _survival_pct(age_adj: float, p: dict) -> float:
    """생존율 곡선 보간 (age_adj = 미국 stretch 보정 후 개월).

    KR: [10m→0.50, 12m→0.30, 24m→0.05]. 오후 구간(12~24)은 선형 보간.
    """
    noon, aft, close = p["clock_noon"], p["clock_afternoon"], p["clock_close"]
    if age_adj < noon:
        return 0.5
    if age_adj < aft:
        return 0.3
    if age_adj < close:
        # 12개월 0.30 → 24개월 0.05 선형 보간
        frac = (age_adj - aft) / (close - aft)
        return round(0.30 + frac * (0.05 - 0.30), 3)
    return 0.05


def _clock(age_adj: float, p: dict) -> str:
    if age_adj < p["clock_noon"]:
        return "오전"
    if age_adj < p["clock_afternoon"]:
        return "정오"
    if age_adj < p["clock_close"]:
        return "오후"
    return "마감"


def _compute_alignment(w: pd.DataFrame, p: dict) -> pd.DataFrame:
    """주봉 MA + 정배열 strict/loose 마스크 계산 (게이트 1 공통 기반).

    - 성립(strict): MA5>MA20>MA60 AND 세 선 기울기>0(우상향) AND 종가>MA5
    - 유지(loose):  MA5>MA20>MA60 (기울기 무시)
    """
    s, m, l = p["ma"]
    w = w.copy()
    w["ma_s"] = w["close"].rolling(s).mean()
    w["ma_m"] = w["close"].rolling(m).mean()
    w["ma_l"] = w["close"].rolling(l).mean()
    up_s = w["ma_s"].diff() > 0
    up_m = w["ma_m"].diff() > 0
    up_l = w["ma_l"].diff() > 0
    order = (w["ma_s"] > w["ma_m"]) & (w["ma_m"] > w["ma_l"])
    w["strict"] = (order & up_s & up_m & up_l & (w["close"] > w["ma_s"])).fillna(False)
    w["loose"] = order.fillna(False)
    return w


def _scan_cycles(w: pd.DataFrame, p: dict) -> list[tuple[int, int]]:
    """상태기계로 정배열 사이클 구간 [start_idx, end_idx] 리스트 추출(잔파동 tolerance 허용).

    strict 성립 시 사이클 시작, loose가 tolerance_weeks 연속 깨지면 종료(깨짐 시작 직전을 end로).
    """
    strict = w["strict"].to_numpy()
    loose = w["loose"].to_numpy()
    tol = p["tolerance_weeks"]
    cycles: list[tuple[int, int]] = []
    active = False
    start_i = 0
    last_intact = 0   # 마지막으로 loose 유지된 인덱스
    broken_run = 0
    for i in range(len(w)):
        if not active:
            if strict[i]:
                active = True
                start_i = i
                last_intact = i
                broken_run = 0
        else:
            if loose[i]:
                last_intact = i
                broken_run = 0
            else:
                broken_run += 1
                if broken_run > tol:
                    cycles.append((start_i, last_intact))
                    active = False
                    broken_run = 0
    if active:
        # 루프 끝까지 active여도 end는 마지막으로 정배열 유지된 주(깨진 꼬리 제외)
        cycles.append((start_i, last_intact))
    return cycles


def _trace_cycle(w: pd.DataFrame, p: dict) -> dict:
    """게이트 1 — 현재(마지막 주) 기준 정배열 활성 여부 + cycle_start 역추적.

    active = 마지막 정배열 유지 주가 데이터 끝에서 tolerance_weeks 이내(=유예중).
    그 이상 깨졌으면 사이클 종료(비active). cycle 구간 end는 깨진 꼬리를 제외한 last_intact.
    """
    w = _compute_alignment(w, p)
    cycles = _scan_cycles(w, p)
    idx = w.index
    last_i = len(w) - 1
    tol = p["tolerance_weeks"]
    last_cycle = cycles[-1] if cycles else None
    active = bool(last_cycle) and (last_i - last_cycle[1]) <= tol
    cur_start = idx[last_cycle[0]] if active else None
    last_aligned_week = None
    if not active and last_cycle is not None:
        last_aligned_week = str(idx[last_cycle[1]].date())
    return {
        "active": active,
        "cycle_start": cur_start,
        "loose_intact": bool(w["loose"].iloc[-1]),
        "aligned_strict": bool(w["strict"].iloc[-1]),
        "ma_s": float(w["ma_s"].iloc[-1]) if pd.notna(w["ma_s"].iloc[-1]) else None,
        "ma_m": float(w["ma_m"].iloc[-1]) if pd.notna(w["ma_m"].iloc[-1]) else None,
        "ma_l": float(w["ma_l"].iloc[-1]) if pd.notna(w["ma_l"].iloc[-1]) else None,
        "last_aligned_week": last_aligned_week,
    }


def extract_all_cycles(weekly: pd.DataFrame, params: dict | None = None) -> list[dict]:
    """전체 주봉에서 모든 정배열 사이클 구간을 추출(백테스트 캘리브레이션용).

    Returns: [{start, end, duration_months, peak_date, peak_close, ret_to_peak_pct}, ...]
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    w = _compute_alignment(_normalize_weekly(weekly), p)
    if len(w) < p["min_weeks"]:
        return []
    out = []
    for s_i, e_i in _scan_cycles(w, p):
        seg = w.iloc[s_i:e_i + 1]
        start_dt, end_dt = w.index[s_i], w.index[e_i]
        dur = (end_dt - start_dt).days / 30.4
        peak_i = seg["close"].idxmax()
        peak_close = float(seg["close"].max())
        start_close = float(w["close"].iloc[s_i])
        out.append({
            "start": str(start_dt.date()),
            "end": str(end_dt.date()),
            "duration_months": round(dur, 1),
            "peak_date": str(peak_i.date()),
            "peak_close": peak_close,
            "ret_to_peak_pct": round((peak_close / start_close - 1) * 100, 1) if start_close else None,
        })
    return out


def _delta_negative(op_growth: pd.Series | None, as_of: pd.Timestamp | None) -> tuple[bool | None, float | None]:
    """게이트 3 델타 — 영업이익 성장률의 변화(2차 미분).

    delta = growth(최근분기) - growth(직전분기). 0 미만이면 음전환(경계 플래그, 매도 아님).
    데이터 없으면 (None, None) → 신뢰도 하향.
    """
    if op_growth is None or len(op_growth) < 2:
        return None, None
    g = op_growth.dropna()
    # tz 정규화(as_of와 비교 위해 둘 다 naive). US/yfinance는 tz-aware 흔함.
    try:
        if getattr(g.index, "tz", None) is not None:
            g = g.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    if as_of is not None:
        try:
            g = g[g.index <= as_of]
        except TypeError:
            return None, None   # tz/타입 불일치 → 비교 불가 시 안전하게 데이터 없음 처리
    g = g.sort_index()
    if len(g) < 2:
        return None, None
    delta = float(g.iloc[-1]) - float(g.iloc[-2])
    return (delta < 0), round(delta, 2)


def diagnose_leader_cycle(
    weekly: pd.DataFrame,
    *,
    market: str = "KR",
    op_growth: pd.Series | None = None,
    as_of: str | pd.Timestamp | None = None,
    params: dict | None = None,
    ticker: str = "",
    name: str = "",
) -> dict:
    """주도주 사이클 진단 (지시서 §3~§5).

    Args:
        weekly: 주봉 OHLCV. DatetimeIndex + [open,high,low,close,volume] (대/소문자 무관).
        market: "KR" | "US" (US는 생존율 곡선 ×us_stretch).
        op_growth: 분기 영업이익 성장률(YoY%) 시계열(index=분기말 date). 선택 — 델타 게이트용.
        as_of: 진단 기준일(백테스트용). None이면 데이터 마지막 주.
        params: DEFAULT_PARAMS 오버라이드.
        ticker/name: 표기용.

    Returns:
        dict — 지시서 §4 출력 스키마 + reasons/limitations/confidence.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    out: dict = {"ticker": ticker, "name": name, "market": market}

    try:
        w = _normalize_weekly(weekly)
    except Exception as e:  # noqa: BLE001
        return {**out, "data_available": False, "error": f"주봉 정규화 실패: {e}"}

    if as_of is not None:
        try:
            w = w[w.index <= pd.to_datetime(as_of)]
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
            pass   # 파싱 불가한 as_of(범위초과 등)는 무시 → 최신 기준
    if w is None or w.empty or "close" not in w.columns:
        return {**out, "data_available": False, "error": "주봉 close 데이터 없음"}
    if len(w) < p["min_weeks"]:
        return {**out, "data_available": False,
                "error": f"주봉 {len(w)}주 부족(>= {p['min_weeks']} 필요: MA{p['ma'][2]} + 역추적)"}

    as_of_ts = w.index[-1]
    cyc = _trace_cycle(w, p)
    delta_neg, delta_val = _delta_negative(op_growth, as_of_ts)

    out.update({
        "data_available": True,
        "as_of": str(as_of_ts.date()),
        "ma": {"ma5": cyc["ma_s"], "ma20": cyc["ma_m"], "ma60": cyc["ma_l"],
               "labels": list(p["ma"])},
        "last_close": float(w["close"].iloc[-1]),
        "delta_negative": delta_neg,
        "delta_value": delta_val,
    })

    # ── 사이클 비활성: 비주도주(가설 미적용) ──
    if not cyc["active"]:
        out.update({
            "now_aligned": False,
            "cycle_start": None,
            "age_months": None,
            "clock": "사이클없음",
            "survival_pct": None,
            "trend_intact": False,
            "mdd_from_high": None,
            "signal": "해당없음",
            "reasons": [
                "주봉 정배열 미성립(또는 사이클 종료) → 비주도주 구간, 본 가설 적용 금지.",
                f"최근 정배열 주: {cyc['last_aligned_week']}" if cyc["last_aligned_week"] else "정배열 이력 없음(관측 기간 내).",
            ],
            "confidence": "low" if delta_neg is None else "mid",
            "limitations": LIMITATIONS,
        })
        return out

    # ── 사이클 활성 ──
    cycle_start = cyc["cycle_start"]
    age_months = (as_of_ts - cycle_start).days / 30.4
    mult = p["us_stretch"] if market == "US" else 1.0
    age_adj = age_months / mult
    clock = _clock(age_adj, p)
    survival = _survival_pct(age_adj, p)

    # 추세 건강도 (게이트 3)
    seg = w[w.index >= cycle_start]
    high_since = float(seg["close"].max())
    last_close = float(w["close"].iloc[-1])
    mdd = (last_close - high_since) / high_since if high_since > 0 else 0.0
    ma20_breach = (cyc["ma_m"] is not None) and (last_close < cyc["ma_m"])
    align_broken = not cyc["loose_intact"]
    trend_break = ma20_breach and align_broken           # 주봉 추세 붕괴 = 청산 트리거
    trend_intact = cyc["loose_intact"] and not (mdd < p["mdd_threshold"] and ma20_breach)

    # ── 신호 규칙 (우선순위 순, 지시서 §5) ──
    # 청산 age 임계는 age_adj 기준: US는 stretch(÷1.25) 적용되어 raw 30개월(=2.5년)에 청산.
    # 사양 §1 "미국 ~2.5년"과 일치. KR은 mult=1이라 raw 24개월.
    reasons: list[str] = []
    if trend_break or age_adj > p["clock_close"]:
        signal = "청산"
        if trend_break:
            reasons.append("주봉 추세 붕괴: MA20 종가 이탈 + 정배열 붕괴.")
        if age_adj > p["clock_close"]:
            reasons.append(f"사이클 나이 {age_months:.1f}개월 > 24개월(보정 {age_adj:.1f}) — 종료 임박.")
    elif age_adj > p["clock_afternoon"] or delta_neg:
        signal = "경계"
        if age_adj > p["clock_afternoon"]:
            reasons.append(f"사이클 {age_months:.1f}개월 — 오후 구간 진입(신규 진입 금지, 보유는 가능).")
        if delta_neg:
            reasons.append(f"델타 음전환(영업이익 성장 둔화, Δ={delta_val}) — 경계 플래그.")
    elif age_adj <= p["buy_window_months"] and (delta_neg is not True) and trend_intact:
        signal = "매수적기"
        delta_note = "델타 양수" if delta_neg is False else "델타 데이터 없음(추세 기준)"
        reasons.append(f"정배열 성립 {age_months:.1f}개월(오전) + {delta_note} + 추세 견조.")
    else:
        signal = "보유"
        reasons.append(f"사이클 {age_months:.1f}개월, 추세 살아있음 — 추세 무너지기 전까지 끌고 간다.")
        if delta_neg is None:
            reasons.append("델타 데이터 없음 → 추세만으로 운용(신뢰도 하향).")

    # 보조 reason
    reasons.append(f"정배열 성립일 {cycle_start.date()} (준비기 제외 카운트).")
    if ma20_breach:
        reasons.append("MA20 주봉 종가 이탈 — 추세 경고.")
    reasons.append(f"고점 대비 낙폭 {mdd*100:.1f}% "
                   f"({'정상 조정' if mdd >= p['mdd_threshold'] else '위험 구간'}).")

    out.update({
        "now_aligned": True,
        "cycle_start": str(cycle_start.date()),
        "age_months": round(age_months, 1),
        "clock": clock,
        "survival_pct": survival,
        "trend_intact": bool(trend_intact),
        "mdd_from_high": round(mdd, 4),
        "signal": signal,
        "reasons": reasons,
        "confidence": "low" if delta_neg is None else "high",
        "limitations": LIMITATIONS,
    })
    return out
