"""
전체 종목 매수 후보 스캔 -> 순위 -> 텔레그램 발송

v5.0: 4축 100점 (Quant 30 + SD 25 + News 25 + Consensus 20)
v9.0: C+E Hybrid Kill→Rank→Tag (선행 100%, 후행 0%)

사용법:
    python scripts/scan_buy_candidates.py --grade AB                     # v5.0 (4축 100점)
    python scripts/scan_buy_candidates.py --v9 --grade AB                # v9.0 C+E Hybrid
    python scripts/scan_buy_candidates.py --v9 --grade AB --no-news --no-send  # 빠른 테스트
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.market_signal_scanner import MarketSignalScanner

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# =========================================================
# 4축 점수 체계 (100점)
# =========================================================

def calc_composite_score(sig: dict) -> dict:
    """
    4축 종합 점수 계산 (v5.0: Consensus 축 추가).

    1) Quant Score (30점) - 퀀텀전략 핵심
       Zone Score(15) + Trigger(6) + R:R(5) + Trend(4)

    2) Supply/Demand Score (25점) - 수급
       외국인(8) + 기관(8) + OBV/거래량(4) + ADX방향(5)

    3) News Score (25점) - Grok 뉴스
       전반감성(7) + 뉴스영향(7) + 살아있는이슈(5) + 실적전망(6)

    4) Consensus Score (20점) - v5.0 Sci-CoE 합의
       Geometric Reward(10) + Consistency(4) + Reliability(3) + Diversity(3)
    """
    scores = {}

    # ── 1. Quant Score (30점) ──
    q_zone = min(sig.get("zone_score", 0) * 15, 15)  # 0~1 float → 0~15

    # Trigger 품질 (6점)
    trigger_type = sig.get("trigger_type", "none")
    confidence = sig.get("confidence", 0)
    if trigger_type == "impulse":
        q_trigger = 6 if sig.get("impulse_met", 0) >= 4 else 5
    elif trigger_type == "confirm":
        q_trigger = 4 if sig.get("confirm_met", 0) >= 4 else 3
    elif trigger_type.startswith("T1_"):
        # v8 T1_TRIX_Golden: 모멘텀 전환 → impulse급
        q_trigger = 6 if confidence >= 0.7 else 5
    elif trigger_type.startswith("T2_") or trigger_type.startswith("T3_"):
        # v8 T2_Volume_RSI, T3_Curvature_OBV → confirm급
        q_trigger = 4 if confidence >= 0.6 else 3
    else:
        q_trigger = 0

    # Market Signal 보너스 (trigger 내 포함)
    ms_list = sig.get("market_signals", [])
    critical_count = sum(1 for m in ms_list if m.get("importance") in ("critical", "high"))
    if critical_count >= 2:
        q_trigger = min(q_trigger + 1, 6)

    # R:R (5점)
    rr = sig.get("risk_reward", 0)
    if rr >= 8:
        q_rr = 5
    elif rr >= 5:
        q_rr = 4
    elif rr >= 3:
        q_rr = 3
    elif rr >= 2:
        q_rr = 2
    elif rr >= 1.5:
        q_rr = 1
    else:
        q_rr = 0

    # Trend (4점)
    trend = sig.get("trend", "unknown")
    q_trend = {"strong_up": 4, "up": 3, "neutral": 2, "down": 0}.get(trend, 0)

    scores["quant"] = round(q_zone + q_trigger + q_rr + q_trend, 1)
    scores["q_detail"] = {
        "zone": round(q_zone, 1),
        "trigger": round(q_trigger, 1),
        "rr": round(q_rr, 1),
        "trend": round(q_trend, 1),
    }

    # ── 2. Supply/Demand Score (30점) ──

    # 외국인 수급 (10점)
    f_streak = sig.get("foreign_streak", 0)
    f_5d = sig.get("foreign_amount_5d", 0)
    sd_foreign = 0
    if f_streak >= 5:
        sd_foreign = 10
    elif f_streak >= 3:
        sd_foreign = 7
    elif f_streak >= 2:
        sd_foreign = 5
    elif f_streak == 1 and f_5d > 0:
        sd_foreign = 3
    elif f_5d > 0:
        sd_foreign = 1

    # 기관 수급 (10점)
    i_streak = sig.get("inst_streak", 0)
    i_5d = sig.get("inst_amount_5d", 0)
    sd_inst = 0
    if i_streak >= 5:
        sd_inst = 10
    elif i_streak >= 3:
        sd_inst = 7
    elif i_streak >= 2:
        sd_inst = 5
    elif i_streak == 1 and i_5d > 0:
        sd_inst = 3
    elif i_5d > 0:
        sd_inst = 1

    # OBV + 거래량 (5점)
    sd_obv = 0
    if sig.get("obv_trend") == "up":
        sd_obv += 3
    vol = sig.get("vol_surge", 1.0)
    if vol >= 2.0:
        sd_obv += 2
    elif vol >= 1.5:
        sd_obv += 1

    # ADX 방향성 (5점)
    sd_adx = 0
    adx = sig.get("adx", 0)
    if adx >= 25 and sig.get("plus_di", 0) > sig.get("minus_di", 0):
        sd_adx = 5
    elif adx >= 20 and sig.get("plus_di", 0) > sig.get("minus_di", 0):
        sd_adx = 3
    elif sig.get("plus_di", 0) > sig.get("minus_di", 0):
        sd_adx = 1

    scores["supply_demand"] = round(min(sd_foreign + sd_inst + sd_obv + sd_adx, 25), 1)
    scores["sd_detail"] = {
        "foreign": sd_foreign,
        "inst": sd_inst,
        "obv_vol": sd_obv,
        "adx_dir": sd_adx,
    }

    # ── 3. News Score (25점) ──
    news_data = sig.get("news_data")
    if news_data:
        ns, ns_detail = _calc_news_score(news_data)
    else:
        ns = 0
        ns_detail = {"sentiment": 0, "impact": 0, "living": 0, "earnings": 0}
    scores["news"] = round(min(ns, 25), 1)
    scores["news_detail"] = ns_detail

    # ── 4. Consensus Score (20점, v5.0) ──
    consensus = sig.get("consensus")
    if consensus:
        # Geometric Reward (10점): 0~1 → 0~10
        c_reward = min(consensus.get("geometric_reward", 0) * 10, 10)
        # Consistency (4점): 0~1 → 0~4
        c_consistency = min(consensus.get("consistency", 0) * 4, 4)
        # Reliability (3점): 0~1 → 0~3
        c_reliability = min(consensus.get("reliability", 0) * 3, 3)
        # Diversity (3점): 0~1 → 0~3
        c_diversity = min(consensus.get("diversity", 0) * 3, 3)
        c_total = round(c_reward + c_consistency + c_reliability + c_diversity, 1)
        scores["consensus"] = min(c_total, 20)
        scores["consensus_detail"] = {
            "reward": round(c_reward, 1),
            "consistency": round(c_consistency, 1),
            "reliability": round(c_reliability, 1),
            "diversity": round(c_diversity, 1),
            "grade": consensus.get("grade", "reject"),
        }
    else:
        scores["consensus"] = 0
        scores["consensus_detail"] = {
            "reward": 0, "consistency": 0, "reliability": 0, "diversity": 0,
            "grade": "N/A",
        }

    # ── 합산 ──
    scores["total"] = round(
        scores["quant"] + scores["supply_demand"]
        + scores["news"] + scores["consensus"], 1
    )

    return scores


def _calc_news_score(news_data: dict) -> tuple[float, dict]:
    """Grok 뉴스 분석 결과 -> 0~30점 변환."""
    detail = {"sentiment": 0, "impact": 0, "living": 0, "earnings": 0}

    # 전반적 sentiment (8점)
    sentiment = news_data.get("overall_sentiment", "중립")
    if sentiment in ("긍정", "positive"):
        detail["sentiment"] = 8
    elif sentiment in ("중립", "neutral"):
        detail["sentiment"] = 3
    else:
        detail["sentiment"] = 0

    # 최신 뉴스 영향도 (8점)
    news_list = news_data.get("latest_news", [])
    pos_high = sum(
        1 for n in news_list
        if n.get("sentiment") in ("positive", "긍정") and n.get("impact_score", 0) >= 7
    )
    if pos_high >= 3:
        detail["impact"] = 8
    elif pos_high >= 2:
        detail["impact"] = 6
    elif pos_high >= 1:
        detail["impact"] = 3
    else:
        neg_count = sum(1 for n in news_list if n.get("sentiment") in ("negative", "부정"))
        if neg_count >= 2:
            detail["impact"] = -3

    # 살아있는 이슈 (7점)
    living = news_data.get("living_issues", [])
    active_pos = sum(
        1 for li in living
        if li.get("status") == "active" and li.get("sentiment") in ("positive", "긍정")
    )
    if active_pos >= 3:
        detail["living"] = 7
    elif active_pos >= 2:
        detail["living"] = 5
    elif active_pos >= 1:
        detail["living"] = 3

    # 실적 전망 (7점)
    earnings = news_data.get("earnings_estimate", {})
    if earnings.get("surprise_direction") == "beat":
        detail["earnings"] = 7
    elif earnings.get("surprise_direction") == "in_line" and earnings.get("yoy_growth_pct", 0) > 20:
        detail["earnings"] = 4
    elif earnings.get("yoy_growth_pct", 0) > 10:
        detail["earnings"] = 2

    total = sum(detail.values())
    return max(total, 0), detail


# =========================================================
# v9.0 C+E 하이브리드 파이프라인
# =========================================================

def detect_regime_v9() -> dict:
    """Layer 0: 7D 레짐 Gate — 공매도 상태에 따라 Kill 기준값 결정."""
    import yaml
    from datetime import date

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    today = date.today()
    calendar = cfg.get("short_selling_calendar", [])
    status = "active"

    for period in calendar:
        start = date.fromisoformat(str(period["start"]))
        end = date.fromisoformat(str(period["end"]))
        if start <= today <= end:
            status = period["status"]
            break

    # 공매도 허용/재개: 엄격, 금지: 완화
    if status in ("active", "reopened"):
        return {"status": status, "zone_th": 7 / 15, "rr_th": 2.0}
    else:  # banned
        return {"status": status, "zone_th": 5 / 15, "rr_th": 1.5}


def kill_filters_v9(sig: dict, regime: dict) -> tuple[bool, list[str]]:
    """Layer 1: Kill Filters (5개, 하나라도 걸리면 탈락)."""
    kills = []

    # K1: Zone < regime.zone_threshold
    zone = sig.get("zone_score", 0)
    if zone < regime["zone_th"]:
        kills.append(f"K1:Zone({zone:.2f}<{regime['zone_th']:.2f})")

    # K2: R:R < regime.rr_threshold
    rr = sig.get("risk_reward", 0)
    if rr < regime["rr_th"]:
        kills.append(f"K2:RR({rr:.1f}<{regime['rr_th']:.1f})")

    # K3: Trigger 등급 D (미발동)
    trigger = sig.get("trigger_type", "none")
    if trigger in ("none", "waiting", "setup"):
        kills.append(f"K3:Trigger({trigger})")

    # K4: 20일 평균 거래대금 < 10억
    avg_tv = sig.get("avg_trading_value_20d", 0)
    if avg_tv < 1_000_000_000:
        kills.append(f"K4:유동성({avg_tv / 1e8:.0f}억<10억)")

    # K5: 52주 고점 대비 -5% 이내
    pct_high = sig.get("pct_of_52w_high", 0)
    if pct_high > 0.95:
        kills.append(f"K5:고점근접({pct_high:.1%})")

    return len(kills) == 0, kills


def trap_filter_v9(sig: dict) -> tuple[bool, str]:
    """Layer 2: 6D 함정 필터 — 기술적 근거 없이 수급+뉴스만 좋은 종목 탈락."""
    scores = sig.get("scores", {})
    quant = scores.get("quant", 0)
    sd = scores.get("supply_demand", 0)
    news = scores.get("news", 0)

    if quant < 18 and sd >= 20 and news >= 15:
        return True, f"TRAPPED(Q{quant:.0f}<18 & SD{sd:.0f}>=20 & N{news:.0f}>=15)"
    return False, ""


def generate_v9_tags(sig: dict) -> list[str]:
    """Layer 4: 정보 태그 (순위에 영향 없음, tiebreaker만)."""
    tags = []
    scores = sig.get("scores", {})

    # SD 태그
    sd = scores.get("supply_demand", 0)
    if sd >= 20:
        tags.append("수급전환")
    elif sd >= 10:
        tags.append("초기전환")
    elif sd >= 5:
        tags.append("수급미약")

    # News 태그
    news = scores.get("news", 0)
    if news >= 15:
        tags.append("이슈+실적")
    elif news >= 5:
        tags.append("이슈존재")

    # Consensus 태그
    cons = scores.get("consensus", 0)
    if cons >= 15:
        tags.append("강한상향")
    elif cons >= 10:
        tags.append("상향")

    # 외국인/기관 연속 매수
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    if f_streak >= 5:
        tags.append(f"외{f_streak}D연속")
    if i_streak >= 5:
        tags.append(f"기{i_streak}D연속")

    return tags


def run_v9_pipeline(
    candidates: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """v9.0 C+E 하이브리드 파이프라인 오케스트레이터.

    Layer 0 → 1 → 2 → 3(순위) → 4(태그).
    반환: (survivors, killed_list, trapped_list)
    """
    # Layer 0: 레짐 감지
    regime = detect_regime_v9()
    print(f"  v9 Regime: {regime['status']} (zone_th={regime['zone_th']:.3f}, rr_th={regime['rr_th']:.1f})")

    # 4축 점수 계산 (함정 필터용)
    for sig in candidates:
        sig["scores"] = calc_composite_score(sig)

    killed_list = []
    trapped_list = []
    survivors = []

    for sig in candidates:
        # Layer 1: Kill Filters
        passed, kill_reasons = kill_filters_v9(sig, regime)
        if not passed:
            sig["v9_kill_reasons"] = kill_reasons
            killed_list.append(sig)
            continue

        # Layer 2: 6D 함정 필터
        is_trapped, trap_reason = trap_filter_v9(sig)
        if is_trapped:
            sig["v9_trap_reason"] = trap_reason
            trapped_list.append(sig)
            continue

        survivors.append(sig)

    # Layer 3: 최종 순위 (rank = R:R × zone_score × catalyst_boost)
    for sig in survivors:
        zone = sig.get("zone_score", 0)
        rr = sig.get("risk_reward", 0)

        # 촉매 부스트
        catalyst_boost = 1.0
        news_data = sig.get("news_data")
        if news_data:
            earnings = news_data.get("earnings_estimate", {})
            if earnings.get("surprise_direction") == "beat":
                catalyst_boost = 1.10

        sig["v9_rank_score"] = round(rr * zone * catalyst_boost, 4)
        sig["v9_catalyst_boost"] = catalyst_boost

    # Layer 4: 태그
    for sig in survivors:
        sig["v9_tags"] = generate_v9_tags(sig)

    # 순위 정렬
    survivors.sort(key=lambda s: s["v9_rank_score"], reverse=True)

    return survivors, killed_list, trapped_list


# =========================================================
# Grok 뉴스 검색
# =========================================================

def fetch_grok_news(name: str, ticker: str) -> dict | None:
    """Grok API로 종목 심층 분석 (동기 직접 호출)."""
    try:
        import requests as _req
        from dotenv import load_dotenv as _ld
        _ld(Path(__file__).resolve().parent.parent / ".env")

        from src.adapters.grok_news_adapter import GrokNewsAdapter
        adapter = GrokNewsAdapter()
        if not adapter.api_key:
            return None

        prompt = adapter._deep_analysis_prompt(name, ticker)
        payload = {
            "model": "grok-4-1-fast",
            "input": [
                {"role": "system", "content": (
                    "너는 한국 주식시장 전문 리서치 애널리스트다. "
                    "웹과 X(트위터)를 검색해서 종목의 최신 뉴스뿐 아니라, "
                    "아직 해소되지 않은 과거 이슈, 실적 전망, 수급 동향을 "
                    "종합적으로 분석한다. 반드시 요청된 JSON 형식으로만 응답한다."
                )},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
        }
        resp = _req.post(
            "https://api.x.ai/v1/responses",
            headers=adapter.headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        return adapter._parse_response(resp.json())
    except Exception as e:
        print(f"  ! Grok fail ({name}): {e}")
        return None


# =========================================================
# 데이터 로드 + SignalEngine Pipeline
# =========================================================

def load_all_parquets() -> dict[str, pd.DataFrame]:
    """data/processed/*.parquet 로드 → {ticker: DataFrame}"""
    data = {}
    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq)
        if len(df) >= 200:
            data[pq.stem] = df
    return data


def load_name_map() -> dict:
    """stock_data_daily CSV 파일명에서 종목명 추출."""
    name_map = {}
    csv_dir = Path(__file__).resolve().parent.parent / "stock_data_daily"
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            match = re.search(r"_(\d{6})$", f.stem)
            if match:
                ticker = match.group(1)
                name = f.stem[: f.stem.rfind("_")]
                name_map[ticker] = name
    return name_map


def _calc_di(df: pd.DataFrame, idx: int, period: int = 14) -> tuple[float, float]:
    """ADX의 +DI, -DI를 직접 계산 (parquet에 미포함이므로)."""
    if len(df) < period + 2 or idx < period + 1:
        return 0.0, 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # 상대적으로 큰 쪽만 살림
    mask_plus = plus_dm < minus_dm
    mask_minus = minus_dm < plus_dm
    plus_dm = plus_dm.where(~mask_plus, 0)
    minus_dm = minus_dm.where(~mask_minus, 0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period).mean()
    p_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
    m_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

    return float(p_di.iloc[idx] or 0), float(m_di.iloc[idx] or 0)


def _classify_trend(row) -> str:
    """이평선 기반 추세 분류."""
    close = row.get("close", 0) or 0
    ma5 = row.get("sma_5", 0) or 0
    ma20 = row.get("sma_20", 0) or 0
    ma60 = row.get("sma_60", 0) or 0
    if close > ma5 and close > ma20 and close > ma60:
        return "strong_up"
    elif close > ma5 and close > ma20:
        return "up"
    elif close > ma60:
        return "neutral"
    return "down"


def _fit_regime(data_dict: dict):
    """전종목에 HMM 레짐 확률 추가 (main.py와 동일 패턴)."""
    try:
        from src.regime_detector import RegimeDetector
        detector = RegimeDetector()
        for ticker, df in data_dict.items():
            try:
                regime_proba = detector.fit_predict(df)
                for col in ["P_Advance", "P_Distrib", "P_Accum"]:
                    data_dict[ticker][col] = regime_proba[col]
            except Exception:
                for col in ["P_Advance", "P_Distrib", "P_Accum"]:
                    data_dict[ticker][col] = 1 / 3
    except ImportError:
        for df in data_dict.values():
            for col in ["P_Advance", "P_Distrib", "P_Accum"]:
                df[col] = 1 / 3


def _calc_streak(series: pd.Series) -> int:
    """최근 연속 순매수 일수 계산 (음수면 순매도 연속)."""
    if series.empty:
        return 0
    vals = series.values
    if pd.isna(vals[-1]) or vals[-1] == 0:
        return 0
    direction = 1 if vals[-1] > 0 else -1
    count = 0
    for v in reversed(vals):
        if pd.isna(v) or v == 0:
            break
        if (v > 0 and direction > 0) or (v < 0 and direction < 0):
            count += 1
        else:
            break
    return count * direction




# =========================================================
# 메인 스캔
# =========================================================

def scan_all(
    grade_filter: str = "A",
    use_news: bool = True,
    use_v9: bool = False,
) -> tuple[list[dict], dict]:
    """전 종목 스캔 -> Grade 필터 -> 점수/Kill -> 순위 반환.

    use_v9=True: v9.0 C+E Kill→Rank→Tag 파이프라인
    use_v9=False: 기존 v8.1 4축 100점 합산
    """
    from src.signal_engine import SignalEngine

    # 데이터 로드 (parquet)
    data_dict = load_all_parquets()
    name_map = load_name_map()
    print(f"scan: {len(data_dict)} stocks (parquet) | grade={grade_filter} | news={'ON' if use_news else 'OFF'}")

    # HMM 레짐 피팅
    print("HMM regime fitting...")
    _fit_regime(data_dict)

    # SignalEngine 초기화
    engine = SignalEngine("config/settings.yaml")
    scanner = MarketSignalScanner()

    candidates = []
    stats = {
        "total": len(data_dict),
        "loaded": 0,
        "passed_pipeline": 0,
        "trigger_impulse": 0,
        "trigger_confirm": 0,
        "grade_A": 0,
        "grade_B": 0,
        "grade_C": 0,
        "after_grade_filter": 0,
    }

    t0 = time.time()

    for i, (ticker, df) in enumerate(data_dict.items()):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(data_dict)} ({elapsed:.1f}s)")

        stats["loaded"] += 1
        idx = len(df) - 1

        try:
            result = engine.calculate_signal(ticker, df, idx)
        except Exception:
            continue

        if not result["signal"]:
            continue

        grade = result["grade"]
        stats["passed_pipeline"] += 1

        trigger_type = result.get("trigger_type", "none")
        if trigger_type == "impulse":
            stats["trigger_impulse"] += 1
        elif trigger_type == "confirm":
            stats["trigger_confirm"] += 1

        grade_key = f"grade_{grade}"
        stats[grade_key] = stats.get(grade_key, 0) + 1

        if grade not in grade_filter:
            continue

        stats["after_grade_filter"] += 1

        # DataFrame에서 4축 점수 계산용 필드 추출
        row = df.iloc[idx]
        name = name_map.get(ticker, ticker)

        # 수급 streak 계산
        foreign_streak = 0
        inst_streak = 0
        foreign_amount_5d = 0
        inst_amount_5d = 0

        if "foreign_net" in df.columns:
            f_series = df["foreign_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            foreign_streak = _calc_streak(f_series)
            foreign_amount_5d = int(df["foreign_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        if "inst_net" in df.columns:
            i_series = df["inst_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            inst_streak = _calc_streak(i_series)
            inst_amount_5d = int(df["inst_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        sig = {
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "zone_score": result["zone_score"],
            "trigger_type": trigger_type,
            "confidence": result.get("trigger_confidence", 0),
            "entry_price": result["entry_price"],
            "stop_loss": result["stop_loss"],
            "target_price": result["target_price"],
            "risk_reward": result.get("risk_reward_ratio", 0),
            # DataFrame에서 직접 추출
            "rsi": float(row.get("rsi_14", 50) or 50),
            "adx": float(row.get("adx_14", 0) or 0),
            "plus_di": 0.0,  # 아래에서 _calc_di로 계산
            "minus_di": 0.0,
            "vol_surge": float(row.get("volume_surge_ratio", 1.0) or 1.0),
            "obv_trend": (
                "up" if (row.get("obv", 0) or 0) > (df.iloc[max(0, idx - 20)].get("obv", 0) or 0)
                else "down"
            ),
            "trend": _classify_trend(row),
            # 수급
            "foreign_streak": foreign_streak,
            "inst_streak": inst_streak,
            "foreign_amount_5d": foreign_amount_5d,
            "inst_amount_5d": inst_amount_5d,
            # SignalEngine 고급 필드
            "consensus": result.get("consensus"),
            # v9 필수 필드
            "avg_trading_value_20d": float(
                (df["close"] * df["volume"]).iloc[max(0, idx - 19) : idx + 1].mean()
            ),
            "pct_of_52w_high": float(row.get("pct_of_52w_high", 0) or 0),
        }

        # +DI/-DI 직접 계산 (parquet에 미포함)
        try:
            p_di, m_di = _calc_di(df, idx)
            sig["plus_di"] = p_di
            sig["minus_di"] = m_di
        except Exception:
            pass

        # Market Signal Scanner
        try:
            market_signals = scanner.scan_all(df, idx)
            sig["market_signals"] = [
                {"title": s.title, "importance": s.importance, "confidence": s.confidence}
                for s in market_signals
            ] if market_signals else []
        except Exception:
            sig["market_signals"] = []

        candidates.append(sig)

    scan_elapsed = time.time() - t0
    stats["scan_sec"] = round(scan_elapsed, 1)

    # -- Grok 뉴스 적용 --
    if use_news and candidates:
        print(f"\nGrok news ({len(candidates)} stocks)...")
        news_t0 = time.time()
        for sig in candidates:
            print(f"  {sig['name']}({sig['ticker']})...", end=" ", flush=True)
            news_data = fetch_grok_news(sig["name"], sig["ticker"])
            sig["news_data"] = news_data
            if news_data:
                sentiment = news_data.get("overall_sentiment", "?")
                takeaway = news_data.get("key_takeaway", "")[:30]
                print(f"OK [{sentiment}] {takeaway}")
            else:
                print("- (empty)")
        stats["news_sec"] = round(time.time() - news_t0, 1)
    else:
        for sig in candidates:
            sig["news_data"] = None
        stats["news_sec"] = 0

    # -- 점수 계산 + 정렬 --
    if use_v9:
        survivors, killed, trapped = run_v9_pipeline(candidates)
        stats["v9_killed"] = len(killed)
        stats["v9_trapped"] = len(trapped)
        stats["v9_survivors"] = len(survivors)
        stats["elapsed_sec"] = round(time.time() - t0, 1)
        # v9: killed/trapped 정보를 stats에 저장 (텔레그램/HTML용)
        stats["v9_killed_list"] = killed
        stats["v9_trapped_list"] = trapped
        return survivors, stats
    else:
        for sig in candidates:
            sig["scores"] = calc_composite_score(sig)
        candidates.sort(key=lambda s: s["scores"]["total"], reverse=True)
        stats["elapsed_sec"] = round(time.time() - t0, 1)
        return candidates, stats


# =========================================================
# 텔레그램 메시지 포맷 (KISBOT v5.0 스타일)
# =========================================================

LINE = "\u2500" * 28  # ────────────────────────────


def _supply_tag(streak: int, amount_5d: int, label: str) -> str:
    """수급 태그 생성."""
    if streak == 0 and amount_5d == 0:
        return ""
    if streak >= 3:
        return f"{label}{streak}D+"
    elif streak >= 1:
        return f"{label}{streak}D"
    elif amount_5d > 0:
        return f"{label}+"
    elif amount_5d < 0:
        return f"{label}-"
    return ""


def format_telegram_message(candidates: list[dict], stats: dict) -> str:
    """KISBOT 스타일 텔레그램 메시지 포맷 (작은 이모지)."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # -- Header --
    lines.append(f"[Quant v5.0] {now} 4-Axis \uc2a4\uce94")
    lines.append("")

    # -- 점수 체계 --
    lines.append("[ \uc810\uc218 \uccb4\uacc4 ]")
    lines.append("4\ucd95: Quant(30) + \uc218\uae09(25) + \ub274\uc2a4(25) + Consensus(20) = 100")
    lines.append("  Q: Zone(15)+\ud2b8\ub9ac\uac70(6)+\uc190\uc775\ube44(5)+\ucd94\uc138(4)")
    lines.append("  SD: \uc678\uad6d\uc778(10)+\uae30\uad00(10)+OBV/\uac70\ub798\ub7c9(5)")
    lines.append("  N: \uac10\uc131(8)+\uc601\ud5a5(8)+\uc774\uc288(7)+\uc2e4\uc801(7)")
    lines.append("  C: Reward(10)+Consist(4)+Rely(3)+Diver(3)")
    lines.append("")

    # -- 스캔 통계 --
    lines.append("[ \uc2a4\uce94 \ud1b5\uacc4 ]")
    lines.append(f"\uc804\uccb4: {stats['total']:,}\uc885\ubaa9 > Pipeline: {stats['passed_pipeline']}\uc885\ubaa9")
    lines.append(
        f"\ub4f1\uae09: A:{stats.get('grade_A',0)} B:{stats.get('grade_B',0)} "
        f"C:{stats.get('grade_C',0)} | \ud544\ud130 \ud6c4: {stats.get('after_grade_filter',0)}\uc885\ubaa9"
    )
    lines.append(f"\uc18c\uc694: \uc2a4\uce94 {stats.get('scan_sec',0)}\ucd08 + \ub274\uc2a4 {stats.get('news_sec',0)}\ucd08")
    lines.append("")

    if not candidates:
        lines.append("\ub9e4\uc218 \uc2dc\uadf8\ub110 \ubc1c\ub3d9 \uc885\ubaa9 \uc5c6\uc74c")
        return "\n".join(lines)

    # -- 1순위 추천 매수 --
    top = candidates[0]
    top_sc = top["scores"]
    top_trigger = "\ud655\uc778\ub9e4\uc218" if top["trigger_type"] == "confirm" else "IMP"
    top_news = top.get("news_data")

    # 수급 태그
    top_supply = _format_supply_tag(top)

    lines.append(f"[ \U0001f3c6 1\uc21c\uc704 \ucd94\ucc9c \ub9e4\uc218 ]")
    lines.append(f"{top['name']} ({top['ticker']}) [{top_trigger}]{top_supply}")
    lines.append(f"\ud604\uc7ac {top['entry_price']:,}\uc6d0 | {top_sc['total']:.0f}\uc810 | \ub274\uc2a4 {_news_icon(top_news)}")
    loss_pct = ((top['stop_loss'] / top['entry_price']) - 1) * 100
    lines.append(f"\ubaa9\ud45c {top['target_price']:,} (+{((top['target_price']/top['entry_price'])-1)*100:.1f}%) / \uc190\uc808 {top['stop_loss']:,} ({loss_pct:.1f}%)")
    lines.append(f"\uc190\uc775\ube44 1:{top['risk_reward']:.1f} | RSI {top['rsi']:.0f} | ADX {top['adx']:.0f}")

    # 긍정요소 + 뉴스
    positives = _collect_positives(top)
    if positives:
        lines.append(f"\U0001f44d {positives}")
    if top_news:
        takeaway = top_news.get("key_takeaway", "")
        if takeaway:
            lines.append(f"\u25b8 {takeaway[:60]}")

    # -- 매수 후보 --
    if len(candidates) > 1:
        lines.append("")
        lines.append(f"[ \ub9e4\uc218 \ud6c4\ubcf4 ({len(candidates)-1}\uac1c) ]")
        for i, sig in enumerate(candidates[1:], start=2):
            sc = sig["scores"]
            trigger = "IMP" if sig["trigger_type"] == "impulse" else "\ud655\uc778\ub9e4\uc218"
            supply = _format_supply_tag(sig)
            news = sig.get("news_data")

            lines.append(f"{i}. {sig['name']} ({sig['ticker']}){supply} {trigger}")
            lines.append(f"   {sig['entry_price']:,}\uc6d0 | {sc['total']:.0f}\uc810 | \ub274\uc2a4 {_news_icon(news)} | RR 1:{sig['risk_reward']:.1f}")

            positives = _collect_positives(sig)
            warnings = _collect_warnings(sig)
            line_parts = []
            if positives:
                line_parts.append(f"\U0001f44d {positives}")
            if warnings:
                line_parts.append(f"\u26a0\ufe0f {warnings}")
            if line_parts:
                lines.append(f"   {' | '.join(line_parts)}")

    # -- 전체 후보 한줄 요약 --
    lines.append("")
    lines.append(f"[ \ucc38\uace0: \uc804\uccb4 \ud6c4\ubcf4 (\uc810\uc218\uc21c) ]")
    for i, sig in enumerate(candidates, start=1):
        sc = sig["scores"]
        supply = _format_supply_tag(sig)
        lines.append(
            f"{i}. {sig['name']}({sig['ticker']}) "
            f"{sig['entry_price']:,} ({sc['total']:.0f}) "
            f"Q{sc['quant']:.0f} SD{sc['supply_demand']:.0f} N{sc['news']:.0f}"
            f"{supply}"
        )

    return "\n".join(lines)


def format_telegram_message_v9(candidates: list[dict], stats: dict) -> str:
    """v9.0 C+E 하이브리드 텔레그램 메시지 포맷."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # -- Header --
    lines.append(f"[Quant v9.0] {now} C+E Hybrid")
    lines.append("")

    # -- 파이프라인 설명 --
    lines.append("[ 파이프라인 ]")
    lines.append("Kill(5) \u2192 Trap(6D) \u2192 Rank(R:R\u00d7Zone) \u2192 Tag")
    lines.append("  \u00b7 선행 100%: Zone + R:R + Trigger")
    lines.append("  \u00b7 후행 0%: SD/News/Consensus \u2192 태그만")
    lines.append("")

    # -- 스캔 통계 --
    killed = stats.get("v9_killed", 0)
    trapped = stats.get("v9_trapped", 0)
    survivors = stats.get("v9_survivors", 0)
    lines.append("[ 스캔 통계 ]")
    lines.append(
        f"전체: {stats['total']:,}종목 > Pipeline: {stats['passed_pipeline']}종목"
    )
    lines.append(
        f"등급: A:{stats.get('grade_A',0)} B:{stats.get('grade_B',0)} "
        f"C:{stats.get('grade_C',0)} | 필터 후: {stats.get('after_grade_filter',0)}종목"
    )
    lines.append(f"Kill: {killed}종목 | Trap: {trapped}종목 | 생존: {survivors}종목")
    lines.append(f"소요: 스캔 {stats.get('scan_sec',0)}초 + 뉴스 {stats.get('news_sec',0)}초")
    lines.append("")

    if not candidates:
        lines.append("v9 Kill 필터 통과 종목 없음")
        # Kill/Trap 상세를 보여주고 종료
        killed_list = stats.get("v9_killed_list", [])
        trapped_list = stats.get("v9_trapped_list", [])
        if killed_list:
            lines.append("")
            lines.append(f"[ Kill ({len(killed_list)}종목) ]")
            for sig in killed_list:
                reasons = ", ".join(sig.get("v9_kill_reasons", []))
                lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        if trapped_list:
            lines.append("")
            lines.append(f"[ Trap ({len(trapped_list)}종목) ]")
            for sig in trapped_list:
                lines.append(f"  {sig['name']}({sig['ticker']}): {sig.get('v9_trap_reason','')}")
        return "\n".join(lines)

    # -- 1순위 추천 매수 --
    top = candidates[0]
    top_trigger = "확인매수" if top["trigger_type"] == "confirm" else "IMP"
    top_tags = ", ".join(top.get("v9_tags", []))
    boost = top.get("v9_catalyst_boost", 1.0)
    boost_str = " x1.10촉매" if boost > 1.0 else ""

    lines.append("[ 1순위 추천 매수 ]")
    lines.append(f"{top['name']} ({top['ticker']}) [{top_trigger}]")
    lines.append(
        f"Rank {top['v9_rank_score']:.3f} = "
        f"R:R({top['risk_reward']:.1f}) x Zone({top['zone_score']:.2f}){boost_str}"
    )
    lines.append(
        f"현재 {top['entry_price']:,}원 | "
        f"목표 {top['target_price']:,} (+{((top['target_price']/top['entry_price'])-1)*100:.1f}%) | "
        f"손절 {top['stop_loss']:,} ({((top['stop_loss']/top['entry_price'])-1)*100:.1f}%)"
    )
    if top_tags:
        lines.append(f"태그: {top_tags}")

    # -- 나머지 후보 --
    if len(candidates) > 1:
        lines.append("")
        lines.append(f"[ 매수 후보 ({len(candidates)-1}개) ]")
        for i, sig in enumerate(candidates[1:], start=2):
            tags = ", ".join(sig.get("v9_tags", []))
            b = sig.get("v9_catalyst_boost", 1.0)
            b_str = " x1.10" if b > 1.0 else ""
            lines.append(
                f"{i}. {sig['name']}({sig['ticker']}) "
                f"Rank {sig['v9_rank_score']:.3f} "
                f"RR:{sig['risk_reward']:.1f} Zone:{sig['zone_score']:.2f}{b_str}"
            )
            if tags:
                lines.append(f"   [{tags}]")

    # -- Kill/Trap 요약 --
    killed_list = stats.get("v9_killed_list", [])
    trapped_list = stats.get("v9_trapped_list", [])
    if killed_list:
        lines.append("")
        lines.append(f"[ Kill ({len(killed_list)}종목) ]")
        for sig in killed_list[:5]:
            reasons = ", ".join(sig.get("v9_kill_reasons", []))
            lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        if len(killed_list) > 5:
            lines.append(f"  ... +{len(killed_list)-5}종목")

    if trapped_list:
        lines.append("")
        lines.append(f"[ Trap ({len(trapped_list)}종목) ]")
        for sig in trapped_list:
            lines.append(f"  {sig['name']}({sig['ticker']}): {sig.get('v9_trap_reason','')}")

    return "\n".join(lines)


def _format_supply_tag(sig: dict) -> str:
    """수급 태그 문자열 생성. 예: ' [외+기+]', ' [외-기-]'"""
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    f_5d = sig.get("foreign_amount_5d", 0)
    i_5d = sig.get("inst_amount_5d", 0)
    parts = []
    if f_streak > 0 or f_5d > 0:
        parts.append("\uc678+")
    elif f_streak < 0 or f_5d < 0:
        parts.append("\uc678-")
    if i_streak > 0 or i_5d > 0:
        parts.append("\uae30+")
    elif i_streak < 0 or i_5d < 0:
        parts.append("\uae30-")
    return f" [{''.join(parts)}]" if parts else ""


def _news_icon(news: dict | None) -> str:
    """뉴스 sentiment 아이콘."""
    if not news:
        return "-"
    s = news.get("overall_sentiment", "")
    if s in ("\uae0d\uc815", "positive"):
        return "\U0001f7e2"
    elif s in ("\ubd80\uc815", "negative"):
        return "\U0001f534"
    return "\U0001f7e1"


def _collect_positives(sig: dict) -> str:
    """긍정 요소 수집."""
    parts = []
    if sig.get("obv_trend") == "up":
        parts.append("OBV\uc0c1\uc2b9")
    ms_list = sig.get("market_signals", [])
    for m in ms_list:
        if m.get("importance") in ("critical", "high"):
            parts.append(m["title"][:12])
    news = sig.get("news_data")
    if news:
        s = news.get("overall_sentiment", "")
        if s in ("\uae0d\uc815", "positive"):
            parts.append("\ub274\uc2a4\uae0d\uc815")
        # 살아있는 이슈
        living = news.get("living_issues", [])
        active = [li for li in living if li.get("status") == "active" and li.get("sentiment") in ("positive", "\uae0d\uc815")]
        if active:
            parts.append(active[0].get("title", "")[:15])
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    if f_streak >= 3:
        parts.append(f"\uc678{f_streak}\uc77c\uc21c\ub9e4\uc218")
    if i_streak >= 3:
        parts.append(f"\uae30{i_streak}\uc77c\uc21c\ub9e4\uc218")
    f_5d = sig.get("foreign_amount_5d", 0)
    if abs(f_5d) >= 1000 and f_5d > 0:
        parts.append(f"\uc678 5\uc77c+{f_5d//1000}K")
    return ", ".join(parts[:5])


def _collect_warnings(sig: dict) -> str:
    """경고 요소 수집."""
    parts = []
    news = sig.get("news_data")
    if news:
        s = news.get("overall_sentiment", "")
        if s in ("\ubd80\uc815", "negative"):
            parts.append("\ub274\uc2a4\ubd80\uc815")
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    if f_streak <= -3:
        parts.append(f"\uc678{abs(f_streak)}\uc77c\uc21c\ub9e4\ub3c4")
    if i_streak <= -3:
        parts.append(f"\uae30{abs(i_streak)}\uc77c\uc21c\ub9e4\ub3c4")
    if sig.get("vol_surge", 1) < 0.5:
        parts.append("\uac70\ub798\ub7c9\ubd80\uc871")
    return ", ".join(parts[:3])


def main():
    parser = argparse.ArgumentParser(description="Quant Buy Scan (SignalEngine)")
    parser.add_argument("--no-send", action="store_true", help="No telegram send")
    parser.add_argument("--grade", type=str, default="A", help="Grade filter (A, AB, ABC)")
    parser.add_argument("--no-news", action="store_true", help="Skip Grok news")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--v9", action="store_true", help="v9.0 C+E Hybrid Pipeline")
    args = parser.parse_args()

    if args.v9:
        print("=" * 50)
        print("  [Quant v9.0] C+E Hybrid Kill\u2192Rank\u2192Tag")
        print("  Kill(5) \u2192 Trap(6D) \u2192 Rank(R:R\u00d7Zone) \u2192 Tag")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  [Quant v5.0] 4-Axis Score Buy Scan (SignalEngine)")
        print("  Quant(30) + SD(25) + News(25) + Consensus(20) = 100")
        print("=" * 50)

    candidates, stats = scan_all(
        grade_filter=args.grade.upper(),
        use_news=not args.no_news,
        use_v9=args.v9,
    )

    if args.v9:
        msg = format_telegram_message_v9(candidates, stats)
    else:
        msg = format_telegram_message(candidates, stats)
    print("\n" + msg)

    # HTML 보고서 생성 + PNG 변환
    png_path = None
    if not args.no_html and candidates:
        try:
            from src.html_report import generate_premarket_report
            print("\nHTML 보고서 생성 중...")
            html_path, png_path = generate_premarket_report(candidates, stats)
            print(f"HTML: {html_path}")
            if png_path:
                print(f"PNG:  {png_path}")
        except Exception as e:
            print(f"HTML 보고서 생성 실패: {e}")

    if not args.no_send:
        from src.telegram_sender import send_message

        # 1) PNG 이미지 전송 (보고서)
        if png_path and png_path.exists():
            from src.html_report import send_report_to_telegram
            print("\nSending report image to Telegram...")
            ver = "v9.0" if args.v9 else "v5.0"
            caption = f"[Quant {ver}] 장시작전 분석 | {len(candidates)}종목 | Grade {args.grade.upper()}"
            img_ok = send_report_to_telegram(png_path, caption)
            print("OK - Report image sent" if img_ok else "FAIL - Image send")

        # 2) 텍스트 메시지 전송
        print("Sending text to Telegram...")
        success = send_message(msg)
        print("OK - Text sent" if success else "FAIL - Check .env")
    else:
        print("\n(--no-send: skipped)")

    output_path = Path(__file__).parent.parent / "data" / "scan_result.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(msg, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
