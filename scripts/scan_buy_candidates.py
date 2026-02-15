"""
전체 종목 매수 후보 스캔 -> 3축 종합순위 -> 텔레그램 발송

v4.3: 3축 점수체계 (100점 만점)
  - Quant Score (퀀텀전략): 40점
  - Supply/Demand (수급):   30점
  - News Score (뉴스):      30점

사용법:
    python scripts/scan_buy_candidates.py              # Grade A만 + Grok뉴스 + 텔레그램
    python scripts/scan_buy_candidates.py --no-send    # 스캔만 (발송 안함)
    python scripts/scan_buy_candidates.py --grade A    # Grade A만 (기본)
    python scripts/scan_buy_candidates.py --grade AB   # Grade A+B
    python scripts/scan_buy_candidates.py --no-news    # Grok 뉴스 건너뛰기
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

DATA_DIR = Path(__file__).resolve().parent.parent / "stock_data_daily"

# -- 최소 필터 (사전 스크리닝) --
MIN_TRADING_VALUE = 5_0000_0000   # 일거래대금 5억 이상
MIN_PRICE = 1000                   # 최소 주가 1,000원
MIN_DATA_ROWS = 120                # 최소 120거래일 데이터


# =========================================================
# 3축 점수 체계 (100점)
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
    q_zone = min(sig.get("zone_score", 0) / 65 * 15, 15)

    # Trigger 품질 (6점)
    if sig.get("trigger_type") == "impulse":
        q_trigger = 6 if sig.get("impulse_met", 0) >= 4 else 5
    elif sig.get("trigger_type") == "confirm":
        q_trigger = 4 if sig.get("confirm_met", 0) >= 4 else 3
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
# 종목 로드 + Pipeline
# =========================================================

def load_all_stocks() -> list[tuple[str, str, Path]]:
    """stock_data_daily에서 전체 종목 로드 (스팩/채권 제외)."""
    files = list(DATA_DIR.glob("*.csv"))
    stocks = []
    for f in files:
        name = f.stem
        if name.startswith("Stock_") or name.startswith("_"):
            continue
        match = re.search(r"_(\d{6})$", name)
        if not match:
            continue
        ticker = match.group(1)
        stock_name = name[: name.rfind("_")]
        if "스팩" in stock_name:
            continue
        stocks.append((stock_name, ticker, f))
    return stocks


def compute_extra(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 추가 지표 계산."""
    df["volume_surge_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    if "MA60" in df.columns:
        df["slope_ma60"] = df["MA60"].pct_change(5) * 100
    df["atr_14"] = df.get("ATR", pd.Series(0, index=df.index))
    return df


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


def quick_pipeline(df: pd.DataFrame, idx: int) -> dict | None:
    """
    빠른 6-Layer Pipeline 판정.
    통과 시 시그널 dict 반환, 실패 시 None.
    """
    if idx < 60:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    close = row["Close"]

    # -- L0 Pre-Gate: 일거래대금 --
    avg_vol_20 = df["Volume"].iloc[max(0, idx - 19) : idx + 1].mean()
    avg_trading_val = avg_vol_20 * close
    if avg_trading_val < MIN_TRADING_VALUE:
        return None
    if close < MIN_PRICE:
        return None

    # -- L0 Grade --
    atr = row.get("ATR", 0)
    if pd.isna(atr) or atr <= 0:
        return None
    ma20 = row.get("MA20", close)
    if pd.isna(ma20):
        ma20 = close

    pullback_atr = (ma20 - close) / atr if atr > 0 else 0

    if 0.5 <= pullback_atr <= 1.5:
        atr_score = 30
    elif 0.25 <= pullback_atr < 0.5 or 1.5 < pullback_atr <= 2.0:
        atr_score = 20
    elif 0 <= pullback_atr < 0.25:
        atr_score = 10
    else:
        atr_score = 5

    rsi = row.get("RSI", 50)
    if pd.isna(rsi):
        rsi = 50
    if 30 <= rsi <= 45:
        rsi_score = 20
    elif 45 < rsi <= 55:
        rsi_score = 15
    elif rsi < 30:
        rsi_score = 10
    else:
        rsi_score = 5

    stoch_k = row.get("Stoch_K", 50)
    if pd.isna(stoch_k):
        stoch_k = 50
    if stoch_k < 20:
        stoch_score = 15
    elif stoch_k < 40:
        stoch_score = 10
    else:
        stoch_score = 5

    zone_score = atr_score + rsi_score + stoch_score
    if zone_score >= 55:
        grade = "A"
    elif zone_score >= 40:
        grade = "B"
    elif zone_score >= 25:
        grade = "C"
    else:
        grade = "F"

    if grade == "F":
        return None

    # -- L3 Momentum --
    vol_surge = row.get("volume_surge_ratio", 1.0)
    slope_60 = row.get("slope_ma60", 0)
    if pd.isna(vol_surge):
        vol_surge = 1.0
    if pd.isna(slope_60):
        slope_60 = 0
    if vol_surge < 1.2 and slope_60 < -0.5:
        return None

    # -- L4 Smart Money --
    obv_now = row.get("OBV", 0)
    obv_20ago = df.iloc[max(0, idx - 19)].get("OBV", obv_now)
    if pd.isna(obv_now):
        obv_now = 0
    if pd.isna(obv_20ago):
        obv_20ago = obv_now
    obv_trend = "up" if obv_now > obv_20ago else "down"
    if obv_trend == "down" and close < ma20:
        return None

    # -- L5 Risk --
    swing_low = df["Low"].iloc[max(0, idx - 9) : idx + 1].min()
    stop_price = max(swing_low * 0.995, close * 0.97)
    target_price = close + atr * 3
    risk = close - stop_price
    reward = target_price - close
    rr_ratio = reward / risk if risk > 0 else 0
    if rr_ratio < 1.5:
        return None

    # -- L6 Trigger --
    ma5 = row.get("MA5", 0)
    if pd.isna(ma5):
        ma5 = 0
    macd = row.get("MACD", 0)
    macd_sig = row.get("MACD_Signal", 0)
    if pd.isna(macd):
        macd = 0
    if pd.isna(macd_sig):
        macd_sig = 0
    hist = macd - macd_sig
    prev_rsi = prev.get("RSI", 50) if not pd.isna(prev.get("RSI", np.nan)) else 50
    prev_hist = prev.get("MACD", 0) - prev.get("MACD_Signal", 0)
    if pd.isna(prev_hist):
        prev_hist = 0

    stoch_d = row.get("Stoch_D", 50)
    if pd.isna(stoch_d):
        stoch_d = 50

    imp_conds = {
        "close>MA5": close > ma5 if ma5 > 0 else False,
        "RSI_up": rsi > prev_rsi and rsi < 70,
        "MACD_hist_up": hist > prev_hist,
        "vol_surge": vol_surge > 1.2 if not pd.isna(vol_surge) else False,
        "stoch_GC": stoch_k > stoch_d,
    }
    imp_met = sum(imp_conds.values())

    adx = row.get("ADX", 0)
    if pd.isna(adx):
        adx = 0
    plus_di = row.get("Plus_DI", 0)
    minus_di = row.get("Minus_DI", 0)
    if pd.isna(plus_di):
        plus_di = 0
    if pd.isna(minus_di):
        minus_di = 0

    conf_conds = {
        "close>MA20": close > ma20,
        "RSI>50": rsi > 50,
        "ADX>20": adx > 20,
        "+DI>-DI": plus_di > minus_di,
    }
    conf_met = sum(conf_conds.values())

    if imp_met >= 3:
        trigger_type = "impulse"
        confidence = imp_met / 5
    elif conf_met >= 3:
        trigger_type = "confirm"
        confidence = conf_met / 4
    else:
        return None

    # -- Trend --
    ma60 = row.get("MA60", 0)
    if pd.isna(ma60):
        ma60 = 0
    ma120 = row.get("MA120", 0)
    if pd.isna(ma120):
        ma120 = 0

    if close > ma5 and close > ma20 and close > ma60:
        trend = "strong_up"
    elif close > ma5 and close > ma20:
        trend = "up"
    elif close > ma60:
        trend = "neutral"
    else:
        trend = "down"

    # -- 수급 데이터 추출 --
    foreign_streak = 0
    inst_streak = 0
    foreign_amount_5d = 0
    inst_amount_5d = 0

    if "Foreign_Net" in df.columns:
        f_series = df["Foreign_Net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
        foreign_streak = _calc_streak(f_series)
        foreign_amount_5d = int(df["Foreign_Net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

    if "Inst_Net" in df.columns:
        i_series = df["Inst_Net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
        inst_streak = _calc_streak(i_series)
        inst_amount_5d = int(df["Inst_Net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

    return {
        "grade": grade,
        "zone_score": zone_score,
        "trigger_type": trigger_type,
        "confidence": confidence,
        "entry_price": int(close),
        "stop_loss": int(stop_price),
        "target_price": int(target_price),
        "risk_reward": round(rr_ratio, 2),
        "rsi": round(rsi, 1),
        "adx": round(adx, 1),
        "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "vol_surge": round(vol_surge, 2),
        "obv_trend": obv_trend,
        "trend": trend,
        "avg_trading_val": round(avg_trading_val / 1e8, 1),
        "impulse_met": imp_met,
        "confirm_met": conf_met,
        # 수급 데이터
        "foreign_streak": foreign_streak,
        "inst_streak": inst_streak,
        "foreign_amount_5d": foreign_amount_5d,
        "inst_amount_5d": inst_amount_5d,
    }


# =========================================================
# 메인 스캔
# =========================================================

def scan_all(
    grade_filter: str = "A",
    use_news: bool = True,
) -> tuple[list[dict], dict]:
    """전 종목 스캔 -> Grade 필터 -> 3축 점수 -> 순위 반환."""
    stocks = load_all_stocks()
    print(f"scan: {len(stocks)} stocks | grade={grade_filter} | news={'ON' if use_news else 'OFF'}")

    scanner = MarketSignalScanner()
    candidates = []
    stats = {
        "total": len(stocks),
        "loaded": 0,
        "filtered_data": 0,
        "passed_pipeline": 0,
        "trigger_impulse": 0,
        "trigger_confirm": 0,
        "grade_A": 0,
        "grade_B": 0,
        "grade_C": 0,
        "after_grade_filter": 0,
    }

    t0 = time.time()

    for i, (name, ticker, fpath) in enumerate(stocks):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(stocks)} ({elapsed:.1f}s)")

        try:
            df = pd.read_csv(fpath, index_col="Date", parse_dates=True)
        except Exception:
            continue

        stats["loaded"] += 1

        if len(df) < MIN_DATA_ROWS:
            stats["filtered_data"] += 1
            continue

        df = compute_extra(df)
        idx = len(df) - 1

        signal = quick_pipeline(df, idx)
        if signal is None:
            continue

        signal["ticker"] = ticker
        signal["name"] = name

        stats["passed_pipeline"] += 1

        if signal["trigger_type"] == "impulse":
            stats["trigger_impulse"] += 1
        elif signal["trigger_type"] == "confirm":
            stats["trigger_confirm"] += 1

        grade_key = f"grade_{signal['grade']}"
        stats[grade_key] = stats.get(grade_key, 0) + 1

        if signal["grade"] not in grade_filter:
            continue

        stats["after_grade_filter"] += 1

        # Market Signal Scanner
        try:
            market_signals = scanner.scan_all(df, idx)
            signal["market_signals"] = [
                {"title": s.title, "importance": s.importance, "confidence": s.confidence}
                for s in market_signals
            ] if market_signals else []
        except Exception:
            signal["market_signals"] = []

        candidates.append(signal)

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

    # -- 종합 점수 계산 --
    for sig in candidates:
        sig["scores"] = calc_composite_score(sig)

    # -- 종합 점수 기준 정렬 --
    candidates.sort(key=lambda s: s["scores"]["total"], reverse=True)

    stats["elapsed_sec"] = round(time.time() - t0, 1)

    return candidates, stats


# =========================================================
# 텔레그램 메시지 포맷 (KISBOT v4.3 스타일)
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
    lines.append(f"[Quant v4.3] {now} 3-Axis \uc2a4\uce94")
    lines.append("")

    # -- 점수 체계 --
    lines.append("[ \uc810\uc218 \uccb4\uacc4 ]")
    lines.append("3\ucd95: Quant(40) + \uc218\uae09(30) + \ub274\uc2a4(30) = 100")
    lines.append("  Q: Zone(20)+\ud2b8\ub9ac\uac70(8)+\uc190\uc775\ube44(7)+\ucd94\uc138(5)")
    lines.append("  SD: \uc678\uad6d\uc778(10)+\uae30\uad00(10)+OBV(5)+ADX(5)")
    lines.append("  N: \uac10\uc131(8)+\uc601\ud5a5(8)+\uc774\uc288(7)+\uc2e4\uc801(7)")
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
    parser = argparse.ArgumentParser(description="3-Axis Score Buy Scan v4.3")
    parser.add_argument("--no-send", action="store_true", help="No telegram send")
    parser.add_argument("--grade", type=str, default="A", help="Grade filter (A, AB, ABC)")
    parser.add_argument("--no-news", action="store_true", help="Skip Grok news")
    args = parser.parse_args()

    print("=" * 50)
    print("  [Quant v4.3] 3-Axis Score Buy Scan")
    print("  Quant(40) + Supply/Demand(30) + News(30) = 100")
    print("=" * 50)

    candidates, stats = scan_all(
        grade_filter=args.grade.upper(),
        use_news=not args.no_news,
    )

    msg = format_telegram_message(candidates, stats)
    print("\n" + msg)

    if not args.no_send:
        from src.telegram_sender import send_message
        print("\nSending to Telegram...")
        success = send_message(msg)
        if success:
            print("OK - Telegram sent")
        else:
            print("FAIL - Check .env")
    else:
        print("\n(--no-send: skipped)")

    output_path = Path(__file__).parent.parent / "data" / "scan_result.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(msg, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
