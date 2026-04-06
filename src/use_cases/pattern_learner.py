"""패턴 학습 엔진 — 급등/급락 원인 분석 + 패턴 누적 + 내일 후보 추출.

daily_market_learner.py에서 호출.
parquet 131개 컬럼에서 D-1 기술지표를 뽑아 "왜 올랐나/떨어졌나"를 분류하고,
패턴별 통계를 누적하여 수익 확률을 학습한다.

4/10 1차 완성 목표.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PATTERN_STATS_PATH = DATA_DIR / "market_learning" / "pattern_stats.json"

# ═══════════════════════════════════════════════
# D-1 기술지표 수집 (parquet 131컬럼 중 핵심 추출)
# ═══════════════════════════════════════════════

# 수집할 D-1 지표 목록
_INDICATOR_COLS = [
    # 가격
    "close", "sma_20", "sma_60", "sma_120",
    "bb_position", "bb_width", "pct_of_52w_high",
    # 추세
    "sar_trend", "slope_ma60", "days_above_sma20",
    "higher_low_3d", "higher_low_5d", "linreg_slope_20",
    # 모멘텀
    "rsi_14", "rsi_zscore", "stoch_slow_k", "stoch_slow_d", "stoch_slow_golden",
    "macd_histogram", "macd_histogram_prev", "adx_14",
    "trix", "trix_golden_cross",
    # 거래량
    "volume_surge_ratio", "volume_contraction_ratio", "vol_z",
    "trading_value",
    # 수급
    "foreign_net_5d", "foreign_net_20d", "foreign_consecutive_buy",
    "inst_net_5d", "inst_net_20d", "inst_consecutive_buy",
    "pension_net_5d",
    # 공매도
    "short_ratio", "short_cover_signal", "short_spike",
    # 기타
    "gap_up_pct", "pullback_atr_zscore", "smart_z",
    # 피보나치 대용 (스윙 고/저)
    "high_20", "high_60", "rolling_low_10", "rolling_low_3",
]


def _sf(v) -> float:
    """NaN-safe float."""
    if v is None:
        return 0.0
    try:
        fv = float(v)
        return 0.0 if pd.isna(fv) else round(fv, 4)
    except (ValueError, TypeError):
        return 0.0


def collect_deep_snapshot(ticker: str) -> dict | None:
    """단일 종목 parquet에서 D-1 기술지표 + 오늘 수익률 추출."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        return None
    try:
        df = pd.read_parquet(pq)
        if len(df) < 3:
            return None

        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        day_before = df.iloc[-3]

        close_t = float(today.get("close", 0) or 0)
        close_y = float(yesterday.get("close", 0) or 0)
        close_yy = float(day_before.get("close", 0) or 0)

        if close_y <= 0:
            return None

        ret_1d = round((close_t / close_y - 1) * 100, 2)
        ret_2d = round((close_t / close_yy - 1) * 100, 2) if close_yy > 0 else 0

        # D-1 지표 수집
        indicators: dict[str, Any] = {}
        for col in _INDICATOR_COLS:
            if col in yesterday.index:
                indicators[col] = _sf(yesterday[col])
            else:
                indicators[col] = 0.0

        # 피보나치 레벨 계산 (D-1 기준)
        fib = _calc_fibonacci(df.iloc[:-1])  # 오늘 제외한 데이터

        # MA20 이격도
        sma20 = _sf(yesterday.get("sma_20", 0))
        ma20_gap = round((close_y / sma20 - 1) * 100, 2) if sma20 > 0 else 0

        # MA60 이격도
        sma60 = _sf(yesterday.get("sma_60", 0))
        ma60_gap = round((close_y / sma60 - 1) * 100, 2) if sma60 > 0 else 0

        return {
            "ticker": ticker,
            "close_today": close_t,
            "close_yesterday": close_y,
            "ret_1d": ret_1d,
            "ret_2d": ret_2d,
            "ma20_gap": ma20_gap,
            "ma60_gap": ma60_gap,
            "indicators": indicators,
            "fibonacci": fib,
        }
    except Exception as e:
        logger.debug("collect_deep_snapshot %s 실패: %s", ticker, e)
        return None


def _calc_fibonacci(df: pd.DataFrame) -> dict:
    """최근 60일 스윙 고/저에서 피보나치 되돌림 레벨 계산."""
    if len(df) < 20:
        return {}
    try:
        recent = df.tail(60)
        swing_high = float(recent["close"].max())
        swing_low = float(recent["close"].min())
        last_close = float(df.iloc[-1]["close"])

        if swing_high <= swing_low:
            return {}

        diff = swing_high - swing_low
        fib_levels = {
            "swing_high": round(swing_high),
            "swing_low": round(swing_low),
            "fib_236": round(swing_high - diff * 0.236),
            "fib_382": round(swing_high - diff * 0.382),
            "fib_500": round(swing_high - diff * 0.500),
            "fib_618": round(swing_high - diff * 0.618),
            "fib_786": round(swing_high - diff * 0.786),
            "close_vs_fib": "",
        }

        # 현재가가 어떤 피보나치 구간에 있는지
        if last_close >= fib_levels["fib_236"]:
            fib_levels["close_vs_fib"] = "ABOVE_236"
        elif last_close >= fib_levels["fib_382"]:
            fib_levels["close_vs_fib"] = "236_382"
        elif last_close >= fib_levels["fib_500"]:
            fib_levels["close_vs_fib"] = "382_500"
        elif last_close >= fib_levels["fib_618"]:
            fib_levels["close_vs_fib"] = "500_618"
        elif last_close >= fib_levels["fib_786"]:
            fib_levels["close_vs_fib"] = "618_786"
        else:
            fib_levels["close_vs_fib"] = "BELOW_786"

        return fib_levels
    except Exception:
        return {}


# ═══════════════════════════════════════════════
# 패턴 분류 — "왜 올랐나 / 왜 떨어졌나"
# ═══════════════════════════════════════════════

def classify_pattern(snap: dict) -> list[str]:
    """D-1 기술지표 기반 패턴 분류. 복수 패턴 가능."""
    ind = snap.get("indicators", {})
    fib = snap.get("fibonacci", {})
    patterns = []

    rsi = ind.get("rsi_14", 50)
    bb = ind.get("bb_position", 0.5)
    macd_h = ind.get("macd_histogram", 0)
    adx = ind.get("adx_14", 20)
    vol_surge = ind.get("volume_surge_ratio", 1)
    vol_z = ind.get("vol_z", 0)
    sar = ind.get("sar_trend", 0)
    slope60 = ind.get("slope_ma60", 0)
    stoch_k = ind.get("stoch_slow_k", 50)
    stoch_golden = ind.get("stoch_slow_golden", 0)
    foreign_5d = ind.get("foreign_net_5d", 0)
    inst_5d = ind.get("inst_net_5d", 0)
    foreign_consec = ind.get("foreign_consecutive_buy", 0)
    inst_consec = ind.get("inst_consecutive_buy", 0)
    short_cover = ind.get("short_cover_signal", 0)
    short_spike = ind.get("short_spike", 0)
    higher_low_3d = ind.get("higher_low_3d", 0)
    higher_low_5d = ind.get("higher_low_5d", 0)
    trix_gx = ind.get("trix_golden_cross", 0)
    pct_52w = ind.get("pct_of_52w_high", -50)
    ma20_gap = snap.get("ma20_gap", 0)
    vol_contract = ind.get("volume_contraction_ratio", 1)
    fib_zone = fib.get("close_vs_fib", "")

    ret = snap.get("ret_1d", 0)
    is_gainer = ret > 0

    if is_gainer:
        # === 급등 패턴 ===

        # 1. 눌림목 반등 (Pullback Bounce)
        if rsi < 40 and bb < 0.3 and ma20_gap < -5:
            patterns.append("PULLBACK_BOUNCE")

        # 2. 극단 과매도 반등
        if rsi < 30 and bb < 0.15:
            patterns.append("OVERSOLD_BOUNCE")

        # 3. 볼린저 하단 반등
        if bb < 0.1 and vol_surge > 1.5:
            patterns.append("BB_LOWER_BOUNCE")

        # 4. 피보나치 지지 반등
        if fib_zone in ("500_618", "618_786", "BELOW_786"):
            patterns.append("FIB_SUPPORT_BOUNCE")

        # 5. 돌파 (신고가 근처 + 거래량)
        if pct_52w > -10 and vol_surge > 2.0:
            patterns.append("BREAKOUT")

        # 6. 거래량 폭발
        if vol_surge > 3.0 or vol_z > 2.5:
            patterns.append("VOLUME_EXPLOSION")

        # 7. 수급 반전 (외인+기관 동시 매수)
        if foreign_5d > 0 and inst_5d > 0:
            patterns.append("DUAL_BUY")
        elif foreign_consec >= 3 or inst_consec >= 3:
            patterns.append("SUPPLY_ACCUMULATION")

        # 8. 모멘텀 지속
        if rsi > 50 and higher_low_5d and macd_h > 0 and slope60 > 0:
            patterns.append("MOMENTUM_CONTINUATION")

        # 9. 골든크로스 (Stochastic 또는 TRIX)
        if stoch_golden or trix_gx:
            patterns.append("GOLDEN_CROSS")

        # 10. 공매도 숏커버
        if short_cover or short_spike:
            patterns.append("SHORT_SQUEEZE")

        # 11. 거래량 진공 후 폭발
        if vol_contract < 0.5 and vol_surge > 2.0:
            patterns.append("VOLUME_VACUUM_BREAK")

        # 12. SAR 전환
        if sar == -1 and ret > 3:  # 하락추세였는데 급등
            patterns.append("SAR_REVERSAL")

    else:
        # === 급락 패턴 ===

        # 1. 과열 후 이익실현
        if rsi > 70 or bb > 0.9:
            patterns.append("OVERBOUGHT_SELLOFF")

        # 2. 수급 이탈
        if foreign_5d < 0 and inst_5d < 0:
            patterns.append("DUAL_SELL")
        elif foreign_consec <= -3 or inst_consec <= -3:
            patterns.append("SUPPLY_EXODUS")

        # 3. 지지선 붕괴
        if fib_zone in ("BELOW_786",) and slope60 < 0:
            patterns.append("SUPPORT_BREAKDOWN")

        # 4. 갭다운 패닉
        if vol_surge > 3.0 and ret < -5:
            patterns.append("PANIC_SELL")

        # 5. 거래량 없는 하락 (무관심 하락)
        if vol_surge < 0.7 and ret < -2:
            patterns.append("LOW_VOLUME_DRIFT")

        # 6. 추세 하락 지속
        if sar == -1 and slope60 < 0 and rsi < 45:
            patterns.append("TREND_DOWN_CONTINUATION")

    if not patterns:
        patterns.append("UNCLASSIFIED")

    return patterns


# ═══════════════════════════════════════════════
# 급등/급락 원인 분석 (Phase 1.5)
# ═══════════════════════════════════════════════

def analyze_movers(
    name_map: dict[str, str],
    top_n_gainers: int = 30,
    top_n_losers: int = 20,
) -> dict:
    """전체 종목에서 급등/급락 원인 분석. 패턴 분류 포함."""
    all_snaps = []

    for pq_path in sorted(PROCESSED_DIR.glob("*.parquet")):
        ticker = pq_path.stem
        snap = collect_deep_snapshot(ticker)
        if snap:
            snap["name"] = name_map.get(ticker, ticker)
            all_snaps.append(snap)

    all_snaps.sort(key=lambda x: x["ret_1d"], reverse=True)
    total = len(all_snaps)

    gainers = [s for s in all_snaps if s["ret_1d"] >= 3.0][:top_n_gainers]
    losers = [s for s in all_snaps if s["ret_1d"] <= -3.0]
    losers.sort(key=lambda x: x["ret_1d"])
    losers = losers[:top_n_losers]

    # 패턴 분류
    for snap in gainers + losers:
        snap["patterns"] = classify_pattern(snap)

    # 패턴 통계 (오늘만)
    pattern_counts: dict[str, dict] = {}
    for snap in gainers:
        for p in snap["patterns"]:
            if p not in pattern_counts:
                pattern_counts[p] = {"gain_count": 0, "gain_rets": [], "loss_count": 0, "loss_rets": []}
            pattern_counts[p]["gain_count"] += 1
            pattern_counts[p]["gain_rets"].append(snap["ret_1d"])
    for snap in losers:
        for p in snap["patterns"]:
            if p not in pattern_counts:
                pattern_counts[p] = {"gain_count": 0, "gain_rets": [], "loss_count": 0, "loss_rets": []}
            pattern_counts[p]["loss_count"] += 1
            pattern_counts[p]["loss_rets"].append(snap["ret_1d"])

    # 요약
    today_patterns = {}
    for pname, pdata in pattern_counts.items():
        g_rets = pdata["gain_rets"]
        l_rets = pdata["loss_rets"]
        today_patterns[pname] = {
            "gain_count": pdata["gain_count"],
            "gain_avg": round(sum(g_rets) / len(g_rets), 2) if g_rets else 0,
            "loss_count": pdata["loss_count"],
            "loss_avg": round(sum(l_rets) / len(l_rets), 2) if l_rets else 0,
        }

    logger.info(
        "[패턴분석] 급등 %d건, 급락 %d건, 패턴 %d종",
        len(gainers), len(losers), len(today_patterns),
    )

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_stocks": total,
        "gainers": _slim_movers(gainers),
        "losers": _slim_movers(losers),
        "today_patterns": today_patterns,
    }


def _slim_movers(movers: list[dict]) -> list[dict]:
    """저장용 슬림 버전. indicators는 핵심만."""
    result = []
    for m in movers:
        ind = m.get("indicators", {})
        result.append({
            "ticker": m["ticker"],
            "name": m.get("name", m["ticker"]),
            "ret_1d": m["ret_1d"],
            "patterns": m.get("patterns", []),
            "key_indicators": {
                "rsi": ind.get("rsi_14", 0),
                "bb_position": ind.get("bb_position", 0),
                "macd_histogram": ind.get("macd_histogram", 0),
                "adx": ind.get("adx_14", 0),
                "stoch_k": ind.get("stoch_slow_k", 0),
                "vol_surge": ind.get("volume_surge_ratio", 0),
                "foreign_5d": ind.get("foreign_net_5d", 0),
                "inst_5d": ind.get("inst_net_5d", 0),
                "foreign_consec": ind.get("foreign_consecutive_buy", 0),
                "inst_consec": ind.get("inst_consecutive_buy", 0),
                "sar_trend": ind.get("sar_trend", 0),
                "slope_ma60": ind.get("slope_ma60", 0),
                "pct_52w": ind.get("pct_of_52w_high", 0),
                "short_ratio": ind.get("short_ratio", 0),
            },
            "ma20_gap": m.get("ma20_gap", 0),
            "ma60_gap": m.get("ma60_gap", 0),
            "fibonacci": m.get("fibonacci", {}),
        })
    return result


# ═══════════════════════════════════════════════
# 패턴 누적 통계 (Phase 4.5)
# ═══════════════════════════════════════════════

def accumulate_patterns(today_analysis: dict) -> dict:
    """패턴별 누적 통계 갱신. pattern_stats.json 관리."""
    stats = _load_pattern_stats()

    today_date = today_analysis.get("date", "")

    # 이미 같은 날짜 처리됐으면 스킵
    if today_date in stats.get("processed_dates", []):
        logger.info("[패턴누적] %s 이미 처리됨", today_date)
        return stats

    # 급등주 패턴 누적
    for g in today_analysis.get("gainers", []):
        for p in g.get("patterns", []):
            if p not in stats["patterns"]:
                stats["patterns"][p] = {
                    "total_gain": 0, "total_loss": 0,
                    "gain_rets": [], "loss_rets": [],
                    "avg_indicators": {},
                }
            entry = stats["patterns"][p]
            entry["total_gain"] += 1
            entry["gain_rets"].append(g["ret_1d"])
            # 지표 누적 (이동평균)
            _update_avg_indicators(entry, g.get("key_indicators", {}), is_gain=True)

    # 급락주 패턴 누적
    for l in today_analysis.get("losers", []):
        for p in l.get("patterns", []):
            if p not in stats["patterns"]:
                stats["patterns"][p] = {
                    "total_gain": 0, "total_loss": 0,
                    "gain_rets": [], "loss_rets": [],
                    "avg_indicators": {},
                }
            entry = stats["patterns"][p]
            entry["total_loss"] += 1
            entry["loss_rets"].append(l["ret_1d"])

    # 처리일 기록
    stats.setdefault("processed_dates", [])
    stats["processed_dates"].append(today_date)
    stats["processed_dates"] = stats["processed_dates"][-60:]  # 60일 유지

    # 패턴별 요약 통계 계산
    for pname, pdata in stats["patterns"].items():
        g_rets = pdata.get("gain_rets", [])[-60:]  # 최근 60일
        l_rets = pdata.get("loss_rets", [])[-60:]
        total = len(g_rets) + len(l_rets)
        pdata["gain_rets"] = g_rets
        pdata["loss_rets"] = l_rets
        pdata["win_rate"] = round(len(g_rets) / total * 100, 1) if total > 0 else 0
        pdata["avg_gain"] = round(sum(g_rets) / len(g_rets), 2) if g_rets else 0
        pdata["avg_loss"] = round(sum(l_rets) / len(l_rets), 2) if l_rets else 0
        pdata["sample_size"] = total
        pdata["profit_factor"] = round(
            (sum(g_rets) / len(g_rets)) / abs(sum(l_rets) / len(l_rets)), 2
        ) if l_rets and g_rets and sum(l_rets) != 0 else 0

    stats["updated_at"] = today_date
    _save_pattern_stats(stats)

    logger.info("[패턴누적] %d개 패턴 업데이트", len(stats["patterns"]))
    return stats


def _update_avg_indicators(entry: dict, indicators: dict, is_gain: bool):
    """급등 시 지표의 이동평균 업데이트 (학습용)."""
    if not is_gain:
        return
    avg = entry.setdefault("avg_indicators", {})
    n = entry.get("total_gain", 1)
    for k, v in indicators.items():
        if isinstance(v, (int, float)):
            old_avg = avg.get(k, 0)
            # incremental mean
            avg[k] = round(old_avg + (v - old_avg) / n, 4)


def _load_pattern_stats() -> dict:
    if PATTERN_STATS_PATH.exists():
        try:
            with open(PATTERN_STATS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"patterns": {}, "processed_dates": [], "updated_at": ""}


def _save_pattern_stats(stats: dict):
    PATTERN_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERN_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 내일 후보 추출 — 오늘 수익성 패턴과 일치하는 종목
# ═══════════════════════════════════════════════

def find_tomorrow_candidates(
    pattern_stats: dict,
    name_map: dict[str, str],
    min_win_rate: float = 55.0,
    min_samples: int = 5,
    max_candidates: int = 20,
) -> list[dict]:
    """오늘 지표가 수익성 패턴과 일치하는 종목 추출."""
    # 수익성 패턴 필터
    profitable_patterns = []
    for pname, pdata in pattern_stats.get("patterns", {}).items():
        if (pdata.get("win_rate", 0) >= min_win_rate
                and pdata.get("sample_size", 0) >= min_samples
                and pname != "UNCLASSIFIED"):
            profitable_patterns.append(pname)

    if not profitable_patterns:
        logger.info("[내일후보] 수익성 패턴 없음 (기준: WR>=%.0f%%, n>=%d)", min_win_rate, min_samples)
        return []

    logger.info("[내일후보] 수익성 패턴 %d개: %s", len(profitable_patterns), profitable_patterns)

    # 전종목 오늘 지표로 패턴 매칭
    candidates = []
    for pq_path in sorted(PROCESSED_DIR.glob("*.parquet")):
        ticker = pq_path.stem
        snap = collect_deep_snapshot(ticker)
        if not snap:
            continue

        # "오늘"의 지표로 패턴 분류 (내일 급등할 조건인지 확인)
        # 트릭: ret_1d를 양수로 가정하여 급등 패턴 매칭
        test_snap = snap.copy()
        test_snap["ret_1d"] = 5.0  # 가상 급등으로 설정해서 패턴 분류
        matched = classify_pattern(test_snap)

        # 수익성 패턴과 교집합
        hits = [p for p in matched if p in profitable_patterns]
        if not hits:
            continue

        # 패턴별 가중 점수
        total_score = 0
        for p in hits:
            pdata = pattern_stats["patterns"].get(p, {})
            wr = pdata.get("win_rate", 50)
            n = pdata.get("sample_size", 1)
            # 신뢰도 가중: win_rate * log(sample_size)
            import math
            total_score += wr * math.log2(max(n, 2))

        candidates.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "close": snap.get("close_today", 0),
            "matched_patterns": hits,
            "pattern_score": round(total_score, 1),
            "key_indicators": {
                "rsi": snap["indicators"].get("rsi_14", 0),
                "bb_position": snap["indicators"].get("bb_position", 0),
                "vol_surge": snap["indicators"].get("volume_surge_ratio", 0),
                "foreign_5d": snap["indicators"].get("foreign_net_5d", 0),
                "inst_5d": snap["indicators"].get("inst_net_5d", 0),
                "stoch_k": snap["indicators"].get("stoch_slow_k", 0),
                "ma20_gap": snap.get("ma20_gap", 0),
            },
            "fibonacci": snap.get("fibonacci", {}),
        })

    candidates.sort(key=lambda x: x["pattern_score"], reverse=True)
    result = candidates[:max_candidates]
    logger.info("[내일후보] %d종목 (전체 %d 중)", len(result), len(candidates))
    return result


# ═══════════════════════════════════════════════
# 학습 인사이트 텍스트 생성
# ═══════════════════════════════════════════════

def build_pattern_summary(
    analysis: dict,
    pattern_stats: dict,
    candidates: list[dict],
) -> str:
    """텔레그램 + 저장용 학습 인사이트 텍스트."""
    lines = []

    # 급등 패턴 분석
    gainers = analysis.get("gainers", [])
    if gainers:
        lines.append("🔥 급등 패턴 분석 (D-1 조건):")
        pattern_groups: dict[str, list] = {}
        for g in gainers[:15]:
            for p in g.get("patterns", []):
                pattern_groups.setdefault(p, []).append(g)
        for pname, stocks in sorted(pattern_groups.items(), key=lambda x: -len(x[1]))[:5]:
            names = ", ".join(s["name"] for s in stocks[:3])
            avg_ret = sum(s["ret_1d"] for s in stocks) / len(stocks)
            lines.append(f"  {pname}: {len(stocks)}건 avg+{avg_ret:.1f}% ({names})")
        lines.append("")

    # 급락 패턴
    losers = analysis.get("losers", [])
    if losers:
        lines.append("📉 급락 패턴:")
        pattern_groups2: dict[str, list] = {}
        for l in losers[:10]:
            for p in l.get("patterns", []):
                pattern_groups2.setdefault(p, []).append(l)
        for pname, stocks in sorted(pattern_groups2.items(), key=lambda x: -len(x[1]))[:3]:
            names = ", ".join(s["name"] for s in stocks[:3])
            avg_ret = sum(s["ret_1d"] for s in stocks) / len(stocks)
            lines.append(f"  {pname}: {len(stocks)}건 avg{avg_ret:.1f}% ({names})")
        lines.append("")

    # 누적 패턴 승률 TOP 5
    stats_patterns = pattern_stats.get("patterns", {})
    if stats_patterns:
        ranked = sorted(
            [(k, v) for k, v in stats_patterns.items()
             if v.get("sample_size", 0) >= 3 and k != "UNCLASSIFIED"],
            key=lambda x: -x[1].get("win_rate", 0),
        )
        if ranked:
            lines.append("📊 누적 패턴 승률 TOP 5:")
            for pname, pdata in ranked[:5]:
                wr = pdata.get("win_rate", 0)
                n = pdata.get("sample_size", 0)
                avg_g = pdata.get("avg_gain", 0)
                lines.append(f"  {pname}: WR {wr}% (n={n}) avg+{avg_g:.1f}%")
            lines.append("")

    # 내일 후보
    if candidates:
        lines.append("🎯 내일 주목 (패턴 매칭):")
        for c in candidates[:5]:
            pats = "+".join(c["matched_patterns"][:2])
            ki = c["key_indicators"]
            lines.append(
                f"  {c['name']} RSI={ki['rsi']:.0f} BB={ki['bb_position']:.2f} "
                f"수급={ki['foreign_5d']:+.0f}/{ki['inst_5d']:+.0f} [{pats}]"
            )

    return "\n".join(lines)
