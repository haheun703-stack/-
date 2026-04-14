"""
Daily Market Learner — 84종목 유니버스 일일 학습 시스템

BAT-D 20.5단계: scan_tomorrow_picks 이후, daily_archive 이전에 실행.
이 시점에서 data/ JSON들은 아직 "어제" 데이터 → parquet은 "오늘" 데이터.

6-Phase 파이프라인:
  Phase 1: Universe Snapshot (84종목 오늘 실적)
  Phase 2: Signal Accuracy (10개 시그널 적중 검증)
  Phase 3: Missed Opportunity (놓친 급등주 역추적)
  Phase 4: Cumulative Update (누적 적중률 갱신)
  Phase 5: AI Synthesis (Claude 인사이트, 선택)
  Phase 6: Save + Telegram

Usage:
    python scripts/daily_market_learner.py              # 기본 실행
    python scripts/daily_market_learner.py --no-send    # 텔레그램 미발송
    python scripts/daily_market_learner.py --no-ai      # AI 인사이트 생략
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── 경로 ──
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
LEARNING_DIR = DATA_DIR / "market_learning"
ACCURACY_PATH = LEARNING_DIR / "signal_accuracy.json"
WEIGHTS_PATH = LEARNING_DIR / "learning_weights.json"
INDEX_PATH = LEARNING_DIR / "_index.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ═══════════════════════════════════════════════
# 설정 로드
# ═══════════════════════════════════════════════

def _load_config() -> dict:
    defaults = {
        "enabled": True,
        "ai_synthesis": True,
        "ai_model": "claude-sonnet-4-5",
        "rolling_window_days": 20,
        "min_return_pct": 3.0,
        "send_telegram": True,
        "signals_to_track": [
            "tomorrow_picks", "pullback_scan", "whale_detect",
            "accumulation_tracker", "dart_event", "volume_spike",
            "dual_buying", "safety_margin", "ai_v3_picks",
            "overnight_signal",
        ],
    }
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.get("market_learner", {}).items():
            if k in defaults:
                defaults[k] = v
    except Exception:
        pass
    return defaults


# ═══════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_name_map() -> dict[str, str]:
    """ticker → 종목명 매핑 (universe.csv 최우선 → CSV 파일명 폴백)."""
    name_map = {}
    # 1) universe.csv (VPS에서도 안정적)
    uni_path = DATA_DIR / "universe.csv"
    if uni_path.exists():
        try:
            import csv as csv_mod
            with open(uni_path, encoding="utf-8-sig") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    t = row.get("ticker", row.get("code", "")).strip()
                    n = row.get("name", row.get("종목명", "")).strip()
                    if t and n:
                        name_map[t] = n
        except Exception:
            pass
    if name_map:
        return name_map
    # 2) stock_data_daily/ CSV 파일명 (폴백)
    if CSV_DIR.exists():
        for csv_path in CSV_DIR.glob("*.csv"):
            stem = csv_path.stem
            parts = stem.split("_", 1)
            if len(parts) == 2:
                name_map[parts[0]] = parts[1]
    return name_map


def _sf(v) -> float:
    """NaN-safe float."""
    if v is None:
        return 0.0
    try:
        fv = float(v)
        return 0.0 if pd.isna(fv) else round(fv, 2)
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════
# Phase 1: Universe Daily Snapshot
# ═══════════════════════════════════════════════

def phase1_universe_snapshot(name_map: dict) -> list[dict]:
    """84종목 parquet에서 오늘의 실적 스냅샷 추출."""
    snapshots = []
    for pq_path in sorted(PROCESSED_DIR.glob("*.parquet")):
        ticker = pq_path.stem
        try:
            df = pd.read_parquet(pq_path)
            if len(df) < 2:
                continue
            today = df.iloc[-1]
            yesterday = df.iloc[-2]

            close_t = _sf(today.get("close", 0))
            close_y = _sf(yesterday.get("close", 0))
            ret_1d = round((close_t / close_y - 1) * 100, 2) if close_y > 0 else 0.0

            vol = _sf(today.get("volume", 0))
            vol_ma20 = _sf(today.get("volume_ma20", 0))
            vol_ratio = round(vol / vol_ma20, 2) if vol_ma20 > 0 else 0.0

            snapshots.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "close": close_t,
                "ret_1d": ret_1d,
                "volume_ratio": vol_ratio,
                "rsi": _sf(today.get("rsi_14", 0)),
                "adx": _sf(today.get("adx_14", 0)),
                "sar_trend": int(_sf(today.get("sar_trend", 0))),
                "trix_bull": bool(today.get("trix_golden_cross", False)),
                "foreign_5d": _sf(today.get("foreign_net_5d", 0)),
                "bb_position": _sf(today.get("bb_position", 0)),
                # 어제 지표 (missed opportunity 역추적용)
                "y_rsi": _sf(yesterday.get("rsi_14", 0)),
                "y_adx": _sf(yesterday.get("adx_14", 0)),
                "y_sar_trend": int(_sf(yesterday.get("sar_trend", 0))),
            })
        except Exception as e:
            logger.debug("Parquet 로드 실패 %s: %s", ticker, e)

    snapshots.sort(key=lambda x: x["ret_1d"], reverse=True)
    logger.info("[Phase 1] 유니버스 스냅샷: %d종목", len(snapshots))
    return snapshots


# ═══════════════════════════════════════════════
# Phase 2: Signal Accuracy Check
# ═══════════════════════════════════════════════

def phase2_signal_accuracy(
    snapshots: list[dict], cfg: dict
) -> dict[str, dict]:
    """어제 시그널 vs 오늘 수익률 비교."""
    # 오늘 수익률 맵
    ret_map = {s["ticker"]: s["ret_1d"] for s in snapshots}
    results = {}

    tracked = cfg.get("signals_to_track", [])

    # 1. tomorrow_picks
    if "tomorrow_picks" in tracked:
        results["tomorrow_picks"] = _check_tomorrow_picks(ret_map)

    # 2. pullback_scan
    if "pullback_scan" in tracked:
        results["pullback_scan"] = _check_pullback(ret_map)

    # 3. whale_detect
    if "whale_detect" in tracked:
        results["whale_detect"] = _check_whale(ret_map)

    # 4. accumulation_tracker
    if "accumulation_tracker" in tracked:
        results["accumulation_tracker"] = _check_accumulation(ret_map)

    # 5. dart_event
    if "dart_event" in tracked:
        results["dart_event"] = _check_dart_event(ret_map)

    # 6. volume_spike
    if "volume_spike" in tracked:
        results["volume_spike"] = _check_volume_spike(ret_map)

    # 7. dual_buying
    if "dual_buying" in tracked:
        results["dual_buying"] = _check_dual_buying(ret_map)

    # 8. safety_margin
    if "safety_margin" in tracked:
        results["safety_margin"] = _check_safety_margin(ret_map)

    # 9. ai_v3_picks
    if "ai_v3_picks" in tracked:
        results["ai_v3_picks"] = _check_ai_v3(ret_map)

    # 10. overnight_signal
    if "overnight_signal" in tracked:
        results["overnight_signal"] = _check_overnight(snapshots)

    logger.info("[Phase 2] 시그널 적중 검증: %d개 소스", len(results))
    return results


def _accuracy_stat(tickers: list[str], ret_map: dict) -> dict:
    """공통: 시그널 종목 리스트 → 적중 통계."""
    returns = []
    for t in tickers:
        if t in ret_map:
            returns.append(ret_map[t])
    if not returns:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0, "tickers": []}
    hit = sum(1 for r in returns if r > 0)
    return {
        "total": len(returns),
        "hit": hit,
        "hit_rate": round(hit / len(returns) * 100, 1),
        "avg_ret": round(sum(returns) / len(returns), 2),
        "tickers": tickers[:10],  # 상위 10개만
    }


def _check_tomorrow_picks(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "tomorrow_picks.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    buy_grades = {"강력 포착", "포착", "관심", "적극매수", "매수", "관심매수"}
    tickers = [
        p["ticker"] for p in data.get("picks", [])
        if p.get("grade") in buy_grades
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_pullback(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "pullback_scan.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = [
        c["ticker"] for c in data.get("candidates", [])
        if c.get("grade") == "매수대기"
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_whale(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "whale_detect.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = [
        item["ticker"] for item in data.get("items", [])
        if item.get("grade") in ("세력포착", "매집의심")
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_accumulation(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "accumulation_tracker.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = [
        item["ticker"] for item in data.get("items", [])
        if item.get("phase") == "재돌파"
    ][:30]  # 최대 30개
    return _accuracy_stat(tickers, ret_map)


def _check_dart_event(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "dart_event_signals.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = [
        s["ticker"] for s in data.get("signals", [])
        if s.get("action") == "BUY"
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_volume_spike(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "volume_spike_watchlist.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    watching = data.get("watching", {})
    tickers = [
        t for t, info in watching.items()
        if info.get("status") == "signal"
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_dual_buying(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "dual_buying_watch.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = []
    for grade_key in ("s_grade", "a_grade", "b_grade"):
        for item in data.get(grade_key, []):
            t = item.get("ticker", "")
            if t:
                tickers.append(t)
    return _accuracy_stat(tickers, ret_map)


def _check_safety_margin(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "safety_margin_daily.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    tickers = [
        item.get("ticker", "") for item in data.get("green", [])
        if item.get("ticker")
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_ai_v3(ret_map: dict) -> dict:
    data = _load_json(DATA_DIR / "ai_v3_picks.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    # ai_v3_picks는 리스트 or dict
    picks = data if isinstance(data, list) else data.get("picks", [])
    tickers = [
        p.get("ticker", "") for p in picks
        if p.get("action") in ("BUY", "강력 포착", "포착", "적극매수", "매수")
           or p.get("confidence", 0) >= 0.7
    ]
    return _accuracy_stat(tickers, ret_map)


def _check_overnight(snapshots: list[dict]) -> dict:
    data = _load_json(DATA_DIR / "us_market" / "overnight_signal.json")
    if not data:
        return {"total": 0, "hit": 0, "hit_rate": 0, "avg_ret": 0}
    grade = data.get("final_grade", data.get("grade", "NEUTRAL"))
    # 유니버스 평균 수익률로 판정
    rets = [s["ret_1d"] for s in snapshots if s["ret_1d"] != 0]
    avg_ret = round(sum(rets) / len(rets), 2) if rets else 0
    is_bull = grade in ("STRONG_BULL", "MILD_BULL")
    is_bear = grade in ("STRONG_BEAR", "MILD_BEAR")
    is_neutral = grade == "NEUTRAL"
    # NEUTRAL은 측정 불가 → 적중률 계산에서 제외 (total=0)
    if is_neutral:
        return {
            "total": 0, "hit": 0, "hit_rate": 0, "avg_ret": avg_ret,
            "grade": grade, "note": "NEUTRAL 제외 (방향성 판단 없음)",
        }
    hit = (
        (is_bull and avg_ret > 0)
        or (is_bear and avg_ret < 0)
    )
    return {
        "total": 1,
        "hit": 1 if hit else 0,
        "hit_rate": 100.0 if hit else 0.0,
        "avg_ret": avg_ret,
        "grade": grade,
    }


# ═══════════════════════════════════════════════
# Phase 3: Missed Opportunity Detector
# ═══════════════════════════════════════════════

def phase3_missed_opportunities(
    snapshots: list[dict], accuracy: dict, cfg: dict
) -> list[dict]:
    """오늘 급등했지만 어제 시그널이 없었던 종목 역추적."""
    min_ret = cfg.get("min_return_pct", 3.0)
    top_movers = [s for s in snapshots if s["ret_1d"] >= min_ret]

    # 모든 시그널에 등장한 종목 집합
    signal_tickers = set()
    for sig_name, sig_data in accuracy.items():
        for t in sig_data.get("tickers", []):
            signal_tickers.add(t)

    missed = []
    for mover in top_movers:
        ticker = mover["ticker"]
        had_signals = [
            sig_name for sig_name, sig_data in accuracy.items()
            if ticker in sig_data.get("tickers", [])
        ]
        if not had_signals:
            # 어떤 게이트가 차단했는지 추정
            blocked_by = _guess_block_reason(mover)
            missed.append({
                "ticker": ticker,
                "name": mover["name"],
                "ret_1d": mover["ret_1d"],
                "had_signals": [],
                "blocked_by": blocked_by,
                "yesterday_indicators": {
                    "rsi": mover.get("y_rsi", 0),
                    "adx": mover.get("y_adx", 0),
                    "sar_trend": mover.get("y_sar_trend", 0),
                },
                "in_universe": True,
            })

    logger.info("[Phase 3] 놓친 급등주: %d건 (+%.1f%% 이상)", len(missed), min_ret)
    return missed


def _guess_block_reason(snap: dict) -> str:
    """어제 지표 기반으로 차단 사유 추정."""
    reasons = []
    y_adx = snap.get("y_adx", 0)
    y_rsi = snap.get("y_rsi", 0)
    y_sar = snap.get("y_sar_trend", 0)

    if y_adx < 14:
        reasons.append(f"G1_ADX ({y_adx:.0f}<14)")
    if y_rsi > 70:
        reasons.append(f"과열 (RSI {y_rsi:.0f})")
    if y_sar == -1:
        reasons.append("SAR 하락추세")
    if not reasons:
        reasons.append("시그널 미감지 (조건 불충족)")
    return " + ".join(reasons)


def _analyze_sar_blocks(missed: list[dict], snapshots: list[dict]) -> dict:
    """SAR 차단 종목의 진짜 놓침 vs 정당 차단 비율 분석.

    판정 기준:
      MISS: MA60↑ + RSI 35~65 + 52주고점 대비 -5%↓ → 기술적 양호한데 차단
      OK:   MA60↓ or RSI>70 or 신고가권(할인 -3%↑) → 차단 정당
      ?:    애매한 경우
    """
    sar_blocked = [m for m in missed if "SAR" in m.get("blocked_by", "")]
    if not sar_blocked:
        return {"total": 0, "genuine_miss": 0, "noise_block": 0, "ambiguous": 0,
                "miss_rate": 0.0, "top_misses": []}

    # snapshots에서 종목별 데이터 조회 (MA60, 52주고점 등)
    snap_map = {s["ticker"]: s for s in snapshots}

    genuine, noise, ambiguous = 0, 0, 0
    top_misses = []

    for m in sar_blocked:
        ticker = m.get("ticker", "")
        snap = snap_map.get(ticker, {})
        y_rsi = snap.get("y_rsi", m.get("yesterday_indicators", {}).get("rsi", 50))

        # parquet에서 MA60, 52주 할인 확인
        pq = PROCESSED_DIR / f"{ticker}.parquet"
        above_ma60 = False
        discount = 0.0
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
                if len(df) >= 60:
                    last = df.iloc[-1]
                    close = float(last.get("close", 0))
                    sma60 = float(last.get("sma_60", 0))
                    above_ma60 = close > sma60 > 0
                    high_52w = float(
                        df["close"].rolling(250).max().iloc[-1]
                        if len(df) >= 250
                        else df["close"].max()
                    )
                    discount = (close / high_52w - 1) * 100 if high_52w > 0 else 0
            except Exception:
                pass

        if above_ma60 and 35 <= y_rsi <= 65 and discount <= -5:
            genuine += 1
            top_misses.append({
                "ticker": ticker, "name": m.get("name", ""),
                "ret_1d": m.get("ret_1d", 0), "verdict": "MISS",
            })
        elif not above_ma60 or y_rsi > 70 or discount > -3:
            noise += 1
        else:
            ambiguous += 1

    total = len(sar_blocked)
    top_misses.sort(key=lambda x: -x["ret_1d"])

    return {
        "total": total,
        "genuine_miss": genuine,
        "noise_block": noise,
        "ambiguous": ambiguous,
        "miss_rate": round(genuine / total * 100, 1) if total > 0 else 0.0,
        "top_misses": top_misses[:5],
    }


# ═══════════════════════════════════════════════
# Phase 4: Cumulative Learning DB
# ═══════════════════════════════════════════════

def phase4_cumulative_update(
    today_date: str, accuracy: dict, cfg: dict
) -> dict:
    """Rolling N일 누적 적중률 갱신."""
    window = cfg.get("rolling_window_days", 20)

    # 기존 누적 데이터 로드
    cum = _load_json(ACCURACY_PATH) or {
        "updated_at": "",
        "window_days": window,
        "signals": {},
        "daily_log": [],
    }

    # 오늘 로그 추가 (중복 방지: 같은 날짜 덮어쓰기)
    day_entry = {"date": today_date}
    for sig_name, sig_data in accuracy.items():
        day_entry[f"{sig_name}_hr"] = sig_data.get("hit_rate", 0)
        day_entry[f"{sig_name}_n"] = sig_data.get("total", 0)
        day_entry[f"{sig_name}_ret"] = sig_data.get("avg_ret", 0)
    cum["daily_log"] = [d for d in cum["daily_log"] if d.get("date") != today_date]
    cum["daily_log"].append(day_entry)

    # rolling window 유지
    cum["daily_log"] = cum["daily_log"][-window:]
    cum["updated_at"] = today_date
    cum["window_days"] = window

    # 시그널별 누적 통계 재계산
    signals_cum = {}
    for sig_name in accuracy:
        total_sum = 0
        hit_sum = 0
        ret_list = []
        for day in cum["daily_log"]:
            n = day.get(f"{sig_name}_n", 0)
            hr = day.get(f"{sig_name}_hr", 0)
            ret = day.get(f"{sig_name}_ret", 0)
            total_sum += n
            hit_sum += int(n * hr / 100) if n > 0 else 0
            if n > 0:
                ret_list.append(ret)
        signals_cum[sig_name] = {
            "total": total_sum,
            "hit": hit_sum,
            "hit_rate": round(hit_sum / total_sum * 100, 1) if total_sum > 0 else 0,
            "avg_ret": round(sum(ret_list) / len(ret_list), 2) if ret_list else 0,
            "days_tracked": len([d for d in cum["daily_log"] if d.get(f"{sig_name}_n", 0) > 0]),
        }
    cum["signals"] = signals_cum

    _save_json(ACCURACY_PATH, cum)

    # ── learning_weights.json 생성 (피드백 루프 핵심) ──
    _compute_learning_weights(cum)

    logger.info("[Phase 4] 누적 적중률 갱신 (window=%d일)", window)
    return cum


def _compute_learning_weights(cum: dict) -> dict:
    """signal_accuracy → score_multiplier 변환.

    적중률이 평균보다 높은 시그널은 가중치 UP, 낮으면 DOWN.
    scan_tomorrow_picks.py 축2(개별점수)에서 소비.

    공식: multiplier = 0.5 + (hit_rate / baseline) * 0.5
    범위: clamp(0.5, 1.5)
    조건: days_tracked >= 3 (초기 3일부터 적용, 데이터 안정화 후 상향 가능)
    """
    MIN_DAYS = 3

    signals = cum.get("signals", {})
    if not signals:
        return {}

    # 전체 평균 적중률 (baseline)
    rates = [
        s["hit_rate"]
        for s in signals.values()
        if s.get("days_tracked", 0) >= MIN_DAYS
    ]
    if not rates:
        logger.info("[Weights] 충분한 데이터 없음 (%d일 미만) — 기본값 유지", MIN_DAYS)
        return {}
    baseline = max(sum(rates) / len(rates), 10)  # 최소 10% (0 나누기 방지)

    weights: dict = {}
    for name, stats in signals.items():
        if stats.get("days_tracked", 0) < MIN_DAYS:
            weights[name] = {"multiplier": 1.0, "reason": f"데이터 부족 (<{MIN_DAYS}일)"}
            continue
        hr = stats["hit_rate"]
        raw = 0.5 + (hr / baseline) * 0.5
        multiplier = round(max(0.5, min(1.5, raw)), 2)
        weights[name] = {
            "multiplier": multiplier,
            "hit_rate": hr,
            "avg_ret": stats["avg_ret"],
            "total": stats["total"],
            "days_tracked": stats["days_tracked"],
            "reason": f"적중률 {hr}% (baseline {baseline:.1f}%)",
        }

    result = {
        "updated_at": cum.get("updated_at", ""),
        "baseline_hit_rate": round(baseline, 1),
        "min_days_required": MIN_DAYS,
        "weights": weights,
    }
    _save_json(WEIGHTS_PATH, result)
    logger.info(
        "[Weights] learning_weights.json 갱신 — baseline %.1f%%, %d개 시그널",
        baseline, len(weights),
    )
    return result


# ═══════════════════════════════════════════════
# Phase 5: AI Synthesis
# ═══════════════════════════════════════════════

def phase5_ai_synthesis(
    snapshot_summary: dict, accuracy: dict,
    missed: list[dict], cumulative: dict, cfg: dict,
) -> str:
    """Claude Sonnet으로 학습 인사이트 생성."""
    if not cfg.get("ai_synthesis", True):
        return ""

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic 패키지 없음 — AI 인사이트 생략")
        return ""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    model = cfg.get("ai_model", "claude-sonnet-4-5")

    # 시그널 적중 요약
    acc_lines = []
    for sig, stat in accuracy.items():
        if stat.get("total", 0) > 0:
            acc_lines.append(
                f"  {sig}: {stat['hit']}/{stat['total']} "
                f"({stat['hit_rate']}%) avg {stat['avg_ret']:+.1f}%"
            )
    acc_text = "\n".join(acc_lines) if acc_lines else "  (시그널 없음)"

    # 놓친 종목 요약
    missed_text = "\n".join(
        f"  {m['name']} +{m['ret_1d']}% — {m['blocked_by']}"
        for m in missed[:5]
    ) if missed else "  (없음)"

    # 누적 베스트/워스트
    cum_signals = cumulative.get("signals", {})
    best = max(cum_signals.items(), key=lambda x: x[1].get("hit_rate", 0), default=("없음", {}))
    worst = min(
        [(k, v) for k, v in cum_signals.items() if v.get("total", 0) > 5],
        key=lambda x: x[1].get("hit_rate", 0),
        default=("없음", {}),
    )

    prompt = (
        f"오늘 한국 주식시장 84종목 학습 결과:\n\n"
        f"시장: 상승 {snapshot_summary['up_count']}종목, "
        f"하락 {snapshot_summary['down_count']}종목, "
        f"평균 수익률 {snapshot_summary['avg_ret']:+.2f}%\n\n"
        f"시그널 적중률:\n{acc_text}\n\n"
        f"놓친 급등주:\n{missed_text}\n\n"
        f"20일 누적 최고: {best[0]} ({best[1].get('hit_rate', 0)}%)\n"
        f"20일 누적 최저: {worst[0]} ({worst[1].get('hit_rate', 0)}%)\n\n"
        f"이 데이터에서 배울 수 있는 핵심 인사이트 3줄로 요약해줘. "
        f"각 줄은 30자 이내. 한국어로."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        insight = resp.content[0].text.strip()
        logger.info("[Phase 5] AI 인사이트 생성 완료")
        return insight
    except Exception as e:
        logger.warning("[Phase 5] AI 인사이트 실패: %s", e)
        return ""


# ═══════════════════════════════════════════════
# Phase 6: Save + Telegram
# ═══════════════════════════════════════════════

def phase6_save_and_send(
    today_date: str, snapshots: list[dict], accuracy: dict,
    missed: list[dict], cumulative: dict, ai_insight: str,
    send_telegram: bool = True,
    sar_analysis: dict | None = None,
    pattern_analysis: dict | None = None,
    pattern_text: str = "",
    pattern_candidates: list | None = None,
):
    """결과 저장 + 텔레그램."""
    # ── 스냅샷 요약 ──
    up_count = sum(1 for s in snapshots if s["ret_1d"] > 0)
    down_count = sum(1 for s in snapshots if s["ret_1d"] < 0)
    flat_count = len(snapshots) - up_count - down_count
    vol_spike = [s for s in snapshots if s["volume_ratio"] >= 2.0]
    rets = [s["ret_1d"] for s in snapshots]
    avg_ret = round(sum(rets) / len(rets), 2) if rets else 0

    summary = {
        "date": today_date,
        "total": len(snapshots),
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "avg_ret": avg_ret,
        "vol_spike_count": len(vol_spike),
        "top5": [{"name": s["name"], "ret": s["ret_1d"]} for s in snapshots[:5]],
        "bottom5": [{"name": s["name"], "ret": s["ret_1d"]} for s in snapshots[-5:]],
    }

    # ── 일별 JSON 저장 ──
    daily_data = {
        "date": today_date,
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "signal_accuracy": accuracy,
        "missed_opportunities": missed,
        "sar_block_analysis": sar_analysis,
        "ai_insight": ai_insight,
        "pattern_analysis": pattern_analysis or {},
        "pattern_candidates": pattern_candidates or [],
    }
    daily_path = LEARNING_DIR / f"{today_date}.json"
    _save_json(daily_path, daily_data)
    logger.info("일별 학습 저장: %s", daily_path)

    # ── 인덱스 갱신 ──
    idx = _load_json(INDEX_PATH) or {"start_date": today_date, "logs": []}
    if today_date not in idx.get("logs", []):
        idx["logs"].append(today_date)
    idx["total_days"] = len(idx["logs"])
    _save_json(INDEX_PATH, idx)

    # ── 텔레그램 메시지 ──
    if not send_telegram:
        logger.info("텔레그램 미발송 (--no-send)")
        return

    msg = _build_telegram_message(
        today_date, summary, accuracy, missed, cumulative, ai_insight,
        sar_analysis=sar_analysis,
        pattern_text=pattern_text,
    )
    try:
        from src.telegram_sender import send_message
        ok = send_message(msg)
        if ok:
            logger.info("텔레그램 발송 성공")
        else:
            logger.warning("텔레그램 발송 실패")
    except Exception as e:
        logger.error("텔레그램 오류: %s", e)


def _build_telegram_message(
    date_str: str, summary: dict, accuracy: dict,
    missed: list[dict], cumulative: dict, ai_insight: str,
    sar_analysis: dict | None = None,
    pattern_text: str = "",
) -> str:
    lines = [f"📚 일일 시장 학습 ({date_str[5:]})", ""]

    # 유니버스 요약
    lines.append(
        f"📊 유니버스: ↑{summary['up_count']} / ↓{summary['down_count']} "
        f"/ 거래량폭발 {summary['vol_spike_count']}종목"
    )
    lines.append(f"   평균 수익률: {summary['avg_ret']:+.2f}%")
    lines.append("")

    # 시그널 적중률
    lines.append("🎯 시그널 적중률 (오늘):")
    sig_labels = {
        "tomorrow_picks": "추천",
        "pullback_scan": "눌림목",
        "whale_detect": "세력",
        "accumulation_tracker": "매집",
        "dart_event": "DART",
        "volume_spike": "수급폭발",
        "dual_buying": "동반매수",
        "safety_margin": "안전마진",
        "ai_v3_picks": "AI추천",
        "overnight_signal": "오버나잇",
    }
    for sig_name, sig_data in accuracy.items():
        total = sig_data.get("total", 0)
        if total == 0:
            continue
        label = sig_labels.get(sig_name, sig_name)
        hit = sig_data.get("hit", 0)
        hr = sig_data.get("hit_rate", 0)
        avg = sig_data.get("avg_ret", 0)
        lines.append(f"  {label}: {hit}/{total} ({hr}%) avg {avg:+.1f}%")
    lines.append("")

    # 놓친 급등주
    if missed:
        lines.append("⚠️ 놓친 급등주:")
        for m in missed[:3]:
            lines.append(f"  {m['name']} +{m['ret_1d']}% ({m['blocked_by']})")
        lines.append("")

    # SAR 차단 분석
    if sar_analysis and sar_analysis.get("total", 0) > 0:
        sa = sar_analysis
        lines.append(
            f"🔍 SAR 차단 분석: {sa['total']}건 중 "
            f"놓침 {sa['genuine_miss']}건({sa['miss_rate']:.0f}%) / "
            f"정당 {sa['noise_block']}건"
        )
        for tm in sa.get("top_misses", [])[:3]:
            lines.append(f"  {tm['name']} +{tm['ret_1d']:.1f}% [MISS]")
        lines.append("")

    # AI 인사이트
    if ai_insight:
        lines.append("🧠 AI 인사이트:")
        for line in ai_insight.split("\n")[:3]:
            lines.append(f"  {line.strip()}")
        lines.append("")

    # 20일 누적
    cum_sigs = cumulative.get("signals", {})
    cum_parts = []
    for sig_name in ("tomorrow_picks", "pullback_scan", "whale_detect"):
        if sig_name in cum_sigs and cum_sigs[sig_name].get("total", 0) > 0:
            label = sig_labels.get(sig_name, sig_name)
            hr = cum_sigs[sig_name]["hit_rate"]
            cum_parts.append(f"{label} {hr}%")
    if cum_parts:
        lines.append(f"📈 20일 누적: {' | '.join(cum_parts)}")

    # 패턴 학습 결과
    if pattern_text:
        lines.append("")
        lines.append(pattern_text)

    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

def run_daily_learner(
    send_telegram: bool = True,
    use_ai: bool = True,
) -> dict:
    """메인 학습 루프."""
    cfg = _load_config()
    if not cfg.get("enabled", True):
        logger.info("market_learner 비활성화")
        return {}

    if not use_ai:
        cfg["ai_synthesis"] = False

    today_date = datetime.now().strftime("%Y-%m-%d")
    name_map = _build_name_map()

    logger.info("=" * 50)
    logger.info("[Daily Market Learner] %s 시작", today_date)
    logger.info("=" * 50)

    # Phase 1
    snapshots = phase1_universe_snapshot(name_map)
    if not snapshots:
        logger.warning("스냅샷 없음 — 종료")
        return {}

    # Phase 2
    accuracy = phase2_signal_accuracy(snapshots, cfg)

    # Phase 3
    missed = phase3_missed_opportunities(snapshots, accuracy, cfg)
    sar_analysis = _analyze_sar_blocks(missed, snapshots)
    if sar_analysis["total"] > 0:
        logger.info(
            "[Phase 3] SAR 차단 분석: %d건 중 MISS %d건(%.0f%%), OK %d건",
            sar_analysis["total"], sar_analysis["genuine_miss"],
            sar_analysis["miss_rate"], sar_analysis["noise_block"],
        )

    # Phase 4
    cumulative = phase4_cumulative_update(today_date, accuracy, cfg)

    # ── Phase 4.5: 패턴 학습 (급등/급락 원인 분석 + 누적 통계) ──
    pattern_analysis = {}
    pattern_stats = {}
    pattern_candidates = []
    try:
        from src.use_cases.pattern_learner import (
            analyze_movers, accumulate_patterns, find_tomorrow_candidates,
            build_pattern_summary,
        )
        logger.info("[Phase 4.5] 패턴 학습 시작...")
        pattern_analysis = analyze_movers(name_map)
        pattern_stats = accumulate_patterns(pattern_analysis)
        pattern_candidates = find_tomorrow_candidates(pattern_stats, name_map)
        pattern_text = build_pattern_summary(pattern_analysis, pattern_stats, pattern_candidates)
        logger.info("[Phase 4.5] 패턴 학습 완료 — 급등 %d건, 패턴 %d종, 내일 후보 %d건",
                     len(pattern_analysis.get("gainers", [])),
                     len(pattern_stats.get("patterns", {})),
                     len(pattern_candidates))
    except Exception as e:
        logger.warning("[Phase 4.5] 패턴 학습 실패: %s", e)
        pattern_text = ""

    # 스냅샷 요약 (Phase 5용)
    rets = [s["ret_1d"] for s in snapshots]
    snapshot_summary = {
        "up_count": sum(1 for r in rets if r > 0),
        "down_count": sum(1 for r in rets if r < 0),
        "avg_ret": round(sum(rets) / len(rets), 2) if rets else 0,
    }

    # Phase 5
    ai_insight = phase5_ai_synthesis(
        snapshot_summary, accuracy, missed, cumulative, cfg
    )

    # Phase 6
    phase6_save_and_send(
        today_date, snapshots, accuracy, missed,
        cumulative, ai_insight, send_telegram,
        sar_analysis=sar_analysis,
        pattern_analysis=pattern_analysis,
        pattern_text=pattern_text,
        pattern_candidates=pattern_candidates,
    )

    # 콘솔 요약
    _print_console_summary(snapshots, accuracy, missed, cumulative,
                           pattern_analysis, pattern_candidates)

    return {
        "date": today_date,
        "universe_count": len(snapshots),
        "signals_checked": len(accuracy),
        "missed_count": len(missed),
        "pattern_gainers": len(pattern_analysis.get("gainers", [])),
        "pattern_candidates": len(pattern_candidates),
    }


def _print_console_summary(snapshots, accuracy, missed, cumulative,
                           pattern_analysis=None, pattern_candidates=None):
    """콘솔 출력."""
    print("\n" + "=" * 60)
    print("📚 Daily Market Learner — 학습 결과")
    print("=" * 60)

    up = sum(1 for s in snapshots if s["ret_1d"] > 0)
    dn = sum(1 for s in snapshots if s["ret_1d"] < 0)
    print(f"\n유니버스: {len(snapshots)}종목 (↑{up} / ↓{dn})")

    print(f"\n  TOP 3: ", end="")
    for s in snapshots[:3]:
        print(f"{s['name']} {s['ret_1d']:+.1f}%  ", end="")
    print(f"\n  BOT 3: ", end="")
    for s in snapshots[-3:]:
        print(f"{s['name']} {s['ret_1d']:+.1f}%  ", end="")
    print()

    print("\n시그널 적중:")
    for sig_name, stat in accuracy.items():
        total = stat.get("total", 0)
        if total == 0:
            continue
        print(f"  {sig_name:20s}: {stat['hit']}/{total} "
              f"({stat['hit_rate']}%) avg {stat['avg_ret']:+.2f}%")

    if missed:
        print(f"\n놓친 급등주 ({len(missed)}건):")
        for m in missed[:5]:
            print(f"  {m['name']:12s} +{m['ret_1d']}% — {m['blocked_by']}")

    cum_sigs = cumulative.get("signals", {})
    if cum_sigs:
        print(f"\n20일 누적 적중률:")
        for sig, stat in sorted(cum_sigs.items(), key=lambda x: -x[1].get("hit_rate", 0)):
            if stat.get("total", 0) > 0:
                print(f"  {sig:20s}: {stat['hit_rate']}% "
                      f"(n={stat['total']}, avg {stat['avg_ret']:+.2f}%)")

    # 패턴 학습 결과
    if pattern_analysis:
        gainers = pattern_analysis.get("gainers", [])
        losers = pattern_analysis.get("losers", [])
        print(f"\n🔬 패턴 학습: 급등 {len(gainers)}건 / 급락 {len(losers)}건 분석")
        for g in gainers[:3]:
            patterns = ", ".join(g.get("patterns", [])[:2]) or "미분류"
            print(f"  ↑ {g['name']:12s} {g['ret_1d']:+.1f}% [{patterns}]")
        for l in losers[:3]:
            patterns = ", ".join(l.get("patterns", [])[:2]) or "미분류"
            print(f"  ↓ {l['name']:12s} {l['ret_1d']:+.1f}% [{patterns}]")

    if pattern_candidates:
        print(f"\n🎯 패턴 매칭 후보 ({len(pattern_candidates)}건):")
        for c in pattern_candidates[:5]:
            pats = "+".join(c.get("matched_patterns", [])[:2])
            score = c.get("pattern_score", 0)
            print(f"  {c['name']:12s} [{pats}] score={score:.0f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="일일 시장 학습 시스템")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--no-ai", action="store_true", help="AI 인사이트 생략")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_daily_learner(
        send_telegram=not args.no_send,
        use_ai=not args.no_ai,
    )


if __name__ == "__main__":
    main()
