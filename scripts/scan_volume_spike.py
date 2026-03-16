"""
수급 폭발 → 조정 매수 스캐너 (전략 A)

매일 전체 parquet에서:
  1. vol_z >= 3.0 또는 volume_surge_ratio >= 3.0 종목 감지 → watchlist 추가
  2. watchlist 중 스파이크 고가 대비 -3% ~ -10% 조정 중인 종목 → 매수 시그널 발동
  3. 30일 경과 or -15% 이상 급락 → watchlist 제거

실증: 이 패턴에서 10일 반등 승률 77.3%, 평균 +9.23%

출력: data/volume_spike_watchlist.json
BAT-D 12.5단계 (눌림목 직후)

Usage:
    python scripts/scan_volume_spike.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
KOSPI_PATH = DATA_DIR / "kospi_index.csv"
OUTPUT_PATH = DATA_DIR / "volume_spike_watchlist.json"

# ── 파라미터 ──
SPIKE_VOL_Z = 3.0              # vol_z 임계치
SPIKE_VSR = 3.0                # volume_surge_ratio 임계치
SPIKE_AMOUNT_RATIO = 3.0       # 거래대금 20일MA 대비 배율 (AND 조건)
# 레짐 게이트: BEAR/CRISIS에서는 스파이크 매수 비활성
REGIME_GATE_ENABLED = True     # False로 끄면 레짐 무시
PULLBACK_MIN_PCT = -3.0        # 조정 최소 % (스파이크 고가 대비)
PULLBACK_MAX_PCT = -10.0       # 조정 최대 %
WATCHLIST_EXPIRY_DAYS = 30     # 감시 기간 (일)
MAX_DROP_PCT = -15.0           # 급락 제거 기준


def build_name_map() -> dict[str, str]:
    """종목코드 → 종목명 매핑."""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def load_existing_watchlist() -> dict:
    """기존 watchlist 로드 (누적 관리)."""
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"watching": {}, "signals": [], "stats": {}}


def scan_new_spikes(name_map: dict) -> dict[str, dict]:
    """전체 parquet 스캔하여 오늘의 새 스파이크 감지."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_spikes = {}

    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 21:
                continue
            last = df.iloc[-1]
            vol_z = float(last.get("vol_z", 0) or 0)
            vsr = float(last.get("volume_surge_ratio", 0) or 0)

            if pd.isna(vol_z):
                vol_z = 0
            if pd.isna(vsr):
                vsr = 0

            if vol_z >= SPIKE_VOL_Z or vsr >= SPIKE_VSR:
                close = float(last.get("close", 0))
                high = float(last.get("high", close))
                volume = float(last.get("volume", 0) or 0)

                # ── 거래대금 AND 필터 (저가주 거짓 신호 제거) ──
                amount = close * volume
                amount_ma20 = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1]
                if amount_ma20 > 0:
                    amount_ratio = amount / amount_ma20
                else:
                    amount_ratio = 0.0

                if amount_ratio < SPIKE_AMOUNT_RATIO:
                    continue  # 거래대금 기준 미달 → 스킵

                rsi = float(last.get("rsi_14", 50) or 50)
                foreign_5d = float(last.get("foreign_net_5d", 0) or 0)

                new_spikes[ticker] = {
                    "spike_date": today_str,
                    "spike_close": round(close, 0),
                    "spike_high": round(max(high, close), 0),
                    "vol_z": round(vol_z, 2),
                    "vsr": round(vsr, 2),
                    "amount_ratio": round(amount_ratio, 2),
                    "name": name_map.get(ticker, ticker),
                    "rsi_at_spike": round(rsi, 1) if not pd.isna(rsi) else 50,
                    "status": "watching",
                }
        except Exception as e:
            logger.debug("스캔 실패 %s: %s", ticker, e)

    return new_spikes


def check_pullback_signals(
    watching: dict[str, dict], name_map: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """감시 중인 종목의 조정 여부 체크 → 시그널 또는 만료 판정."""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    signals = []
    updated = {}

    for ticker, info in watching.items():
        pq_path = PROCESSED_DIR / f"{ticker}.parquet"
        if not pq_path.exists():
            continue

        try:
            df = pd.read_parquet(pq_path)
            if len(df) < 2:
                continue
            last = df.iloc[-1]
            current_close = float(last.get("close", 0))
            spike_high = float(info.get("spike_high", 0))
            spike_date_str = info.get("spike_date", today_str)

            # 경과일 계산
            try:
                spike_dt = datetime.strptime(spike_date_str, "%Y-%m-%d")
                days_elapsed = (today - spike_dt).days
            except ValueError:
                days_elapsed = 0

            # 만료 조건
            if days_elapsed > WATCHLIST_EXPIRY_DAYS:
                continue  # 만료 → 제거
            if spike_high <= 0:
                continue

            # 조정폭 (스파이크 고가 대비)
            pullback_pct = (current_close / spike_high - 1) * 100

            if pullback_pct <= MAX_DROP_PCT:
                continue  # 급락 → 제거

            # 시그널 발동 조건: -3% ~ -10% 조정 + RSI < 65
            rsi = float(last.get("rsi_14", 50) or 50)
            if pd.isna(rsi):
                rsi = 50

            if PULLBACK_MAX_PCT <= pullback_pct <= PULLBACK_MIN_PCT and rsi < 65:
                foreign_5d = float(last.get("foreign_net_5d", 0) or 0)
                if pd.isna(foreign_5d):
                    foreign_5d = 0
                vol_z_now = float(last.get("vol_z", 0) or 0)

                score = _calc_spike_score(
                    pullback_pct, days_elapsed, rsi, foreign_5d,
                    info.get("vol_z", 0),
                )

                signals.append({
                    "ticker": ticker,
                    "name": info.get("name", name_map.get(ticker, ticker)),
                    "spike_date": spike_date_str,
                    "spike_high": int(spike_high),
                    "current_close": int(current_close),
                    "pullback_pct": round(pullback_pct, 1),
                    "days_since_spike": days_elapsed,
                    "rsi": round(rsi, 1),
                    "foreign_5d": round(foreign_5d, 0),
                    "vol_z_at_spike": info.get("vol_z", 0),
                    "score": score,
                })

                info["status"] = "signal"
            else:
                info["status"] = "watching"

            updated[ticker] = info

        except Exception as e:
            logger.debug("체크 실패 %s: %s", ticker, e)
            updated[ticker] = info

    # 점수 내림차순 정렬
    signals.sort(key=lambda x: -x["score"])

    return updated, signals


def _calc_spike_score(
    pullback_pct: float, days: int, rsi: float,
    foreign_5d: float, vol_z: float,
) -> int:
    """조정 매수 점수 (0~100)."""
    score = 50  # 기본 (조정 구간 진입 자체로 50점)

    # 조정폭: -5% ~ -8%가 최적
    if -8 <= pullback_pct <= -5:
        score += 20
    elif -10 <= pullback_pct <= -3:
        score += 10

    # 경과일: 5~15일이 최적
    if 5 <= days <= 15:
        score += 10
    elif 3 <= days <= 20:
        score += 5

    # RSI: 30~50이면 보너스
    if 30 <= rsi <= 50:
        score += 10
    elif rsi <= 60:
        score += 5

    # 외인 순매수 전환
    if foreign_5d > 0:
        score += 10

    return min(score, 100)


def get_kospi_regime() -> str:
    """KOSPI 레짐 판별 (v10.3 기준).

    BULL: MA20 위 + RV20 < 50%ile
    CAUTION: MA20 위 + RV20 >= 50%ile
    BEAR: MA20~MA60 사이
    CRISIS: MA60 아래
    """
    try:
        df = pd.read_csv(KOSPI_PATH, index_col=0, parse_dates=True)
        if len(df) < 60:
            return "UNKNOWN"
        col = "close"
        close = df[col].iloc[-1]
        ma20 = df[col].rolling(20).mean().iloc[-1]
        ma60 = df[col].rolling(60).mean().iloc[-1]
        rv_pctile = df[col].pct_change().rolling(20).std().rank(pct=True).iloc[-1]

        if close > ma20 and rv_pctile < 0.5:
            return "BULL"
        elif close > ma20:
            return "CAUTION"
        elif close > ma60:
            return "BEAR"
        else:
            return "CRISIS"
    except Exception as e:
        logger.warning("KOSPI 레짐 판별 실패: %s", e)
        return "UNKNOWN"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── 레짐 게이트 ──
    regime = get_kospi_regime()
    print(f"[레짐] KOSPI: {regime}")

    if REGIME_GATE_ENABLED and regime in ("BEAR", "CRISIS"):
        print(f"[레짐 게이트] {regime} → 스파이크 매수 비활성. 하락장에서 스파이크는 투매.")
        # watchlist는 유지하되 새 스파이크 감지/시그널 발동 안 함
        existing = load_existing_watchlist()
        existing["regime"] = regime
        existing["regime_gate"] = "BLOCKED"
        existing["date"] = datetime.now().strftime("%Y-%m-%d")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
        print(f"[저장] {OUTPUT_PATH} (레짐 게이트 차단)")
        return

    name_map = build_name_map()

    # 기존 watchlist 로드
    existing = load_existing_watchlist()
    watching = existing.get("watching", {})

    # 1) 새 스파이크 감지
    new_spikes = scan_new_spikes(name_map)
    print(f"[수급 폭발] 신규 스파이크: {len(new_spikes)}종목")

    # 2) 기존 watching에 신규 추가 (이미 있으면 최신으로 갱신)
    for ticker, info in new_spikes.items():
        if ticker not in watching:
            watching[ticker] = info

    # 3) 조정 시그널 체크
    watching, signals = check_pullback_signals(watching, name_map)
    print(f"[수급 폭발] 감시 중: {len(watching)}종목, 시그널: {len(signals)}종목")

    # 4) 저장
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "regime_gate": "PASS",
        "watching": watching,
        "signals": signals,
        "stats": {
            "new_spikes": len(new_spikes),
            "total_watching": len(watching),
            "active_signals": len(signals),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"[저장] {OUTPUT_PATH}")

    # 신규 스파이크 요약 출력
    if new_spikes:
        print(f"\n  신규 스파이크:")
        for t, info in list(new_spikes.items())[:10]:
            ar = info.get("amount_ratio", 0)
            print(f"    {info['name']}({t}) vol_z:{info['vol_z']} vsr:{info['vsr']} 거래대금:{ar:.1f}x")

    # 시그널 요약 출력
    if signals:
        print(f"\n{'─' * 55}")
        print(f"  수급 폭발 → 조정 매수 시그널 ({len(signals)}건)")
        print(f"{'─' * 55}")
        for s in signals[:15]:
            print(
                f"  {s['name']}({s['ticker']}) "
                f"점수:{s['score']} 조정:{s['pullback_pct']:+.1f}% "
                f"RSI:{s['rsi']:.0f} "
                f"({s['days_since_spike']}일전 폭발)"
            )
    else:
        print("  현재 조정 매수 시그널 없음")


if __name__ == "__main__":
    main()
