"""
소형주 급등 포착 스캐너 (v12.4) — SmallCap Explosion Scanner

CSV 전종목(2,860+)에서 3가지 폭발 패턴 탐지:
  A) 거래량 폭발 + 돌파 (Volume Breakout)
  B) 세력 털기 후 V자 반등 (Shakeout Reversal)
  C) 연속 양봉 가속 (Consecutive Momentum)

자동매수 연결 안 함 — 정보 제공 + 텔레그램 알림 전용.

Usage:
    python scripts/scan_smallcap_explosion.py
    python scripts/scan_smallcap_explosion.py --no-send
    python scripts/scan_smallcap_explosion.py --backtest 2026-03-05
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# ── 경로 ──
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
UNIVERSE_PATH = PROJECT_ROOT / "data" / "universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "smallcap_explosion.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    """settings.yaml에서 smallcap_explosion 섹션 로드."""
    defaults = {
        "enabled": True,
        "min_trading_value": 5e8,
        "min_price": 1000,
        "exclude_main_universe": True,
        "vol_breakout": {"min_vol_ratio": 2.5},
        "shakeout": {"min_drop_pct": -5, "min_bounce_pct": 3, "min_vol_ratio": 1.0},
        "consecutive": {"min_streak": 3, "min_cum_return": 8},
        "telegram_top_n": 10,
    }
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("smallcap_explosion", defaults)
    except Exception:
        return defaults


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_main_universe() -> set[str]:
    """폭발 패턴 스캐너는 전종목 대상 — 제외 없음.
    메인 파이프라인과 목적이 다름 (알림 전용 vs 매수 시그널).
    exclude_main_universe=true여도 빈 셋 반환 → 사실상 비활성.
    """
    return set()  # 전종목 스캔


def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑."""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def load_csv_universe(cfg: dict, main_universe: set[str],
                      target_date: str | None = None) -> list[tuple[str, pd.DataFrame]]:
    """CSV 전종목에서 소형주 후보 필터."""
    min_tv = float(cfg.get("min_trading_value", 5e8))
    min_price = float(cfg.get("min_price", 1000))
    exclude_main = cfg.get("exclude_main_universe", True)

    candidates = []
    skipped = {"main_universe": 0, "short_data": 0, "low_price": 0, "low_liquidity": 0}

    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        parts = csv_path.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        ticker = parts[1]

        if exclude_main and ticker in main_universe:
            skipped["main_universe"] += 1
            continue

        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df = df.sort_index()
        except Exception:
            continue

        if len(df) < 60:
            skipped["short_data"] += 1
            continue

        # target_date 기준 인덱스 찾기
        if target_date:
            td = pd.Timestamp(target_date)
            valid = df.index[df.index <= td]
            if len(valid) < 60:
                skipped["short_data"] += 1
                continue
            df = df.loc[:td]

        last = df.iloc[-1]
        close = float(last.get("Close", 0))

        if close < min_price:
            skipped["low_price"] += 1
            continue

        vol_20d = df["Volume"].iloc[-20:].mean()
        trading_value_20d = close * vol_20d

        if trading_value_20d < min_tv:
            skipped["low_liquidity"] += 1
            continue

        candidates.append((ticker, df))

    return candidates, skipped


# ═══════════════════════════════════════════════
# Pattern A: 거래량 폭발 + 돌파
# ═══════════════════════════════════════════════

def detect_volume_breakout(df: pd.DataFrame, idx: int, cfg: dict) -> dict | None:
    """거래량 급증 + 가격 돌파 감지."""
    min_vr = cfg.get("vol_breakout", {}).get("min_vol_ratio", 2.5)

    vol = float(df["Volume"].iloc[idx])
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx].mean())  # 오늘 제외
    if vol_ma20 <= 0:
        return None
    vol_ratio = vol / vol_ma20

    if vol_ratio < min_vr:
        return None

    close = float(df["Close"].iloc[idx])
    high_20 = float(df["High"].iloc[max(0, idx - 20):idx].max())
    high_60 = float(df["High"].iloc[max(0, idx - 60):idx].max())

    bb_upper = float(df["Upper_Band"].iloc[idx]) if "Upper_Band" in df.columns else high_20

    breakout_type = None
    if close > high_60:
        breakout_type = "60일_신고가"
    elif close > high_20:
        breakout_type = "20일_돌파"
    elif close > bb_upper and bb_upper > 0:
        breakout_type = "BB상단_돌파"
    else:
        return None

    # 점수
    score = 50
    score += min(int(vol_ratio * 5), 25)
    if breakout_type == "60일_신고가":
        score += 15
    elif breakout_type == "20일_돌파":
        score += 10
    else:
        score += 5

    rsi = float(df["RSI"].iloc[idx]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[idx]) else 50
    if rsi < 75:
        score += 5

    return {
        "pattern": "A_VOLUME_BREAKOUT",
        "score": min(score, 100),
        "vol_ratio": round(vol_ratio, 1),
        "breakout": breakout_type,
        "rsi": round(rsi, 1),
    }


# ═══════════════════════════════════════════════
# Pattern B: 세력 털기 후 V자 반등
# ═══════════════════════════════════════════════

def detect_shakeout_reversal(df: pd.DataFrame, idx: int, cfg: dict) -> dict | None:
    """급락(-5%+) 후 반등(+3%+) = V자 반전."""
    if idx < 5:
        return None

    sh_cfg = cfg.get("shakeout", {})
    min_drop = sh_cfg.get("min_drop_pct", -5)
    min_bounce = sh_cfg.get("min_bounce_pct", 3)
    min_vr = sh_cfg.get("min_vol_ratio", 1.0)

    closes = df["Close"].values

    today_chg = (closes[idx] / closes[idx - 1] - 1) * 100 if closes[idx - 1] > 0 else 0
    yest_chg = (closes[idx - 1] / closes[idx - 2] - 1) * 100 if closes[idx - 2] > 0 else 0

    shakeout = None

    # 케이스 1: 어제 급락 → 오늘 급반등
    if yest_chg <= min_drop and today_chg >= min_bounce:
        shakeout = {"drop_day": -1, "drop_pct": yest_chg, "bounce_pct": today_chg}

    # 케이스 2: 2일전 급락 → 오늘까지 반등
    if not shakeout and idx >= 3 and closes[idx - 3] > 0:
        day2_chg = (closes[idx - 2] / closes[idx - 3] - 1) * 100
        bounce_2d = (closes[idx] / closes[idx - 2] - 1) * 100
        if day2_chg <= min_drop and bounce_2d >= min_bounce * 1.5:
            shakeout = {"drop_day": -2, "drop_pct": day2_chg, "bounce_pct": bounce_2d}

    if not shakeout:
        return None

    # 더 큰 추세 안의 shakeout인지 확인
    close_5d_ago = closes[max(0, idx - 5)]
    trend_5d = (closes[idx] / close_5d_ago - 1) * 100 if close_5d_ago > 0 else 0
    if trend_5d < 0:
        return None  # 하락 추세 중 반등 = 데드캣 바운스 위험

    # 거래량 확인
    vol = float(df["Volume"].iloc[idx])
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx].mean())
    vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio < min_vr:
        return None

    # 점수
    score = 55
    score += min(int(abs(shakeout["drop_pct"]) * 2), 15)
    score += min(int(shakeout["bounce_pct"] * 2), 15)
    score += min(int(vol_ratio * 5), 10)
    if trend_5d >= 10:
        score += 5

    return {
        "pattern": "B_SHAKEOUT_REVERSAL",
        "score": min(score, 100),
        "drop_pct": round(shakeout["drop_pct"], 1),
        "bounce_pct": round(shakeout["bounce_pct"], 1),
        "drop_day": shakeout["drop_day"],
        "vol_ratio": round(vol_ratio, 1),
        "trend_5d": round(trend_5d, 1),
    }


# ═══════════════════════════════════════════════
# Pattern C: 연속 양봉 가속
# ═══════════════════════════════════════════════

def detect_consecutive_momentum(df: pd.DataFrame, idx: int, cfg: dict) -> dict | None:
    """3일+ 연속 양봉 + 거래량 증가 + 누적 8%+."""
    if idx < 5:
        return None

    con_cfg = cfg.get("consecutive", {})
    min_streak = con_cfg.get("min_streak", 3)
    min_cum = con_cfg.get("min_cum_return", 8)

    closes = df["Close"].values

    # 연속 양봉 카운트
    streak = 0
    for i in range(idx, max(idx - 10, 0), -1):
        if closes[i] > closes[i - 1]:
            streak += 1
        else:
            break

    if streak < min_streak:
        return None

    # 누적 상승률
    base_close = closes[idx - streak]
    if base_close <= 0:
        return None
    cum_return = (closes[idx] / base_close - 1) * 100

    if cum_return < min_cum:
        return None

    # 거래량 추세
    vol_recent = float(df["Volume"].iloc[max(0, idx - 2):idx + 1].mean())
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx + 1].mean())
    vol_accel = vol_recent / vol_ma20 if vol_ma20 > 0 else 1

    if vol_accel < 0.8:
        return None

    # 점수
    score = 50
    score += min(streak * 5, 20)
    score += min(int(cum_return), 20)
    score += min(int(vol_accel * 5), 10)

    return {
        "pattern": "C_CONSECUTIVE_MOMENTUM",
        "score": min(score, 100),
        "streak": streak,
        "cum_return": round(cum_return, 1),
        "vol_accel": round(vol_accel, 2),
    }


# ═══════════════════════════════════════════════
# 메인 스캔
# ═══════════════════════════════════════════════

def scan_all(cfg: dict, main_universe: set[str], name_map: dict[str, str],
             target_date: str | None = None) -> list[dict]:
    """전종목 스캔 → 3패턴 탐지 → 결과 정렬."""
    t0 = time.time()

    candidates, skipped = load_csv_universe(cfg, main_universe, target_date)
    print(f"  후보: {len(candidates)}종목 (제외: 메인유니{skipped['main_universe']}, "
          f"데이터부족{skipped['short_data']}, 저가{skipped['low_price']}, "
          f"유동성{skipped['low_liquidity']})")

    results = []
    scan_count = 0

    for ticker, df in candidates:
        idx = len(df) - 1
        if idx < 10:
            continue

        signals = []

        a = detect_volume_breakout(df, idx, cfg)
        b = detect_shakeout_reversal(df, idx, cfg)
        c = detect_consecutive_momentum(df, idx, cfg)

        for det in [a, b, c]:
            if det:
                signals.append(det)

        if not signals:
            continue

        # 복합 패턴 보너스
        best_score = max(s["score"] for s in signals)
        if len(signals) >= 2:
            best_score = min(best_score + 10, 100)
        if len(signals) >= 3:
            best_score = min(best_score + 10, 100)

        close = float(df["Close"].iloc[idx])
        prev_close = float(df["Close"].iloc[idx - 1]) if idx > 0 else close
        change_pct = (close / prev_close - 1) * 100 if prev_close > 0 else 0

        # 20일 평균 거래대금
        vol_20d = float(df["Volume"].iloc[-20:].mean())
        trading_value = close * vol_20d

        results.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "close": int(close),
            "change_pct": round(change_pct, 1),
            "trading_value_억": round(trading_value / 1e8, 1),
            "patterns": signals,
            "pattern_names": [s["pattern"] for s in signals],
            "top_score": best_score,
            "combined_count": len(signals),
        })

        scan_count += 1

    results.sort(key=lambda x: -x["top_score"])

    elapsed = time.time() - t0
    print(f"  스캔 완료: {len(candidates)}종목 → {len(results)}건 감지 ({elapsed:.1f}s)")

    return results


def send_telegram(results: list[dict], cfg: dict):
    """텔레그램 알림 발송."""
    top_n = cfg.get("telegram_top_n", 10)
    if not results:
        return

    top = results[:top_n]
    lines = [f"🔥 소형주 급등 포착 ({len(results)}건 중 상위 {len(top)})"]
    lines.append("")

    for i, r in enumerate(top, 1):
        patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
        lines.append(
            f"{i}. {r['name']}({r['ticker']}) {r['change_pct']:+.1f}% "
            f"점수:{r['top_score']} [{patterns}]"
        )
        for p in r["patterns"]:
            if p["pattern"] == "A_VOLUME_BREAKOUT":
                lines.append(f"   ↳ 거래량 {p['vol_ratio']}배 + {p['breakout']}")
            elif p["pattern"] == "B_SHAKEOUT_REVERSAL":
                lines.append(f"   ↳ {p['drop_pct']:+.1f}% 급락 → {p['bounce_pct']:+.1f}% 반등 (5일 {p['trend_5d']:+.1f}%)")
            elif p["pattern"] == "C_CONSECUTIVE_MOMENTUM":
                lines.append(f"   ↳ {p['streak']}일 연속↑ 누적 {p['cum_return']:+.1f}%")

    msg = "\n".join(lines)
    print(f"\n{msg}")

    try:
        from src.telegram_sender import send_message
        send_message(msg)
        print("  [텔레그램 발송 완료]")
    except Exception as e:
        print(f"  [텔레그램 실패: {e}]")


def main():
    parser = argparse.ArgumentParser(description="소형주 급등 포착 스캐너")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--backtest", type=str, default=None,
                        help="특정 날짜 기준 스캔 (예: 2026-03-05)")
    args = parser.parse_args()

    cfg = load_settings()
    if not cfg.get("enabled", True):
        print("소형주 급등 스캐너 비활성 (settings.yaml)")
        return

    scan_date = args.backtest or datetime.now().strftime("%Y-%m-%d")
    print(f"[소형주 급등 포착] {scan_date}")
    print("=" * 55)

    main_universe = load_main_universe()
    name_map = build_name_map()

    print(f"  메인 유니버스: {len(main_universe)}종목 (제외 대상)")
    print(f"  CSV 전종목: {len(list(CSV_DIR.glob('*.csv')))}개")

    results = scan_all(cfg, main_universe, name_map,
                       target_date=args.backtest)

    # 저장
    output = {
        "scan_date": scan_date,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_detected": len(results),
        "signals": results[:50],  # 상위 50건만 저장
        "pattern_summary": {
            "A_VOLUME_BREAKOUT": len([r for r in results if "A_VOLUME_BREAKOUT" in r["pattern_names"]]),
            "B_SHAKEOUT_REVERSAL": len([r for r in results if "B_SHAKEOUT_REVERSAL" in r["pattern_names"]]),
            "C_CONSECUTIVE_MOMENTUM": len([r for r in results if "C_CONSECUTIVE_MOMENTUM" in r["pattern_names"]]),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {OUTPUT_PATH}")

    # 상위 결과 출력
    if results:
        print(f"\n{'─' * 70}")
        print(f"  소형주 급등 포착 — 상위 {min(20, len(results))}건")
        print(f"{'─' * 70}")
        print(f"  {'#':>2} {'종목':>10} {'종가':>8} {'변동':>7} {'점수':>4} {'패턴':>20} {'거래대금':>8}")
        print(f"  {'-'*2} {'-'*10} {'-'*8} {'-'*7} {'-'*4} {'-'*20} {'-'*8}")
        for i, r in enumerate(results[:20], 1):
            patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
            print(f"  {i:>2} {r['name'][:8]:>10} {r['close']:>8,} {r['change_pct']:>+6.1f}% "
                  f"{r['top_score']:>4} {patterns:>20} {r['trading_value_억']:>7.0f}억")
    else:
        print("\n  감지된 패턴 없음")

    # 텔레그램
    if not args.no_send and results:
        send_telegram(results, cfg)

    print(f"\n  패턴 분포: A(거래량돌파) {output['pattern_summary']['A_VOLUME_BREAKOUT']}건 "
          f"/ B(V자반등) {output['pattern_summary']['B_SHAKEOUT_REVERSAL']}건 "
          f"/ C(연속양봉) {output['pattern_summary']['C_CONSECUTIVE_MOMENTUM']}건")


if __name__ == "__main__":
    main()
