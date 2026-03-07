"""
소형주 급등 포착 스캐너 v2 — C 중심 재설계

백테스트 검증 결과 (12,425건):
  - A+B 복합: 승률 35%, 20d -0.7% → 데드캣 바운스 함정
  - C 단독:   승률 51-52%, 20d +5.8% → 가장 신뢰
  - A+C 복합: 승률 50%, 20d +7.0% → 최고 수익

시그널 등급:
  PRIMARY  → C 단독, A+C 복합 (매수 후보)
  WARNING  → A+B 복합 (데드캣 바운스 경고)
  ALERT    → B 단독 (고위험, 정보 전용)
  WATCH    → A 단독 (모니터링)

수급 확인: Foreign_Net + Inst_Net → CONFIRMED / PARTIAL / NONE

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
OUTPUT_PATH = PROJECT_ROOT / "data" / "smallcap_explosion.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# ── 시그널 등급 (백테스트 검증 기반) ──
GRADE_PRIMARY = "PRIMARY"    # C, A+C → 실전 유효
GRADE_WARNING = "WARNING"    # A+B   → 데드캣 바운스 경고
GRADE_ALERT = "ALERT"        # B     → 고위험 정보
GRADE_WATCH = "WATCH"        # A     → 모니터링

# 등급별 정렬 우선순위 (낮을수록 우선)
GRADE_PRIORITY = {GRADE_PRIMARY: 0, GRADE_ALERT: 1, GRADE_WATCH: 2, GRADE_WARNING: 3}


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


def load_main_universe() -> set[str]:
    """전종목 스캔 — 제외 없음."""
    return set()


def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑."""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def load_csv_universe(cfg: dict, main_universe: set[str],
                      target_date: str | None = None) -> tuple[list, dict]:
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
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx].mean())
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

    return {
        "pattern": "A_VOLUME_BREAKOUT",
        "vol_ratio": round(vol_ratio, 1),
        "breakout": breakout_type,
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

    if yest_chg <= min_drop and today_chg >= min_bounce:
        shakeout = {"drop_day": -1, "drop_pct": yest_chg, "bounce_pct": today_chg}

    if not shakeout and idx >= 3 and closes[idx - 3] > 0:
        day2_chg = (closes[idx - 2] / closes[idx - 3] - 1) * 100
        bounce_2d = (closes[idx] / closes[idx - 2] - 1) * 100
        if day2_chg <= min_drop and bounce_2d >= min_bounce * 1.5:
            shakeout = {"drop_day": -2, "drop_pct": day2_chg, "bounce_pct": bounce_2d}

    if not shakeout:
        return None

    close_5d_ago = closes[max(0, idx - 5)]
    trend_5d = (closes[idx] / close_5d_ago - 1) * 100 if close_5d_ago > 0 else 0
    if trend_5d < 0:
        return None

    vol = float(df["Volume"].iloc[idx])
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx].mean())
    vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio < min_vr:
        return None

    return {
        "pattern": "B_SHAKEOUT_REVERSAL",
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

    streak = 0
    for i in range(idx, max(idx - 10, 0), -1):
        if closes[i] > closes[i - 1]:
            streak += 1
        else:
            break

    if streak < min_streak:
        return None

    base_close = closes[idx - streak]
    if base_close <= 0:
        return None
    cum_return = (closes[idx] / base_close - 1) * 100

    if cum_return < min_cum:
        return None

    vol_recent = float(df["Volume"].iloc[max(0, idx - 2):idx + 1].mean())
    vol_ma20 = float(df["Volume"].iloc[max(0, idx - 19):idx + 1].mean())
    vol_accel = vol_recent / vol_ma20 if vol_ma20 > 0 else 1

    if vol_accel < 0.8:
        return None

    return {
        "pattern": "C_CONSECUTIVE_MOMENTUM",
        "streak": streak,
        "cum_return": round(cum_return, 1),
        "vol_accel": round(vol_accel, 2),
    }


# ═══════════════════════════════════════════════
# 수급 확인 (Foreign_Net + Inst_Net)
# ═══════════════════════════════════════════════

def check_supply_demand(df: pd.DataFrame, idx: int) -> dict:
    """수급 확인 — OBV 기반 프록시.

    CSV의 Foreign_Net/Inst_Net은 소형주에서 2~4%만 비영 → 사실상 사용 불가.
    대안: OBV(On-Balance Volume) 추세로 수급 판단.
      - OBV 5일 > OBV 20일: 자금 유입 추세
      - 거래량 5일/20일 비율: 관심도 증가

    Returns:
        supply_grade: CONFIRMED / PARTIAL / NONE
        obv_trend: OBV 5일 변화율
        vol_ratio_5_20: 거래량 5일/20일 비율
    """
    # OBV 추세 (5일 vs 20일 평균)
    has_obv = "OBV" in df.columns
    obv_trend = 0.0
    if has_obv:
        obv = df["OBV"].values
        obv_5 = np.mean(obv[max(0, idx - 4):idx + 1])
        obv_20 = np.mean(obv[max(0, idx - 19):idx + 1])
        if obv_20 != 0:
            obv_trend = (obv_5 / obv_20 - 1) * 100
        # OBV 방향: 최근 3일 연속 상승?
        obv_rising = all(obv[i] > obv[i - 1] for i in range(max(1, idx - 2), idx + 1))
    else:
        obv_rising = False

    # 거래량 비율 (5d/20d)
    vol = df["Volume"].values.astype(float)
    vol_5 = np.mean(vol[max(0, idx - 4):idx + 1])
    vol_20 = np.mean(vol[max(0, idx - 19):idx + 1])
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

    # Foreign_Net/Inst_Net (있으면 보너스)
    foreign_buying = False
    inst_buying = False
    start = max(0, idx - 2)
    if "Foreign_Net" in df.columns:
        fn_sum = np.nansum(df["Foreign_Net"].iloc[start:idx + 1].values)
        foreign_buying = fn_sum > 0
    if "Inst_Net" in df.columns:
        in_sum = np.nansum(df["Inst_Net"].iloc[start:idx + 1].values)
        inst_buying = in_sum > 0

    # 등급 판정
    # CONFIRMED: OBV 상승 + 거래량 1.5배+ (또는 기관/외인 매수)
    # PARTIAL: OBV 상승 OR 거래량 1.2배+
    # NONE: 둘 다 아님
    if (obv_rising and vol_ratio >= 1.5) or (foreign_buying and inst_buying):
        grade = "CONFIRMED"
    elif obv_rising or vol_ratio >= 1.2 or foreign_buying or inst_buying:
        grade = "PARTIAL"
    else:
        grade = "NONE"

    return {
        "supply_grade": grade,
        "obv_trend": round(obv_trend, 1),
        "vol_ratio_5_20": round(vol_ratio, 2),
    }


# ═══════════════════════════════════════════════
# 시그널 등급 결정
# ═══════════════════════════════════════════════

def classify_signal(pattern_names: list[str]) -> str:
    """백테스트 검증 기반 시그널 등급 결정.

    검증 결과:
      C 단독:  51-52% 승률, +5.8% 20d → PRIMARY
      A+C:     50% 승률, +7.0% 20d   → PRIMARY
      A+B:     35% 승률, -0.7% 20d   → WARNING (함정)
      B 단독:  40% 승률, +4.0% 20d   → ALERT (고위험)
      A 단독:  44% 승률, +4.6% 20d   → WATCH
    """
    has_a = "A_VOLUME_BREAKOUT" in pattern_names
    has_b = "B_SHAKEOUT_REVERSAL" in pattern_names
    has_c = "C_CONSECUTIVE_MOMENTUM" in pattern_names

    # A+B 복합 = 데드캣 바운스 (승률 35%, 20d -0.7%)
    if has_a and has_b and not has_c:
        return GRADE_WARNING

    # C 포함 패턴 = 검증된 모멘텀
    if has_c:
        return GRADE_PRIMARY  # C 단독, A+C, B+C, A+B+C 모두

    # B 단독 = 고위험 (승률 40%, 대박 가능성)
    if has_b:
        return GRADE_ALERT

    # A 단독 = 모니터링 (승률 44%)
    if has_a:
        return GRADE_WATCH

    return GRADE_WATCH


# ═══════════════════════════════════════════════
# 메인 스캔
# ═══════════════════════════════════════════════

def scan_all(cfg: dict, main_universe: set[str], name_map: dict[str, str],
             target_date: str | None = None) -> list[dict]:
    """전종목 스캔 → 3패턴 탐지 → 등급/수급 분류 → 결과 정렬."""
    t0 = time.time()

    candidates, skipped = load_csv_universe(cfg, main_universe, target_date)
    print(f"  후보: {len(candidates)}종목 (제외: 메인유니{skipped['main_universe']}, "
          f"데이터부족{skipped['short_data']}, 저가{skipped['low_price']}, "
          f"유동성{skipped['low_liquidity']})")

    results = []

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

        pattern_names = [s["pattern"] for s in signals]

        # 등급 결정 (백테스트 검증 기반)
        grade = classify_signal(pattern_names)

        # 수급 확인
        supply = check_supply_demand(df, idx)

        close = float(df["Close"].iloc[idx])
        prev_close = float(df["Close"].iloc[idx - 1]) if idx > 0 else close
        change_pct = (close / prev_close - 1) * 100 if prev_close > 0 else 0

        vol_20d = float(df["Volume"].iloc[-20:].mean())
        trading_value = close * vol_20d

        results.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "close": int(close),
            "change_pct": round(change_pct, 1),
            "trading_value_억": round(trading_value / 1e8, 1),
            "grade": grade,
            "supply_grade": supply["supply_grade"],
            "obv_trend": supply["obv_trend"],
            "vol_ratio_5_20": supply["vol_ratio_5_20"],
            "patterns": signals,
            "pattern_names": pattern_names,
            "combined_count": len(signals),
        })

    # 정렬: 등급 우선순위 → 수급 확인 → 거래대금
    supply_order = {"CONFIRMED": 0, "PARTIAL": 1, "NONE": 2, "N/A": 3}
    results.sort(key=lambda x: (
        GRADE_PRIORITY.get(x["grade"], 9),
        supply_order.get(x["supply_grade"], 9),
        -x["trading_value_억"],
    ))

    elapsed = time.time() - t0
    print(f"  스캔 완료: {len(candidates)}종목 → {len(results)}건 감지 ({elapsed:.1f}s)")

    return results


def send_telegram(results: list[dict], cfg: dict):
    """텔레그램 알림 — 등급별 구분 발송."""
    top_n = cfg.get("telegram_top_n", 10)
    if not results:
        return

    primary = [r for r in results if r["grade"] == GRADE_PRIMARY]
    warnings = [r for r in results if r["grade"] == GRADE_WARNING]
    alerts = [r for r in results if r["grade"] == GRADE_ALERT]

    lines = [f"[소형주 급등 v2] {len(results)}건 감지"]
    lines.append("")

    # PRIMARY 시그널 (C, A+C)
    if primary:
        confirmed = [r for r in primary if r["supply_grade"] == "CONFIRMED"]
        partial = [r for r in primary if r["supply_grade"] == "PARTIAL"]
        none_supply = [r for r in primary if r["supply_grade"] not in ("CONFIRMED", "PARTIAL")]

        if confirmed:
            lines.append(f"== PRIMARY + 수급확인 ({len(confirmed)}건) ==")
            for r in confirmed[:5]:
                patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
                lines.append(f"  {r['name']}({r['ticker']}) {r['change_pct']:+.1f}% [{patterns}]")
                lines.append(f"    OBV:{r['obv_trend']:+.1f}% Vol:{r['vol_ratio_5_20']:.1f}x")
            lines.append("")

        if partial:
            lines.append(f"== PRIMARY + 편측수급 ({len(partial)}건) ==")
            for r in partial[:5]:
                patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
                lines.append(f"  {r['name']}({r['ticker']}) {r['change_pct']:+.1f}% [{patterns}]"
                              f" Vol:{r['vol_ratio_5_20']:.1f}x")
            lines.append("")

        if none_supply:
            lines.append(f"== PRIMARY 수급미확인 ({len(none_supply)}건) ==")
            for r in none_supply[:3]:
                patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
                lines.append(f"  {r['name']}({r['ticker']}) {r['change_pct']:+.1f}% [{patterns}]")
            lines.append("")

    # WARNING (A+B 데드캣)
    if warnings:
        lines.append(f"-- WARNING: 데드캣 주의 ({len(warnings)}건) --")
        for r in warnings[:3]:
            lines.append(f"  {r['name']}({r['ticker']}) {r['change_pct']:+.1f}%")
        lines.append("")

    # ALERT (B 고위험)
    if alerts:
        lines.append(f"-- ALERT: 고위험 ({len(alerts)}건) --")
        for r in alerts[:3]:
            b_info = next((p for p in r["patterns"] if p["pattern"] == "B_SHAKEOUT_REVERSAL"), {})
            drop = b_info.get("drop_pct", 0)
            bounce = b_info.get("bounce_pct", 0)
            lines.append(f"  {r['name']}({r['ticker']}) {drop:+.1f}%→{bounce:+.1f}%")

    msg = "\n".join(lines)
    print(f"\n{msg}")

    try:
        from src.telegram_sender import send_message
        send_message(msg)
        print("  [텔레그램 발송 완료]")
    except Exception as e:
        print(f"  [텔레그램 실패: {e}]")


def main():
    parser = argparse.ArgumentParser(description="소형주 급등 포착 스캐너 v2")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--backtest", type=str, default=None,
                        help="특정 날짜 기준 스캔 (예: 2026-03-05)")
    args = parser.parse_args()

    cfg = load_settings()
    if not cfg.get("enabled", True):
        print("소형주 급등 스캐너 비활성 (settings.yaml)")
        return

    scan_date = args.backtest or datetime.now().strftime("%Y-%m-%d")
    print(f"[소형주 급등 포착 v2] {scan_date}")
    print("=" * 65)

    main_universe = load_main_universe()
    name_map = build_name_map()

    print(f"  CSV 전종목: {len(list(CSV_DIR.glob('*.csv')))}개")

    results = scan_all(cfg, main_universe, name_map,
                       target_date=args.backtest)

    # 등급별 집계
    grade_counts = {}
    for r in results:
        g = r["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1

    supply_counts = {}
    primary_list = [r for r in results if r["grade"] == GRADE_PRIMARY]
    for r in primary_list:
        sg = r["supply_grade"]
        supply_counts[sg] = supply_counts.get(sg, 0) + 1

    # 저장
    output = {
        "scan_date": scan_date,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_detected": len(results),
        "grade_summary": grade_counts,
        "primary_supply": supply_counts,
        "signals": results[:50],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {OUTPUT_PATH}")

    # ── 출력 ──
    if results:
        # PRIMARY 시그널
        print(f"\n{'═' * 75}")
        print(f"  PRIMARY 시그널 (C단독/A+C복합) — 백테스트 승률 50%+")
        print(f"{'═' * 75}")

        primary = [r for r in results if r["grade"] == GRADE_PRIMARY]
        if primary:
            print(f"  {'#':>2} {'종목':>10} {'종가':>8} {'변동':>7}"
                  f" {'패턴':>8} {'수급':>10} {'OBV추세':>8} {'Vol5/20':>8} {'거래대금':>8}")
            print(f"  {'─'*2} {'─'*10} {'─'*8} {'─'*7}"
                  f" {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
            for i, r in enumerate(primary[:20], 1):
                patterns = "+".join(p.split("_")[0] for p in r["pattern_names"])
                print(f"  {i:>2} {r['name'][:8]:>10} {r['close']:>8,} {r['change_pct']:>+6.1f}%"
                      f" {patterns:>8} {r['supply_grade']:>10}"
                      f" {r['obv_trend']:>+7.1f}% {r['vol_ratio_5_20']:>7.2f}x"
                      f" {r['trading_value_억']:>7.0f}억")
        else:
            print("  (없음)")

        # WARNING (A+B 데드캣)
        warnings = [r for r in results if r["grade"] == GRADE_WARNING]
        if warnings:
            print(f"\n{'─' * 75}")
            print(f"  WARNING: A+B 데드캣 바운스 경고 ({len(warnings)}건) — 승률 35%, 매수 금지")
            print(f"{'─' * 75}")
            for i, r in enumerate(warnings[:10], 1):
                print(f"  {i:>2} {r['name'][:8]:>10} {r['close']:>8,} {r['change_pct']:>+6.1f}%"
                      f" 거래대금:{r['trading_value_억']:.0f}억")

        # ALERT (B 고위험)
        alerts = [r for r in results if r["grade"] == GRADE_ALERT]
        if alerts:
            print(f"\n{'─' * 75}")
            print(f"  ALERT: B 단독 고위험 ({len(alerts)}건) — 승률 40%, 대박or폭락")
            print(f"{'─' * 75}")
            for i, r in enumerate(alerts[:10], 1):
                b = next((p for p in r["patterns"] if p["pattern"] == "B_SHAKEOUT_REVERSAL"), {})
                print(f"  {i:>2} {r['name'][:8]:>10} {r['close']:>8,}"
                      f" 급락{b.get('drop_pct',0):+.1f}%→반등{b.get('bounce_pct',0):+.1f}%"
                      f" 수급:{r['supply_grade']}")

        # WATCH (A 단독)
        watches = [r for r in results if r["grade"] == GRADE_WATCH]
        if watches:
            print(f"\n  WATCH: A 단독 ({len(watches)}건) — 승률 44%, 모니터링")

    else:
        print("\n  감지된 패턴 없음")

    # 요약
    print(f"\n  {'═' * 50}")
    print(f"  등급별: PRIMARY {grade_counts.get(GRADE_PRIMARY, 0)} │"
          f" WARNING {grade_counts.get(GRADE_WARNING, 0)} │"
          f" ALERT {grade_counts.get(GRADE_ALERT, 0)} │"
          f" WATCH {grade_counts.get(GRADE_WATCH, 0)}")
    if supply_counts:
        print(f"  PRIMARY 수급: CONFIRMED {supply_counts.get('CONFIRMED', 0)} │"
              f" PARTIAL {supply_counts.get('PARTIAL', 0)} │"
              f" NONE {supply_counts.get('NONE', 0)}")

    # 텔레그램
    if not args.no_send and results:
        send_telegram(results, cfg)


if __name__ == "__main__":
    main()
