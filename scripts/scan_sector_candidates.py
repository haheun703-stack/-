"""섹터 순환매 통합 스캐너 — 테마머니 vs 스마트머니 구분.

섹터 모멘텀 + z-score + 수급 데이터를 결합하여
두 가지 카테고리로 종목을 분류한다:

  [테마머니] 개인+기관 주도, 외인 매도 → 단타/순환매 (하프사이즈)
  [스마트머니] 외인+기관 동시 매수 → 포물선 시작점 (풀사이즈)

추가 필터:
  - processed parquet에서 Stoch Slow 골든크로스 확인
  - v10.3 Gate (ADX, 풀백, 오버히트) 통과 여부 확인

사용법:
  python scripts/scan_sector_candidates.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_json(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stock_indicators(ticker: str) -> pd.DataFrame | None:
    """종목 processed parquet에서 최근 지표 로드."""
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────
# 종목 필터: 기술적 지표 체크
# ─────────────────────────────────────────────

def check_stock_technicals(ticker: str) -> dict:
    """종목의 기술적 지표를 체크.

    Returns:
        dict with stoch_golden, rsi, adx, pullback, overheat, gate_pass
    """
    df = load_stock_indicators(ticker)
    if df is None or len(df) < 5:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    result = {
        "close": float(last.get("close", 0)),
        "rsi": float(last.get("rsi", 50)),
        "adx": float(last.get("adx", 0)),
    }

    # Stoch Slow 골든크로스
    stoch_k = last.get("stoch_slow_k")
    stoch_d = last.get("stoch_slow_d")
    prev_k = prev.get("stoch_slow_k")
    prev_d = prev.get("stoch_slow_d")

    if all(v is not None and not np.isnan(v) for v in [stoch_k, stoch_d, prev_k, prev_d]):
        result["stoch_k"] = round(float(stoch_k), 1)
        result["stoch_d"] = round(float(stoch_d), 1)
        result["stoch_golden"] = bool(stoch_k > stoch_d and prev_k <= prev_d)
        # 최근 5일 내 골든크로스
        recent_golden = False
        for i in range(-5, 0):
            if i + 1 >= 0:
                continue
            row = df.iloc[i]
            row_next = df.iloc[i + 1]
            sk = row.get("stoch_slow_k")
            sd = row.get("stoch_slow_d")
            sk_n = row_next.get("stoch_slow_k")
            sd_n = row_next.get("stoch_slow_d")
            if all(v is not None and not np.isnan(v) for v in [sk, sd, sk_n, sd_n]):
                if sk_n > sd_n and sk <= sd:
                    recent_golden = True
                    break
        result["stoch_golden_recent"] = recent_golden
    else:
        result["stoch_golden"] = False
        result["stoch_golden_recent"] = False

    # v10.3 Gate 체크 (간이 판정)
    adx = float(last.get("adx", 0))
    rsi_val = float(last.get("rsi", 50))
    ma20 = float(last.get("ma20", 0))
    close_val = float(last.get("close", 0))

    gate_adx = adx >= 18
    gate_pullback = close_val <= ma20 * 1.0 if ma20 > 0 else True  # MA20 근처
    gate_overheat = rsi_val < 80

    result["gate_adx"] = gate_adx
    result["gate_pullback"] = gate_pullback
    result["gate_overheat"] = gate_overheat
    result["gate_pass"] = gate_adx and gate_overheat  # 풀백은 참고만

    return result


# ─────────────────────────────────────────────
# 메인 스캔
# ─────────────────────────────────────────────

def scan_candidates() -> tuple[list[dict], list[dict]]:
    """테마머니 vs 스마트머니 후보를 분류하여 반환."""

    momentum = load_json("sector_momentum.json")
    zscore = load_json("sector_zscore.json")
    flow = load_json("investor_flow.json")

    if not momentum or not zscore or not flow:
        logger.error("데이터 부족 — sector_daily_report.py 먼저 실행")
        return [], []

    # 수급 데이터를 섹터명 키로
    flow_map = {}
    for s in flow.get("sectors", []):
        flow_map[s["sector"]] = s

    # 모멘텀 순위 맵
    mom_map = {}
    for s in momentum.get("sectors", []):
        mom_map[s["sector"]] = s

    theme_money = []   # 테마머니: 모멘텀 강 + 외인 매도
    smart_money = []   # 스마트머니: 외인+기관 매수

    # 각 섹터별 z-score 후보 순회
    for sector_name, stocks in zscore.get("sectors", {}).items():
        mom = mom_map.get(sector_name, {})
        fl = flow_map.get(sector_name, {})

        mom_rank = mom.get("rank", 99)
        mom_score = mom.get("momentum_score", 0)
        ret_20 = mom.get("ret_20", 0)
        rsi_sector = mom.get("rsi_14", 50)

        foreign_cum = fl.get("foreign_cum_bil", 0)
        inst_cum = fl.get("inst_cum_bil", 0)

        # 수급 분류
        is_smart = foreign_cum > 0 and inst_cum > 0
        is_theme = foreign_cum < -500 and (inst_cum > 0 or mom_rank <= 5)
        is_foreign_buy = foreign_cum > 0

        for stock in stocks:
            z_20 = stock.get("z_20", 0)

            # 래깅 종목만 (z < -0.5, 조금 완화)
            if z_20 > -0.5:
                continue

            ticker = stock.get("ticker", "")
            name = stock.get("name", "")
            stock_ret_20 = stock.get("stock_ret_20", 0)
            etf_ret_20 = stock.get("etf_ret_20", 0)

            # 기술적 지표 체크
            tech = check_stock_technicals(ticker)
            if not tech:
                continue

            candidate = {
                "sector": sector_name,
                "ticker": ticker,
                "name": name,
                "z_20": z_20,
                "stock_ret_20": stock_ret_20,
                "etf_ret_20": etf_ret_20,
                "mom_rank": mom_rank,
                "mom_score": mom_score,
                "sector_ret_20": ret_20,
                "foreign_cum": foreign_cum,
                "inst_cum": inst_cum,
                "rsi": tech.get("rsi", 50),
                "adx": tech.get("adx", 0),
                "stoch_k": tech.get("stoch_k", 0),
                "stoch_d": tech.get("stoch_d", 0),
                "stoch_golden": tech.get("stoch_golden", False),
                "stoch_golden_recent": tech.get("stoch_golden_recent", False),
                "gate_pass": tech.get("gate_pass", False),
                "gate_overheat": tech.get("gate_overheat", True),
            }

            # 분류
            if is_smart or is_foreign_buy:
                # 스마트머니: 외인+기관 동시 or 외인 매수
                candidate["money_type"] = "SMART"
                candidate["sizing"] = "FULL" if is_smart else "HALF"
                smart_money.append(candidate)
            elif is_theme:
                # 테마머니: 외인 매도 + 기관/개인 주도
                candidate["money_type"] = "THEME"
                candidate["sizing"] = "HALF"
                theme_money.append(candidate)
            elif mom_rank <= 7 and z_20 <= -0.8:
                # 모멘텀 상위이지만 수급 중립 → 테마 쪽
                candidate["money_type"] = "THEME"
                candidate["sizing"] = "HALF"
                theme_money.append(candidate)

    # 정렬
    smart_money.sort(key=lambda x: (x["z_20"]))
    theme_money.sort(key=lambda x: (x["mom_rank"], x["z_20"]))

    return theme_money, smart_money


# ─────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────

def print_candidates(theme_money: list, smart_money: list):
    """분류 결과 출력."""

    date_str = ""
    momentum = load_json("sector_momentum.json")
    if momentum:
        date_str = momentum.get("date", "")

    print(f"\n{'━' * 65}")
    print(f"  섹터 순환매 통합 스캔 — {date_str}")
    print(f"  테마머니 vs 스마트머니 분류")
    print(f"{'━' * 65}")

    # ── 스마트머니 ──
    print(f"\n{'=' * 65}")
    print(f"  ◆ 스마트머니 (외인+기관 매수, 포물선 시작점)")
    print(f"  → 사이즈: FULL or HALF | 전략: 중기 보유")
    print(f"{'=' * 65}")

    if smart_money:
        print(f"  {'섹터':<8} {'종목':<10} {'z_20':>5} {'종목%':>7} {'섹터%':>7} {'외인':>7} {'기관':>7} {'Stoch':>6} {'사이즈'}")
        print(f"  {'─' * 62}")

        for c in smart_money[:15]:
            stoch_str = ""
            if c["stoch_golden"]:
                stoch_str = "GX★"
            elif c["stoch_golden_recent"]:
                stoch_str = "GX(r)"
            else:
                stoch_str = f"{c['stoch_k']:.0f}"

            gate_str = "✓" if c["gate_pass"] else "✗"
            oh_str = "" if c["gate_overheat"] else " OH!"

            print(
                f"  {c['sector']:<8} {c['name']:<10} {c['z_20']:>+5.2f} "
                f"{c['stock_ret_20']:>+7.1f} {c['sector_ret_20']:>+7.1f} "
                f"{c['foreign_cum']:>+7.0f} {c['inst_cum']:>+7.0f} "
                f"{stoch_str:>6} {c['sizing']}{oh_str}"
            )
    else:
        print("  후보 없음")

    # ── 테마머니 ──
    print(f"\n{'=' * 65}")
    print(f"  ★ 테마머니 (개인+기관 주도, 외인 매도)")
    print(f"  → 사이즈: HALF | 전략: 단타/순환매 (빠른 익절)")
    print(f"{'=' * 65}")

    if theme_money:
        print(f"  {'섹터':<8} {'종목':<10} {'z_20':>5} {'종목%':>7} {'섹터%':>7} {'외인':>7} {'기관':>7} {'Stoch':>6} {'사이즈'}")
        print(f"  {'─' * 62}")

        for c in theme_money[:15]:
            stoch_str = ""
            if c["stoch_golden"]:
                stoch_str = "GX★"
            elif c["stoch_golden_recent"]:
                stoch_str = "GX(r)"
            else:
                stoch_str = f"{c['stoch_k']:.0f}"

            oh_str = "" if c["gate_overheat"] else " OH!"

            print(
                f"  {c['sector']:<8} {c['name']:<10} {c['z_20']:>+5.2f} "
                f"{c['stock_ret_20']:>+7.1f} {c['sector_ret_20']:>+7.1f} "
                f"{c['foreign_cum']:>+7.0f} {c['inst_cum']:>+7.0f} "
                f"{stoch_str:>6} {c['sizing']}{oh_str}"
            )
    else:
        print("  후보 없음")

    # ── 종합 매수 리스트 ──
    print(f"\n{'━' * 65}")
    print(f"  종합 매수 후보 (내일 2/20)")
    print(f"{'━' * 65}")

    # 스마트머니 FULL + 기술적 신호
    top_smart = [c for c in smart_money if c["sizing"] == "FULL" and c["gate_pass"]]
    top_smart_stoch = [c for c in smart_money if c["sizing"] == "FULL" and (c["stoch_golden"] or c["stoch_golden_recent"])]

    # 테마머니 + Stoch 골든크로스
    top_theme = [c for c in theme_money if c["stoch_golden"] or c["stoch_golden_recent"]]

    if top_smart:
        print(f"\n  ◆ 1순위 (스마트머니 FULL + Gate 통과):")
        for c in top_smart[:5]:
            print(f"    {c['sector']}/{c['name']} z={c['z_20']:+.2f} 외인{c['foreign_cum']:+.0f}억 기관{c['inst_cum']:+.0f}억")

    if top_smart_stoch:
        print(f"\n  ◆ 2순위 (스마트머니 FULL + Stoch 골든크로스):")
        for c in top_smart_stoch[:5]:
            print(f"    {c['sector']}/{c['name']} z={c['z_20']:+.2f} Stoch {c['stoch_k']:.0f}/{c['stoch_d']:.0f}")

    remaining_smart = [c for c in smart_money if c not in top_smart and c not in top_smart_stoch]
    if remaining_smart:
        print(f"\n  ◇ 3순위 (스마트머니 관찰):")
        for c in remaining_smart[:5]:
            print(f"    {c['sector']}/{c['name']} z={c['z_20']:+.2f} [{c['sizing']}]")

    if top_theme:
        print(f"\n  ★ 4순위 (테마머니 + Stoch 골든크로스):")
        for c in top_theme[:5]:
            print(f"    {c['sector']}/{c['name']} z={c['z_20']:+.2f} 단타전략 [HALF]")

    remaining_theme = [c for c in theme_money if c not in top_theme]
    if remaining_theme:
        print(f"\n  ☆ 5순위 (테마머니 관찰):")
        for c in remaining_theme[:5]:
            oh = " ⚠OH" if not c["gate_overheat"] else ""
            print(f"    {c['sector']}/{c['name']} z={c['z_20']:+.2f} [HALF]{oh}")

    if not top_smart and not top_smart_stoch and not top_theme and not remaining_smart and not remaining_theme:
        print("  매수 후보 없음")

    # 저장
    all_candidates = {
        "date": date_str,
        "smart_money": smart_money,
        "theme_money": theme_money,
    }
    out_path = DATA_DIR / "scan_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2, default=str)
    logger.info("스캔 결과 → %s", out_path)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    theme, smart = scan_candidates()
    print_candidates(theme, smart)

    print(f"\n  테마머니: {len(theme)}종목 | 스마트머니: {len(smart)}종목")


if __name__ == "__main__":
    main()
