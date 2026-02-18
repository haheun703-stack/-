"""
장시작전 마켓 브리핑 텔레그램 발송 스크립트.

데이터 소스:
  - data/us_market/overnight_signal.json (US overnight signal → Phase 1에서 생성)
  - results/signals_log.csv (전일 스캔 결과 → Phase 10에서 생성)
  - stock_data_daily/*.csv (종목명 매핑)

수동 실행: python scripts/send_market_briefing.py [--send]
스케줄러: daily_scheduler.phase_morning_briefing()에서 호출
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

SIGNAL_PATH = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
SIGNALS_LOG = PROJECT_ROOT / "results" / "signals_log.csv"
STOCK_DATA_DIR = PROJECT_ROOT / "stock_data_daily"


# ─────────────────────────────────────────────
# 유틸: 종목명 조회
# ─────────────────────────────────────────────

def _get_stock_name(ticker: str) -> str:
    """ticker(6자리) → 종목명. CSV 파일명에서 추출."""
    ticker = str(ticker).zfill(6)
    if STOCK_DATA_DIR.exists():
        for csv in STOCK_DATA_DIR.glob(f"*_{ticker}.csv"):
            # 파일명 형식: "종목명_006400.csv"
            stem = csv.stem  # "종목명_006400"
            name = stem.rsplit("_", 1)[0]
            return name
    return ticker


# ─────────────────────────────────────────────
# 확률 계산
# ─────────────────────────────────────────────

def calc_market_probability(
    ewy_ret: float,
    vix: float,
    spy_ret: float,
    qqq_ret: float,
    regime: str,
) -> dict:
    """해외시장 데이터 기반 금일 KOSPI 상승/하락 확률."""
    down_prob = 50.0

    # EWY (40% 가중) — KOSPI 가장 강한 선행지표
    if ewy_ret < -1:
        down_prob += min(abs(ewy_ret) * 8, 25)
    elif ewy_ret < 0:
        down_prob += abs(ewy_ret) * 5
    elif ewy_ret > 1:
        down_prob -= min(ewy_ret * 8, 25)
    elif ewy_ret > 0:
        down_prob -= ewy_ret * 5

    # VIX (20% 가중)
    if vix > 25:
        down_prob += 8
    elif vix > 20:
        down_prob += 5
    elif vix < 15:
        down_prob -= 5

    # US 종합 (15% 가중)
    us_avg = (spy_ret + qqq_ret) / 2
    if us_avg < -0.5:
        down_prob += 5
    elif us_avg > 0.5:
        down_prob -= 5

    # KOSPI 레짐 (15% 가중)
    regime_adj = {
        "BULL": -5, "STRONG_BULL": -8, "MILD_BULL": -3,
        "CAUTION": 3, "NEUTRAL": 0,
        "BEAR": 8, "MILD_BEAR": 5, "STRONG_BEAR": 12,
        "CRISIS": 12,
    }.get(regime.upper(), 0)
    down_prob += regime_adj

    # 클램핑
    down_prob = max(5, min(95, down_prob))
    up_prob = 100 - down_prob

    # 예상 레인지 (EWY 기반)
    if ewy_ret != 0:
        est_low = round(ewy_ret * 1.2, 1)
        est_high = round(ewy_ret * 0.6, 1)
    else:
        est_low, est_high = -0.5, 0.5

    return {
        "up_prob": round(up_prob),
        "down_prob": round(down_prob),
        "est_low": min(est_low, est_high),
        "est_high": max(est_low, est_high),
    }


def make_prob_bar(down_pct: int, width: int = 20) -> str:
    """확률 시각화 바."""
    filled = round(down_pct / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_overnight_signal() -> dict:
    """US overnight signal JSON 로드. 없으면 빈 dict."""
    if SIGNAL_PATH.exists():
        with open(SIGNAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    logger.warning("overnight_signal.json 없음 — 기본값 사용")
    return {}


def load_scan_results() -> list[dict]:
    """최신 스캔 결과 로드 (signals_log.csv)."""
    if not SIGNALS_LOG.exists():
        return []
    df = pd.read_csv(SIGNALS_LOG)
    if df.empty:
        return []
    if "date" in df.columns:
        df = df[df["date"] == df["date"].max()]
    # zone_score 내림차순 정렬
    if "zone_score" in df.columns:
        df = df.sort_values("zone_score", ascending=False)
    results = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).zfill(6)
        entry = row.get("entry_price", 0)
        rr = row.get("rr_ratio", 0)
        results.append({
            "ticker": ticker,
            "name": _get_stock_name(ticker),
            "zone_score": row.get("zone_score", 0),
            "grade": row.get("grade", "?"),
            "trigger_type": row.get("trigger_type", ""),
            "entry_price": entry,
            "rr_ratio": rr,
            "regime": row.get("regime", ""),
        })
    return results


# ─────────────────────────────────────────────
# 메시지 빌더
# ─────────────────────────────────────────────

def build_briefing_message() -> str:
    """오늘의 장전 마켓 브리핑 메시지 자동 생성."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === 데이터 로드 ===
    sig = load_overnight_signal()
    idx = sig.get("index_direction", {})

    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    dia_ret = idx.get("DIA", {}).get("ret_1d", 0)
    vix_data = sig.get("vix", {})
    vix = vix_data.get("level", 20)
    vix_status = vix_data.get("status", "정상")
    grade = sig.get("grade", "NEUTRAL")
    combined = sig.get("combined_score_100", 0)

    # SOX: l2_pattern → today_us_vector에서 추출
    l2 = sig.get("l2_pattern", {})
    us_vec = l2.get("today_us_vector", {})
    sox_ret = us_vec.get("us_soxx_chg", 0)

    # 레짐 매핑
    regime_map = {
        "STRONG_BULL": "BULL", "MILD_BULL": "BULL",
        "STRONG_BEAR": "BEAR", "MILD_BEAR": "BEAR",
    }
    regime = regime_map.get(grade, grade)

    prob = calc_market_probability(ewy_ret, vix, spy_ret, qqq_ret, regime)
    bar = make_prob_bar(prob["down_prob"])

    # 스캔 결과
    candidates = load_scan_results()

    lines = []

    # ── Header ──
    lines.append(f"\U0001f3c6 Quantum Master v10.3 | {now}")
    lines.append("\u2501" * 28)
    lines.append("")

    # ── 상승/하락 확률 ──
    lines.append("\U0001f3b2 금일 KOSPI 예측")
    if prob["down_prob"] >= 55:
        dir_emoji = "\U0001f4c9"
        dir_label = "하락"
    elif prob["up_prob"] >= 55:
        dir_emoji = "\U0001f4c8"
        dir_label = "상승"
    else:
        dir_emoji = "\u27a1\ufe0f"
        dir_label = "보합"

    lines.append(
        f"  {dir_emoji} {dir_label} {prob['down_prob']}% "
        f"\u2502 \U0001f4c8 상승 {prob['up_prob']}%"
    )
    lines.append(f"  {bar}")
    lines.append(f"  예상 레인지: {prob['est_low']:+.1f}% ~ {prob['est_high']:+.1f}%")
    lines.append("")

    # ── 근거 ──
    lines.append("\U0001f4cc 근거")
    # EWY
    ewy_emoji = "\U0001f53b" if ewy_ret < 0 else "\U0001f53a"
    ewy_comment = "강한 하방" if ewy_ret < -1.5 else ("약세" if ewy_ret < 0 else "양호")
    lines.append(f"  \u251c EWY {ewy_ret:+.2f}% {ewy_emoji} \u2192 {ewy_comment}")
    # VIX
    vix_emoji = "\u26a0\ufe0f" if vix > 20 else "\u2705"
    lines.append(f"  \u251c VIX {vix:.1f} [{vix_emoji}{vix_status}]")
    # SOX
    if sox_ret != 0:
        sox_comment = "반도체 약세 동조" if sox_ret < -0.5 else (
            "반도체 강세" if sox_ret > 0.5 else "반도체 중립"
        )
        lines.append(f"  \u251c SOX {sox_ret:+.1f}% \u2192 {sox_comment}")
    # US 종합
    us_avg = (spy_ret + qqq_ret) / 2
    us_label = "글로벌 약세" if us_avg < -0.3 else ("글로벌 강세" if us_avg > 0.3 else "글로벌 중립")
    lines.append(f"  \u251c S&P {spy_ret:+.1f}% QQQ {qqq_ret:+.1f}% \u2192 {us_label}")
    # Grade
    grade_emoji = {
        "STRONG_BULL": "\U0001f7e2", "MILD_BULL": "\U0001f7e2",
        "NEUTRAL": "\U0001f7e1", "MILD_BEAR": "\U0001f7e0",
        "STRONG_BEAR": "\U0001f534",
    }.get(grade, "\U0001f7e1")
    lines.append(f"  \u2514 US Signal {grade_emoji} {grade} ({combined:+.1f})")
    lines.append("")

    # ── 전략 한줄 ──
    if prob["down_prob"] >= 60:
        lines.append("\U0001f4a1 전략: 갭다운 예상 — 우량주 저가매수 기회!")
    elif prob["up_prob"] >= 60:
        lines.append("\U0001f4a1 전략: 갭업 예상 — 목표가 부근 분할매도!")
    else:
        lines.append("\U0001f4a1 전략: 혼조 예상 — 시가 확인 후 대응!")
    lines.append("")

    # ── 해외시장 요약 ──
    us_date = sig.get("us_close_date", "")
    lines.append(f"\U0001f30d 해외시장 ({us_date})")
    lines.append(
        f"  \u251c \U0001f1fa\U0001f1f8 SPY {spy_ret:+.1f}% | "
        f"QQQ {qqq_ret:+.1f}% | DIA {dia_ret:+.1f}%"
    )
    lines.append(f"  \u251c \U0001f1f0\U0001f1f7 EWY {ewy_ret:+.2f}%")
    lines.append(f"  \u2514 \U0001f4ca VIX {vix:.1f} [{vix_emoji}{vix_status}]")
    lines.append("")

    # ── 섹터 Kill 경고 ──
    kills = sig.get("sector_kills", {})
    killed_sectors = [s for s, v in kills.items() if v.get("killed")]
    if killed_sectors:
        lines.append("\U0001f6a8 섹터 Kill 경고")
        for s in killed_sectors:
            info = kills[s]
            lines.append(
                f"  \u274c {s}: {info['driver_ret']:+.1f}% "
                f"(임계 {info['threshold_pct']:+.1f}%)"
            )
        lines.append("")

    # ── 패턴 매칭 인사이트 ──
    l2_kospi = l2.get("kospi", {})
    if l2_kospi:
        pos_rate = l2_kospi.get("positive_rate", 50)
        mean_chg = l2_kospi.get("mean_chg", 0)
        sample = l2.get("sample_count", 0)
        if sample > 30:
            lines.append(f"\U0001f50d 패턴매칭 (유사 {sample}건)")
            lines.append(
                f"  \u251c KOSPI 상승률: {pos_rate:.0f}% (평균 {mean_chg:+.2f}%)"
            )
            # 섹터별 탑/바텀
            sec_l2 = l2.get("sectors", {})
            if sec_l2:
                sorted_sec = sorted(sec_l2.items(), key=lambda x: x[1].get("mean_chg", 0), reverse=True)
                top = sorted_sec[0]
                bot = sorted_sec[-1]
                lines.append(
                    f"  \u251c \U0001f53a 강세섹터: {top[0]} "
                    f"(상승 {top[1]['positive_rate']:.0f}%)"
                )
                lines.append(
                    f"  \u2514 \U0001f53b 약세섹터: {bot[0]} "
                    f"(상승 {bot[1]['positive_rate']:.0f}%)"
                )
            lines.append("")

    # ── 매수 후보 (S/A/B/C) ──
    if candidates:
        lines.append("\u2501" * 28)
        lines.append("\U0001f525 매수 후보")
        lines.append("\u2501" * 28)

        grade_labels = {0: ("S", "\U0001f525"), 1: ("A", "\u2b50"), 2: ("B", "\U0001f539"), 3: ("C", "\u26d4")}

        for i, c in enumerate(candidates[:5]):
            g_label, g_emoji = grade_labels.get(i, ("D", "\u2796"))
            name = c["name"]
            ticker = c["ticker"]
            entry = c["entry_price"]
            rr = c["rr_ratio"]
            zone = c["zone_score"]
            trigger = c["trigger_type"]

            lines.append(f"{g_emoji} {g_label}등급 — {name} ({ticker})")
            lines.append(
                f"  \U0001f4b0 진입: {entry:,.0f}원 | "
                f"R:R {rr:.1f}배 | Zone {zone:.2f}"
            )
            lines.append(f"  \u26a1 트리거: {trigger} | 등급 {c['grade']}")
            lines.append("")
    else:
        lines.append("")
        lines.append("\U0001f525 매수 후보: 스캔 결과 없음")
        lines.append("")

    # ── Disclaimer ──
    lines.append("\u26a0\ufe0f 투자 판단은 본인 책임 | Quantum Master v10.3")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    msg = build_briefing_message()

    print("=" * 50)
    print("[미리보기]")
    print("=" * 50)
    print(msg)
    print("=" * 50)
    print(f"총 {len(msg)}자 (텔레그램 제한: 4096자)")
    print()

    if "--send" in sys.argv:
        from src.telegram_sender import send_message
        ok = send_message(msg)
        if ok:
            print("\u2705 텔레그램 발송 완료!")
        else:
            print("\u274c 텔레그램 발송 실패")
    else:
        print("실제 발송: python scripts/send_market_briefing.py --send")
