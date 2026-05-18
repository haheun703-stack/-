"""5/18 일일 강세 종목 발굴기 — 사장님 EYE 확장 (2026-05-18 신규)

배경: 사장님 15:00 통찰 "급등주 + 음→양 전환 + 상한가 발굴 + 학습"
- 자비스 추천 9건 외에도 진짜 강세 종목 자동 발굴
- 음→양 전환 패턴 = 사장님 안전선 "10:30 이후 진입" 정확 보완

3 카테고리 자동 발굴:
  🚀 급등주        — 전일 대비 +5% 이상
  🔄 음→양 전환    — 시초가 음봉, 현재가 양봉 (반등 종목, 사장님 안전선 패턴)
  🔥 상한가        — 30% 이상 도달

학습 데이터:
  data/winners/winners_{YYYYMMDD}.json — 발굴 종목 + EYE 점수 + 패턴

Universe (현재 시점):
  - 자비스 화이트리스트 26개
  - 자비스 강력포착 9건
  - 시총 상위 100 (universe.csv 활용)

Usage:
  python scripts/daily_winners.py             # 1회 실행 + 텔레그램
  python scripts/daily_winners.py --no-tg     # 텔레그램 OFF
  python scripts/daily_winners.py --top 5     # 카테고리별 TOP 5만
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter  # noqa: E402
from src.use_cases.eye_filters import evaluate_filters  # noqa: E402

WINNERS_DIR = PROJECT_ROOT / "data" / "winners"
TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"

# 임계값
THRESHOLD_SURGE_PCT = 5.0    # 급등주 (전일 대비 +5%)
THRESHOLD_LIMIT_PCT = 28.0   # 상한가 근접 (KOSPI 상한 30%, 28% 이상)
THRESHOLD_REVERSAL_OPN = -1.5  # 음봉 시초가 (시가 대비 -1.5% 이하)
THRESHOLD_REVERSAL_CUR = 1.0   # 양봉 현재가 (전일 대비 +1% 이상)

logger = logging.getLogger(__name__)


def fetch_price_data(broker, ticker: str) -> dict | None:
    """KIS fetch_price → 분류 필요 필드 추출."""
    try:
        resp = broker.fetch_price(ticker)
        out = resp.get("output", {}) if resp else {}
        if not out:
            return None
        cur = int(out.get("stck_prpr", 0) or 0)
        opn = int(out.get("stck_oprc", 0) or 0)
        sdpr = int(out.get("stck_sdpr", 0) or 0)  # 전일 종가
        ctrt = float(out.get("prdy_ctrt", 0) or 0)  # 전일 대비 %
        return {
            "ticker": ticker,
            "current": cur,
            "open": opn,
            "prev_close": sdpr,
            "prdy_ctrt": ctrt,
            "intra_chg_pct": ((cur - opn) / opn * 100) if opn > 0 else 0,
            "open_chg_pct": ((opn - sdpr) / sdpr * 100) if sdpr > 0 else 0,
            "vol_ratio": float(out.get("prdy_vrss_vol_rate", 0) or 0),
            "pgtr_ntby": int(out.get("pgtr_ntby_qty", 0) or 0),
            "name_dummy": out.get("bstp_kor_isnm", ""),  # 업종
        }
    except Exception as e:
        logger.debug("fetch_price 실패 %s: %s", ticker, e)
        return None


def get_universe() -> list[tuple[str, str]]:
    """스캔 대상 종목 리스트.

    1. tomorrow_picks 강력포착 + 포착
    2. 화이트리스트 26 (대형주)
    3. 시총 상위 100 (universe.csv)
    """
    universe = {}

    # 1) tomorrow_picks
    if TOMORROW_PICKS.exists():
        data = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
        for p in data.get("picks", []):
            tk = p.get("ticker", "")
            nm = p.get("name", tk)
            if tk:
                universe[tk] = nm

    # 2) universe.csv (시총 상위)
    uni_csv = PROJECT_ROOT / "data" / "universe.csv"
    if uni_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(uni_csv, dtype=str)
            for _, row in df.head(150).iterrows():
                tk = row.get("code", row.get("ticker", ""))
                nm = row.get("name", tk)
                if tk and tk not in universe:
                    universe[str(tk).zfill(6)] = str(nm)
        except Exception as e:
            logger.warning("universe.csv 로드 실패: %s", e)

    # 3) 대형주 화이트리스트 (백업)
    big_caps = [
        ("005935", "삼성전자우"), ("005930", "삼성전자"), ("000660", "SK하이닉스"),
        ("005380", "현대차"), ("012330", "현대모비스"), ("051910", "LG화학"),
        ("066570", "LG전자"), ("028260", "삼성물산"), ("035420", "NAVER"),
        ("068270", "셀트리온"), ("207940", "삼성바이오로직스"), ("373220", "LG에너지솔루션"),
        ("069500", "KODEX 200"), ("122630", "KODEX 레버리지"),
    ]
    for tk, nm in big_caps:
        if tk not in universe:
            universe[tk] = nm

    return [(tk, nm) for tk, nm in universe.items()]


def classify_winners(broker, universe: list[tuple[str, str]], top_n: int = 10) -> dict:
    """전체 universe 스캔 후 3 카테고리 분류."""
    surges = []      # 🚀 급등주 (전일 대비 +5%)
    reversals = []   # 🔄 음→양 전환
    limits = []      # 🔥 상한가 근접

    print(f"스캔 중... ({len(universe)} 종목)")
    for i, (tk, nm) in enumerate(universe):
        if i % 20 == 0:
            print(f"  진행 {i}/{len(universe)}")
        px = fetch_price_data(broker, tk)
        if not px:
            continue
        px["name"] = nm

        ctrt = px["prdy_ctrt"]
        opn_chg = px["open_chg_pct"]
        intra_chg = px["intra_chg_pct"]

        # 상한가 (전일 대비 +28% 이상)
        if ctrt >= THRESHOLD_LIMIT_PCT:
            limits.append(px)
        # 급등 (전일 대비 +5% 이상)
        elif ctrt >= THRESHOLD_SURGE_PCT:
            surges.append(px)

        # 음→양 전환 (시초가 -1.5% 이하 + 현재가 +1% 이상)
        if opn_chg <= THRESHOLD_REVERSAL_OPN and ctrt >= THRESHOLD_REVERSAL_CUR:
            reversals.append({**px, "reversal_pp": ctrt - opn_chg})

    # 정렬 + TOP N
    surges.sort(key=lambda x: x["prdy_ctrt"], reverse=True)
    reversals.sort(key=lambda x: x.get("reversal_pp", 0), reverse=True)
    limits.sort(key=lambda x: x["prdy_ctrt"], reverse=True)

    return {
        "surges": surges[:top_n],
        "reversals": reversals[:top_n],
        "limits": limits[:top_n],
    }


def apply_eye_filters(broker, winners: dict, date: str) -> dict:
    """발굴 종목에 EYE 필터 적용."""
    for category in ("surges", "reversals", "limits"):
        for item in winners[category]:
            eye = evaluate_filters(broker, item["ticker"], date)
            item["eye_skip"] = eye["should_skip"]
            item["eye_reasons"] = eye["skip_reasons"]
    return winners


def format_telegram(winners: dict, now: datetime) -> str:
    lines = [f"🎯 일일 강세 종목 발굴 ({now.strftime('%H:%M')} KST)"]
    lines.append("━" * 12)

    if winners["limits"]:
        lines.append("🔥 상한가 근접 (+28%↑)")
        for w in winners["limits"][:5]:
            flag = "🔴EYE_SKIP" if w["eye_skip"] else "🟢"
            lines.append(f"  {flag} {w['name']} {w['prdy_ctrt']:+.2f}%")
        lines.append("")

    if winners["surges"]:
        lines.append("🚀 급등주 (+5%↑)")
        for w in winners["surges"][:8]:
            flag = "🔴" if w["eye_skip"] else "🟢"
            lines.append(f"  {flag} {w['name']} {w['prdy_ctrt']:+.2f}%")
        lines.append("")

    if winners["reversals"]:
        lines.append("🔄 음→양 전환 (사장님 안전선 패턴)")
        for w in winners["reversals"][:8]:
            flag = "🔴" if w["eye_skip"] else "🟢"
            rev = w.get("reversal_pp", 0)
            lines.append(f"  {flag} {w['name']} 시초{w['open_chg_pct']:+.1f}% → 현재{w['prdy_ctrt']:+.1f}% (반등 {rev:+.1f}%p)")
        lines.append("")

    if not (winners["limits"] or winners["surges"] or winners["reversals"]):
        lines.append("(오늘 발굴 없음)")

    lines.append("━" * 12)
    lines.append("★ EYE 통과 (🟢) 종목 = 5/20 출격 추가 후보")
    return "\n".join(lines)


def save_winners(winners: dict, date: str) -> Path:
    WINNERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WINNERS_DIR / f"winners_{date.replace('-', '')}.json"
    out_path.write_text(
        json.dumps({
            "date": date,
            "n_surges": len(winners["surges"]),
            "n_reversals": len(winners["reversals"]),
            "n_limits": len(winners["limits"]),
            "winners": winners,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--no-tg", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    adp = KisStockDataAdapter()
    broker = adp.broker

    universe = get_universe()
    print(f"Universe: {len(universe)} 종목")

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    winners = classify_winners(broker, universe, top_n=args.top)
    winners = apply_eye_filters(broker, winners, today)

    out_path = save_winners(winners, today)
    print(f"[SAVED] {out_path.name}")

    msg = format_telegram(winners, now)
    print(msg)

    if not args.no_tg:
        try:
            from src.telegram_sender import send_message

            send_message(msg)
            print("\n[TG SENT]")
        except Exception as e:
            print(f"\n[TG FAIL] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
