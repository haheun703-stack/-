"""
NXT(넥스트레이드) 추천 엔진 — BAT-L 3단계

tomorrow_picks(본장 추천) × NXT 애프터마켓 수급 교차 분석 → NXT 매매 추천.

플로우:
  1. tomorrow_picks.json에서 적극매수/매수/관심매수 종목 추출
  2. nxt_after_{date}.json 애프터마켓 수급 데이터 교차
  3. NXT 프리미엄/순매수/거래량 스코어링
  4. 최종 NXT 추천 → nxt_picks.json 저장
  5. 텔레그램 알림 발송

사용: python -u -X utf8 scripts/nxt_recommend.py [--date 2026-03-20]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# PYTHONPATH 안전장치
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
NXT_DIR = DATA_DIR / "nxt"
PICKS_PATH = NXT_DIR / "nxt_picks.json"

# 등급별 기본 점수
GRADE_SCORE = {
    "강력 포착": 100, "적극매수": 100,
    "포착": 70, "매수": 70,
    "관심": 40, "관심매수": 40,
}

# NXT 추천 임계값
MIN_SCORE = 50           # 최소 추천 점수
MIN_VOLUME = 300          # 최소 NXT 거래량
MAX_PICKS = 10            # 최대 추천 수


def _build_name_map() -> dict[str, str]:
    """종목코드 → 종목명 매핑 (CSV 파일명 기반)."""
    csv_dir = DATA_DIR / "csv"
    name_map = {}
    if csv_dir.exists():
        for csv in csv_dir.glob("*.csv"):
            parts = csv.stem.rsplit("_", 1)
            if len(parts) == 2:
                name_map[parts[1]] = parts[0]
    return name_map


def _load_tomorrow_picks() -> dict:
    """tomorrow_picks.json 로드 → {ticker: grade} 매핑."""
    path = DATA_DIR / "tomorrow_picks.json"
    if not path.exists():
        logger.warning("tomorrow_picks.json 없음")
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))

    # picks 배열에서 등급별 종목 추출
    picks_by_grade = {}

    # picks 배열 (전체 종목 + 등급 + 이름 포함)
    picks_list = data.get("picks", data.get("candidates", []))
    if picks_list:
        for c in picks_list:
            ticker = c.get("ticker", "")
            grade = c.get("grade", "")
            if ticker and grade in GRADE_SCORE:
                picks_by_grade[ticker] = {
                    "grade": grade,
                    "score": c.get("total_score", 0),
                    "name": c.get("name", ""),
                }

    # ai_largecap도 포함
    for ai in data.get("ai_largecap", []):
        ticker = ai.get("ticker", "")
        if ticker and ticker not in picks_by_grade:
            picks_by_grade[ticker] = {
                "grade": "관심매수",
                "score": 50,
                "name": ai.get("name", ""),
            }

    logger.info("tomorrow_picks 로드: %d종목", len(picks_by_grade))
    return picks_by_grade


def _load_aftermarket_data(target_date: str) -> dict:
    """NXT 애프터마켓 데이터 로드 → {ticker: summary_info}."""
    after_path = NXT_DIR / f"nxt_after_{target_date}.json"
    if not after_path.exists():
        logger.warning("애프터마켓 데이터 없음: %s", after_path.name)
        return {}

    data = json.loads(after_path.read_text(encoding="utf-8"))

    # summary 먼저 확인
    summary = data.get("summary", {})
    if summary:
        return summary

    # summary 없으면 마지막 스냅샷에서 추출
    snapshots = data.get("snapshots", [])
    if not snapshots:
        return {}

    last_snap = snapshots[-1]
    tickers = last_snap.get("tickers", {})
    result = {}
    for ticker, info in tickers.items():
        prev_close = info.get("prev_close", 0)
        price = info.get("price", 0)
        if prev_close > 0 and price > 0:
            result[ticker] = {
                "last_price": price,
                "prev_close": prev_close,
                "volume": info.get("volume", 0),
                "gap_pct": round((price / prev_close - 1) * 100, 2),
                "session_change_pct": 0,
                "net_buy_ratio": (
                    round(
                        info.get("total_bid_volume", 0)
                        / (info.get("total_bid_volume", 0) + info.get("total_ask_volume", 1)),
                        3,
                    )
                    if info.get("total_bid_volume") is not None
                    else 0.5
                ),
            }
    return result


def _load_nxt_signal() -> dict:
    """기존 NXT 시그널 로드."""
    signal_path = NXT_DIR / "nxt_signal.json"
    if not signal_path.exists():
        return {}
    data = json.loads(signal_path.read_text(encoding="utf-8"))
    # aftermarket_picks를 ticker 키 dict로 변환
    picks_map = {}
    for p in data.get("aftermarket_picks", []):
        picks_map[p["ticker"]] = p
    return picks_map


def _load_learning_weights() -> dict:
    """학습 가중치 로드 (nxt_track_results.py가 생성)."""
    weights_path = NXT_DIR / "nxt_learning_weights.json"
    if not weights_path.exists():
        return {}
    try:
        return json.loads(weights_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate_nxt_picks(target_date: str | None = None) -> dict:
    """
    NXT 추천 엔진 메인 로직.

    tomorrow_picks × 애프터마켓 수급 → 교차 추천.
    학습 가중치(nxt_learning_weights.json)가 있으면 스코어 보정.
    """
    today = target_date or date.today().isoformat()
    name_map = _build_name_map()
    learning = _load_learning_weights()

    # 1. tomorrow_picks 로드 (본장 추천)
    main_picks = _load_tomorrow_picks()

    # 2. 애프터마켓 데이터 로드
    after_data = _load_aftermarket_data(today)

    # 3. 기존 NXT 시그널 로드
    nxt_signals = _load_nxt_signal()

    if not main_picks:
        logger.warning("본장 추천 종목 없음 → NXT 추천 불가")
        return _empty_result(today)

    if not after_data:
        logger.warning("애프터마켓 데이터 없음 → 본장 추천만으로 NXT 후보 생성")

    # 4. 교차 분석 + 스코어링
    scored_picks = []

    for ticker, pick_info in main_picks.items():
        grade = pick_info["grade"]
        base_score = GRADE_SCORE.get(grade, 0)

        nxt_info = after_data.get(ticker, {})
        nxt_sig = nxt_signals.get(ticker, {})

        # NXT 데이터가 있으면 보너스 점수
        nxt_volume = nxt_info.get("volume", 0)
        premium_pct = nxt_info.get("gap_pct", 0)
        net_buy_ratio = nxt_info.get("net_buy_ratio", 0.5)
        nxt_last_price = nxt_info.get("last_price", 0)
        prev_close = nxt_info.get("prev_close", 0)

        # 스코어 계산
        score = base_score

        # 순매수 보너스: 0.6 이상이면 +20, 0.55 이상 +10
        if net_buy_ratio >= 0.60:
            score += 20
        elif net_buy_ratio >= 0.55:
            score += 10
        elif net_buy_ratio < 0.40:
            score -= 20  # 순매도 페널티

        # 프리미엄 보너스: 양이면 +, 음이면 -
        if premium_pct >= 1.0:
            score += 15
        elif premium_pct >= 0.3:
            score += 8
        elif premium_pct <= -1.0:
            score -= 10

        # 거래량 보너스
        if nxt_volume >= 10000:
            score += 10
        elif nxt_volume >= 3000:
            score += 5

        # 기존 NXT 시그널 보너스
        sig = nxt_sig.get("signal", "")
        if sig == "STRONG_BUY":
            score += 15
        elif sig == "BUY":
            score += 8
        elif sig == "SELL":
            score -= 15

        # NXT 데이터 존재 여부
        has_nxt = bool(nxt_info)
        has_volume = nxt_volume >= MIN_VOLUME

        # 학습 가중치 반영 (데이터 10건 이상일 때만)
        if learning and learning.get("data_points", 0) >= 10:
            # NXT 데이터 보너스 보정
            if has_nxt:
                score += learning.get("nxt_data_bonus", 0)

        # 추천 등급 결정
        if score >= 120:
            rec_grade = "NXT강력포착"
        elif score >= 90:
            rec_grade = "NXT포착"
        elif score >= 60:
            rec_grade = "NXT주목"
        else:
            rec_grade = "보류"

        # 매수가격 제안 (NXT 현재가 기반)
        suggested_price = 0
        if nxt_last_price > 0:
            # 프리미엄 1% 이하면 NXT 현재가 매수 추천
            # 프리미엄 높으면 전일 종가 + α 제안
            if premium_pct <= 1.0:
                suggested_price = nxt_last_price
            else:
                suggested_price = int(prev_close * 1.005)  # 전일종가 +0.5%

        name = pick_info.get("name") or name_map.get(ticker, ticker)

        scored_picks.append({
            "ticker": ticker,
            "name": name,
            "main_grade": grade,
            "nxt_grade": rec_grade,
            "total_score": score,
            "base_score": base_score,
            "nxt_premium_pct": round(premium_pct, 2),
            "nxt_net_buy_ratio": round(net_buy_ratio, 3),
            "nxt_volume": nxt_volume,
            "nxt_last_price": nxt_last_price,
            "prev_close": prev_close,
            "suggested_price": suggested_price,
            "has_nxt_data": has_nxt,
            "nxt_signal": sig,
        })

    # 점수순 정렬
    scored_picks.sort(key=lambda x: x["total_score"], reverse=True)

    # 상위 MAX_PICKS만
    top_picks = [p for p in scored_picks if p["total_score"] >= MIN_SCORE][:MAX_PICKS]

    # NXT 데이터가 있는 종목 우선
    nxt_active = [p for p in top_picks if p["has_nxt_data"]]
    nxt_inactive = [p for p in top_picks if not p["has_nxt_data"]]

    result = {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "target_session": "after",
        "total_evaluated": len(scored_picks),
        "nxt_active_count": len(nxt_active),
        "picks": top_picks,
        "summary": {
            "nxt_강력포착": sum(1 for p in top_picks if p["nxt_grade"] == "NXT강력포착"),
            "nxt_포착": sum(1 for p in top_picks if p["nxt_grade"] == "NXT포착"),
            "nxt_주목": sum(1 for p in top_picks if p["nxt_grade"] == "NXT주목"),
            "avg_premium": round(
                sum(p["nxt_premium_pct"] for p in nxt_active) / len(nxt_active), 2
            ) if nxt_active else 0,
            "avg_net_buy": round(
                sum(p["nxt_net_buy_ratio"] for p in nxt_active) / len(nxt_active), 3
            ) if nxt_active else 0,
        },
    }

    # 저장
    NXT_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("NXT 추천 저장: %s (%d종목)", PICKS_PATH, len(top_picks))

    return result


def _empty_result(today: str) -> dict:
    """빈 결과 반환."""
    result = {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "target_session": "after",
        "total_evaluated": 0,
        "nxt_active_count": 0,
        "picks": [],
        "summary": {
            "nxt_강력포착": 0,
            "nxt_포착": 0,
            "nxt_주목": 0,
            "avg_premium": 0,
            "avg_net_buy": 0,
        },
    }
    NXT_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _send_telegram(result: dict):
    """NXT 추천 결과 텔레그램 발송."""
    picks = result.get("picks", [])
    summary = result.get("summary", {})

    if not picks:
        logger.info("NXT 추천 0건 → 텔레그램 스킵")
        return

    lines = [f"🌙 NXT 추천 ({result['date']})"]
    lines.append(
        f"강력포착 {summary.get('nxt_강력포착', 0)} / "
        f"포착 {summary.get('nxt_포착', 0)} / "
        f"주목 {summary.get('nxt_주목', 0)}"
    )
    lines.append("")

    for i, p in enumerate(picks[:7], 1):
        grade_emoji = {
            "NXT강력포착": "🔴",
            "NXT포착": "🟠",
            "NXT주목": "🟡",
        }.get(p["nxt_grade"], "⚪")

        nxt_info = ""
        if p["has_nxt_data"]:
            nxt_info = (
                f" P{p['nxt_premium_pct']:+.1f}%"
                f" 순매수{p['nxt_net_buy_ratio']:.0%}"
                f" V{p['nxt_volume']:,}"
            )
            if p["suggested_price"] > 0:
                nxt_info += f" → {p['suggested_price']:,}원"
        else:
            nxt_info = " (NXT거래없음)"

        lines.append(
            f"{grade_emoji} {i}. {p['name']}({p['ticker']}) "
            f"[{p['main_grade']}→{p['nxt_grade']}]"
            f"{nxt_info}"
        )

    msg = "\n".join(lines)

    try:
        from src.telegram_sender import send_message
        send_message(msg)
        logger.info("텔레그램 발송 완료")
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NXT 추천 엔진")
    parser.add_argument("--date", default=None, help="분석 날짜 (YYYY-MM-DD)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 안 함")
    args = parser.parse_args()

    result = generate_nxt_picks(target_date=args.date)

    # 결과 출력
    picks = result.get("picks", [])
    summary = result.get("summary", {})

    print(f"\n=== NXT 추천 엔진 ({result['date']}) ===")
    print(f"  분석: {result['total_evaluated']}종목, NXT거래: {result['nxt_active_count']}종목")
    print(f"  NXT강력포착: {summary.get('nxt_강력포착', 0)}")
    print(f"  NXT포착: {summary.get('nxt_포착', 0)}")
    print(f"  NXT주목: {summary.get('nxt_주목', 0)}")

    if picks:
        print(f"\n  TOP {len(picks)}:")
        for i, p in enumerate(picks, 1):
            nxt_tag = ""
            if p["has_nxt_data"]:
                nxt_tag = (
                    f" | NXT: P{p['nxt_premium_pct']:+.1f}% "
                    f"순매수{p['nxt_net_buy_ratio']:.0%} "
                    f"V{p['nxt_volume']:,}"
                )
            print(
                f"    {i}. {p['name']}({p['ticker']}) "
                f"[{p['main_grade']}→{p['nxt_grade']}] "
                f"점수{p['total_score']}"
                f"{nxt_tag}"
            )

    # 텔레그램 발송
    if not args.no_telegram:
        _send_telegram(result)


if __name__ == "__main__":
    main()
