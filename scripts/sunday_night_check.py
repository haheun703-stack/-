"""일요일 밤 미장 체크 스크립트.

일요일 밤 21:00~22:00 실행.
나스닥 선물, VIX, DXY, 블루아울(OWL) 등 주요 지표를 수집하여
월요일 한국장 시나리오 A/B/C를 판정하고 텔레그램으로 발송.

사용법:
    python scripts/sunday_night_check.py           # 체크 + 텔레그램 발송
    python scripts/sunday_night_check.py --no-send # 체크만 (발송 안 함)

스케줄러 연동:
    daily_scheduler.py에서 일요일 21:00에 호출
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# ── 프로젝트 루트 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ═══════════════════════════════════════════════
# 1. 데이터 수집
# ═══════════════════════════════════════════════

TICKERS = {
    "NQ": "NQ=F",       # 나스닥 100 선물
    "ES": "ES=F",       # S&P 500 선물
    "VIX": "^VIX",      # VIX 공포지수
    "DXY": "DX-Y.NYB",  # 달러 인덱스
    "OWL": "OWL",       # 블루아울 캐피탈
    "EWY": "EWY",       # 한국 ETF
}


def fetch_market_data() -> dict:
    """yfinance로 주요 지표 수집."""
    result = {}
    end = datetime.now()
    start = end - timedelta(days=10)  # 최근 10일

    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False)
            if df.empty:
                result[name] = {"error": "데이터 없음"}
                continue

            # yfinance MultiIndex 처리
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

            close = float(last["Close"])
            prev_close = float(prev["Close"])
            change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0
            last_date = df.index[-1].strftime("%Y-%m-%d")

            # 5일 변화율
            if len(df) >= 5:
                close_5d = float(df.iloc[-5]["Close"])
                change_5d = (close - close_5d) / close_5d * 100
            else:
                change_5d = 0

            result[name] = {
                "close": close,
                "prev_close": prev_close,
                "change_pct": round(change_pct, 2),
                "change_5d": round(change_5d, 2),
                "last_date": last_date,
            }
        except Exception as e:
            logger.warning("[%s] 수집 실패: %s", name, e)
            result[name] = {"error": str(e)}

    return result


# ═══════════════════════════════════════════════
# 2. 시나리오 판정
# ═══════════════════════════════════════════════

def judge_scenario(data: dict) -> dict:
    """시나리오 A/B/C 판정.

    시나리오 A (공격): 나스닥선물 +, VIX 안정, EWY +
    시나리오 B (중립): 혼조세 또는 약보합
    시나리오 C (방어): 나스닥선물 -, VIX 급등, 달러 강세

    Returns:
        {"scenario": "A"|"B"|"C", "label": str, "reasons": list, "details": dict}
    """
    nq = data.get("NQ", {})
    es = data.get("ES", {})
    vix = data.get("VIX", {})
    dxy = data.get("DXY", {})
    owl = data.get("OWL", {})
    ewy = data.get("EWY", {})

    nq_chg = nq.get("change_pct", 0)
    es_chg = es.get("change_pct", 0)
    vix_level = vix.get("close", 20)
    vix_chg = vix.get("change_pct", 0)
    dxy_chg = dxy.get("change_pct", 0)
    ewy_chg = ewy.get("change_pct", 0)
    ewy_5d = ewy.get("change_5d", 0)

    reasons = []
    score = 0  # 양수=공격, 음수=방어

    # 나스닥 선물
    if nq_chg > 0.5:
        score += 2
        reasons.append(f"나스닥선물 강세({nq_chg:+.1f}%)")
    elif nq_chg < -0.5:
        score -= 2
        reasons.append(f"나스닥선물 약세({nq_chg:+.1f}%)")
    else:
        reasons.append(f"나스닥선물 보합({nq_chg:+.1f}%)")

    # VIX
    if vix_level >= 30:
        score -= 3
        reasons.append(f"VIX 극공포({vix_level:.0f})")
    elif vix_level >= 25:
        score -= 2
        reasons.append(f"VIX 공포({vix_level:.0f})")
    elif vix_level >= 20:
        score -= 1
        reasons.append(f"VIX 경계({vix_level:.0f})")
    else:
        score += 1
        reasons.append(f"VIX 안정({vix_level:.0f})")

    if vix_chg > 10:
        score -= 2
        reasons.append(f"VIX 급등({vix_chg:+.1f}%)")
    elif vix_chg < -10:
        score += 1
        reasons.append(f"VIX 급락({vix_chg:+.1f}%)")

    # 달러 인덱스
    if dxy_chg > 0.5:
        score -= 1
        reasons.append(f"달러 강세({dxy_chg:+.1f}%) → 외인 매도 경계")
    elif dxy_chg < -0.5:
        score += 1
        reasons.append(f"달러 약세({dxy_chg:+.1f}%) → 외인 유입 기대")

    # EWY (한국 ETF)
    if ewy_chg > 1.0:
        score += 1
        reasons.append(f"EWY 강세({ewy_chg:+.1f}%)")
    elif ewy_chg < -1.0:
        score -= 1
        reasons.append(f"EWY 약세({ewy_chg:+.1f}%)")

    if ewy_5d > 3.0:
        score += 1
        reasons.append(f"EWY 5일 모멘텀({ewy_5d:+.1f}%)")
    elif ewy_5d < -3.0:
        score -= 1
        reasons.append(f"EWY 5일 하락({ewy_5d:+.1f}%)")

    # OWL (블루아울 — 대체투자 센티먼트)
    owl_chg = owl.get("change_pct", 0)
    if owl_chg < -3.0:
        score -= 1
        reasons.append(f"OWL 급락({owl_chg:+.1f}%) → 유동성 경계")

    # 판정
    if score >= 2:
        scenario = "A"
        label = "공격 (적극 매수)"
    elif score <= -2:
        scenario = "C"
        label = "방어 (매수 자제)"
    else:
        scenario = "B"
        label = "중립 (선별 매수)"

    return {
        "scenario": scenario,
        "label": label,
        "score": score,
        "reasons": reasons,
        "details": {
            "NQ": nq_chg,
            "ES": es_chg,
            "VIX": vix_level,
            "VIX_chg": vix_chg,
            "DXY": dxy_chg,
            "EWY": ewy_chg,
            "EWY_5d": ewy_5d,
            "OWL": owl_chg,
        },
    }


# ═══════════════════════════════════════════════
# 3. 메시지 포맷
# ═══════════════════════════════════════════════

def format_message(data: dict, judgment: dict) -> str:
    """텔레그램 텍스트 메시지 생성."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"[Sunday Night Check] {now}")
    lines.append("=" * 30)

    # 시나리오 판정
    lines.append(f"시나리오 {judgment['scenario']}: {judgment['label']}")
    lines.append(f"종합 점수: {judgment['score']:+d}")
    lines.append("")

    # 근거
    lines.append("[판정 근거]")
    for r in judgment["reasons"]:
        lines.append(f"  - {r}")
    lines.append("")

    # 지표 요약
    lines.append("[시장 지표]")
    for name in ["NQ", "ES", "VIX", "DXY", "OWL", "EWY"]:
        d = data.get(name, {})
        if "error" in d:
            lines.append(f"  {name}: 수집 실패")
            continue
        close = d.get("close", 0)
        chg = d.get("change_pct", 0)
        chg_5d = d.get("change_5d", 0)
        date = d.get("last_date", "")

        if name == "VIX":
            lines.append(f"  VIX: {close:.1f} ({chg:+.1f}%) [5d {chg_5d:+.1f}%]")
        elif name in ("NQ", "ES"):
            lines.append(f"  {name}: {close:,.0f} ({chg:+.1f}%) [5d {chg_5d:+.1f}%]")
        else:
            lines.append(f"  {name}: {close:,.2f} ({chg:+.1f}%) [5d {chg_5d:+.1f}%]")

    lines.append("")

    # 액션 가이드
    lines.append("[월요일 액션]")
    if judgment["scenario"] == "A":
        lines.append("  - Quantum 시그널 적극 반영")
        lines.append("  - Relay 진입적기 즉시 매수")
        lines.append("  - 분할매수 1차 비중 확대 가능")
    elif judgment["scenario"] == "B":
        lines.append("  - Quantum 시그널 선별 반영")
        lines.append("  - Relay 승률 75%+ 패턴만 진입")
        lines.append("  - 분할매수 1차 50% 원칙 유지")
    else:  # C
        lines.append("  - Quantum 매수 보류")
        lines.append("  - Relay 신규 진입 자제")
        lines.append("  - 기존 포지션 방어 우선 (손절 타이트)")

    lines.append("=" * 30)
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 4. 실행
# ═══════════════════════════════════════════════

def run(send: bool = True) -> dict:
    """일요일 밤 체크 실행."""
    print("=" * 40)
    print("  Sunday Night Market Check")
    print("=" * 40)

    # 1. 데이터 수집
    print("\n[1/3] 시장 데이터 수집...")
    data = fetch_market_data()

    for name, d in data.items():
        if "error" in d:
            print(f"  {name}: 실패 ({d['error']})")
        else:
            print(f"  {name}: {d['close']:,.2f} ({d['change_pct']:+.1f}%)")

    # 2. 시나리오 판정
    print("\n[2/3] 시나리오 판정...")
    judgment = judge_scenario(data)
    print(f"  결과: 시나리오 {judgment['scenario']} — {judgment['label']} (점수 {judgment['score']:+d})")

    # 3. 메시지 생성 + 발송
    print("\n[3/3] 메시지 생성...")
    msg = format_message(data, judgment)
    print(msg)

    if send:
        try:
            from telegram_sender import send_message
            ok = send_message(msg)
            if ok:
                print("\n텔레그램 발송 완료")
            else:
                print("\n텔레그램 발송 실패")
        except Exception as e:
            print(f"\n텔레그램 발송 오류: {e}")

    return {"data": data, "judgment": judgment, "message": msg}


def main():
    parser = argparse.ArgumentParser(description="일요일 밤 미장 체크")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 발송 안 함")
    args = parser.parse_args()

    run(send=not args.no_send)


if __name__ == "__main__":
    main()
