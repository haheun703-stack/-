#!/usr/bin/env python
"""스나이퍼 워치 — 밸류트랩 사냥 + 관심종목 통합 일일 모니터링

data/sniper_watch/watchlist.json 기반으로 10종목을 매일 스캔하여
기술적 진입 시그널을 판정하고, 일별 히스토리를 누적한다.

진입 판단 기준 (7가지):
  1. 52주 저점 근처 (10% 이내)
  2. 거래량 급증/증가 추세
  3. 5/20 골든크로스
  4. 20일선 돌파
  5. RSI 과매도 반등 (30 이하→반등)
  6. 볼린저 하단 반등
  7. 외국인/기관 순매수

판정: STRONG_ENTRY(4+) > ENTRY(3) > WATCH(2) > WAIT

출력:
  data/sniper_watch/latest.json          — 최신 스캔 결과
  data/sniper_watch/history/YYYY-MM-DD.json — 일별 히스토리

사용법:
  python -u -X utf8 scripts/scan_value_trap.py [--telegram] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mojito
from dotenv import load_dotenv

load_dotenv()

# ── 경로 ─────────────────────────────────────────
WATCH_DIR = Path("data/sniper_watch")
WATCHLIST_PATH = WATCH_DIR / "watchlist.json"
LATEST_PATH = WATCH_DIR / "latest.json"
HISTORY_DIR = WATCH_DIR / "history"
PERIOD_DAYS = 240  # 약 1년치


def load_watchlist() -> list[dict]:
    """워치리스트 전체 로드"""
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["stocks"]


# ── KIS API ───────────────────────────────────────
def fetch_ohlcv(broker: mojito.KoreaInvestment, code: str) -> list[dict]:
    """일봉 OHLCV 조회 (최근 ~1년)"""
    end_day = date.today().strftime("%Y%m%d")
    start_day = (date.today() - timedelta(days=int(PERIOD_DAYS * 1.5))).strftime("%Y%m%d")
    resp = broker.fetch_ohlcv(code, timeframe="D", start_day=start_day, end_day=end_day)
    rows = resp.get("output2", [])
    candles = []
    for r in reversed(rows):
        dt = r.get("stck_bsop_date", "")
        clpr = r.get("stck_clpr")
        if not dt or not clpr:
            continue
        candles.append({
            "date": dt,
            "open": int(r["stck_oprc"]),
            "high": int(r["stck_hgpr"]),
            "low": int(r["stck_lwpr"]),
            "close": int(r["stck_clpr"]),
            "volume": int(r.get("acml_vol", 0)),
        })
    return candles[-PERIOD_DAYS:]


def fetch_investor(broker: mojito.KoreaInvestment, code: str) -> dict:
    """투자자별 매매동향"""
    import requests
    base_url = "https://openapi.koreainvestment.com:9443"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {broker.access_token}",
        "appkey": os.getenv("KIS_APP_KEY"),
        "appsecret": os.getenv("KIS_APP_SECRET"),
        "tr_id": "FHKST01010900",
        "custtype": "P",
    }
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
    try:
        resp = requests.get(
            f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-investor",
            headers=headers, params=params, timeout=10,
        )
        items = resp.json().get("output", [])
        if not items:
            return {"frgn_net": 0, "inst_net": 0}
        item = items[0]
        return {
            "frgn_net": int(item.get("frgn_ntby_qty", 0)),
            "inst_net": int(item.get("orgn_ntby_qty", 0)),
        }
    except Exception:
        return {"frgn_net": 0, "inst_net": 0}


# ── 기술적 분석 (변경 없음) ─────────────────────────
def analyze(candles: list[dict], investor: dict) -> dict:
    """진입 시그널 종합 분석"""
    if len(candles) < 60:
        return {"error": "데이터 부족", "signal_count": 0, "verdict": "WAIT", "signals": []}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    cur_price = closes[-1]
    prev_price = closes[-2] if len(closes) >= 2 else cur_price

    def sma(data, period):
        return sum(data[-period:]) / period if len(data) >= period else None

    # 52주 저/고점
    low_52w = min(lows[-min(len(lows), 240):])
    high_52w = max(highs[-min(len(highs), 240):])
    vs_low = (cur_price - low_52w) / low_52w * 100 if low_52w > 0 else 0
    vs_high = (cur_price - high_52w) / high_52w * 100 if high_52w > 0 else 0

    # 이동평균
    ma5 = sma(closes, 5)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    ma120 = sma(closes, 120)

    # 골든크로스 (5/20, 최근 5일 내)
    golden_cross = False
    if len(closes) >= 25:
        for i in range(-5, 0):
            m5p = sum(closes[i - 5:i]) / 5
            m20p = sum(closes[i - 20:i]) / 20
            m5c = sum(closes[i - 4:i + 1]) / 5
            m20c = sum(closes[i - 19:i + 1]) / 20
            if m5p <= m20p and m5c > m20c:
                golden_cross = True
                break

    # 20일선 돌파
    ma20_breakout = False
    if ma20 and len(closes) >= 21:
        ma20_y = sum(closes[-21:-1]) / 20
        if prev_price < ma20_y and cur_price >= ma20:
            ma20_breakout = True

    # RSI 14
    rsi = None
    if len(closes) >= 15:
        gains = [max(closes[i] - closes[i - 1], 0) for i in range(-14, 0)]
        losses = [max(closes[i - 1] - closes[i], 0) for i in range(-14, 0)]
        ag, al = sum(gains) / 14, sum(losses) / 14
        rsi = 100 - (100 / (1 + ag / al)) if al > 0 else 100.0

    # RSI 과매도 반등
    rsi_bounce = False
    if len(closes) >= 20 and rsi:
        for lb in range(2, 7):
            idx = -lb
            if len(closes) + idx - 14 >= 1:
                g2 = [max(closes[j] - closes[j - 1], 0) for j in range(idx - 14, idx)]
                l2 = [max(closes[j - 1] - closes[j], 0) for j in range(idx - 14, idx)]
                ag2, al2 = sum(g2) / 14, sum(l2) / 14
                pr = 100 - (100 / (1 + ag2 / al2)) if al2 > 0 else 100
                if pr <= 30 and rsi > pr:
                    rsi_bounce = True
                    break

    # 거래량
    va20 = sma(volumes, 20) or 1
    vol_ratio = volumes[-1] / va20 if va20 > 0 else 0
    va5 = sma(volumes, 5)
    va20p = sma(volumes[:-5], 20) if len(volumes) >= 25 else va20
    vol_up = (va5 or 0) > (va20p or 1) * 1.3

    # 볼린저
    bb_bounce = False
    bb_width = 0
    if ma20 and len(closes) >= 20:
        std = (sum((c - ma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        bb_l = ma20 - 2 * std
        bb_u = ma20 + 2 * std
        bb_bounce = cur_price <= bb_l * 1.02
        bb_width = round((bb_u - bb_l) / ma20 * 100, 1) if ma20 > 0 else 0

    # 수급
    frgn = investor.get("frgn_net", 0) > 0
    inst = investor.get("inst_net", 0) > 0

    # 시그널 수집
    signals = []
    if vs_low <= 10:
        signals.append("52주저점근처")
    if golden_cross:
        signals.append("골든크로스(5/20)")
    if ma20_breakout:
        signals.append("20일선돌파")
    if rsi_bounce:
        signals.append("RSI과매도반등")
    if vol_ratio >= 2.0:
        signals.append(f"거래량폭발({vol_ratio:.1f}x)")
    elif vol_up:
        signals.append("거래량증가추세")
    if bb_bounce:
        signals.append("볼린저하단반등")
    if frgn or inst:
        who = []
        if frgn: who.append("외국인")
        if inst: who.append("기관")
        signals.append(f"{'+'.join(who)}순매수")

    sc = len(signals)
    verdict = "STRONG_ENTRY" if sc >= 4 else "ENTRY" if sc >= 3 else "WATCH" if sc >= 2 else "WAIT"

    ma_st = "N/A"
    if ma5 and ma20 and ma60:
        ma_st = "정배열" if ma5 > ma20 > ma60 else "역배열" if ma5 < ma20 < ma60 else "혼조"

    return {
        "price": cur_price,
        "change_pct": round((cur_price - prev_price) / prev_price * 100, 2) if prev_price else 0,
        "52w_low": low_52w, "52w_high": high_52w,
        "vs_52w_low_pct": round(vs_low, 1), "vs_52w_high_pct": round(vs_high, 1),
        "ma5": int(ma5) if ma5 else None, "ma20": int(ma20) if ma20 else None,
        "ma60": int(ma60) if ma60 else None, "ma120": int(ma120) if ma120 else None,
        "ma_status": ma_st, "rsi": round(rsi, 1) if rsi else None,
        "vol_ratio": round(vol_ratio, 2), "vol_increasing": vol_up, "bb_width": bb_width,
        "frgn_net": investor.get("frgn_net", 0), "inst_net": investor.get("inst_net", 0),
        "signals": signals, "signal_count": sc, "verdict": verdict,
    }


# ── 리포트 ────────────────────────────────────────
EMOJI = {"STRONG_ENTRY": "🔴", "ENTRY": "🟠", "WATCH": "🟡", "WAIT": "⚪"}
GROUP_BANNER = {"밸류트랩": "🎯 밸류트랩 사냥", "관심종목": "📌 관심종목"}


def format_report(results: list[dict]) -> str:
    """그룹별 배너 + 시그널순 정렬 텔레그램 리포트"""
    lines = [f"🔫 스나이퍼 워치 리포트 ({date.today()})", ""]

    for group_key, banner in GROUP_BANNER.items():
        group = [r for r in results if r["group"] == group_key]
        if not group:
            continue
        group.sort(key=lambda x: x["analysis"]["signal_count"], reverse=True)

        lines.append(f"{'━' * 30}")
        lines.append(f"{banner} ({len(group)}종목)")
        lines.append(f"{'━' * 30}")

        for r in group:
            a = r["analysis"]
            e = EMOJI.get(a["verdict"], "⚪")
            grade_tag = f"[{r['grade']}급]" if r["grade"] != "W" else ""
            lines.append(f"{e} {r['name']} ({r['code']}) {grade_tag}")
            lines.append(f"  {a['price']:,}원 ({a['change_pct']:+.1f}%) | RSI {a['rsi']}")
            lines.append(f"  52주저점+{a['vs_52w_low_pct']}% | {a['ma_status']} | Vol {a['vol_ratio']}x")
            if a["frgn_net"] or a["inst_net"]:
                lines.append(f"  외국인 {a['frgn_net']:+,} | 기관 {a['inst_net']:+,}")
            if a["signals"]:
                lines.append(f"  → {', '.join(a['signals'])}")
            lines.append(f"  ▶ {a['verdict']} ({a['signal_count']}개)")
            lines.append("")

    # 요약
    s = sum(1 for r in results if r["analysis"]["verdict"] == "STRONG_ENTRY")
    e = sum(1 for r in results if r["analysis"]["verdict"] == "ENTRY")
    w = sum(1 for r in results if r["analysis"]["verdict"] == "WATCH")
    lines.append(f"─ 총 {len(results)}종목 | 🔴{s} 🟠{e} 🟡{w}")

    # ENTRY 이상만 요약 강조
    hot = [r for r in results if r["analysis"]["verdict"] in ("STRONG_ENTRY", "ENTRY")]
    if hot:
        lines.append("")
        lines.append("⚡ 진입 검토 대상:")
        for r in hot:
            lines.append(f"  • {r['name']}: {', '.join(r['analysis']['signals'])}")

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="스나이퍼 워치 스캐너")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 발송")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🔫 스나이퍼 워치 스캐너 — {date.today()}")
    print("=" * 60)

    stocks = load_watchlist()
    print(f"\n워치리스트: {len(stocks)}종목")
    for s in stocks:
        print(f"  [{s['group']}] {s['name']} ({s['code']})")

    broker = mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=os.getenv("MODEL") != "REAL",
    )

    results = []
    for s in stocks:
        code, name, group, grade = s["code"], s["name"], s["group"], s["grade"]
        print(f"\n[{group}] {name} ({code}) 분석 중...")

        try:
            candles = fetch_ohlcv(broker, code)
            if not candles or len(candles) < 60:
                print(f"  ⚠ 데이터 부족 ({len(candles)}일)")
                continue

            investor = fetch_investor(broker, code)
            analysis = analyze(candles, investor)

            result = {
                "code": code, "name": name, "group": group, "grade": grade,
                "sector": s.get("sector", ""),
                "thesis": s.get("thesis", ""),
                "analysis": analysis,
                "scan_date": str(date.today()),
            }
            results.append(result)

            v = analysis["verdict"]
            sc = analysis["signal_count"]
            e = EMOJI.get(v, "⚪")
            print(f"  {e} {v} ({sc}개) | {analysis['price']:,}원 | RSI {analysis['rsi']}")
            if analysis["signals"]:
                print(f"  → {', '.join(analysis['signals'])}")

        except Exception as ex:
            print(f"  ✗ 에러: {ex}")
            import traceback
            traceback.print_exc()

        time.sleep(0.5)

    # 결과 저장
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    hist_path = HISTORY_DIR / f"{date.today()}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {LATEST_PATH}")
    print(f"히스토리: {hist_path}")

    # 리포트
    report = format_report(results)
    print(f"\n{report}")

    # 텔레그램
    if args.telegram and results:
        try:
            from src.telegram_sender import send_message
            send_message(report)
            print("\n✓ 텔레그램 발송 완료")
        except Exception as ex:
            print(f"\n✗ 텔레그램 에러: {ex}")


if __name__ == "__main__":
    main()
