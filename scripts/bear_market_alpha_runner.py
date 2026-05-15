"""약세장 알파 학습 + 인버스 시그널 자동 실행

흐름 (매일 BAT-D 후반부에서 자동):
1. 시장 레짐 판단 (KOSPI MA20 대비 등락률)
2. 약세장 모드 (-2%↓): 알파 종목 자동 추출 + 학습 누적
3. 인버스 ETF 시그널 (외인 5일 매도 -3조원+ 시 매수 권장)
4. 텔레그램 종합 알림

근거 (5/12~5/15 실측):
- 시장 -4.2%에도 LG전자/두산로보틱스/하나마이크론/자화전자/영원무역 +10~53% 상승
- 공통: 금투 또는 연기금 단독 매수 (외인 매도 중에도)
- KODEX 200선물인버스2X 4일 +7.3% (레버리지 효과)

출력:
- data/learning/bear_market_picks_YYYYMMDD.parquet (약세일에만)
- data/learning/inverse_signal_YYYYMMDD.json (외인 매도 폭증 시)
- 텔레그램: 약세장 알파 + 인버스 시그널
"""

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
LEARNING_DIR = DATA_DIR / "learning"
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = DATA_DIR / "investor_flow" / "investor_daily.db"
KOSPI_CSV = DATA_DIR / "kospi_index.csv"

# 인버스 ETF 마스터
INVERSE_ETFS = {
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "251340": "KODEX 코스닥150선물인버스",
}


def detect_market_regime() -> dict:
    """KOSPI 5일 등락률 OR MA20 대비 등락률로 레짐 판단.

    bear 조건 (둘 중 하나):
    - 5일 등락률 ≤ -3.0% (단기 약세)
    - MA20 대비 등락률 ≤ -2.0% (중기 약세)
    """
    if not KOSPI_CSV.exists():
        return {"regime": "unknown", "reason": "kospi_index.csv 없음"}
    df = pd.read_csv(KOSPI_CSV, encoding="utf-8-sig").tail(25)
    if len(df) < 20:
        return {"regime": "unknown", "reason": f"데이터 부족 ({len(df)}일)"}
    df["MA20"] = df["close"].rolling(20, min_periods=10).mean()
    last = df.iloc[-1]
    dev_ma20 = (last["close"] - last["MA20"]) / last["MA20"] * 100
    # 5일 등락률 (5일 전 종가 대비)
    ret_5d = None
    if len(df) >= 6:
        base_5d = df.iloc[-6]["close"]
        ret_5d = (last["close"] - base_5d) / base_5d * 100
    # 1일 등락률 (당일 폭락 감지)
    ret_1d = None
    if len(df) >= 2:
        prev = df.iloc[-2]["close"]
        ret_1d = (last["close"] - prev) / prev * 100

    # bear: MA20 -2%↓ OR 5일 -3%↓ OR 1일 -2.5%↓ (5/15 같은 폭락일 즉시 감지)
    is_bear = (
        (dev_ma20 <= -2.0)
        or (ret_5d is not None and ret_5d <= -3.0)
        or (ret_1d is not None and ret_1d <= -2.5)
    )
    is_bull = (dev_ma20 >= 2.0) and (ret_5d is None or ret_5d >= 0)
    if is_bear:
        regime = "bear"
    elif is_bull:
        regime = "bull"
    else:
        regime = "neutral"
    return {
        "regime": regime,
        "kospi_close": float(last["close"]),
        "ma20": float(last["MA20"]),
        "dev_ma20_pct": round(dev_ma20, 2),
        "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
        "ret_1d_pct": round(ret_1d, 2) if ret_1d is not None else None,
        "date": last["Date"] if "Date" in df.columns else None,
    }


def extract_alpha_stocks(today_str: str) -> list[dict]:
    """오늘 시장이 빠지는데도 +5%+ 오른 종목 추출 (학습 후보)."""
    today_date = f"{today_str[:4]}-{today_str[4:6]}-{today_str[6:8]}"
    yesterday = (datetime.strptime(today_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    alphas = []
    for csv_path in CSV_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["Date", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            sub = df[df["Date"].isin([today_date, yesterday])]
            if len(sub) < 2:
                continue
            sub = sub.set_index("Date")
            if today_date not in sub.index or yesterday not in sub.index:
                continue
            base = sub.loc[yesterday, "Close"]
            end = sub.loc[today_date, "Close"]
            if base <= 0:
                continue
            ret = (end - base) / base * 100
            # 시장 약세에도 +5%↑ 종목만
            if ret >= 5.0:
                ticker = csv_path.stem.split("_")[-1]
                name = csv_path.stem.rsplit("_", 1)[0]
                alphas.append({
                    "ticker": ticker,
                    "name": name,
                    "ret_today": round(ret, 1),
                    "close": int(end),
                    "volume": int(sub.loc[today_date, "Volume"]),
                })
        except Exception:
            continue
    return sorted(alphas, key=lambda x: x["ret_today"], reverse=True)


def get_flow_pattern(ticker: str, date_str: str, conn) -> dict:
    """종목의 당일 6유형 수급 패턴 (억원)."""
    rows = conn.execute(
        """SELECT investor, SUM(net_val) FROM investor_daily
           WHERE date=? AND ticker=? GROUP BY investor""",
        (date_str, ticker),
    ).fetchall()
    return {r[0]: round(r[1] / 1e8, 1) for r in rows}


def detect_inverse_signal() -> dict:
    """외인 5일 누적 매도 -3조원+ 시 인버스 매수 시그널."""
    if not DB_PATH.exists():
        return {"signal": False, "reason": "DB 없음"}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT date, SUM(net_val) FROM investor_daily
           WHERE investor='외국인' AND date >= ?
           GROUP BY date ORDER BY date DESC LIMIT 5""",
        ((date.today() - timedelta(days=14)).strftime("%Y%m%d"),),
    ).fetchall()
    conn.close()
    if not rows:
        return {"signal": False, "reason": "외인 데이터 없음"}
    total_5d_eok = sum(r[1] for r in rows) / 1e8
    threshold = -30000  # 5일 누적 -3조원
    return {
        "signal": total_5d_eok <= threshold,
        "foreign_5d_eok": round(total_5d_eok, 0),
        "threshold_eok": threshold,
        "days_count": len(rows),
        "recent_dates": [r[0] for r in rows],
    }


def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        for chunk in [msg[i : i + 3900] for i in range(0, len(msg), 3900)]:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
        return True
    except Exception:
        return False


def format_message(today: str, regime: dict, alphas: list, inverse: dict, flow_top: list) -> str:
    lines = [
        f"🧠 *약세장 알파 학습 ({today})*",
        "",
        f"📊 시장 레짐: *{regime['regime'].upper()}* (KOSPI MA20 대비 {regime.get('dev_pct', 0):+.2f}%)",
        "",
    ]
    if regime["regime"] == "bear":
        lines.append(f"🔴 *약세장 모드 활성* — 알파 종목 자동 학습 中")
        lines.append("")
        if alphas:
            lines.append(f"⭐ *역행 상승 종목 {len(alphas)}건 (+5%↑)*")
            for a in alphas[:10]:
                lines.append(f"• `{a['ticker']}` {a['name'][:12]} {a['ret_today']:+.1f}%")
            lines.append("")
        if flow_top:
            lines.append("📈 *수급 패턴 (TOP 5 알파)*")
            for f in flow_top[:5]:
                flow = f["flow"]
                lines.append(
                    f"• {f['name'][:12]}: 금투 {flow.get('금융투자', 0):+.0f}억 / "
                    f"연 {flow.get('연기금', 0):+.0f}억 / 외 {flow.get('외국인', 0):+.0f}억"
                )
            lines.append("")
    if inverse.get("signal"):
        lines.append(f"⚠️ *외인 매도 폭증 감지* — 5일 *{inverse['foreign_5d_eok']:,.0f}억* (임계 {inverse['threshold_eok']:,}억)")
        lines.append(f"")
        lines.append(f"📊 *인버스 ETF 참고 (Phase 9 백테스트)*")
        lines.append(f"• 1년 14건 평균 D+5 *-3.95%* (적중률 28.6%)")
        lines.append(f"• 단기 대박 +24% 가끔 / 대규모 손실 -21% 자주")
        lines.append(f"• 5/12~15 +7.3%는 *운*. 자동매매 *부적합*")
        lines.append(f"• 252670/114800/251340 — *수동 신중 판단* 영역")
        lines.append("")
    lines.append("_자동매매: OFF 유지 (학습/알림 전용)_")
    return "\n".join(lines)


def main():
    today_str = date.today().strftime("%Y%m%d")
    today_date = f"{today_str[:4]}-{today_str[4:6]}-{today_str[6:8]}"

    # 1. 시장 레짐
    regime = detect_market_regime()
    print(f"[regime] {regime}")

    # 2. 약세장이면 알파 학습
    alphas = []
    flow_top = []
    if regime["regime"] == "bear":
        alphas = extract_alpha_stocks(today_str)
        print(f"[alpha] {len(alphas)}개 역행 상승 종목 발견")

        # 상위 10개 수급 패턴
        if alphas and DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            for a in alphas[:10]:
                flow = get_flow_pattern(a["ticker"], today_str, conn)
                flow_top.append({**a, "flow": flow})
            conn.close()

            # 학습 데이터 저장
            df_save = pd.DataFrame(flow_top)
            save_path = LEARNING_DIR / f"bear_market_picks_{today_str}.parquet"
            df_save.to_parquet(save_path, index=False)
            print(f"[save] {save_path} ({len(df_save)}행)")

    # 3. 인버스 시그널
    inverse = detect_inverse_signal()
    print(f"[inverse] {inverse}")

    if inverse.get("signal"):
        inv_save = LEARNING_DIR / f"inverse_signal_{today_str}.json"
        inv_save.write_text(json.dumps(inverse, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {inv_save}")

    # 4. 텔레그램
    if regime["regime"] == "bear" or inverse.get("signal"):
        msg = format_message(today_date, regime, alphas, inverse, flow_top)
        ok = send_telegram(msg)
        print(f"[telegram] {'발송 OK' if ok else 'skip/실패'}")
    else:
        print("[info] 신호 없음 (강세/중립), 텔레그램 skip")

    print(f"\n[OK] bear_market_alpha_runner 완료 ({today_date})")


if __name__ == "__main__":
    main()
