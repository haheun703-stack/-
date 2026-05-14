"""정보봇 ETF 수급 일일 자동 수집 + theme dual_buy 시그널 추출

흐름:
- 16:28 정보봇 etf_investor_flow Supabase 업데이트
- 17:00+ 퀀트봇 BAT-D 후반부에서 fetch
- 5일 누적 계산 → dual_buy + theme 카테고리 추출
- 텔레그램 발송 (Phase 8 백테스트 88.9% 적중률 시그널)

출력:
- data/etf_flow_history.parquet (일별 누적)
- data/etf_theme_signals_YYYYMMDD.json (당일 시그널)
- 텔레그램: theme dual_buy 종목 리스트

실행:
    python -X utf8 scripts/fetch_etf_flow_daily.py             # fetch + 저장
    python -X utf8 scripts/fetch_etf_flow_daily.py --analyze   # + 시그널 추출
    python -X utf8 scripts/fetch_etf_flow_daily.py --telegram  # + 텔레그램 발송
"""

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
HISTORY_PATH = DATA_DIR / "etf_flow_history.parquet"

# Phase 8 결과: theme 카테고리 88.9% 적중, D+3 +1.52%
PHASE8_THRESHOLD_HIT_RATE = 60  # 최소 적중률 기준 (참고용)


def fetch_etf_flow_full() -> pd.DataFrame:
    """Supabase etf_investor_flow 전체 export"""
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    res = sb.table("etf_investor_flow").select("*").order("date").execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_5d_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["ticker", "date"])
    df["fgn_5d"] = df.groupby("ticker")["foreign_net_amt"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["inst_5d"] = df.groupby("ticker")["institution_net_amt"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["signal"] = "none"
    df.loc[(df["fgn_5d"] > 0) & (df["inst_5d"] > 0), "signal"] = "dual_buy"
    df.loc[(df["fgn_5d"] > 0) & (df["inst_5d"] <= 0), "signal"] = "fgn_only"
    df.loc[(df["fgn_5d"] <= 0) & (df["inst_5d"] > 0), "signal"] = "inst_only"
    df.loc[(df["fgn_5d"] <= 0) & (df["inst_5d"] <= 0), "signal"] = "dual_sell"
    return df


def extract_theme_signals(df: pd.DataFrame, target_date: str) -> list[dict]:
    """target_date의 theme 카테고리 dual_buy 추출 (최강 시그널)"""
    today = df[df["date"] == target_date].copy()
    if today.empty:
        return []
    # Phase 8 최강 시그널: theme + dual_buy
    theme_dual = today[(today["category"] == "theme") & (today["signal"] == "dual_buy")]
    if theme_dual.empty:
        return []
    # 외+기 합산 기준 정렬
    theme_dual = theme_dual.assign(
        combined=theme_dual["fgn_5d"] + theme_dual["inst_5d"]
    ).sort_values("combined", ascending=False)
    return theme_dual.to_dict("records")


def extract_all_dual_buy(df: pd.DataFrame, target_date: str) -> list[dict]:
    """target_date 전체 dual_buy (참고용)"""
    today = df[df["date"] == target_date].copy()
    if today.empty:
        return []
    dual = today[today["signal"] == "dual_buy"].copy()
    return dual.to_dict("records")


def send_telegram(message: str):
    """텔레그램 발송"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] TELEGRAM 환경변수 미설정 → 발송 스킵")
        return
    # 4096자 초과 분할
    chunks = [message[i : i + 3900] for i in range(0, len(message), 3900)]
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"[WARN] 텔레그램 발송 실패: {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"[WARN] 텔레그램 에러: {e}")


def format_telegram_message(target_date: str, theme_signals: list[dict], all_duals: list[dict]) -> str:
    """텔레그램 메시지 포맷"""
    lines = [
        f"🔥 *ETF 수급 자동 시그널 ({target_date})*",
        "",
        f"📊 Phase 8 백테스트 기반 (theme 88.9% / D+3 +1.52%)",
        "",
    ]
    if theme_signals:
        lines.append(f"⭐ *Theme dual\\_buy {len(theme_signals)}개 (최강 시그널)*")
        for s in theme_signals[:10]:
            name = s["name"][:20]
            ticker = s["ticker"]
            fgn = s["fgn_5d"]
            inst = s["inst_5d"]
            lines.append(f"• `{ticker}` {name} 외인+{fgn:.0f} 기관+{inst:.0f}")
        lines.append("")
    else:
        lines.append("⚠️ Theme dual\\_buy 시그널 없음")
        lines.append("")

    # 다른 카테고리 dual_buy 요약
    by_cat = {}
    for d in all_duals:
        by_cat.setdefault(d.get("category", "?"), []).append(d)
    lines.append(f"📈 *전체 dual\\_buy 분포 ({len(all_duals)}건)*")
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        lines.append(f"• {cat}: {len(items)}건")
    lines.append("")
    lines.append(f"_매수 시점: D+0 종가 / 매도 권장: D+3_")
    lines.append(f"_자동매매: OFF (백테스트 누적 후 활성 결정)_")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="시그널 추출")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 발송")
    parser.add_argument("--date", default=None, help="대상 날짜 (YYYY-MM-DD, 기본 오늘)")
    args = parser.parse_args()

    target_date = args.date or date.today().strftime("%Y-%m-%d")
    print(f"[fetch] target_date={target_date}")

    df = fetch_etf_flow_full()
    if df.empty:
        print("[ERROR] etf_investor_flow 데이터 없음")
        sys.exit(1)
    print(f"  → {len(df)}건 ({df['date'].min()} ~ {df['date'].max()})")

    df = add_5d_cumulative(df)

    # 누적 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HISTORY_PATH, index=False)
    print(f"[save] {HISTORY_PATH}")

    if args.analyze or args.telegram:
        theme_signals = extract_theme_signals(df, target_date)
        all_duals = extract_all_dual_buy(df, target_date)
        print(f"\n[시그널] theme dual_buy: {len(theme_signals)}개, 전체 dual_buy: {len(all_duals)}개")

        # 당일 시그널 JSON 저장
        d_compact = target_date.replace("-", "")
        signals_out = DATA_DIR / f"etf_theme_signals_{d_compact}.json"
        signals_out.write_text(
            json.dumps(
                {
                    "date": target_date,
                    "phase8_backtest": "theme 88.9% / D+3 +1.52%",
                    "theme_dual_buy": theme_signals[:20],
                    "all_dual_buy_count": len(all_duals),
                    "all_dual_buy_by_category": {
                        cat: sum(1 for d in all_duals if d.get("category") == cat)
                        for cat in set(d.get("category") for d in all_duals)
                    },
                },
                ensure_ascii=False,
                default=str,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[save] {signals_out}")

        if theme_signals:
            print(f"\n[Theme dual_buy 상위 5]")
            for s in theme_signals[:5]:
                print(f"  {s['ticker']} {s['name']} 외+{s['fgn_5d']:.0f} 기+{s['inst_5d']:.0f}")

        if args.telegram:
            msg = format_telegram_message(target_date, theme_signals, all_duals)
            send_telegram(msg)
            print("[telegram] 발송 완료")

    print(f"\n[OK] ETF 수급 자동 수집 완료 ({target_date})")


if __name__ == "__main__":
    main()
