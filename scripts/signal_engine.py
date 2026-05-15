"""자비스 시그널 엔진 — 검증된 8개 시그널 통합 점수

목적:
- 매일 BAT-D 후반부 자동 실행
- 검증된 백테스트 시그널을 통합 점수화
- 상위 5종목 추출 → 텔레그램 알림 (MOCK/실전 매수 후보)

검증된 시그널 (백테스트 기준):
A급 (PF 2.5+, 점수 100): PB15_BB / 우량주TOP100 / PULLBACK_20MA
B급 (적중률 50%+, 점수 50): Phase5_stage3 / Phase8_theme_dual / 외인폭발+눌림
보너스 (점수 +10): 다중 시그널 동시 발생

활용:
- Phase 1 (5/16~): 텔레그램 알림만 (MOCK X)
- Phase 2 (5/23~): MOCK 자동매수 (모의투자 계좌)
- Phase 5 (6/27~): 실전 활성화

실행 (BAT-D 후반부 자동):
    python -X utf8 scripts/signal_engine.py            # 일별 추출
    python -X utf8 scripts/signal_engine.py --top 5    # 상위 N개
    python -X utf8 scripts/signal_engine.py --notify   # 텔레그램
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
OUT_DIR = PROJECT_ROOT / "data" / "signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 시그널 점수
# ──────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    # A급 (백테스트 PF 2.5+, 점수 100)
    "PB15_BB": 100,              # PULLBACK15 + 볼린저밴드 (WR 60.2%, PF 2.64)
    "BLUECHIP_TIMING": 100,       # 우량주TOP100 매매타이밍 (PF 3.35)
    "PULLBACK_20MA_15": 90,       # 20MA 눌림 15% 단일 (PF 2.02)
    # B급 (적중률 50%+, 점수 50~70)
    "PHASE5_STAGE3": 70,          # 수급 3단계 완성 (적중률 59.1%)
    "PHASE8_THEME_DUAL": 70,      # theme ETF dual_buy (적중률 88.9%)
    "FOREIGN_SURGE_PB": 60,        # 외인폭발+20MA눌림+D+1양봉 (PF 1.82)
    "SILENT_GOLD_COMBO": 50,       # Silent Accumulation Gold (D+5 +1.52%)
    "LAGGARD_FOLLOW": 50,          # 래거드 추격 (WR 61.9%)
}


def calc_indicators(df: pd.DataFrame) -> dict:
    """단일 종목 OHLCV로 핵심 지표 계산"""
    if len(df) < 20:
        return {}
    close = df["Close"].iloc[-1]
    ret_1d = (close - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100 if len(df) >= 2 else 0
    ret_5d = (close - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100 if len(df) >= 6 else 0
    ret_60d = (close - df["Close"].iloc[-61]) / df["Close"].iloc[-61] * 100 if len(df) >= 61 else 0
    ma5 = df["Close"].iloc[-5:].mean()
    ma20 = df["Close"].iloc[-20:].mean()
    ma60 = df["Close"].iloc[-60:].mean() if len(df) >= 60 else ma20

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1] if len(df) >= 15 else 50

    # 볼린저밴드 (20MA ± 2std)
    std20 = df["Close"].iloc[-20:].std()
    bb_lower = ma20 - 2 * std20
    bb_upper = ma20 + 2 * std20
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # 20MA 이격
    ma20_dev = (close - ma20) / ma20 * 100

    # 거래량
    vol = df["Volume"].iloc[-1]
    vol_5d_avg = df["Volume"].iloc[-5:].mean()
    vol_ratio = vol / vol_5d_avg if vol_5d_avg > 0 else 1

    return {
        "close": close,
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_60d": ret_60d,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ma20_dev": ma20_dev,
        "rsi": rsi if not pd.isna(rsi) else 50,
        "bb_position": bb_position,
        "vol_ratio": vol_ratio,
    }


def detect_signals(ticker: str, ind: dict, supply: dict) -> dict:
    """시그널 발생 여부 판정 → {시그널명: True/False}"""
    sigs = {}

    # A1. PB15_BB: 20MA 눌림 15% + 볼린저 하단 근처
    sigs["PB15_BB"] = (
        ind.get("ma20_dev", 0) <= -8 and ind.get("ma20_dev", 0) >= -15
        and ind.get("bb_position", 0.5) <= 0.25
        and ind.get("rsi", 50) <= 40
    )

    # A2. BLUECHIP_TIMING: 우량주 + 눌림 + 외인 매수
    # (시총 5조원+ 우량주만 / 별도 화이트리스트 필요)
    sigs["BLUECHIP_TIMING"] = (
        ind.get("ma20_dev", 0) <= -5
        and ind.get("rsi", 50) <= 45
        and supply.get("fgn_5d", 0) >= 50  # 외인 5일 +50억+
        and ticker in BLUECHIP_WHITELIST
    )

    # A3. PULLBACK_20MA_15: 20MA 눌림 -10~-15%
    sigs["PULLBACK_20MA_15"] = (
        ind.get("ma20_dev", 0) <= -10 and ind.get("ma20_dev", 0) >= -18
        and ind.get("rsi", 50) <= 45
    )

    # B1. PHASE5_STAGE3: 금투+연기금+기타+기관+외인 모두 매수
    sigs["PHASE5_STAGE3"] = (
        supply.get("finance_5d", 0) > 0
        and supply.get("pension_5d", 0) > 0
        and supply.get("corp_5d", 0) > 0
        and supply.get("inst_5d", 0) > 0
        and supply.get("fgn_5d", 0) > 0
    )

    # B2. PHASE8_THEME_DUAL: ETF 데이터 필요 (별도 처리)
    sigs["PHASE8_THEME_DUAL"] = False  # 별도 fetch_etf_flow_daily에서 추출

    # B3. FOREIGN_SURGE_PB: 외인폭발 + 20MA눌림 + D+1양봉
    sigs["FOREIGN_SURGE_PB"] = (
        supply.get("fgn_5d", 0) >= 100  # 외인 5일 +100억+
        and ind.get("ma20_dev", 0) <= -3
        and ind.get("ret_1d", 0) >= 1.0  # 어제 양봉 1%+
    )

    # B4. SILENT_GOLD_COMBO: RSI≤32 + 60일 횡보 + 수급 10~50억
    sigs["SILENT_GOLD_COMBO"] = (
        ind.get("rsi", 50) <= 32
        and abs(ind.get("ret_60d", 0)) <= 3
        and 10 <= supply.get("fgn_5d", 0) + supply.get("inst_5d", 0) <= 50
    )

    # B5. LAGGARD_FOLLOW: 어제 +3%+ + 거래량 2배+
    sigs["LAGGARD_FOLLOW"] = (
        ind.get("ret_1d", 0) >= 3.0
        and ind.get("vol_ratio", 1) >= 2.0
    )

    return sigs


# 시총 5조원+ 우량주 (KOSPI200 대표 30종목, 추후 확장)
BLUECHIP_WHITELIST = {
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "373220",  # LG에너지솔루션
    "207940",  # 삼성바이오로직스
    "005380",  # 현대차
    "000270",  # 기아
    "068270",  # 셀트리온
    "035420",  # NAVER
    "035720",  # 카카오
    "012330",  # 현대모비스
    "066570",  # LG전자
    "017670",  # SK텔레콤
    "055550",  # 신한지주
    "105560",  # KB금융
    "086790",  # 하나금융지주
    "138930",  # BNK금융지주
    "316140",  # 우리금융지주
    "032830",  # 삼성생명
    "051910",  # LG화학
    "006400",  # 삼성SDI
    "247540",  # 에코프로비엠
    "003670",  # 포스코퓨처엠
    "009540",  # HD한국조선해양
    "010130",  # 고려아연
    "024110",  # 기업은행
    "316140",  # 우리금융
    "030200",  # KT
    "015760",  # 한국전력
    "036460",  # 한국가스공사
    "267260",  # HD현대일렉트릭
}


def load_supply_5d(ticker: str, end_date: str, conn) -> dict:
    """종목의 5일 누적 6유형 수급 (억원)"""
    d_compact = end_date.replace("-", "")
    sql = """
    SELECT investor, SUM(net_val)
    FROM investor_daily
    WHERE ticker = ?
      AND date <= ?
      AND date >= (SELECT MIN(date) FROM (
          SELECT DISTINCT date FROM investor_daily
          WHERE date <= ?
          ORDER BY date DESC LIMIT 5))
      AND investor IN ('금융투자','연기금','기타법인','기관합계','외국인')
    GROUP BY investor
    """
    rows = conn.execute(sql, (ticker, d_compact, d_compact)).fetchall()
    return {
        "finance_5d": next((r[1] / 1e8 for r in rows if r[0] == "금융투자"), 0),
        "pension_5d": next((r[1] / 1e8 for r in rows if r[0] == "연기금"), 0),
        "corp_5d": next((r[1] / 1e8 for r in rows if r[0] == "기타법인"), 0),
        "inst_5d": next((r[1] / 1e8 for r in rows if r[0] == "기관합계"), 0),
        "fgn_5d": next((r[1] / 1e8 for r in rows if r[0] == "외국인"), 0),
    }


def calc_total_score(sigs: dict) -> tuple[int, list[str]]:
    """시그널 → 점수 + 발생 시그널 리스트"""
    score = 0
    triggered = []
    for sig, is_active in sigs.items():
        if is_active:
            score += SIGNAL_WEIGHTS.get(sig, 0)
            triggered.append(sig)
    # 다중 시그널 보너스
    if len(triggered) >= 2:
        score += 10 * (len(triggered) - 1)
    return score, triggered


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="상위 N개")
    parser.add_argument("--notify", action="store_true", help="텔레그램 발송")
    parser.add_argument("--date", default=None, help="기준일 (YYYY-MM-DD)")
    args = parser.parse_args()

    today = args.date or date.today().strftime("%Y-%m-%d")
    print(f"[signal_engine] target={today}")

    conn = sqlite3.connect(DB_PATH)
    candidates = []

    csv_files = list(CSV_DIR.glob("*.csv"))
    print(f"  종목 스캔: {len(csv_files)}개")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["Date", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values("Date").reset_index(drop=True)
            # today 이전 데이터만 (백테스트용)
            df = df[df["Date"] <= today]
            if len(df) < 20:
                continue

            ticker = csv_path.stem.split("_")[-1]
            name = csv_path.stem.rsplit("_", 1)[0]

            ind = calc_indicators(df)
            if not ind:
                continue

            supply = load_supply_5d(ticker, today, conn)
            sigs = detect_signals(ticker, ind, supply)
            score, triggered = calc_total_score(sigs)

            if score > 0:
                candidates.append({
                    "ticker": ticker,
                    "name": name,
                    "score": score,
                    "signals": triggered,
                    "close": int(ind["close"]),
                    "rsi": round(ind["rsi"], 1),
                    "ma20_dev": round(ind["ma20_dev"], 1),
                    "fgn_5d": round(supply["fgn_5d"], 0),
                    "inst_5d": round(supply["inst_5d"], 0),
                    "finance_5d": round(supply["finance_5d"], 0),
                    "pension_5d": round(supply["pension_5d"], 0),
                })
        except Exception:
            continue

    conn.close()

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[: args.top]
    print(f"\n[result] 후보 {len(candidates)}개, TOP {len(top)}:\n")

    for i, c in enumerate(top, 1):
        print(
            f"  {i}. {c['name']}({c['ticker']}) 점수 {c['score']}점 — "
            f"{', '.join(c['signals'])}"
        )
        print(
            f"     RSI {c['rsi']} / MA20 {c['ma20_dev']:+.1f}% / "
            f"외+{c['fgn_5d']:.0f}억 기+{c['inst_5d']:.0f}억 "
            f"금투+{c['finance_5d']:.0f}억 연+{c['pension_5d']:.0f}억"
        )

    # 저장
    today_compact = today.replace("-", "")
    out_file = OUT_DIR / f"signals_{today_compact}.json"
    out_file.write_text(
        json.dumps(
            {
                "date": today,
                "total_candidates": len(candidates),
                "top_n": args.top,
                "picks": top,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[save] {out_file}")

    # 텔레그램
    if args.notify and top:
        lines = [
            f"🤖 *자비스 시그널 엔진 ({today})*",
            f"",
            f"📊 검증된 8개 시그널 통합 점수 (Phase 1)",
            f"",
        ]
        for i, c in enumerate(top, 1):
            lines.append(f"*{i}. {c['name']}({c['ticker']})* — {c['score']}점")
            lines.append(f"   {', '.join(c['signals'])}")
            lines.append(
                f"   RSI {c['rsi']} / MA20 {c['ma20_dev']:+.1f}% / "
                f"외+{c['fgn_5d']:.0f}억 기+{c['inst_5d']:.0f}억"
            )
            lines.append("")
        lines.append("_자동매매: OFF (Phase 1 알림 전용)_")
        msg = "\n".join(lines)
        ok = send_telegram(msg)
        print(f"[telegram] {'발송 OK' if ok else '발송 실패'}")


if __name__ == "__main__":
    main()
