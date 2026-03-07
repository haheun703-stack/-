"""
KIS 앱 스크린샷 기반 과거 매매 내역 일괄 등록

스크린샷 6장에서 추출한 결제 내역 + CSV 종가로 주당 가격/수량 역산.
trades.json을 완전 재생성한다.

결제일 → 거래일 매핑 (T+1, 설 연휴 02.16-18 반영):
  결제 02.11(수) → 거래 02.10(화)
  결제 02.12(목) → 거래 02.11(수)
  결제 02.13(금) → 거래 02.12(목)
  결제 02.19(목) → 거래 02.13(금) — 설 연휴로 결제 지연
  결제 02.20(금) → 거래 02.19(목)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JOURNAL_DIR = PROJECT_ROOT / "data" / "trade_journal"
TRADES_FILE = JOURNAL_DIR / "trades.json"
MONTHLY_DIR = JOURNAL_DIR / "monthly"


# ──────────────────────────────────────────
# 검증 완료된 전체 거래 목록
# ──────────────────────────────────────────
# 수량/가격: CSV 종가 기반 역산 + 현재 보유잔고 교차검증
# PnL: KIS 앱 총 거래금액 기준 (실제 수수료+세금 반영된 순손익)

ALL_TRADES = [
    # ── 스크린샷 범위 이전 매수 (보유 중, 매수일 미상 → 02.09 추정) ──
    {"date": "2026-02-09", "time": "09:30:00", "type": "BUY", "ticker": "004020",
     "name": "현대제철", "quantity": 30, "price": 34250, "amount": 1027500,
     "note": "보유종목 (매수일 추정)"},
    {"date": "2026-02-09", "time": "09:30:00", "type": "BUY", "ticker": "005930",
     "name": "삼성전자", "quantity": 36, "price": 187200, "amount": 6739200,
     "note": "보유종목 (매수일 추정)"},
    {"date": "2026-02-09", "time": "09:30:00", "type": "BUY", "ticker": "010950",
     "name": "S-Oil", "quantity": 27, "price": 107900, "amount": 2913300,
     "note": "보유종목 (매수일 추정)"},
    {"date": "2026-02-09", "time": "09:30:00", "type": "BUY", "ticker": "068270",
     "name": "셀트리온", "quantity": 13, "price": 242500, "amount": 3152500,
     "note": "보유종목 (매수일 추정)"},
    {"date": "2026-02-09", "time": "09:30:00", "type": "BUY", "ticker": "323410",
     "name": "카카오뱅크", "quantity": 73, "price": 27900, "amount": 2036700,
     "note": "보유종목 (매수일 추정)"},

    # ── 거래일 02.10 (결제 02.11) — 7건 매수 + 2건 매도 ──
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "001430",
     "name": "세아베스틸지주", "quantity": 23, "price": 80157, "amount": 1843600,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "006805",
     "name": "미래에셋증권우", "quantity": 98, "price": 19392, "amount": 1900460,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "023160",
     "name": "태광", "quantity": 67, "price": 28707, "amount": 1923400,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "038500",
     "name": "삼표시멘트", "quantity": 118, "price": 16086, "amount": 1898130,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "041920",
     "name": "메디아나", "quantity": 70, "price": 27394, "amount": 1917600,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "069500",
     "name": "KODEX 200", "quantity": 185, "price": 36943, "amount": 6834470,
     "note": "KIS 내역 (02.11 결제)"},
    {"date": "2026-02-10", "time": "09:30:00", "type": "BUY", "ticker": "098460",
     "name": "고영", "quantity": 58, "price": 32677, "amount": 1895250,
     "note": "KIS 내역 (02.11 결제)"},
    # 매도 — PnL = sell_amount - buy_amount (실제 순손익)
    {"date": "2026-02-10", "time": "14:00:00", "type": "SELL", "ticker": "038500",
     "name": "삼표시멘트", "quantity": 118, "price": 15447, "amount": 1822800,
     "buy_price": 16086, "pnl": -75330, "pnl_pct": -3.97,
     "note": "KIS 내역 (02.11 결제) — 당일 매도"},
    {"date": "2026-02-10", "time": "14:00:00", "type": "SELL", "ticker": "041920",
     "name": "메디아나", "quantity": 70, "price": 26569, "amount": 1859800,
     "buy_price": 27394, "pnl": -57800, "pnl_pct": -3.01,
     "note": "KIS 내역 (02.11 결제) — 당일 매도"},

    # ── 거래일 02.11 (결제 02.12) — 1건 매수 + 2건 매도 ──
    {"date": "2026-02-11", "time": "09:30:00", "type": "BUY", "ticker": "272110",
     "name": "케이엔제이", "quantity": 113, "price": 33451, "amount": 3780000,
     "note": "KIS 내역 (02.12 결제)"},
    {"date": "2026-02-11", "time": "14:00:00", "type": "SELL", "ticker": "001430",
     "name": "세아베스틸지주", "quantity": 23, "price": 76617, "amount": 1762200,
     "buy_price": 80157, "pnl": -81400, "pnl_pct": -4.41,
     "note": "KIS 내역 (02.12 결제)"},
    {"date": "2026-02-11", "time": "14:00:00", "type": "SELL", "ticker": "272110",
     "name": "케이엔제이", "quantity": 113, "price": 31350, "amount": 3542550,
     "buy_price": 33451, "pnl": -237450, "pnl_pct": -6.28,
     "note": "KIS 내역 (02.12 결제) — 당일 매도"},

    # ── 거래일 02.12 (결제 02.13) — 3건 매수 + 1건 매도 ──
    {"date": "2026-02-12", "time": "09:30:00", "type": "BUY", "ticker": "006360",
     "name": "GS건설", "quantity": 50, "price": 21300, "amount": 1065000,
     "note": "KIS 내역 (02.13 결제)"},
    {"date": "2026-02-12", "time": "09:30:00", "type": "BUY", "ticker": "021240",
     "name": "코웨이", "quantity": 36, "price": 86900, "amount": 3128400,
     "note": "KIS 내역 (02.13 결제)"},
    {"date": "2026-02-12", "time": "09:30:00", "type": "BUY", "ticker": "039130",
     "name": "하나투어", "quantity": 20, "price": 50000, "amount": 1000000,
     "note": "KIS 내역 (02.13 결제) — 1차 매수"},
    {"date": "2026-02-12", "time": "14:00:00", "type": "SELL", "ticker": "098460",
     "name": "고영", "quantity": 58, "price": 31203, "amount": 1809750,
     "buy_price": 32677, "pnl": -85500, "pnl_pct": -4.51,
     "note": "KIS 내역 (02.13 결제)"},

    # ── 거래일 02.13 (결제 02.19, 설 연휴) — 2건 매수 + 3건 매도 ──
    {"date": "2026-02-13", "time": "09:30:00", "type": "BUY", "ticker": "039130",
     "name": "하나투어", "quantity": 41, "price": 51249, "amount": 2101200,
     "note": "KIS 내역 (02.19 결제) — 2차 매수"},
    {"date": "2026-02-13", "time": "09:30:00", "type": "BUY", "ticker": "248070",
     "name": "솔루엠", "quantity": 96, "price": 17770, "amount": 1705950,
     "note": "KIS 내역 (02.19 결제)"},
    {"date": "2026-02-13", "time": "14:00:00", "type": "SELL", "ticker": "023160",
     "name": "태광", "quantity": 67, "price": 29076, "amount": 1948200,
     "buy_price": 28707, "pnl": 24800, "pnl_pct": 1.29,
     "note": "KIS 내역 (02.19 결제)"},
    {"date": "2026-02-13", "time": "14:00:00", "type": "SELL", "ticker": "069500",
     "name": "KODEX 200", "quantity": 185, "price": 38362, "amount": 7097025,
     "buy_price": 36943, "pnl": 262555, "pnl_pct": 3.84,
     "note": "KIS 내역 (02.19 결제)"},
    {"date": "2026-02-13", "time": "14:30:00", "type": "SELL", "ticker": "248070",
     "name": "솔루엠", "quantity": 96, "price": 18666, "amount": 1791900,
     "buy_price": 17770, "pnl": 85950, "pnl_pct": 5.04,
     "note": "KIS 내역 (02.19 결제) — 당일 매도"},

    # ── 거래일 02.19 (결제 02.20) — 2건 매수 ──
    {"date": "2026-02-19", "time": "09:30:00", "type": "BUY", "ticker": "011200",
     "name": "HMM", "quantity": 22, "price": 22909, "amount": 504000,
     "note": "KIS 내역 (02.20 결제)"},
    {"date": "2026-02-19", "time": "09:30:00", "type": "BUY", "ticker": "039130",
     "name": "하나투어", "quantity": 51, "price": 50029, "amount": 2551500,
     "note": "KIS 내역 (02.20 결제) — 3차 매수"},
]


def update_monthly(trades: list[dict]):
    """월별 요약 갱신."""
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    months = sorted(set(t["date"][:7] for t in trades))
    for month in months:
        month_trades = [t for t in trades if t["date"].startswith(month)]
        buys = [t for t in month_trades if t["type"] == "BUY"]
        sells = [t for t in month_trades if t["type"] == "SELL"]

        total_pnl = sum(t.get("pnl", 0) for t in sells)
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        loses = [t for t in sells if t.get("pnl", 0) < 0]

        summary = {
            "month": month,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "total_pnl": total_pnl,
            "win_count": len(wins),
            "lose_count": len(loses),
            "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
            "avg_pnl_pct": round(
                sum(t.get("pnl_pct", 0) for t in sells) / len(sells), 2
            ) if sells else 0,
            "best_trade": max(sells, key=lambda t: t.get("pnl", 0)) if sells else None,
            "worst_trade": min(sells, key=lambda t: t.get("pnl", 0)) if sells else None,
            "trades": month_trades,
        }

        path = MONTHLY_DIR / f"{month}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{month}] 매수 {len(buys)}건 | 매도 {len(sells)}건 | "
              f"승 {len(wins)}/패 {len(loses)} | 손익 {total_pnl:+,}원 | "
              f"승률 {summary['win_rate']}%")


def main():
    print("=" * 60)
    print("  KIS 앱 스크린샷 기반 과거 거래 일괄 등록")
    print("=" * 60)

    trades = ALL_TRADES

    # trades.json 저장
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_FILE.write_text(
        json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    buys = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]
    total_pnl = sum(t.get("pnl", 0) for t in sells)

    print(f"\n[거래 내역]")
    for t in trades:
        pnl_str = ""
        if t["type"] == "SELL" and "pnl" in t:
            pnl_str = f"  손익 {t['pnl']:+,}원 ({t['pnl_pct']:+.1f}%)"
        emoji = "BUY " if t["type"] == "BUY" else "SELL"
        print(f"  {t['date']} [{emoji}] {t['name']:<12s} {t['quantity']:>5d}주 × {t['price']:>8,}원{pnl_str}")

    print(f"\n[저장] trades.json: {len(trades)}건")

    # 월별 요약
    print(f"\n[월별 요약]")
    update_monthly(trades)

    # 총 요약
    print(f"\n{'='*60}")
    print(f"  총 {len(trades)}건 (매수 {len(buys)} / 매도 {len(sells)})")
    print(f"  매도 실현손익: {total_pnl:+,}원")
    if sells:
        wins = sum(1 for s in sells if s.get("pnl", 0) > 0)
        print(f"  승률: {wins}/{len(sells)} ({wins/len(sells)*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
