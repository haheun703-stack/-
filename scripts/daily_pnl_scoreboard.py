#!/usr/bin/env python3
"""일일 수익 스코어보드 (7/17 신설 — 데일리 루프 A-0 정본).

목적: "매일 뭔가를 하지만 수익과 연결되나?"에 매일 숫자로 답한다.
페이퍼 원장 5종의 daily_equity를 읽어 계좌별 전일 대비 / 누적 / 동일구간
KOSPI 대비 알파를 계산 → JSONL 누적 + 텔레그램 1줄 요약.

- 입력: data/paper_*.json (daily_equity: [{date, equity, ...}]) + data/kospi_index.csv
- 출력: data/metrics/pnl_scoreboard.jsonl (누적) + pnl_scoreboard_latest.json + 텔레그램
- cron: BAT-D 후반 (대시보드 적재 후, 메트릭 수집 전) — run_bat.sh
- 알파 정의: 계좌 누적수익률 − 같은 기간(계좌 가동일~최신) KOSPI 수익률 (%p)

실행:
    python scripts/daily_pnl_scoreboard.py            # 계산+저장+텔레그램
    python scripts/daily_pnl_scoreboard.py --no-send  # 텔레그램 생략
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
KOSPI_CSV = DATA_DIR / "kospi_index.csv"

# (표시명, 원장 파일) — daily_equity 스키마 공통 5종. indexbh는 스키마 상이 → 참고 랭킹만.
LEDGERS = [
    ("메인A", "paper_portfolio.json"),
    ("B안", "paper_portfolio_b.json"),
    ("블루칩V3", "paper_bluechip.json"),
    ("파도VF", "paper_portfolio_vf.json"),
    ("현금방어NAV", "paper_portfolio_holdnav.json"),
]
INDEXBH_FILE = "paper_portfolio_indexbh.json"


def _norm_date(s: str) -> str:
    """'2026-07-16' / '20260716' → '2026-07-16'."""
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def load_kospi_closes() -> dict[str, float]:
    """kospi_index.csv → {YYYY-MM-DD: close} (컬럼명 자동 탐지)."""
    import pandas as pd

    df = pd.read_csv(KOSPI_CSV)
    date_col = next((c for c in df.columns if c.lower() in ("date", "날짜")), df.columns[0])
    close_col = next((c for c in df.columns if c.lower() in ("close", "종가")), df.columns[-1])
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            out[_norm_date(row[date_col])] = float(row[close_col])
        except (ValueError, TypeError):
            continue
    return out


def kospi_at_or_before(closes: dict[str, float], date_str: str) -> float | None:
    """해당 일자 종가, 없으면(주말/휴장) 직전 거래일 종가."""
    if date_str in closes:
        return closes[date_str]
    prior = [d for d in closes if d <= date_str]
    return closes[max(prior)] if prior else None


def build_scoreboard() -> dict:
    closes = load_kospi_closes()
    accounts = []
    for label, fname in LEDGERS:
        path = DATA_DIR / fname
        if not path.exists():
            logger.warning("[PNL] %s 없음 — 스킵", fname)
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            logger.warning("[PNL] %s 로드 실패: %s — 스킵", fname, e)
            continue
        de = d.get("daily_equity") or []
        init = float(d.get("initial_capital", 0))
        if not de or init <= 0:
            logger.warning("[PNL] %s daily_equity/초기자본 없음 — 스킵", fname)
            continue

        first, last = de[0], de[-1]
        prev = de[-2] if len(de) >= 2 else first
        eq_last = float(last.get("equity", 0))
        eq_prev = float(prev.get("equity", 0)) or eq_last
        d0, d1 = _norm_date(first.get("date", "")), _norm_date(last.get("date", ""))

        cum_pct = (eq_last / init - 1) * 100
        day_pct = (eq_last / eq_prev - 1) * 100 if eq_prev else 0.0

        k0, k1 = kospi_at_or_before(closes, d0), kospi_at_or_before(closes, d1)
        kospi_cum = (k1 / k0 - 1) * 100 if (k0 and k1) else None
        alpha = cum_pct - kospi_cum if kospi_cum is not None else None

        accounts.append({
            "account": label,
            "date": d1,
            "since": d0,
            "equity": round(eq_last),
            "day_pct": round(day_pct, 2),
            "cum_pct": round(cum_pct, 2),
            "kospi_cum_pct": round(kospi_cum, 2) if kospi_cum is not None else None,
            "alpha_pct": round(alpha, 2) if alpha is not None else None,
            "positions": last.get("positions"),
            "stock_ratio": last.get("stock_ratio"),
        })

    # 지수BH 랭킹 상위 3 (참고 — 페이퍼 6번째 영구벤치마크)
    idx_top = []
    idx_path = DATA_DIR / INDEXBH_FILE
    if idx_path.exists():
        try:
            b = json.load(open(idx_path, encoding="utf-8"))
            bms = b.get("benchmarks", {})
            ranked = sorted(bms.values(), key=lambda x: x.get("return_pct", 0), reverse=True)
            idx_top = [{"name": r.get("name", r.get("symbol", "?")),
                        "return_pct": round(float(r.get("return_pct", 0)), 2)} for r in ranked[:3]]
        except Exception as e:
            logger.warning("[PNL] indexbh 로드 실패: %s", e)

    return {
        "date": max((a["date"] for a in accounts), default=datetime.now().strftime("%Y-%m-%d")),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "accounts": accounts,
        "indexbh_top3": idx_top,
    }


def format_report(sb: dict) -> str:
    lines = [f"💰 수익 스코어보드 ({sb['date']})"]
    for a in sb["accounts"]:
        alpha = f"α{a['alpha_pct']:+.1f}%p" if a["alpha_pct"] is not None else "α n/a"
        lines.append(
            f"{a['account']}: 누적 {a['cum_pct']:+.1f}% ({alpha}) | 오늘 {a['day_pct']:+.2f}%"
        )
    if sb["accounts"]:
        best = max(sb["accounts"], key=lambda x: (x["alpha_pct"] if x["alpha_pct"] is not None else -1e9))
        k = next((a["kospi_cum_pct"] for a in sb["accounts"] if a["kospi_cum_pct"] is not None), None)
        if k is not None:
            lines.append(f"(동일구간 KOSPI {k:+.1f}% | 알파 1위: {best['account']})")
    if sb["indexbh_top3"]:
        top = " / ".join(f"{t['name']} {t['return_pct']:+.1f}%" for t in sb["indexbh_top3"])
        lines.append(f"지수BH Top3: {top}")
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    import requests

    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return True
    except Exception as e:
        logger.warning("[PNL] 텔레그램 실패: %s", e)
        return False


def main():
    ap = argparse.ArgumentParser(description="일일 수익 스코어보드")
    ap.add_argument("--no-send", action="store_true", help="텔레그램 생략")
    args = ap.parse_args()

    sb = build_scoreboard()
    if not sb["accounts"]:
        logger.error("[PNL] 계산된 계좌 0개 — 중단 (exit 1)")
        sys.exit(1)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "pnl_scoreboard.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(sb, ensure_ascii=False) + "\n")
    with open(METRICS_DIR / "pnl_scoreboard_latest.json", "w", encoding="utf-8") as f:
        json.dump(sb, f, ensure_ascii=False, indent=2)

    report = format_report(sb)
    print(report)
    if not args.no_send:
        ok = send_telegram(report)
        logger.info("[PNL] 텔레그램 %s", "발송 OK" if ok else "발송 SKIP")
    logger.info("[PNL] 저장 완료: %s (%d계좌)", METRICS_DIR / "pnl_scoreboard.jsonl", len(sb["accounts"]))


if __name__ == "__main__":
    main()
