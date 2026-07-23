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


def load_universe_prices() -> dict:
    """data/raw/*.parquet 종가 — 동일가중 벤치마크용.

    실패해도 빈 dict를 돌려 동일가중 알파만 n/a가 되게 한다.
    (7/20 교훈: 신설 기능이 스코어보드 본체 발송을 막으면 안 된다)
    """
    prices: dict = {}
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[PNL] pandas 없음 — 동일가중 벤치마크 생략")
        return prices
    raw = DATA_DIR / "raw"
    if not raw.exists():
        return prices
    for f in raw.glob("*.parquet"):
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
            s.index = pd.to_datetime(s.index)
            prices[f.stem] = s.sort_index()
        except Exception:
            continue
    return prices


def equal_weight_return(prices: dict, d0: str, d1: str):
    """d0 종가에 전 종목 동일금액 매수 → d1 보유 시 평균 수익률(%).

    ★왜 필요한가 (7/23): KOSPI는 시총가중이라 최상위 주도장에서는 개별종목
      포트폴리오가 종목을 아무리 잘 골라도 구조적으로 뒤진다. 실측 3/24~7/22에
      KOSPI +22.4% vs 유니버스 동일가중 -17.4% = **격차 39.8%p**.
      이 격차 탓에 메인A 알파가 -34.1%p로 찍혔지만, 동일가중 대비로는 **+5.7%p**다.
      KOSPI 알파만 보면 "종목 선택 실패"로 오독하게 되므로 둘을 병기한다.
    표본 100종목 미만이면 None(무의미한 비교 방지).
    """
    if not prices:
        return None
    try:
        import pandas as pd
        t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
    except Exception:
        return None
    rets = []
    for s in prices.values():
        if len(s) == 0 or s.index[-1] < t0:
            continue  # 구간 시작 전에 데이터가 끊긴 종목(상폐 등) 제외
        i0 = s.index.searchsorted(t0, side="right") - 1
        i1 = s.index.searchsorted(t1, side="right") - 1
        if i0 < 0 or i1 < 0:
            continue
        p0, p1 = float(s.iloc[i0]), float(s.iloc[i1])
        if p0 > 0:
            rets.append((p1 / p0 - 1) * 100)
    return sum(rets) / len(rets) if len(rets) >= 100 else None


def build_scoreboard() -> dict:
    closes = load_kospi_closes()
    try:
        uni_prices = load_universe_prices()
    except Exception as e:  # noqa: BLE001 — 본체 발송 보호
        logger.warning("[PNL] 유니버스 로드 실패: %s — 동일가중 생략", e)
        uni_prices = {}
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

        # 동일가중 벤치마크 (7/23 신설) — 종목 선택의 순수 성과
        try:
            ew_cum = equal_weight_return(uni_prices, d0, d1)
        except Exception as e:  # noqa: BLE001 — 본체 발송 보호
            logger.warning("[PNL] %s 동일가중 계산 실패: %s", label, e)
            ew_cum = None
        alpha_ew = cum_pct - ew_cum if ew_cum is not None else None
        # 구조적 격차 = 개별주를 들고 있다는 사실만으로 생기는 지수 대비 손실
        struct_gap = (kospi_cum - ew_cum) if (kospi_cum is not None and ew_cum is not None) else None

        accounts.append({
            "account": label,
            "date": d1,
            "since": d0,
            "equity": round(eq_last),
            "day_pct": round(day_pct, 2),
            "cum_pct": round(cum_pct, 2),
            "kospi_cum_pct": round(kospi_cum, 2) if kospi_cum is not None else None,
            "alpha_pct": round(alpha, 2) if alpha is not None else None,
            "ew_cum_pct": round(ew_cum, 2) if ew_cum is not None else None,
            "alpha_ew_pct": round(alpha_ew, 2) if alpha_ew is not None else None,
            "struct_gap_pct": round(struct_gap, 2) if struct_gap is not None else None,
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
        alpha = f"αK{a['alpha_pct']:+.1f}" if a["alpha_pct"] is not None else "αK n/a"
        # αEW = 동일가중 대비 = 종목 선택의 순수 성과 (7/23 신설)
        alpha_ew = f"αEW{a['alpha_ew_pct']:+.1f}" if a.get("alpha_ew_pct") is not None else "αEW n/a"
        lines.append(
            f"{a['account']}: 누적 {a['cum_pct']:+.1f}% ({alpha} / {alpha_ew}) | 오늘 {a['day_pct']:+.2f}%"
        )
    if sb["accounts"]:
        head = sb["accounts"][0]  # 구간 기준 계좌(메인A) — 계좌마다 가동일이 달라 대표 1개만 표기
        k, ew = head.get("kospi_cum_pct"), head.get("ew_cum_pct")
        if k is not None and ew is not None:
            lines.append(
                f"({head['account']} 구간 KOSPI {k:+.1f}% / 동일가중 {ew:+.1f}% "
                f"→ 구조격차 {head['struct_gap_pct']:+.1f}%p)")
        elif k is not None:
            lines.append(f"({head['account']} 구간 KOSPI {k:+.1f}%)")
        best_k = max(sb["accounts"], key=lambda x: (x["alpha_pct"] if x["alpha_pct"] is not None else -1e9))
        best_ew = max(sb["accounts"], key=lambda x: (x.get("alpha_ew_pct") if x.get("alpha_ew_pct") is not None else -1e9))
        lines.append(f"알파 1위: KOSPI기준 {best_k['account']} / 동일가중기준 {best_ew['account']}")
    if sb["indexbh_top3"]:
        top = " / ".join(f"{t['name']} {t['return_pct']:+.1f}%" for t in sb["indexbh_top3"])
        lines.append(f"지수BH Top3: {top}")
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    import requests

    # .env 정본 키명은 TELEGRAM_BOT_TOKEN (7/20 실측: TELEGRAM_TOKEN은 미정의라 조용히 SKIP됨).
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        # 7/21 검수: 응답 미검증 시 401/400/429가 예외 없이 통과해 "발송 OK" 오보.
        # A-0(1번 지표)라 실물 발송 여부를 응답 ok로 확정한다.
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        data = r.json()
        if not data.get("ok"):
            logger.warning("[PNL] 텔레그램 거부: HTTP %s %s",
                           r.status_code, data.get("description", ""))
            return False
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
    latest = METRICS_DIR / "pnl_scoreboard_latest.json"
    tmp_latest = latest.with_suffix(".json.tmp")
    with open(tmp_latest, "w", encoding="utf-8") as f:
        json.dump(sb, f, ensure_ascii=False, indent=2)
    tmp_latest.replace(latest)  # 7/21 검수: 원자적 쓰기(중단 시 손상 방지)

    report = format_report(sb)
    print(report)
    if not args.no_send:
        ok = send_telegram(report)
        logger.info("[PNL] 텔레그램 %s", "발송 OK" if ok else "발송 SKIP")
    logger.info("[PNL] 저장 완료: %s (%d계좌)", METRICS_DIR / "pnl_scoreboard.jsonl", len(sb["accounts"]))


if __name__ == "__main__":
    main()
