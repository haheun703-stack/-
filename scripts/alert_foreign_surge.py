#!/usr/bin/env python3
"""장중 외인 폭발 텔레그램 실시간 알림 (P3)

장중 10:00~15:00, 30분 간격 crontab 실행
체결강도 120%+ AND 거래량 5일평균 3배+ → 외인 폭발 추정
텔레그램 [ALERT] 태그로 발송 (QUIET 모드에서도 통과)

Usage:
    python scripts/alert_foreign_surge.py           # 1회 스캔 + 알림
    python scripts/alert_foreign_surge.py --dry-run  # 알림 안 보내고 결과만 출력
    python scripts/alert_foreign_surge.py --top 100  # 상위 100종목 스캔
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "universe.csv"

# 체결강도/거래량 임계값
STRENGTH_THRESHOLD = 120  # 체결강도 120% 이상
VOLUME_MULTIPLIER = 3.0   # 5일 평균 거래량 대비 3배 이상
MIN_PRICE = 3000          # 최소 주가 (동전주 제외)


# ─────────────────────────────────────────────
# 1. 유니버스 + 5일 평균 거래량 로드
# ─────────────────────────────────────────────

def load_universe(top_n: int = 50) -> pd.DataFrame:
    """universe.csv에서 시총 상위 N종목 로드."""
    if not UNIVERSE_PATH.exists():
        logger.warning("universe.csv 없음: %s", UNIVERSE_PATH)
        return pd.DataFrame()

    df = pd.read_csv(UNIVERSE_PATH)
    df = df.dropna(subset=["ticker", "market_cap"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.sort_values("market_cap", ascending=False).head(top_n)
    logger.info("유니버스: %d종목 로드 (시총 상위)", len(df))
    return df


def load_avg_volume(ticker: str, days: int = 5) -> float:
    """CSV에서 최근 N일 평균 거래량 계산."""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return 0.0
    try:
        df = pd.read_csv(csvs[0], header=0)
        if len(df) < days + 1:
            return 0.0
        # Volume 컬럼 (6번째)
        vol_col = df.columns[5] if len(df.columns) > 5 else None
        if vol_col is None:
            return 0.0
        recent = df[vol_col].iloc[-(days + 1):-1]  # 오늘 제외 최근 N일
        avg = recent.mean()
        return float(avg) if not np.isnan(avg) else 0.0
    except Exception:
        return 0.0


def precompute_avg_volumes(tickers: list[str]) -> dict[str, float]:
    """전 종목 5일 평균 거래량 사전 계산."""
    result = {}
    for t in tickers:
        result[t] = load_avg_volume(t)
    loaded = sum(1 for v in result.values() if v > 0)
    logger.info("5일 평균 거래량: %d/%d종목 로드", loaded, len(tickers))
    return result


# ─────────────────────────────────────────────
# 2. KIS API 조회
# ─────────────────────────────────────────────

def init_broker():
    """mojito 브로커 초기화."""
    try:
        import mojito
        is_mock = os.getenv("MODEL") != "REAL"
        broker = mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY", ""),
            api_secret=os.getenv("KIS_APP_SECRET", ""),
            acc_no=os.getenv("KIS_ACC_NO", ""),
            mock=is_mock,
        )
        logger.info("[KIS] 브로커 초기화 완료 (mock=%s)", is_mock)
        return broker
    except Exception as e:
        logger.error("[KIS] 브로커 초기화 실패: %s", e)
        return None


def fetch_realtime_data(broker, ticker: str) -> dict | None:
    """KIS API로 현재가 + 체결 데이터 조회."""
    try:
        time.sleep(0.06)  # Rate limit 보호 (초당 ~16건)
        data = broker.fetch_price(ticker)
        output = data.get("output", {})
        if not output:
            return None

        price = int(output.get("stck_prpr", 0) or 0)
        if price < MIN_PRICE:
            return None

        volume = int(output.get("acml_vol", 0) or 0)
        change_pct = float(output.get("prdy_ctrt", 0) or 0)

        # 체결강도 계산: 매수체결량 / 매도체결량 × 100
        sell_vol = int(output.get("seln_cntg_smtn", 0) or 0)
        buy_vol = int(output.get("shnu_cntg_smtn", 0) or 0)

        # 대안: 체결강도 필드가 직접 있는 경우
        raw_strength = float(output.get("tday_rltv", 0) or 0)

        if raw_strength > 0:
            strength = raw_strength
        elif sell_vol > 0 and buy_vol > 0:
            strength = (buy_vol / sell_vol) * 100
        elif sell_vol > 0:
            strength = 0.0
        else:
            strength = 100.0  # 매도 0이면 기본값

        return {
            "ticker": ticker,
            "price": price,
            "volume": volume,
            "change_pct": change_pct,
            "strength": round(strength, 1),
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
        }
    except Exception as e:
        logger.debug("[KIS] %s 조회 실패: %s", ticker, e)
        return None


# ─────────────────────────────────────────────
# 3. 외인 폭발 감지
# ─────────────────────────────────────────────

def scan_foreign_surge(
    broker,
    universe: pd.DataFrame,
    avg_volumes: dict[str, float],
) -> list[dict]:
    """전 종목 스캔 → 외인 폭발 후보 반환."""
    results = []
    scanned = 0
    errors = 0

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        name = str(row.get("name", ticker))[:10]

        data = fetch_realtime_data(broker, ticker)
        if data is None:
            errors += 1
            continue
        scanned += 1

        avg_vol = avg_volumes.get(ticker, 0)
        if avg_vol <= 0:
            continue

        vol_ratio = data["volume"] / avg_vol
        strength = data["strength"]

        # 외인 폭발 조건
        is_surge = strength >= STRENGTH_THRESHOLD and vol_ratio >= VOLUME_MULTIPLIER
        # 완화 조건: 체결강도 매우 높거나 거래량 매우 폭발
        is_near = (
            (strength >= 150 and vol_ratio >= 2.0)
            or (strength >= STRENGTH_THRESHOLD and vol_ratio >= 5.0)
        )

        if not (is_surge or is_near):
            continue

        grade = "S" if strength >= 150 and vol_ratio >= 5.0 else \
                "A" if strength >= 150 or vol_ratio >= 5.0 else "B"

        results.append({
            "ticker": ticker,
            "name": name,
            "price": data["price"],
            "change_pct": data["change_pct"],
            "strength": strength,
            "volume": data["volume"],
            "avg_vol_5d": int(avg_vol),
            "vol_ratio": round(vol_ratio, 1),
            "grade": grade,
        })

    results.sort(key=lambda x: (-{"S": 3, "A": 2, "B": 1}.get(x["grade"], 0), -x["strength"]))
    logger.info("스캔 완료: %d종목 중 %d건 감지 (에러 %d)", scanned, len(results), errors)
    return results


# ─────────────────────────────────────────────
# 4. 텔레그램 알림
# ─────────────────────────────────────────────

def send_alert(results: list[dict], dry_run: bool = False):
    """텔레그램 알림 발송."""
    if not results:
        logger.info("감지 종목 없음 — 알림 생략")
        return

    now = datetime.now().strftime("%H:%M")
    lines = [f"[ALERT] 외인폭발 감지 ({now})\n"]

    for r in results:
        emoji = {"S": "🔴", "A": "🟠", "B": "🟡"}.get(r["grade"], "⚪")
        lines.append(
            f"{emoji} <b>{r['name']}</b> ({r['ticker']})\n"
            f"   {r['price']:,}원 {r['change_pct']:+.1f}%\n"
            f"   체결강도 {r['strength']:.0f}% | 거래량 {r['vol_ratio']:.1f}배\n"
        )

    lines.append(
        "\n<i>체결강도 120%+ AND 거래량 3배+</i>\n"
        "<i>D+1 양봉 확인 후 진입 검토</i>"
    )

    message = "\n".join(lines)

    if dry_run:
        print(f"\n[DRY-RUN] 텔레그램 발송 생략\n{message}")
        return

    try:
        from src.telegram_sender import send_message
        ok = send_message(message, parse_mode="HTML")
        if ok:
            logger.info("텔레그램 발송 성공 (%d종목)", len(results))
        else:
            logger.warning("텔레그램 발송 실패")
    except Exception as e:
        logger.error("텔레그램 발송 오류: %s", e)


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="장중 외인 폭발 텔레그램 알림 (P3)")
    parser.add_argument("--dry-run", action="store_true", help="알림 안 보내고 결과만 출력")
    parser.add_argument("--top", type=int, default=50, help="시총 상위 N종목 스캔")
    args = parser.parse_args()

    # 장중 체크 (주말/장외시간 실행 방지)
    now = datetime.now()
    if now.weekday() >= 5:
        logger.info("주말 — 스캔 생략")
        return
    hour = now.hour
    if hour < 9 or hour >= 16:
        logger.info("장외 시간(%02d시) — 스캔 생략", hour)
        return

    # 유니버스 + 5일 평균 거래량
    universe = load_universe(top_n=args.top)
    if universe.empty:
        logger.error("유니버스 로드 실패")
        return

    tickers = universe["ticker"].tolist()
    avg_volumes = precompute_avg_volumes(tickers)

    # KIS 브로커 초기화
    broker = init_broker()
    if broker is None:
        logger.error("KIS 브로커 초기화 실패 — 종료")
        return

    # 스캔
    results = scan_foreign_surge(broker, universe, avg_volumes)

    # 콘솔 리포트
    print(f"\n{'=' * 85}")
    print(f"  외인 폭발 감지 — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  조건: 체결강도 >={STRENGTH_THRESHOLD}% AND 거래량 >={VOLUME_MULTIPLIER}배")
    print(f"{'=' * 85}")

    if results:
        print(f"\n  [감지] {len(results)}종목")
        print(f"  {'등급':>4} {'종목':>10} {'현재가':>9} {'등락':>6} {'체결강도':>8} {'거래량배':>7}")
        print(f"  {'─' * 55}")
        for r in results:
            print(
                f"  {r['grade']:>4} {r['name']:>10} {r['price']:>9,} "
                f"{r['change_pct']:>+5.1f}% {r['strength']:>7.0f}% "
                f"{r['vol_ratio']:>6.1f}x"
            )
    else:
        print("\n  감지 종목 없음")
    print(f"{'=' * 85}")

    # 텔레그램
    send_alert(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
