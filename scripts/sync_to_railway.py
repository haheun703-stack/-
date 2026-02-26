"""
로컬 데이터 → Railway 서버 동기화

BAT-D 완료 후 호출:
  python scripts/sync_to_railway.py

환경변수:
  RAILWAY_URL    — Railway 배포 URL (예: https://xxx.railway.app)
  JARVIS_SECRET  — 동기화 토큰 (.env에서 로드)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sync")

RAILWAY_URL = os.getenv("RAILWAY_URL", "")
SYNC_TOKEN = os.getenv("JARVIS_SECRET", "")
DATA_DIR = PROJECT_ROOT / "data"

# 동기화 대상 파일 목록 (실제 스크립트 출력 파일명 기준)
SYNC_FILES = [
    "tomorrow_picks.json",
    "etf_master.json",
    "picks_history.json",
    "us_market/overnight_signal.json",
    "sector_rotation/sector_momentum.json",
    "sector_rotation/etf_trading_signal.json",
    "sector_rotation/sector_zscore.json",
    "sector_rotation/investor_flow.json",
    "sector_rotation/relay_trading_signal.json",
    "whale_detect.json",
    "force_hybrid.json",
    "market_news.json",
    "dual_buying_watch.json",
    "pullback_scan.json",
    "group_relay/group_relay_today.json",
    "scan_cache.json",
    "dart_disclosures.json",
    "integrated_report.json",
    "leverage_etf/leverage_etf_scan.json",
    # 전략 업그레이드 (Timefolio 벤치마크)
    "regime_macro_signal.json",
    "dart_event_signals.json",
    "portfolio_allocation.json",
    # 라이브 데이터 (동기화 전 자동 캡처)
    "kis_balance.json",
    "kospi_regime.json",
    "volume_spike_watchlist.json",
    "institutional_targets.json",
    "accumulation_tracker.json",
]


def capture_live_data():
    """동기화 전 KIS 보유종목 + KOSPI 레짐을 JSON으로 캡처."""

    # ── KIS 보유종목 캡처 ──
    try:
        if os.getenv("KIS_APP_KEY"):
            from src.adapters.kis_order_adapter import KisOrderAdapter
            kis = KisOrderAdapter()
            balance = kis.fetch_balance()
            balance["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            dest = DATA_DIR / "kis_balance.json"
            dest.write_text(json.dumps(balance, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            logger.info("[CAPTURE] KIS 보유종목 %d건 → kis_balance.json", len(balance.get("holdings", [])))
        else:
            logger.warning("[SKIP] KIS_APP_KEY 미설정 → kis_balance.json 캡처 불가")
    except Exception as e:
        logger.error("[CAPTURE] KIS 보유종목 실패: %s", e)

    # ── KOSPI 레짐 캡처 ──
    try:
        import numpy as np
        import pandas as pd

        kospi_path = DATA_DIR / "kospi_index.csv"
        if kospi_path.exists():
            df = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma60"] = df["close"].rolling(60).mean()

            if len(df) >= 60:
                row = df.iloc[-1]
                prev = df.iloc[-2]
                close = float(row["close"])
                ma20 = float(row["ma20"])
                ma60 = float(row["ma60"])

                log_ret = np.log(df["close"] / df["close"].shift(1))
                rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
                rv20_pct = rv20.rolling(252, min_periods=60).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
                rv_pct = float(rv20_pct.iloc[-1]) if not pd.isna(rv20_pct.iloc[-1]) else 0.5

                if close > ma20:
                    regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
                elif close > ma60:
                    regime, slots = "BEAR", 2
                else:
                    regime, slots = "CRISIS", 0

                prev_close = float(prev["close"])
                change = round((close / prev_close - 1) * 100, 2) if prev_close > 0 else 0

                regime_data = {
                    "regime": regime, "slots": slots,
                    "close": round(close, 2), "change": change,
                    "ma20": round(ma20, 2), "ma60": round(ma60, 2),
                    "rv_pct": round(rv_pct, 2),
                    "date": str(df.index[-1].date()),
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                dest = DATA_DIR / "kospi_regime.json"
                dest.write_text(json.dumps(regime_data, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("[CAPTURE] KOSPI 레짐: %s (슬롯 %d) → kospi_regime.json", regime, slots)
        else:
            logger.warning("[SKIP] kospi_index.csv 없음")
    except Exception as e:
        logger.error("[CAPTURE] KOSPI 레짐 실패: %s", e)


def sync_file(filename: str) -> bool:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("[SKIP] %s 없음", filename)
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[FAIL] %s 파싱 오류: %s", filename, e)
        return False

    try:
        resp = requests.post(
            f"{RAILWAY_URL}/api/sync",
            json={"file": filename, "data": data},
            headers={"X-Sync-Token": SYNC_TOKEN},
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("[OK] %s", filename)
            return True
        else:
            logger.error("[FAIL] %s → %d: %s", filename, resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.error("[FAIL] %s → %s", filename, e)
        return False


def git_commit_data():
    """데이터 JSON을 git에 커밋+푸시 (Railway 재배포 시 데이터 유지용)."""
    import subprocess

    try:
        # data/ 내 JSON 파일만 추가
        subprocess.run(
            ["git", "add", "data/*.json", "data/**/*.json"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=10,
        )
        # 변경사항 확인
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            logger.info("[GIT] 데이터 변경 없음 — 커밋 스킵")
            return

        subprocess.run(
            ["git", "commit", "-m", f"data: 자동 데이터 스냅샷 {time.strftime('%Y-%m-%d %H:%M')}"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60,
        )
        if push.returncode == 0:
            logger.info("[GIT] 데이터 커밋+푸시 완료")
        else:
            logger.warning("[GIT] 푸시 실패: %s", push.stderr.decode()[:200])
    except Exception as e:
        logger.warning("[GIT] 자동 커밋 실패 (무시): %s", e)


def main():
    if not RAILWAY_URL:
        logger.error("RAILWAY_URL 환경변수가 설정되지 않았습니다.")
        logger.info("  .env에 추가: RAILWAY_URL=https://xxx.railway.app")
        sys.exit(1)

    if not SYNC_TOKEN:
        logger.error("JARVIS_SECRET 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 동기화 전 라이브 데이터 캡처
    capture_live_data()

    logger.info("동기화 시작: %s → %s", DATA_DIR, RAILWAY_URL)

    ok = fail = skip = 0
    for f in SYNC_FILES:
        if (DATA_DIR / f).exists():
            if sync_file(f):
                ok += 1
            else:
                fail += 1
        else:
            skip += 1

    logger.info("동기화 완료: 성공 %d / 실패 %d / 스킵 %d", ok, fail, skip)

    # Railway API 동기화 후, git에도 데이터 커밋 (재배포 대비)
    git_commit_data()


if __name__ == "__main__":
    main()
