# -*- coding: utf-8 -*-
"""
fetch_theme_intel.py — Supabase에서 테마 인텔리전스 읽기
========================================================
단타봇이 Supabase에 업로드한 quant_theme_flow / quant_theme_momentum을
읽어서 로컬 data/theme_intel.json에 저장.

다른 퀀트봇 스크립트(build_brain_upload, scan_* 등)가 이 JSON을 참조한다.

Usage:
    python scripts/fetch_theme_intel.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def get_supabase():
    """Supabase 클라이언트 생성."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.error("SUPABASE_URL/KEY 미설정")
        return None
    from supabase import create_client
    return create_client(url, key)


def fetch_latest(client, table: str) -> dict | None:
    """테이블에서 최신 1건 조회."""
    try:
        resp = client.table(table).select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            return resp.data[0]
        logger.warning("[%s] 데이터 없음", table)
        return None
    except Exception as e:
        logger.error("[%s] 조회 실패: %s", table, e)
        return None


def main():
    client = get_supabase()
    if not client:
        return

    theme_flow = fetch_latest(client, "quant_theme_flow")
    theme_momentum = fetch_latest(client, "quant_theme_momentum")

    intel = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "theme_flow": None,
        "theme_momentum": None,
        "hot_themes": [],
        "cold_themes": [],
        "top_inflow": [],
        "top_outflow": [],
        "signal": "",
        "rotation_signal": "",
    }

    if theme_flow:
        intel["theme_flow"] = theme_flow
        intel["top_inflow"] = theme_flow.get("top_inflow", [])
        intel["top_outflow"] = theme_flow.get("top_outflow", [])
        intel["signal"] = theme_flow.get("signal", "")
        flow_date = theme_flow.get("date", "N/A")
        total = theme_flow.get("total_themes", 0)
        logger.info("[theme_flow] date=%s, %d개 테마", flow_date, total)
    else:
        logger.warning("[theme_flow] 데이터 없음 — 단타봇 미업로드 또는 테이블 미생성")

    if theme_momentum:
        intel["theme_momentum"] = theme_momentum
        intel["hot_themes"] = theme_momentum.get("hot_themes", [])
        intel["cold_themes"] = theme_momentum.get("cold_themes", [])
        intel["rotation_signal"] = theme_momentum.get("rotation_signal", "")
        mom_date = theme_momentum.get("date", "N/A")
        total = theme_momentum.get("total_themes", 0)
        logger.info("[theme_momentum] date=%s, %d개 테마, HOT=%d, COLD=%d",
                    mom_date, total,
                    len(intel["hot_themes"]),
                    len(intel["cold_themes"]))
    else:
        logger.warning("[theme_momentum] 데이터 없음 — 단타봇 미업로드 또는 테이블 미생성")

    # 저장
    out = DATA / "theme_intel.json"
    DATA.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(intel, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("저장: %s (%s bytes)", out, out.stat().st_size)

    # 요약 출력
    if intel["hot_themes"]:
        print(f"\n🔥 HOT 테마: {', '.join(intel['hot_themes'][:10])}")
    if intel["cold_themes"]:
        print(f"🧊 COLD 테마: {', '.join(intel['cold_themes'][:5])}")
    if intel["top_inflow"]:
        print(f"💰 자금유입 TOP: {', '.join(intel['top_inflow'][:5])}")
    if intel["signal"]:
        print(f"📊 시그널: {intel['signal']}")


if __name__ == "__main__":
    main()
