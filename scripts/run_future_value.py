"""미래가치 통합 엔진 러너 — data/shadow/future_value.json 산출 (관측 전용).

BAT-D 배선 후보 (검증 후). graceful — 어떤 입력이 없어도 exit 0 보장
(run_rss_theme_scan 선례: 옛 archive+sys.exit(1) 폐지 전례 방지).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_future_value")


def main() -> int:
    try:
        from src.use_cases.future_value_engine import OUTPUT_PATH, build_scorecards

        result = build_scorecards()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        cards = result.get("scorecards", [])
        logger.info("[FV] 산출 %d종목 | 국면=%s → 권장 %s | 축 커버리지 %s",
                    len(cards), result.get("regime"),
                    result.get("recommended_horizon"), result.get("axes_coverage"))
        for c in cards[:10]:
            logger.info("  %s(%s) 단기%.0f 중기%.0f 장기%.0f [%s] %s",
                        c["name"], c["ticker"], c["fv_short"], c["fv_mid"], c["fv_long"],
                        c["fv_best"], ",".join(c["tags"][:4]))
        logger.info("[FV] 저장: %s", OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001 — 관측 전용, BAT 절대 무손상
        logger.warning("[FV] 실패(관측 전용이라 무해): %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
