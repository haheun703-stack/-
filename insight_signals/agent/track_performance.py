# -*- coding: utf-8 -*-
"""누적 픽 성과 평가 — 진입점.

실행:
    python -u -X utf8 -m insight_signals.agent.track_performance

산출물:
    data/insight_signals/performance_YYYY-MM-DD.csv
    (일일 리포트에도 요약이 자동 포함되므로 필요할 때만 단독 실행)
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from insight_signals.adapters import price_client                  # noqa: E402
from insight_signals.adapters.kis_flow_client import KisFlowClient  # noqa: E402
from insight_signals.agent import _env                              # noqa: E402
from insight_signals.use_cases import evaluate                      # noqa: E402

log = logging.getLogger("insight_signals.track")


def main() -> int:
    root = _env.project_root()
    _env.setup_logging(root, "insight_signals_track")
    _env.load_dotenv_manual(root)
    cfg = _env.load_config(root)

    data_dir = os.path.join(root, cfg["paths"]["data_dir"])
    picks_log = os.path.join(data_dir, "picks_log.csv")
    env_names = cfg["env_names"]
    kis = KisFlowClient(
        app_key=os.environ.get(env_names["kis_app_key"], ""),
        app_secret=os.environ.get(env_names["kis_app_secret"], ""),
        cache_dir=data_dir,
        base_url=cfg["flow"]["kis_base_url"],
        token_cache_path=cfg["flow"].get("kis_token_cache", ""),
    )

    result = evaluate.evaluate(
        picks_log, price_fn=lambda c: price_client.get_price(c, kis_client=kis)
    )
    rows = result["rows"]
    if not rows:
        log.info("평가할 픽이 아직 없습니다 (최소 +5일 경과 필요)")
        return 0

    today = dt.date.today().isoformat()
    out_path = os.path.join(data_dir, f"performance_{today}.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log.info("성과 평가 %d건 저장: %s", len(rows), out_path)
    for h, s in sorted(result["summary"].items()):
        log.info("  +%d일: n=%d 평균 %+.2f%% 승률 %.1f%%", h, s["n"], s["avg"], s["win_rate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
