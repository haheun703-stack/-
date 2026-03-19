"""LENS LAYER 실행 스크립트 (STEP 7-7)

BRAIN → LENS → scan_buy 순서로 실행됨.
brain_decision.json을 읽어 lens_context.json을 생성한다.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.alpha.lens_layer import LensLayer


def main():
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    layer = LensLayer(settings)
    ctx = layer.compute()

    gb = ctx.get("game_board", {})
    asym = ctx.get("asymmetry", {})
    flow = ctx.get("flow_map", {})

    print(f"LENS: {gb.get('mode', '?')} | {gb.get('reason', '')}")
    print(f"  R:R>={asym.get('min_rr_ratio', 0):.1f} | "
          f"target={asym.get('target_atr_mult', 0):.1f}x | "
          f"stop={asym.get('stop_atr_mult', 0):.1f}x")

    hot = flow.get("hot_sectors", [])
    cold = flow.get("cold_sectors", [])
    if hot:
        print(f"  HOT: {', '.join(hot)}")
    if cold:
        print(f"  COLD: {', '.join(cold)}")

    sv = ctx.get("structural_value", {})
    print(f"  밸류: {sv.get('valuation_mode', '?')} | "
          f"min_Q={sv.get('min_quality_score', 0):.1f} | "
          f"trap={sv.get('trap_filter', False)}")


if __name__ == "__main__":
    main()
