"""STEP 6-2: 실제 BacktestEngine 기반 V1 vs V2 비교

게이트/트리거가 적용된 실제 백테스트 엔진으로 비교:
  V1: use_unified_scorer=false (기존 스코어링)
  V2: use_unified_scorer=true  (4팩터 재채점)

실행:
  python -u -X utf8 scripts/v2_engine_comparison.py
"""

import json
import logging
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_with_config(config: dict, label: str) -> dict:
    """임시 설정 파일로 BacktestEngine 실행."""
    from src.backtest_engine import BacktestEngine

    # 임시 설정 파일 생성
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    )
    yaml.dump(config, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()

    try:
        logger.info("[%s] 백테스트 시작...", label)
        engine = BacktestEngine(config_path=tmp.name)
        data = engine.load_data()

        if not data:
            logger.error("[%s] 데이터 없음", label)
            return {}

        logger.info("[%s] %d종목 로딩 완료", label, len(data))
        results = engine.run(data)
        return results
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("STEP 6-2: 실제 BacktestEngine V1 vs V2 비교")
    logger.info("=" * 70)

    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # --- V1 (기존 스코어링) ---
    logger.info("\n--- V1 (기존 스코어링) ---")
    cfg_v1 = deepcopy(base_config)
    cfg_v1.setdefault("alpha_v2", {})["use_unified_scorer"] = False
    r_v1 = run_with_config(cfg_v1, "V1")

    # --- V2 (4팩터 재채점) ---
    logger.info("\n--- V2 (4팩터 재채점) ---")
    cfg_v2 = deepcopy(base_config)
    cfg_v2.setdefault("alpha_v2", {})["enabled"] = True
    cfg_v2["alpha_v2"]["use_unified_scorer"] = True
    r_v2 = run_with_config(cfg_v2, "V2")

    if not r_v1 or not r_v2:
        logger.error("결과 부족 — 비교 불가")
        return

    # --- 비교 출력 ---
    m_v1 = r_v1.get("stats", {})
    m_v2 = r_v2.get("stats", {})

    print("\n" + "=" * 70)
    print("  실제 BacktestEngine — V1 vs V2 비교")
    print("=" * 70)

    keys = [
        ("total_return_pct", "수익률(%)"),
        ("sharpe_ratio", "Sharpe"),
        ("profit_factor", "PF"),
        ("max_drawdown_pct", "MDD(%)"),
        ("win_rate", "승률(%)"),
        ("total_trades", "거래(건)"),
        ("avg_hold_days", "보유(일)"),
        ("avg_win_pct", "평균수익(%)"),
        ("avg_loss_pct", "평균손실(%)"),
    ]

    print(f"  {'지표':<14} {'V1(기존)':>12} {'V2(4팩터)':>12}  {'차이':>10}")
    print("  " + "-" * 55)

    for key, label in keys:
        v1_val = m_v1.get(key, 0) or 0
        v2_val = m_v2.get(key, 0) or 0
        if isinstance(v1_val, int):
            v1_val = float(v1_val)
        if isinstance(v2_val, int):
            v2_val = float(v2_val)
        diff = v2_val - v1_val
        sign = "+" if diff > 0 else ""
        print(f"  {label:<14} {v1_val:>12.2f} {v2_val:>12.2f}  {sign}{diff:>9.2f}")

    print("=" * 70)

    # 판정
    v2_pf = float(m_v2.get("profit_factor", 0) or 0)
    v1_pf = float(m_v1.get("profit_factor", 0) or 0)
    v2_mdd = float(m_v2.get("max_drawdown_pct", -999) or -999)
    v1_mdd = float(m_v1.get("max_drawdown_pct", -999) or -999)
    v2_ret = float(m_v2.get("total_return_pct", 0) or 0)
    v1_ret = float(m_v1.get("total_return_pct", 0) or 0)

    abs_pass = v2_pf >= 1.3 and v2_mdd >= -15.0
    print(f"\n  절대 기준: PF >= 1.3 AND MDD >= -15% → {'PASS' if abs_pass else 'FAIL'}")
    print(f"  PF: V2({v2_pf:.2f}) vs V1({v1_pf:.2f}) → {'V2 우위' if v2_pf >= v1_pf else 'V1 유지'}")
    print(f"  MDD: V2({v2_mdd:.1f}%) vs V1({v1_mdd:.1f}%) → {'V2 우위' if v2_mdd >= v1_mdd else 'V1 유지'}")
    print(f"  수익: V2({v2_ret:.1f}%) vs V1({v1_ret:.1f}%) → {'V2 우위' if v2_ret >= v1_ret else 'V1 유지'}")

    # 결과 저장
    output_dir = PROJECT_ROOT / "data" / "v2_migration"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "v1_engine": {k: m_v1.get(k, 0) for k, _ in keys},
        "v2_engine": {k: m_v2.get(k, 0) for k, _ in keys},
    }

    with open(output_dir / "v2_engine_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info("저장: %s", output_dir / "v2_engine_comparison.json")


if __name__ == "__main__":
    main()
