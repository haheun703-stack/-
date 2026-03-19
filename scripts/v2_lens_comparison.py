"""STEP 7-8: V2 vs V2+LENS 백테스트 비교

LENS asymmetry 효과 검증:
- V1: 기존 (lens_enabled=false)
- V2+LENS: 레짐별 동적 min_rr (BULL:1.2, CAUTION:1.5, BEAR:2.0, CRISIS:3.0)
"""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def run_backtest(label: str, overrides: dict) -> dict:
    """임시 config로 백테스트 실행."""
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 오버라이드 적용
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    # 임시 config 파일
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, allow_unicode=True)
    tmp.close()

    try:
        from src.backtest_engine import BacktestEngine

        engine = BacktestEngine(tmp.name)
        data = engine.load_data()
        if not data:
            print(f"  [{label}] 데이터 없음!")
            return {}
        result = engine.run(data)
        stats = result.get("stats", {})
        return {
            "label": label,
            "pf": stats.get("profit_factor", 0),
            "return": stats.get("total_return_pct", 0),
            "mdd": stats.get("max_drawdown_pct", 0),
            "trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0),
            "sharpe": stats.get("sharpe_ratio", 0),
            "avg_hold": stats.get("avg_hold_days", 0),
        }
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main():
    print("=" * 60)
    print("STEP 7-8: V2 vs V2+LENS 백테스트 비교")
    print("=" * 60)

    # V1: LENS OFF (기존)
    print("\n[1/2] V1 (LENS OFF)...")
    v1 = run_backtest("V1_baseline", {
        "alpha_v2.lens_enabled": False,
    })

    # V2: LENS ON (asymmetry 적용)
    print("[2/2] V2+LENS (asymmetry ON)...")
    v2 = run_backtest("V2_LENS", {
        "alpha_v2.lens_enabled": True,
    })

    # 결과 비교
    print("\n" + "=" * 60)
    print(f"{'':15s} {'V1(baseline)':>14s} {'V2+LENS':>14s} {'delta':>10s}")
    print("-" * 60)

    metrics = [
        ("PF", "pf", ".2f"),
        ("Return %", "return", ".1f"),
        ("MDD %", "mdd", ".1f"),
        ("Trades", "trades", "d"),
        ("Win Rate %", "win_rate", ".1f"),
        ("Sharpe", "sharpe", ".2f"),
        ("Avg Hold", "avg_hold", ".1f"),
    ]

    for name, key, fmt in metrics:
        a = v1[key]
        b = v2[key]
        delta = b - a
        sign = "+" if delta > 0 else ""
        print(f"{name:15s} {a:>14{fmt}} {b:>14{fmt}} {sign}{delta:>9{fmt}}")

    # PASS/FAIL 판정
    print("\n" + "=" * 60)
    checks = []

    # 절대 기준
    if v2["pf"] >= 1.3:
        checks.append(("PF >= 1.3", True, f"{v2['pf']:.2f}"))
    else:
        checks.append(("PF >= 1.3", False, f"{v2['pf']:.2f}"))

    if v2["mdd"] >= -15:
        checks.append(("MDD >= -15%", True, f"{v2['mdd']:.1f}%"))
    else:
        checks.append(("MDD >= -15%", False, f"{v2['mdd']:.1f}%"))

    # 상대 기준 (V1 대비 악화 방어)
    if v2["pf"] >= v1["pf"] * 0.9:
        checks.append(("PF 10%이상 악화 없음", True, f"{v2['pf']:.2f} vs {v1['pf']:.2f}"))
    else:
        checks.append(("PF 10%이상 악화 없음", False, f"{v2['pf']:.2f} vs {v1['pf']:.2f}"))

    all_pass = all(ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}: {detail}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n최종: {verdict}")

    # 결과 저장
    out = {"v1": v1, "v2_lens": v2, "verdict": verdict}
    out_path = PROJECT_ROOT / "data" / "v2_migration" / "v2_lens_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
