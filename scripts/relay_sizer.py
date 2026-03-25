"""[B] 릴레이 포지션 비중 계산.

패턴 신뢰도(승률) x 발화 강도 기반 투입 비중 자동 계산.

공식:
  base_weight = (win_rate - 50) / 50 * 20%   # 승률50%→0%, 100%→20%

  발화강도 보정 (선행 섹터 등락률 기반):
    EXTREME (+10%↑): x 1.5
    STRONG  (+7%↑) : x 1.2
    MODERATE(+5%↑) : x 1.0
    WEAK    (~3.5%): x 0.5

  최종 비중: max(3%, min(15%, base * 보정))
  투입 금액 = 총자산 * (1 - 현금유보율) * 비중

사용법:
  python scripts/relay_sizer.py --win-rate 70.6 --fire-return 14.0 --portfolio 30000000
"""

from __future__ import annotations

import argparse


# 발화 강도 등급 + 보정 계수
INTENSITY_GRADES = [
    ("EXTREME", 10.0, 1.5),
    ("STRONG",   7.0, 1.2),
    ("MODERATE",  5.0, 1.0),
    ("WEAK",      0.0, 0.5),
]

def _load_cash_reserve() -> float:
    """settings.yaml에서 cash_reserve_pct 로드 (없으면 0.20)."""
    try:
        from pathlib import Path
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("live_trading", {}).get("position", {}).get("cash_reserve_pct", 0.20)
    except Exception:
        return 0.20

CASH_RESERVE = _load_cash_reserve()
MAX_WEIGHT = 0.15      # 최대 15%
MIN_WEIGHT = 0.03      # 최소 3%


def grade_fire_intensity(lead_return: float) -> tuple[str, float]:
    """선행 섹터 등락률 → 발화 등급 + 보정 계수."""
    for grade, threshold, mult in INTENSITY_GRADES:
        if lead_return >= threshold:
            return grade, mult
    return "WEAK", 0.5


def calc_relay_weight(
    win_rate: float,
    lead_return: float,
    confidence: str = "HIGH",
) -> dict:
    """릴레이 포지션 비중 계산.

    Args:
        win_rate: 백테스트 승률 (%)
        lead_return: 선행 섹터 오늘 등락률 (%)
        confidence: 신뢰도 등급 (HIGH/MED/LOW)

    Returns:
        {weight_pct, grade, multiplier, base_weight, reason}
    """
    # 기본 비중: 승률 50%→0%, 100%→20%
    base = max(0, (win_rate - 50) / 50 * 20)

    # 발화 강도 보정
    grade, mult = grade_fire_intensity(lead_return)

    # 신뢰도 보정 (LOW면 추가 감점)
    if confidence == "LOW":
        mult *= 0.5
    elif confidence == "MED":
        mult *= 0.8

    # 최종 비중
    weight = base * mult
    weight = max(MIN_WEIGHT * 100, min(MAX_WEIGHT * 100, weight))

    reason_parts = [
        f"승률{win_rate:.0f}%→기본{base:.1f}%",
        f"발화{grade}({lead_return:+.1f}%)→x{mult:.1f}",
        f"신뢰도{confidence}",
    ]

    return {
        "weight_pct": round(weight, 1),
        "base_weight_pct": round(base, 1),
        "grade": grade,
        "multiplier": mult,
        "confidence": confidence,
        "reason": " | ".join(reason_parts),
    }


def calc_invest_amount(
    weight_pct: float,
    total_portfolio: int,
    cash_reserve: float = CASH_RESERVE,
) -> dict:
    """투입 금액 계산.

    Args:
        weight_pct: 비중 (%)
        total_portfolio: 총 포트폴리오 (원)
        cash_reserve: 현금 유보 비율 (기본 20%)

    Returns:
        {invest_amount, available, cash_reserved}
    """
    available = int(total_portfolio * (1 - cash_reserve))
    invest = int(available * weight_pct / 100)

    return {
        "invest_amount": invest,
        "available": available,
        "cash_reserved": int(total_portfolio * cash_reserve),
        "weight_pct": weight_pct,
        "total_portfolio": total_portfolio,
    }


def calc_full_sizing(
    win_rate: float,
    lead_return: float,
    confidence: str,
    total_portfolio: int,
) -> dict:
    """비중 + 금액 통합 계산."""
    w = calc_relay_weight(win_rate, lead_return, confidence)
    a = calc_invest_amount(w["weight_pct"], total_portfolio)
    return {**w, **a}


def main():
    parser = argparse.ArgumentParser(description="릴레이 포지션 비중 계산")
    parser.add_argument("--win-rate", type=float, required=True,
                        help="백테스트 승률 (%)")
    parser.add_argument("--fire-return", type=float, required=True,
                        help="선행 섹터 오늘 등락률 (%)")
    parser.add_argument("--confidence", default="HIGH",
                        choices=["HIGH", "MED", "LOW"],
                        help="백테스트 신뢰도 (기본 HIGH)")
    parser.add_argument("--portfolio", type=int, default=30_000_000,
                        help="총 포트폴리오 (기본 3000만원)")
    args = parser.parse_args()

    result = calc_full_sizing(
        args.win_rate, args.fire_return,
        args.confidence, args.portfolio,
    )

    print(f"\n{'=' * 50}")
    print(f"  릴레이 포지션 사이징")
    print(f"{'=' * 50}")
    print(f"  승률: {args.win_rate:.1f}%")
    print(f"  발화: {args.fire_return:+.1f}% ({result['grade']})")
    print(f"  신뢰도: {result['confidence']}")
    print(f"")
    print(f"  기본 비중: {result['base_weight_pct']:.1f}%")
    print(f"  보정 계수: x{result['multiplier']:.1f}")
    print(f"  최종 비중: {result['weight_pct']:.1f}%")
    print(f"")
    print(f"  총 자산: {result['total_portfolio']:>14,}원")
    print(f"  현금 유보: {result['cash_reserved']:>14,}원 (20%)")
    print(f"  투입 가능: {result['available']:>14,}원")
    print(f"  투입 금액: {result['invest_amount']:>14,}원")
    print(f"")
    print(f"  {result['reason']}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
