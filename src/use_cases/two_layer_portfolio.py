"""src/use_cases/two_layer_portfolio.py — 2층 구조 포트 현황 + 드로다운 알림 골격.

설계서(docs/02-design/two_layer_portfolio_design_2026-06-15.md) 확정 구조를 대시보드
2테이블(데이터계약 §2 dashboard_two_layer / §3 dashboard_drawdown_alert) 행으로 변환.

  코어 82%(밸류밴드 우량주) + 새틀라이트 18%(미국 레버리지 B 상시 SOXL/TQQQ/NVDL)
  + 드로다운 −15% 알림(자동대응 X / 수동결정 — 리먼급 판별 대시보드)

★골격(관측) 단계: 누적수익률·MDD·현재DD 등 실데이터는 포트 실운용(unfreeze·페이퍼20일) 후
  채워진다. 지금은 설계 상수 + None(실데이터 자리) + level=normal. 매매로직 0·freeze 무손상.
순수 함수만 — I/O 없음(적재는 scripts/upload_two_layer.py).
"""
from __future__ import annotations

# 설계 확정값(SSoT = 설계서 §1). 변경 시 설계서와 동기화.
CORE_PCT = 82.0
SATELLITE_PCT = 18.0
DD_ALERT_THRESHOLD = -15.0  # 계좌 고점 대비 −15% → alert 전개

# 새틀라이트 = 미국 레버리지 B 상시(한국은 예탁금 1,000만 장벽 → 별첨). 비중은 unfreeze 시 미세조정.
SATELLITE_DETAIL = [
    {"ticker": "SOXL", "name": "반도체 3x", "weight": None, "return": None},
    {"ticker": "TQQQ", "name": "나스닥100 3x", "weight": None, "return": None},
    {"ticker": "NVDL", "name": "엔비디아 2x", "weight": None, "return": None},
]


def classify_dd_level(current_dd: float | None) -> str:
    """현재 드로다운 → 알림 레벨. −15% 이상 하락이면 alert, 평소 normal.

    current_dd는 음수(예: −12.3). None(실데이터 전)이면 평소로 간주(normal).
    """
    if current_dd is not None and current_dd <= DD_ALERT_THRESHOLD:
        return "alert"
    return "normal"


def build_two_layer_row(
    date_str: str,
    snapshot_iso: str,
    cum_return: float | None = None,
    mdd: float | None = None,
    current_dd: float | None = None,
) -> dict:
    """dashboard_two_layer 행(데이터계약 §2). 실데이터(cum_return/mdd/current_dd)는 unfreeze 후."""
    return {
        "date": date_str,
        "core_pct": CORE_PCT,
        "satellite_pct": SATELLITE_PCT,
        "cum_return": cum_return,
        "mdd": mdd,
        "current_dd": current_dd,
        "satellite_detail": SATELLITE_DETAIL,
        "snapshot_time": snapshot_iso,
    }


def build_drawdown_alert_row(
    date_str: str,
    snapshot_iso: str,
    current_dd: float | None = None,
    history_analog: list | None = None,
    crisis_signals: dict | None = None,
    foreign_outflow: dict | None = None,
    port_exposure: dict | None = None,
    recommended_actions: list | None = None,
) -> dict:
    """dashboard_drawdown_alert 행(데이터계약 §3).

    ★평소(current_dd > −15)엔 level=normal(녹색)·JSONB 요약/None 허용. −15% 도달 시 전 필드 채움
      (stress_test 역사닮은꼴·매크로 위기신호·외인이탈·포트노출·권장행동) — unfreeze 후 실배선.
    """
    level = classify_dd_level(current_dd)
    verdict = None
    if level == "alert":
        # 실제 시스템위기/일반조정 판정은 알림 대시보드(stress_test+매크로) 구현 시 채움.
        verdict = "판정대기"
    return {
        "date": date_str,
        "current_dd": current_dd,
        "level": level,
        "verdict": verdict,
        "history_analog": history_analog,
        "crisis_signals": crisis_signals,
        "foreign_outflow": foreign_outflow,
        "port_exposure": port_exposure,
        "recommended_actions": recommended_actions,
        "snapshot_time": snapshot_iso,
    }
