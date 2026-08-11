"""정보봇 위험감지 SDK 래퍼 (P0-7 통합).

사용:
    from src.utils.risk_gate import get_risk_gate
    rg = get_risk_gate()
    multiplier = rg.get_position_multiplier()  # 0.2~1.0
    if rg.should_block_new_entry():  # CRISIS 차단
        return

데이터 출처: Supabase macro_risk_daily (정보봇 매일 16:49 갱신)
원본 SDK: src/utils/risk_gate_client.py (5/16 정보봇 → 퀀트봇)
가이드: jgis/docs/[정보봇 → 퀀트봇] 위험감지시스템 통합 가이드.md
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.utils.risk_gate_client import RiskGateClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Singleton (모듈 import 시 1회 초기화)
_INSTANCE: Optional[RiskGateClient] = None


def get_risk_gate() -> Optional[RiskGateClient]:
    """싱글톤 RiskGateClient 반환. Supabase 환경변수 누락 시 None.

    Fail-safe: 호출 측에서 None 체크 후 NORMAL 가정.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        logger.warning("[risk_gate] SUPABASE_URL/KEY 없음 → 위험감지 비활성 (NORMAL fallback)")
        return None

    try:
        _INSTANCE = RiskGateClient(
            supabase_url=url,
            supabase_key=key,
            bot_name="quant",
            cache_minutes=10,
        )
        logger.info("[risk_gate] 초기화 OK (bot_name=quant)")
        return _INSTANCE
    except Exception as e:
        logger.error(f"[risk_gate] 초기화 실패: {e}")
        return None


#: 두 리스크 체계의 보수성 비교용 순위. 자체 레짐(BULL/CAUTION/BEAR/CRISIS)과
#: 정보봇 등급(NORMAL/CAUTION/WARNING/DANGER/CRISIS)을 한 축에 세운다.
_SEVERITY_RANK = {
    "BULL": 0, "NORMAL": 0,
    "CAUTION": 1,
    "BEAR": 2, "WARNING": 2,
    "DANGER": 3,
    "CRISIS": 4,
}


def _local_regime_view() -> tuple[Optional[str], Optional[float]]:
    """자체 레짐 파일에서 (레짐, **그 체계가 스스로 낸** 포지션 배수).

    ★임의 매핑을 만들지 않는다 — 자체 레짐을 정보봇 등급표(CRISIS=0.2 등)에
    끼워 넣으면 근거 없는 숫자가 사이징에 들어간다. 각 체계가 이미 산출한
    값만 쓰고, 비교는 아래 교차검증에서 한다.
    """
    try:
        path = PROJECT_ROOT / "data" / "regime_macro_signal.json"
        if not path.exists():
            return None, None
        import json
        d = json.loads(path.read_text(encoding="utf-8"))
        regime = next(
            (str(d[k]) for k in ("current_regime", "kospi_regime", "regime")
             if d.get(k)), None,
        )
        mult = d.get("position_multiplier")
        # ★bool 배제: 파이썬에서 `isinstance(False, int)`는 True다. 채택을 철회한
        # 지금은 로그 표기용이지만, 되살릴 때 `float(False)=0.0`이 사이징 0으로
        # 새는 경로라 여기서 막아둔다(검수 재현 확인).
        ok = isinstance(mult, (int, float)) and not isinstance(mult, bool)
        return regime, (float(mult) if ok else None)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[risk_gate] 자체 레짐 조회 실패(무시): {e}")
        return None, None


def get_position_multiplier_safe() -> float:
    """매수금액 곱하기 계수. SDK 실패 시 1.0 (NORMAL).

    ★★8/11 신설 — **보수적 우선 교차검증**(퐝가님 승인). 사이징을 지배하는
    `macro_risk_daily`는 정보봇 산출물인데, 그 점수 체계에 **국내 시장의
    실현변동성 항목이 없다**(구성: MSCI 이벤트·DXY·US 2Y·미국 VIX·외인 플로우·
    디커플링). 그래서 8/11 실측에서 국내 20일 실현변동성이 **96.8% = 과거 3년
    99.9퍼센타일**(중앙값 17.9%)인데도 등급은 `NORMAL`, 배수 1.0, 신규진입
    차단 없음이었다. 같은 날 우리 자체 레짐은 **CRISIS 5일 연속**이었다.
    **두 체계가 정반대 답을 내는데 집행은 느슨한 쪽을 따르고 있었다.**

    조치: 두 체계가 **각자 산출한 배수 중 낮은 쪽**을 쓴다. 자체 레짐을
    정보봇 등급표에 임의로 매핑하지 않는 이유는 그 숫자에 근거가 없기
    때문이다(국내 RV를 반영한 새 배수 체계는 백테스트 선행 — 별건).
    등급 차가 2단계 이상 벌어지면 경고를 남겨 불일치가 보이게 한다.
    """
    rg = get_risk_gate()
    base = 1.0
    if rg is not None:
        try:
            base = rg.get_position_multiplier()
        except Exception as e:
            logger.warning(f"[risk_gate] multiplier 조회 실패 → 1.0: {e}")
            base = 1.0

    # ── 불일치 감지만 한다(채택은 하지 않는다) ──
    # ★★8/11 저녁 철회: 최초 구현은 자체 레짐의 `position_multiplier`와
    # `min()`을 취해 "보수적 쪽 채택"을 했는데, **두 값은 서로 다른 축이었다.**
    #   · 이 함수의 반환값 = 정보봇 등급 기반 **매수금액 계수**(0.2~1.0)
    #   · regime_macro_signal.position_multiplier = **최종 점수 배수**
    #     (`scan_tomorrow_picks.py:24` "최종 점수 × position_multiplier", 0.5~**1.3**)
    # 축이 다르므로 min()은 의미가 없고, 실제로 `macro_score>=45`인 날은 그 값이
    # 1.0~1.3이라 **보호가 0인데도 "보수적 쪽 채택" 로그만 남는다**(검수 재현).
    # 게다가 그 값은 레짐이 아니라 macro_score만으로 정해져(`regime_macro_signal.py`
    # GRADE_MAP) CRISIS에 반응하지도 않는다 — 5일 연속 CRISIS인데 0.9였던 이유다.
    # 8/1 켈리 `f*`를 수량배수로 쓴 것과 같은 계열의 의미론 오류라 채택을 철회한다.
    # **판정이 갈리는 날을 드러내는 경고는 유지**한다 — 그건 사실이고 유용하다.
    # 국내 실현변동성을 반영한 사이징은 백테스트 선행(B-63 근본).
    regime, local_mult = _local_regime_view()
    if regime is None:
        return base
    try:
        gate_level = rg.get_current_level() if rg is not None else "NORMAL"
    except Exception:  # noqa: BLE001
        gate_level = "NORMAL"
    gap = (_SEVERITY_RANK.get(str(regime).upper(), 0)
           - _SEVERITY_RANK.get(str(gate_level).upper(), 0))
    if gap >= 2:
        logger.warning(
            "[risk_gate] 리스크 판정 불일치 — 자체 레짐 %s vs 정보봇 등급 %s "
            "(%d단계). 사이징은 정보봇 계수 %.2f를 그대로 쓴다"
            "(자체 배수 %s는 점수 축이라 혼용 불가 — B-63 근본 조치 대기)",
            regime, gate_level, gap, base,
            f"{local_mult:.2f}" if local_mult is not None else "n/a",
        )

    return base


def should_block_new_entry_safe() -> bool:
    """신규 진입 차단 여부. SDK 실패 시 False (정상 진입)."""
    rg = get_risk_gate()
    if rg is None:
        return False
    try:
        return rg.should_block_new_entry()
    except Exception:
        return False


def get_risk_status_safe() -> dict:
    """현재 위험 상태 전체 (로그/알림용). 실패 시 빈 dict."""
    rg = get_risk_gate()
    if rg is None:
        return {}
    try:
        return rg.get_full_status() or {}
    except Exception as e:
        logger.warning(f"[risk_gate] status 조회 실패: {e}")
        return {}


def get_exhaustion_threshold() -> Optional[float]:
    """현재 등급별 외인소진율 차단 임계값 (높을수록 위험).

    정상: None (차단 없음)
    주의: 80%+
    경고: 60%+
    위험: 50%+
    위기: 30%+
    """
    rg = get_risk_gate()
    if rg is None:
        return None
    try:
        level = rg.get_current_level()
        thresholds = {
            "NORMAL": None,
            "CAUTION": 80.0,
            "WARNING": 60.0,
            "DANGER": 50.0,
            "CRISIS": 30.0,
        }
        return thresholds.get(level)
    except Exception:
        return None
