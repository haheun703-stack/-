"""정보봇 regime_macro_fact.json shadow 관측 로더 — 매매 미반영 (freeze).

정보봇(jgis)이 `jgis_to_quant/regime_macro_fact.json`에 적재한 거시 국면 fact를
읽어 `regime_macro_signal.json`에 shadow 필드로 기록한다.

★중요: macro_score / position_multiplier 에 절대 반영하지 않는다 (관측 전용).
회신서 [퀀트봇 → 정보봇] 2026-06-22 §C 약속 — 1단계는 shadow 관측만,
N거래일(≥10) 검증 후 실제 7축 배선은 freeze 해제 하에 진행.

읽기 경로는 퀀트봇의 기존 jgis 채널(scenario_detector.py:42)과 동일한 OS 분기.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 정보봇 → 퀀트봇 공유 채널 (scenario_detector.py 와 동일 경로 규칙)
JGIS_SHARED_DIR = (
    Path("/home/ubuntu/shared-bot-data/jgis_to_quant") if os.name == "posix"
    else Path("D:/shared-bot-data/jgis_to_quant")
)
REGIME_FACT_FILE = JGIS_SHARED_DIR / "regime_macro_fact.json"

# 회신서 §A 정의 4필드 (정보봇 제공 예정 — graceful 추출)
_FACT_FIELDS = (
    "macro_regime_bias",      # EXPANSION / NEUTRAL / CONTRACTION
    "say_do_divergence",      # 말-돈 괴리 (air pocket 조기경보)
    "bear_pressure_score",    # 0~100 베어 압력
    "theme_money_quality",    # 테마별 말=돈 품질
)


def load_regime_fact_shadow(path: Path = REGIME_FACT_FILE) -> dict:
    """정보봇 regime_macro_fact 를 shadow 관측용으로 읽는다.

    graceful:
      - 파일 없음 → loaded=False, status=no_file
      - 읽기 오류 → loaded=False, status=read_error
      - 미확정(is_final=False) → loaded=False, status=not_final (finality 가드, 회신서 §B)
      - 정상 → loaded=True, 4필드 + snapshot_time 기록

    ★반환 dict 는 관측 전용. 호출처(regime_macro_signal)는 이것을 macro_score 에
      절대 더하지 않고 출력 필드로만 둔다.
    """
    shadow: dict = {
        "loaded": False,
        "note": "SHADOW 관측 전용 — macro_score 미반영 (freeze, 회신서 §C)",
    }
    try:
        if not path.exists():
            shadow["status"] = "no_file"
            return shadow
        with open(path, encoding="utf-8") as f:
            fact = json.load(f)
    except Exception as e:  # noqa: BLE001 — 관측 로더는 어떤 실패도 매매에 영향 0
        logger.warning("[regime_fact_shadow] 읽기 실패: %s", e)
        shadow["status"] = "read_error"
        return shadow

    # finality 가드 (회신서 §B): 미확정 fact 는 관측에서도 제외
    if fact.get("is_final") is False:
        shadow["status"] = "not_final"
        shadow["fact_snapshot_time"] = fact.get("snapshot_time")
        return shadow

    shadow["loaded"] = True
    shadow["status"] = "ok"
    shadow["fact_date"] = fact.get("date")
    shadow["fact_snapshot_time"] = fact.get("snapshot_time")
    for k in _FACT_FIELDS:
        shadow[k] = fact.get(k)
    return shadow


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    result = load_regime_fact_shadow()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 파일 없을 때 graceful 확인용 스모크
    sys.exit(0)
