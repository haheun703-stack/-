"""정보봇 kr_us_shock_summary shadow 관측 로더 — 매매 미반영 (freeze).

정보봇(jgis)이 daily_intelligence.json["kr_us_shock_summary"]에 적재하는 한미 충격
비교 요약(환율·금리 → 한국 vs 미국 취약도 + 드라이버 3건)을 읽어
regime_macro_signal.json에 shadow 필드로 기록한다.

★중요: 레짐 / macro_score / SHIELD 에 절대 반영하지 않는다 (관측 전용).
  [[jgis_regime_fact_adapter]] 와 동일 정책 — 1단계 shadow 관측, N거래일(≥10) 검증 후
  freeze 해제 하에 매매(레짐/방어) 반영. (2026-06-27 사장님 '매매/레짐 반영 배선' 결정의
  정석 1단계: 데이터 6/26 1건뿐 → 백테스트 불가 → shadow 선수신, 누적 후 활성화)

데이터 계약 (정보봇 shared_pipeline_service._build_kr_us_shock_summary, 2026-06-27 코드 실측):
  {date, kr_shock(한국취약도), us_shock(미국취약도), diff(=kr-us), verdict, drivers[:3]}
전달 경로: jgis_to_quant/daily_intelligence.json["kr_us_shock_summary"] (평일 08:00 적재)
  ※ kr_us_shock 첫 산출=6/26 16:49 → daily_intelligence 첫 반영은 6/29(월) 08:00.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 정보봇 → 퀀트봇 공유 채널 (jgis_regime_fact_adapter 와 동일 경로 규칙)
JGIS_SHARED_DIR = (
    Path("/home/ubuntu/shared-bot-data/jgis_to_quant") if os.name == "posix"
    else Path("D:/shared-bot-data/jgis_to_quant")
)
DAILY_INTEL_FILE = JGIS_SHARED_DIR / "daily_intelligence.json"

# 데이터 계약 6필드 (정보봇 _build_kr_us_shock_summary 반환 키)
_SHOCK_FIELDS = ("date", "kr_shock", "us_shock", "diff", "verdict", "drivers")


def load_kr_us_shock_shadow(path: Path = DAILY_INTEL_FILE) -> dict:
    """정보봇 kr_us_shock_summary 를 shadow 관측용으로 읽는다 (graceful).

    graceful:
      - 파일 없음 → loaded=False, status=no_file
      - 읽기 오류 → loaded=False, status=read_error
      - 필드 없음(정보봇 미적재, 월요일 08:00 전) → loaded=False, status=no_field
      - 정상 → loaded=True, 6필드 + intel_date 기록

    ★반환 dict 는 관측 전용. 호출처(regime_macro_signal)는 이것을 레짐/macro_score/
      SHIELD 에 절대 반영하지 않고 출력 필드로만 둔다 (freeze).
    """
    shadow: dict = {
        "loaded": False,
        "note": "SHADOW 관측 전용 — 레짐/macro_score/SHIELD 미반영 (freeze)",
    }
    try:
        if not path.exists():
            shadow["status"] = "no_file"
            return shadow
        with open(path, encoding="utf-8") as f:
            intel = json.load(f)
    except Exception as e:  # noqa: BLE001 — 관측 로더는 어떤 실패도 매매에 영향 0
        logger.warning("[kr_us_shock_shadow] 읽기 실패: %s", e)
        shadow["status"] = "read_error"
        return shadow

    summ = intel.get("kr_us_shock_summary")
    if not summ:  # 정보봇이 아직 필드 미적재 (graceful — 월요일 08:00 첫 반영)
        shadow["status"] = "no_field"
        shadow["intel_date"] = intel.get("date")
        return shadow

    shadow["loaded"] = True
    shadow["status"] = "ok"
    shadow["intel_date"] = intel.get("date")
    for k in _SHOCK_FIELDS:
        shadow[k] = summ.get(k)
    return shadow


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(load_kr_us_shock_shadow(), ensure_ascii=False, indent=2))
