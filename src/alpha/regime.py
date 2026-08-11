"""
Alpha L1 REGIME Engine — Hysteresis 기반 레짐 판정

핵심 메커니즘:
- 백테스트: RegimeGate 출력 → Alpha 4등급 매핑
- 라이브: regime_macro_signal.json 직접 읽기
- Hysteresis: 업그레이드 3일 연속 확인, 다운그레이드 즉시
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import AlphaRegimeLevel, RegimeParams

logger = logging.getLogger(__name__)


class AlphaRegime:
    """통합 레짐 판정 엔진 (Hysteresis 적용)"""

    # RegimeGate → Alpha 매핑
    _GATE_TO_ALPHA = {
        "favorable": AlphaRegimeLevel.BULL,
        "neutral": AlphaRegimeLevel.CAUTION,
        "caution": AlphaRegimeLevel.BEAR,
        "hostile": AlphaRegimeLevel.CRISIS,
    }

    # KOSPI 레짐 → Alpha 매핑 (regime_macro_signal.json 용)
    _KOSPI_TO_ALPHA = {
        "BULL": AlphaRegimeLevel.BULL,
        "CAUTION": AlphaRegimeLevel.CAUTION,
        "BEAR": AlphaRegimeLevel.BEAR,
        "CRISIS": AlphaRegimeLevel.CRISIS,
    }

    # 등급 순위 (다운그레이드/업그레이드 판단용)
    _REGIME_RANK = {
        AlphaRegimeLevel.CRISIS: 0,
        AlphaRegimeLevel.BEAR: 1,
        AlphaRegimeLevel.CAUTION: 2,
        AlphaRegimeLevel.BULL: 3,
    }

    def __init__(self, config: dict):
        regime_cfg = config.get("alpha_engine", {}).get("regime", {})
        self._hysteresis_days = regime_cfg.get("hysteresis_upgrade_days", 3)
        self._config = config

        # 현재 확정 레짐 (초기값: CAUTION)
        self._current = AlphaRegimeLevel.CAUTION
        # 업그레이드 연속 카운터
        self._upgrade_count = 0
        # 업그레이드 후보 레짐
        self._upgrade_candidate: AlphaRegimeLevel | None = None

    def reset(self) -> None:
        """백테스트 시작 시 상태 초기화"""
        self._current = AlphaRegimeLevel.CAUTION
        self._upgrade_count = 0
        self._upgrade_candidate = None

    @property
    def current(self) -> AlphaRegimeLevel:
        return self._current

    def get_params(self) -> RegimeParams:
        """현재 레짐의 운영 파라미터"""
        return RegimeParams.from_regime(self._current, self._config)

    # ──────────────────────────────────────────────
    # 백테스트용: RegimeGate 결과 변환
    # ──────────────────────────────────────────────

    def detect_backtest(self, regime_state) -> AlphaRegimeLevel:
        """RegimeGate.detect() 결과 → Alpha 레짐 (Hysteresis 적용)

        Args:
            regime_state: RegimeGate.detect()가 반환한 RegimeState
                          (regime 필드: favorable/neutral/caution/hostile)
        """
        raw = self._GATE_TO_ALPHA.get(
            regime_state.regime, AlphaRegimeLevel.CAUTION,
        )
        return self._apply_hysteresis(raw)

    # ──────────────────────────────────────────────
    # 라이브용: JSON 파일에서 직접 읽기
    # ──────────────────────────────────────────────

    #: `regime_macro_signal.json`의 레짐 키 — 정본이 앞. 뒤는 과거/호환 표기.
    _LIVE_REGIME_KEYS = ("current_regime", "kospi_regime", "regime")

    def detect_live(self) -> AlphaRegimeLevel:
        """regime_macro_signal.json → Alpha 레짐 (Hysteresis 적용).

        ★★8/11 교정(B-60 ⑴) — `data.get("kospi_regime", "CAUTION")`이었는데
        이 JSON에 `kospi_regime` 키는 **존재하지 않는다**(정본은 `current_regime`).
        그래서 실제 레짐이 CRISIS인 날에도 매일 조용히 CAUTION을 반환했다.
        키 부재가 예외를 내지 않고 **하필 '안전해 보이는' 기본값으로 위장**돼
        로그도 알람도 뜨지 않았고, 그 값이 알파 엔진 VETO와 포지션 사이징까지
        내려갔다(8/11 실측: 파일 `current_regime`=CRISIS ↔ 이 함수 반환 CAUTION).

        그래서 ⑴ 정본 키를 앞에 둔 폴백 체인으로 읽고 ⑵ **어느 키도 못 찾으면
        경고를 남긴다** — 조용한 기본값이 이 버그를 63일간 숨긴 장본인이라,
        같은 상황이 다시 오면 이번엔 소리가 나야 한다.
        """
        raw = AlphaRegimeLevel.CAUTION  # 기본값

        try:
            path = Path("data/regime_macro_signal.json")
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                kospi_regime = next(
                    (str(data[k]) for k in self._LIVE_REGIME_KEYS
                     if data.get(k)), None,
                )
                if kospi_regime is None:
                    logger.warning(
                        "Alpha 레짐: %s에 레짐 키 없음(%s) — CAUTION으로 대체. "
                        "생산자 스키마 변경 여부 확인 필요",
                        path.name, "/".join(self._LIVE_REGIME_KEYS),
                    )
                elif kospi_regime not in self._KOSPI_TO_ALPHA:
                    logger.warning(
                        "Alpha 레짐: 알 수 없는 값 %r — CAUTION으로 대체",
                        kospi_regime,
                    )
                else:
                    raw = self._KOSPI_TO_ALPHA[kospi_regime]
        except Exception as e:
            logger.warning("Alpha 레짐 JSON 읽기 실패: %s", e)

        return self._apply_hysteresis(raw)

    # ──────────────────────────────────────────────
    # Hysteresis 적용
    # ──────────────────────────────────────────────

    def _apply_hysteresis(self, raw: AlphaRegimeLevel) -> AlphaRegimeLevel:
        """
        업그레이드: 동일 상위 레짐이 N일 연속 → 확정
        다운그레이드: 즉시 반영 (위험 대응은 빠르게)
        동일: 카운터 리셋
        """
        raw_rank = self._REGIME_RANK[raw]
        current_rank = self._REGIME_RANK[self._current]

        if raw_rank > current_rank:
            # 업그레이드 시도
            if self._upgrade_candidate == raw:
                self._upgrade_count += 1
            else:
                # 새로운 후보 등장 → 카운터 리셋
                self._upgrade_candidate = raw
                self._upgrade_count = 1

            if self._upgrade_count >= self._hysteresis_days:
                prev = self._current
                self._current = raw
                self._upgrade_count = 0
                self._upgrade_candidate = None
                logger.info(
                    "Alpha 레짐 업그레이드: %s → %s (%d일 확인)",
                    prev.value, raw.value, self._hysteresis_days,
                )

        elif raw_rank < current_rank:
            # 다운그레이드 → 즉시 반영
            prev = self._current
            self._current = raw
            self._upgrade_count = 0
            self._upgrade_candidate = None
            logger.info(
                "Alpha 레짐 다운그레이드: %s → %s (즉시)",
                prev.value, raw.value,
            )

        else:
            # 동일 레짐 유지
            self._upgrade_count = 0
            self._upgrade_candidate = None

        return self._current
