"""risk/account_peak.py — 계좌 고점(equity peak) 영속 추적 + G8 사다리 평가. Phase 3c.

DD = (current_equity - peak) / peak. peak는 단조 증가(고점 갱신). 직전 사다리 step도 함께
영속(히스테리시스 — 다음 평가의 prev_step). nonce_store와 같은 graceful 파일 영속 패턴.

graceful degradation: 파일 글리치 시 기본값(peak=0→current가 곧 peak, step=0)으로 흘려보낸다.
영속 실패가 모든 매수를 막는 가용성 붕괴보다 낫다(진짜 fail-closed 백스톱은 킬스위치 L4).
"""
from __future__ import annotations

import json
from pathlib import Path

from risk.config import RISK_CONFIG, RiskConfig
from risk.drawdown_ladder import LadderState, ladder_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PEAK_PATH = PROJECT_ROOT / "data" / "risk" / "equity_peak.json"


class EquityPeakStore:
    """계좌 고점(단조 증가) + 직전 사다리 step 영속. update_and_ladder()가 한 번에 갱신·평가·저장."""

    def __init__(self, path: Path = DEFAULT_PEAK_PATH):
        self.path = Path(path)

    def _load(self) -> tuple[float, int]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return float(d.get("peak", 0.0) or 0.0), int(d.get("step", 0) or 0)
        except (OSError, ValueError, TypeError):
            return 0.0, 0

    def _save(self, peak: float, step: int) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"peak": peak, "step": step}), encoding="utf-8")
        except OSError:
            pass  # graceful — 영속 실패가 매수를 막지 않는다(백스톱=킬스위치)

    def update_and_ladder(
        self, current_equity: float, cfg: RiskConfig = RISK_CONFIG,
    ) -> LadderState:
        """현재 자본으로 고점 갱신 + DD 계산 + 사다리 평가(히스테리시스). 영속 저장 후 반환.

        peak=0(이력 없음)이면 current가 곧 고점 → DD 0(정상). 신규 계좌/첫 평가에서 과차단 방지.
        """
        prev_peak, prev_step = self._load()
        cur = float(current_equity)
        peak = max(prev_peak, cur)
        dd = (cur - peak) / peak if peak > 0 else 0.0
        ls = ladder_state(dd, prev_step, cfg)
        self._save(peak, ls.step)
        return ls
