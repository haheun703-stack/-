"""
프레데터 모드 엔진 — "룰을 깨는 전략"
==========================================
3가지 룰 파괴:
  1. 모멘텀 가속도 → "다음 1위"를 선제 매수
  2. 확신 집중 → HIGH에 60%, MID에 30%, LOW에 10%
  3. 이벤트 트리거 → 격주 기본 + 트리거 시 즉시 대응
"""

from dataclasses import dataclass, field


@dataclass
class SectorAcceleration:
    """섹터 모멘텀 가속도."""
    sector: str
    code: str
    current_rank: int
    prev_rank: int         # 5일 전 순위
    rank_change: int       # prev - current (양수 = 순위 상승)
    rank_velocity: float   # 5일 가중 순위 변화 속도
    momentum_score: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    acceleration_score: float = 0.0


@dataclass
class ConvictionResult:
    """섹터 확신도 판정."""
    sector: str
    code: str
    level: str              # HIGH / MID / LOW / SKIP
    score: float            # 0~100
    weight_pct: float       # 배분 비중 (%)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EventTrigger:
    """이벤트 트리거."""
    triggered: bool = False
    trigger_type: str = ""   # SURGE / SMART_MONEY / REGIME_SHIFT / US_BULL
    sector: str = ""
    reason: str = ""


class PredatorEngine:
    """프레데터 모드 엔진."""

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        cfg = self.settings.get("predator", {})

        # 가속도 설정
        self.accel_lookback = cfg.get("accel_lookback", 5)         # 순위 변화 관찰 기간 (일)
        self.accel_weight_recent = cfg.get("accel_weight_recent", 0.6)  # 최근 가중

        # 확신도 설정
        self.conviction_thresholds = cfg.get("conviction_thresholds", {
            "HIGH": 70, "MID": 45, "LOW": 25,
        })
        # 확신도별 비중 배분 (섹터 전체 비중 대비 %)
        self.conviction_weights = cfg.get("conviction_weights", {
            "HIGH": 60, "MID": 30, "LOW": 10,
        })
        self.max_sector_holdings = cfg.get("max_sector_holdings", 3)

        # 이벤트 트리거 설정
        self.surge_threshold_pct = cfg.get("surge_threshold_pct", 5.0)
        self.smart_money_threshold_bil = cfg.get("smart_money_threshold_bil", 500)
        self.smart_money_consecutive_days = cfg.get("smart_money_consecutive_days", 3)

    # ─────────────────────────────────────────────
    # 1. 모멘텀 가속도
    # ─────────────────────────────────────────────
    def calc_acceleration(
        self,
        current_ranks: dict[str, dict],
        prev_ranks: dict[str, dict],
    ) -> list[SectorAcceleration]:
        """모멘텀 순위 변화 속도(가속도) 계산.

        Args:
            current_ranks: {sector: {rank, ret_5, ret_20, ret_60, momentum_score, code}}
            prev_ranks: 5일 전 동일 형식

        Returns:
            가속도 순으로 정렬된 SectorAcceleration 리스트
        """
        results = []
        for sector, cur in current_ranks.items():
            prev = prev_ranks.get(sector, {})
            cur_rank = cur.get("rank", 20)
            prev_rank = prev.get("rank", cur_rank)

            # 순위 변화 (양수 = 상승)
            rank_change = prev_rank - cur_rank

            # 가중 속도: 최근 변화에 더 큰 가중치
            # rank_change가 크고 + 모멘텀 스코어가 높으면 → 가속 중
            mom_score = cur.get("momentum_score", 50)
            rank_velocity = rank_change * self.accel_weight_recent + (mom_score / 100) * (1 - self.accel_weight_recent)

            # 가속도 점수 (0~100)
            # 순위 상승 + 단기 수익률 + 모멘텀 스코어 종합
            ret_5 = cur.get("ret_5", 0)
            accel_score = self._calc_accel_score(rank_change, ret_5, mom_score, cur_rank)

            results.append(SectorAcceleration(
                sector=sector,
                code=cur.get("code", ""),
                current_rank=cur_rank,
                prev_rank=prev_rank,
                rank_change=rank_change,
                rank_velocity=rank_velocity,
                momentum_score=mom_score,
                ret_5d=ret_5,
                ret_20d=cur.get("ret_20", 0),
                ret_60d=cur.get("ret_60", 0),
                acceleration_score=accel_score,
            ))

        results.sort(key=lambda x: x.acceleration_score, reverse=True)
        return results

    def _calc_accel_score(self, rank_change: int, ret_5: float, mom_score: float, current_rank: int) -> float:
        """가속도 종합 점수.

        가중치:
          - 순위 변화 속도 (35%): 순위가 빠르게 올라오는 섹터
          - 단기 수익률 (25%): 최근 5일 상승세
          - 모멘텀 스코어 (25%): 전체 모멘텀 기반
          - 순위 보너스 (15%): 이미 상위에 있으면 가점
        """
        # 순위 변화: -10~+10 → 0~100
        rank_norm = min(max((rank_change + 10) / 20 * 100, 0), 100)

        # 단기 수익률: -10~+20% → 0~100
        ret_norm = min(max((ret_5 + 10) / 30 * 100, 0), 100)

        # 모멘텀 스코어: 이미 0~100
        mom_norm = min(max(mom_score, 0), 100)

        # 순위 보너스: 1위=100, 5위=60, 10위=20, 15+=0
        rank_bonus = max(0, (15 - current_rank)) / 14 * 100

        score = rank_norm * 0.35 + ret_norm * 0.25 + mom_norm * 0.25 + rank_bonus * 0.15
        return round(score, 2)

    # ─────────────────────────────────────────────
    # 2. 확신도 판정 + 비대칭 배분
    # ─────────────────────────────────────────────
    def calc_conviction(
        self,
        accelerations: list[SectorAcceleration],
        supply_data: dict = None,
        total_sector_pct: float = 30.0,
    ) -> list[ConvictionResult]:
        """가속도 + 수급 → 확신도 판정 + 비대칭 비중 배분.

        Args:
            accelerations: calc_acceleration() 결과
            supply_data: {sector: {foreign_cum, inst_cum, stealth_buying}}
            total_sector_pct: 섹터 축 전체 비중 (%)

        Returns:
            확신도 판정 + 비중 배분 리스트
        """
        supply_data = supply_data or {}
        results = []

        for acc in accelerations[:self.max_sector_holdings + 2]:  # 여유분 포함
            # 기본 점수 = 가속도
            conviction_score = acc.acceleration_score

            # 수급 가점
            sd = supply_data.get(acc.sector, {})
            foreign_cum = sd.get("foreign_cum_bil", 0)
            inst_cum = sd.get("inst_cum_bil", 0)
            stealth = sd.get("stealth_buying", False)

            supply_bonus = 0
            reasons = [f"가속도 {acc.acceleration_score:.0f}점"]

            if foreign_cum > 0 and inst_cum > 0:
                supply_bonus += 15
                reasons.append("외인+기관 동시매수")
            elif foreign_cum > 0:
                supply_bonus += 8
                reasons.append("외인 순매수")
            elif inst_cum > 0:
                supply_bonus += 5
                reasons.append("기관 순매수")

            if stealth:
                supply_bonus += 10
                reasons.append("스텔스 매집")

            # 순위 급상승 가점
            if acc.rank_change >= 5:
                supply_bonus += 10
                reasons.append(f"순위 {acc.rank_change}단계 급상승")
            elif acc.rank_change >= 3:
                supply_bonus += 5
                reasons.append(f"순위 {acc.rank_change}단계 상승")

            conviction_score = min(conviction_score + supply_bonus, 100)

            # 레벨 판정
            thresholds = self.conviction_thresholds
            if conviction_score >= thresholds["HIGH"]:
                level = "HIGH"
            elif conviction_score >= thresholds["MID"]:
                level = "MID"
            elif conviction_score >= thresholds["LOW"]:
                level = "LOW"
            else:
                level = "SKIP"

            reasons.insert(0, f"확신 {level}")

            results.append(ConvictionResult(
                sector=acc.sector,
                code=acc.code,
                level=level,
                score=conviction_score,
                weight_pct=0,
                reasons=reasons,
            ))

        # 비대칭 배분
        results = [r for r in results if r.level != "SKIP"]
        results = results[:self.max_sector_holdings]
        results = self._allocate_asymmetric(results, total_sector_pct)
        return results

    def _allocate_asymmetric(self, results: list[ConvictionResult], total_pct: float) -> list[ConvictionResult]:
        """확신도 기반 비대칭 비중 배분."""
        if not results:
            return results

        weights = self.conviction_weights
        raw_weights = []
        for r in results:
            raw_weights.append(weights.get(r.level, 10))

        total_raw = sum(raw_weights) or 1
        for i, r in enumerate(results):
            r.weight_pct = round(total_pct * raw_weights[i] / total_raw, 2)

        return results

    # ─────────────────────────────────────────────
    # 3. 이벤트 트리거
    # ─────────────────────────────────────────────
    def check_event_triggers(
        self,
        sector_returns_1d: dict[str, float],
        supply_data: dict = None,
        regime_changed: bool = False,
        regime_direction: str = "",
        us_overnight_grade: int = 3,
    ) -> list[EventTrigger]:
        """이벤트 드리븐 리밸런싱 트리거 체크.

        Returns:
            발동된 트리거 리스트
        """
        triggers = []
        supply_data = supply_data or {}

        # 트리거 1: 섹터 급등 (+5% 이상)
        for sector, ret_1d in sector_returns_1d.items():
            if ret_1d >= self.surge_threshold_pct:
                triggers.append(EventTrigger(
                    triggered=True,
                    trigger_type="SURGE",
                    sector=sector,
                    reason=f"{sector} 일간 +{ret_1d:.1f}% 급등",
                ))

        # 트리거 2: 스마트머니 진입
        for sector, sd in supply_data.items():
            if sd.get("foreign_cum_bil", 0) > self.smart_money_threshold_bil:
                if sd.get("stealth_buying", False):
                    triggers.append(EventTrigger(
                        triggered=True,
                        trigger_type="SMART_MONEY",
                        sector=sector,
                        reason=f"{sector} 스마트머니 스텔스 매집 감지",
                    ))

        # 트리거 3: 레짐 전환
        if regime_changed:
            triggers.append(EventTrigger(
                triggered=True,
                trigger_type="REGIME_SHIFT",
                reason=f"레짐 전환 감지: {regime_direction}",
            ))

        # 트리거 4: US Overnight 1등급
        if us_overnight_grade == 1:
            triggers.append(EventTrigger(
                triggered=True,
                trigger_type="US_BULL",
                reason="US Overnight 1등급 — 강한 상승 시그널",
            ))

        return triggers

    # ─────────────────────────────────────────────
    # 통합 실행
    # ─────────────────────────────────────────────
    def run(
        self,
        current_ranks: dict[str, dict],
        prev_ranks: dict[str, dict],
        supply_data: dict = None,
        total_sector_pct: float = 30.0,
        sector_returns_1d: dict[str, float] = None,
        regime_changed: bool = False,
        regime_direction: str = "",
        us_overnight_grade: int = 3,
    ) -> dict:
        """프레데터 모드 통합 실행.

        Returns:
            {accelerations, convictions, event_triggers, summary}
        """
        # 1. 가속도 스캔
        accelerations = self.calc_acceleration(current_ranks, prev_ranks)

        # 2. 확신도 + 비대칭 배분
        convictions = self.calc_conviction(accelerations, supply_data, total_sector_pct)

        # 3. 이벤트 트리거
        triggers = self.check_event_triggers(
            sector_returns_1d=sector_returns_1d or {},
            supply_data=supply_data,
            regime_changed=regime_changed,
            regime_direction=regime_direction,
            us_overnight_grade=us_overnight_grade,
        )

        # 요약
        top3 = accelerations[:3]
        accel_summary = " > ".join([f"{a.sector}({a.acceleration_score:.0f})" for a in top3])
        conv_summary = " | ".join([f"{c.sector} {c.level}({c.weight_pct:.1f}%)" for c in convictions])
        trigger_summary = f"{len(triggers)}건 발동" if triggers else "없음"

        summary = (
            f"[프레데터] 가속도 TOP3: {accel_summary} | "
            f"배분: {conv_summary} | "
            f"트리거: {trigger_summary}"
        )

        return {
            "accelerations": [self._accel_to_dict(a) for a in accelerations],
            "convictions": [self._conv_to_dict(c) for c in convictions],
            "event_triggers": [self._trigger_to_dict(t) for t in triggers],
            "summary": summary,
        }

    def _accel_to_dict(self, a: SectorAcceleration) -> dict:
        return {
            "sector": a.sector, "code": a.code,
            "current_rank": a.current_rank, "prev_rank": a.prev_rank,
            "rank_change": a.rank_change, "acceleration_score": a.acceleration_score,
            "ret_5d": a.ret_5d, "ret_20d": a.ret_20d, "ret_60d": a.ret_60d,
        }

    def _conv_to_dict(self, c: ConvictionResult) -> dict:
        return {
            "sector": c.sector, "code": c.code,
            "level": c.level, "score": c.score,
            "weight_pct": c.weight_pct, "reasons": c.reasons,
        }

    def _trigger_to_dict(self, t: EventTrigger) -> dict:
        return {
            "triggered": t.triggered, "trigger_type": t.trigger_type,
            "sector": t.sector, "reason": t.reason,
        }
