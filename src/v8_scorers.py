"""
Quantum Master v8.0 — Phase 2: Scoring Engine
"초점에 얼마나 가까운가를 점수로 매긴다"

5개 스코어러: 각각 0.0~1.0 연속값 → 가중합으로 등급 결정
v7.0의 L2(OU), L3(Momentum) 게이트를 스코어로 전환한 핵심 모듈
"""

from dataclasses import dataclass

import pandas as pd


def _graduated_consecutive_score(days: int, cfg: dict) -> float:
    """연속 매수 일수 → 구간별 점수 (0.0 ~ 1.0)

    8일+: 1.00 (확신 매집) | 5~7일: 0.75 (강한 매집)
    3~4일: 0.50 (추세 형성) | 1~2일: 0.25 (관심 시작) | 0일: 0.00
    """
    thresholds = cfg.get('consecutive_thresholds', [
        [8, 1.00], [5, 0.75], [3, 0.50], [1, 0.25],
    ])
    for min_days, score in thresholds:
        if days >= min_days:
            return score
    return 0.0


@dataclass
class ScoreResult:
    """개별 스코어 결과"""
    name: str
    score: float           # 0.0 ~ 1.0
    weight: float
    weighted: float = 0.0  # score * weight
    breakdown: dict = None

    def __post_init__(self):
        self.weighted = self.score * self.weight
        if self.breakdown is None:
            self.breakdown = {}


@dataclass
class GradeResult:
    """최종 등급 결과"""
    total_score: float
    grade: str              # "A", "B", "C", "F"
    scores: list            # [ScoreResult, ...]
    position_size_pct: float = 0.0
    tradeable: bool = False


class ScoringEngine:
    """Phase 2: Scoring Engine — 포물선 초점 근접도 측정"""

    def __init__(self, config: dict):
        v8_cfg = config.get('v8_hybrid', {})
        self.cfg = v8_cfg.get('scoring', {})
        self.weights = self.cfg.get('weights', {
            'energy_depletion': 0.30,
            'valuation': 0.20,
            'ou_reversion': 0.20,
            'momentum_decel': 0.15,
            'smart_money': 0.15,
        })
        self.cutoffs = self.cfg.get('grade_cutoffs', {'A': 0.65, 'B': 0.50, 'C': 0.35})
        self.pos_cfg = v8_cfg.get('position', {})
        # v10.1: 마스터 스위치 — use_short_selling_filter: false면 S5 공매도 보너스 비활성
        self._short_filter_enabled = config.get('use_short_selling_filter', False)

    def score_all(self, row: pd.Series) -> GradeResult:
        """5개 스코어를 계산하고 가중합으로 등급을 결정합니다."""
        scores = [
            self.score_energy_depletion(row),
            self.score_valuation(row),
            self.score_ou_reversion(row),
            self.score_momentum_deceleration(row),
            self.score_smart_money(row),
        ]

        total = sum(s.weighted for s in scores)
        grade = self._determine_grade(total)

        pos_pct = 0.0
        tradeable = False
        if grade == "A":
            pos_pct = self.pos_cfg.get('A_grade_pct', 0.20)
            tradeable = True
        elif grade == "B":
            pos_pct = self.pos_cfg.get('B_grade_pct', 0.10)
            tradeable = True

        return GradeResult(
            total_score=total,
            grade=grade,
            scores=scores,
            position_size_pct=pos_pct,
            tradeable=tradeable,
        )

    def _determine_grade(self, total: float) -> str:
        if total >= self.cutoffs.get('A', 0.80):
            return "A"
        elif total >= self.cutoffs.get('B', 0.65):
            return "B"
        elif total >= self.cutoffs.get('C', 0.50):
            return "C"
        else:
            return "F"

    # ═══════════════════════════════════════════════════════
    # S1: 에너지 소진 스코어 (가중치 0.30)
    # ═══════════════════════════════════════════════════════
    def score_energy_depletion(self, row: pd.Series) -> ScoreResult:
        """
        매도 에너지 고갈 정도 — 초점의 핵심 조건

        (a) RSI 위치: 38~52가 최적 (45 중심)
        (b) 거래량 감소: 5MA/20MA < 0.7이면 관심 철수
        (c) 볼린저 밴드 하단 근접
        """
        w = self.weights.get('energy_depletion', 0.30)
        cfg = self.cfg.get('energy', {})
        score = 0.0
        bd = {}

        # (a) RSI
        rsi = row.get('rsi_14', 50)
        rsi_center = cfg.get('rsi_optimal_center', 45)
        rsi_range = cfg.get('rsi_optimal_range', [38, 52])

        if rsi_range[0] <= rsi <= rsi_range[1]:
            rsi_score = 0.40 * (1.0 - abs(rsi - rsi_center) / 14)
        elif 30 <= rsi < rsi_range[0]:
            rsi_score = 0.25
        elif rsi < 30:
            rsi_score = 0.15
        else:
            rsi_score = 0.0
        score += rsi_score
        bd['rsi'] = round(rsi_score, 3)

        # (b) 거래량 감소
        vol_5ma = row.get('volume_ma5', row.get('vol_5ma', 1))
        vol_20ma = row.get('volume_ma20', row.get('vol_20ma', 1))
        vol_ratio = vol_5ma / max(vol_20ma, 1)

        vol_thresh = cfg.get('vol_ratio_threshold', 0.7)
        if vol_ratio < vol_thresh:
            vol_score = 0.35
        elif vol_ratio < 0.9:
            vol_score = 0.20
        elif vol_ratio < 1.1:
            vol_score = 0.10
        else:
            vol_score = 0.0
        score += vol_score
        bd['volume'] = round(vol_score, 3)

        # (c) 볼린저 밴드 위치
        bb_upper = row.get('bb_upper', 0)
        bb_lower = row.get('bb_lower', 0)
        close = row.get('close', 0)

        bb_width = max(bb_upper - bb_lower, 0.01)
        bb_pos = (close - bb_lower) / bb_width

        bb_optimal = cfg.get('bb_position_optimal', [0.05, 0.30])
        if bb_optimal[0] <= bb_pos <= bb_optimal[1]:
            bb_score = 0.25
        elif bb_optimal[1] < bb_pos <= 0.45:
            bb_score = 0.15
        elif bb_pos < bb_optimal[0]:
            bb_score = 0.10
        else:
            bb_score = 0.0
        score += bb_score
        bd['bollinger'] = round(bb_score, 3)

        # (d) 센티먼트 바닥 보너스 (L5, 0.10)
        if row.get('sentiment_extreme', 0) == 1:
            sent_score = 0.10  # 토론실 비관 극단 → 역발상 매수
        else:
            sent_score = 0.0
        score += sent_score
        if sent_score > 0:
            bd['panic_sentiment'] = round(sent_score, 3)

        return ScoreResult(name="S1_Energy", score=min(score, 1.0), weight=w, breakdown=bd)

    # ═══════════════════════════════════════════════════════
    # S2: 밸류에이션 스코어 (가중치 0.20)
    # ═══════════════════════════════════════════════════════
    def score_valuation(self, row: pd.Series) -> ScoreResult:
        """합리적 가격인가 — PER/PBR 기반 밸류 체크"""
        w = self.weights.get('valuation', 0.20)
        cfg = self.cfg.get('valuation', {})
        score = 0.0
        bd = {}

        # (a) PER vs 업종 평균
        per = row.get('per', 0)
        sector_per = row.get('sector_per', per)

        if sector_per > 0 and per > 0:
            per_ratio = per / sector_per
            discount = cfg.get('per_discount_threshold', 0.8)
            if per_ratio <= discount:
                per_score = 0.50
            elif per_ratio <= 1.0:
                per_score = 0.35
            elif per_ratio <= 1.2:
                per_score = 0.15
            else:
                per_score = 0.0
        else:
            per_score = 0.25  # 데이터 없으면 중립 (상향)
        score += per_score
        bd['per'] = round(per_score, 3)

        # (b) PBR 역사적 위치
        pbr_pctile = row.get('pbr_percentile_3y', 50)
        good_pctile = cfg.get('pbr_percentile_good', 30)

        if pbr_pctile <= good_pctile:
            pbr_score = 0.30
        elif pbr_pctile <= 50:
            pbr_score = 0.20
        elif pbr_pctile <= 70:
            pbr_score = 0.10
        else:
            pbr_score = 0.0
        score += pbr_score
        bd['pbr'] = round(pbr_score, 3)

        # (c) Forward PER
        forward_per = row.get('forward_per', 0)
        fwd_discount = cfg.get('forward_per_discount', 0.85)
        if forward_per > 0 and per > 0 and forward_per < per * fwd_discount:
            fwd_score = 0.20
        elif forward_per <= 0 and per <= 0:
            fwd_score = 0.05  # 데이터 없으면 중립
        else:
            fwd_score = 0.0
        score += fwd_score
        bd['forward_per'] = round(fwd_score, 3)

        # (d) QoQ 턴어라운드 보너스 (L3 DART, 0.20)
        if row.get('earnings_surprise', 0) == 1:
            turn_score = 0.20  # 적자→흑자 전환
            bd['turnaround'] = round(turn_score, 3)
            score += turn_score

        return ScoreResult(name="S2_Valuation", score=min(score, 1.0), weight=w, breakdown=bd)

    # ═══════════════════════════════════════════════════════
    # S3: OU 평균회귀 스코어 (가중치 0.20) — v7.0 L2 대체
    # ═══════════════════════════════════════════════════════
    def score_ou_reversion(self, row: pd.Series) -> ScoreResult:
        """
        OU 프로세스 기반 평균회귀 — v7.0 L2의 Gate→Score 전환

        v7.0: z_score 미달 → 즉시 탈락 (98% 차단)
        v8.0: z_score에 따라 0.0~1.0 연속값 부여
        """
        w = self.weights.get('ou_reversion', 0.20)
        cfg = self.cfg.get('ou', {})
        score = 0.0
        bd = {}

        # (a) z_score
        z = row.get('ou_z', row.get('ou_z_score', 0))

        z_optimal = cfg.get('z_score_optimal', [-2.5, -1.0])
        z_weak = cfg.get('z_score_weak', [-1.0, -0.5])

        if z_optimal[0] <= z <= z_optimal[1]:
            z_score = 0.45
        elif z_weak[0] < z <= z_weak[1]:
            z_score = 0.30
        elif z < z_optimal[0] and z >= -3.5:
            z_score = 0.20
        elif z_weak[1] < z <= 0:
            z_score = 0.10
        else:
            z_score = 0.0
        score += z_score
        bd['z_score'] = round(z_score, 3)

        # (b) half_life
        hl = row.get('half_life', row.get('ou_half_life', 999))
        hl_fast = cfg.get('half_life_fast', [3, 15])
        hl_medium = cfg.get('half_life_medium', [15, 30])

        if hl_fast[0] <= hl <= hl_fast[1]:
            hl_score = 0.35
        elif hl_medium[0] < hl <= hl_medium[1]:
            hl_score = 0.20
        elif 30 < hl <= 60:
            hl_score = 0.10
        else:
            hl_score = 0.0
        score += hl_score
        bd['half_life'] = round(hl_score, 3)

        # (c) theta (mean-reversion strength)
        theta = row.get('ou_theta', 0)
        theta_strong = cfg.get('theta_strong', 0.05)

        if theta > theta_strong:
            theta_score = 0.20
        elif theta > 0.02:
            theta_score = 0.10
        else:
            theta_score = 0.0
        score += theta_score
        bd['theta'] = round(theta_score, 3)

        return ScoreResult(name="S3_OU", score=min(score, 1.0), weight=w, breakdown=bd)

    # ═══════════════════════════════════════════════════════
    # S4: 모멘텀 감속 스코어 (가중치 0.15) — v7.0 L3 대체
    # ═══════════════════════════════════════════════════════
    def score_momentum_deceleration(self, row: pd.Series) -> ScoreResult:
        """
        모멘텀 감속 감지 — 포물선 초점 원리의 핵심

        v7.0: martin_dead_zone → 570건 전량 차단 (사실상 고장)
        v8.0: 곡률 전환 감지 → 하락 곡선이 완만해지는 변곡점 포착
        """
        w = self.weights.get('momentum_decel', 0.15)
        score = 0.0
        bd = {}

        # (a) 기울기 감속
        slope_20 = row.get('linreg_slope_20', 0)
        slope_5 = row.get('linreg_slope_5', 0)

        if slope_20 < 0 and slope_5 > slope_20:
            decel_ratio = min(1.0 - (slope_5 / min(slope_20, -0.001)), 1.0)
            slope_score = 0.35 * max(decel_ratio, 0)
        elif slope_5 >= 0 and slope_20 < 0:
            slope_score = 0.35
        elif slope_5 >= 0 and slope_20 >= 0:
            slope_score = 0.15
        else:
            slope_score = 0.0
        score += slope_score
        bd['slope_decel'] = round(slope_score, 3)

        # (b) 곡률 전환 (EMA 2차 미분)
        curvature = row.get('ema_curvature', 0)
        curvature_prev = row.get('ema_curvature_prev', 0)

        if curvature > 0 and curvature_prev <= 0:
            curv_score = 0.40  # 변곡점 발생!
        elif curvature > 0:
            curv_score = 0.20
        elif curvature > curvature_prev:
            curv_score = 0.10
        else:
            curv_score = 0.0
        score += curv_score
        bd['curvature'] = round(curv_score, 3)

        # (c) MACD 히스토그램 감속
        macd_hist = row.get('macd_histogram', 0)
        macd_hist_prev = row.get('macd_histogram_prev', macd_hist)

        if macd_hist < 0 and macd_hist > macd_hist_prev:
            macd_score = 0.25
        elif macd_hist >= 0:
            macd_score = 0.15
        else:
            macd_score = 0.0
        score += macd_score
        bd['macd'] = round(macd_score, 3)

        return ScoreResult(name="S4_MomentumDecel", score=min(score, 1.0), weight=w, breakdown=bd)

    # ═══════════════════════════════════════════════════════
    # S5: 수급/스마트머니 스코어 (가중치 0.15)
    # ═══════════════════════════════════════════════════════
    def score_smart_money(self, row: pd.Series) -> ScoreResult:
        """수급/스마트머니 — SD V2 우선, 없으면 V1 폴백.

        SD V2: 금액 기반 멀티타임프레임 (5/20/60일) + 5패턴 분류
        V1 (legacy): 연속일수 기반 7개 구성요소 + ETF 왜곡 보정
        """
        w = self.weights.get('smart_money', 0.15)
        cfg = self.cfg.get('smart_money', {})

        # ── SD V2 스코어가 있으면 직접 사용 (scan_buy에서 주입) ──
        sd_v2_score = row.get('sd_score_v2', None)
        if sd_v2_score is not None and sd_v2_score >= 0:
            bd = {
                'source': 'SD_V2',
                'sd_score': round(float(sd_v2_score), 4),
                'sd_pattern': str(row.get('sd_pattern', 'X')),
            }
            return ScoreResult(
                name="S5_SmartMoney",
                score=min(float(sd_v2_score), 1.0),
                weight=w,
                breakdown=bd,
            )

        # ── V1 폴백: 기존 연속일수 기반 로직 ──
        score = 0.0
        bd = {}

        # ── ETF 왜곡 보정 파라미터 ──
        etf_distortion = row.get('etf_distortion_pct', 0)
        etf_cfg = self.cfg.get('etf_flow_distortion', {}).get('distortion_correction', {})
        high_thresh = etf_cfg.get('high_distortion_threshold', 50)
        mid_thresh = etf_cfg.get('medium_distortion_threshold', 30)

        # (a) OBV 다이버전스 (0.25)
        price_trend = row.get('price_trend_5d', 0)
        obv_trend = row.get('obv_trend_5d', 0)

        if price_trend < 0 and obv_trend > 0:
            obv_score = 0.25  # 가격 하락 + OBV 상승 = 매집
        elif obv_trend > 0:
            obv_score = 0.12
        else:
            obv_score = 0.0
        score += obv_score
        bd['obv_divergence'] = round(obv_score, 3)

        # (b) 외국인 연속 매수 (0.20) — 구간별 채점
        f_consec = int(row.get('foreign_consecutive_buy', 0))
        f_ratio = _graduated_consecutive_score(f_consec, cfg)
        f_sub = round(f_ratio * 0.20, 4)
        score += f_sub
        bd['foreign'] = round(f_sub, 3)
        bd['foreign_consec_days'] = f_consec

        # (c) 기관 연속 매수 (0.20) — ETF 왜곡 비율만큼 할인
        i_consec = int(row.get('inst_consecutive_buy', 0))
        i_ratio = _graduated_consecutive_score(i_consec, cfg)

        if etf_distortion > high_thresh:
            i_ratio *= 0.5   # 50%+ 왜곡 → 기관 점수 반감
        elif etf_distortion > mid_thresh:
            i_ratio *= 0.7   # 30~50% 왜곡 → 30% 할인

        i_sub = round(i_ratio * 0.20, 4)
        score += i_sub
        bd['institutional'] = round(i_sub, 3)
        bd['inst_consec_days'] = i_consec
        if etf_distortion > 0:
            bd['etf_distortion_pct'] = round(etf_distortion, 1)

        # (d) DRS (0.15)
        drs = row.get('distribution_risk_score', row.get('smart_z', 0.5))
        drs_safe = cfg.get('drs_safe_threshold', 0.3)
        drs_neutral = cfg.get('drs_neutral_threshold', 0.5)

        if drs < drs_safe:
            drs_score = 0.15
        elif drs <= drs_neutral:
            drs_score = 0.08
        else:
            drs_score = 0.0
        score += drs_score
        bd['drs'] = round(drs_score, 3)

        # (e) 수급 다이버전스 (±0.10) — ETF 왜곡 시 기관매집 신뢰 하락
        div = int(row.get('supply_divergence', 0))
        div_bonus = cfg.get('divergence_bonus', 0.10)
        div_penalty = cfg.get('divergence_penalty', -0.05)
        if div == 1:      # 외인매도 + 기관매수 = 기관 매집
            if etf_distortion > high_thresh:
                div_score = div_bonus * 0.3  # ETF 기계적 매수 가능성 → 대폭 할인
            else:
                div_score = div_bonus
        elif div == -1:   # 쌍매도 = 위험
            div_score = div_penalty
        else:
            div_score = 0.0
        score += div_score
        bd['divergence'] = round(div_score, 3)

        # (f) 공매도 보너스 (0~0.05)
        if self._short_filter_enabled:
            short_ratio = row.get('short_ratio', 0)
            has_short_data = not pd.isna(short_ratio) and short_ratio > 0

            if has_short_data:
                short_cover = row.get('short_cover_signal', 0)
                short_spike = row.get('short_spike', 1.0)

                if short_cover:
                    short_score = 0.05
                elif short_spike < 0.5:
                    short_score = 0.03
                elif short_spike > 2.0:
                    short_score = 0.0
                else:
                    short_score = 0.02
                score += short_score
                bd['short_selling'] = round(short_score, 3)

        # (g) 연기금 보너스 (0~0.05)
        if row.get('pension_top_buyer', 0) == 1:
            pension_score = 0.05
            score += pension_score
            bd['pension'] = round(pension_score, 3)
        elif row.get('pension_net_5d', 0) > 0:
            pension_score = 0.03
            score += pension_score
            bd['pension'] = round(pension_score, 3)

        return ScoreResult(name="S5_SmartMoney", score=min(score, 1.0), weight=w, breakdown=bd)
