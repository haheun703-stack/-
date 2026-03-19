"""
Step 6: backtest_engine.py — v3.0 퀀트 백테스트 엔진

v2.1 → v3.0 변경:
- 4단계 부분청산: 25%씩 2R/4R/8R/10R (Price Action 논문)
- 레짐 기반 포지션 축소 (Distribution 레짐 시)
- 최대 보유 10일 제한 (OU+BB 논문)
- 진단 리포트 + Quant Metrics 통합
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .alpha.engine import AlphaEngine
from .position_sizer import PositionSizer
from .quant_metrics import assess_reliability, calc_full_metrics, print_metrics
from .regime_gate import RegimeGate
from .signal_engine import SignalEngine
from .use_cases.adaptive_exit import AdaptiveExitManager
from .use_cases.anchor_learner import AnchorLearner
from .use_cases.daily_hold_scorer import DailyHoldScorer
from .walk_forward import BootstrapValidator, MonteCarloSimulator

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """보유 포지션"""
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    target_price: float
    atr_value: float
    grade: str
    bes_score: float
    trigger_type: str = "confirm"   # impulse / confirm / breakout
    stop_loss_pct: float = 0.05     # 모드별 손절 비율
    highest_price: float = 0.0
    trailing_stop: float = 0.0
    partial_sold: bool = False
    # 분할 매수 추적
    total_stages_entered: int = 1   # 몇 차 매수까지 들어갔나
    max_investment: float = 0.0     # 최대 투입 가능 금액
    # v3.0: 4단계 부분청산 추적
    partial_exits_done: int = 0     # 완료된 부분청산 횟수 (0~4)
    initial_shares: int = 0         # 최초 매수 수량 (부분청산 비율 계산용)


@dataclass
class Trade:
    """완료된 거래 기록"""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str
    grade: str
    bes_score: float
    commission: float
    trigger_type: str = "confirm"


class BacktestEngine:
    """v3.0 퀀트 백테스트 엔진"""

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        use_v9: bool = False,
        use_parabola: bool = False,
        bt_start: str | None = None,
        bt_end: str | None = None,
    ):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.use_v9 = use_v9
        self.use_parabola = use_parabola
        self.max_parabola_positions = 1  # Mode B 최대 동시 보유 (별도 슬롯)
        self.bt_start = bt_start  # 백테스트 시작일 (YYYY-MM-DD)
        self.bt_end = bt_end      # 백테스트 종료일 (YYYY-MM-DD)

        bt = self.config["backtest"]
        self.initial_capital = bt["initial_capital"]
        self.max_positions = bt["max_positions"]
        self.commission_rate = bt["commission_rate"]
        self.slippage_rate = bt["slippage_rate"]
        self.tax_rate = bt.get("tax_rate", 0.0018)  # 증권거래세 0.18% (매도 시만)
        # 동적 슬리피지: 저가주일수록 슬리피지 증가 (호가단위 영향)
        self.dynamic_slippage = bt.get("dynamic_slippage", True)
        self.slippage_price_ref = bt.get("slippage_price_ref", 10000)  # 기준가
        self.partial_tp_pct = bt.get("partial_take_profit_pct", 0.50)
        self.trend_exit_days = bt["trend_exit_consecutive_days"]
        self.atr_stop_mult = bt["trailing_stop_atr_mult"]

        # v3.0: 4단계 부분청산 설정
        exit_cfg = self.config.get("quant_engine", {}).get("exit", {})
        self.partial_exit_r = exit_cfg.get("partial_exit_r", [2, 4, 8, 10])
        self.partial_exit_pct = exit_cfg.get("partial_exit_pct", 0.25)
        self.max_hold_days = exit_cfg.get("max_hold_days", 10)

        # v8.5: 시간 감쇠 청산 모델 (Sinc-inspired)
        # trailing_stop = highest - ATR_mult × decay_factor
        # decay_factor = 1 / (1 + decay_rate × days_held)
        decay_cfg = exit_cfg.get("time_decay", {})
        self.time_decay_enabled = decay_cfg.get("enabled", False)
        self.time_decay_rate = decay_cfg.get("decay_rate", 0.05)
        self.time_decay_min_days = decay_cfg.get("min_days", 2)  # 감쇠 시작일

        # v6.0: Martin 최적 보유기간 연동
        martin_cfg = self.config.get("martin_momentum", {})
        martin_hold_cfg = martin_cfg.get("optimal_hold", {})
        if martin_cfg.get("enabled", False) and martin_hold_cfg.get("enabled", True):
            n_fast = martin_cfg.get("n_fast", 8)
            n_slow = martin_cfg.get("n_slow", 24)
            optimal = int(1.7 * (n_fast + n_slow))
            min_days = martin_hold_cfg.get("min_days", 5)
            max_days = martin_hold_cfg.get("max_days", 30)
            self.max_hold_days = max(min_days, min(optimal, max_days))

        self.signal_engine = SignalEngine(config_path)
        self.position_sizer = PositionSizer(self.config)

        # v8.2: 7D 시장 체제 감지 Gate
        self.regime_gate = RegimeGate(self.config)
        self.regime_log: list[dict] = []  # 체제 변화 추적용

        # v8.3: 공매도 캘린더 + Regime Profile
        # v10.1: 마스터 스위치 — use_short_selling_filter: false면 체제 프로파일 비활성
        self._short_filter_enabled = self.config.get("use_short_selling_filter", False)
        if self._short_filter_enabled:
            self._short_calendar = self._parse_short_calendar(
                self.config.get("short_selling_calendar", [])
            )
            self._regime_profiles = self.config.get("regime_profiles", {})
        else:
            self._short_calendar = []   # 빈 캘린더 → 항상 기본값 사용
            self._regime_profiles = {}  # 프로파일 없음 → _apply_regime_profile 무동작
        self._base_max_positions = self.max_positions
        self._base_max_hold_days = self.max_hold_days
        self._base_atr_stop_mult = self.atr_stop_mult
        self._current_short_status = None  # 캐시
        self._current_short_profile = {}  # 현재 활성 프로파일 캐시

        # v8.5.1: 동시 손실 한도 (Portfolio Stop)
        ps_cfg = self.config.get("portfolio_stop", {})
        self.portfolio_stop_enabled = ps_cfg.get("enabled", False)
        self.portfolio_stop_threshold = ps_cfg.get("avg_pnl_threshold", -0.03)

        # v4.1: 적응형 청산
        adaptive_cfg = self.config.get("adaptive_exit", {})
        self.adaptive_exit_enabled = adaptive_cfg.get("enabled", False)
        if self.adaptive_exit_enabled:
            self.adaptive_exit = AdaptiveExitManager(config_path)
            self.hold_scorer = DailyHoldScorer(config_path)
        else:
            self.adaptive_exit = None
            self.hold_scorer = None

        # v5.0: 앵커 학습 (auto_update 설정 시)
        anchor_cfg = self.config.get("consensus_engine", {}).get("anchor", {})
        self.anchor_auto_update = anchor_cfg.get("auto_update", False)
        if self.anchor_auto_update:
            self.anchor_learner = AnchorLearner(
                db_path=anchor_cfg.get("db_path", "data/anchors.json"),
                min_success_pnl_pct=anchor_cfg.get("min_success_pnl_pct", 3.0),
                min_failure_pnl_pct=anchor_cfg.get("min_failure_pnl_pct", -2.0),
            )
        else:
            self.anchor_learner = None

        # v6.1: WaveLSFormer 리스크 예산 정규화
        wavels_cfg = self.config.get("wavelsformer", {})
        risk_norm_cfg = wavels_cfg.get("risk_normalization", {})
        self.risk_norm_enabled = risk_norm_cfg.get("enabled", False)
        if self.risk_norm_enabled:
            from .use_cases.risk_normalizer import RiskBudgetNormalizer
            self.risk_normalizer = RiskBudgetNormalizer(
                target_daily_vol=risk_norm_cfg.get("target_daily_vol", 0.02),
                lookback=risk_norm_cfg.get("lookback", 60),
                max_scale=risk_norm_cfg.get("max_scale", 2.0),
                min_scale=risk_norm_cfg.get("min_scale", 0.3),
            )
        else:
            self.risk_normalizer = None

        # v6.2: Config 범위 검증
        from .config_validator import ConfigValidator
        config_warnings = ConfigValidator.validate(self.config)
        for w in config_warnings:
            logger.warning("Config: %s", w)

        # Alpha Engine — L1 REGIME + L4 RISK (Phase I)
        self.alpha = AlphaEngine(self.config)

        # Alpha V2 — 4팩터 통합 스코어러 (STEP 6)
        self._v2_scorer = None
        v2_cfg = self.config.get("alpha_v2", {})
        if v2_cfg.get("enabled", False) and v2_cfg.get("use_unified_scorer", False):
            from .alpha.factors.unified_scorer import UnifiedV2Scorer
            self._v2_scorer = UnifiedV2Scorer(self.config)

        # LENS LAYER — 레짐별 동적 R:R (STEP 7)
        self._lens_enabled = v2_cfg.get("lens_enabled", False)
        self._lens_cfg = v2_cfg.get("lens", {})
        self._lens_asymmetry = None
        if self._lens_enabled:
            from .alpha.lens import asymmetry as _lens_asym_mod
            self._lens_asymmetry = _lens_asym_mod

        logger.info(
            "BacktestEngine v6.2: risk_norm=%s, max_hold=%d, tax=%.2f%%, dyn_slip=%s, alpha=%s, v2=%s",
            self.risk_norm_enabled, self.max_hold_days,
            self.tax_rate * 100, self.dynamic_slippage, self.alpha.enabled,
            self._v2_scorer is not None,
        )

        # 상태
        self.cash = self.initial_capital
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.signal_log: list[dict] = []

    def load_data(self) -> dict:
        """processed 디렉토리에서 전종목 데이터 로딩"""
        processed_dir = Path("data/processed")
        data = {}
        for fpath in sorted(processed_dir.glob("*.parquet")):
            data[fpath.stem] = pd.read_parquet(fpath)
        logger.info(f"데이터 로딩: {len(data)}종목")
        return data

    # ──────────────────────────────────────────────
    # v8.3: 공매도 캘린더 / Regime Profile
    # ──────────────────────────────────────────────

    @staticmethod
    def _parse_short_calendar(calendar_list: list) -> list:
        """공매도 캘린더 파싱 → (start_date, end_date, status) 리스트"""
        from datetime import datetime
        parsed = []
        for entry in calendar_list:
            start = datetime.strptime(str(entry["start"]), "%Y-%m-%d").date()
            end = datetime.strptime(str(entry["end"]), "%Y-%m-%d").date()
            parsed.append((start, end, entry["status"]))
        return parsed

    def _get_short_status(self, date_str: str) -> str:
        """날짜 기준 공매도 상태 반환: 'active' or 'banned'"""
        from datetime import datetime
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        for start, end, status in self._short_calendar:
            if start <= d <= end:
                return status
        return "active"  # 캘린더에 없으면 기본 active

    def _v9_kill_check(
        self, sig: dict, date_str: str, data_dict: dict, idx: int,
    ) -> tuple[bool, list[str]]:
        """v9.0 Kill Filters — 하나라도 걸리면 (True, reasons) 반환.

        K1: Zone < regime threshold
        K2: R:R < regime threshold
        K3: Trigger D (미발동 — 이미 signal=True 통과했으므로 사실상 미적용)
        K4: 20일 평균 거래대금 < 10억
        K5: 52주 고점 대비 -5% 이내
        """
        kills = []
        status = self._get_short_status(date_str)

        if status in ("active", "reopened"):
            zone_th, rr_th = 7 / 15, 2.0
        else:
            zone_th, rr_th = 5 / 15, 1.5

        # K1: Zone
        zone = sig.get("zone_score", 0)
        if zone < zone_th:
            kills.append(f"K1:Zone({zone:.2f}<{zone_th:.2f})")

        # K2: R:R
        rr = sig.get("risk_reward_ratio", 0)
        if rr < rr_th:
            kills.append(f"K2:RR({rr:.1f}<{rr_th:.1f})")

        # K3: Trigger (scan_universe는 signal=True만 반환하므로 거의 안 걸림)
        trigger = sig.get("trigger_type", "none")
        if trigger in ("none", "waiting", "setup"):
            kills.append(f"K3:Trigger({trigger})")

        # K4: 20일 평균 거래대금 < 10억
        ticker = sig.get("ticker", "")
        df = data_dict.get(ticker)
        if df is not None and idx < len(df):
            tv = (df["close"] * df["volume"]).iloc[max(0, idx - 19) : idx + 1]
            avg_tv = float(tv.mean()) if len(tv) > 0 else 0
            if avg_tv < 1_000_000_000:
                kills.append(f"K4:유동성({avg_tv / 1e8:.0f}억)")

        # K5: 52주 고점 대비 -5% 이내
        if df is not None and idx < len(df):
            pct_high = float(df.iloc[idx].get("pct_of_52w_high", 0) or 0)
            if pct_high > 0.95:
                kills.append(f"K5:고점({pct_high:.1%})")

        return len(kills) > 0, kills

    def _apply_regime_profile(self, date_str: str) -> dict:
        """날짜 기준 공매도 체제 프로파일 적용. 현재 활성 프로파일 반환."""
        status = self._get_short_status(date_str)

        # 상태 변화 시에만 재적용 (매일 반복 방지)
        if status == self._current_short_status:
            return self._current_short_profile

        self._current_short_status = status
        profile_key = f"short_selling_{status}"
        profile = self._regime_profiles.get(profile_key, {})
        self._current_short_profile = profile

        if not profile:
            return profile

        # SA Floor 비대칭 적용
        sa_floor = profile.get("sa_floor", 0.55)
        self.regime_gate.set_sa_floor(sa_floor)

        # max_positions 조정
        pos_scale = profile.get("max_positions_scale", 1.0)
        self.max_positions = max(1, int(self._base_max_positions * pos_scale))

        # max_hold_days 조정
        hold_scale = profile.get("max_hold_days_scale", 1.0)
        self.max_hold_days = max(5, int(self._base_max_hold_days * hold_scale))

        # G2 pullback 상한 동적 조정 (v8.4.1)
        pullback_max = profile.get("pullback_max")
        if self.signal_engine.v8_pipeline:
            self.signal_engine.v8_pipeline.gate_engine.set_pullback_max(pullback_max)

        # stop_loss_scale 적용 (v8.4.1 — dead code 활성화)
        stop_scale = profile.get("stop_loss_scale", 1.0)
        self.atr_stop_mult = self._base_atr_stop_mult * stop_scale

        # G4 공매도 압력 게이트 활성화 (v10.0)
        short_gate_active = status in ("active", "reopened")
        if self.signal_engine.v8_pipeline:
            self.signal_engine.v8_pipeline.gate_engine.set_short_gate_active(short_gate_active)

        logger.info(
            "  공매도 체제 전환: %s → max_pos=%d, max_hold=%d, sa_floor=%.2f, "
            "pullback_max=%s, stop_scale=%.2f, G4=%s",
            status, self.max_positions, self.max_hold_days, sa_floor,
            pullback_max or "default", stop_scale,
            "ON" if short_gate_active else "OFF",
        )

        return profile

    # ──────────────────────────────────────────────
    # 동적 슬리피지 계산
    # ──────────────────────────────────────────────

    def _effective_slippage(self, price: float) -> float:
        """가격대별 실효 슬리피지 계산.

        저가주(호가단위 큰 종목)의 실제 체결 슬리피지를 반영.
        - 10,000원 이상: base slippage 그대로
        - 5,000원: 2배
        - 3,000원: ~3.3배
        """
        if not self.dynamic_slippage or price <= 0:
            return self.slippage_rate
        scale = max(1.0, self.slippage_price_ref / price)
        return self.slippage_rate * scale

    # ──────────────────────────────────────────────
    # 매수 (분할 매수 지원)
    # ──────────────────────────────────────────────

    def _execute_buy(self, signal: dict, next_open: float,
                     stage_pct: float = 1.0,
                     df: pd.DataFrame = None, idx: int = 0) -> Position | None:
        """
        시그널 기반 매수.
        stage_pct: 이 단계에서 사용할 비중 (Impulse=0.4, Confirm=0.4, Breakout=0.2)
        """
        entry_price = next_open * (1 + self._effective_slippage(next_open))

        available_cash = self.cash

        current_risk = sum(
            p.shares * (p.entry_price - p.stop_loss)
            for p in self.positions
        )

        # v6.0: Martin 변동성 정규화 비중
        vol_weight = 1.0
        martin_data = signal.get("martin_momentum")
        if martin_data:
            vol_weight = martin_data.get("vol_weight", 1.0)

        sizing = self.position_sizer.calculate(
            account_balance=available_cash,
            entry_price=entry_price,
            atr_value=signal["atr_value"],
            grade_ratio=signal["position_ratio"],
            current_portfolio_risk=current_risk,
            stage_pct=stage_pct,  # 분할 비중 전달
            vol_normalized_weight=vol_weight,
        )

        # v6.1: WaveLSFormer 리스크 예산 정규화
        if self.risk_norm_enabled and self.risk_normalizer and df is not None:
            try:
                normalized = self.risk_normalizer.normalize_position(
                    df, idx, sizing["shares"], entry_price,
                )
                if normalized != sizing["shares"] and normalized > 0:
                    sizing["shares"] = normalized
                    sizing["investment"] = normalized * entry_price
            except Exception as e:
                logger.warning("Risk normalization failed: %s", e)

        if sizing["shares"] <= 0:
            return None

        commission = entry_price * sizing["shares"] * self.commission_rate
        total_cost = sizing["investment"] + commission

        if total_cost > available_cash:
            affordable = int((available_cash - commission) / entry_price)
            if affordable <= 0:
                return None
            sizing["shares"] = affordable
            sizing["investment"] = affordable * entry_price
            commission = sizing["investment"] * self.commission_rate
            total_cost = sizing["investment"] + commission

        self.cash -= total_cost

        # 모드별 손절선
        stop_price = signal["stop_loss"]
        trigger_type = signal.get("trigger_type", "confirm")
        stop_pct = signal.get("stop_loss_pct", 0.05)

        pos = Position(
            ticker=signal["ticker"],
            entry_date=str(signal["date"]),
            entry_price=entry_price,
            shares=sizing["shares"],
            stop_loss=stop_price,
            target_price=signal["target_price"],
            atr_value=signal["atr_value"],
            grade=signal["grade"],
            bes_score=signal.get("zone_score", signal.get("bes_score", 0)),
            trigger_type=trigger_type,
            stop_loss_pct=stop_pct,
            highest_price=entry_price,
            trailing_stop=entry_price - signal["atr_value"] * self.atr_stop_mult,
            initial_shares=sizing["shares"],
        )
        self.positions.append(pos)

        mode_emoji = "⚡" if trigger_type == "impulse" else ("🎯" if trigger_type == "confirm" else "🚀")
        logger.debug(f"  {mode_emoji} 매수: {signal['ticker']} {sizing['shares']}주 @{int(entry_price):,} "
                     f"[{trigger_type}] Zone={pos.bes_score:.2f} {signal['grade']}등급")
        return pos

    def _execute_add(self, pos: Position, df: pd.DataFrame, idx: int,
                     trigger_result, stage_pct: float = 0.20):
        """기존 포지션에 추가 매수 (Breakout 트리거)"""
        if idx + 1 >= len(df):
            return

        next_open = df["open"].iloc[idx + 1]
        entry_price = next_open * (1 + self._effective_slippage(next_open))

        # 추가 매수 수량 계산 (전체 포지션의 stage_pct만큼)
        add_amount = self.cash * 0.3 * stage_pct  # 잔고의 30% × stage_pct
        add_shares = int(add_amount / entry_price)

        if add_shares <= 0:
            return

        commission = entry_price * add_shares * self.commission_rate
        total_cost = entry_price * add_shares + commission

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        # 평균 단가 갱신
        total_shares = pos.shares + add_shares
        avg_price = (pos.entry_price * pos.shares + entry_price * add_shares) / total_shares
        pos.entry_price = avg_price
        pos.shares = total_shares
        pos.total_stages_entered += 1

        # 트레일링 스탑 재설정
        pos.trailing_stop = max(pos.trailing_stop, entry_price - pos.atr_value * self.atr_stop_mult)

        logger.debug(f"  🚀 추가매수: {pos.ticker} +{add_shares}주 @{int(entry_price):,} "
                     f"[breakout] 총 {pos.shares}주")

    # ──────────────────────────────────────────────
    # 매도
    # ──────────────────────────────────────────────

    def _execute_sell(self, pos: Position, exit_price: float,
                      exit_date: str, reason: str, shares: int | None = None):
        """포지션 매도 (증권거래세 + 동적 슬리피지 반영)"""
        sell_shares = shares if shares else pos.shares
        actual_price = exit_price * (1 - self._effective_slippage(exit_price))
        sell_commission = actual_price * sell_shares * self.commission_rate
        buy_commission = pos.entry_price * sell_shares * self.commission_rate
        # 증권거래세: 매도 금액 × tax_rate (매도 시만 부과)
        tax = actual_price * sell_shares * self.tax_rate
        gross_pnl = (actual_price - pos.entry_price) * sell_shares
        net_pnl = gross_pnl - sell_commission - buy_commission - tax
        # v8.6: pnl_pct에 수수료+세금 반영 (기존: gross → 수정: net 기준)
        net_per_share = actual_price - (sell_commission + buy_commission + tax) / sell_shares
        pnl_pct = (net_per_share / pos.entry_price - 1) * 100

        sell_proceeds = actual_price * sell_shares - sell_commission - tax
        self.cash += sell_proceeds

        # v8.2.1: Self-Adaptive 체제 감지 피드백
        self.regime_gate.update_trade_outcome(is_win=(net_pnl > 0))

        hold_days = 0
        try:
            hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(pos.entry_date)).days
        except Exception:
            pass

        trade = Trade(
            ticker=pos.ticker,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=actual_price,
            shares=sell_shares,
            pnl=round(net_pnl),
            pnl_pct=round(pnl_pct, 2),
            hold_days=hold_days,
            exit_reason=reason,
            grade=pos.grade,
            bes_score=pos.bes_score,
            commission=round(sell_commission + buy_commission + tax),
            trigger_type=pos.trigger_type,
        )
        self.trades.append(trade)

        if shares and shares < pos.shares:
            pos.shares -= sell_shares
            pos.partial_sold = True
        else:
            self.positions.remove(pos)

        emoji = "🔴" if net_pnl < 0 else "🟢"
        logger.debug(f"  {emoji} 매도: {pos.ticker} {sell_shares}주 @{int(actual_price):,} "
                     f"({reason}) [{pos.trigger_type}] PnL={pnl_pct:+.1f}%")

    # ──────────────────────────────────────────────
    # 보유 종목 관리
    # ──────────────────────────────────────────────

    def _manage_parabola_position(self, pos, df, idx: int, date_str: str):
        """Mode B 전용 청산: 포물선 시작점 홀딩 로직.

        - trend_exit 비활성화
        - 손절: 베이스 하단 고정 (pos.stop_loss)
        - 1차 익절: 베이스 높이 × 2 → 50% 매도
        - 2차 익절: 베이스 높이 × 3 → 전량 매도
        - 시간 손절: 20일 후 베이스 높이 × 1 미도달 → 탈출
        """
        row = df.iloc[idx]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        hold_days = 0
        try:
            hold_days = (pd.Timestamp(date_str) - pd.Timestamp(pos.entry_date)).days
        except Exception:
            pass

        # 베이스 높이 역산 (target = entry + base_height × 3)
        base_height = (pos.target_price - pos.entry_price) / 3.0
        if base_height <= 0:
            base_height = pos.atr_value * 2  # fallback

        # 최고가 갱신
        if high > pos.highest_price:
            pos.highest_price = high

        # 1. 절대 손절 (베이스 하단 고정)
        if low <= pos.stop_loss:
            self._execute_sell(pos, pos.stop_loss, date_str, "stop_loss_parabola")
            return

        # 2. 2차 익절: 베이스 높이 × 3 → 전량 매도
        base_3x = pos.entry_price + base_height * 3.0
        if high >= base_3x:
            self._execute_sell(pos, base_3x, date_str, "target_3x_parabola")
            return

        # 3. 1차 익절: 베이스 높이 × 2 → 50% 매도
        base_2x = pos.entry_price + base_height * 2.0
        if high >= base_2x and not pos.partial_sold:
            exit_shares = max(1, pos.shares // 2)
            self._execute_sell(
                pos, base_2x, date_str, "partial_2x_parabola",
                shares=exit_shares,
            )
            pos.partial_sold = True
            return

        # 4. 시간 손절: 20일 경과 + 베이스 높이 × 1 미도달
        base_1x = pos.entry_price + base_height
        if hold_days >= 20 and close < base_1x:
            self._execute_sell(pos, close, date_str, "time_exit_parabola")
            return

    def _manage_positions(self, data_dict: dict, idx: int, date_str: str):
        """v4.1 포지션 관리: 적응형 청산 + 4단계 부분청산 + 일일 보유 판단"""
        for pos in list(self.positions):
            if pos.ticker not in data_dict:
                continue
            df = data_dict[pos.ticker]
            if idx >= len(df):
                continue

            # Mode B 전용 청산 (trend_exit 등 비활성화)
            if pos.trigger_type == "parabola":
                self._manage_parabola_position(pos, df, idx, date_str)
                continue

            row = df.iloc[idx]
            close = row["close"]
            high = row["high"]
            low = row["low"]

            # 0. 보유일 계산
            hold_days = 0
            try:
                hold_days = (pd.Timestamp(date_str) - pd.Timestamp(pos.entry_date)).days
            except Exception:
                pass

            # 0-1. 최고가 갱신 (적응형 청산에 필요)
            if high > pos.highest_price:
                pos.highest_price = high

            # Alpha Engine X1-X5 청산 규칙 (Point D)
            if self.alpha.enabled:
                alpha_exit = self.alpha.check_exits(
                    entry_price=pos.entry_price,
                    current_price=close,
                    highest_price=pos.highest_price,
                    atr_value=pos.atr_value,
                    hold_days=hold_days,
                    partial_sold=pos.partial_sold,
                    df=df,
                    idx=idx,
                )
                if alpha_exit is not None and alpha_exit.triggered:
                    reason = f"alpha_{alpha_exit.rule.name}"
                    if alpha_exit.partial_pct < 1.0:
                        sell_shares = max(1, int(pos.shares * alpha_exit.partial_pct))
                        self._execute_sell(pos, close, date_str, reason, shares=sell_shares)
                        pos.partial_sold = True
                    else:
                        self._execute_sell(pos, close, date_str, reason)
                    if pos not in self.positions:
                        continue

            # v6.1: 극한 변동성 시 포지션 긴급 관리
            if (hasattr(self.signal_engine, 'extreme_vol_detector')
                    and self.signal_engine.extreme_vol_enabled
                    and self.signal_engine.extreme_vol_detector):
                try:
                    evol_result = self.signal_engine.extreme_vol_detector.detect(df, idx)
                    if evol_result.is_extreme:
                        pct_loss = (close / pos.entry_price - 1)
                        if evol_result.direction == "bearish_breakdown":
                            self._execute_sell(pos, close, date_str, "extreme_vol_bearish")
                            continue
                        elif evol_result.direction == "capitulation" and pct_loss < -0.03:
                            self._execute_sell(pos, close, date_str, "extreme_vol_capitulation")
                            continue
                except Exception as e:
                    logger.warning("Extreme vol position check failed for %s: %s", pos.ticker, e)

            # 0-2. 일일 보유 점수 + 최대 보유일 (일일 점수로 연장 가능)
            effective_max_hold = self.max_hold_days
            hold_result = None
            if self.adaptive_exit_enabled and self.hold_scorer and hold_days >= 2:
                hold_result = self.hold_scorer.score(
                    df, idx, pos.entry_price, pos.highest_price,
                    trigger_type=pos.trigger_type, hold_days=hold_days,
                )
                effective_max_hold += hold_result.hold_days_adjustment

                # 일일 점수 EXIT → 즉시 청산
                if hold_result.action == "exit":
                    self._execute_sell(pos, close, date_str, "hold_score_exit")
                    continue

            if hold_days >= effective_max_hold:
                self._execute_sell(pos, close, date_str, "max_hold_days")
                continue

            # 1. 절대 손절 (ATR 기반 — 건강도와 무관)
            if low <= pos.stop_loss:
                self._execute_sell(pos, pos.stop_loss, date_str, f"stop_loss_{pos.trigger_type}")
                continue

            # 2. 퍼센트 손절 — 적응형 또는 고정
            pct_loss = (close / pos.entry_price - 1)

            if self.adaptive_exit_enabled and self.adaptive_exit:
                # v4.1: 조정 건강도 평가 후 동적 손절
                health = self.adaptive_exit.evaluate_pullback(
                    df, idx, pos.entry_price, pos.highest_price, pos.trigger_type,
                )

                if health.classification == "critical":
                    # 긴급 → 즉시 청산
                    self._execute_sell(pos, close, date_str, "adaptive_critical")
                    continue
                elif health.classification == "dangerous":
                    # 위험 → 타이트 손절 적용
                    if pct_loss <= -health.adjusted_stop_pct:
                        self._execute_sell(pos, close, date_str, "adaptive_dangerous")
                        continue
                elif health.classification == "healthy":
                    # 건강 → 넓은 손절 (MA20 기반)
                    if health.adjusted_stop_price > 0 and low <= health.adjusted_stop_price:
                        self._execute_sell(pos, health.adjusted_stop_price, date_str, "adaptive_healthy_stop")
                        continue
                else:
                    # caution → 기존 로직 유지
                    if pct_loss <= -pos.stop_loss_pct:
                        self._execute_sell(pos, close, date_str, f"pct_stop_{pos.trigger_type}")
                        continue
            else:
                # 기존 고정 손절
                if pct_loss <= -pos.stop_loss_pct:
                    self._execute_sell(pos, close, date_str, f"pct_stop_{pos.trigger_type}")
                    continue

            # 2. 4단계 부분청산 (Price Action 논문: 25%씩 2R/4R/8R/10R)
            if pos.partial_exits_done < len(self.partial_exit_r):
                risk_per_share = pos.entry_price - pos.stop_loss
                if risk_per_share > 0:
                    next_r = self.partial_exit_r[pos.partial_exits_done]
                    target_r_price = pos.entry_price + risk_per_share * next_r

                    if high >= target_r_price:
                        exit_shares = max(
                            1,
                            int(pos.initial_shares * self.partial_exit_pct),
                        )
                        exit_shares = min(exit_shares, pos.shares)

                        if exit_shares > 0:
                            self._execute_sell(
                                pos, target_r_price, date_str,
                                f"partial_{next_r}R",
                                shares=exit_shares,
                            )
                            pos.partial_exits_done += 1
                            pos.partial_sold = True

                            # 부분청산 후 잔여 포지션 없으면 다음으로
                            if pos not in self.positions:
                                continue

            # 3. 트레일링 스탑 갱신 (v4.1 + v8.5 시간 감쇠)
            trailing_mult = self.atr_stop_mult
            if hold_result is not None:
                trailing_mult *= hold_result.trailing_tightness

            # v8.5: 시간 감쇠 — 보유일 증가 시 ATR 승수 축소
            if self.time_decay_enabled and hold_days >= self.time_decay_min_days:
                decay_factor = 1.0 / (1.0 + self.time_decay_rate * hold_days)
                trailing_mult *= decay_factor

            trailing_mult = max(trailing_mult, 0.5)  # v6.2: 최소 0.5배 ATR

            pos.trailing_stop = pos.highest_price - pos.atr_value * trailing_mult

            # 4. 트레일링 스탑 히트 (부분청산 시작 후)
            if pos.partial_sold and low <= pos.trailing_stop:
                self._execute_sell(pos, pos.trailing_stop, date_str, "trailing_stop")
                continue

            # 5. 추세 이탈
            if "sma_60" in df.columns and "sma_120" in df.columns:
                start_check = max(0, idx - self.trend_exit_days + 1)
                trend_broken = all(
                    df["sma_60"].iloc[i] < df["sma_120"].iloc[i]
                    for i in range(start_check, idx + 1)
                    if not pd.isna(df["sma_60"].iloc[i])
                )
                if trend_broken and (idx - start_check + 1) >= self.trend_exit_days:
                    self._execute_sell(pos, close, date_str, "trend_exit")
                    continue

    # ──────────────────────────────────────────────
    # 포트폴리오 가치
    # ──────────────────────────────────────────────

    def _calc_portfolio_value(self, data_dict: dict, idx: int) -> float:
        holdings = 0
        for pos in self.positions:
            if pos.ticker in data_dict:
                df = data_dict[pos.ticker]
                if idx < len(df):
                    holdings += df["close"].iloc[idx] * pos.shares
        return self.cash + holdings

    # ──────────────────────────────────────────────
    # 메인 백테스트 루프
    # ──────────────────────────────────────────────

    def run(self, data_dict: dict) -> dict:
        """
        v2.1 백테스트 루프.

        매일:
        1. 보유 종목 관리 (모드별 손절/익절/트레일링)
        2. 보유 종목 돌파 체크 (Trigger-3 추가 매수)
        3. 전종목 스캔 (Trigger-1/2 신규 진입)
        4. 에쿼티 커브 기록
        """
        first_ticker = list(data_dict.keys())[0]
        all_dates = data_dict[first_ticker].index
        start_idx = 200  # 최소 워밍업

        # 날짜 기반 구간 설정
        if self.bt_start:
            ts = pd.Timestamp(self.bt_start)
            candidates = np.where(all_dates >= ts)[0]
            if len(candidates) > 0:
                start_idx = max(start_idx, int(candidates[0]))
        end_idx = len(all_dates)
        if self.bt_end:
            ts = pd.Timestamp(self.bt_end)
            candidates = np.where(all_dates <= ts)[0]
            if len(candidates) > 0:
                end_idx = int(candidates[-1]) + 1

        # Alpha Engine 상태 초기화
        self.alpha.reset(self.initial_capital)

        logger.info(f"백테스트 v3.0 시작: {all_dates[start_idx]} ~ {all_dates[min(end_idx-1, len(all_dates)-1)]}")
        logger.info(f"  초기자본: {self.initial_capital:,}원 | 종목수: {len(data_dict)} | 6-Layer Pipeline")

        for idx in tqdm(range(start_idx, end_idx), desc="v3.0 backtest"):
            date_str = str(all_dates[idx].date()) if hasattr(all_dates[idx], "date") else str(all_dates[idx])

            # ── 0. 공매도 체제 프로파일 적용 (v8.3) ──
            short_profile = self._apply_regime_profile(date_str)

            # ── 0.5 동시 손실 한도 (Portfolio Stop) ──
            # 포지션 관리(손절 청산) 전에 체크해야 실제 포트폴리오 상태 반영
            # "오늘 아침 내 포트폴리오가 안 좋으면 신규 진입하지 않는다"
            portfolio_stopped = False
            if self.portfolio_stop_enabled and self.positions:
                unrealized_pcts = []
                for p in self.positions:
                    p_df = data_dict.get(p.ticker)
                    if p_df is not None and idx < len(p_df):
                        cur_price = p_df["close"].iloc[idx]
                        unrealized_pcts.append((cur_price - p.entry_price) / p.entry_price)
                if unrealized_pcts:
                    avg_unrealized = sum(unrealized_pcts) / len(unrealized_pcts)
                    if avg_unrealized < self.portfolio_stop_threshold:
                        portfolio_stopped = True
                        logger.debug(
                            f"  Portfolio Stop: avg={avg_unrealized:.2%} < "
                            f"{self.portfolio_stop_threshold:.0%} ({date_str})"
                        )

            # ── 1. 보유 종목 관리 ──
            self._manage_positions(data_dict, idx, date_str)

            # ── 2. 보유 종목 Breakout 체크 (Trigger-3 추가 매수) ──
            held_tickers = {p.ticker for p in self.positions}
            if held_tickers:
                breakouts = self.signal_engine.scan_breakout(data_dict, idx, held_tickers)
                for bo in breakouts:
                    # 해당 포지션 찾기
                    pos_list = [p for p in self.positions if p.ticker == bo["ticker"]]
                    if pos_list and pos_list[0].total_stages_entered < 3:
                        self._execute_add(
                            pos_list[0],
                            data_dict[bo["ticker"]],
                            idx,
                            bo["trigger"],
                            stage_pct=bo["trigger"].entry_stage_pct,
                        )

            # ── 2.5 시장 체제 감지 (7D Regime Gate) ──
            regime = self.regime_gate.detect(data_dict, idx)

            # Alpha Engine: 레짐 변환 + 파라미터 추출 (Point B)
            alpha_level, alpha_params = self.alpha.get_regime(regime)

            # 체제 변화 로깅 (매일이 아닌 변화 시점만)
            if not self.regime_log or self.regime_log[-1]["regime"] != regime.regime:
                self.regime_log.append({
                    "date": date_str,
                    "regime": regime.regime,
                    "scale": regime.position_scale,
                    "composite": regime.composite_score,
                    "details": regime.details,
                })
                if len(self.regime_log) > 1:
                    prev = self.regime_log[-2]["regime"]
                    logger.info(f"  체제 전환: {prev} → {regime.regime} ({date_str}) {regime.details}")

            # ── 3. 신규 진입 (Trigger-1 또는 Trigger-2) ──
            # hostile 체제에서는 신규 진입 차단
            # v8.3.1: 공매도 프로파일의 min_regime_scale 적용
            min_regime = short_profile.get("min_regime_scale", 0.0)
            effective_max_pos = self.max_positions
            if portfolio_stopped:
                effective_max_pos = 0  # 포트폴리오 동시 손실 → 신규 진입 불가
            elif regime.position_scale <= 0 or regime.position_scale < min_regime:
                effective_max_pos = 0  # 신규 진입 불가
            elif regime.position_scale < 1.0:
                # neutral/caution: 최대 포지션 수 축소
                effective_max_pos = max(1, int(self.max_positions * regime.position_scale))

            # Mode A 포지션 수 (Mode B 제외하여 슬롯 독립)
            mode_a_count = sum(1 for p in self.positions if p.trigger_type != "parabola")

            if mode_a_count < effective_max_pos:
                signals = self.signal_engine.scan_universe(
                    data_dict, idx, held_positions=self.positions
                )

                # V2 4팩터 재채점 (게이트/트리거 유지, 스코어링만 V2)
                if self._v2_scorer is not None:
                    signals = self._v2_scorer.rescore_signals(
                        signals, data_dict, idx, alpha_level,
                    )

                for sig in signals:
                    self.signal_log.append({
                        "date": date_str,
                        "ticker": sig["ticker"],
                        "zone_score": sig["zone_score"],
                        "grade": sig["grade"],
                        "trigger_type": sig["trigger_type"],
                        "trigger_confidence": sig.get("trigger_confidence", 0),
                        "entry_price": sig["entry_price"],
                        "rr_ratio": sig["risk_reward_ratio"],
                        "regime": regime.regime,
                        "regime_scale": regime.position_scale,
                    })

                held_tickers = {p.ticker for p in self.positions}

                for sig in signals:
                    mode_a_count = sum(1 for p in self.positions if p.trigger_type != "parabola")
                    if mode_a_count >= effective_max_pos:
                        break
                    if sig["ticker"] in held_tickers:
                        continue

                    # v9.0 Kill 필터 (--v9 모드 시)
                    if self.use_v9:
                        killed, kill_reasons = self._v9_kill_check(
                            sig, date_str, data_dict, idx
                        )
                        if killed:
                            continue

                    # 손익비 기준: 공매도 체제에 따라 상향 가능 (v8.3)
                    profile_min_rr = short_profile.get("min_rr_ratio", 1.5)
                    base_min_rr = max(profile_min_rr, 1.5 if sig["trigger_type"] == "impulse" else 2.0)

                    # LENS asymmetry: 레짐별 동적 min_rr 오버라이드 (STEP 7)
                    if self._lens_asymmetry is not None:
                        _lens_regime = {
                            "favorable": "BULL", "neutral": "CAUTION",
                            "caution": "BEAR", "hostile": "CRISIS",
                        }.get(regime.regime, "CAUTION")
                        _asym = self._lens_asymmetry.compute(_lens_regime, self._lens_cfg)
                        base_min_rr = max(base_min_rr, _asym["min_rr_ratio"])

                    if sig["risk_reward_ratio"] < base_min_rr:
                        continue

                    if idx + 1 < len(all_dates):
                        next_df = data_dict.get(sig["ticker"])
                        if next_df is not None and idx + 1 < len(next_df):
                            next_open = next_df["open"].iloc[idx + 1]
                            # 체제 스케일 반영: regime + 공매도 프로파일 (v8.3)
                            pos_mult = short_profile.get("position_scale_mult", 1.0)
                            base_stage = sig.get("entry_stage_pct", 0.40)
                            scaled_stage = base_stage * regime.position_scale * pos_mult

                            # Alpha Engine VETO 검사 (Point C)
                            if self.alpha.enabled:
                                # 포지션 사이저와 동일한 방식으로 투입금 추정
                                _stop_dist = sig["atr_value"] * self.atr_stop_mult
                                if _stop_dist > 0:
                                    _risk_amt = self.cash * self.position_sizer.max_risk_pct
                                    _est_shares = int(_risk_amt / _stop_dist)
                                    _est_shares = int(_est_shares * sig.get("position_ratio", 0.67) * scaled_stage)
                                    est_investment = _est_shares * next_open
                                else:
                                    est_investment = 0
                                portfolio_value = self._calc_portfolio_value(data_dict, idx)
                                veto = self.alpha.veto_buy(
                                    regime_level=alpha_level,
                                    regime_params=alpha_params,
                                    capital=portfolio_value,
                                    cash=self.cash,
                                    num_positions=mode_a_count,
                                    investment_amount=est_investment,
                                )
                                if veto.vetoed:
                                    continue

                            self._execute_buy(
                                sig, next_open,
                                stage_pct=scaled_stage,
                                df=next_df, idx=idx,
                            )
                            held_tickers.add(sig["ticker"])

            # ── 3b. Mode B: 포물선 탐지 (--parabola, 별도 슬롯) ──
            if self.use_parabola:
                from .parabola_detector import scan_parabola_universe

                held_tickers = {p.ticker for p in self.positions}
                para_held = sum(1 for p in self.positions if p.trigger_type == "parabola")
                if para_held >= self.max_parabola_positions:
                    para_signals = []
                else:
                    para_signals = scan_parabola_universe(data_dict, idx, held_tickers)

                for sig in para_signals:
                    para_held = sum(1 for p in self.positions if p.trigger_type == "parabola")
                    if para_held >= self.max_parabola_positions:
                        break

                    self.signal_log.append({
                        "date": date_str,
                        "ticker": sig["ticker"],
                        "zone_score": sig["zone_score"],
                        "grade": sig["grade"],
                        "trigger_type": sig["trigger_type"],
                        "trigger_confidence": sig.get("trigger_confidence", 0),
                        "entry_price": sig["entry_price"],
                        "rr_ratio": sig["risk_reward_ratio"],
                        "regime": regime.regime,
                        "regime_scale": regime.position_scale,
                    })

                    if idx + 1 < len(all_dates):
                        next_df = data_dict.get(sig["ticker"])
                        if next_df is not None and idx + 1 < len(next_df):
                            next_open = next_df["open"].iloc[idx + 1]
                            # Mode B: 전체 자본의 15% 상한 (단일 종목)
                            max_para_pct = 0.15
                            base_stage = min(sig.get("entry_stage_pct", 0.30), max_para_pct)
                            pos_mult = short_profile.get("position_scale_mult", 1.0)
                            scaled_stage = base_stage * regime.position_scale * pos_mult
                            self._execute_buy(
                                sig, next_open,
                                stage_pct=scaled_stage,
                                df=next_df, idx=idx,
                            )
                            held_tickers.add(sig["ticker"])

            # ── 4. 에쿼티 커브 ──
            portfolio_value = self._calc_portfolio_value(data_dict, idx)

            # Alpha Engine 일일 업데이트
            if self.alpha.enabled:
                self.alpha.update_equity(portfolio_value)
                prev_value = self.equity_curve[-1]["portfolio_value"] if self.equity_curve else self.initial_capital
                daily_pnl_pct = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
                self.alpha.end_of_day(daily_pnl_pct)

                # 포트폴리오 레벨 방어선 체크
                port_exit = self.alpha.check_portfolio()
                if port_exit and port_exit.triggered:
                    logger.warning(
                        "  Alpha 포트폴리오 방어선: %s (%s)",
                        port_exit.rule.value, port_exit.details,
                    )

            self.equity_curve.append({
                "date": date_str,
                "portfolio_value": round(portfolio_value),
                "cash": round(self.cash),
                "n_positions": len(self.positions),
                "positions": [f"{p.ticker}[{p.trigger_type[0]}]" for p in self.positions],
            })

        # ── 잔여 포지션 강제 청산 ──
        last_idx = min(end_idx - 1, len(all_dates) - 1)
        last_date = str(all_dates[last_idx].date()) if hasattr(all_dates[last_idx], "date") else str(all_dates[last_idx])
        for pos in list(self.positions):
            if pos.ticker in data_dict:
                df_t = data_dict[pos.ticker]
                close_idx = min(last_idx, len(df_t) - 1)
                close = df_t["close"].iloc[close_idx]
                self._execute_sell(pos, close, last_date, "backtest_end")

        return self._compile_results()

    # ──────────────────────────────────────────────
    # 결과 정리
    # ──────────────────────────────────────────────

    def _compile_results(self) -> dict:
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        equity_df = pd.DataFrame(self.equity_curve)
        signals_df = pd.DataFrame(self.signal_log)
        regime_df = pd.DataFrame(self.regime_log)

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        if not trades_df.empty:
            trades_df.to_csv(results_dir / "trades_log.csv", index=False, encoding="utf-8-sig")
        if not regime_df.empty:
            regime_df.to_csv(results_dir / "regime_log.csv", index=False, encoding="utf-8-sig")
            logger.info(f"체제 전환 이력: {len(regime_df)}건")
            for _, row in regime_df.iterrows():
                logger.info(f"  {row['date']}: {row['regime']} (score={row['composite']:.2f})")
        if not equity_df.empty:
            equity_df.to_csv(results_dir / "daily_equity.csv", index=False, encoding="utf-8-sig")
        if not signals_df.empty:
            signals_df.to_csv(results_dir / "signals_log.csv", index=False, encoding="utf-8-sig")

        # v3.0: Quant Metrics 사용
        stats = calc_full_metrics(
            trades_df, equity_df,
            initial_capital=self.initial_capital,
        )
        print_metrics(stats)

        # 6-Layer 진단 리포트
        self.signal_engine.diagnostic.print_summary()

        # v4.5: Bootstrap 검증
        bootstrap_cfg = self.config.get("quant_engine", {}).get("bootstrap", {})
        bootstrap_results = None
        if bootstrap_cfg.get("enabled", False) and not trades_df.empty:
            bs_validator = BootstrapValidator(
                n_iterations=bootstrap_cfg.get("n_iterations", 1000),
                block_size=bootstrap_cfg.get("block_size", 10),
                seed=bootstrap_cfg.get("seed", 42),
            )
            bootstrap_results = bs_validator.run(trades_df)
            bs_validator.print_results(bootstrap_results)

        # v4.6: Monte Carlo 시뮬레이션
        mc_cfg = self.config.get("quant_engine", {}).get("monte_carlo", {})
        monte_carlo_results = None
        if mc_cfg.get("enabled", False) and not trades_df.empty:
            mc_simulator = MonteCarloSimulator(
                n_simulations=mc_cfg.get("n_simulations", 1000),
                ruin_threshold_pct=mc_cfg.get("ruin_threshold_pct", -50.0),
                seed=mc_cfg.get("seed", 42),
            )
            monte_carlo_results = mc_simulator.run(trades_df, self.initial_capital)
            mc_simulator.print_results(monte_carlo_results)

        # v4.6: Bootstrap 결과 반영하여 통계적 신뢰도 재평가
        if bootstrap_results:
            stats["statistical_reliability"] = assess_reliability(
                n_trades=len(trades_df),
                bootstrap_results=bootstrap_results,
            )

        # v5.0: 앵커 DB 자동 업데이트
        if self.anchor_auto_update and self.anchor_learner and not trades_df.empty:
            trade_dicts = trades_df.to_dict("records")
            self.anchor_learner.learn_from_trades(trade_dicts)
            self.anchor_learner.save()
            logger.info(
                "앵커 DB 자동 업데이트: %d cases (성공: %d, 실패: %d)",
                len(self.anchor_learner.db.cases),
                sum(1 for c in self.anchor_learner.db.cases if c.outcome == "success"),
                sum(1 for c in self.anchor_learner.db.cases if c.outcome == "failure"),
            )

        return {
            "stats": stats,
            "trades_df": trades_df,
            "equity_df": equity_df,
            "signals_df": signals_df,
            "regime_df": regime_df,
            "diagnostic": self.signal_engine.diagnostic.summarize(),
            "bootstrap": bootstrap_results,
            "monte_carlo": monte_carlo_results,
        }

    def _calc_stats(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:  # noqa: C901
        """하위호환용 — calc_full_metrics로 위임"""
        return calc_full_metrics(trades_df, equity_df, self.initial_capital)

    def _calc_stats_legacy(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        stats = {
            "total_trades": 0, "win_rate": 0.0, "avg_rr_ratio": 0.0,
            "total_return_pct": 0.0, "cagr_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "avg_hold_days": 0, "total_commission": 0,
            "profit_factor": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "grade_breakdown": {}, "trigger_breakdown": {},
        }

        if trades_df.empty or equity_df.empty:
            return stats

        stats["total_trades"] = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        stats["win_rate"] = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        stats["avg_hold_days"] = int(trades_df["hold_days"].mean())
        stats["total_commission"] = int(trades_df["commission"].sum())
        stats["avg_win_pct"] = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        stats["avg_loss_pct"] = losses["pnl_pct"].mean() if len(losses) > 0 else 0

        avg_win = abs(wins["pnl_pct"].mean()) if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 1
        stats["avg_rr_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

        total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
        stats["profit_factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else 0

        final_value = equity_df["portfolio_value"].iloc[-1]
        stats["total_return_pct"] = round((final_value / self.initial_capital - 1) * 100, 1)

        n_years = len(equity_df) / 252
        if n_years > 0 and final_value > 0:
            stats["cagr_pct"] = round(((final_value / self.initial_capital) ** (1 / n_years) - 1) * 100, 1)

        equity_series = equity_df["portfolio_value"]
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        stats["max_drawdown_pct"] = round(drawdown.min(), 1)

        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess = daily_returns.mean() - 0.03 / 252
            stats["sharpe_ratio"] = round(excess / daily_returns.std() * np.sqrt(252), 2)

        # 등급별 분석
        for grade in ["A", "B", "C"]:
            gt = trades_df[trades_df["grade"] == grade]
            if len(gt) > 0:
                gw = gt[gt["pnl"] > 0]
                stats["grade_breakdown"][grade] = {
                    "count": len(gt),
                    "win_rate": round(len(gw) / len(gt) * 100, 1),
                    "avg_pnl_pct": round(gt["pnl_pct"].mean(), 2),
                }

        # 🔥 트리거별 분석 (v2.1 신규)
        if "trigger_type" in trades_df.columns:
            for ttype in ["impulse", "confirm", "breakout"]:
                tt = trades_df[trades_df["trigger_type"] == ttype]
                if len(tt) > 0:
                    tw = tt[tt["pnl"] > 0]
                    stats["trigger_breakdown"][ttype] = {
                        "count": len(tt),
                        "win_rate": round(len(tw) / len(tt) * 100, 1),
                        "avg_pnl_pct": round(tt["pnl_pct"].mean(), 2),
                        "avg_hold_days": int(tt["hold_days"].mean()),
                        "total_pnl": int(tt["pnl"].sum()),
                    }

        return stats
