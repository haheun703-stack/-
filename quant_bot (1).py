"""
============================================================================
 퀀트봇 (QuantBot) — 인내형 지지 반등 스윙 엔진  [ "허준 머신" ]
============================================================================

철학:  "기회는 준비된 자에게 온다."
       약할 때 산다. 단, 떨어지는 칼을 잡는 게 아니라
       '지지선 테스트 후 반등 확인'된 순간에만 산다.

핵심 4원칙:
  1) 상태머신으로 "확인 후 진입"을 강제 — 한 단계라도 건너뛰면 신호 없음 (FOMO 차단)
  2) 손절선 = 지지 바로 아래 → 손익비(RR)가 구조적으로 좋아짐
  3) NXT 매크로 게이트 = 최상단 공통 차단기
  4) 봇은 '결정 + 알림'만, 체결은 사람이 (NXT 수동 체결 대응)

엔진 흐름:
  [0] 유니버스 빌더 (정적 필터: 거래대금/시총/관리종목/동전주/NXT 650 교집합)
        ↓
  [1] 추세 게이트 (주봉 20주선 위 + ADX>25 + 일봉 정배열)
        ↓
  [2] 지지 반등 상태머신 (IDLE→PULLBACK→SUPPORT_TEST→ARMED)
        ↓
  [3] 수급/매집 확인 (외인+기관 순매수 + OBV 상승)  ← 점수 가산/필터
        ↓
  [4] 리스크 (지지 아래 SL / 사이징 / TP1·TP2 / 트레일링) — RR 나쁘면 거부
        ↓
  [5] 매크로 게이트 통과 시 → 텔레그램 매수 알림 (사람이 체결)

NOTE: 외부 의존성 없이 순수 파이썬으로 지표를 직접 구현했다(검증 가능하도록).
      실데이터는 DataProvider를 상속해 KIS API에 연결하면 된다.
      맨 아래 __main__ 에 MockDataProvider 데모가 있어 바로 돌려볼 수 있다.
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import math


# ===========================================================================
# 0. 설정 (Config) — 모든 파라미터를 여기서 제어
# ===========================================================================
@dataclass
class QuantConfig:
    # --- 유니버스 정적 필터 ---
    MIN_TRADING_VALUE: float = 10_000_000_000   # 일 거래대금 하한 (100억). 유동성 — 못 빠져나오면 죽음
    MIN_PRICE: int = 3_000                       # 동전주 제외 (세력판이라 프레임 안 맞음)
    MIN_MARKET_CAP: float = 300_000_000_000      # 시총 하한 (3천억) — 소형주 작전 리스크 회피
    MAX_MARKET_CAP: float = 10_000_000_000_000   # 시총 상한 (10조) — 대형주는 너무 느림. 소부장·중형 중심
    REQUIRE_NXT_TRADABLE: bool = True            # NXT 수동 체결 → NXT 650 종목 교집합만

    # --- 추세 게이트 ---
    ADX_PERIOD: int = 14
    ADX_TREND_MIN: float = 25.0                  # ADX 이 값 미만이면 횡보 → 진입 금지 (휩쏘 차단)
    WEEKLY_MA_SHORT: int = 20                    # 주봉 20주선 (≈ 일봉 100거래일)
    WEEKLY_MA_LONG: int = 60                     # 주봉 60주선 (≈ 14개월 대세선)
    MA_FAST: int = 5
    MA_MID: int = 20
    MA_SLOW: int = 60
    MA_LONG: int = 120

    # --- 지지 반등 상태머신 ---
    PULLBACK_PROXIMITY: float = 0.03             # 종가가 20일선 +3% 이내로 눌리면 'PULLBACK' 진입
    SUPPORT_BREAK_TOL: float = 0.02              # 종가가 지지선 -2% 아래로 마감하면 지지 붕괴 → 리셋
    PULLBACK_VOL_DECLINE: float = 0.9            # 눌림 구간 거래량이 평균의 0.9배 이하 = 건강한 눌림(매물 소화)
    BOUNCE_VOL_RATIO: float = 1.2                # 반등 캔들 거래량이 평균의 1.2배 이상 = 거래량 회복
    SETUP_EXPIRY_DAYS: int = 7                   # 지지 테스트 후 N일 내 반등 확인 못 하면 셋업 만료
    VOL_MA_PERIOD: int = 20

    # --- 수급/매집 확인 ---
    OBV_RISING_DAYS: int = 3                     # OBV N일 연속 상승 = 매집 신호
    SUPPLY_CONFIRM_DAYS: int = 2                 # 외인+기관 N일 연속 순매수 = 수급 확인

    # --- 리스크 / 사이징 ---
    ACCOUNT_EQUITY: float = 100_000_000          # 계좌 평가금 (운전자본). 실운영 시 KIS 잔고로 갱신
    RISK_PER_TRADE_PCT: float = 0.01             # 1회 거래 위험 = 계좌의 1% (= 손절 시 잃을 금액)
    MAX_POSITION_PCT: float = 0.15               # 종목당 최대 비중 15%
    MAX_POSITIONS: int = 5                       # 동시 보유 최대 5종목 (집중)
    SL_BUFFER: float = 0.01                      # 손절선 = 지지 저점 -1% (지지 바로 아래)
    MIN_RR: float = 2.0                          # TP1까지 최소 손익비. 진입가가 지지서 너무 멀면(=RR<2) 거부
    TP1_R: float = 2.0                           # TP1 = 진입 + 위험폭 × 2.0R (절반 익절)
    TP2_R: float = 3.5                           # TP2 = 진입 + 위험폭 × 3.5R (나머지 익절)
    TRAILING_ACTIVATE_R: float = 1.5             # +1.5R 도달 시 트레일링 스탑 활성화
    TRAILING_STOP_PCT: float = 0.05              # 최고가 대비 -5% 트레일링

    # --- 매크로 게이트 (NXT 야간 공포 스코어, -10 ~ +10) ---
    MACRO_HALT_BELOW: float = -7.0               # 이 값 미만이면 신규 진입 전면 정지
    MACRO_SCALE_BELOW: float = -3.0              # 이 값 미만이면 사이징 절반으로 축소


# ===========================================================================
# 1. 데이터 구조
# ===========================================================================
@dataclass
class Bar:
    """일봉 1개"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FlowData:
    """주체별 순매수 (단위: 원 또는 주). 부호: +매수 / -매도"""
    foreign: list[float] = field(default_factory=list)      # 외인 일별 순매수 (최신이 끝)
    institution: list[float] = field(default_factory=list)  # 기관 일별 순매수
    pension: list[float] = field(default_factory=list)      # 연기금 일별 순매수 (바닥 지지 신호)


@dataclass
class SymbolMeta:
    """종목 기본정보 (유니버스 필터용)"""
    code: str
    name: str
    price: float
    trading_value: float        # 당일 거래대금
    market_cap: float
    is_managed: bool            # 관리/경고/단기과열 등 시장조치종목
    is_nxt_tradable: bool       # NXT 거래 가능 종목 여부


# ===========================================================================
# 2. 지표 (Indicators) — 순수 함수로 직접 구현
# ===========================================================================
class Indicators:

    @staticmethod
    def sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def sma_series(values: list[float], period: int) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        for i in range(len(values)):
            if i + 1 < period:
                out.append(None)
            else:
                out.append(sum(values[i + 1 - period:i + 1]) / period)
        return out

    @staticmethod
    def obv(closes: list[float], volumes: list[float]) -> list[float]:
        obv = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        return obv

    @staticmethod
    def _wilder_smooth(values: list[float], period: int) -> list[Optional[float]]:
        """Wilder 평활 (ADX/ATR 표준)"""
        out: list[Optional[float]] = [None] * len(values)
        if len(values) < period:
            return out
        first = sum(values[:period])
        out[period - 1] = first
        for i in range(period, len(values)):
            out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
        return out

    @classmethod
    def adx(cls, bars: list[Bar], period: int = 14) -> dict[str, Optional[float]]:
        """ADX / +DI / -DI 반환 (Wilder 방식). 추세 강도 + 방향."""
        if len(bars) < period * 2:
            return {"adx": None, "plus_di": None, "minus_di": None}

        tr, plus_dm, minus_dm = [], [], []
        for i in range(1, len(bars)):
            h, l = bars[i].high, bars[i].low
            ph, pl, pc = bars[i - 1].high, bars[i - 1].low, bars[i - 1].close
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
            up, down = h - ph, pl - l
            plus_dm.append(up if (up > down and up > 0) else 0.0)
            minus_dm.append(down if (down > up and down > 0) else 0.0)

        atr = cls._wilder_smooth(tr, period)
        sm_plus = cls._wilder_smooth(plus_dm, period)
        sm_minus = cls._wilder_smooth(minus_dm, period)

        dx_series: list[float] = []
        for i in range(len(atr)):
            if atr[i] and atr[i] != 0 and sm_plus[i] is not None and sm_minus[i] is not None:
                pdi = 100 * sm_plus[i] / atr[i]
                mdi = 100 * sm_minus[i] / atr[i]
                denom = pdi + mdi
                dx = 100 * abs(pdi - mdi) / denom if denom != 0 else 0.0
                dx_series.append(dx)

        if len(dx_series) < period:
            return {"adx": None, "plus_di": None, "minus_di": None}

        adx_val = sum(dx_series[:period]) / period
        for i in range(period, len(dx_series)):
            adx_val = (adx_val * (period - 1) + dx_series[i]) / period

        last_atr = atr[-1]
        pdi = 100 * sm_plus[-1] / last_atr if last_atr else None
        mdi = 100 * sm_minus[-1] / last_atr if last_atr else None
        return {"adx": adx_val, "plus_di": pdi, "minus_di": mdi}

    @staticmethod
    def daily_to_weekly(bars: list[Bar]) -> list[Bar]:
        """일봉 → 주봉 리샘플 (date는 'YYYY-MM-DD' 가정, 5거래일 묶음 근사)."""
        weekly: list[Bar] = []
        chunk: list[Bar] = []
        for b in bars:
            chunk.append(b)
            if len(chunk) == 5:
                weekly.append(Bar(
                    date=chunk[-1].date,
                    open=chunk[0].open,
                    high=max(c.high for c in chunk),
                    low=min(c.low for c in chunk),
                    close=chunk[-1].close,
                    volume=sum(c.volume for c in chunk),
                ))
                chunk = []
        if chunk:  # 남은 미완성 주
            weekly.append(Bar(
                date=chunk[-1].date,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
            ))
        return weekly


# ===========================================================================
# 3. 상태머신 정의
# ===========================================================================
class SetupState(Enum):
    IDLE = "IDLE"                  # 셋업 없음 / 추세 게이트 미통과
    PULLBACK = "PULLBACK"          # 정배열 유지 중, 20일선으로 눌림 진행
    SUPPORT_TEST = "SUPPORT_TEST"  # 지지 구간 터치/근접, 아직 반등 미확인
    ARMED = "ARMED"                # 반등 확인됨 → 진입 신호 발생


@dataclass
class SymbolState:
    """종목별 상태머신 상태 (실운영 시 디스크/DB에 영속화 필요)"""
    code: str
    state: SetupState = SetupState.IDLE
    support_level: Optional[float] = None    # 이번 셋업의 지지선 (20일선 등)
    support_test_low: Optional[float] = None # 지지 테스트 시 찍은 저점 (→ SL 기준)
    days_in_setup: int = 0


# ===========================================================================
# 4. 데이터 제공자 인터페이스 (KIS API를 여기에 연결)
# ===========================================================================
class DataProvider(ABC):
    @abstractmethod
    def get_universe(self) -> list[SymbolMeta]:
        """스캔 후보 종목 메타정보 리스트 (거래대금 순위/조건검색 API 활용)."""
        ...

    @abstractmethod
    def get_daily_bars(self, code: str, count: int = 250) -> list[Bar]:
        """일봉 (오래된→최신 순). 주봉 60선 계산 위해 최소 300거래일 권장."""
        ...

    @abstractmethod
    def get_flow(self, code: str, days: int = 10) -> FlowData:
        """주체별 순매수 (외인/기관/연기금)."""
        ...

    @abstractmethod
    def get_macro_score(self) -> float:
        """NXT 야간 공포 스코어 (-10 ~ +10). 별도 nightwatch 모듈에서 산출."""
        ...


# ===========================================================================
# 5. 유니버스 빌더 (정적 필터)
# ===========================================================================
class UniverseBuilder:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def build(self, metas: list[SymbolMeta]) -> list[SymbolMeta]:
        out = []
        for m in metas:
            if m.is_managed:                                   # 관리/경고/단기과열 제외
                continue
            if m.price < self.cfg.MIN_PRICE:                   # 동전주 제외
                continue
            if m.trading_value < self.cfg.MIN_TRADING_VALUE:   # 유동성 하한
                continue
            if not (self.cfg.MIN_MARKET_CAP <= m.market_cap <= self.cfg.MAX_MARKET_CAP):
                continue                                        # 소부장·중형 사냥터
            if self.cfg.REQUIRE_NXT_TRADABLE and not m.is_nxt_tradable:
                continue                                        # NXT 수동 체결 가능 종목만
            out.append(m)
        return out


# ===========================================================================
# 6. 추세 게이트
# ===========================================================================
class TrendGate:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def passes(self, bars: list[Bar]) -> tuple[bool, dict]:
        c = self.cfg
        closes = [b.close for b in bars]
        info: dict = {}

        ma5 = Indicators.sma(closes, c.MA_FAST)
        ma20 = Indicators.sma(closes, c.MA_MID)
        ma60 = Indicators.sma(closes, c.MA_SLOW)
        ma120 = Indicators.sma(closes, c.MA_LONG)
        if None in (ma5, ma20, ma60, ma120):
            return False, {"reason": "데이터 부족(120일선 미산출)"}

        # 일봉 정배열 (5>20>60>120)
        aligned = ma5 > ma20 > ma60 > ma120
        info.update(ma5=ma5, ma20=ma20, ma60=ma60, ma120=ma120, aligned=aligned)
        if not aligned:
            return False, {**info, "reason": "정배열 아님"}

        # ADX (추세장만)
        adx = Indicators.adx(bars, c.ADX_PERIOD)
        info["adx"] = adx
        if adx["adx"] is None or adx["adx"] < c.ADX_TREND_MIN:
            return False, {**info, "reason": f"ADX<{c.ADX_TREND_MIN} (횡보)"}
        if adx["plus_di"] is None or adx["minus_di"] is None or adx["plus_di"] <= adx["minus_di"]:
            return False, {**info, "reason": "+DI<=-DI (상승 방향 아님)"}

        # 주봉 게이트 (가격 > 20주선 > 60주선)
        weekly = Indicators.daily_to_weekly(bars)
        w_closes = [b.close for b in weekly]
        w20 = Indicators.sma(w_closes, c.WEEKLY_MA_SHORT)
        w60 = Indicators.sma(w_closes, c.WEEKLY_MA_LONG)
        info.update(w20=w20, w60=w60)
        if w20 is None:
            return False, {**info, "reason": "주봉 데이터 부족"}
        if w60 is not None and not (w_closes[-1] > w20 > w60):
            return False, {**info, "reason": "주봉 대세 약함(가격<20주 or 20주<60주)"}
        if w60 is None and not (w_closes[-1] > w20):
            return False, {**info, "reason": "주봉 20주선 이탈"}

        return True, info


# ===========================================================================
# 7. 지지 반등 상태머신 (핵심 — 허준 로직)
# ===========================================================================
class SupportBounceEngine:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def update(self, st: SymbolState, bars: list[Bar], gate_ok: bool) -> tuple[SymbolState, bool, dict]:
        """
        하루치 진행. (state 갱신, 진입신호 여부, 진단정보) 반환.
        진입 신호는 state가 ARMED로 전이되는 그 순간에만 True.
        """
        c = self.cfg
        diag: dict = {}

        # 추세 게이트 깨지면 즉시 리셋 (셋업 무효)
        if not gate_ok:
            st.state = SetupState.IDLE
            st.support_level = st.support_test_low = None
            st.days_in_setup = 0
            return st, False, {"note": "게이트 미통과 → IDLE 리셋"}

        closes = [b.close for b in bars]
        vols = [b.volume for b in bars]
        today = bars[-1]
        yday = bars[-2]
        ma5 = Indicators.sma(closes, c.MA_FAST)
        ma20 = Indicators.sma(closes, c.MA_MID)
        vol_ma = Indicators.sma(vols, c.VOL_MA_PERIOD)
        support = ma20  # 1차 지지 = 20일선
        diag.update(ma5=ma5, ma20=ma20, vol_ma=vol_ma, close=today.close)

        if st.state != SetupState.IDLE:
            st.days_in_setup += 1
            # 셋업 만료
            if st.days_in_setup > c.SETUP_EXPIRY_DAYS:
                st.state = SetupState.IDLE
                st.support_level = st.support_test_low = None
                st.days_in_setup = 0
                return st, False, {"note": "셋업 만료 → IDLE"}
            # 지지 붕괴 → 리셋
            if support and today.close < support * (1 - c.SUPPORT_BREAK_TOL):
                st.state = SetupState.IDLE
                st.support_level = st.support_test_low = None
                st.days_in_setup = 0
                return st, False, {"note": "지지 붕괴(종가 이탈) → IDLE"}

        # ---- 상태 전이 ----
        if st.state == SetupState.IDLE:
            # 정배열 유지 중 가격이 20일선 근처로 눌림 → PULLBACK
            near_support = support and today.close <= support * (1 + c.PULLBACK_PROXIMITY)
            vol_declining = vol_ma and today.volume <= vol_ma * c.PULLBACK_VOL_DECLINE
            if near_support and vol_declining:
                st.state = SetupState.PULLBACK
                st.support_level = support
                st.days_in_setup = 0
                diag["note"] = "PULLBACK 진입(20일선 근접 + 거래량 감소=건강한 눌림)"
            return st, False, diag

        if st.state == SetupState.PULLBACK:
            # 저점이 지지 구간을 테스트(터치)하되 종가는 지지 위에서 버팀 → SUPPORT_TEST
            if support and today.low <= support * (1 + c.PULLBACK_PROXIMITY) \
                    and today.close >= support * (1 - c.SUPPORT_BREAK_TOL):
                st.state = SetupState.SUPPORT_TEST
                st.support_test_low = today.low
                diag["note"] = "SUPPORT_TEST(지지 터치 후 종가 사수)"
            return st, False, diag

        if st.state == SetupState.SUPPORT_TEST:
            # 반등 확인 3종: ① 양봉  ② 거래량 회복  ③ 전일고가 돌파(또는 5일선 회복)
            is_bullish = today.close > today.open
            vol_recover = vol_ma and today.volume >= vol_ma * c.BOUNCE_VOL_RATIO
            reclaim = today.close > yday.high or (ma5 and today.close > ma5)
            diag.update(is_bullish=is_bullish, vol_recover=vol_recover, reclaim=reclaim)
            if is_bullish and vol_recover and reclaim:
                st.state = SetupState.ARMED
                diag["note"] = "★ 반등 확인 → ARMED (진입 신호 발생)"
                return st, True, diag   # ← 진입 신호 발생 순간
            return st, False, diag

        if st.state == SetupState.ARMED:
            # 이미 신호 낸 셋업 — 포지션 관리로 넘어가고 셋업은 리셋
            st.state = SetupState.IDLE
            st.support_level = st.support_test_low = None
            st.days_in_setup = 0
            return st, False, {"note": "ARMED 처리 완료 → IDLE"}

        return st, False, diag


# ===========================================================================
# 8. 수급/매집 확인
# ===========================================================================
class SupplyDemandConfirm:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def score(self, bars: list[Bar], flow: FlowData) -> tuple[float, dict]:
        """
        0~3 점. 진입 신호의 '확신 가산점'. 0점이어도 진입 자체는 막지 않되,
        실운영에서 최소 1점 이상을 요구하도록 QuantBot에서 컷 가능.
        """
        c = self.cfg
        s = 0.0
        diag: dict = {}

        # ① 외인+기관 N일 연속 순매수
        def consec_buy(series: list[float], days: int) -> bool:
            if len(series) < days:
                return False
            return all(x > 0 for x in series[-days:])

        fi_ok = consec_buy([f + i for f, i in zip(flow.foreign, flow.institution)],
                           c.SUPPLY_CONFIRM_DAYS) if flow.foreign and flow.institution else False
        if fi_ok:
            s += 1.0
        diag["foreign_inst_consec"] = fi_ok

        # ② 연기금 순매수 (바닥 지지)
        pension_ok = flow.pension and flow.pension[-1] > 0
        if pension_ok:
            s += 0.5
        diag["pension_buy"] = bool(pension_ok)

        # ③ OBV N일 연속 상승 (매집)
        closes = [b.close for b in bars]
        vols = [b.volume for b in bars]
        obv = Indicators.obv(closes, vols)
        obv_rising = len(obv) > c.OBV_RISING_DAYS and \
            all(obv[-i] > obv[-i - 1] for i in range(1, c.OBV_RISING_DAYS + 1))
        if obv_rising:
            s += 1.5
        diag["obv_rising"] = obv_rising

        return s, diag


# ===========================================================================
# 9. 리스크 매니저
# ===========================================================================
@dataclass
class TradePlan:
    code: str
    name: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    shares: int
    risk_amount: float
    rr: float
    position_value: float
    supply_score: float
    note: str = ""


class RiskManager:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def build_plan(self, meta: SymbolMeta, entry: float, support_low: float,
                   supply_score: float, macro_scale: float,
                   open_positions: int) -> Optional[TradePlan]:
        c = self.cfg
        if open_positions >= c.MAX_POSITIONS:
            return None  # 보유 한도 초과

        stop_loss = support_low * (1 - c.SL_BUFFER)
        risk_per_share = entry - stop_loss
        if risk_per_share <= 0:
            return None

        tp1 = entry + risk_per_share * c.TP1_R
        tp2 = entry + risk_per_share * c.TP2_R
        rr = (tp1 - entry) / risk_per_share  # = TP1_R

        # 진입가가 지지에서 너무 멀면(RR 나쁨) 거부 — "좋은 진입가" 강제
        if rr < c.MIN_RR:
            return None

        # 사이징: 1회 위험 = 계좌의 RISK_PER_TRADE_PCT, 매크로에 따라 축소
        risk_budget = c.ACCOUNT_EQUITY * c.RISK_PER_TRADE_PCT * macro_scale
        shares = int(risk_budget / risk_per_share)
        if shares <= 0:
            return None

        position_value = shares * entry
        max_value = c.ACCOUNT_EQUITY * c.MAX_POSITION_PCT
        if position_value > max_value:  # 종목당 비중 캡
            shares = int(max_value / entry)
            position_value = shares * entry
        if shares <= 0:
            return None

        return TradePlan(
            code=meta.code, name=meta.name, entry=entry, stop_loss=stop_loss,
            tp1=tp1, tp2=tp2, shares=shares,
            risk_amount=shares * risk_per_share, rr=rr,
            position_value=position_value, supply_score=supply_score,
        )


# ===========================================================================
# 10. 매크로 게이트 (공통 차단기)
# ===========================================================================
class MacroGate:
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def evaluate(self, macro_score: float) -> tuple[bool, float, str]:
        """(신규진입 허용?, 사이징 배수, 사유)"""
        c = self.cfg
        if macro_score < c.MACRO_HALT_BELOW:
            return False, 0.0, f"매크로 {macro_score:.1f} < {c.MACRO_HALT_BELOW} → 신규진입 전면 정지"
        if macro_score < c.MACRO_SCALE_BELOW:
            return True, 0.5, f"매크로 {macro_score:.1f} → 사이징 50% 축소"
        return True, 1.0, f"매크로 {macro_score:.1f} → 정상"


# ===========================================================================
# 11. 텔레그램 알림 (봇 결정 → 사람 체결)
# ===========================================================================
class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id

    def send(self, text: str) -> None:
        if not self.token or not self.chat_id:
            print(text)  # 미설정 시 콘솔 출력 (테스트)
            return
        # 실운영: requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage", ...)
        print(f"[TELEGRAM→{self.chat_id}]\n{text}")

    @staticmethod
    def format_signal(plan: TradePlan, macro_note: str) -> str:
        return (
            "🟢 [퀀트봇] 매수 신호 (지지 반등 확인)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"종목 : {plan.name} ({plan.code})\n"
            f"진입 : {plan.entry:,.0f}  (내일 시가 또는 지지 위 분할)\n"
            f"손절 : {plan.stop_loss:,.0f}  ({(plan.stop_loss/plan.entry-1)*100:+.1f}%)\n"
            f"목표 : TP1 {plan.tp1:,.0f} (절반) / TP2 {plan.tp2:,.0f}\n"
            f"수량 : {plan.shares:,}주  (≈{plan.position_value:,.0f}원)\n"
            f"위험 : {plan.risk_amount:,.0f}원  /  RR {plan.rr:.1f}\n"
            f"수급 : 매집점수 {plan.supply_score:.1f}/3.0\n"
            f"매크로: {macro_note}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            "체결은 사람이. 신호=결정, 클릭=실행."
        )


# ===========================================================================
# 12. 오케스트레이터
# ===========================================================================
class QuantBot:
    def __init__(self, provider: DataProvider, cfg: QuantConfig = QuantConfig(),
                 notifier: Optional[TelegramNotifier] = None):
        self.p = provider
        self.cfg = cfg
        self.universe_builder = UniverseBuilder(cfg)
        self.trend_gate = TrendGate(cfg)
        self.engine = SupportBounceEngine(cfg)
        self.supply = SupplyDemandConfirm(cfg)
        self.risk = RiskManager(cfg)
        self.macro = MacroGate(cfg)
        self.notifier = notifier or TelegramNotifier()
        self.states: dict[str, SymbolState] = {}  # 종목별 상태머신 (영속화 대상)
        self.open_positions = 0                    # 실운영 시 KIS 잔고로 갱신

    def run_daily_scan(self, min_supply_score: float = 1.0) -> list[TradePlan]:
        """장 마감 후 1회 실행. 진입 신호 → 텔레그램 알림."""
        # 0) 매크로 게이트 (최상단)
        macro_score = self.p.get_macro_score()
        allow, scale, macro_note = self.macro.evaluate(macro_score)
        if not allow:
            self.notifier.send(f"🔴 [퀀트봇] {macro_note}\n오늘 신규 진입 없음.")
            return []

        # 1) 유니버스
        universe = self.universe_builder.build(self.p.get_universe())
        plans: list[TradePlan] = []

        for meta in universe:
            bars = self.p.get_daily_bars(meta.code)
            if len(bars) < self.cfg.MA_LONG + 5:
                continue

            gate_ok, _ = self.trend_gate.passes(bars)
            st = self.states.setdefault(meta.code, SymbolState(code=meta.code))
            st, fired, _ = self.engine.update(st, bars, gate_ok)

            if not fired:
                continue

            # 수급 확인
            flow = self.p.get_flow(meta.code)
            sscore, _ = self.supply.score(bars, flow)
            if sscore < min_supply_score:
                continue  # 매집 확인 부족 → 진입 보류

            # 리스크 플랜
            entry = bars[-1].close
            support_low = st.support_test_low or bars[-1].low
            plan = self.risk.build_plan(meta, entry, support_low, sscore, scale,
                                        self.open_positions + len(plans))
            if plan is None:
                continue

            plans.append(plan)
            self.notifier.send(self.notifier.format_signal(plan, macro_note))

        if not plans:
            self.notifier.send("⚪ [퀀트봇] 오늘 조건 충족 종목 없음. (준비된 자에게 기회가 온다 — 기다린다)")
        return plans


# ===========================================================================
# 13. 데모 (MockDataProvider) — 바로 실행해 흐름 검증
# ===========================================================================
class MockDataProvider(DataProvider):
    """가짜 데이터로 '정배열 → 완만한 눌림 → 지지테스트 → 반등' 시나리오를 만들어 신호 발화 확인.

    as_of 인덱스로 '그날까지 보이는 데이터'를 흉내내, 상태머신이 하루씩 진행되게 한다.
    """

    def __init__(self):
        self._all = self._make_setup_bars()
        self.as_of = len(self._all)  # 현재 보이는 봉 개수 (데모에서 하루씩 늘림)

    def _make_setup_bars(self) -> list[Bar]:
        bars: list[Bar] = []
        price = 50_000
        # 단계1: 장기 상승 추세 (정배열 + ADX 형성)
        for i in range(150):
            price *= 1.003
            bars.append(Bar(f"D{i}", price * 0.997, price * 1.008, price * 0.993, price, 1_000_000))
        # 단계2: 20일선으로 '완만한' 눌림 (거래량 감소 = 건강한 눌림)
        for i in range(4):
            price *= 0.995
            bars.append(Bar(f"P{i}", price * 1.001, price * 1.004, price * 0.994, price, 550_000))
        # 단계3: 지지 테스트 (저점 찍고 종가 사수)
        bars.append(Bar("ST", price * 0.999, price * 1.001, price * 0.990, price * 0.998, 600_000))
        # 단계4: 반등 확인 (양봉 + 거래량 폭증 + 전일고가 돌파)
        price = price * 1.03
        bars.append(Bar("BNC", price * 0.985, price * 1.005, price * 0.983, price, 2_000_000))
        return bars

    def get_universe(self) -> list[SymbolMeta]:
        return [SymbolMeta("000660", "테스트소부장", 60_000, 50_000_000_000,
                           2_000_000_000_000, False, True)]

    def get_daily_bars(self, code: str, count: int = 250) -> list[Bar]:
        return self._all[:self.as_of]

    def get_flow(self, code: str, days: int = 10) -> FlowData:
        return FlowData(
            foreign=[1e8, 2e8, 3e8],         # 외인 3일 연속 매수
            institution=[0.5e8, 1e8, 1.2e8],
            pension=[0, 0.3e8, 0.5e8],
        )

    def get_macro_score(self) -> float:
        return 2.0  # 매크로 정상


if __name__ == "__main__":
    provider = MockDataProvider()
    bot = QuantBot(provider, QuantConfig())
    total = len(provider._all)
    print("=" * 60)
    print(" 퀀트봇 데모 실행 (정배열→완만 눌림→지지테스트→반등)")
    print("=" * 60)
    # 마지막 8일을 하루씩 순차 진입시켜 상태머신을 흘려보낸다
    for day in range(total - 8, total + 1):
        provider.as_of = day
        bot.run_daily_scan()
    print("\n데모 종료 — 위에 🟢 매수 신호(반등 확인)가 떠야 정상.")
