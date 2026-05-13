"""역발상 저점매집 전략 — 백테스트.

기간: 2024-03 ~ 2026-03 (2년)
대상: 84종목 parquet + KOSPI + VIX
BRAIN 레짐 연동 포함.

사용법:
  python -u -X utf8 scripts/backtest_contrarian.py
  python -u -X utf8 scripts/backtest_contrarian.py --start 2024-06 --end 2026-03
  python -u -X utf8 scripts/backtest_contrarian.py --no-brain   # BRAIN 레짐 무시
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.contrarian.config import (
    COMMISSION_PCT,
    COOLDOWN_DAYS,
    DUAL_STOPLOSS_PAUSE,
    MACRO_ENTRY_THRESHOLD,
    MACRO_ENTRY_THRESHOLD_BEAR,
    MAX_PER_STOCK_PCT,
    MAX_STOCKS,
    MONTHLY_LOSS_LIMIT,
    REGIME_CAP,
    SLIPPAGE_PCT,
    SLOT_CAPITAL_PCT,
    SLOT_MDD_LIMIT,
    SPLIT_BUY,
    STOCK_ENTRY_THRESHOLD,
    STOP_LOSS_PCT,
    TAX_PCT,
    TAKE_PROFIT_TIERS,
    TRAILING_ACTIVATE_PCT,
    TRAILING_STOP_PCT,
)


# ═══════════════════════════════════════════════
# 데이터 로더
# ═══════════════════════════════════════════════

def load_all_data(
    start: str,
    end: str,
    min_market_cap: float = 1000,  # 억원 단위. 기본 1000억
    min_trading_value: float | None = None,  # None이면 시총 필터만 사용
):
    """시총 필터링된 종목 + KOSPI + US(VIX) 로드.

    Args:
        min_market_cap: 시총 기준 (억원). 기본 1000억. 0이면 필터 안함.
        min_trading_value: 평균 거래대금 기준 (원). None이면 시총 필터만 사용.
    """
    import json as _json

    proc_dir = PROJECT_ROOT / "data" / "processed"
    parquets = sorted(proc_dir.glob("*.parquet"))

    stocks: dict[str, pd.DataFrame] = {}
    stock_names: dict[str, str] = {}

    # 시총 캐시 로드
    cap_cache_path = PROJECT_ROOT / "data" / "market_cap_cache.json"
    cap_cache: dict[str, dict] = {}
    if cap_cache_path.exists():
        with open(cap_cache_path, "r", encoding="utf-8") as f:
            cap_cache = _json.load(f)

    # 종목명 매핑 (CSV에서 + 시총 캐시에서)
    csv_dir = PROJECT_ROOT / "stock_data_daily"
    name_map = {}
    if csv_dir.exists():
        for csv_f in csv_dir.glob("*.csv"):
            parts = csv_f.stem.split("_", 1)
            if len(parts) == 2:
                name_map[parts[0]] = parts[1]
    # 시총 캐시의 종목명도 보완
    for code, info in cap_cache.items():
        if code not in name_map and info.get("name"):
            name_map[code] = info["name"]

    skipped_cap = 0
    skipped_data = 0
    for pf in parquets:
        code = pf.stem

        # 시총 필터
        if min_market_cap > 0:
            cap_info = cap_cache.get(code, {})
            avls = cap_info.get("hts_avls", 0)
            if avls < min_market_cap:
                skipped_cap += 1
                continue

        df = pd.read_parquet(pf)
        df = df[(df.index >= start) & (df.index <= end)]
        if len(df) < 60:
            skipped_data += 1
            continue

        # 거래대금 필터 (선택적)
        if min_trading_value is not None:
            avg_tv = df["trading_value"].mean() if "trading_value" in df.columns else 0
            # trading_value=0인 종목은 close*volume으로 대체 추정
            if avg_tv == 0 and "close" in df.columns and "volume" in df.columns:
                avg_tv = (df["close"] * df["volume"]).mean()
            if avg_tv < min_trading_value:
                skipped_data += 1
                continue

        stocks[code] = df
        stock_names[code] = name_map.get(code, code)

    print(f"  [필터] 시총 {min_market_cap}억+ 미달: {skipped_cap}개, "
          f"데이터 부족: {skipped_data}개")

    # KOSPI
    kospi = pd.read_csv(
        PROJECT_ROOT / "data" / "kospi_index.csv",
        index_col=0, parse_dates=True,
    )
    kospi = kospi[(kospi.index >= start) & (kospi.index <= end)]

    # US Market (VIX)
    us = pd.read_parquet(PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet")
    us = us[(us.index >= start) & (us.index <= end)]

    return stocks, stock_names, kospi, us


# ═══════════════════════════════════════════════
# BRAIN 레짐 계산 (KOSPI 기반)
# ═══════════════════════════════════════════════

def calc_brain_regime(kospi: pd.DataFrame) -> pd.Series:
    """KOSPI 데이터로 레짐 판정.

    - MA20 위 + RV20<50%ile → BULL
    - MA20 위 + RV20≥50%ile → CAUTION
    - MA20~MA60 사이 → BEAR
    - MA60 아래 → CRISIS
    """
    close = kospi["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # Realized Volatility 20일
    log_ret = np.log(close / close.shift(1))
    rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
    rv20_pctile = rv20.rolling(120, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    regime = pd.Series("CAUTION", index=kospi.index)

    for i in range(len(kospi)):
        if pd.isna(ma20.iloc[i]) or pd.isna(ma60.iloc[i]):
            continue

        c = close.iloc[i]
        m20 = ma20.iloc[i]
        m60 = ma60.iloc[i]
        pctile = rv20_pctile.iloc[i] if not pd.isna(rv20_pctile.iloc[i]) else 0.5

        if c < m60:
            regime.iloc[i] = "CRISIS"
        elif c < m20:
            regime.iloc[i] = "BEAR"
        elif pctile >= 0.50:
            regime.iloc[i] = "CAUTION"
        else:
            regime.iloc[i] = "BULL"

    return regime


# ═══════════════════════════════════════════════
# 매크로 공포 점수
# ═══════════════════════════════════════════════

def calc_macro_fear_score(
    date: pd.Timestamp,
    kospi: pd.DataFrame,
    us: pd.DataFrame,
    stocks: dict[str, pd.DataFrame],
) -> tuple[int, dict]:
    """일별 매크로 공포 점수 계산. (60점 만점)"""

    details = {}
    total = 0

    # 1. KOSPI 5일 낙폭
    kospi_up_to = kospi[kospi.index <= date]
    if len(kospi_up_to) >= 6:
        ret_5d = (kospi_up_to["close"].iloc[-1] / kospi_up_to["close"].iloc[-6] - 1) * 100
        if ret_5d <= -5:
            total += 15
            details["kospi_drop"] = f"{ret_5d:.1f}% (15pt)"
        elif ret_5d <= -3:
            total += 10
            details["kospi_drop"] = f"{ret_5d:.1f}% (10pt)"

    # 2. 외국인 5일 연속 순매도 (유니버스 중위값)
    foreign_sells = 0
    count = 0
    for code, df in stocks.items():
        if date in df.index:
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                idx = idx.start
            if idx >= 0:
                val = df["foreign_net_5d"].iloc[idx]
                if not pd.isna(val):
                    count += 1
                    if val < 0:
                        foreign_sells += 1
    if count > 0 and foreign_sells / count > 0.70:  # 70% 이상 종목이 외국인 순매도
        total += 10
        details["foreign_sell"] = f"{foreign_sells}/{count} ({foreign_sells/count*100:.0f}%, 10pt)"

    # 3. VIX (시차 보정: 한국 장중 기준 전일 미국 VIX)
    # 한국 날짜보다 1~2일 전 미국 데이터 사용
    us_up_to = us[us.index <= date]
    if len(us_up_to) > 0:
        vix = us_up_to["vix_close"].iloc[-1]
        if not pd.isna(vix):
            if vix >= 30:
                total += 15  # 10 + 5 bonus
                details["vix"] = f"{vix:.1f} (15pt)"
            elif vix >= 25:
                total += 10
                details["vix"] = f"{vix:.1f} (10pt)"

    # 4. 공포탐욕 지수 (간이 계산)
    fear_greed = _calc_fear_greed(date, kospi, us, stocks)
    if fear_greed <= 25:
        total += 15
        details["fear_greed"] = f"{fear_greed:.0f} (15pt)"
    elif fear_greed <= 35:
        total += 10
        details["fear_greed"] = f"{fear_greed:.0f} (10pt)"
    elif fear_greed <= 40:
        total += 5
        details["fear_greed"] = f"{fear_greed:.0f} (5pt)"

    return total, details


def _calc_fear_greed(
    date: pd.Timestamp,
    kospi: pd.DataFrame,
    us: pd.DataFrame,
    stocks: dict[str, pd.DataFrame],
) -> float:
    """간이 공포탐욕 지수 (0~100, 낮을수록 공포)."""
    scores = []

    # 1. VIX 반전 (VIX 높을수록 공포)
    us_up_to = us[us.index <= date]
    if len(us_up_to) > 0:
        vix = us_up_to["vix_close"].iloc[-1]
        if not pd.isna(vix):
            # VIX 12=탐욕(90), 20=중립(50), 30=공포(20), 40=극공포(5)
            vix_score = max(0, min(100, 100 - (vix - 12) * 3.5))
            scores.append(vix_score)

    # 2. KOSPI 20일 모멘텀
    kospi_up_to = kospi[kospi.index <= date]
    if len(kospi_up_to) >= 21:
        mom = (kospi_up_to["close"].iloc[-1] / kospi_up_to["close"].iloc[-21] - 1) * 100
        # +5%=80, 0%=50, -5%=20, -10%=5
        mom_score = max(0, min(100, 50 + mom * 5))
        scores.append(mom_score)

    # 2b. KOSPI 5일 급락 (단기 공포 가중)
    if len(kospi_up_to) >= 6:
        mom5 = (kospi_up_to["close"].iloc[-1] / kospi_up_to["close"].iloc[-6] - 1) * 100
        # -5%=15, -3%=30, 0%=50
        short_score = max(0, min(100, 50 + mom5 * 7))
        scores.append(short_score)

    # 3. RSI breadth (RSI<30 종목 비율)
    rsi_below30 = 0
    total_valid = 0
    for code, df in stocks.items():
        if date in df.index:
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                idx = idx.start
            rsi = df["rsi_14"].iloc[idx]
            if not pd.isna(rsi):
                total_valid += 1
                if rsi < 30:
                    rsi_below30 += 1
    if total_valid > 0:
        rsi_pct = rsi_below30 / total_valid
        # 0%=70, 10%=40, 20%=10
        breadth_score = max(0, min(100, 70 - rsi_pct * 300))
        scores.append(breadth_score)

    return np.mean(scores) if scores else 50.0


# ═══════════════════════════════════════════════
# 종목 레벨 과매도 스크리닝
# ═══════════════════════════════════════════════

def screen_stocks(
    date: pd.Timestamp,
    stocks: dict[str, pd.DataFrame],
    stock_names: dict[str, str],
) -> list[dict]:
    """과매도 종목 스크리닝. score ≥ 18 반환."""
    candidates = []

    for code, df in stocks.items():
        if date not in df.index:
            continue

        idx = df.index.get_loc(date)
        if isinstance(idx, slice):
            idx = idx.start

        row = df.iloc[idx]
        score = 0
        reasons = []

        # 1. RSI ≤ 30
        if not pd.isna(row.get("rsi_14")) and row["rsi_14"] <= 30:
            score += 8
            reasons.append(f"RSI {row['rsi_14']:.1f}")

        # 2. 볼린저밴드 하단 이탈
        if not pd.isna(row.get("bb_position")) and row["bb_position"] <= 0:
            score += 5
            reasons.append(f"BB하단이탈 {row['bb_position']:.2f}")

        # 3. 52주 고점 대비 -30% 이상
        if not pd.isna(row.get("pct_of_52w_high")):
            drawdown = row["pct_of_52w_high"] - 100  # pct_of_52w_high가 70이면 -30%
            if drawdown <= -30:
                score += 7
                reasons.append(f"52주고점-{abs(drawdown):.0f}%")

        # 4. 거래량 200% 이상 (투매 신호)
        if not pd.isna(row.get("volume_surge_ratio")) and row["volume_surge_ratio"] >= 2.0:
            score += 5
            reasons.append(f"거래량{row['volume_surge_ratio']:.1f}x")

        # 5. 이격도 ≤ 92 (close/sma20 ≤ 0.92)
        if not pd.isna(row.get("sma_20")) and row["sma_20"] > 0:
            disparity = row["close"] / row["sma_20"] * 100
            if disparity <= 92:
                score += 5
                reasons.append(f"이격도{disparity:.0f}")

        if score >= STOCK_ENTRY_THRESHOLD:
            candidates.append({
                "code": code,
                "name": stock_names.get(code, code),
                "score": score,
                "price": row["close"],
                "reasons": reasons,
            })

    # 점수 내림차순
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════
# 포지션 관리
# ═══════════════════════════════════════════════

@dataclass
class Position:
    code: str
    name: str
    entries: list = field(default_factory=list)  # [(date, price, qty, phase)]
    avg_price: float = 0.0
    total_qty: int = 0
    peak_price: float = 0.0  # 트레일링용 최고가
    tier1_sold: bool = False
    tier2_sold: bool = False
    trailing_active: bool = False

    def add_entry(self, date, price, qty, phase):
        cost = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
        self.entries.append((date, cost, qty, phase))
        total_cost = sum(p * q for _, p, q, _ in self.entries)
        self.total_qty = sum(q for _, _, q, _ in self.entries)
        self.avg_price = total_cost / self.total_qty if self.total_qty > 0 else 0
        self.peak_price = max(self.peak_price, price)

    @property
    def current_phase(self) -> int:
        return max((ph for _, _, _, ph in self.entries), default=0)

    def invested_amount(self) -> float:
        return sum(p * q for _, p, q, _ in self.entries)


@dataclass
class SlotState:
    initial_capital: float
    capital: float              # 현금
    positions: dict = field(default_factory=dict)  # code → Position
    trades: list = field(default_factory=list)      # 완료된 거래 기록
    daily_equity: list = field(default_factory=list)  # (date, equity)
    cooldown_until: str = ""    # 킬스위치 냉각기
    monthly_loss: dict = field(default_factory=lambda: defaultdict(float))
    stoploss_today: int = 0     # 오늘 손절 횟수
    paused: bool = False
    fear_scores: list = field(default_factory=list)   # (date, score, details)

    def total_equity(self, prices: dict) -> float:
        eq = self.capital
        for code, pos in self.positions.items():
            if code in prices:
                eq += prices[code] * pos.total_qty
        return eq

    def invested_amount(self) -> float:
        return sum(pos.invested_amount() for pos in self.positions.values())


# ═══════════════════════════════════════════════
# 백테스트 엔진
# ═══════════════════════════════════════════════

def run_backtest(
    start: str = "2024-03-01",
    end: str = "2026-03-08",
    total_account: float = 100_000_000,  # 1억
    use_brain: bool = True,
    min_market_cap: float = 1000,  # 억원
):
    print("=" * 65)
    print("  역발상 저점매집 전략 백테스트")
    print(f"  기간: {start} ~ {end}")
    print(f"  계좌: {total_account:,.0f}원 | 슬롯: {SLOT_CAPITAL_PCT*100:.0f}%")
    print(f"  BRAIN 연동: {'ON' if use_brain else 'OFF'}")
    print(f"  유니버스: 시총 {min_market_cap:,.0f}억+ 필터")
    print("=" * 65)

    # 데이터 로드
    stocks, stock_names, kospi, us = load_all_data(
        start, end, min_market_cap=min_market_cap,
    )
    print(f"  종목: {len(stocks)}개 | KOSPI: {len(kospi)}일 | US: {len(us)}일")

    # BRAIN 레짐 계산
    if use_brain:
        regimes = calc_brain_regime(kospi)
    else:
        regimes = pd.Series("CAUTION", index=kospi.index)

    # 슬롯 초기화
    slot_capital = total_account * SLOT_CAPITAL_PCT
    state = SlotState(initial_capital=slot_capital, capital=slot_capital)

    # 거래일 리스트
    trading_days = sorted(kospi.index)
    print(f"  거래일: {len(trading_days)}일")
    print()

    # 일별 시뮬레이션
    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")
        month_key = day.strftime("%Y-%m")

        # ── 킬스위치 체크 ──
        if state.cooldown_until and day_str < state.cooldown_until:
            _update_daily_equity(state, day, stocks)
            continue

        if state.paused:
            _update_daily_equity(state, day, stocks)
            continue

        # 월간 손실 한도 체크
        if state.monthly_loss[month_key] <= MONTHLY_LOSS_LIMIT:
            _update_daily_equity(state, day, stocks)
            continue

        state.stoploss_today = 0

        # ── 1. BRAIN 레짐 확인 ──
        regime = regimes.get(day, "CAUTION")
        regime_mult = REGIME_CAP.get(regime, 1.0)

        effective_capital = state.initial_capital * regime_mult
        current_invested = state.invested_amount()

        # ── 2. 보유 종목 관리 (매도 체크) ──
        codes_to_close = []
        for code, pos in list(state.positions.items()):
            if code not in stocks or day not in stocks[code].index:
                continue

            idx = stocks[code].index.get_loc(day)
            if isinstance(idx, slice):
                idx = idx.start
            price = stocks[code]["close"].iloc[idx]

            # 최고가 갱신
            pos.peak_price = max(pos.peak_price, price)

            pnl_pct = price / pos.avg_price - 1

            # 손절 -7%
            if pnl_pct <= STOP_LOSS_PCT:
                sell_price = price * (1 - SLIPPAGE_PCT)
                proceeds = sell_price * pos.total_qty * (1 - TAX_PCT)
                state.capital += proceeds
                state.trades.append({
                    "code": code, "name": pos.name,
                    "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                    "exit_date": day_str,
                    "avg_price": pos.avg_price,
                    "exit_price": sell_price,
                    "qty": pos.total_qty,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "STOP_LOSS",
                    "phases": pos.current_phase,
                })
                codes_to_close.append(code)
                state.stoploss_today += 1
                continue

            # 트레일링 스탑
            if pos.trailing_active:
                trail_pnl = price / pos.peak_price - 1
                if trail_pnl <= TRAILING_STOP_PCT:
                    sell_price = price * (1 - SLIPPAGE_PCT)
                    proceeds = sell_price * pos.total_qty * (1 - TAX_PCT)
                    state.capital += proceeds
                    state.trades.append({
                        "code": code, "name": pos.name,
                        "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                        "exit_date": day_str,
                        "avg_price": pos.avg_price,
                        "exit_price": sell_price,
                        "qty": pos.total_qty,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "TRAILING_STOP",
                        "phases": pos.current_phase,
                    })
                    codes_to_close.append(code)
                    continue

            # 구간 익절
            if not pos.tier1_sold and pnl_pct >= TAKE_PROFIT_TIERS[0]["from"]:
                sell_qty = max(1, int(pos.total_qty * TAKE_PROFIT_TIERS[0]["sell_pct"]))
                sell_price = price * (1 - SLIPPAGE_PCT)
                proceeds = sell_price * sell_qty * (1 - TAX_PCT)
                state.capital += proceeds
                pos.total_qty -= sell_qty
                pos.tier1_sold = True
                if pos.total_qty <= 0:
                    state.trades.append({
                        "code": code, "name": pos.name,
                        "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                        "exit_date": day_str,
                        "avg_price": pos.avg_price,
                        "exit_price": sell_price,
                        "qty": sell_qty,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "TAKE_PROFIT_T1",
                        "phases": pos.current_phase,
                    })
                    codes_to_close.append(code)
                    continue

            if not pos.tier2_sold and pnl_pct >= TAKE_PROFIT_TIERS[1]["from"]:
                sell_qty = max(1, int(pos.total_qty * TAKE_PROFIT_TIERS[1]["sell_pct"]))
                sell_price = price * (1 - SLIPPAGE_PCT)
                proceeds = sell_price * sell_qty * (1 - TAX_PCT)
                state.capital += proceeds
                pos.total_qty -= sell_qty
                pos.tier2_sold = True
                if pos.total_qty <= 0:
                    state.trades.append({
                        "code": code, "name": pos.name,
                        "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                        "exit_date": day_str,
                        "avg_price": pos.avg_price,
                        "exit_price": sell_price,
                        "qty": sell_qty,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "TAKE_PROFIT_T2",
                        "phases": pos.current_phase,
                    })
                    codes_to_close.append(code)
                    continue

            # 트레일링 활성화
            if pnl_pct >= TRAILING_ACTIVATE_PCT:
                pos.trailing_active = True

        for code in codes_to_close:
            del state.positions[code]

        # 동시 2종목 손절 → 일시 중지
        if DUAL_STOPLOSS_PAUSE and state.stoploss_today >= 2:
            state.paused = True
            state.cooldown_until = ""  # 수동 재개 필요 → 백테스트에서는 다음달 자동 해제
            # 백테스트 편의: 5일 냉각
            cool_idx = trading_days.index(day) + COOLDOWN_DAYS
            if cool_idx < len(trading_days):
                state.cooldown_until = trading_days[cool_idx].strftime("%Y-%m-%d")
            state.paused = False  # 냉각기로 대체

        # ── 3. 분할매수 2차/3차 체크 ──
        if regime_mult > 0:
            for code, pos in list(state.positions.items()):
                if code not in stocks or day not in stocks[code].index:
                    continue
                if pos.current_phase >= 3:
                    continue  # 이미 3차까지 완료

                idx = stocks[code].index.get_loc(day)
                if isinstance(idx, slice):
                    idx = idx.start
                price = stocks[code]["close"].iloc[idx]

                next_phase = pos.current_phase + 1
                should_buy = False

                if next_phase == 2:
                    # 1차 대비 -3% 추가 하락 OR 3영업일 경과 후 반등
                    entry1_price = pos.entries[0][1]
                    entry1_date = pos.entries[0][0]
                    days_since = (day - entry1_date).days
                    if price <= entry1_price * 0.97:
                        should_buy = True
                    elif days_since >= 3 and price > pos.avg_price:
                        should_buy = True

                elif next_phase == 3:
                    # 2차 대비 -5% 추가 하락 OR 거래량 동반 양봉
                    if price <= pos.avg_price * 0.95:
                        should_buy = True
                    elif (not pd.isna(stocks[code]["volume_surge_ratio"].iloc[idx])
                          and stocks[code]["volume_surge_ratio"].iloc[idx] >= 1.5
                          and stocks[code]["is_bullish"].iloc[idx]):
                        should_buy = True

                if should_buy:
                    buy_pct = SPLIT_BUY[next_phase]["pct"]
                    buy_amount = min(
                        effective_capital * buy_pct,
                        effective_capital * MAX_PER_STOCK_PCT - pos.invested_amount(),
                        state.capital * 0.95,  # 예수금 5% 여유
                    )
                    if buy_amount > price * 1.01:
                        qty = int(buy_amount / (price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)))
                        if qty > 0:
                            cost = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT) * qty
                            state.capital -= cost
                            pos.add_entry(day, price, qty, next_phase)

        # ── 4. 신규 진입 (매크로 공포 → 종목 스크리닝) ──
        if (regime_mult > 0
            and len(state.positions) < MAX_STOCKS
            and state.capital > effective_capital * 0.10):

            fear_score, fear_details = calc_macro_fear_score(day, kospi, us, stocks)

            # 레짐별 적응형 임계치: BEAR에서는 공포 초기(20pt)에 진입 허용
            entry_threshold = (
                MACRO_ENTRY_THRESHOLD_BEAR if regime == "BEAR"
                else MACRO_ENTRY_THRESHOLD
            )

            # 공포 점수 기록 (항상)
            if fear_score >= 15:
                state.fear_scores.append((day_str, fear_score, fear_details, regime))

            if fear_score >= entry_threshold:

                candidates = screen_stocks(day, stocks, stock_names)
                # 이미 보유 중인 종목 제외
                candidates = [c for c in candidates if c["code"] not in state.positions]

                slots_available = MAX_STOCKS - len(state.positions)
                for cand in candidates[:slots_available]:
                    buy_pct = SPLIT_BUY[1]["pct"]
                    buy_amount = min(
                        effective_capital * buy_pct,
                        state.capital * 0.95,
                    )
                    price = cand["price"]
                    if buy_amount > price * 1.01:
                        qty = int(buy_amount / (price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)))
                        if qty > 0:
                            pos = Position(code=cand["code"], name=cand["name"])
                            cost = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT) * qty
                            state.capital -= cost
                            pos.add_entry(day, price, qty, 1)
                            state.positions[cand["code"]] = pos

        # ── 5. 일일 자산 기록 ──
        _update_daily_equity(state, day, stocks)

        # 월간 손실 추적
        if len(state.daily_equity) >= 2:
            prev_eq = state.daily_equity[-2][1]
            curr_eq = state.daily_equity[-1][1]
            daily_ret = curr_eq / prev_eq - 1 if prev_eq > 0 else 0
            state.monthly_loss[month_key] += daily_ret

        # 슬롯 MDD 킬스위치
        if len(state.daily_equity) > 0:
            eq = state.daily_equity[-1][1]
            peak = max(e for _, e in state.daily_equity)
            mdd = eq / peak - 1 if peak > 0 else 0
            if mdd <= SLOT_MDD_LIMIT:
                # 전량 매도
                for code, pos in list(state.positions.items()):
                    if code in stocks and day in stocks[code].index:
                        idx = stocks[code].index.get_loc(day)
                        if isinstance(idx, slice):
                            idx = idx.start
                        price = stocks[code]["close"].iloc[idx]
                        sell_price = price * (1 - SLIPPAGE_PCT)
                        proceeds = sell_price * pos.total_qty * (1 - TAX_PCT)
                        state.capital += proceeds
                        pnl_pct = sell_price / pos.avg_price - 1
                        state.trades.append({
                            "code": code, "name": pos.name,
                            "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                            "exit_date": day_str,
                            "avg_price": pos.avg_price,
                            "exit_price": sell_price,
                            "qty": pos.total_qty,
                            "pnl_pct": pnl_pct,
                            "exit_reason": "KILLSWITCH_MDD",
                            "phases": pos.current_phase,
                        })
                state.positions.clear()
                cool_idx = trading_days.index(day) + COOLDOWN_DAYS
                if cool_idx < len(trading_days):
                    state.cooldown_until = trading_days[cool_idx].strftime("%Y-%m-%d")

    # ── 잔여 포지션 강제 청산 ──
    last_day = trading_days[-1]
    for code, pos in list(state.positions.items()):
        if code in stocks and last_day in stocks[code].index:
            idx = stocks[code].index.get_loc(last_day)
            if isinstance(idx, slice):
                idx = idx.start
            price = stocks[code]["close"].iloc[idx]
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * pos.total_qty * (1 - TAX_PCT)
            state.capital += proceeds
            pnl_pct = sell_price / pos.avg_price - 1
            state.trades.append({
                "code": code, "name": pos.name,
                "entry_date": pos.entries[0][0].strftime("%Y-%m-%d"),
                "exit_date": last_day.strftime("%Y-%m-%d"),
                "avg_price": pos.avg_price,
                "exit_price": sell_price,
                "qty": pos.total_qty,
                "pnl_pct": pnl_pct,
                "exit_reason": "BACKTEST_END",
                "phases": pos.current_phase,
            })
    state.positions.clear()

    # 결과 출력
    _print_results(state, start, end, use_brain)
    return state


def _update_daily_equity(state: SlotState, day, stocks):
    prices = {}
    for code in state.positions:
        if code in stocks and day in stocks[code].index:
            idx = stocks[code].index.get_loc(day)
            if isinstance(idx, slice):
                idx = idx.start
            prices[code] = stocks[code]["close"].iloc[idx]
    eq = state.total_equity(prices)
    state.daily_equity.append((day, eq))


# ═══════════════════════════════════════════════
# 결과 출력
# ═══════════════════════════════════════════════

def _print_results(state: SlotState, start: str, end: str, use_brain: bool):
    trades = state.trades
    initial = state.initial_capital

    if not trades:
        print("\n거래 없음!")
        print(f"공포 시그널 발생 횟수: {len(state.fear_scores)}")
        if state.fear_scores:
            for d, s, det in state.fear_scores[:10]:
                print(f"  {d}: {s}점 — {det}")
        return

    # 기본 통계
    n = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n * 100 if n > 0 else 0

    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) * 100 if losses else 0

    # 최종 수익률
    final_eq = state.daily_equity[-1][1] if state.daily_equity else initial
    total_return = (final_eq / initial - 1) * 100

    # MDD
    equities = [e for _, e in state.daily_equity]
    if equities:
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            peak = max(peak, eq)
            dd = (eq / peak - 1) * 100
            max_dd = min(max_dd, dd)
    else:
        max_dd = 0

    # 엑싯 유형별 통계
    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t["exit_reason"]] += 1

    # 분할매수 단계별 통계
    phase_counts = defaultdict(int)
    for t in trades:
        phase_counts[t["phases"]] += 1

    # 레짐별 거래 수 (fear_scores에서)
    # 보유 기간
    hold_days = []
    for t in trades:
        d1 = pd.Timestamp(t["entry_date"])
        d2 = pd.Timestamp(t["exit_date"])
        hold_days.append((d2 - d1).days)

    print()
    print("=" * 65)
    print("  백테스트 결과")
    print("=" * 65)
    print(f"  기간: {start} ~ {end}")
    print(f"  BRAIN 연동: {'ON' if use_brain else 'OFF'}")
    print(f"  초기 자본: {initial:,.0f}원 (계좌의 {SLOT_CAPITAL_PCT*100:.0f}%)")
    print(f"  최종 자산: {final_eq:,.0f}원")
    print()
    print(f"  총 수익률:  {total_return:+.1f}%")
    print(f"  MDD:        {max_dd:.1f}%")
    print(f"  PF:         {pf:.2f}")
    print(f"  거래 수:    {n}건")
    print(f"  승률:       {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  평균 수익:  {avg_win:+.1f}%")
    print(f"  평균 손실:  {avg_loss:+.1f}%")
    print(f"  평균 보유:  {np.mean(hold_days):.1f}일")
    print()

    print("  --- 엑싯 유형 ---")
    for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {cnt}건")
    print()

    print("  --- 분할매수 단계 ---")
    for phase, cnt in sorted(phase_counts.items()):
        print(f"    {phase}차 매수까지: {cnt}건")
    print()

    print(f"  공포 시그널 발생: {len(state.fear_scores)}회")
    print()

    # 개별 거래 상세
    print("  --- 거래 상세 (최근 20건) ---")
    for t in trades[-20:]:
        marker = "+" if t["pnl_pct"] > 0 else ""
        print(
            f"    {t['entry_date']}~{t['exit_date']} "
            f"{t['name'][:6]:>6s} "
            f"P{t['phases']} "
            f"{marker}{t['pnl_pct']*100:+.1f}% "
            f"[{t['exit_reason']}]"
        )
    print()

    # 공포 시그널 상세 (최근 15건)
    if state.fear_scores:
        print("  --- 공포 시그널 (최근 15건) ---")
        for item in state.fear_scores[-15:]:
            d, s, det = item[0], item[1], item[2]
            reg = item[3] if len(item) > 3 else "?"
            detail_str = ", ".join(f"{k}={v}" for k, v in det.items())
            blocked = " [BLOCKED]" if reg in ("CRISIS", "PANIC") else ""
            print(f"    {d} [{reg}]: {s}점{blocked} — {detail_str}")

    # 통과 기준 판정
    print()
    print("=" * 65)
    passed = pf >= 1.5 and max_dd >= -15 and win_rate >= 45
    if passed:
        print(f"  통과 기준: PF≥1.5({pf:.2f}) MDD≥-15%({max_dd:.1f}%) 승률≥45%({win_rate:.1f}%)")
        print("  ✓ 백테스트 통과")
    else:
        fails = []
        if pf < 1.5:
            fails.append(f"PF {pf:.2f}<1.5")
        if max_dd < -15:
            fails.append(f"MDD {max_dd:.1f}%<-15%")
        if win_rate < 45:
            fails.append(f"승률 {win_rate:.1f}%<45%")
        print(f"  통과 기준 미달: {', '.join(fails)}")
        print("  × 파라미터 조정 필요")
    print("=" * 65)


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="역발상 저점매집 백테스트")
    parser.add_argument("--start", default="2024-03-01")
    parser.add_argument("--end", default="2026-03-08")
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--no-brain", action="store_true", help="BRAIN 레짐 연동 비활성화")
    parser.add_argument("--min-cap", type=float, default=1000,
                        help="시총 기준 (억원). 기본 1000억. 0이면 전종목")
    args = parser.parse_args()

    import time as _time
    t0 = _time.perf_counter()
    run_backtest(
        start=args.start,
        end=args.end,
        total_account=args.capital,
        use_brain=not args.no_brain,
        min_market_cap=args.min_cap,
    )
    elapsed = _time.perf_counter() - t0
    print(f"\n  처리 시간: {elapsed:.1f}초")
