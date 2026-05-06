"""
Paper Trader — 실시간 페이퍼 트레이딩 엔진
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[실행]
  pip install yfinance
  python paper_trader.py

[동작]
  장중(09:05~15:20): yfinance 실시간 5분봉 수집 → Body Hunter 실행
  장외:             어제 데이터로 시뮬레이션 (테스트 가능)

[특징]
  - KIS API, 계좌 연결 불필요
  - 실제 시장 데이터 기반 (yfinance .KS)
  - 가상 자금 100만원으로 시작
  - 터미널 실시간 대시보드
  - 결과 자동 저장 (paper_trades.csv)
"""

import os, sys, time, json, csv
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("paper_trader")

# ══════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════

CONFIG = {
    "initial_capital": 1_000_000,   # 초기 가상 자금
    "risk_per_trade":   50_000,     # 거래당 리스크
    "top_n":            3,          # 동시 감시 종목 수
    "etf_min_change":   0.15,       # ETF 최소 변화율 (%)
    "refresh_seconds":  30,         # 데이터 갱신 주기 (초)
    "result_file":      "paper_trades.csv",
}

# 종목 유니버스 (yfinance .KS 형식)
UNIVERSE = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035720.KS": "카카오",
    "066570.KS": "LG전자",
    "055550.KS": "신한지주",
    "086790.KS": "하나금융지주",
    "032830.KS": "삼성생명",
    "003550.KS": "LG",
    "012330.KS": "현대모비스",
    "017670.KS": "SK텔레콤",
    "096770.KS": "SK이노베이션",
    "034730.KS": "SK",
    "028260.KS": "삼성물산",
    "009540.KS": "HD한국조선해양",
    "011170.KS": "롯데케미칼",
}
ETF_TICKER = "069500.KS"  # KODEX200


# ══════════════════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════════════════

def fetch_5min(ticker: str, bars: int = 80) -> Optional[object]:
    """yfinance 5분봉 수집"""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 5:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]

        # 오늘 장중 데이터만
        today = datetime.now().date()
        today_df = df[df.index.date == today]
        if len(today_df) >= 5:
            return today_df
        # 장외면 어제 데이터
        return df.tail(80)
    except Exception as e:
        logger.debug(f"fetch [{ticker}]: {e}")
        return None


def is_market_hours() -> bool:
    """장중 여부 (09:00~15:30)"""
    now = datetime.now()
    h, m = now.hour, now.minute
    return (9, 0) <= (h, m) <= (15, 30) and now.weekday() < 5


def is_simulation_mode() -> bool:
    """장외면 시뮬레이션 모드"""
    return not is_market_hours()


# ══════════════════════════════════════════════════
# DrawdownShield
# ══════════════════════════════════════════════════

class DrawdownShield:
    TIERS = [50_000, 25_000, 15_000, 10_000]

    def __init__(self):
        self.peak    = 0.0
        self.current = 0.0
        self.losses  = 0

    def update(self, pnl: float):
        self.current += pnl
        if self.current > self.peak:
            self.peak  = self.current
            self.losses = 0
        elif pnl < 0:
            self.losses += 1

    @property
    def risk(self) -> int:
        idx = min(self.losses // 2, len(self.TIERS) - 1)
        return self.TIERS[idx]

    @property
    def tier(self) -> int:
        return min(self.losses // 2, len(self.TIERS) - 1) + 1


# ══════════════════════════════════════════════════
# 간이 Body Hunter (의존성 없는 독립형)
# ══════════════════════════════════════════════════

@dataclass
class PaperPosition:
    ticker:      str
    name:        str
    direction:   str          # LONG / SHORT
    entry_price: float
    stop_loss:   float
    take_profit: float
    risk:        float
    entry_time:  str
    hold_bars:   int = 0
    peak_price:  float = 0.0


@dataclass
class PaperTrade:
    date:        str
    time:        str
    ticker:      str
    name:        str
    direction:   str
    entry_price: float
    exit_price:  float
    stop_loss:   float
    take_profit: float
    reason:      str
    rr:          float
    pnl:         float
    hold_bars:   int


class MiniBodyHunter:
    """
    독립형 간이 Body Hunter
    Fixed 2:1 TP + SL 0.6 구조
    """
    def __init__(self, ticker, name, direction, avg_vol,
                 sl_ratio=0.6, tp_rr=2.0,
                 vol_surge=1.5, breakeven_rr=0.3):
        self.ticker       = ticker
        self.name         = name
        self.direction    = direction
        self.avg_vol      = avg_vol
        self.sl_ratio     = sl_ratio
        self.tp_rr        = tp_rr
        self.vol_surge    = vol_surge
        self.breakeven_rr = breakeven_rr

        self.first_bar    = None
        self.position: Optional[PaperPosition] = None
        self.state        = "WATCHING"   # WATCHING / IN_TRADE / DONE
        self.breakout_confirmed = False
        self.retest_done  = False

    def update(self, bar) -> dict:
        if self.state == "DONE":
            return {"action": "DONE"}

        if self.first_bar is None:
            self.first_bar = bar
            return {"action": "WAIT"}

        if self.state == "WATCHING":
            return self._watch(bar)
        elif self.state == "IN_TRADE":
            return self._manage(bar)
        return {"action": "WAIT"}

    def _watch(self, bar) -> dict:
        fb   = self.first_bar
        mid  = (fb["high"] + fb["low"]) / 2
        rng  = fb["high"] - fb["low"]
        vol_ok = bar["volume"] >= self.avg_vol * self.vol_surge

        if self.direction == "LONG":
            breakout = bar["close"] > fb["high"]
            if breakout and vol_ok and not self.breakout_confirmed:
                self.breakout_confirmed = True
                self.breakout_level = fb["high"]
                return {"action": "WAIT", "note": "이탈확인"}

            if self.breakout_confirmed:
                # 리테스트: 이탈레벨 근처까지 되돌렸다가 다시 위에서 마감
                near = bar["low"] <= self.breakout_level * 1.003
                outside = bar["close"] > self.breakout_level
                if near and outside:
                    return self._enter(bar, fb, mid, rng, "LONG")

        else:  # SHORT
            breakout = bar["close"] < fb["low"]
            if breakout and vol_ok and not self.breakout_confirmed:
                self.breakout_confirmed = True
                self.breakout_level = fb["low"]
                return {"action": "WAIT", "note": "이탈확인"}

            if self.breakout_confirmed:
                near = bar["high"] >= self.breakout_level * 0.997
                outside = bar["close"] < self.breakout_level
                if near and outside:
                    return self._enter(bar, fb, mid, rng, "SHORT")

        return {"action": "WAIT"}

    def _enter(self, bar, fb, mid, rng, direction) -> dict:
        entry = bar["close"]
        if direction == "LONG":
            sl = entry - rng * self.sl_ratio
            tp = entry + rng * self.sl_ratio * self.tp_rr
        else:
            sl = entry + rng * self.sl_ratio
            tp = entry - rng * self.sl_ratio * self.tp_rr

        risk_pts = abs(entry - sl)
        self.position = PaperPosition(
            ticker      = self.ticker,
            name        = self.name,
            direction   = direction,
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            risk        = risk_pts,
            entry_time  = bar.name.strftime("%H:%M") if hasattr(bar.name, "strftime") else "09:05",
            peak_price  = entry,
        )
        self.state = "IN_TRADE"
        return {"action": "ENTER", "position": self.position}

    def _manage(self, bar) -> dict:
        pos = self.position
        pos.hold_bars += 1

        h, l = bar["high"], bar["low"]

        # 고점/저점 갱신
        if pos.direction == "LONG":
            pos.peak_price = max(pos.peak_price, h)
        else:
            pos.peak_price = min(pos.peak_price, l)

        # 현재 RR
        risk = abs(pos.entry_price - pos.stop_loss)
        if pos.direction == "LONG":
            current_rr = (bar["close"] - pos.entry_price) / risk if risk else 0
        else:
            current_rr = (pos.entry_price - bar["close"]) / risk if risk else 0

        # Breakeven 이동 (RR 0.3 이상 달성 시)
        if current_rr >= self.breakeven_rr:
            if pos.direction == "LONG":
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
            else:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)

        # SL 히트
        sl_hit = (l <= pos.stop_loss) if pos.direction == "LONG" else (h >= pos.stop_loss)
        tp_hit = (h >= pos.take_profit) if pos.direction == "LONG" else (l <= pos.take_profit)

        if sl_hit or tp_hit:
            if sl_hit and tp_hit:
                exit_price = pos.stop_loss  # SL 우선 (보수적)
                reason = "손절"
            elif tp_hit:
                exit_price = pos.take_profit
                reason = "익절(2R)"
            else:
                exit_price = pos.stop_loss
                reason = "손절" if exit_price < pos.entry_price and pos.direction == "LONG" \
                         else ("손절" if exit_price > pos.entry_price and pos.direction == "SHORT"
                               else "본전")

            rr = (exit_price - pos.entry_price) / risk if pos.direction == "LONG" \
                 else (pos.entry_price - exit_price) / risk

            self.state = "DONE"
            return {
                "action":     "EXIT",
                "position":   pos,
                "exit_price": exit_price,
                "reason":     reason,
                "rr":         round(rr, 2),
                "hold_bars":  pos.hold_bars,
            }

        return {"action": "HOLD", "rr": round(current_rr, 2)}


# ══════════════════════════════════════════════════
# 종목 스캐너 (간이)
# ══════════════════════════════════════════════════

def scan_candidates(etf_bar, stock_bars: dict, avg_vols: dict,
                    direction: str, top_n=3) -> list:
    """상대강도 상위 종목 추출"""
    etf_change = (etf_bar["close"] - etf_bar["open"]) / etf_bar["open"] * 100

    scored = []
    for ticker, bar in stock_bars.items():
        stock_change = (bar["close"] - bar["open"]) / bar["open"] * 100
        avg_vol = avg_vols.get(ticker, 1)
        if avg_vol == 0:
            continue

        # 방향 일치
        if direction == "LONG" and stock_change <= 0:
            continue
        if direction == "SHORT" and stock_change >= 0:
            continue

        rel_str  = abs(stock_change) / max(0.01, abs(etf_change))
        vol_ratio = bar["volume"] / avg_vol

        if rel_str < 1.5 or vol_ratio < 1.2:
            continue

        score = rel_str * 40 + vol_ratio * 40 + abs(stock_change) * 20
        scored.append((ticker, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


# ══════════════════════════════════════════════════
# 터미널 대시보드
# ══════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_dashboard(state: dict):
    clear()
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode  = "🟢 장중" if not state["sim_mode"] else "🟡 시뮬레이션"
    shield = state["shield"]

    print(f"{'━'*60}")
    print(f"  📈 Body Hunter — Paper Trader   {mode}  {now}")
    print(f"{'━'*60}")
    print(f"  가상 자금:  {state['capital']:>12,.0f}원")
    print(f"  누적 손익:  {state['total_pnl']:>+12,.0f}원")
    print(f"  Shield:    Tier {shield.tier}  "
          f"리스크 {shield.risk:,}원  연패 {shield.losses}회")
    print(f"{'─'*60}")

    # ETF 방향
    etf = state.get("etf_signal", {})
    dir_str = {"LONG": "🔴 상승장", "SHORT": "🔵 하락장", "NEUTRAL": "⚪ 중립"}.get(
        etf.get("direction", "NEUTRAL"), "⚪ 중립")
    print(f"  ETF 방향:  {dir_str}  ({etf.get('change', 0):+.2f}%)")
    print(f"{'─'*60}")

    # 활성 포지션
    positions = state.get("positions", {})
    if positions:
        print(f"  📌 활성 포지션:")
        for ticker, info in positions.items():
            pos  = info["position"]
            rr   = info.get("current_rr", 0)
            bars = pos.hold_bars
            rr_bar = "█" * max(0, int(rr * 5)) if rr > 0 else "░" * max(0, int(abs(rr) * 5))
            icon = "📈" if pos.direction == "LONG" else "📉"
            print(f"    {icon} {pos.name:<10} "
                  f"진입:{pos.entry_price:>8,.0f}  "
                  f"RR:{rr:>+5.2f}  "
                  f"{bars}봉  {rr_bar}")
    else:
        print(f"  📌 활성 포지션: 없음")

    print(f"{'─'*60}")

    # 오늘 거래 내역
    today_trades = state.get("today_trades", [])
    print(f"  📋 오늘 거래 ({len(today_trades)}건):")
    if today_trades:
        for t in today_trades[-5:]:
            icon = "✅" if t.rr > 0 else ("⬜" if t.rr == 0 else "❌")
            print(f"    {icon} {t.time} {t.name:<10} "
                  f"{t.direction}  {t.reason:<8}  RR:{t.rr:>+.2f}  "
                  f"손익:{t.pnl:>+,.0f}원")
    else:
        print(f"    (없음)")

    print(f"{'─'*60}")

    # 전체 누적 통계
    all_trades = state.get("all_trades", [])
    if all_trades:
        wins   = sum(1 for t in all_trades if t.rr > 0)
        losses = sum(1 for t in all_trades if t.rr < 0)
        be     = sum(1 for t in all_trades if t.rr == 0)
        total  = len(all_trades)
        avg_rr = sum(t.rr for t in all_trades) / total
        good   = sum(1 for t in all_trades if t.reason in ["익절(2R)"])
        print(f"  📊 누적 통계 ({total}거래):")
        print(f"    승:{wins}  패:{losses}  본전:{be}  "
              f"승률:{wins/total*100:.0f}%  "
              f"평균RR:{avg_rr:+.2f}  "
              f"익절율:{good/total*100:.0f}%")

    print(f"{'━'*60}")
    print(f"  Ctrl+C 로 종료  |  {CONFIG['refresh_seconds']}초마다 갱신")
    print(f"{'━'*60}")


# ══════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════

class PaperTrader:
    def __init__(self):
        self.shield       = DrawdownShield()
        self.capital      = CONFIG["initial_capital"]
        self.total_pnl    = 0.0
        self.hunters: Dict[str, MiniBodyHunter] = {}
        self.positions: Dict[str, dict]         = {}
        self.today_trades: List[PaperTrade]     = []
        self.all_trades:   List[PaperTrade]     = []
        self.avg_vols:     Dict[str, float]     = {}
        self.etf_signal    = {"direction": "NEUTRAL", "change": 0.0}
        self.sim_mode      = False
        self.last_date     = None

        # 결과 저장 파일 초기화
        self._init_csv()

    def _init_csv(self):
        path = CONFIG["result_file"]
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "date","time","ticker","name","direction",
                    "entry_price","exit_price","reason","rr","pnl","hold_bars"
                ])
                writer.writeheader()

    def _save_trade(self, trade: PaperTrade):
        with open(CONFIG["result_file"], "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date","time","ticker","name","direction",
                "entry_price","exit_price","reason","rr","pnl","hold_bars"
            ])
            d = asdict(trade)
            d.pop("stop_loss", None)
            d.pop("take_profit", None)
            writer.writerow(d)

    def _reset_day(self):
        """새 거래일 초기화"""
        self.hunters      = {}
        self.positions    = {}
        self.today_trades = []
        self.avg_vols     = {}
        self.etf_signal   = {"direction": "NEUTRAL", "change": 0.0}

    def _calc_avg_vol(self):
        """평균 거래량 계산 (5일치)"""
        import yfinance as yf
        tickers = [ETF_TICKER] + list(UNIVERSE.keys())
        try:
            data = yf.download(tickers, period="5d", interval="1d",
                               progress=False, auto_adjust=True)
            if data.empty:
                return
            vol = data["Volume"] if "Volume" in data else data.xs("volume", axis=1, level=0)
            for t in tickers:
                if t in vol.columns:
                    self.avg_vols[t] = float(vol[t].mean())
        except Exception as e:
            logger.debug(f"avg_vol: {e}")

    def tick(self):
        """한 틱 실행 (5분마다 호출)"""
        now   = datetime.now()
        today = now.date()

        # 새 거래일 체크
        if self.last_date != today:
            self._reset_day()
            self.last_date = today
            self._calc_avg_vol()

        self.sim_mode = is_simulation_mode()

        # ETF 데이터
        etf_df = fetch_5min(ETF_TICKER)
        if etf_df is None or len(etf_df) < 2:
            return

        etf_first = etf_df.iloc[0]
        etf_last  = etf_df.iloc[-1]
        etf_change = (etf_last["close"] - etf_first["open"]) / etf_first["open"] * 100

        if abs(etf_change) < CONFIG["etf_min_change"]:
            self.etf_signal = {"direction": "NEUTRAL", "change": etf_change}
            return

        direction = "LONG" if etf_change > 0 else "SHORT"
        self.etf_signal = {"direction": direction, "change": etf_change}

        # 09:05 이후 스캔 (첫봉 마감 후)
        if len(etf_df) < 2:
            return

        # 전 종목 최신봉 수집
        stock_latest = {}
        stock_dfs    = {}
        for ticker in UNIVERSE:
            df = fetch_5min(ticker)
            if df is not None and len(df) >= 2:
                stock_latest[ticker] = df.iloc[-1]
                stock_dfs[ticker]    = df

        # 신규 후보 스캔 (아직 감시 중이 아닌 종목만)
        if len(self.hunters) < CONFIG["top_n"]:
            cands = scan_candidates(
                etf_last, stock_latest, self.avg_vols,
                direction, CONFIG["top_n"]
            )
            for ticker in cands:
                if ticker not in self.hunters:
                    name = UNIVERSE[ticker]
                    avg_vol = self.avg_vols.get(ticker, 500_000)
                    self.hunters[ticker] = MiniBodyHunter(
                        ticker    = ticker,
                        name      = name,
                        direction = direction,
                        avg_vol   = avg_vol,
                    )
                    # 과거 봉 feed
                    df = stock_dfs.get(ticker)
                    if df is not None:
                        for _, row in df.iterrows():
                            self.hunters[ticker].update(row)

        # 활성 Hunter 업데이트 (최신봉)
        for ticker, hunter in list(self.hunters.items()):
            df = stock_dfs.get(ticker)
            if df is None:
                continue
            bar = df.iloc[-1]
            result = hunter.update(bar)

            if result["action"] == "ENTER":
                pos = result["position"]
                risk = self.shield.risk
                self.positions[ticker] = {"position": pos, "current_rr": 0.0, "risk": risk}

            elif result["action"] == "EXIT":
                pos       = result["position"]
                rr        = result["rr"]
                risk      = self.positions.get(ticker, {}).get("risk", self.shield.risk)
                pnl       = risk * rr
                self.total_pnl += pnl
                self.shield.update(pnl)

                trade = PaperTrade(
                    date        = today.strftime("%Y-%m-%d"),
                    time        = now.strftime("%H:%M"),
                    ticker      = ticker,
                    name        = pos.name,
                    direction   = pos.direction,
                    entry_price = pos.entry_price,
                    exit_price  = result["exit_price"],
                    stop_loss   = pos.stop_loss,
                    take_profit = pos.take_profit,
                    reason      = result["reason"],
                    rr          = rr,
                    pnl         = pnl,
                    hold_bars   = result["hold_bars"],
                )
                self.today_trades.append(trade)
                self.all_trades.append(trade)
                self._save_trade(trade)

                if ticker in self.positions:
                    del self.positions[ticker]
                del self.hunters[ticker]

            elif result["action"] == "HOLD":
                if ticker in self.positions:
                    self.positions[ticker]["current_rr"] = result.get("rr", 0)

        # 대시보드 출력
        print_dashboard({
            "capital":      self.capital + self.total_pnl,
            "total_pnl":    self.total_pnl,
            "shield":       self.shield,
            "etf_signal":   self.etf_signal,
            "positions":    self.positions,
            "today_trades": self.today_trades,
            "all_trades":   self.all_trades,
            "sim_mode":     self.sim_mode,
        })

    def run(self):
        print("Body Hunter Paper Trader 시작...")
        print(f"데이터: yfinance (실시간 or 어제 데이터)")
        print(f"종목: {len(UNIVERSE)}개 유니버스")
        print(f"갱신: {CONFIG['refresh_seconds']}초마다")
        print("Ctrl+C 로 종료\n")

        try:
            while True:
                try:
                    self.tick()
                except Exception as e:
                    logger.error(f"tick 오류: {e}")
                time.sleep(CONFIG["refresh_seconds"])
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            if self.all_trades:
                print(f"\n총 {len(self.all_trades)}거래 기록 → {CONFIG['result_file']}")
                wins = sum(1 for t in self.all_trades if t.rr > 0)
                total = len(self.all_trades)
                print(f"승률: {wins}/{total} ({wins/total*100:.0f}%)")
                print(f"총 손익: {self.total_pnl:+,.0f}원")


# ══════════════════════════════════════════════════
# 엔트리포인트
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # yfinance 설치 확인
    try:
        import yfinance
    except ImportError:
        print("yfinance 미설치. 설치 중...")
        os.system(f"{sys.executable} -m pip install yfinance -q")

    trader = PaperTrader()
    trader.run()
