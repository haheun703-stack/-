"""src/use_cases/gate_wiring.py — 라이브 매수 게이트 통행증 발급의 *유일* 경로 (RISK_ENGINE C-ii).

★production에서 `risk.pre_trade_gate.evaluate_pre_trade`를 직접 호출하는 곳은 이 모듈뿐이다
  (`tests/test_gate_wiring.py::test_evaluate_pre_trade_sole_caller`가 grep으로 강제).
  live_trading 등 호출자는 `build_gate_result()`만 부른다 → 우회 발급 변형 차단.

발급은 양 모드 공통(mock/paper에서도 토큰을 만들어 전달), 강제만 REAL(KisOrderAdapter):
  → 페이퍼 20일(체크리스트 E) 동안 이 경로가 매 매수마다 돌고, REJECT가 0으로 수렴하는 것이
    '배선 준비 완료' 증거가 된다.

보강 3종 (6/12 part2 적대검증 승인):
  R1 잔고 조회 실패(fetch_balance ok=False) → REJECT(reason=balance_unavailable). equity=0을
     'G3 비중 위반'으로 오판하는 misleading 감사사유를 차단(진실을 말하는 감사 로그).
  R2 adv20 OHLCV 없음/**stale**(마지막 봉이 기준일 대비 max_stale_trading_days 이상 뒤처짐) →
     adv20=None로 G6 fail-closed REJECT. "데이터 없으면 안 산다 + 썩어도 안 산다"
     (6/11 finality 교훈='조용한 미확정값'의 유동성 일반화).
  R3 이 함수가 sole issuance(위 grep 테스트) + 양 모드 발급.

설계: 게이트는 사이징을 *대체*하지 않는다 — 호출자(구 PositionSizer)가 shares를 정하고,
  이 헬퍼는 그 결과를 G3~G6로 검증 + 토큰 발급(검증+발급 층). verify는 호출하지 않는다
  (nonce 소비는 주문 어댑터 _enforce_gate_token이 단 1회 — 이중 소비 방지).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from risk.config import RISK_CONFIG, RiskConfig
from risk.pre_trade_gate import (
    GateRequest,
    GateResult,
    Holding,
    evaluate_pre_trade,
)
from risk.sizing import adv_krw

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_LOG_DIR = PROJECT_ROOT / "data" / "risk" / "gate_logs"
_DEFAULT_MAX_STALE_TRADING_DAYS = 3


# ── 기본 데이터 소스(주입 가능 — 테스트는 fake 주입) ────────────────────────
def _default_ohlcv_loader(ticker: str):
    """기본 OHLCV 로더 — 로컬 parquet 우선(오프라인 가용). 실패 시 None.

    days=400: adv20(window=20)에는 60이면 충분하나 VaR(G1/G2, Phase 2c)는 FHS lookback +
    EWMA 워밍업에 더 긴 이력이 필요하다(_MIN_OBS=60). 로컬 parquet이라 긴 이력 로드 비용은
    무시 가능. pykrx 폴백은 KRX 로그인 만료로 막혀 있어 로컬 parquet 있는 종목만 VaR 가용.
    """
    from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv
    try:
        return load_daily_ohlcv(ticker, days=400)
    except Exception as exc:  # noqa: BLE001 — 로드 실패는 '유동성 모름'으로 흘려보낸다(R2)
        logger.warning("[gate_wiring] OHLCV 로드 실패 %s: %s", ticker, exc)
        return None


def _returns_from_ohlcv(df):
    """OHLCV df → 일별 단순수익률 Series(close.pct_change). 데이터 부족/컬럼 부재 → None."""
    if df is None or len(df) < 2:
        return None
    if "close" not in getattr(df, "columns", []):
        return None
    r = df["close"].pct_change().dropna()
    return r if len(r) > 0 else None


def _build_var_returns(tickers, ohlcv_loader) -> dict:
    """보유+신규 종목의 일별 수익률 {ticker: Series} (VaR G1/G2 입력, Phase 2c).

    ★VaR 최소표본(_MIN_OBS) 미만 종목은 제외 — 짧은 이력으로 VaR를 신뢰할 수 없다. 신규 종목이
    제외되면 호출부가 returns=None으로 넘겨 G1/G2 not_active(G3~G6 폴백, 과차단 방지). 이는
    Phase 1a 철학(데이터 없으면 기존 정적 게이트로) + graceful degradation과 일관 — VaR는 추가
    보호 층이며, 데이터 부족으로 모든 매수를 막는 가용성 붕괴가 더 큰 리스크(진짜 백스톱=킬스위치).
    한 종목 로드 실패가 전체를 막지 않는다(독립 try).
    """
    from risk.var_engine import _MIN_OBS
    out: dict = {}
    for t in tickers:
        if not t:
            continue
        try:
            r = _returns_from_ohlcv(ohlcv_loader(t))
        except Exception:  # noqa: BLE001
            r = None
        if r is not None and len(r) >= _MIN_OBS:
            out[t] = r
    return out


def _default_sector_resolver(ticker: str) -> str | None:
    """기본 sector 리졸버 — stock_to_sector.json. 미매핑/깨짐 → None(게이트가 UNKNOWN 버킷)."""
    from src.use_cases.sector_momentum_label import load_stock_to_sector
    m = load_stock_to_sector()
    v = m.get(ticker)
    if not v:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            s = first.get("sector") or first.get("name")
            return str(s) if s else None
    if isinstance(v, dict):
        s = v.get("sector") or v.get("name")
        return str(s) if s else None
    return None


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────
def _reject(ticker: str, proposed: float, gate: str, reason: str, now_iso: str) -> GateResult:
    """게이트 평가 이전 단계에서의 fail-closed REJECT(토큰 없음). 호출자가 매수 중단."""
    return GateResult(
        verdict="REJECT",
        final_size_krw=0.0,
        original_size_krw=float(proposed),
        violations=[{"gate": gate, "reason": reason}],
        checks={},
        issued_at=now_iso,
        token=None,
        signed=False,
        ticker=ticker,
    )


def _last_bar_date(df) -> date | None:
    """df의 마지막 봉 날짜. 인덱스(DatetimeIndex) 우선, 없으면 'date' 컬럼. 불가 시 None."""
    if df is None or len(df) == 0:
        return None
    import pandas as pd
    try:
        return pd.Timestamp(df.index[-1]).date()
    except Exception:  # noqa: BLE001
        pass
    if "date" in getattr(df, "columns", []):
        try:
            return pd.Timestamp(df["date"].iloc[-1]).date()
        except Exception:  # noqa: BLE001
            return None
    return None


def _trading_days_behind(last_bar: date, as_of: date) -> int:
    """as_of 기준 최신 거래일 대비 last_bar가 몇 거래일 뒤처졌나(0=최신). 상한 캡 10."""
    from src.trading_calendar import last_kr_trading_day, prev_kr_trading_day

    ref = last_kr_trading_day(as_of)
    if last_bar >= ref:
        return 0
    behind = 0
    cur = ref
    while cur > last_bar and behind < 10:
        behind += 1
        cur = prev_kr_trading_day(cur)
    return behind


def _adv20_or_none(
    ticker: str,
    ohlcv_loader: Callable,
    as_of: date,
    max_stale_trading_days: int,
) -> float | None:
    """adv20 거래대금(원). 데이터 없음/계산불능/stale → None(상류 G6 fail-closed REJECT)."""
    df = ohlcv_loader(ticker)
    if df is None or len(df) == 0:
        logger.warning("[gate_wiring] %s OHLCV 없음 → adv20=None(G6 REJECT)", ticker)
        return None
    last_bar = _last_bar_date(df)
    if last_bar is None:
        logger.warning("[gate_wiring] %s 마지막 봉 날짜 불명 → adv20=None(G6 REJECT)", ticker)
        return None
    behind = _trading_days_behind(last_bar, as_of)
    if behind >= max_stale_trading_days:
        logger.warning(
            "[gate_wiring] %s OHLCV stale(%d거래일 뒤처짐, last=%s) → adv20=None(G6 REJECT)",
            ticker, behind, last_bar,
        )
        return None
    val = adv_krw(df, window=20)
    if not val or val <= 0:
        logger.warning("[gate_wiring] %s adv20 계산불능(=0) → None(G6 REJECT)", ticker)
        return None
    return float(val)


# ── 공개 API ─────────────────────────────────────────────────────────────────
def build_gate_result(
    ticker: str,
    proposed_size_krw: float,
    *,
    balance_port,
    ohlcv_loader: Callable | None = None,
    sector_resolver: Callable[[str], str | None] | None = None,
    hmac_key: str | None = None,
    now_kst: datetime | None = None,
    as_of_date: date | None = None,
    log_dir: Path | None = None,
    cfg: RiskConfig = RISK_CONFIG,
    max_stale_trading_days: int = _DEFAULT_MAX_STALE_TRADING_DAYS,
    equity_peak_store=None,
) -> GateResult:
    """라이브 매수 직전 게이트 통행증 발급(유일 경로). 항상 GateResult 반환(REJECT는 사유 포함).

    호출자 처리: verdict=='REJECT'→매수 중단 / 'RESIZE'→final_size_krw로 shares 축소 후 진행 /
    'PASS'→그대로. PASS/RESIZE면 token/signed가 채워져 주문 어댑터가 검증한다.
    """
    ohlcv_loader = ohlcv_loader or _default_ohlcv_loader
    sector_resolver = sector_resolver or _default_sector_resolver
    log_dir = log_dir if log_dir is not None else DEFAULT_GATE_LOG_DIR
    now = now_kst or datetime.now()
    now_iso = now.isoformat()
    as_of = as_of_date or date.today()

    # R1 — 잔고 조회 실패 → fail-closed REJECT(equity=0 오판 차단)
    try:
        balance = balance_port.fetch_balance()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[gate_wiring] %s fetch_balance 예외 → REJECT: %s", ticker, exc)
        return _reject(ticker, proposed_size_krw, "BALANCE", "balance_fetch_exception", now_iso)
    if not isinstance(balance, dict) or not balance.get("ok", False):
        logger.warning("[gate_wiring] %s 잔고 조회 실패(ok=False) → REJECT(balance_unavailable)", ticker)
        return _reject(ticker, proposed_size_krw, "BALANCE", "balance_unavailable", now_iso)

    cash = float(balance.get("available_cash", 0) or 0)
    raw_holdings = balance.get("holdings") or []
    holdings_value = sum(float(h.get("eval_amount", 0) or 0) for h in raw_holdings)
    equity_krw = cash + holdings_value
    if equity_krw <= 0:
        logger.warning("[gate_wiring] %s equity<=0(cash=%s,hold=%s) → REJECT", ticker, cash, holdings_value)
        return _reject(ticker, proposed_size_krw, "BALANCE", "equity_non_positive", now_iso)

    holdings = [
        Holding(
            ticker=str(h.get("ticker", "")),
            value_krw=float(h.get("eval_amount", 0) or 0),
            sector=sector_resolver(str(h.get("ticker", ""))),
            corr_with_new=None,  # 기본 None — 아래 G5 배선(Phase 4)이 returns 가용 시 ρ_stress 주입
        )
        for h in raw_holdings
        if str(h.get("ticker", ""))
    ]

    # R2 — adv20 없음/stale → None → G6 fail-closed REJECT(게이트가 audit 로그까지 기록)
    adv20 = _adv20_or_none(ticker, ohlcv_loader, as_of, max_stale_trading_days)

    # VaR용 수익률(보유 + 신규) — G1/G2 활성화 (Phase 2c).
    var_tickers = {ticker} | {h.ticker for h in holdings if h.ticker}
    returns_by_ticker = _build_var_returns(var_tickers, ohlcv_loader)
    # ★신규 종목 VaR 데이터(≥_MIN_OBS)가 없으면 G1/G2 not_active(Phase 1a 폴백, 과차단 방지).
    #   신규 리스크를 못 본 채 VaR를 통과시키는 fail-open을 막기 위해 '전부 끔'(보유분만으론 무의미).
    if ticker not in returns_by_ticker:
        returns_by_ticker = None

    # G5 상관 클러스터 (Phase 4) — returns 가용 시 신규 vs 보유 ρ_stress 주입(없으면 corr_with_new
    #   =None 유지 → G5 군집 제외 + unknown_corr_count 노출, 기존 계약). ρ_stress=평시 상관을 1
    #   방향 슈링크(위기 동조화 '둘 중 나쁜 값') → 위기에 함께 무너질 종목을 평시 저상관으로
    #   따로 세는 분산 착시 차단. 계산 불가(공통표본<60) 종목은 None 유지(과차단 방지).
    if returns_by_ticker:
        from risk.correlation import stress_correlation_with
        other_tickers = [h.ticker for h in holdings if h.ticker]
        corr_map = stress_correlation_with(ticker, other_tickers, returns_by_ticker, cfg=cfg)
        if corr_map:
            holdings = [
                Holding(ticker=h.ticker, value_krw=h.value_krw, sector=h.sector,
                        corr_with_new=corr_map.get(h.ticker))
                for h in holdings
            ]

    # G8 드로다운 사다리 (Phase 3c) — equity_peak_store 주입 시만 활성(없으면 ladder=None=not_active).
    #   계좌 고점 추적 → DD → ladder_state(히스테리시스). 영속/계산 실패는 G8 비활성(graceful).
    ladder = None
    if equity_peak_store is not None:
        try:
            ladder = equity_peak_store.update_and_ladder(equity_krw, cfg)
        except Exception as exc:  # noqa: BLE001 — 영속/계산 실패는 G8 끔(가용성 우선, 백스톱=킬스위치)
            logger.warning("[gate_wiring] %s equity_peak ladder 실패 → G8 not_active: %s", ticker, exc)
            ladder = None

    request = GateRequest(
        ticker=ticker,
        sector=sector_resolver(ticker),
        proposed_size_krw=float(proposed_size_krw),
        equity_krw=equity_krw,
        adv20_krw=adv20,
    )
    return evaluate_pre_trade(
        request, holdings, cfg=cfg, log_dir=log_dir, hmac_key=hmac_key, now_kst=now_kst,
        returns_by_ticker=returns_by_ticker or None, ladder=ladder,
    )


def _derive_enforce(balance_port) -> bool:
    """강제 여부 도출 — ★어댑터의 강제 변수와 동일 신호: `_is_mock is False`(MODEL=REAL)만 True.

    KisOrderAdapter REAL(_is_mock=False) → True(차단). 그 외 전부 False(비차단):
      KisOrderAdapter mock(_is_mock=True)·PaperOrderAdapter(_is_mock 없음→기본 True)·테스트
      MagicMock(_is_mock=truthy) → 모두 `... is False`가 False. → paper/mock/테스트는 자동 비차단.
    REAL만 강제하므로 어댑터 하드 차단(_enforce_gate_token raise)이 일어나는 경우와 정확히 일치한다.
    """
    return getattr(balance_port, "_is_mock", True) is False


def gate_check(
    balance_port,
    ticker: str,
    unit_price: float,
    qty: int,
    *,
    enforce: bool | None = None,
    **build_kwargs,
) -> tuple[bool, GateResult | None, int]:
    """호출처용 얇은 래퍼 — REJECT/RESIZE/모드별 거동을 1곳에 모은다("로직 1곳=버그 1곳").

    각 BUY 호출처(smart_entry/adaptive_*/limit_up_scanner/live_trading)는 이것만 부르면 된다:
        proceed, gate, qty = gate_check(adapter, ticker, unit_price, qty)
        if not proceed:         # REAL에서 REJECT/입력불가 → 매수 스킵
            ...continue/skip
        order = adapter.buy_limit(ticker, price, qty, gate_result=gate, **kw)

    ★강제만 REAL, 발급은 양 모드(보강 R3): enforce는 기본적으로 `_is_mock is False`로 도출되어
      REAL만 차단한다. mock/paper에선 절대 차단하지 않고(어댑터가 토큰 무시·또는 GATE-DRYRUN 경고),
      발급된 토큰이 있으면 그대로 전달(audit/E 증거). enforce를 명시 전달해 강제 override 가능.

    Args:
        balance_port: fetch_balance()를 가진 어댑터(KisOrderAdapter는 BalancePort 겸함).
        unit_price: 사이징 기준 단가 — 지정가는 주문가, 시장가는 현재가 추정. ≤0이면 입력불가.
        qty: 주문 수량.
        enforce: None이면 balance_port._is_mock에서 도출(REAL만 True). 명시 시 그 값 사용.
        build_kwargs: build_gate_result로 전달(ohlcv_loader/sector_resolver/hmac_key/now_kst/log_dir 등).

    Returns:
        (proceed, gate_result, final_qty).
        REAL(enforce=True): REJECT/입력불가 → (False, None, 0) / RESIZE → (True, gate, 축소qty) /
          PASS → (True, gate, qty).
        mock·paper(enforce=False): 항상 proceed=True. 발급 토큰(PASS/RESIZE)이면 (True, gate, qty),
          REJECT/입력불가면 (True, None, qty) — 차단 없이 원수량 그대로(어댑터가 무시 또는 경고).
    """
    if enforce is None:
        enforce = _derive_enforce(balance_port)

    if not unit_price or unit_price <= 0 or qty <= 0:
        if enforce:
            logger.warning("[gate_check] %s 단가/수량 부적격(price=%s, qty=%s) → REAL fail-closed 스킵",
                           ticker, unit_price, qty)
            return False, None, 0
        return True, None, qty  # mock/paper: 차단 안 함(어댑터가 무시/경고)

    try:
        gate = build_gate_result(ticker, float(qty) * float(unit_price),
                                 balance_port=balance_port, **build_kwargs)
    except Exception as exc:  # noqa: BLE001 — 발급 중 예외는 REAL에선 차단, 그 외엔 통과
        logger.warning("[gate_check] %s 발급 예외: %s", ticker, exc)
        return (False, None, 0) if enforce else (True, None, qty)

    if not enforce:
        # mock/paper: 절대 차단 안 함. 발급된 토큰이면 전달(audit), REJECT면 None(어댑터 무시).
        return True, (gate if gate.verdict in ("PASS", "RESIZE") else None), qty

    # ── REAL 강제 ──
    if gate.verdict == "REJECT":
        logger.warning("[gate_check] %s REAL 게이트 거부 — %s", ticker, gate.violations)
        return False, None, 0
    if gate.verdict == "RESIZE":
        new_qty = int(gate.final_size_krw // unit_price)
        if new_qty <= 0:
            logger.warning("[gate_check] %s RESIZE 후 수량 0 → 스킵", ticker)
            return False, None, 0
        if new_qty != qty:
            logger.info("[gate_check] %s 게이트 RESIZE: %d→%d주(승인 %.0f원)",
                        ticker, qty, new_qty, gate.final_size_krw)
        return True, gate, new_qty
    return True, gate, qty
