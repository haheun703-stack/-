"""통합 Paper Trading 엔진 — 1주 Rolling 방식.

매일 장마감 후 BAT-D에서 실행:
  python -u -X utf8 scripts/paper_trading_unified.py

동작:
  1. tomorrow_picks.json에서 추천 종목 수집 (AI대형주 + 전략종합)
  2. 등급별 가상 포트폴리오 진입 (max 3개/일, max 8개 보유)
  3. 보유 종목 일별 현재가 업데이트 + 매도 조건 체크
  4. 금요일 주간 리밸런싱: 미추천 청산 → 겹치면 유지 → 신규 진입
  5. 텔레그램 일일/주간 리포트 ([PAPER] 태그)
  6. FLOWX Supabase paper_trades 업로드

Rolling 규칙:
  - 최대 보유일 5영업일 (MAX_HOLDING_DAYS=5)
  - 매주 금요일(또는 --rebalance) 리밸런싱 강제 실행
  - 리밸런싱: 이번 주 추천에 없는 보유종목 → 전량 청산
  - 이번 주 추천에 있는 기존 보유 → 유지 (보유일 리셋)
  - 새 추천 중 미보유 → 신규 진입

데이터:
  - 입력: data/tomorrow_picks.json, data/processed/*.parquet
  - 출력: data/paper_portfolio.json (포지션 + 성적)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pykrx_quiet import silence_pykrx_logging

silence_pykrx_logging()  # pykrx 로그인/로깅 노이즈 억제 (진입부 1회)

# .env 로드 (systemd/cron 환경에서 AUTO_TRADING_* 등 환경변수 읽기 보장)
load_dotenv(PROJECT_ROOT / ".env")

from src.stock_name_resolver import ticker_to_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════

DATA_DIR = PROJECT_ROOT / "data"

# ── 확신모델 A/B 페이퍼 (conviction-reversal-redesign 6/18 · §6 사장님 승인 6/22) ──
# A = 현행 baseline(무손상) / B = 과매집(crowding) 감점 적용.
# 환경변수로 포트폴리오 파일을 격리 → A안은 손도 안 댐, B안만 별도 누적.
# 실주문 경로(KisOrderAdapter·리스크게이트 6호출처) 무접촉 — 페이퍼 SCAN 한정.
CONVICTION_MODE = os.environ.get("PAPER_CONVICTION_MODE", "A").upper()
_PF_FILE = "paper_portfolio_b.json" if CONVICTION_MODE == "B" else "paper_portfolio.json"
PORTFOLIO_PATH = DATA_DIR / _PF_FILE
# 매도경로 변형(6/23 검증 기반) — 독립 플래그. 기본 OFF → A안 동작 100% 불변.
#   근거: parquet 200만+ 관측 백테스트. 손실구간(-4~-7%) 쌍끌이는 forward 최저(+0.106%
#         vs 중립 +0.480%) + 추가하락 위험 최고(57%). "매집 지속에도 주가 밀림=약세".
#   변경 ①매수신호 보유일 연장 제거(hold_adj=min(score,0)) ②손실구간 쌍끌이 조기탈출.
#   run_bat.sh B안 블록에서 PAPER_CONVICTION_MODE=B와 함께 ON → B안 = 진입역전+매도변형.
SELL_REVISION = os.environ.get("PAPER_SELL_REVISION", "0") == "1"
PICKS_PATH = DATA_DIR / "tomorrow_picks.json"
JARVIS_PATH = DATA_DIR / "jarvis_direction.json"
PROCESSED_DIR = DATA_DIR / "processed"

# 포지션 규칙
INITIAL_CAPITAL = 30_000_000   # 3,000만원 가상 자본
MAX_POSITIONS = 8              # 최대 동시 보유
MAX_NEW_PER_DAY = 3            # 하루 최대 신규 진입
SLIPPAGE_PCT = 0.001           # 슬리피지 0.1%
COMMISSION_PCT = 0.00015       # 수수료 0.015%
TAX_PCT = 0.0018               # 매도세 0.18%

# 등급별 사이징 (자본 대비 %)
SIZING = {
    "AA": 0.15,    # 강력 포착/confidence>=0.85: 자본의 15%
    "A": 0.12,     # 포착/confidence>=0.75: 12%
    "B": 0.10,     # 관심/기타: 10%
}

# 매도 규칙
STOP_LOSS_PCT = -0.07          # -7% 손절
TAKE_PROFIT_T1_PCT = 0.10      # +10% 1차 익절 (50% 매도)
TAKE_PROFIT_T2_PCT = 0.20      # +20% 2차 익절 (전량 매도)
TRAILING_ACTIVATE_PCT = 0.08   # +8% 이후 트레일링 활성화
TRAILING_STOP_PCT = -0.04      # 고점 대비 -4% 하락 시 매도
MAX_HOLDING_DAYS = int(os.environ.get("PAPER_MAX_HOLD", 5))  # 기본 5일, D+1=1

# ETF 방향 트레이딩 (JARVIS 연동)
ETF_CAPITAL = 10_000_000       # 1,000만원 별도 자본 (개별 종목과 분리)
ETF_MAP = {
    "STRONG_LONG":  {"code": "122630", "name": "KODEX 레버리지"},
    "LONG":         {"code": "069500", "name": "KODEX 200"},
    "SHORT":        {"code": "114800", "name": "KODEX 인버스"},
    "STRONG_SHORT": {"code": "252670", "name": "KODEX 200선물인버스2X"},
}
ETF_STOP_LOSS_PCT = -0.05      # ETF 손절 -5% (레버리지 특성 고려)
# 5/16 추가 (Phase 8/9 백테스트 + 5/12~15 학습):
ETF_TAKE_PROFIT_T1 = 0.05      # ETF +5% 도달 시 trailing 활성화
ETF_TAKE_PROFIT_T2 = 0.10      # ETF +10% 분할 익절 (50%, 단순화: 전량)
ETF_TRAILING_STOP = -0.02      # +5% 도달 후 고점 -2% 하락 시 매도
# 인버스/레버리지 보수적 (Phase 9 -21% 손실 학습)
ETF_INVERSE_STOP_LOSS = -0.03  # 인버스 손절 -3% (Phase 9 -21% 방지)
ETF_INVERSE_MAX_HOLD = 2       # 인버스 최대 보유 2일 (D+1~D+2)
# Phase 11 정교화 — 인버스 진입 strict 조건
INVERSE_STRICT_KOSPI_1D = -2.5      # KOSPI 1일 -2.5%↓
INVERSE_STRICT_FOREIGN_5D_EOK = -50000  # 외인 5일 누적 -5조↓
INVERSE_STRICT_FOREIGN_1D_EOK = -3000   # 외인 1일 -3000억↓
INVESTOR_DB_PATH = DATA_DIR / "investor_flow" / "investor_daily.db"
KOSPI_CSV_PATH = DATA_DIR / "kospi_index.csv"

# ── 알파 필터 설정 (2026-04-07 도입) ──
# Shield 레벨별 최대 보유 수
SHIELD_MAX_POSITIONS = {
    "RED": 3,       # CRISIS/위기 → 3종목 집중
    "YELLOW": 5,    # 경계 → 5종목
    "GREEN": 8,     # 정상 → 기존 8종목
}
# STRONG_ALPHA 시그널 (PF>=1.8, 백테스트 검증 → AA 승격 + 50점 부스트)
STRONG_ALPHA_SIGNALS = {
    "PULLBACK15_VOL3x",          # PF=2.58 (급락15%+거래량3배+수급)
    "PULLBACK15_DUAL",           # PF=1.94 (급락15%+쌍끌이)
    "BREAKOUT60_VOL3x_DUAL",    # PF=1.81 (60일돌파+거래량3배+쌍끌이)
    "MEGA_VOL_8x",               # 메가거래량(8배+)+수급 (주간급등주 패턴)
}
# MODERATE_ALPHA 시그널 (PF 1.4~1.8, 보조 → A 승격 + 30점 부스트)
MODERATE_ALPHA_SIGNALS = {
    "PULLBACK10_SUPPLY",         # PF=1.48 (급락10%+수급)
    "PULLBACK7_SUPPLY",          # PF=1.35 (급락7%+수급)
}
# 알파 부스트 점수 (후보 정렬 시 가산)
ALPHA_BOOST = 50    # STRONG_ALPHA → +50점 부스트 (100점 = AA 최상위)
MODERATE_BOOST = 30  # MODERATE_ALPHA → +30점 부스트 (80점 = A급)
PULLBACK_BOOST = 20 # pullback_scan 등재 → +20점 부스트
SHIELD_PATH = DATA_DIR / "shield_report.json"
PULLBACK_PATH = DATA_DIR / "pullback_scan.json"

# ── 과매집(crowding) 감점 — 확신모델 역전 재설계 B안 (conviction-reversal-redesign §3·§4.1) ──
# 진범: 외인/기관 "연속 매수(과매집)"가 길수록 천장 진입 → 손실 (paper 63건 해부 §3).
#   STRONG_ALPHA의 쌍끌이 2종(PULLBACK15_DUAL·BREAKOUT60_VOL3x_DUAL)이 연속길이를 무시하고
#   AA를 강제 → 과매집을 최고확신·최대비중으로 천장에서 진입하는 구조였음.
# 입력: §3 해부와 동일 출처 = parquet inst/foreign_consecutive_buy (pick.dual_days는 picks에 부재).
# 계수 고정 — 백테스트 튜닝 금지(§5 과최적화 방지). 페이퍼 A/B 실측으로만 검증.
STRONG_CROWDING_DAYS = 6   # 외인/기관 연속매수 ≥6일 = 과매집 천장 → AA 승격 차단(→A)
_CROWD_COLS = ["inst_consecutive_buy", "foreign_consecutive_buy"]


def crowding_streak(ticker: str) -> int:
    """parquet 최신 행의 외인/기관 연속매수일 중 최대값(과매집 지표). graceful=0."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        return 0
    try:
        df = pd.read_parquet(pq, columns=_CROWD_COLS)
        if df.empty:
            return 0
        row = df.iloc[-1]
        return max(int(row.get("inst_consecutive_buy", 0) or 0),
                   int(row.get("foreign_consecutive_buy", 0) or 0))
    except Exception:  # noqa: BLE001 — 데이터 결손은 감점 미적용(보수적, A안과 동일 거동)
        return 0


def demote_if_crowded(grade: str, ticker: str) -> str:
    """B안 전용: AA가 과매집(연속 ≥STRONG_CROWDING_DAYS) 천장이면 A로 강등.
    A안(CONVICTION_MODE != 'B')에선 무조건 no-op → 현행 거동 100% 보존."""
    if CONVICTION_MODE != "B" or grade != "AA":
        return grade
    days = crowding_streak(ticker)
    if days >= STRONG_CROWDING_DAYS:
        logger.info("[CONVICTION-B] %s 과매집 %d연속 → AA 차단(→A)", ticker, days)
        return "A"
    return grade


def get_shield_max_positions() -> int:
    """Shield 레벨에 따른 동적 최대 보유 수 결정."""
    try:
        with open(SHIELD_PATH, "r", encoding="utf-8") as f:
            shield = json.load(f)
        level = shield.get("overall_level", "GREEN").upper()
        max_pos = SHIELD_MAX_POSITIONS.get(level, MAX_POSITIONS)
        logger.info("[ALPHA] Shield=%s → MAX_POSITIONS=%d", level, max_pos)
        return max_pos
    except Exception as e:
        logger.warning("[ALPHA] Shield 읽기 실패: %s → 기본값 %d", e, MAX_POSITIONS)
        return MAX_POSITIONS


def _load_pullback_tickers() -> set:
    """pullback_scan.json에서 풀백 종목 티커 세트 반환."""
    try:
        with open(PULLBACK_PATH, "r", encoding="utf-8") as f:
            ps = json.load(f)
        tickers = set()
        for cand in ps.get("candidates", []):
            if isinstance(cand, dict) and cand.get("ticker"):
                tickers.add(cand["ticker"])
        return tickers
    except Exception:
        return set()


# ═══════════════════════════════════════════════
# 포트폴리오 관리
# ═══════════════════════════════════════════════

def _default_portfolio() -> dict:
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "initial_capital": INITIAL_CAPITAL,
        "capital": INITIAL_CAPITAL,
        "positions": {},
        "closed_trades": [],
        "daily_equity": [],
        "stats": {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "max_equity": INITIAL_CAPITAL,
            "mdd": 0.0,
        },
        "updated": "",
    }


def load_portfolio() -> dict:
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 마이그레이션: 이전 포맷 호환
        if data.get("initial_capital", 0) != INITIAL_CAPITAL and not data.get("positions"):
            logger.info("포트폴리오 자본금 갱신: %s → %s", data.get("initial_capital"), INITIAL_CAPITAL)
            data["initial_capital"] = INITIAL_CAPITAL
            data["capital"] = INITIAL_CAPITAL
        return data
    pf = _default_portfolio()
    save_portfolio(pf)
    return pf


def save_portfolio(pf: dict) -> None:
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 종가 조회
# ═══════════════════════════════════════════════

def get_latest_price(ticker: str) -> tuple[float, str]:
    """processed parquet에서 최신 종가 + 날짜 반환."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        # raw fallback
        pq = DATA_DIR / "raw" / f"{ticker}.parquet"
    if not pq.exists():
        return 0.0, ""
    try:
        df = pd.read_parquet(pq)
        if len(df) == 0:
            return 0.0, ""
        last = df.iloc[-1]
        return float(last["close"]), df.index[-1].strftime("%Y-%m-%d")
    except Exception:
        return 0.0, ""


def get_etf_price(code: str) -> float:
    """ETF 현재가 조회 (parquet → pykrx 순 fallback)."""
    price, _ = get_latest_price(code)
    if price > 0:
        return price
    try:
        from pykrx import stock as pykrx_stock
        to_date = datetime.now().strftime("%Y%m%d")
        from_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        df = pykrx_stock.get_market_ohlcv_by_date(from_date, to_date, code)
        if len(df) > 0:
            return float(df["종가"].iloc[-1])
    except Exception as e:
        logger.warning("[ETF] %s pykrx 가격 조회 실패: %s", code, e)
    return 0.0


# ═══════════════════════════════════════════════
# 추천 종목 수집
# ═══════════════════════════════════════════════

def collect_candidates() -> list[dict]:
    """tomorrow_picks.json에서 paper trading 후보 수집.

    Returns:
        [{"ticker", "name", "grade", "score", "price", "strategy", "reason"}]
        score 내림차순 정렬.
    """
    if not PICKS_PATH.exists():
        logger.warning("tomorrow_picks.json 없음")
        return []

    with open(PICKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    seen = set()

    # 알파 시그널 조회 준비: picks 데이터에서 ticker→alpha_signals 매핑
    _picks_alpha = {}
    for p in data.get("picks", []):
        t = p.get("ticker", "")
        if t:
            _picks_alpha[t] = p.get("alpha_signals", [])
    _pullback_tickers = _load_pullback_tickers()

    # 1) AI 대형주 (confidence >= 0.75, B등급 제거)
    for item in data.get("ai_largecap", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        conf = float(item.get("confidence", 0))
        if conf < 0.75:     # B등급(0.70~0.75) 차단
            continue

        price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        if conf >= 0.85:
            grade = "AA"
        else:
            grade = "A"

        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "name": item.get("name", ticker),
            "grade": grade,
            "score": round(conf * 100, 1),
            "price": price,
            "strategy": "AI_BRAIN",
            "reason": item.get("reasoning", "")[:80],
        })

    # 2) 전략 종합 picks (score >= 40, 상위 15개 — 시장 주도주 포착 확대)
    picks_sorted = sorted(
        data.get("picks", []),
        key=lambda x: x.get("total_score", 0),
        reverse=True,
    )
    for pick in picks_sorted[:15]:
        ticker = pick.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        score = pick.get("total_score", 0)
        if score < 40:
            continue

        price = pick.get("close", 0)
        if price <= 0:
            price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        # 알파 시그널 기반 등급 우회
        alpha_sigs = set(pick.get("alpha_signals", []))
        has_strong = bool(alpha_sigs & STRONG_ALPHA_SIGNALS)
        has_moderate = bool(alpha_sigs & MODERATE_ALPHA_SIGNALS)
        has_multi = len(alpha_sigs) >= 3  # 3개+ 시그널 = 복합 강세

        grade_kr = pick.get("grade", "")
        if has_strong:
            # 부검(메인A 73건: SCAN/AA -458만원 = 전체 손실의 원흉, 2026-06-27) → SCAN/AA(STRONG) 전면 차단.
            #   ★6/30 forward A/B 확정(메인A -2.7%[SCAN활성] vs 메인B +0.5%[SCAN차단]) → 메인A에도 확대.
            #   대조군 종료(SCAN/AA가 손실 원흉으로 forward 검증됨). CONVICTION_MODE는 이제 과매집감점·매도변형 분기에만.
            logger.info("[SCAN-BLOCK] SCAN/AA(STRONG) 차단(부검 -458만·6/30 메인A 확대): %s",
                        pick.get("name", ticker))
            continue
        elif (has_moderate or has_multi) and score >= 75:
            grade = "A"   # 멀티시그널(3+) + 고점수 → A 승격
            logger.info("[MULTI-SIG] %s grade=%s → A 승격 (시그널 %d개, score=%.1f)",
                        pick.get("name", ticker), grade_kr, len(alpha_sigs), score)
        elif grade_kr in ("강력 포착", "적극매수"):
            # SCAN/AA 전면 차단(부검 -458만 · 6/30 forward A/B 확정 → 메인A 확대)
            logger.info("[SCAN-BLOCK] SCAN/AA(강력포착) 차단(부검 -458만·6/30 메인A 확대): %s",
                        pick.get("name", ticker))
            continue
        elif grade_kr in ("포착", "매수"):
            grade = "A"
        else:
            # B등급(관심/관찰/보류) 차단
            logger.debug("[ALPHA] %s B등급 제외 (grade=%s)", ticker, grade_kr)
            continue

        seen.add(ticker)
        reasons = pick.get("reasons", [])
        reason_str = ", ".join(reasons[:3]) if reasons else ""

        candidates.append({
            "ticker": ticker,
            "name": pick.get("name", ticker),
            "grade": grade,
            "score": round(score, 1),
            "price": float(price),
            "strategy": "SCAN",
            "reason": reason_str[:80],
        })

    # 3) top5_swing (스윙 전용, 있으면 — ticker 문자열 또는 dict)
    for item in data.get("top5_swing", []):
        if isinstance(item, str):
            ticker = item
            item_name = ticker
            item_score = 50
        elif isinstance(item, dict):
            ticker = item.get("ticker", "")
            item_name = item.get("name", ticker)
            item_score = item.get("total_score", 50)
        else:
            continue
        if not ticker or ticker in seen:
            continue
        price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "name": item_name,
            "grade": "A",
            "score": round(item_score, 1),
            "price": price,
            "strategy": "SWING",
            "reason": "스윙 전략 추천",
        })

    # 4) ALPHA OVERRIDE: 검증된 알파시그널 보유 종목 → 등급 무시, 직접 진입
    #    STRONG(PF>=1.8) → AA+50점, MODERATE(PF 1.4~1.8) → A+30점
    for pick in data.get("picks", []):
        ticker = pick.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        alpha_sigs = set(pick.get("alpha_signals", []))
        strong_match = alpha_sigs & STRONG_ALPHA_SIGNALS
        moderate_match = alpha_sigs & MODERATE_ALPHA_SIGNALS
        in_pullback = ticker in _pullback_tickers
        if not strong_match and not moderate_match and not in_pullback:
            continue

        price = pick.get("close", 0)
        if price <= 0:
            price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        # 알파 점수 산출: 기본 50 + 부스트 (STRONG > MODERATE > PULLBACK)
        base_score = 50.0
        alpha_tags = []
        if strong_match:
            base_score += ALPHA_BOOST    # +50
            alpha_tags.extend(strong_match)
        if moderate_match:
            base_score += MODERATE_BOOST  # +30
            alpha_tags.extend(moderate_match)
        if in_pullback:
            base_score += PULLBACK_BOOST  # +20
            alpha_tags.append("PULLBACK_SCAN")

        # 등급: STRONG → AA(B안: 과매집 천장이면 차단), MODERATE/풀백 → A
        alpha_grade = demote_if_crowded("AA", ticker) if strong_match else "A"

        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "name": pick.get("name", ticker),
            "grade": alpha_grade,
            "score": base_score,
            "price": float(price),
            "strategy": "ALPHA",
            "reason": ",".join(alpha_tags)[:80],
            "alpha_boost": base_score - 50,
            "alpha_tags": alpha_tags,
        })
        logger.info("[ALPHA] %s → %s 승격 (score=%.1f) [%s]",
                    pick.get("name", ticker), alpha_grade, base_score,
                    ",".join(alpha_tags))

    # 5) INTRADAY 학습 시그널 (Phase 12c)
    #    어제 장중 학습이 추출한 오늘 진입 후보 (early_ret + 체결강도 + 매수비율 통과)
    today_compact = datetime.now().strftime("%Y%m%d")
    intra_sig_path = DATA_DIR / "intraday" / f"intraday_signals_{today_compact}.json"
    if intra_sig_path.exists():
        try:
            sig = json.loads(intra_sig_path.read_text(encoding="utf-8"))
            for c in sig.get("candidates", [])[:10]:
                ticker = c.get("code", "")
                if not ticker or ticker in seen:
                    continue
                price, _ = get_latest_price(ticker)
                if price <= 0:
                    continue
                seen.add(ticker)
                candidates.append({
                    "ticker": ticker,
                    "name": ticker,
                    "grade": "A",  # 학습 시그널은 A등급 (B 차단 통과)
                    "score": 60.0 + min(c.get("early_ret_pct", 0) * 2, 30),
                    "price": price,
                    "strategy": "INTRADAY_LEARNED",
                    "reason": (
                        f"early {c.get('early_ret_pct', 0):+.2f}% / "
                        f"strength {c.get('strength_avg', 0):.1f} / "
                        f"buy_ratio {c.get('buy_ratio', 0):.2f}"
                    ),
                    "alpha_tags": ["INTRADAY_LEARNED"],
                })
            logger.info(f"[INTRADAY] 학습 시그널 {len(sig.get('candidates', []))}건 후보 합류")
        except Exception as e:
            logger.warning(f"[INTRADAY] 시그널 로드 실패: {e}")

    # 종목명 보정: name이 ticker 코드 그대로인 경우 resolver로 해결
    for cand in candidates:
        if cand["name"] == cand["ticker"] or not cand["name"]:
            cand["name"] = ticker_to_name(cand["ticker"])

    # ── 기존 후보에도 알파 부스트 적용 ──
    for cand in candidates:
        if "alpha_boost" in cand:
            continue  # 이미 ALPHA 소스에서 부스트됨
        ticker = cand["ticker"]
        alpha_sigs = set(_picks_alpha.get(ticker, []))
        boost = 0
        alpha_tags = []

        strong_match = alpha_sigs & STRONG_ALPHA_SIGNALS
        moderate_match = alpha_sigs & MODERATE_ALPHA_SIGNALS
        if strong_match:
            boost += ALPHA_BOOST
            alpha_tags.extend(strong_match)
        if moderate_match:
            boost += MODERATE_BOOST
            alpha_tags.extend(moderate_match)
        if ticker in _pullback_tickers:
            boost += PULLBACK_BOOST
            alpha_tags.append("PULLBACK_SCAN")

        if boost > 0:
            cand["score"] += boost
            cand["alpha_boost"] = boost
            cand["alpha_tags"] = alpha_tags
            logger.info("[ALPHA] %s(%s) +%d boost → score=%.1f [%s]",
                        cand["name"], ticker, boost, cand["score"],
                        ",".join(alpha_tags))

    # 3) 추세추종 (B안 전용 — 강추세주 발굴, 풀백 스캐너의 빈칸 보완)
    #    진입 확신모델은 과매집(천장) 감점이 핵심이나, 그것만으론 '안 떨어지고 오르는'
    #    강추세주를 통째로 놓침(6/22 발견 — 하이닉스·스퀘어). trend_follow_scanner 로직
    #    재사용해 후보 가세. A안(현행)은 무손상 → A/B 페이퍼로 추세 가세 효과 검증.
    if CONVICTION_MODE == "B":
        try:
            import os as _os
            import sys as _sys
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from trend_follow_scanner import scan as _trend_scan

            _, _trend_hits = _trend_scan()
            _added = 0
            for h in _trend_hits:
                if h.get("grade") == "과열":          # 과열 이상치 제외(추격 위험)
                    continue
                tk = h["ticker"]
                if tk in seen:
                    continue
                seen.add(tk)
                candidates.append({
                    "ticker": tk,
                    "name": h["name"],
                    "grade": "A",                      # 검증 전 보수적 A급(쌍끌이 AA보다 아래)
                    "score": round(80 + min(h["mom5_pct"] / 5.0, 15.0), 1),
                    "price": h["close"],
                    "strategy": "TREND_FOLLOW",
                    "reason": f"추세추종 5일{h['mom5_pct']:+.0f}%·신고가·정배열"
                              f"(손절=5일선 {h['stop_price']:,})",
                    "stop_price": h["stop_price"],     # 5일선 = 손절 기준(매도경로 후속)
                })
                _added += 1
            if _added:
                logger.info("[TREND] 추세추종 후보 %d종 추가(B안)", _added)
        except Exception as exc:  # noqa: BLE001 — 페이퍼 절대 무손상
            logger.warning("[TREND] 추세추종 통합 스킵: %s", exc)

    # score(부스트 포함) 내림차순 정렬
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════
# 수급 시그널
# ═══════════════════════════════════════════════

_SD_COLS = ["inst_net_5d", "foreign_net_5d", "inst_consecutive_buy",
            "foreign_consecutive_buy", "inst_net_streak", "foreign_net_streak"]


def get_supply_demand_signal(ticker: str) -> dict:
    """parquet에서 수급 시그널 추출 → 보유/매도 판단 보정값 반환.

    Returns:
        {
            "score": int,       # -2(강한 이탈) ~ +2(쌍끌이 매집)
            "detail": str,      # 로그용 설명
            "stop_adj": float,  # 손절선 보정 (양수=완화, 음수=강화)
            "hold_adj": int,    # MAX_HOLD 보정일수
        }
    """
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        return {"score": 0, "detail": "no_data", "stop_adj": 0.0, "hold_adj": 0}
    try:
        df = pd.read_parquet(pq, columns=_SD_COLS)
        if df.empty:
            return {"score": 0, "detail": "empty", "stop_adj": 0.0, "hold_adj": 0}
        row = df.iloc[-1]
    except Exception:
        return {"score": 0, "detail": "read_err", "stop_adj": 0.0, "hold_adj": 0}

    inst_5d = float(row.get("inst_net_5d", 0) or 0)
    foreign_5d = float(row.get("foreign_net_5d", 0) or 0)
    inst_streak = int(row.get("inst_consecutive_buy", 0) or 0)
    foreign_streak = int(row.get("foreign_consecutive_buy", 0) or 0)

    score = 0
    reasons = []

    # 쌍끌이 (기관+외인 5일 양수) → 강한 매집
    if inst_5d > 0 and foreign_5d > 0:
        score += 2
        reasons.append("쌍끌이매집")
    # 한쪽만 매수
    elif inst_5d > 0 or foreign_5d > 0:
        score += 1
        buyer = "기관" if inst_5d > 0 else "외인"
        reasons.append(f"{buyer}매수")
    # 쌍매도
    elif inst_5d < 0 and foreign_5d < 0:
        score -= 2
        reasons.append("쌍매도")
    # 한쪽만 매도
    elif inst_5d < 0 or foreign_5d < 0:
        score -= 1
        seller = "기관" if inst_5d < 0 else "외인"
        reasons.append(f"{seller}매도")

    # 연속 매수 3일+ → 추가 부스트
    if inst_streak >= 3 or foreign_streak >= 3:
        score = min(score + 1, 2)
        who = []
        if inst_streak >= 3:
            who.append(f"기관{inst_streak}연속")
        if foreign_streak >= 3:
            who.append(f"외인{foreign_streak}연속")
        reasons.append("+".join(who))

    # 보정값 계산 (★주석 정정 6/23: 실제 동작은 손절'선' 이동 — 구주석의 완화/강화 표기가
    #   동작과 반대로 기재돼 있었음. adj_stop = STOP_LOSS_PCT + stop_adj.)
    #   score +2(쌍끌이): -7% + 2% = -5% → 손절선 상향 = 더 빨리 손절(=실제로는 강화).
    #   score -2(쌍매도): -7% - 2% = -9% → 손절선 하향 = 늦게 손절(=실제로는 완화).
    #   백테스트(200만+ 관측): 손실구간 쌍끌이는 forward 최저 → 빨리 자르는 현 동작이 데이터 부합.
    stop_adj = score * 0.01   # ±1~2%p (A안 동작 불변)
    hold_adj = score           # ±1~2일
    if SELL_REVISION:
        # 매수신호(score>0)의 보유일 연장은 데이터와 반대(쌍끌이 forward 최저) → 제거.
        #   매도신호(score<0)의 보유일 단축은 그대로 유지.
        hold_adj = min(score, 0)

    detail = " / ".join(reasons) if reasons else "중립"
    return {"score": score, "detail": detail, "stop_adj": stop_adj, "hold_adj": hold_adj}


# ═══════════════════════════════════════════════
# 매도 체크
# ═══════════════════════════════════════════════

def check_exits(pf: dict, today_str: str) -> list[dict]:
    """보유 종목 매도 조건 체크. 수급 시그널로 손절/보유일 동적 조정."""
    exits = []
    codes_to_remove = []

    for ticker, pos in list(pf["positions"].items()):
        price, price_date = get_latest_price(ticker)
        if price <= 0:
            continue

        avg_price = pos["avg_price"]
        if avg_price <= 0:
            logger.warning("[check_exits] %s avg_price=0, 스킵", ticker)
            continue
        peak_price = pos.get("peak_price", avg_price)
        pnl_pct = price / avg_price - 1

        # 최고가 갱신
        if price > peak_price:
            pos["peak_price"] = price
            peak_price = price

        # 보유일수 계산
        entry_date = pos.get("entry_date", today_str)
        try:
            days_held = (pd.Timestamp(today_str) - pd.Timestamp(entry_date)).days
        except Exception:
            days_held = 0

        # MAX_HOLD는 원진입일 기준 — 리밸런스 유지가 entry_date를 리셋해도 영구회피 불가
        # (7/4 발견 정책구멍 픽스: 메인A SK네트웍스가 유지 리셋으로 MAX_HOLD 회피하다 -6.7% 손절)
        try:
            orig_days_held = (pd.Timestamp(today_str)
                              - pd.Timestamp(pos.get("orig_entry_date", entry_date))).days
        except Exception:
            orig_days_held = days_held

        # 수급 시그널 → 손절선/보유일 동적 조정
        sd = get_supply_demand_signal(ticker)
        adj_stop = STOP_LOSS_PCT + sd["stop_adj"]       # 예: -7% + 2% = -5% (완화)
        adj_trailing = TRAILING_STOP_PCT + sd["stop_adj"]  # 트레일링도 같이 조정
        adj_max_hold = MAX_HOLDING_DAYS + sd["hold_adj"]   # 예: 5 + 2 = 7일

        if sd["score"] != 0:
            logger.info("[수급] %s %s → 손절%.1f%% 보유%d일",
                        pos["name"], sd["detail"],
                        adj_stop * 100, adj_max_hold)

        exit_reason = None
        exit_qty = pos["qty"]  # 기본: 전량 매도

        # 1. 손절 (수급 보정 적용)
        if pnl_pct <= adj_stop:
            exit_reason = "STOP_LOSS"

        # 2. 2차 익절 (+20%)
        elif pnl_pct >= TAKE_PROFIT_T2_PCT:
            exit_reason = "TAKE_PROFIT"

        # 3. 1차 익절 (+10%, 50% 매도)
        elif pnl_pct >= TAKE_PROFIT_T1_PCT and not pos.get("t1_sold"):
            exit_qty = max(1, pos["qty"] // 2)
            pos["t1_sold"] = True
            pos["qty"] -= exit_qty
            # 부분 매도 — 포지션 유지
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * exit_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds
            exits.append({
                "ticker": ticker,
                "name": pos["name"],
                "reason": "TAKE_PROFIT_T1",
                "exit_price": round(sell_price),
                "pnl_pct": round(pnl_pct * 100, 2),
                "qty": exit_qty,
                "partial": True,
                "supply_demand": sd["detail"],
            })
            # 트레일링 활성화
            pos["trailing_active"] = True
            continue

        # 4. 트레일링 스탑 (수급 보정 적용)
        elif pos.get("trailing_active"):
            drop_from_peak = price / peak_price - 1
            if drop_from_peak <= adj_trailing:
                exit_reason = "TRAILING_STOP"

        # 5. 트레일링 활성화 (고점 갱신 중)
        elif pnl_pct >= TRAILING_ACTIVATE_PCT:
            pos["trailing_active"] = True

        # 6. 최대 보유일 초과 — 통상은 entry_date(유지 리셋 반영), 단 원진입일 기준
        #    절대상한(4x)으로 리밸런스 유지의 무한 연장 차단 (유지 자체는 정상 동작 보존)
        elif days_held >= adj_max_hold or orig_days_held >= MAX_HOLDING_DAYS * 4:
            exit_reason = "MAX_HOLD"

        # 7. 수급 이탈 경고: 수익 중인데 쌍매도 → 조기 매도
        elif sd["score"] <= -2 and pnl_pct > 0 and days_held >= 2:
            exit_reason = "SUPPLY_EXIT"

        # 8. [매도변형] 손실 구간 쌍끌이 = 매집 지속에도 주가 밀림(약세 확정) → 조기 손절.
        #    근거(6/23 백테스트): 손실 -4~-7% 구간 쌍끌이 forward 최저(+0.106%) + 추가하락
        #    위험 최고(57%). 손절선(-5%) 도달 전 -3%에서 탈출. SELL_REVISION에서만 활성.
        elif SELL_REVISION and sd["score"] >= 2 and pnl_pct <= -0.03 and days_held >= 2:
            exit_reason = "SUPPLY_TRAP_EXIT"

        # 전량 매도 처리
        if exit_reason:
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * exit_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds

            final_pnl = sell_price / avg_price - 1

            pf["closed_trades"].append({
                "ticker": ticker,
                "name": pos["name"],
                "strategy": pos.get("strategy", ""),
                "grade": pos.get("grade", ""),
                "entry_date": pos.get("entry_date", ""),
                "orig_entry_date": pos.get("orig_entry_date", pos.get("entry_date", "")),
                "exit_date": today_str,
                "avg_price": round(avg_price),
                "exit_price": round(sell_price),
                "qty": exit_qty,
                "pnl_pct": round(final_pnl * 100, 2),
                "exit_reason": exit_reason,
                "days_held": days_held,
                "supply_demand": sd["detail"],
            })

            pf["stats"]["total_trades"] += 1
            if final_pnl > 0:
                pf["stats"]["wins"] += 1
            else:
                pf["stats"]["losses"] += 1

            exits.append({
                "ticker": ticker,
                "name": pos["name"],
                "reason": exit_reason,
                "exit_price": round(sell_price),
                "pnl_pct": round(final_pnl * 100, 2),
                "qty": exit_qty,
                "partial": False,
                "supply_demand": sd["detail"],
            })
            codes_to_remove.append(ticker)

    for ticker in codes_to_remove:
        if ticker in pf["positions"]:
            del pf["positions"][ticker]

    return exits


# ═══════════════════════════════════════════════
# 주간 리밸런싱 (1주 Rolling)
# ═══════════════════════════════════════════════

def weekly_rebalance(pf: dict, candidates: list[dict], today_str: str) -> list[dict]:
    """금요일 주간 리밸런싱: 미추천 청산, 겹치는 종목 유지, 신규 진입.

    Returns:
        exits: 리밸런싱으로 청산된 포지션 리스트
    """
    recommended_tickers = {c["ticker"] for c in candidates}
    exits = []
    codes_to_remove = []

    for ticker, pos in list(pf["positions"].items()):
        if ticker in recommended_tickers:
            # 차단 전략 잔존분만 유지 제외 — 진입차단 집합(SCAN AND AA)과 동일하게.
            # (OR로 넓히면 진입 허용되는 SCAN/A·AI_BRAIN/AA가 금요일 청산→같은날 재매수 공회전)
            if pos.get("strategy") == "SCAN" and pos.get("grade") == "AA":
                logger.info("[REBALANCE] %s 유지 거부 — 차단 전략 잔존분(%s/%s)",
                            pos["name"], pos.get("strategy", ""), pos.get("grade", ""))
            else:
                # 이번 주도 추천 → 유지, 보유일 리셋 (MAX_HOLD는 orig_entry_date 원천 계산)
                logger.info("[REBALANCE] %s 유지 (이번 주 추천 포함)", pos["name"])
                # 규칙D-a 관찰 (TRADING_PRINCIPLES 7/9): 유지 포지션의 손익을 기록해
                # "손실 먼저 잘라라" 우선순위와 현행(추천목록 기준)의 성과 비교 데이터 축적.
                # 청산측 손익은 closed_trades에 이미 기록됨 — 유지측만 여기서 보강. 로그만.
                _p, _ = get_latest_price(ticker)
                if _p > 0:
                    logger.info("[규칙D-a 관찰] 유지 %s pnl %+.2f%%",
                                pos["name"], (_p / pos["avg_price"] - 1) * 100)
                pos.setdefault("orig_entry_date", pos.get("entry_date", today_str))
                pos["entry_date"] = today_str
                pos["trailing_active"] = False
                pos["t1_sold"] = False
                pos["peak_price"] = pos["avg_price"]
                continue

        # 미추천 → 전량 청산
        price, _ = get_latest_price(ticker)
        if price <= 0:
            price = pos["avg_price"]

        sell_price = price * (1 - SLIPPAGE_PCT)
        proceeds = sell_price * pos["qty"] * (1 - TAX_PCT)
        pf["capital"] += proceeds

        final_pnl = sell_price / pos["avg_price"] - 1
        days_held = 0
        try:
            days_held = (pd.Timestamp(today_str) - pd.Timestamp(pos["entry_date"])).days
        except Exception:
            pass

        pf["closed_trades"].append({
            "ticker": ticker,
            "name": pos["name"],
            "strategy": pos.get("strategy", ""),
            "grade": pos.get("grade", ""),
            "entry_date": pos.get("entry_date", ""),
            "orig_entry_date": pos.get("orig_entry_date", pos.get("entry_date", "")),
            "exit_date": today_str,
            "avg_price": round(pos["avg_price"]),
            "exit_price": round(sell_price),
            "qty": pos["qty"],
            "pnl_pct": round(final_pnl * 100, 2),
            "exit_reason": "REBALANCE",
            "days_held": days_held,
        })

        pf["stats"]["total_trades"] += 1
        if final_pnl > 0:
            pf["stats"]["wins"] += 1
        else:
            pf["stats"]["losses"] += 1

        exits.append({
            "ticker": ticker,
            "name": pos["name"],
            "reason": "REBALANCE",
            "exit_price": round(sell_price),
            "pnl_pct": round(final_pnl * 100, 2),
            "qty": pos["qty"],
            "partial": False,
        })
        codes_to_remove.append(ticker)

    for ticker in codes_to_remove:
        if ticker in pf["positions"]:
            del pf["positions"][ticker]

    return exits


def is_rebalance_day() -> bool:
    """금요일 여부 판단 (0=월, 4=금)."""
    return datetime.now().weekday() == 4


# ═══════════════════════════════════════════════
# 신규 진입
# ═══════════════════════════════════════════════

def enter_new_positions(pf: dict, candidates: list[dict], today_str: str) -> list[dict]:
    """후보 종목 가상 매수. Shield 연동 + B등급 차단 + 위험감지 게이트(P0-7)."""
    entries = []

    # ── P0-7 위험감지 게이트 (정보봇 SDK) ──
    from src.utils.risk_gate import (
        get_position_multiplier_safe,
        should_block_new_entry_safe,
        get_risk_status_safe,
    )
    risk_mult = get_position_multiplier_safe()  # 0.2~1.0
    risk_block = should_block_new_entry_safe()  # CRISIS만 True
    risk_status = get_risk_status_safe()
    risk_level = risk_status.get("level_kr", "정상")
    risk_score = risk_status.get("total_score", 0)

    if risk_block:
        logger.warning(f"[위험감지] {risk_level} ({risk_score}점) — 신규 진입 차단")
        return []
    if risk_mult < 1.0:
        logger.info(
            f"[위험감지] {risk_level} ({risk_score}점) — 매수금액 ×{risk_mult}"
        )

    # ── 한미충격 게이트 (정보봇 kr_us_shock, 페이퍼 실반영) ──
    #   "한국 더 취약"이고 취약도差(diff=kr-us)가 클수록 신규 매수금액 축소(방어).
    #   risk_gate(위 risk_mult)와 동일 패턴. graceful: 데이터 없으면(loaded=False) ×1.0(중립).
    #   페이퍼라 곧장 실반영 — diff20→×1.0, diff50+→×0.5 선형. 강도/임계는 forward로 조정.
    from src.adapters.jgis_kr_us_shock_adapter import load_kr_us_shock_shadow
    _shock = load_kr_us_shock_shadow()
    shock_mult = 1.0
    if _shock.get("loaded"):
        # 검수 🟡-1/🔵-2: verdict 문자열("한국 더 취약") 의존 제거 — 정보봇 표기변경 시
        #   침묵무력화 방지 위해 수치로 판정 + diff 타입 방어.
        try:
            _kr = float(_shock.get("kr_shock") or 0)
            _us = float(_shock.get("us_shock") or 0)
            _diff = float(_shock.get("diff") or 0) or (_kr - _us)  # diff 누락 시 kr-us 직접
        except (TypeError, ValueError):
            _kr = _us = _diff = 0.0
        # "한국 더 취약" = KR 취약도 > US, 격차(diff)가 클수록 신규 매수금액 축소.
        if _kr > _us and _diff >= 20:
            shock_mult = max(0.5, 1.0 - (_diff - 20) / 60.0)
            logger.info(
                f"[한미충격] KR 더 취약 (kr={_kr} us={_us} diff={_diff}) — 신규 매수금액 ×{shock_mult:.2f}"
            )

    # Shield 기반 동적 최대 보유 수
    shield_max = get_shield_max_positions()
    slots_available = shield_max - len(pf["positions"])
    new_today = 0

    if slots_available <= 0:
        logger.info("[ALPHA] 보유 %d / 한도 %d → 신규 진입 불가",
                     len(pf["positions"]), shield_max)

    # ── 추세추종 슬롯 보장 (B안) — 알파 부스트(score 100+)에 밀려 추세주가 진입 못 하는 것 방지.
    #    최상위 추세주 1종을 후보 맨 앞으로 = 하루 신규 중 1슬롯 우선(나머지는 기존 score 순).
    #    A안 무손상. 강제 진입이 아니라 '공정 기회' — A/B 비교로 효과 검증.
    if CONVICTION_MODE == "B":
        _trend = [c for c in candidates if c.get("strategy") == "TREND_FOLLOW"
                  and c["ticker"] not in pf["positions"]]
        if _trend:
            _top = max(_trend, key=lambda x: x["score"])
            candidates = [_top] + [c for c in candidates if c is not _top]
            logger.info("[TREND] %s 추세추종 1슬롯 우선(B안)", _top["name"])

    for cand in candidates:
        if new_today >= MAX_NEW_PER_DAY:
            break
        if slots_available <= 0:
            break
        if cand["ticker"] in pf["positions"]:
            continue

        # B등급 진입 차단 (안전장치 — collect_candidates에서 이미 필터하지만 2중 방어)
        grade = cand["grade"]
        if grade == "B":
            logger.info("[ALPHA] %s B등급 진입 차단", cand["name"])
            continue

        # 사이징 (P0-7: 위험감지 multiplier 적용)
        size_pct = SIZING.get(grade, SIZING["A"])  # B 없으므로 A를 기본값으로
        buy_amount = min(
            pf["initial_capital"] * size_pct * risk_mult * shock_mult,
            pf["capital"] * 0.90,  # 현금의 90%까지만
        )

        price = cand["price"]
        buy_price = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
        qty = int(buy_amount / buy_price)
        if qty <= 0:
            continue

        cost = buy_price * qty
        if cost > pf["capital"]:
            continue

        # ── E-0: 페이퍼 게이트 드라이런 (관측만·enforce=False·흐름 무변경) ──
        #    가상매수 직전 스냅샷으로 리스크엔진 게이트를 평가해 GATE-DRYRUN 감사로그를 남긴다.
        #    unfreeze 체크리스트 §2-E 페이퍼 20거래일 증거(gate_log)가 여기서 쌓이기 시작한다.
        #    실매매 6호출처는 무접촉. 게이트 미가용/예외는 페이퍼 엔진에 영향 0(graceful).
        try:
            from src.use_cases.paper_gate import run_paper_gate_dryrun
            run_paper_gate_dryrun(pf, cand["ticker"], buy_price, qty)
        except Exception:  # noqa: BLE001 — 페이퍼 엔진 절대 무손상
            pass

        pf["capital"] -= cost
        pf["positions"][cand["ticker"]] = {
            "name": cand["name"],
            "ticker": cand["ticker"],
            "entry_date": today_str,
            "avg_price": round(buy_price),
            "qty": qty,
            "cost": round(cost),
            "peak_price": price,
            "strategy": cand["strategy"],
            "grade": grade,
            "reason": cand["reason"],
            "trailing_active": False,
            "t1_sold": False,
        }

        entries.append({
            "ticker": cand["ticker"],
            "name": cand["name"],
            "grade": grade,
            "price": round(price),
            "qty": qty,
            "cost": round(cost),
            "strategy": cand["strategy"],
        })

        new_today += 1
        slots_available -= 1

    return entries


# ═══════════════════════════════════════════════
# 일일 자산 기록 + 통계
# ═══════════════════════════════════════════════

def update_equity(pf: dict, today_str: str) -> float:
    """일일 자산 평가 + MDD 업데이트. 현재 equity 반환."""
    equity = pf["capital"]

    for ticker, pos in pf["positions"].items():
        price, _ = get_latest_price(ticker)
        if price > 0:
            equity += price * pos["qty"]
        else:
            equity += pos["avg_price"] * pos["qty"]

    # ── 규칙A 관찰 (TRADING_PRINCIPLES 7/9 — 주식:현금 7:3, 매매 무개입·기록만) ──
    #    "이 규칙이 있었다면 오늘 주문이 어떻게 조정됐을지"의 판단 데이터 축적.
    stock_ratio = round((equity - pf["capital"]) / equity * 100, 1) if equity > 0 else 0.0
    if stock_ratio > 75:
        logger.info("[규칙A 관찰] 주식비중 %.1f%% > 75%% — 규칙 있었다면 리밸런싱 후보 리포트", stock_ratio)
    elif stock_ratio > 70:
        logger.info("[규칙A 관찰] 주식비중 %.1f%% > 70%% — 규칙 있었다면 신규매수 축소/거부", stock_ratio)

    # 중복 날짜 방지
    pf["daily_equity"] = [e for e in pf["daily_equity"] if e["date"] != today_str]
    pf["daily_equity"].append({
        "date": today_str,
        "equity": round(equity),
        "capital": round(pf["capital"]),
        "positions": len(pf["positions"]),
        "stock_ratio": stock_ratio,  # 규칙A 관찰 필드 (7/9 추가, 소비자 없음·순수 기록)
    })

    # MDD
    if pf["daily_equity"]:
        max_eq = max(e["equity"] for e in pf["daily_equity"])
        pf["stats"]["max_equity"] = max_eq
        if max_eq > 0:
            current_dd = (equity / max_eq - 1) * 100
            pf["stats"]["mdd"] = round(min(pf["stats"].get("mdd", 0), current_dd), 2)

    return equity


def calc_stats(pf: dict) -> dict:
    """성과 통계 계산."""
    initial = pf["initial_capital"]
    equity_list = pf.get("daily_equity", [])
    equity = equity_list[-1]["equity"] if equity_list else initial

    total = pf["stats"]["total_trades"]
    wins = pf["stats"]["wins"]
    losses = pf["stats"]["losses"]

    # Profit Factor
    gross_profit = sum(
        t["pnl_pct"] for t in pf["closed_trades"] if t["pnl_pct"] > 0
    )
    gross_loss = abs(sum(
        t["pnl_pct"] for t in pf["closed_trades"] if t["pnl_pct"] < 0
    ))
    pf_ratio = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

    return {
        "equity": round(equity),
        "total_return_pct": round((equity / initial - 1) * 100, 2),
        "pf": pf_ratio,
        "mdd": pf["stats"]["mdd"],
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "open_positions": len(pf["positions"]),
        "cash": round(pf["capital"]),
    }


# ═══════════════════════════════════════════════
# ETF 방향 트레이딩 (JARVIS 연동)
# ═══════════════════════════════════════════════

def _default_etf_state() -> dict:
    return {
        "capital": ETF_CAPITAL,
        "position": None,
        "closed_trades": [],
        "daily_equity": [],
    }


def _close_etf_position(etf: dict, price: float, today_str: str, reason: str) -> dict:
    """ETF 포지션 청산 공통 로직. 청산 결과 dict 반환."""
    pos = etf["position"]
    if not pos:
        return {}

    sell_price = price * (1 - SLIPPAGE_PCT)
    proceeds = sell_price * pos["qty"] * (1 - TAX_PCT)
    etf["capital"] += proceeds

    pnl_pct = sell_price / pos["avg_price"] - 1
    days_held = 0
    try:
        days_held = (pd.Timestamp(today_str) - pd.Timestamp(pos["entry_date"])).days
    except Exception:
        pass

    trade = {
        "code": pos["code"],
        "name": pos["name"],
        "direction": pos.get("direction", ""),
        "entry_date": pos["entry_date"],
        "exit_date": today_str,
        "avg_price": pos["avg_price"],
        "exit_price": round(sell_price),
        "qty": pos["qty"],
        "pnl_pct": round(pnl_pct * 100, 2),
        "exit_reason": reason,
        "days_held": days_held,
    }
    etf["closed_trades"].append(trade)
    etf["position"] = None
    return trade


def _is_inverse_strict_signal(today_str: str) -> tuple[bool, str]:
    """Phase 11 검증된 strict 조건: KOSPI 1d -2.5%↓ AND 외인 5d -5조↓ AND 외인 1d -3000억↓.

    인버스 진입 가드. 미달이면 매수 SKIP.
    Returns: (통과여부, 사유)
    """
    try:
        # KOSPI 1d
        kdf = pd.read_csv(KOSPI_CSV_PATH, encoding="utf-8-sig")
        kdf["Date"] = pd.to_datetime(kdf["Date"]).dt.strftime("%Y-%m-%d")
        kdf = kdf.sort_values("Date")
        today_kospi = kdf[kdf["Date"] == today_str]
        if today_kospi.empty:
            return (False, f"KOSPI {today_str} 데이터 없음")
        prev_close = kdf.iloc[-2]["close"] if len(kdf) >= 2 else None
        if prev_close is None:
            return (False, "KOSPI 전일 데이터 없음")
        ret_1d = (today_kospi.iloc[0]["close"] / prev_close - 1) * 100

        # 외인 5d/1d
        if not INVESTOR_DB_PATH.exists():
            return (False, "investor_daily.db 없음")
        conn = sqlite3.connect(INVESTOR_DB_PATH)
        rows = conn.execute(
            """SELECT date, SUM(net_val) as foreign_net
               FROM investor_daily
               WHERE investor='외국인'
               GROUP BY date ORDER BY date DESC LIMIT 5""",
        ).fetchall()
        conn.close()
        if len(rows) < 5:
            return (False, f"외인 데이터 부족 ({len(rows)}일)")
        foreign_5d_eok = sum(r[1] for r in rows) / 1e8
        foreign_1d_eok = rows[0][1] / 1e8

        # strict 판정
        if ret_1d > INVERSE_STRICT_KOSPI_1D:
            return (False, f"KOSPI 1d {ret_1d:+.2f}% (>{INVERSE_STRICT_KOSPI_1D}%)")
        if foreign_5d_eok > INVERSE_STRICT_FOREIGN_5D_EOK:
            return (False, f"외인 5d {foreign_5d_eok:+.0f}억 (>{INVERSE_STRICT_FOREIGN_5D_EOK}억)")
        if foreign_1d_eok > INVERSE_STRICT_FOREIGN_1D_EOK:
            return (False, f"외인 1d {foreign_1d_eok:+.0f}억 (>{INVERSE_STRICT_FOREIGN_1D_EOK}억)")
        return (True, f"KOSPI {ret_1d:+.2f}% / 외인5d {foreign_5d_eok:+.0f}억 / 외인1d {foreign_1d_eok:+.0f}억")
    except Exception as e:
        return (False, f"strict 체크 실패: {e}")


def manage_etf_position(pf: dict, today_str: str) -> dict:
    """JARVIS 방향 기반 ETF 포지션 관리.

    Returns:
        {"action": "BUY"/"SELL"/"SWITCH"/"HOLD"/"SKIP", ...details}
    """
    etf = pf.setdefault("etf_trading", _default_etf_state())

    # JARVIS 방향 로드
    if not JARVIS_PATH.exists():
        logger.info("[ETF] jarvis_direction.json 없음 → 스킵")
        return {"action": "SKIP", "reason": "jarvis 없음"}

    try:
        with open(JARVIS_PATH, "r", encoding="utf-8") as f:
            jarvis = json.load(f)
    except Exception:
        return {"action": "SKIP", "reason": "jarvis 파싱 실패"}

    direction = jarvis.get("direction", "NEUTRAL")
    meta_score = jarvis.get("meta_score", 0)
    confidence = jarvis.get("confidence", 0)

    current_pos = etf.get("position")
    target_etf = ETF_MAP.get(direction)  # None if NEUTRAL

    result = {
        "action": "HOLD",
        "direction": direction,
        "meta_score": meta_score,
        "confidence": confidence,
    }

    # ── Case 1: NEUTRAL → 포지션 있으면 청산 ──
    if target_etf is None:
        if current_pos:
            price = get_etf_price(current_pos["code"])
            if price <= 0:
                price = current_pos["avg_price"]
            trade = _close_etf_position(etf, price, today_str, "NEUTRAL_EXIT")
            result.update({
                "action": "SELL",
                "name": trade["name"],
                "pnl_pct": trade["pnl_pct"],
                "reason": "방향 NEUTRAL 전환",
            })
        else:
            result["reason"] = "NEUTRAL, 포지션 없음"
        return result

    # ── Case 2: 같은 ETF 보유 중 → 손절/익절/트레일링/유지 체크 (5/16 강화) ──
    if current_pos and current_pos["code"] == target_etf["code"]:
        price = get_etf_price(current_pos["code"])
        if price > 0:
            avg = current_pos["avg_price"]
            pnl = price / avg - 1
            # 고점 갱신
            peak = current_pos.get("peak_price", avg)
            if price > peak:
                current_pos["peak_price"] = price
                peak = price
            peak_pct = (peak - avg) / avg
            drop_from_peak = (price - peak) / peak if peak > 0 else 0

            # 인버스 종목 구분
            is_inverse = target_etf["code"] in ("114800", "252670", "251340")
            stop_loss = ETF_INVERSE_STOP_LOSS if is_inverse else ETF_STOP_LOSS_PCT

            # 보유일 (인버스는 D+2 강제 청산)
            try:
                entry_dt = datetime.strptime(current_pos["entry_date"], "%Y-%m-%d")
                today_dt = datetime.strptime(today_str, "%Y-%m-%d")
                days_held = (today_dt - entry_dt).days
            except (ValueError, KeyError):
                days_held = 0

            exit_reason = None
            # 1. 손절
            if pnl <= stop_loss:
                exit_reason = "ETF_STOP_LOSS"
            # 2. 인버스 최대 보유 2일 강제 청산
            elif is_inverse and days_held >= ETF_INVERSE_MAX_HOLD:
                exit_reason = "INVERSE_MAX_HOLD"
            # 3. +10% 분할 익절 T2 (전량, 단순화)
            elif pnl >= ETF_TAKE_PROFIT_T2:
                exit_reason = "ETF_TAKE_PROFIT_T2"
            # 4. 트레일링 — +5% 도달 후 고점 -2% 하락
            elif peak_pct >= ETF_TAKE_PROFIT_T1 and drop_from_peak <= ETF_TRAILING_STOP:
                exit_reason = "ETF_TRAILING_STOP"

            if exit_reason:
                trade = _close_etf_position(etf, price, today_str, exit_reason)
                result.update({
                    "action": exit_reason,
                    "name": trade["name"],
                    "pnl_pct": trade["pnl_pct"],
                    "peak_pct": round(peak_pct * 100, 2),
                })
            else:
                result.update({
                    "action": "HOLD",
                    "name": current_pos["name"],
                    "pnl_pct": round(pnl * 100, 2),
                    "peak_pct": round(peak_pct * 100, 2),
                })
        return result

    # ── Case 3: 다른 ETF 보유 중 → 스위칭 ──
    if current_pos and current_pos["code"] != target_etf["code"]:
        price = get_etf_price(current_pos["code"])
        if price <= 0:
            price = current_pos["avg_price"]
        sell_trade = _close_etf_position(etf, price, today_str, "DIRECTION_SWITCH")
        result["sell_trade"] = {
            "name": sell_trade["name"],
            "pnl_pct": sell_trade["pnl_pct"],
        }

    # ── Case 4: 신규 매수 ──
    # Phase 11 가드: 인버스(SHORT/STRONG_SHORT) 진입 시 strict 조건 검증
    if direction in ("SHORT", "STRONG_SHORT"):
        ok, reason = _is_inverse_strict_signal(today_str)
        if not ok:
            result.update({
                "action": "SKIP",
                "reason": f"인버스 strict 미달 — {reason}",
            })
            return result
        logger.info(f"[ETF] 인버스 strict 통과: {reason}")

    # P0-7 위험감지 게이트 (정보봇 SDK)
    from src.utils.risk_gate import (
        get_position_multiplier_safe,
        should_block_new_entry_safe,
        get_risk_status_safe,
    )
    risk_block = should_block_new_entry_safe()
    risk_status = get_risk_status_safe()
    risk_score = risk_status.get("total_score", 0)
    risk_level = risk_status.get("level_kr", "정상")
    if risk_block:
        result.update({
            "action": "SKIP",
            "reason": f"위험감지 {risk_level} ({risk_score}점) — ETF 진입 차단",
        })
        return result

    # 롱은 multiplier 적용, 인버스는 1.0 유지 (DANGER에 오히려 reasonable)
    is_long_direction = direction in ("LONG", "STRONG_LONG")
    etf_mult = get_position_multiplier_safe() if is_long_direction else 1.0
    if etf_mult < 1.0:
        logger.info(f"[ETF] 위험감지 {risk_level} ({risk_score}점) — 롱 매수금액 ×{etf_mult}")

    price = get_etf_price(target_etf["code"])
    if price <= 0:
        result.update({"action": "SKIP", "reason": f"{target_etf['name']} 가격 없음"})
        return result

    buy_price = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
    buy_amount = min(etf["capital"] * 0.95 * etf_mult, ETF_CAPITAL * 0.95 * etf_mult)
    qty = int(buy_amount / buy_price)
    if qty <= 0:
        result.update({"action": "SKIP", "reason": "ETF 자본 부족"})
        return result

    cost = buy_price * qty
    if cost > etf["capital"]:
        result.update({"action": "SKIP", "reason": "ETF 현금 부족"})
        return result

    etf["capital"] -= cost
    etf["position"] = {
        "code": target_etf["code"],
        "name": target_etf["name"],
        "entry_date": today_str,
        "avg_price": round(buy_price),
        "qty": qty,
        "cost": round(cost),
        "direction": direction,
        "peak_price": round(buy_price),  # 5/16: 트레일링용
    }

    action_type = "SWITCH" if "sell_trade" in result else "BUY"
    result.update({
        "action": action_type,
        "name": target_etf["name"],
        "price": round(price),
        "qty": qty,
        "cost": round(cost),
    })
    return result


def update_etf_equity(pf: dict, today_str: str) -> float:
    """ETF 일일 자산 평가. equity 반환."""
    etf = pf.get("etf_trading")
    if not etf:
        return 0

    equity = etf["capital"]
    pos = etf.get("position")
    if pos:
        price = get_etf_price(pos["code"])
        if price > 0:
            equity += price * pos["qty"]
        else:
            equity += pos["avg_price"] * pos["qty"]

    etf["daily_equity"] = [e for e in etf.get("daily_equity", []) if e["date"] != today_str]
    etf["daily_equity"].append({
        "date": today_str,
        "equity": round(equity),
        "position": pos["name"] if pos else None,
        "direction": pos["direction"] if pos else None,
    })
    return equity


# ═══════════════════════════════════════════════
# FLOWX 업로드
# ═══════════════════════════════════════════════

def upload_to_flowx(entries: list[dict], exits: list[dict], stats: dict,
                     etf_result: dict | None = None) -> None:
    """FLOWX Supabase에 매매 기록 업로드 (Paper + ETF + 위험감지 메타)."""
    try:
        from src.adapters.flowx_uploader import FlowxUploader, build_paper_trade
        uploader = FlowxUploader()
        if not uploader.is_active:
            return

        # 위험감지 메타 (모든 trade의 memo에 첨부)
        risk_memo = ""
        try:
            from src.utils.risk_gate import get_risk_status_safe, get_position_multiplier_safe
            rs = get_risk_status_safe()
            if rs:
                risk_memo = f" | 위험:{rs.get('level_kr', '-')}({rs.get('total_score', 0)}점) ×{get_position_multiplier_safe()}"
        except Exception:
            pass

        for e in entries:
            trade = build_paper_trade(
                code=e["ticker"], name=e["name"], side="BUY",
                price=e["price"], quantity=e["qty"],
                strategy=e["strategy"], memo=f"등급:{e['grade']}{risk_memo}",
                stats=stats,
            )
            uploader.upload_paper_trade(trade)

        for x in exits:
            trade = build_paper_trade(
                code=x["ticker"], name=x["name"], side="SELL",
                price=x["exit_price"], quantity=x["qty"],
                strategy=x["reason"], pnl_pct=x["pnl_pct"],
                memo=f"{'부분' if x.get('partial') else '전량'}{risk_memo}",
                stats=stats,
            )
            uploader.upload_paper_trade(trade)

        # ETF 매매도 paper_trades에 (strategy로 구분: ETF_BUY/ETF_SELL/ETF_SWITCH)
        if etf_result and etf_result.get("action") in ("BUY", "SWITCH", "SELL",
                                                       "ETF_STOP_LOSS", "ETF_TAKE_PROFIT_T2",
                                                       "ETF_TRAILING_STOP", "INVERSE_MAX_HOLD"):
            action = etf_result.get("action", "")
            direction = etf_result.get("direction", "")
            etf_strategy = f"ETF_{direction}_{action}"
            # SELL 계열은 pnl_pct 첨부
            etf_pnl = etf_result.get("pnl_pct")
            side = "SELL" if action in ("SELL", "ETF_STOP_LOSS", "ETF_TAKE_PROFIT_T2",
                                          "ETF_TRAILING_STOP", "INVERSE_MAX_HOLD") else "BUY"
            etf_trade = build_paper_trade(
                code=etf_result.get("code", ""),
                name=etf_result.get("name", ""),
                side=side,
                price=etf_result.get("price", 0) or etf_result.get("exit_price", 0),
                quantity=etf_result.get("qty", 0),
                strategy=etf_strategy,
                pnl_pct=etf_pnl,
                memo=f"JARVIS{risk_memo}",
                stats=stats,
            )
            uploader.upload_paper_trade(etf_trade)

        logger.info("[FLOWX] Paper+ETF 업로드: BUY %d건, SELL %d건", len(entries), len(exits))
    except Exception as e:
        logger.warning("[FLOWX] Paper 업로드 실패: %s", e)


# ═══════════════════════════════════════════════
# 텔레그램 리포트
# ═══════════════════════════════════════════════

def send_daily_report(
    today_str: str,
    entries: list[dict],
    exits: list[dict],
    pf: dict,
    stats: dict,
    candidates_count: int,
    etf_result: dict | None = None,
) -> None:
    """일일 Paper Trading 텔레그램 리포트."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        return

    # P0-7 위험감지 등급 (헤더에 표시)
    risk_header = ""
    try:
        from src.utils.risk_gate import get_risk_status_safe, get_position_multiplier_safe
        rs = get_risk_status_safe()
        if rs:
            level_kr = rs.get("level_kr", "정상")
            score = rs.get("total_score", 0)
            mult = get_position_multiplier_safe()
            emoji = {"위기": "🔴", "위험": "🟠", "경고": "🟡", "주의": "🟢", "정상": "✅"}.get(level_kr, "ℹ️")
            risk_header = f"{emoji} 위험감지: {level_kr} ({score}점) ×{mult}"
    except Exception:
        pass

    # 전일 대비 일일 손익 계산
    daily_eq = pf.get("daily_equity", [])
    today_eq = stats["equity"]
    diff_line = ""
    if len(daily_eq) >= 2:
        prev_eq = daily_eq[-2].get("equity", today_eq)
        diff = today_eq - prev_eq
        diff_pct = (diff / prev_eq * 100) if prev_eq else 0
        diff_emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        diff_line = f"{diff_emoji} 일일: {diff:+,}원 ({diff_pct:+.2f}%)"

    lines = [
        f"📋 [PAPER] 일일 리포트 ({today_str})",
    ]
    if risk_header:
        lines.append(risk_header)
    lines += [
        f"자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)",
    ]
    if diff_line:
        lines.append(diff_line)
    lines += [
        f"PF: {stats['pf']} | MDD: {stats['mdd']:.1f}% | "
        f"승률: {stats['win_rate']:.0f}% ({stats['wins']}W/{stats['losses']}L)",
        "",
    ]

    # 한국어 매핑
    from src.utils.strategy_kr import strategy_kr as _skr

    if entries:
        lines.append(f"-- 신규 진입 ({len(entries)}건) --")
        for e in entries:
            ename = ticker_to_name(e.get("ticker", "")) if e["name"] == e.get("ticker") else e["name"]
            lines.append(
                f"  [{e['grade']}] {ename} {e['price']:,}원 "
                f"x{e['qty']}주 ({_skr(e['strategy'])})"
            )
        lines.append("")

    if exits:
        lines.append(f"-- 매도 ({len(exits)}건) --")
        for x in exits:
            emoji = "🟢" if x["pnl_pct"] > 0 else "🔴"
            partial = " (부분 매도)" if x.get("partial") else ""
            xname = ticker_to_name(x.get("ticker", "")) if x["name"] == x.get("ticker") else x["name"]
            lines.append(
                f"  {emoji} {xname} {x['pnl_pct']:+.1f}% "
                f"[{_skr(x['reason'])}]{partial}"
            )
        lines.append("")

    if not entries and not exits:
        lines.append(f"스캔: {candidates_count}후보 | 변동 없음")
        lines.append("")

    # 보유 현황
    if pf["positions"]:
        lines.append(f"-- 보유 ({len(pf['positions'])}종목) --")
        for ticker, pos in pf["positions"].items():
            cur_price, _ = get_latest_price(ticker)
            if cur_price > 0:
                pnl = (cur_price / pos["avg_price"] - 1) * 100
            else:
                pnl = 0
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            days = 0
            try:
                days = (pd.Timestamp(today_str) - pd.Timestamp(pos["entry_date"])).days
            except Exception:
                pass
            pname = ticker_to_name(ticker) if pos["name"] == ticker else pos["name"]
            lines.append(
                f"  {emoji} {pname} {pnl:+.1f}% "
                f"({days}일, {pos.get('strategy', '')})"
            )

    # ETF 방향 트레이딩 섹션
    if etf_result and etf_result.get("action") != "SKIP":
        lines.append("")
        etf_state = pf.get("etf_trading", {})
        etf_eq = etf_state.get("daily_equity", [{}])[-1].get("equity", ETF_CAPITAL) if etf_state.get("daily_equity") else ETF_CAPITAL
        etf_ret = (etf_eq / ETF_CAPITAL - 1) * 100
        lines.append(f"-- ETF 방향 ({etf_eq:,}원 {etf_ret:+.1f}%) --")
        lines.append(f"  JARVIS: {etf_result.get('direction', '?')} "
                      f"(점수 {etf_result.get('meta_score', 0):+.1f})")
        action = etf_result.get("action", "")
        if action in ("BUY", "SWITCH"):
            lines.append(f"  매수: {etf_result.get('name', '')} "
                          f"{etf_result.get('price', 0):,}원 x{etf_result.get('qty', 0)}주")
        elif action in ("SELL", "STOP_LOSS"):
            lines.append(f"  청산: {etf_result.get('name', '')} "
                          f"{etf_result.get('pnl_pct', 0):+.1f}%")
        elif action == "HOLD" and etf_result.get("name"):
            lines.append(f"  유지: {etf_result.get('name', '')} "
                          f"{etf_result.get('pnl_pct', 0):+.1f}%")

    # picks_v2 내일 추천 TOP 5 (BAT-PICKV2 17:45 산출분, 17:50 시점에 일일 리포트 메시지에 포함)
    try:
        import csv
        today_compact = today_str.replace("-", "")
        picks_csv = DATA_DIR / f"picks_v2_{today_compact}.csv"
        if picks_csv.exists():
            with picks_csv.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = sorted(
                    [r for r in reader],
                    key=lambda x: int(x.get("score", 0) or 0),
                    reverse=True,
                )[:5]
            if rows:
                lines.append("")
                lines.append("-- 📅 내일 추천 TOP 5 (picks_v2) --")
                for i, r in enumerate(rows, 1):
                    name = r.get("name", "") or ticker_to_name(r.get("ticker", ""))
                    lines.append(f"  {i}. {name} ({r.get('ticker', '')}) score {r.get('score', '?')}")
    except Exception as e:
        logger.warning(f"picks_v2 TOP 5 로드 실패: {e}")

    # [LIVE] 실전 매매 섹션 (5/26 이후 활성. AUTO_TRADING_ENABLED=1 시 실제 데이터)
    try:
        if os.getenv("AUTO_TRADING_ENABLED", "0") == "1":
            from src.utils.auto_trading_volume import get_today_volume
            live = get_today_volume()
            if live["total_trades"] > 0:
                max_amount = int(os.getenv("AUTO_TRADING_MAX_AMOUNT", "300000"))
                max_trades = int(os.getenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "5"))
                lines.append("")
                lines.append(f"-- 💵 [LIVE] 실전 매매 ({today_str}) --")
                lines.append(
                    f"  매수: {live['total_trades']}회 / {max_trades}회 | "
                    f"금액: {live['total_amount']:,}원 / {max_amount:,}원 "
                    f"({live['total_amount']/max_amount*100:.1f}%)"
                )
                for b in live["buys"][-3:]:  # 최근 3건
                    lines.append(
                        f"  • {b['ticker']} x{b['qty']}주 @{b['price']:,}원 ({b['ts'][11:19]})"
                    )
    except Exception as e:
        logger.warning(f"[LIVE] 섹션 로드 실패: {e}")

    # Phase 12 장중 학습 시그널 (어제 수집한 오늘 후보)
    try:
        today_compact = today_str.replace("-", "")
        intra_sig = DATA_DIR / "intraday" / f"intraday_signals_{today_compact}.json"
        if intra_sig.exists():
            sig = json.loads(intra_sig.read_text(encoding="utf-8"))
            cands = sig.get("candidates", [])[:5]
            if cands:
                lines.append("")
                lines.append(f"-- 🧠 장중 학습 후보 ({len(cands)}건) --")
                for c in cands:
                    name = ticker_to_name(c.get("code", ""))
                    lines.append(
                        f"  {name} early {c.get('early_ret_pct', 0):+.1f}% "
                        f"/ 체결강도 {c.get('strength_avg', 0):.0f} "
                        f"/ 매수비 {c.get('buy_ratio', 0):.2f}"
                    )
    except Exception as e:
        logger.warning(f"장중 학습 시그널 로드 실패: {e}")

    msg = "\n".join(lines)
    try:
        send_message(msg)
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


def send_weekly_report(pf: dict, stats: dict) -> None:
    """금요일 주간 리포트."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        return

    equity_list = pf.get("daily_equity", [])
    if len(equity_list) < 2:
        return

    recent5 = equity_list[-5:]
    week_start_eq = recent5[0]["equity"]
    current_eq = equity_list[-1]["equity"]
    week_return = (current_eq / week_start_eq - 1) * 100 if week_start_eq > 0 else 0

    # 이번 주 거래
    week_start_date = recent5[0]["date"]
    week_trades = [
        t for t in pf["closed_trades"]
        if t.get("exit_date", "") >= week_start_date
    ]

    lines = [
        "📊 [PAPER] 주간 리포트",
        f"기간: {week_start_date} ~ {equity_list[-1]['date']}",
        "",
        f"총 자산: {stats['equity']:,}원 (누적 {stats['total_return_pct']:+.1f}%)",
        f"주간 수익: {week_return:+.1f}%",
        f"PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%",
        f"누적: {stats['total_trades']}건 ({stats['wins']}W/{stats['losses']}L)",
        "",
    ]

    if week_trades:
        lines.append(f"-- 금주 거래 ({len(week_trades)}건) --")
        for t in week_trades:
            emoji = "🟢" if t["pnl_pct"] > 0 else "🔴"
            tname = ticker_to_name(t.get("ticker", "")) if t["name"] == t.get("ticker") else t["name"]
            lines.append(
                f"  {emoji} {tname} {t['pnl_pct']:+.1f}% "
                f"[{t['exit_reason']}] {t.get('days_held', '?')}일"
            )
        lines.append("")

    if pf["positions"]:
        lines.append(f"-- 보유 ({len(pf['positions'])}종목) --")
        for ticker, pos in pf["positions"].items():
            cur_price, _ = get_latest_price(ticker)
            pnl = (cur_price / pos["avg_price"] - 1) * 100 if cur_price > 0 else 0
            pname = ticker_to_name(ticker) if pos["name"] == ticker else pos["name"]
            lines.append(f"  {pname} {pnl:+.1f}% ({pos.get('strategy', '')})")

    # 🆕 시그널별 누적 적중률 (전체 closed_trades 기준) — 한국어 풀이
    all_trades = pf.get("closed_trades", [])
    if len(all_trades) >= 5:
        from collections import defaultdict
        from src.utils.strategy_kr import strategy_kr, pf_grade
        bucket = defaultdict(lambda: {"wins": 0, "losses": 0, "gains": 0.0, "losses_amt": 0.0, "total": 0})
        for t in all_trades:
            strat = t.get("strategy") or "기타"
            pnl_pct = t.get("pnl_pct", 0)
            b = bucket[strat]
            b["total"] += 1
            if pnl_pct > 0:
                b["wins"] += 1
                b["gains"] += pnl_pct
            else:
                b["losses"] += 1
                b["losses_amt"] += abs(pnl_pct)

        # 승률 + 손익비 계산
        rows = []
        for strat, b in bucket.items():
            if b["total"] < 3:
                continue
            wr = b["wins"] / b["total"] * 100
            pf_val = b["gains"] / b["losses_amt"] if b["losses_amt"] > 0 else 99.9
            avg_ret = (b["gains"] - b["losses_amt"]) / b["total"]
            rows.append((strat, b["total"], wr, pf_val, avg_ret))

        rows.sort(key=lambda r: r[3], reverse=True)  # 손익비 내림차순

        if rows:
            lines.append("")
            lines.append("-- 🎯 전략별 누적 성과 --")
            for strat, n, wr, pf_val, avg in rows[:8]:
                emoji, grade = pf_grade(pf_val)
                strat_kr = strategy_kr(strat)
                lines.append(f"  {emoji} {strat_kr}: {n}건 승률 {wr:.0f}% 손익비 {pf_val:.2f} 평균 {avg:+.1f}%")

    msg = "\n".join(lines)
    try:
        send_message(msg)
    except Exception as e:
        logger.warning("주간 리포트 전송 실패: %s", e)


# ═══════════════════════════════════════════════
# 메인 일일 실행
# ═══════════════════════════════════════════════

def run_daily(force_rebalance: bool = False) -> dict:
    """일일 Paper Trading 메인 루틴.

    Args:
        force_rebalance: True면 요일 무관하게 리밸런싱 강제 실행
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 주말 가드: 토(5)/일(6)에는 실행하지 않음
    weekday = datetime.now().weekday()
    if weekday >= 5:
        print(f"  [PAPER] 주말({['월','화','수','목','금','토','일'][weekday]}) — Paper Trading 스킵")
        return {"status": "skip", "reason": "weekend", "date": today_str}

    do_rebalance = force_rebalance or is_rebalance_day()
    mode_tag = "리밸런싱" if do_rebalance else "일일"

    print("=" * 60)
    print(f"  [PAPER] {mode_tag} Paper Trading — {today_str}")
    if do_rebalance:
        print(f"  ** 1주 Rolling 리밸런싱 모드 **")
    print("=" * 60)

    # 1. 포트폴리오 로드
    pf = load_portfolio()
    print(f"  자본금: {pf['initial_capital']:,}원 | 현금: {pf['capital']:,}원")
    print(f"  보유: {len(pf['positions'])}종목")

    # 2. 추천 종목 수집 (리밸런싱 전에 필요)
    candidates = collect_candidates()
    print(f"\n  추천 후보: {len(candidates)}종목")
    for c in candidates[:5]:
        print(f"    [{c['grade']}] {c['name']} score={c['score']} "
              f"{c['price']:,}원 ({c['strategy']})")

    # 3. 리밸런싱 또는 일반 매도 체크
    exits = []
    if do_rebalance:
        # 금요일: 미추천 종목 강제 청산 + 추천 유지
        exits = weekly_rebalance(pf, candidates, today_str)
        if exits:
            print(f"\n  리밸런싱 청산: {len(exits)}건")
            for x in exits:
                print(f"    {x['name']} {x['pnl_pct']:+.1f}% [{x['reason']}]")
        kept = len(pf["positions"])
        if kept > 0:
            print(f"  유지: {kept}종목 (이번 주 추천 포함)")
    else:
        # 평일: 손절/익절/보유일 기반 매도
        exits = check_exits(pf, today_str)
        if exits:
            print(f"\n  매도 시그널: {len(exits)}건")
            for x in exits:
                print(f"    {x['name']} {x['pnl_pct']:+.1f}% [{x['reason']}]")

    # 4. 신규 진입
    entries = enter_new_positions(pf, candidates, today_str)
    if entries:
        print(f"\n  신규 진입: {len(entries)}건")
        for e in entries:
            print(f"    [{e['grade']}] {e['name']} {e['price']:,}원 "
                  f"x{e['qty']}주 = {e['cost']:,}원")

    # 4-1. ETF 방향 트레이딩 (JARVIS 연동)
    etf_result = manage_etf_position(pf, today_str)
    etf_equity = update_etf_equity(pf, today_str)
    etf_action = etf_result.get("action", "SKIP")
    print(f"\n  [ETF] JARVIS방향={etf_result.get('direction', '?')} "
          f"| 액션={etf_action}")
    if etf_action in ("BUY", "SWITCH"):
        print(f"    매수: {etf_result.get('name', '')} "
              f"{etf_result.get('price', 0):,}원 x{etf_result.get('qty', 0)}주")
        if etf_action == "SWITCH":
            st = etf_result.get("sell_trade", {})
            print(f"    매도: {st.get('name', '')} {st.get('pnl_pct', 0):+.1f}%")
    elif etf_action in ("SELL", "STOP_LOSS"):
        print(f"    청산: {etf_result.get('name', '')} "
              f"{etf_result.get('pnl_pct', 0):+.1f}% [{etf_action}]")
    elif etf_action == "HOLD" and etf_result.get("name"):
        print(f"    유지: {etf_result.get('name', '')} "
              f"{etf_result.get('pnl_pct', 0):+.1f}%")

    # 5. 일일 자산 기록
    equity = update_equity(pf, today_str)
    stats = calc_stats(pf)

    # 6. 저장
    save_portfolio(pf)

    # 7~8. 외부 발신(텔레그램·FLOWX 업로드) — A안(현행)만 수행.
    #   B안은 관측·격리: 중복 알림 + 대시보드 A·B 혼선 방지 (paper_portfolio_b.json만 누적).
    if CONVICTION_MODE != "B":
        # 7. 텔레그램 리포트
        send_daily_report(today_str, entries, exits, pf, stats, len(candidates), etf_result)
        # 7-1. 리밸런싱일 → 주간 리포트도 자동 전송
        if do_rebalance:
            send_weekly_report(pf, stats)
        # 8. FLOWX 업로드 (Paper + ETF + 위험감지 메타)
        upload_to_flowx(entries, exits, stats, etf_result=etf_result)
    else:
        logger.info("[CONVICTION-B] 외부 발신 스킵 (관측·격리) — entries=%d exits=%d",
                    len(entries), len(exits))

    # 9. 콘솔 요약
    print(f"\n  {'='*40}")
    print(f"  [종목] 자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)")
    print(f"  PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%")
    print(f"  거래: {stats['total_trades']}건 "
          f"({stats['wins']}W/{stats['losses']}L) "
          f"승률 {stats['win_rate']:.0f}%")
    print(f"  보유: {stats['open_positions']}종목 | 현금: {stats['cash']:,}원")
    if etf_equity > 0:
        etf_return = (etf_equity / ETF_CAPITAL - 1) * 100
        print(f"  [ETF]  자산: {etf_equity:,}원 ({etf_return:+.1f}%)")
    print("=" * 60)

    return {
        "status": "ok",
        "date": today_str,
        "rebalanced": do_rebalance,
        "entries": len(entries),
        "exits": len(exits),
        "candidates": len(candidates),
        "etf_action": etf_action,
        **stats,
    }


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="통합 Paper Trading 엔진")
    parser.add_argument("--reset", action="store_true", help="포트폴리오 초기화 (3000만원)")
    parser.add_argument("--rebalance", action="store_true", help="강제 리밸런싱 (금요일 아닌 날도)")
    parser.add_argument("--weekly", action="store_true", help="주간 리포트")
    parser.add_argument("--status", action="store_true", help="현재 상태")
    parser.add_argument("--dry-run", action="store_true", help="매매 없이 후보만 출력")
    args = parser.parse_args()

    if args.reset:
        pf = _default_portfolio()
        pf["etf_trading"] = _default_etf_state()
        save_portfolio(pf)
        print("  [PAPER] 포트폴리오 초기화 완료 (종목 3,000만 + ETF 1,000만)")
        return

    if args.status:
        pf = load_portfolio()
        stats = calc_stats(pf)
        print(f"  [종목] 자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)")
        print(f"  PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%")
        print(f"  거래: {stats['total_trades']}건 "
              f"({stats['wins']}W/{stats['losses']}L) 승률 {stats['win_rate']:.0f}%")
        print(f"  보유: {stats['open_positions']}종목 | 현금: {stats['cash']:,}원")
        if pf["positions"]:
            print("  -- 보유 종목 --")
            for ticker, pos in pf["positions"].items():
                cur_price, _ = get_latest_price(ticker)
                pnl = (cur_price / pos["avg_price"] - 1) * 100 if cur_price > 0 else 0
                print(f"    {pos['name']} {pnl:+.1f}% | "
                      f"진입 {pos['avg_price']:,}원 | {pos.get('strategy', '')}")
        # ETF 상태
        etf = pf.get("etf_trading", {})
        if etf:
            etf_eq_list = etf.get("daily_equity", [])
            etf_eq = etf_eq_list[-1]["equity"] if etf_eq_list else etf.get("capital", ETF_CAPITAL)
            etf_ret = (etf_eq / ETF_CAPITAL - 1) * 100
            print(f"\n  [ETF] 자산: {etf_eq:,}원 ({etf_ret:+.1f}%)")
            etf_pos = etf.get("position")
            if etf_pos:
                ep = get_etf_price(etf_pos["code"])
                etf_pnl = (ep / etf_pos["avg_price"] - 1) * 100 if ep > 0 else 0
                print(f"    {etf_pos['name']} {etf_pnl:+.1f}% "
                      f"| 방향: {etf_pos.get('direction', '?')} "
                      f"| 진입: {etf_pos['avg_price']:,}원")
            else:
                print(f"    포지션 없음 | 현금: {etf.get('capital', 0):,}원")
            etf_trades = etf.get("closed_trades", [])
            if etf_trades:
                etf_wins = sum(1 for t in etf_trades if t["pnl_pct"] > 0)
                print(f"    거래: {len(etf_trades)}건 "
                      f"({etf_wins}W/{len(etf_trades)-etf_wins}L)")
        return

    if args.weekly:
        pf = load_portfolio()
        stats = calc_stats(pf)
        send_weekly_report(pf, stats)
        print("  주간 리포트 전송 완료")
        return

    if args.dry_run:
        candidates = collect_candidates()
        print(f"\n  [DRY-RUN] 후보 {len(candidates)}종목:")
        for i, c in enumerate(candidates, 1):
            print(f"    {i}. [{c['grade']}] {c['name']} score={c['score']} "
                  f"{c['price']:,}원 ({c['strategy']}) — {c['reason']}")
        return

    result = run_daily(force_rebalance=args.rebalance)
    print(f"\n  결과: {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
