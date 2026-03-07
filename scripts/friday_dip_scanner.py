"""금요일 오후 투매 역매수 스캐너 — "강한 종목의 비정상 급락" 감지

백테스트 검증 결과 (3년, 1030종목):
  - RSI > 60 + 거래량 1.5x + SMA20 위 + ADX > 25 + 고가대비 -2% 급락
  - 금요일 기대값 +0.67%, 손익비 1.37
  - -1% 손절 적용 시 기대값 +1.54%

사용법:
  python -u -X utf8 scripts/friday_dip_scanner.py --scan           # 실시간 스캔
  python -u -X utf8 scripts/friday_dip_scanner.py --scan --telegram  # + 텔레그램 발송
  python -u -X utf8 scripts/friday_dip_scanner.py --dry-run        # parquet 기반 시뮬레이션

의존성:
  - KIS API (mojito): 실시간 현재가/호가
  - data/processed/*.parquet: 기술지표 (RSI, ADX, BB, SMA20)
  - data/us_market/overnight_signal.json: 주말 리스크 등급
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stock_name_resolver import ticker_to_name
from src.telegram_sender import send_message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OVERNIGHT_PATH = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "friday_dip_scan.json"

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────

@dataclass
class DipConfig:
    """투매 감지 파라미터"""
    # 핵심 필터 (백테스트 검증됨)
    min_rsi: float = 55.0           # RSI 하한 (강한 종목)
    min_adx: float = 25.0           # ADX 하한 (추세 강도)
    min_drop_pct: float = 2.0       # 고가 대비 최소 하락률 (%)
    min_vol_ratio: float = 1.5      # 20일 평균 대비 거래량 배율
    require_above_sma20: bool = True  # SMA20 위 필수

    # 스코어링 가중치
    w_drop: float = 30.0            # 하락폭 점수 (0~30)
    w_volume: float = 20.0          # 거래량 가속 (0~20)
    w_oversold: float = 20.0        # 기술적 과매도 (0~20)
    w_supply: float = 20.0          # 수급 역발상 (0~20)
    w_weekend: float = 10.0         # 주말 리스크 보너스 (0~10)

    # 리스크 관리
    stop_loss_pct: float = 1.0      # 손절 기준 (%, 매수가 대비)
    max_candidates: int = 5         # 최대 추천 종목 수
    min_trading_value_억: float = 10.0  # 최소 거래대금 (억)

    # 주말 리스크 등급별 매수 허용량
    risk_allow = {
        "STRONG_BULL": 5,
        "MILD_BULL": 5,
        "NEUTRAL": 3,
        "MILD_BEAR": 1,
        "STRONG_BEAR": 0,
    }


@dataclass
class DipCandidate:
    """투매 감지 후보"""
    ticker: str = ""
    name: str = ""
    current_price: int = 0
    day_high: int = 0
    drop_pct: float = 0.0          # 고가 대비 하락률
    rsi: float = 0.0
    adx: float = 0.0
    bb_position: float = 0.0
    volume_ratio: float = 0.0      # 20일 평균 대비
    above_sma20: bool = False
    foreign_net: int = 0           # 장중 외국인 순매수
    inst_net: int = 0              # 장중 기관 순매수
    score: float = 0.0             # 종합 점수 (0~100)
    score_detail: dict = field(default_factory=dict)
    suggested_price: int = 0       # 추천 매수가
    stop_price: int = 0            # 손절가
    target_price: int = 0          # 목표가 (+3%)

    @property
    def supply_signal(self) -> str:
        if self.foreign_net > 0 and self.inst_net > 0:
            return "쌍매수"
        elif self.foreign_net < 0 and self.inst_net < 0:
            return "쌍매도"
        elif self.foreign_net > 0:
            return "외↑기↓"
        elif self.inst_net > 0:
            return "외↓기↑"
        else:
            return "외↓기↓"


# ──────────────────────────────────────────────────────────
# 주말 리스크 평가
# ──────────────────────────────────────────────────────────

def load_weekend_risk() -> dict:
    """US Overnight + NIGHTWATCH 기반 주말 리스크 등급"""
    result = {
        "grade": "NEUTRAL",
        "nightwatch_score": 0.0,
        "bond_vigilante_veto": False,
        "vix_level": 0.0,
        "special_rules": [],
        "allow_count": 3,
    }

    if not OVERNIGHT_PATH.exists():
        return result

    try:
        with open(OVERNIGHT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        result["grade"] = data.get("grade", "NEUTRAL")
        result["vix_level"] = data.get("vix", {}).get("level", 0)

        nw = data.get("nightwatch", {})
        result["nightwatch_score"] = nw.get("nightwatch_score", 0)
        result["bond_vigilante_veto"] = nw.get("bond_vigilante_veto", False)

        rules = data.get("special_rules", [])
        result["special_rules"] = [r.get("name", "") if isinstance(r, dict) else str(r) for r in rules]

        # 매수 허용량 결정
        cfg = DipConfig()
        allow = cfg.risk_allow.get(result["grade"], 3)

        # NIGHTWATCH 비토 → 절대 차단
        if result["bond_vigilante_veto"]:
            allow = 0

        # VIX 30+ → 1종목으로 제한
        if result["vix_level"] >= 30:
            allow = min(allow, 1)

        result["allow_count"] = allow

    except Exception as e:
        logger.warning(f"주말 리스크 로드 실패: {e}")

    return result


# ──────────────────────────────────────────────────────────
# 스코어링 (백테스트 검증 5조건 기반)
# ──────────────────────────────────────────────────────────

def score_dip(candidate: DipCandidate, cfg: DipConfig) -> float:
    """투매 종합 점수 계산 (0~100)"""
    detail = {}

    # 1. 하락폭 점수 (0~30)
    #    -2%: 10점, -3%: 20점, -4%+: 30점
    drop = abs(candidate.drop_pct)
    if drop >= 4.0:
        s_drop = cfg.w_drop
    elif drop >= 3.0:
        s_drop = cfg.w_drop * 0.67
    elif drop >= 2.0:
        s_drop = cfg.w_drop * 0.33
    else:
        s_drop = 0
    detail["drop"] = round(s_drop, 1)

    # 2. 거래량 가속 (0~20)
    #    1.5x: 10점, 2.0x: 15점, 3.0x+: 20점
    vr = candidate.volume_ratio
    if vr >= 3.0:
        s_vol = cfg.w_volume
    elif vr >= 2.0:
        s_vol = cfg.w_volume * 0.75
    elif vr >= 1.5:
        s_vol = cfg.w_volume * 0.5
    else:
        s_vol = 0
    detail["volume"] = round(s_vol, 1)

    # 3. 기술적 상태 (0~20)
    #    RSI 60~70: 10점, 70+: 15점 (더 강한 종목의 급락 = 더 큰 기회)
    #    BB position > 50%: +5점
    s_tech = 0
    if candidate.rsi >= 70:
        s_tech += 15
    elif candidate.rsi >= 60:
        s_tech += 10
    elif candidate.rsi >= 55:
        s_tech += 5

    if candidate.bb_position >= 0.5:
        s_tech += 5
    s_tech = min(s_tech, cfg.w_oversold)
    detail["tech"] = round(s_tech, 1)

    # 4. 수급 역발상 (0~20)
    #    외국인 매도 BUT 기관 매수: 15점 (기관이 받는 중)
    #    외+기 쌍매도: 0점 (진짜 악재 가능)
    #    외국인만 매도: 10점 (기관 중립은 과잉반응일 수 있음)
    s_supply = 0
    if candidate.foreign_net < 0 and candidate.inst_net > 0:
        s_supply = 15  # 기관이 외국인 매도를 흡수 → 최고
    elif candidate.foreign_net > 0 and candidate.inst_net > 0:
        s_supply = cfg.w_supply  # 쌍매수 중 급락 = 일시적 매물 소화
    elif candidate.foreign_net < 0 and candidate.inst_net < 0:
        s_supply = 0  # 쌍매도 → 위험
    elif candidate.foreign_net < 0:
        s_supply = 10  # 외인만 매도
    else:
        s_supply = 5
    detail["supply"] = round(s_supply, 1)

    # 5. 주말 리스크 보너스 (0~10)
    #    금요일 VIX < 20이면 +10점 (주말 리스크 낮음)
    #    장중이므로 전일 데이터로 대체
    # → 이 부분은 전체 스캔 후 외부에서 적용
    detail["weekend"] = 0.0

    total = s_drop + s_vol + s_tech + s_supply
    candidate.score_detail = detail
    return round(total, 1)


# ──────────────────────────────────────────────────────────
# 실시간 스캔 (KIS API)
# ──────────────────────────────────────────────────────────

def scan_live(cfg: DipConfig) -> list[DipCandidate]:
    """장중 실시간 투매 스캔 (KIS API)"""
    from src.adapters.kis_intraday_adapter import KisIntradayAdapter

    adapter = KisIntradayAdapter()
    candidates = []

    # 1) parquet에서 기술지표 + 유니버스 로드
    universe = _load_parquet_indicators()
    logger.info(f"유니버스 {len(universe)}종목 로드")

    for ticker, indicators in universe.items():
        # 필터 1: 기술지표 사전 필터 (API 호출 최소화)
        rsi = indicators.get("rsi_14", 50)
        adx = indicators.get("adx_14", 0)
        sma20 = indicators.get("sma20", 0)
        bb_pos = indicators.get("bb_position", 0.5)
        vol_avg20 = indicators.get("vol_avg20", 0)
        prev_close = indicators.get("close", 0)

        if rsi < cfg.min_rsi:
            continue
        if adx < cfg.min_adx:
            continue

        # 2) KIS API로 현재가 조회
        tick = adapter.fetch_tick(ticker)
        cur_price = tick.get("current_price", 0)
        day_high = tick.get("high_price", 0)
        volume = tick.get("cum_volume", tick.get("volume", 0))

        if cur_price <= 0 or day_high <= 0:
            continue

        # SMA20 필터
        if cfg.require_above_sma20 and sma20 > 0 and cur_price < sma20:
            continue

        # 고가 대비 하락률
        drop_pct = (cur_price - day_high) / day_high * 100
        if abs(drop_pct) < cfg.min_drop_pct:
            continue

        # 거래량 배율
        vol_ratio = volume / vol_avg20 if vol_avg20 > 0 else 1.0
        if vol_ratio < cfg.min_vol_ratio:
            continue

        # 거래대금 필터
        trading_value = cur_price * volume / 100_000_000  # 억원
        if trading_value < cfg.min_trading_value_억:
            continue

        # 3) 수급 조회
        flow = adapter.fetch_investor_flow(ticker)

        c = DipCandidate(
            ticker=ticker,
            name=ticker_to_name(ticker),
            current_price=cur_price,
            day_high=day_high,
            drop_pct=round(drop_pct, 2),
            rsi=round(rsi, 1),
            adx=round(adx, 1),
            bb_position=round(bb_pos, 2),
            volume_ratio=round(vol_ratio, 2),
            above_sma20=cur_price >= sma20 if sma20 > 0 else True,
            foreign_net=flow.get("foreign_net_buy", 0),
            inst_net=flow.get("inst_net_buy", 0),
        )

        c.score = score_dip(c, cfg)

        # 매수가/손절가/목표가 계산
        c.suggested_price = int(cur_price * 0.997)  # 현재가 -0.3%
        c.stop_price = int(c.suggested_price * (1 - cfg.stop_loss_pct / 100))
        c.target_price = int(c.suggested_price * 1.03)  # +3% 목표

        candidates.append(c)

    # 점수순 정렬
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:cfg.max_candidates * 2]  # 리스크 필터 전 여유분


# ──────────────────────────────────────────────────────────
# 드라이런 (parquet 기반 시뮬레이션)
# ──────────────────────────────────────────────────────────

def scan_dry_run(cfg: DipConfig) -> list[DipCandidate]:
    """최신 parquet 데이터로 투매 시뮬레이션 (장 외에서도 테스트 가능)"""
    candidates = []
    universe = _load_parquet_indicators()
    logger.info(f"드라이런: 유니버스 {len(universe)}종목")

    for ticker, ind in universe.items():
        rsi = ind.get("rsi_14", 50)
        adx = ind.get("adx_14", 0)
        close = ind.get("close", 0)
        high = ind.get("high", 0)
        sma20 = ind.get("sma20", 0)
        bb_pos = ind.get("bb_position", 0.5)
        volume = ind.get("volume", 0)
        vol_avg20 = ind.get("vol_avg20", 0)

        if rsi < cfg.min_rsi or adx < cfg.min_adx:
            continue
        if cfg.require_above_sma20 and sma20 > 0 and close < sma20:
            continue

        drop_pct = (close - high) / high * 100 if high > 0 else 0
        if abs(drop_pct) < cfg.min_drop_pct:
            continue

        vol_ratio = volume / vol_avg20 if vol_avg20 > 0 else 1.0
        if vol_ratio < cfg.min_vol_ratio:
            continue

        trading_value = close * volume / 100_000_000
        if trading_value < cfg.min_trading_value_억:
            continue

        # 수급은 parquet에서 추출 (외국인합계, 기관합계)
        foreign = ind.get("foreign_net", 0)
        inst = ind.get("inst_net", 0)

        c = DipCandidate(
            ticker=ticker,
            name=ticker_to_name(ticker),
            current_price=int(close),
            day_high=int(high),
            drop_pct=round(drop_pct, 2),
            rsi=round(rsi, 1),
            adx=round(adx, 1),
            bb_position=round(bb_pos, 2),
            volume_ratio=round(vol_ratio, 2),
            above_sma20=close >= sma20 if sma20 > 0 else True,
            foreign_net=int(foreign),
            inst_net=int(inst),
        )

        c.score = score_dip(c, cfg)
        c.suggested_price = int(close * 0.997)
        c.stop_price = int(c.suggested_price * (1 - cfg.stop_loss_pct / 100))
        c.target_price = int(c.suggested_price * 1.03)

        candidates.append(c)

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:cfg.max_candidates * 2]


# ──────────────────────────────────────────────────────────
# parquet 기술지표 로드
# ──────────────────────────────────────────────────────────

def _load_parquet_indicators() -> dict[str, dict]:
    """각 종목의 최신 기술지표를 parquet에서 추출"""
    result = {}
    needed_cols = [
        "open", "high", "low", "close", "volume",
        "rsi_14", "adx_14", "bb_position",
    ]
    # 있으면 가져오는 추가 컬럼
    optional_cols = ["sma_20", "외국인합계", "기관합계"]

    for f in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            # 필요 컬럼 체크
            available = [c for c in needed_cols if c in df.columns]
            if len(available) < len(needed_cols):
                continue
            if len(df) < 60:
                continue

            df.index = pd.to_datetime(df.index)
            latest = df.iloc[-1]

            # SMA20: 컬럼이 있으면 사용, 없으면 계산
            if "sma_20" in df.columns:
                sma20 = float(latest["sma_20"]) if pd.notna(latest.get("sma_20")) else 0
            else:
                sma20 = float(df["close"].tail(20).mean())

            # 20일 평균 거래량
            vol_avg20 = float(df["volume"].tail(20).mean())

            ind = {
                "close": float(latest["close"]),
                "high": float(latest["high"]),
                "volume": float(latest["volume"]),
                "rsi_14": float(latest["rsi_14"]) if pd.notna(latest.get("rsi_14")) else 50,
                "adx_14": float(latest["adx_14"]) if pd.notna(latest.get("adx_14")) else 0,
                "bb_position": float(latest["bb_position"]) if pd.notna(latest.get("bb_position")) else 0.5,
                "sma20": sma20,
                "vol_avg20": vol_avg20,
            }

            # 수급 (있으면)
            if "외국인합계" in df.columns:
                ind["foreign_net"] = float(latest.get("외국인합계", 0) or 0)
            if "기관합계" in df.columns:
                ind["inst_net"] = float(latest.get("기관합계", 0) or 0)

            result[f.stem] = ind

        except Exception:
            continue

    return result


# ──────────────────────────────────────────────────────────
# 텔레그램 알림 포맷
# ──────────────────────────────────────────────────────────

def format_telegram_message(
    candidates: list[DipCandidate],
    risk: dict,
    is_friday: bool,
    is_dry_run: bool = False,
) -> str:
    """텔레그램 메시지 포맷"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    day_label = "금요일" if is_friday else datetime.now().strftime("%A")

    header = "🔻 [투매 역매수 스캔" + (" - 드라이런]" if is_dry_run else "]")
    header += f"\n{now} ({day_label})"

    # 주말 리스크
    grade = risk.get("grade", "NEUTRAL")
    vix = risk.get("vix_level", 0)
    allow = risk.get("allow_count", 3)
    nw_score = risk.get("nightwatch_score", 0)
    veto = risk.get("bond_vigilante_veto", False)

    risk_line = f"\n주말 리스크: {grade}"
    if vix > 0:
        risk_line += f" | VIX {vix:.1f}"
    if veto:
        risk_line += " | ⛔ 비토 발동"
    risk_line += f"\n매수 허용: {allow}종목"

    rules = risk.get("special_rules", [])
    if rules:
        risk_line += f"\n⚠️ {', '.join(rules)}"

    # 차단 체크
    if allow <= 0:
        return (
            header + "\n━━━━━━━━━━━━━━━━━━━━"
            + risk_line
            + "\n\n🚫 주말 리스크 높음 — 매수 차단"
            + "\n━━━━━━━━━━━━━━━━━━━━"
        )

    if not candidates:
        return (
            header + "\n━━━━━━━━━━━━━━━━━━━━"
            + risk_line
            + "\n\n📭 투매 감지 종목 없음"
            + "\n━━━━━━━━━━━━━━━━━━━━"
        )

    # 허용 수만큼 자르기
    show = candidates[:allow]

    lines = [header, "━━━━━━━━━━━━━━━━━━━━", risk_line, ""]

    for i, c in enumerate(show, 1):
        supply = c.supply_signal
        warn = " ⚠️" if "쌍매도" in supply else ""

        lines.append(f"{i}️⃣ {c.name}({c.ticker}) {c.current_price:,}원")
        lines.append(f"   고가대비 {c.drop_pct:.1f}% | 점수 {c.score:.0f}/100")
        lines.append(f"   RSI={c.rsi:.0f} ADX={c.adx:.0f} BB={c.bb_position:.0%} Vol={c.volume_ratio:.1f}x")
        lines.append(f"   {supply}{warn}")
        lines.append(f"   매수 {c.suggested_price:,}원 → 손절 {c.stop_price:,}원 / 목표 {c.target_price:,}원")

        # 점수 상세
        sd = c.score_detail
        lines.append(f"   [하락{sd.get('drop',0):.0f} 거래량{sd.get('volume',0):.0f} "
                      f"기술{sd.get('tech',0):.0f} 수급{sd.get('supply',0):.0f}]")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("💡 백테스트 기준: 승률 ~49%, 손익비 1.37")
    lines.append("   -1% 손절 시 기대값 +1.54%")

    if is_friday:
        lines.append("   금요일 기대값 +0.67% (최고)")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────────────────

def save_results(candidates: list[DipCandidate], risk: dict):
    """스캔 결과 JSON 저장"""
    data = {
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weekend_risk": risk,
        "candidates": [asdict(c) for c in candidates],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장: {OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="금요일 투매 역매수 스캐너")
    parser.add_argument("--scan", action="store_true", help="실시간 스캔 (KIS API)")
    parser.add_argument("--dry-run", action="store_true", help="parquet 기반 시뮬레이션")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 발송")
    parser.add_argument("--min-rsi", type=float, default=55.0, help="최소 RSI")
    parser.add_argument("--min-drop", type=float, default=2.0, help="최소 하락률 (%%)")
    args = parser.parse_args()

    cfg = DipConfig(min_rsi=args.min_rsi, min_drop_pct=args.min_drop)

    # 주말 리스크 로드
    risk = load_weekend_risk()
    is_friday = datetime.now().weekday() == 4

    logger.info(f"주말 리스크: {risk['grade']} | VIX {risk['vix_level']:.1f} | "
                f"허용 {risk['allow_count']}종목")

    if args.scan:
        logger.info("=== 실시간 투매 스캔 시작 ===")
        candidates = scan_live(cfg)
    elif args.dry_run:
        logger.info("=== 드라이런 (parquet 시뮬레이션) ===")
        candidates = scan_dry_run(cfg)
    else:
        parser.print_help()
        return

    logger.info(f"감지 종목: {len(candidates)}개")

    # 결과 출력
    for i, c in enumerate(candidates, 1):
        logger.info(f"  #{i} {c.name}({c.ticker}) {c.current_price:,}원 "
                     f"하락{c.drop_pct:.1f}% 점수{c.score:.0f}")

    # 저장
    save_results(candidates, risk)

    # 텔레그램
    if args.telegram:
        msg = format_telegram_message(candidates, risk, is_friday, is_dry_run=args.dry_run)
        ok = send_message(msg)
        if ok:
            logger.info("텔레그램 발송 완료")
        else:
            logger.error("텔레그램 발송 실패")
    else:
        # 콘솔 출력
        msg = format_telegram_message(candidates, risk, is_friday, is_dry_run=args.dry_run)
        print("\n" + msg)


if __name__ == "__main__":
    main()
