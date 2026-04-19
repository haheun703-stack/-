#!/usr/bin/env python
"""피보나치 스캐너 — 전체 눌림목 + 대형주 TOP 30 + 섹터 로테이션

단타봇 이관: dashboard_swing.(fib_stocks, fib_leaders, sector_rotation)
→ 퀀트봇: quant_fib_scanner 테이블

탭 구성:
  1) 전체 피보나치 눌림목: 시총 1조+ & 15%+ 하락, fib_zone별 그룹핑 (최대 50)
  2) 대형주 피보나치: 시총 TOP 30, fib_status 색상 매핑
  3) 섹터 로테이션 맵: 17섹터 종합점수 → 선도/추격/대기/후발

스케줄: 매일 BAT-D (장후)
출력: data/fib_scanner.json → quant_fib_scanner 테이블

Usage:
    python -u -X utf8 scripts/scan_fibonacci.py
    python -u -X utf8 scripts/scan_fibonacci.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

import numpy as np

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STOCK_DAILY_DIR = PROJECT_ROOT / "stock_data_daily"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
OUTPUT_PATH = DATA_DIR / "fib_scanner.json"

# ─── 매수 적기 판정 기준 (백테스트 S7 근거) ───
RSI_ENTRY_THRESHOLD = 40    # RSI < 40 = 과매도 구간
SUPPLY_ENTRY_MIN = 5.0      # 외인|기관 순매수 최소 5억원


# ═══════════════════════════════════════════════════
# 피보나치 계산 로직
# ═══════════════════════════════════════════════════

_ZONE_ORDER = {"DEEP": 0, "MID": 1, "MILD": 2, "SHALLOW": 3, "NEAR_HIGH": 4}


def _classify_fib_zone(drop_pct: float) -> tuple[str, str]:
    """하락률(%) → (zone코드, zone라벨)."""
    dd = abs(drop_pct)
    if dd >= 50:
        return "DEEP", "50%+ 하락 (바닥 매수 구간)"
    elif dd >= 40:
        return "MID", "40~50% 하락 (중간 눌림)"
    elif dd >= 30:
        return "MILD", "30~40% 하락 (1차 눌림)"
    elif dd >= 15:
        return "SHALLOW", "15~30% 하락 (얕은 조정)"
    else:
        return "NEAR_HIGH", "고점 근접"


def _fib_status(price: float, fib_382: float, fib_500: float, fib_618: float) -> str:
    """현재가 대비 피보나치 레벨 위치 판정."""
    if price <= fib_382:
        return "38.2% 아래 (깊은 하락)"
    elif price <= fib_500:
        return "38.2%~50% 사이"
    elif price <= fib_618:
        return "50%~61.8% 사이"
    else:
        return "61.8% 위 (회복 중)"


def calc_52w(ticker: str) -> dict | None:
    """parquet에서 52주 고가/저가/현재가 계산."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        return None
    try:
        df = pd.read_parquet(pq)
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            return None
        df = df[list(required)]
        if len(df) < 60:
            return None
        tail = df.tail(252)
        close = float(tail["close"].iloc[-1])
        high_252 = float(tail["high"].max())
        low_252 = float(tail["low"].min())
        if high_252 <= 0 or close <= 0:
            return None
        return {
            "close": close,
            "high_252": high_252,
            "low_252": low_252,
            "drop_pct": round((close / high_252 - 1) * 100, 1),
        }
    except Exception:
        return None


def _load_daily_technicals(ticker: str) -> dict:
    """stock_data_daily CSV에서 RSI, 수급 등 기술적 데이터 로드.

    Returns: {rsi, foreign_net, inst_net, vol_ratio} or empty dict on failure.
    """
    if not STOCK_DAILY_DIR.exists():
        return {}
    # 파일명: {종목명}_{종목코드}.csv — ticker로 glob
    matches = list(STOCK_DAILY_DIR.glob(f"*_{ticker}.csv"))
    if not matches:
        return {}
    try:
        df = pd.read_csv(matches[0], encoding="utf-8-sig")
        if len(df) < 20:
            return {}
        # 최근 행
        last = df.iloc[-1]
        rsi = float(last.get("RSI", np.nan))
        foreign_net = float(last.get("Foreign_Net", 0))
        inst_net = float(last.get("Inst_Net", 0))
        volume = float(last.get("Volume", 0))

        # 거래량 비율 (최근 / MA20)
        vol_ma20 = df["Volume"].tail(20).mean()
        vol_ratio = round(volume / vol_ma20, 2) if vol_ma20 > 0 else 1.0

        return {
            "rsi": round(rsi, 1) if not np.isnan(rsi) else None,
            "foreign_net": round(foreign_net, 1),
            "inst_net": round(inst_net, 1),
            "vol_ratio": vol_ratio,
        }
    except Exception:
        return {}


def _classify_entry_grade(rsi: float | None, foreign_net: float,
                          inst_net: float) -> str:
    """매수 적기 등급 판정.

    근거: 백테스트 S7 (Fib + 수급 + RSI < 40)
      → D+5 +2.97%, WR 61.9%, PF 3.00 = STRONG_ALPHA
    """
    rsi_ok = rsi is not None and rsi < RSI_ENTRY_THRESHOLD
    supply_ok = foreign_net >= SUPPLY_ENTRY_MIN or inst_net >= SUPPLY_ENTRY_MIN

    if rsi_ok and supply_ok:
        return "적기"       # RSI 과매도 + 수급 유입 = STRONG_ALPHA
    elif rsi_ok:
        return "관심"       # RSI 과매도만 (수급 미확인)
    elif supply_ok:
        return "수급 유입"  # 수급만 (RSI 미과매도)
    else:
        return "대기"       # 조건 미충족


def calc_fib_levels(high: float, low: float) -> dict:
    """52주 고/저 기반 피보나치 레벨 계산.

    표준 피보나치: fib_X = low + (high - low) × X
    """
    rng = high - low
    return {
        "fib_382": round(low + rng * 0.382),
        "fib_500": round(low + rng * 0.500),
        "fib_618": round(low + rng * 0.618),
    }


def build_fib_stock(ticker: str, name: str, sector: str, cap_억: float,
                    w52: dict, pykrx: dict) -> dict:
    """개별 종목 피보나치 데이터 빌드 (RSI + 수급 + entry_grade 포함)."""
    close = w52["close"]
    fib = calc_fib_levels(w52["high_252"], w52["low_252"])
    zone, zone_label = _classify_fib_zone(w52["drop_pct"])
    status = _fib_status(close, fib["fib_382"], fib["fib_500"], fib["fib_618"])

    # 목표가: 52주 고/저 범위의 78.6% 회복
    target = round(w52["low_252"] + (w52["high_252"] - w52["low_252"]) * 0.786)
    upside = round((target / close - 1) * 100, 1) if close > 0 else 0

    # 현재가의 52주 범위 내 위치 (0~100%)
    rng = w52["high_252"] - w52["low_252"]
    position_pct = round((close - w52["low_252"]) / rng * 100, 1) if rng > 0 else 50

    # ── RSI + 수급 로드 (백테스트 S7 근거) ──
    tech = _load_daily_technicals(ticker)
    rsi = tech.get("rsi")
    foreign_net = tech.get("foreign_net", 0)
    inst_net = tech.get("inst_net", 0)
    vol_ratio = tech.get("vol_ratio", 1.0)
    entry_grade = _classify_entry_grade(rsi, foreign_net, inst_net)

    return {
        "code": ticker,
        "name": name,
        "sector": sector or "",
        "cap": round(cap_억),
        "price": int(close),
        "w52h": int(w52["high_252"]),
        "w52l": int(w52["low_252"]),
        "drop": w52["drop_pct"],
        "fib_382": fib["fib_382"],
        "fib_500": fib["fib_500"],
        "fib_618": fib["fib_618"],
        "fib_zone": zone,
        "fib_zone_label": zone_label,
        "fib_status": status,
        "position_pct": position_pct,
        "target": target,
        "upside": upside,
        "per": round(pykrx.get("PER", 0), 1),
        "pbr": round(pykrx.get("PBR", 0), 2),
        # ── 신규: RSI + 수급 + 매수 적기 ──
        "rsi": rsi,
        "foreign_net": foreign_net,
        "inst_net": inst_net,
        "vol_ratio": vol_ratio,
        "entry_grade": entry_grade,
    }


# ═══════════════════════════════════════════════════
# 스캔 로직
# ═══════════════════════════════════════════════════

def load_universe() -> pd.DataFrame:
    """universe.csv 로드."""
    if not UNIVERSE_PATH.exists():
        logger.error("universe.csv 없음: %s", UNIVERSE_PATH)
        return pd.DataFrame()
    df = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)
    df["market_cap_억"] = df["market_cap"] / 1e8
    logger.info("유니버스 로드: %d종목", len(df))
    return df


def load_pykrx() -> dict[str, dict]:
    """pykrx PER/PBR 로드 (실패 시 빈 dict)."""
    try:
        from pykrx import stock as pykrx_stock
        from datetime import timedelta

        for days_back in range(0, 8):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                df = pykrx_stock.get_market_fundamental_by_ticker(date_str, market="ALL")
                if not df.empty and "PER" in df.columns:
                    result = {}
                    for ticker, row in df.iterrows():
                        result[ticker] = {
                            "PER": float(row.get("PER", 0)),
                            "PBR": float(row.get("PBR", 0)),
                        }
                    logger.info("pykrx PER/PBR 로드: %s (%d종목)", date_str, len(result))
                    return result
            except Exception:
                continue
    except ImportError:
        pass
    logger.info("pykrx 로드 실패 — PER/PBR 없이 진행")
    return {}


def scan_fib_stocks(universe: pd.DataFrame, pykrx_data: dict,
                    min_cap_억: float = 10000, max_stocks: int = 50) -> list[dict]:
    """전체 피보나치 눌림목 — 시총 1조+ & 15%+ 하락."""
    big = universe[universe["market_cap_억"] >= min_cap_억]
    results = []

    for _, row in big.iterrows():
        ticker = row["ticker"]
        w52 = calc_52w(ticker)
        if w52 is None or abs(w52["drop_pct"]) < 15:
            continue

        stock = build_fib_stock(
            ticker, row.get("name", ticker), row.get("sector", ""),
            row["market_cap_억"], w52, pykrx_data.get(ticker, {}),
        )
        results.append(stock)

    # 정렬: zone 우선 (DEEP→SHALLOW), 같은 zone 내 하락률 큰 순
    results.sort(key=lambda x: (_ZONE_ORDER.get(x["fib_zone"], 9), x["drop"]))
    logger.info("피보나치 눌림목: %d종목 (시총 %.0f억+, 15%%+ 하락)", len(results), min_cap_억)
    return results[:max_stocks]


def scan_fib_leaders(universe: pd.DataFrame, pykrx_data: dict,
                     top_n: int = 30) -> list[dict]:
    """대형주 피보나치 — 시총 TOP 30."""
    top = universe.nlargest(top_n, "market_cap_억")
    results = []

    for _, row in top.iterrows():
        ticker = row["ticker"]
        w52 = calc_52w(ticker)
        if w52 is None:
            continue

        stock = build_fib_stock(
            ticker, row.get("name", ticker), row.get("sector", ""),
            row["market_cap_억"], w52, pykrx_data.get(ticker, {}),
        )
        results.append(stock)

    # 시총 순서 유지 (nlargest 결과 순서)
    logger.info("대형주 피보나치: %d종목", len(results))
    return results


# ═══════════════════════════════════════════════════
# 섹터 로테이션 (기존 데이터 활용)
# ═══════════════════════════════════════════════════

def load_sector_rotation() -> list[dict]:
    """sector_composite.json → 섹터 로테이션 4분면 매핑."""
    path = DATA_DIR / "sector_rotation" / "sector_composite.json"
    if not path.exists():
        logger.warning("sector_composite.json 없음")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        quadrant_map = {
            "STRONG_ROTATION": "선도",
            "MODERATE_ROTATION": "추격",
            "NEUTRAL": "대기",
            "WEAK_ROTATION": "후발",
        }

        result = []
        for s in data.get("sectors", []):
            regime = s.get("regime", "NEUTRAL")
            result.append({
                "sector": s.get("sector", ""),
                "etf_code": s.get("etf_code", ""),
                "score": round(s.get("composite_score", 0), 1),
                "quadrant": quadrant_map.get(regime, "대기"),
                "regime": regime,
                "momentum": round(s.get("momentum_score", 0), 1),
                "inst_flow": round(s.get("institutional_score", 0), 1),
                "rel_strength": round(s.get("relative_strength_score", 0), 1),
                "technical": round(s.get("technical_score", 0), 1),
                "ret_5d": round(s.get("ret_5", 0), 1),
                "ret_20d": round(s.get("ret_20", 0), 1),
                "inst_5d_억": round(s.get("inst_5d_억", 0), 1),
                "foreign_5d_억": round(s.get("foreign_5d_억", 0), 1),
                "dual_buy": s.get("inst_5d_억", 0) > 0 and s.get("foreign_5d_억", 0) > 0,
            })

        result.sort(key=lambda x: x["score"], reverse=True)
        logger.info("섹터 로테이션: %d섹터", len(result))
        return result
    except Exception as e:
        logger.warning("섹터 로테이션 로드 실패: %s", e)
        return []


# ═══════════════════════════════════════════════════
# 업로드 + 출력
# ═══════════════════════════════════════════════════

def _extract_entry_picks(fib_stocks: list, fib_leaders: list) -> list[dict]:
    """매수 적기("적기") 종목만 추출 — 퀀트시스템 메인 노출용.

    근거: 백테스트 S7 (Fib + RSI<40 + 수급) = D+5 +2.97%, WR 61.9%, PF 3.00
    """
    entry_picks = []
    seen = set()
    for stock in fib_stocks + fib_leaders:
        if stock.get("entry_grade") == "적기" and stock["code"] not in seen:
            entry_picks.append({
                "code": stock["code"],
                "name": stock["name"],
                "price": stock["price"],
                "drop": stock["drop"],
                "rsi": stock.get("rsi"),
                "foreign_net": stock.get("foreign_net", 0),
                "inst_net": stock.get("inst_net", 0),
                "fib_zone": stock["fib_zone"],
                "target": stock["target"],
                "upside": stock["upside"],
                "sector": stock.get("sector", ""),
                "cap": stock.get("cap", 0),
            })
            seen.add(stock["code"])
    # 하락률 큰 순 (깊이 눌린 순)
    entry_picks.sort(key=lambda x: x["drop"])
    return entry_picks


def upload_fib_scanner(fib_stocks: list, fib_leaders: list,
                       sector_rotation: list, date_str: str = "") -> bool:
    """quant_fib_scanner 테이블에 업로드."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    entry_picks = _extract_entry_picks(fib_stocks, fib_leaders)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "fib_stocks": fib_stocks,
        "fib_leaders": fib_leaders,
        "sector_rotation": sector_rotation,
        "entry_picks": entry_picks,
        "summary": {
            "fib_stocks_count": len(fib_stocks),
            "fib_leaders_count": len(fib_leaders),
            "sector_count": len(sector_rotation),
            "entry_picks_count": len(entry_picks),
            "zones": {
                zone: sum(1 for s in fib_stocks if s["fib_zone"] == zone)
                for zone in ["DEEP", "MID", "MILD", "SHALLOW"]
            },
        },
    }

    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        row = {"date": date_str, "data": payload}
        uploader.client.table("quant_fib_scanner").upsert(
            row, on_conflict="date"
        ).execute()
        logger.info("[피보나치] 업로드 완료: %s (눌림목 %d, 대형주 %d, 섹터 %d)",
                    date_str, len(fib_stocks), len(fib_leaders), len(sector_rotation))
        return True
    except Exception as e:
        logger.error("[피보나치] 업로드 오류: %s", e)
        return False


def print_report(fib_stocks: list, fib_leaders: list, sector_rotation: list):
    """콘솔 출력."""
    print(f"\n{'='*65}")
    print(f"  피보나치 스캐너 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")

    # ── 매수 적기 종목 (최상단 노출) ──
    entry_picks = _extract_entry_picks(fib_stocks, fib_leaders)
    if entry_picks:
        print(f"\n  ★ 매수 적기 ({len(entry_picks)}종목) — RSI<40 + 수급 유입")
        print(f"  {'─'*60}")
        for s in entry_picks:
            rsi_str = f"RSI {s['rsi']:.0f}" if s['rsi'] is not None else "RSI ?"
            supply_parts = []
            if s['foreign_net'] >= SUPPLY_ENTRY_MIN:
                supply_parts.append(f"외인 +{s['foreign_net']:.0f}억")
            if s['inst_net'] >= SUPPLY_ENTRY_MIN:
                supply_parts.append(f"기관 +{s['inst_net']:.0f}억")
            supply_str = " / ".join(supply_parts) if supply_parts else "수급 ?"
            print(
                f"    {s['name']:12s} ({s['code']}) {s['fib_zone']:6s} "
                f"{rsi_str} | {supply_str} | 목표 {s['upside']:+.1f}%"
            )
    else:
        print(f"\n  [매수 적기 종목 없음] — RSI<40 + 수급 조건 미충족")

    # 눌림목 zone별 요약
    zones = {}
    for s in fib_stocks:
        z = s["fib_zone"]
        zones[z] = zones.get(z, 0) + 1
    zone_str = " | ".join(
        f"{z}: {c}" for z, c in sorted(zones.items(), key=lambda x: _ZONE_ORDER.get(x[0], 9))
    )
    print(f"\n  [눌림목] {len(fib_stocks)}종목 — {zone_str}")
    for s in fib_stocks[:10]:
        grade_mark = "★" if s.get("entry_grade") == "적기" else " "
        rsi_str = f"RSI {s['rsi']:.0f}" if s.get('rsi') is not None else ""
        print(
            f"   {grade_mark} {s['fib_zone']:8s} {s['name']:12s} ({s['code']}) "
            f"하락 {s['drop']:+.1f}% | {s['price']:,} → {s['target']:,} "
            f"({s['upside']:+.1f}%) {rsi_str}"
        )
    if len(fib_stocks) > 10:
        print(f"    ... 외 {len(fib_stocks) - 10}종목")

    # 대형주 요약
    print(f"\n  [대형주 TOP 30] {len(fib_leaders)}종목")
    for s in fib_leaders[:10]:
        cap_조 = s['cap'] / 10000
        grade_mark = "★" if s.get("entry_grade") == "적기" else " "
        rsi_str = f"RSI {s['rsi']:.0f}" if s.get('rsi') is not None else ""
        print(
            f"   {grade_mark} {s['name']:12s} 시총 {cap_조:,.1f}조 | "
            f"하락 {s['drop']:+.1f}% | 위치 {s['position_pct']:.0f}% | "
            f"{s['entry_grade']:4s} {rsi_str}"
        )
    if len(fib_leaders) > 10:
        print(f"    ... 외 {len(fib_leaders) - 10}종목")

    # 섹터 로테이션 요약
    if sector_rotation:
        for q in ["선도", "추격", "대기", "후발"]:
            secs = [s["sector"] for s in sector_rotation if s["quadrant"] == q]
            if secs:
                print(f"\n  [섹터 {q}] {', '.join(secs)}")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="피보나치 스캐너 — 눌림목 + 대형주 + 섹터")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 출력만")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n[피보나치] 스캔 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 유니버스 로드
    universe = load_universe()
    if universe.empty:
        print("[피보나치] 유니버스 없음")
        return

    # PER/PBR (선택)
    pykrx_data = load_pykrx()

    # 피보나치 눌림목 (시총 1조+, 15%+ 하락, 최대 50)
    fib_stocks = scan_fib_stocks(universe, pykrx_data)

    # 대형주 TOP 30
    fib_leaders = scan_fib_leaders(universe, pykrx_data)

    # 섹터 로테이션
    sector_rotation = load_sector_rotation()

    # 콘솔 출력
    print_report(fib_stocks, fib_leaders, sector_rotation)

    # JSON 저장
    date_str = datetime.now().strftime("%Y-%m-%d")
    entry_picks = _extract_entry_picks(fib_stocks, fib_leaders)
    report = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "fib_stocks": fib_stocks,
        "fib_leaders": fib_leaders,
        "sector_rotation": sector_rotation,
        "entry_picks": entry_picks,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 업로드
    if not args.dry_run:
        upload_fib_scanner(fib_stocks, fib_leaders, sector_rotation, date_str)

    print(f"\n[피보나치] 완료 — 눌림목 {len(fib_stocks)} + 대형주 {len(fib_leaders)} + 섹터 {len(sector_rotation)}")


if __name__ == "__main__":
    main()
