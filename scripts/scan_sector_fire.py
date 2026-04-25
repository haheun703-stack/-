#!/usr/bin/env python3
"""섹터 발화 스캔 — FIRE 스코어 + 종목 매수 스코어 + ETF 추천

FIRE = Flow(25) + Inflection(20) + Rhythm(15) + Energy(25) - 과열감산
한국장 핵심: 섹터별 돈의 흐름 → 종목별 수급 반전 → 매수 후보
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from scripts.scan_supply_surge import load_supply_from_db, load_price_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "sector_fire_map.yaml"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
OUTPUT_DIR = PROJECT_ROOT / "data"


# ─────────────────────────────────────────────
# 1. 설정 로더
# ─────────────────────────────────────────────

def load_sector_config() -> dict:
    """config/sector_fire_map.yaml 로드."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sectors", {})


def load_supply_surge_json(date_compact: str) -> dict:
    """당일 supply_surge JSON → ticker→entry 맵."""
    p = OUTPUT_DIR / f"supply_surge_{date_compact}.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {c["ticker"]: c for c in data.get("buy_candidates", [])}


# ─────────────────────────────────────────────
# 2. 섹터 FIRE 스코어
# ─────────────────────────────────────────────

def calc_sector_fire(
    sector_name: str,
    tickers: list[str],
    supply_df: pd.DataFrame,
    dates_5: list[str],
    dates_10: list[str],
) -> dict:
    """섹터 FIRE 스코어 계산. 0~100."""
    result = {"sector": sector_name, "ticker_count": len(tickers)}

    # 섹터 종목 수급 합산 (5일/10일)
    sector_supply = supply_df[supply_df["ticker"].isin(tickers)]

    if sector_supply.empty:
        result.update({
            "fire_score": 0, "fire_grade": "D",
            "flow_score": 0, "inflection_score": 0,
            "rhythm_score": 0, "energy_score": 0, "overheat_penalty": 0,
            "fgn_5d": 0, "inst_5d": 0, "pension_5d": 0,
            "fgn_reversal": 0, "inst_reversal": 0,
            "ma20_avg_dev": 0, "rsi_avg": 50, "vol_ratio_avg": 1,
        })
        return result

    # 5일 수급
    s5 = sector_supply[sector_supply["date"].isin(dates_5)]
    fgn_5d = s5["fgn"].sum()
    inst_5d = s5["inst"].sum()
    pension_5d = s5["pension"].sum()

    # 수급 반전: 최근2일 vs 이전3일
    recent_2d = sector_supply[sector_supply["date"].isin(dates_5[:2])]
    prev_3d = sector_supply[sector_supply["date"].isin(dates_5[2:5])]
    fgn_r2 = recent_2d["fgn"].sum()
    fgn_p3 = prev_3d["fgn"].sum()
    inst_r2 = recent_2d["inst"].sum()
    inst_p3 = prev_3d["inst"].sum()
    fgn_reversal = fgn_r2 - fgn_p3
    inst_reversal = inst_r2 - inst_p3

    # 기술적 지표 (섹터 평균)
    ma20_devs = []
    rsis = []
    vol_ratios = []
    for t in tickers:
        pdf = load_price_data(t)
        if pdf is None or pdf.empty:
            continue
        last = pdf.iloc[-1]
        if not np.isnan(last.get("ma20_dev", np.nan)):
            ma20_devs.append(last["ma20_dev"])
        if not np.isnan(last.get("rsi", np.nan)):
            rsis.append(last["rsi"])
        if not np.isnan(last.get("vol_ratio", np.nan)):
            vol_ratios.append(last["vol_ratio"])

    ma20_avg = np.mean(ma20_devs) if ma20_devs else 0
    rsi_avg = np.mean(rsis) if rsis else 50
    vol_avg = np.mean(vol_ratios) if vol_ratios else 1

    # === FIRE 스코어 계산 ===

    # F: Flow (0~25) — 5일 외인+기관 순매수
    combined_flow = fgn_5d + inst_5d
    if combined_flow >= 500:
        flow_score = 25
    elif combined_flow >= 200:
        flow_score = 20
    elif combined_flow >= 50:
        flow_score = 15
    elif combined_flow > 0:
        flow_score = 8
    elif combined_flow > -50:
        flow_score = 3
    else:
        flow_score = 0

    # I: Inflection (0~20) — 수급 반전
    inflection_score = 0
    if fgn_reversal > 100:
        inflection_score += 12
    elif fgn_reversal > 30:
        inflection_score += 8
    elif fgn_reversal > 0:
        inflection_score += 3

    if inst_reversal > 100:
        inflection_score += 8
    elif inst_reversal > 30:
        inflection_score += 5
    elif inst_reversal > 0:
        inflection_score += 2
    inflection_score = min(inflection_score, 20)

    # R: Rhythm (0~15) — MA20 이격도
    abs_dev = abs(ma20_avg)
    if abs_dev <= 5:
        rhythm_score = 15  # 최적 눌림
    elif abs_dev <= 10:
        rhythm_score = 10  # 양호
    elif abs_dev <= 15:
        rhythm_score = 5   # 확장
    elif abs_dev <= 20:
        rhythm_score = 0   # 과열 영역
    else:
        rhythm_score = -5  # 극단 과열

    # E: Energy (0~25) — RSI + 거래량 + 수급 점증
    energy_score = 0
    if 40 <= rsi_avg <= 60:
        energy_score += 10
    elif 30 <= rsi_avg <= 70:
        energy_score += 5

    if vol_avg >= 2.0:
        energy_score += 5
    elif vol_avg >= 1.5:
        energy_score += 3

    # 수급 점증: 최근 3일 연속 외인+기관 양수인 종목 비율
    if len(dates_5) >= 3:
        streak_tickers = 0
        for t in tickers:
            t_data = sector_supply[sector_supply["ticker"] == t]
            recent_3 = t_data[t_data["date"].isin(dates_5[:3])]
            if len(recent_3) >= 3:
                combined = recent_3["fgn"] + recent_3["inst"]
                if (combined > 0).all():
                    streak_tickers += 1
        streak_ratio = streak_tickers / len(tickers) if tickers else 0
        if streak_ratio >= 0.5:
            energy_score += 10
        elif streak_ratio >= 0.3:
            energy_score += 5
    energy_score = min(energy_score, 25)

    # 과열 감산
    overheat = 0
    if ma20_avg > 20:
        overheat -= 10
    if rsi_avg > 75:
        overheat -= 5

    fire_score = max(0, min(100, flow_score + inflection_score + rhythm_score + energy_score + overheat))

    # FIRE 등급
    if fire_score >= 80:
        fire_grade = "S"
    elif fire_score >= 60:
        fire_grade = "A"
    elif fire_score >= 40:
        fire_grade = "B"
    elif fire_score >= 20:
        fire_grade = "C"
    else:
        fire_grade = "D"

    result.update({
        "fire_score": round(fire_score, 1),
        "fire_grade": fire_grade,
        "flow_score": round(flow_score, 1),
        "inflection_score": round(inflection_score, 1),
        "rhythm_score": round(rhythm_score, 1),
        "energy_score": round(energy_score, 1),
        "overheat_penalty": round(overheat, 1),
        "fgn_5d": round(float(fgn_5d), 1),
        "inst_5d": round(float(inst_5d), 1),
        "pension_5d": round(float(pension_5d), 1),
        "fgn_reversal": round(float(fgn_reversal), 1),
        "inst_reversal": round(float(inst_reversal), 1),
        "ma20_avg_dev": round(float(ma20_avg), 1),
        "rsi_avg": round(float(rsi_avg), 1),
        "vol_ratio_avg": round(float(vol_avg), 1),
    })
    return result


# ─────────────────────────────────────────────
# 3. 종목 매수 스코어
# ─────────────────────────────────────────────

def calc_stock_buy_score(
    ticker: str,
    supply_df: pd.DataFrame,
    dates_5: list[str],
    dates_10: list[str],
    surge_map: dict,
) -> dict | None:
    """종목 매수 스코어 (0~100). None이면 분석 불가."""
    pdf = load_price_data(ticker)
    if pdf is None or pdf.empty:
        return None

    last = pdf.iloc[-1]
    close = last["close"]
    if close <= 0:
        return None

    # 종목명
    ticker_supply = supply_df[supply_df["ticker"] == ticker]
    name = str(ticker_supply["name"].iloc[0]) if not ticker_supply.empty else ticker

    # 기술적 지표
    ma20_dev = last.get("ma20_dev", np.nan)
    rsi = last.get("rsi", np.nan)
    vol_ratio = last.get("vol_ratio", np.nan)
    ret0 = last.get("ret0", np.nan)

    # 수급 데이터
    t5 = ticker_supply[ticker_supply["date"].isin(dates_5)]
    t10 = ticker_supply[ticker_supply["date"].isin(dates_10)]

    fgn_5d = t5["fgn"].sum() if not t5.empty else 0
    inst_5d = t5["inst"].sum() if not t5.empty else 0
    pension_5d = t5["pension"].sum() if not t5.empty else 0

    # 반전 (최근2d vs 이전3d)
    recent_2d = ticker_supply[ticker_supply["date"].isin(dates_5[:2])]
    prev_3d = ticker_supply[ticker_supply["date"].isin(dates_5[2:5])]
    fgn_reversal = recent_2d["fgn"].sum() - prev_3d["fgn"].sum()
    inst_reversal = recent_2d["inst"].sum() - prev_3d["inst"].sum()

    # 연속 매수일
    fgn_streak = 0
    sorted_dates = sorted(dates_10)[::-1]  # 최신→과거
    for d in sorted_dates:
        d_data = ticker_supply[ticker_supply["date"] == d]
        if not d_data.empty and d_data["fgn"].values[0] > 0:
            fgn_streak += 1
        else:
            break

    # 당일 수급
    latest_date = dates_5[0] if dates_5 else ""
    latest_data = ticker_supply[ticker_supply["date"] == latest_date]
    fgn_1d = latest_data["fgn"].values[0] if not latest_data.empty else 0
    inst_1d = latest_data["inst"].values[0] if not latest_data.empty else 0
    retail_5d = t5["retail"].sum() if not t5.empty else 0

    # === 매수 스코어 ===
    score = 0
    reasons = []

    # MA20 눌림
    if not np.isnan(ma20_dev) and -5 <= ma20_dev <= 5:
        score += 15
        reasons.append(f"MA20눌림{ma20_dev:+.1f}%")

    # RSI 적정
    if not np.isnan(rsi) and 40 <= rsi <= 60:
        score += 10
        reasons.append(f"RSI적정{rsi:.0f}")

    # 외인 반전
    if fgn_reversal > 30:
        score += 15
        reasons.append(f"외인반전{fgn_reversal:+.0f}억")
    elif fgn_reversal > 0:
        score += 5
        reasons.append("외인소폭반전")

    # 기관 반전
    if inst_reversal > 30:
        score += 12
        reasons.append(f"기관반전{inst_reversal:+.0f}억")
    elif inst_reversal > 0:
        score += 4
        reasons.append("기관소폭반전")

    # 연기금
    if pension_5d > 20:
        score += 10
        reasons.append(f"연기금{pension_5d:+.0f}억")
    elif pension_5d > 0:
        score += 3

    # 당일 쌍끌이
    if fgn_1d > 0 and inst_1d > 0:
        score += 10
        reasons.append("당일쌍끌이")

    # 거래량
    if not np.isnan(vol_ratio) and vol_ratio >= 1.5:
        score += 5
        reasons.append(f"거래량{vol_ratio:.1f}x")

    # supply_surge 교차
    surge_entry = surge_map.get(ticker)
    surge_type = None
    if surge_entry:
        score += 8
        surge_type = surge_entry.get("type", "")
        reasons.append(f"수급급변{surge_type}")

    # 외인 연속매수
    if fgn_streak >= 3:
        score += 8
        reasons.append(f"외인{fgn_streak}연속")

    # 과열 감산
    if not np.isnan(ma20_dev) and ma20_dev > 15:
        score -= 10
        reasons.append(f"!!과열MA20{ma20_dev:+.0f}%")
    if not np.isnan(rsi) and rsi > 70:
        score -= 5
        reasons.append(f"!!RSI과매수{rsi:.0f}")
    if retail_5d > 100:
        score -= 5
        reasons.append(f"!!개인과다{retail_5d:+.0f}억")

    score = max(0, min(100, score))

    # 등급
    if score >= 50:
        buy_grade = "STRONG"
    elif score >= 35:
        buy_grade = "BUY"
    elif score >= 20:
        buy_grade = "WATCH"
    else:
        buy_grade = "SKIP"

    return {
        "ticker": ticker,
        "name": name[:10],
        "close": int(close),
        "chg_1d": round(float(ret0), 1) if not np.isnan(ret0) else 0.0,
        "buy_score": round(score, 1),
        "buy_grade": buy_grade,
        "ma20_dev": round(float(ma20_dev), 1) if not np.isnan(ma20_dev) else 0.0,
        "rsi": round(float(rsi), 1) if not np.isnan(rsi) else 50.0,
        "vol_ratio": round(float(vol_ratio), 1) if not np.isnan(vol_ratio) else 1.0,
        "fgn_5d": round(float(fgn_5d), 1),
        "inst_5d": round(float(inst_5d), 1),
        "pension_5d": round(float(pension_5d), 1),
        "fgn_reversal": round(float(fgn_reversal), 1),
        "inst_reversal": round(float(inst_reversal), 1),
        "fgn_streak": int(fgn_streak),
        "surge_type": surge_type,
        "buy_reasons": ", ".join(reasons),
    }


# ─────────────────────────────────────────────
# 4. ETF 추천
# ─────────────────────────────────────────────

def recommend_etf(sector_cfg: dict, fire_grade: str) -> dict:
    """FIRE 등급 기반 ETF 추천."""
    etf = sector_cfg.get("etf")
    leverage = sector_cfg.get("leverage_etf")
    rec = {"etf_code": None, "etf_name": None,
           "leverage_etf_code": None, "leverage_etf_name": None,
           "etf_recommend": ""}

    if etf:
        rec["etf_code"] = etf["code"]
        rec["etf_name"] = etf["name"]

    if leverage:
        rec["leverage_etf_code"] = leverage["code"]
        rec["leverage_etf_name"] = leverage["name"]

    if fire_grade == "S" and leverage:
        rec["etf_recommend"] = f"레버리지 강력 추천: {leverage['name']}"
    elif fire_grade == "S" and etf:
        rec["etf_recommend"] = f"ETF 강력 추천: {etf['name']}"
    elif fire_grade == "A" and etf:
        rec["etf_recommend"] = f"ETF 적극 매수: {etf['name']}"
    elif fire_grade == "B" and etf:
        rec["etf_recommend"] = f"ETF 관심: {etf['name']}"
    else:
        rec["etf_recommend"] = ""

    return rec


# ─────────────────────────────────────────────
# 5. 메인 스캔
# ─────────────────────────────────────────────

def scan_sector_fire(
    lookback: int = 10,
) -> tuple[list[dict], list[dict]]:
    """섹터 FIRE + 종목 매수 스캔. (sectors[], picks[]) 반환."""
    sector_cfg = load_sector_config()
    if not sector_cfg:
        logger.error("섹터 설정 로드 실패: %s", CONFIG_PATH)
        return [], []

    supply_df = load_supply_from_db(lookback_days=lookback)
    if supply_df.empty:
        logger.warning("수급 데이터 없음")
        return [], []

    # 거래일 리스트
    all_dates = sorted(supply_df["date"].unique(), reverse=True)
    dates_5 = all_dates[:5]
    dates_10 = all_dates[:10]
    latest_date = all_dates[0]
    date_compact = latest_date.replace("-", "")

    logger.info("최신 거래일: %s / 5일: %s~%s", latest_date, dates_5[-1], dates_5[0])

    # supply_surge 교차 데이터
    surge_map = load_supply_surge_json(date_compact)
    logger.info("supply_surge 교차: %d종목", len(surge_map))

    sectors_result = []
    picks_result = []

    for sector_name, cfg in sector_cfg.items():
        tickers = cfg.get("tickers", [])

        # 섹터 FIRE 스코어
        fire = calc_sector_fire(sector_name, tickers, supply_df, dates_5, dates_10)

        # ETF 추천
        etf_rec = recommend_etf(cfg, fire["fire_grade"])
        fire.update(etf_rec)

        sectors_result.append(fire)

        # 종목 매수 스코어 (FIRE C등급 이상만)
        if fire["fire_grade"] in ("S", "A", "B", "C"):
            for t in tickers:
                stock = calc_stock_buy_score(t, supply_df, dates_5, dates_10, surge_map)
                if stock and stock["buy_score"] >= 20:
                    stock["sector"] = sector_name
                    picks_result.append(stock)

    # 정렬
    sectors_result.sort(key=lambda x: x["fire_score"], reverse=True)
    picks_result.sort(key=lambda x: x["buy_score"], reverse=True)

    logger.info("섹터: %d개 / 매수 후보: %d종목", len(sectors_result), len(picks_result))
    return sectors_result, picks_result


# ─────────────────────────────────────────────
# 6. 리포트 + 저장
# ─────────────────────────────────────────────

def print_report(sectors: list[dict], picks: list[dict]):
    """콘솔 리포트."""
    today = datetime.now().strftime("%Y-%m-%d")
    SEP = "=" * 110

    print(f"\n{SEP}")
    print(f"  섹터 발화 스캔 (FIRE) — {today}")
    print(f"  철학: 섹터 돈의 흐름 → 종목 수급 반전 → 매수 후보")
    print(SEP)

    # ── 섹터 FIRE 순위 ──
    print(f"\n  [섹터 FIRE 순위] {len(sectors)}개")
    print(f"  {'섹터':>10} {'FIRE':>5} {'등급':>3} {'F흐름':>5} {'I반전':>5} "
          f"{'R리듬':>5} {'E에너':>5} {'감산':>4} {'외5d':>7} {'기5d':>7} "
          f"{'연5d':>6} {'MA20':>6} {'RSI':>4} {'ETF추천':>25}")
    print(f"  {'─' * 108}")

    for s in sectors:
        etf_rec = s.get("etf_recommend", "")[:25]
        print(f"  {s['sector']:>10} {s['fire_score']:>5.0f} {s['fire_grade']:>3} "
              f"{s['flow_score']:>5.0f} {s['inflection_score']:>5.0f} "
              f"{s['rhythm_score']:>5.0f} {s['energy_score']:>5.0f} "
              f"{s['overheat_penalty']:>+4.0f} "
              f"{s['fgn_5d']:>+6.0f} {s['inst_5d']:>+6.0f} "
              f"{s['pension_5d']:>+5.0f} "
              f"{s['ma20_avg_dev']:>+5.1f}% {s['rsi_avg']:>4.0f} "
              f"{etf_rec:>25}")

    # ── 종목 매수 후보 ──
    print(f"\n  [종목 매수 후보] {len(picks)}종목 (BUY_SCORE >= 20)")
    if picks:
        print(f"  {'섹터':>10} {'종목':>10} {'종가':>9} {'등락':>6} "
              f"{'점수':>4} {'등급':>6} {'MA20':>6} {'RSI':>4} "
              f"{'외5d':>6} {'기5d':>6} {'연5d':>5} {'외반전':>6} {'기반전':>6} {'근거':>30}")
        print(f"  {'─' * 140}")

        for p in picks:
            reasons = p.get("buy_reasons", "")[:30]
            grade_str = p["buy_grade"]
            if grade_str == "STRONG":
                grade_str = ">>STRG"
            elif grade_str == "BUY":
                grade_str = ">BUY"

            print(f"  {p.get('sector',''):>10} {p['name']:>10} "
                  f"{p['close']:>9,} {p['chg_1d']:>+5.1f}% "
                  f"{p['buy_score']:>4.0f} {grade_str:>6} "
                  f"{p['ma20_dev']:>+5.1f}% {p['rsi']:>4.0f} "
                  f"{p['fgn_5d']:>+5.0f} {p['inst_5d']:>+5.0f} "
                  f"{p['pension_5d']:>+4.0f} "
                  f"{p['fgn_reversal']:>+5.0f} {p['inst_reversal']:>+5.0f} "
                  f"{reasons:>30}")

    # ── FIRE 등급 설명 ──
    print(f"\n  [FIRE 등급]")
    print(f"    S(80+) 최강 발화 — 레버리지 ETF 추천")
    print(f"    A(60~79) 강한 발화 — ETF 적극 매수")
    print(f"    B(40~59) 관심 — ETF 관심")
    print(f"    C(20~39) 약세 / D(<20) 비활성")
    print(f"\n  [종목 등급]")
    print(f"    STRONG(50+) 강력 매수 / BUY(35~49) 매수 / WATCH(20~34) 관찰")
    print(SEP)


def save_output(sectors: list[dict], picks: list[dict]):
    """JSON + CSV 저장."""
    today = datetime.now().strftime("%Y%m%d")

    output = {
        "date": today,
        "type": "sector_fire",
        "sector_count": len(sectors),
        "pick_count": len(picks),
        "sectors": sectors,
        "picks": picks,
    }

    json_path = OUTPUT_DIR / f"sector_fire_{today}.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("JSON 저장: %s", json_path)

    if picks:
        df = pd.DataFrame(picks)
        csv_path = OUTPUT_DIR / f"sector_fire_{today}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV 저장: %s", csv_path)


# ─────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 발화 스캔 (FIRE)")
    parser.add_argument("--lookback", type=int, default=10, help="수급 조회 거래일 수")
    args = parser.parse_args()

    sectors, picks = scan_sector_fire(lookback=args.lookback)
    print_report(sectors, picks)
    save_output(sectors, picks)


if __name__ == "__main__":
    main()
