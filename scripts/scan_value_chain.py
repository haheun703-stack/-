"""밸류체인 릴레이 스캐너 — 대장주 급등 시 소부장 저평가 종목 발굴.

ETF 구성 비중을 활용하여:
  1. 대장주 (비중 10%↑) 당일 급등 감지
  2. 같은 ETF 내 소부장 (비중 10%↓) 중 저평가+준비 종목 발굴
  3. 4축 점수: 미반영도(25) + 기술적(25) + 수급(25) + 저평가도(25)

사용법:
    python scripts/scan_value_chain.py              # 전체 섹터 스캔
    python scripts/scan_value_chain.py --sector 반도체  # 특정 섹터만

출력: data/value_chain_relay.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
ETF_DIR = PROJECT_ROOT / "data" / "sector_rotation" / "etf_compositions"
CONFIG_PATH = PROJECT_ROOT / "config" / "value_chain.yaml"
TARGETS_PATH = PROJECT_ROOT / "data" / "institutional_targets.json"
DART_PATH = PROJECT_ROOT / "data" / "dart_event_signals.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "value_chain_relay.json"


def _sf(val, default=0.0):
    """NaN/Inf 안전 변환"""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────
# 설정 로드
# ──────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_targets() -> dict:
    """기관 추정 목표가 로드 → {ticker: {gap_pct, ...}}"""
    if not TARGETS_PATH.exists():
        return {}
    with open(TARGETS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("targets", {})


def load_dart_avoid() -> set[str]:
    """DART 이벤트 AVOID 종목 로드"""
    if not DART_PATH.exists():
        return set()
    with open(DART_PATH, encoding="utf-8") as f:
        data = json.load(f)
    avoid = set()
    for item in data if isinstance(data, list) else data.get("signals", []):
        if isinstance(item, dict) and item.get("action") == "AVOID":
            avoid.add(item.get("ticker", ""))
    return avoid


# ──────────────────────────────────────────
# 종목명 매핑
# ──────────────────────────────────────────

def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑"""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name, ticker = parts
            name_map[ticker] = name
    return name_map


# ──────────────────────────────────────────
# ETF 구성 로드 + 역할 분류
# ──────────────────────────────────────────

def load_etf_compositions(config: dict) -> dict[str, dict]:
    """활성 섹터의 ETF 구성 데이터 로드.

    Returns:
        {sector: {"etf_code": str, "stocks": [{"code", "name", "weight"}, ...]}}
    """
    sectors = config.get("active_sectors", {})
    result = {}
    for sector_name, etf_code in sectors.items():
        # 파일명: {etf_code}_{sector}.json
        pattern = f"{etf_code}_*.json"
        files = list(ETF_DIR.glob(pattern))
        if not files:
            logger.warning("ETF 파일 없음: %s (%s)", sector_name, pattern)
            continue
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        # weight 0 이거나 이름이 비정상인 종목 제외
        stocks = [
            s for s in data.get("stocks", [])
            if s.get("weight", 0) > 0 and "Empty" not in str(s.get("name", ""))
        ]
        result[sector_name] = {
            "etf_code": etf_code,
            "stocks": stocks,
        }
    return result


def classify_roles(
    etf_data: dict[str, dict],
    config: dict,
) -> dict[str, dict]:
    """ETF 비중 기반 역할 분류: leader / tier1 / tier2.

    Returns:
        {sector: {
            "etf_code": str,
            "leaders": [{"code", "name", "weight"}],
            "tier1": [...],   # 3~10%
            "tier2": [...],   # <3%
        }}
    """
    min_leader_w = config.get("min_leader_weight", 10.0)
    overrides = config.get("leader_override", {})
    result = {}

    for sector, info in etf_data.items():
        leaders, tier1, tier2 = [], [], []
        override_codes = set(overrides.get(sector, []))

        for stock in info["stocks"]:
            code = stock["code"]
            weight = stock.get("weight", 0)

            if weight >= min_leader_w or code in override_codes:
                leaders.append(stock)
            elif weight >= 3.0:
                tier1.append(stock)
            else:
                tier2.append(stock)

        result[sector] = {
            "etf_code": info["etf_code"],
            "leaders": leaders,
            "tier1": tier1,
            "tier2": tier2,
        }
    return result


# ──────────────────────────────────────────
# Parquet 로드
# ──────────────────────────────────────────

def load_parquet(ticker: str) -> pd.Series | None:
    """종목 parquet 마지막 행 로드"""
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df.iloc[-1]
    except Exception as e:
        logger.debug("parquet 로드 실패 %s: %s", ticker, e)
        return None


# ──────────────────────────────────────────
# 대장주 발화 감지
# ──────────────────────────────────────────

def detect_fired_leaders(
    roles: dict[str, dict],
    config: dict,
) -> dict[str, dict]:
    """각 섹터 대장주의 당일 변동률/거래량 확인 → 발화 여부 판정.

    Returns:
        {sector: {
            "leaders": [{"code", "name", "weight", "change_pct", "vol_z", "fire_reason"}],
            "avg_change": float,
        }}
    """
    threshold = config.get("fire_threshold", 2.5)
    vol_z_min = config.get("fire_vol_z", 1.5)
    fired = {}

    for sector, role_info in roles.items():
        fired_leaders = []
        for leader in role_info["leaders"]:
            row = load_parquet(leader["code"])
            if row is None:
                continue
            change = _sf(row.get("change_pct", row.get("price_change", 0)))
            vz = _sf(row.get("vol_z", 0))

            reasons = []
            if change >= threshold:
                reasons.append(f"+{change:.1f}%")
            if vz >= vol_z_min:
                reasons.append(f"거래량z {vz:.1f}")

            if reasons:
                fired_leaders.append({
                    "code": leader["code"],
                    "name": leader["name"],
                    "weight": leader["weight"],
                    "change_pct": round(change, 2),
                    "vol_z": round(vz, 2),
                    "fire_reason": " + ".join(reasons),
                })

        if fired_leaders:
            avg_chg = sum(l["change_pct"] for l in fired_leaders) / len(fired_leaders)
            fired[sector] = {
                "leaders": fired_leaders,
                "avg_change": round(avg_chg, 2),
            }
    return fired


# ──────────────────────────────────────────
# 수급 데이터 (CSV 기반)
# ──────────────────────────────────────────

def get_flow_from_csv(ticker: str) -> dict:
    """CSV에서 5일 수급 + 연속매수 일수"""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}
    try:
        df = pd.read_csv(csvs[0], parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 5:
            return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}

        last5 = df.tail(5)
        f5 = i5 = 0.0
        if "Foreign_Net" in df.columns:
            aligned = pd.DataFrame({
                "net": last5["Foreign_Net"].values,
                "close": last5["Close"].values,
            }).dropna()
            if len(aligned) > 0:
                f5 = round(float((aligned["net"] * aligned["close"]).sum()) / 1e8, 1)
        if "Inst_Net" in df.columns:
            aligned = pd.DataFrame({
                "net": last5["Inst_Net"].values,
                "close": last5["Close"].values,
            }).dropna()
            if len(aligned) > 0:
                i5 = round(float((aligned["net"] * aligned["close"]).sum()) / 1e8, 1)

        def streak(col):
            if col not in df.columns:
                return 0
            vals = df[col].dropna().values
            s = 0
            for v in reversed(vals):
                if v > 0:
                    s += 1
                else:
                    break
            return s

        return {
            "foreign_5d": f5,
            "inst_5d": i5,
            "f_streak": streak("Foreign_Net"),
            "i_streak": streak("Inst_Net"),
        }
    except Exception as e:
        logger.debug("수급 로드 실패 %s: %s", ticker, e)
        return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}


# ──────────────────────────────────────────
# 소부장 4축 점수 (100점)
# ──────────────────────────────────────────

def score_supplier(
    row: pd.Series,
    flow: dict,
    leader_avg_change: float,
    target_info: dict | None,
) -> dict:
    """소부장 4축 점수 계산.

    축1: 미반영도 gap (25점)
    축2: 기술적 준비도 (25점)
    축3: 수급 (25점)
    축4: 저평가도 (25점)
    """
    change = _sf(row.get("change_pct", row.get("price_change", 0)))
    rsi = _sf(row.get("rsi_14", 50), 50)
    bb_pos = _sf(row.get("bb_position", 0.5), 0.5)
    close = _sf(row.get("close", 0))
    ma20 = _sf(row.get("sma_20", 0))
    ma60 = _sf(row.get("sma_60", 0))
    macd_hist = _sf(row.get("macd_histogram", 0))
    macd_hist_prev = _sf(row.get("macd_histogram_prev", 0))
    trix_gc = int(_sf(row.get("trix_golden_cross", 0)))
    vz = _sf(row.get("vol_z", 0))
    f_net_5d = _sf(row.get("foreign_net_5d", 0))
    f_consec = _sf(row.get("foreign_consecutive_buy", 0))

    # ── 축1: 미반영도 (25점) ──
    gap_pct = leader_avg_change - change
    max_gap = 8.0
    gap_score = min(max(gap_pct / max_gap, 0) * 25, 25)

    # ── 축2: 기술적 준비도 (25점) ──
    tech_score = 0.0
    # RSI 구간
    if 30 <= rsi <= 50:
        tech_score += 10
    elif 50 < rsi <= 60:
        tech_score += 7
    elif 25 <= rsi < 30:
        tech_score += 5
    # MA20 위
    if close > 0 and ma20 > 0 and close > ma20:
        tech_score += 5
    # MA60 위
    if close > 0 and ma60 > 0 and close > ma60:
        tech_score += 3
    # MACD 히스토그램 상승
    if macd_hist > macd_hist_prev:
        tech_score += 4
    # TRIX 골든크로스
    if trix_gc:
        tech_score += 3
    tech_score = min(tech_score, 25)

    # ── 축3: 수급 (25점) ──
    flow_score = 0.0
    # parquet foreign_net_5d
    if f_net_5d > 0:
        flow_score += 10
    # CSV inst_5d
    if flow.get("inst_5d", 0) > 0:
        flow_score += 7
    # 외인 연속매수
    if f_consec >= 3:
        flow_score += 5
    elif flow.get("f_streak", 0) >= 3:
        flow_score += 5
    # 거래량 z-score
    if vz >= 1.5:
        flow_score += 3
    flow_score = min(flow_score, 25)

    # ── 축4: 저평가도 (25점) ──
    value_score = 0.0
    if target_info:
        tgt_gap = target_info.get("gap_pct", 0)
        # gap_pct가 음수 = 목표가 아래 = 저평가
        upside = -tgt_gap if tgt_gap < 0 else 0
        if upside >= 20:
            value_score = 25
        elif upside >= 15:
            value_score = 20
        elif upside >= 10:
            value_score = 15
        elif upside >= 5:
            value_score = 8
    else:
        # 목표가 없으면 BB position 활용 (하단일수록 높은 점수)
        if bb_pos <= 0.2:
            value_score = 18
        elif bb_pos <= 0.35:
            value_score = 12
        elif bb_pos <= 0.5:
            value_score = 6

    total = round(gap_score + tech_score + flow_score + value_score)
    return {
        "total": total,
        "gap": round(gap_score),
        "tech": round(tech_score),
        "flow": round(flow_score),
        "value": round(value_score),
        "gap_pct": round(gap_pct, 2),
        "change_pct": round(change, 2),
        "rsi": round(rsi, 1),
        "bb_pos": round(bb_pos, 3),
        "vol_z": round(vz, 2),
        "foreign_5d": flow.get("foreign_5d", 0),
        "inst_5d": flow.get("inst_5d", 0),
        "f_consec": int(f_consec),
    }


# ──────────────────────────────────────────
# 소부장 필터 + 랭킹
# ──────────────────────────────────────────

def scan_suppliers(
    sector: str,
    role_info: dict,
    fire_info: dict,
    config: dict,
    targets: dict,
    dart_avoid: set,
    name_map: dict,
) -> list[dict]:
    """발화 섹터의 소부장 점수 계산 + ready_filter 적용."""
    rf = config.get("ready_filter", {})
    max_change = rf.get("change_pct_max", 3.0)
    rsi_lo, rsi_hi = rf.get("rsi_range", [25, 60])
    need_above_ma20 = rf.get("above_ma20", True)
    avg_leader_change = fire_info["avg_change"]

    candidates = []
    # tier1 + tier2 = 소부장 후보
    suppliers = role_info["tier1"] + role_info["tier2"]

    for stock in suppliers:
        code = stock["code"]
        if code in dart_avoid:
            continue

        row = load_parquet(code)
        if row is None:
            continue

        change = _sf(row.get("change_pct", row.get("price_change", 0)))
        rsi = _sf(row.get("rsi_14", 50), 50)
        close = _sf(row.get("close", 0))
        ma20 = _sf(row.get("sma_20", 0))

        # ready_filter 적용
        if change > max_change:
            continue
        if not (rsi_lo <= rsi <= rsi_hi):
            continue
        if need_above_ma20 and close > 0 and ma20 > 0 and close <= ma20:
            continue

        flow = get_flow_from_csv(code)
        target_info = targets.get(code)
        scores = score_supplier(row, flow, avg_leader_change, target_info)

        # 이름 결정: ETF 데이터 > CSV 파일명
        name = stock.get("name", "") or name_map.get(code, code)

        # 사유 텍스트 생성
        reasons = []
        if scores["gap_pct"] > 1.0:
            reasons.append(f"미반영 {scores['gap_pct']:.1f}%")
        if 25 <= scores["rsi"] <= 55:
            reasons.append(f"RSI {scores['rsi']:.0f} 적정")
        if scores["foreign_5d"] > 0:
            reasons.append("외인 5일 순매수")
        if scores["inst_5d"] > 0:
            reasons.append("기관 5일 순매수")
        if target_info and target_info.get("gap_pct", 0) < -5:
            reasons.append(f"목표가 상승여력 {-target_info['gap_pct']:.0f}%")

        tier = "tier1" if stock["weight"] >= 3.0 else "tier2"
        target_gap = round(-target_info["gap_pct"], 1) if target_info and target_info.get("gap_pct", 0) < 0 else None

        candidates.append({
            "ticker": code,
            "name": name,
            "tier": tier,
            "weight": stock["weight"],
            "change_pct": scores["change_pct"],
            "gap_pct": scores["gap_pct"],
            "score": scores["total"],
            "score_detail": {
                "gap": scores["gap"],
                "tech": scores["tech"],
                "flow": scores["flow"],
                "value": scores["value"],
            },
            "rsi": scores["rsi"],
            "bb_pos": scores["bb_pos"],
            "vol_z": scores["vol_z"],
            "foreign_5d": scores["foreign_5d"],
            "inst_5d": scores["inst_5d"],
            "f_consec": scores["f_consec"],
            "target_gap_pct": target_gap,
            "reasons": reasons,
        })

    # 점수 내림차순 정렬, 상위 10개
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="밸류체인 릴레이 스캐너")
    parser.add_argument("--sector", type=str, default=None, help="특정 섹터만 스캔")
    parser.add_argument("--top", type=int, default=10, help="섹터당 상위 N개")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  밸류체인 릴레이 스캐너 — 대장주→소부장 발굴")
    print("=" * 60)

    config = load_config()
    targets = load_targets()
    dart_avoid = load_dart_avoid()
    name_map = build_name_map()

    # ETF 구성 로드 + 역할 분류
    etf_data = load_etf_compositions(config)
    if args.sector:
        etf_data = {k: v for k, v in etf_data.items() if k == args.sector}
    if not etf_data:
        print("  활성 섹터 없음")
        return

    roles = classify_roles(etf_data, config)
    print(f"\n  활성 섹터: {len(roles)}개")
    for sector, info in roles.items():
        print(f"    {sector}: 대장주 {len(info['leaders'])}개 | "
              f"Tier1 {len(info['tier1'])}개 | Tier2 {len(info['tier2'])}개")

    # 대장주 발화 감지
    fired = detect_fired_leaders(roles, config)
    if not fired:
        print("\n  [결과] 발화 섹터 없음 — 대장주 급등 미감지")
        # 빈 결과 저장
        output = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fired_sectors": [],
            "no_fire_sectors": list(roles.keys()),
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  저장: {OUTPUT_PATH}")
        return

    print(f"\n  발화 섹터: {len(fired)}개")
    for sector, info in fired.items():
        for ld in info["leaders"]:
            print(f"    {sector} | {ld['name']} {ld['fire_reason']} (비중 {ld['weight']:.1f}%)")

    # 소부장 스캔
    fired_sectors = []
    no_fire_sectors = [s for s in roles if s not in fired]

    for sector, fire_info in fired.items():
        print(f"\n  ── {sector} 소부장 스캔 ──")
        candidates = scan_suppliers(
            sector, roles[sector], fire_info, config, targets, dart_avoid, name_map,
        )
        leader_list = [
            {"ticker": l["code"], "name": l["name"],
             "change_pct": l["change_pct"], "weight": l["weight"]}
            for l in fire_info["leaders"]
        ]
        sector_result = {
            "sector": sector,
            "etf_code": roles[sector]["etf_code"],
            "leaders": leader_list,
            "avg_leader_change": fire_info["avg_change"],
            "candidates": candidates[:args.top],
        }
        fired_sectors.append(sector_result)

        if candidates:
            print(f"  {'순위':>4} {'종목':>12} {'티어':>5} {'비중':>5} {'등락':>6} "
                  f"{'갭':>5} {'점수':>4} {'사유'}")
            print(f"  {'─' * 70}")
            for i, c in enumerate(candidates[:args.top], 1):
                reasons_str = ", ".join(c["reasons"][:3]) if c["reasons"] else "-"
                print(f"  {i:>4} {c['name']:>12} {c['tier']:>5} "
                      f"{c['weight']:>5.1f} {c['change_pct']:>+5.1f}% "
                      f"{c['gap_pct']:>+4.1f}% {c['score']:>4} {reasons_str}")
        else:
            print("  준비된 소부장 없음 (필터 미충족)")

    # JSON 저장
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fired_sectors": fired_sectors,
        "no_fire_sectors": no_fire_sectors,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {OUTPUT_PATH}")
    print(f"  발화 {len(fired_sectors)}섹터, 미발화 {len(no_fire_sectors)}섹터")


if __name__ == "__main__":
    main()
