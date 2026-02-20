"""슈퍼섹터 릴레이 감지 엔진.

같은 대분류(슈퍼섹터) 내에서 선행 섹터 → 후행 섹터로 자금이
넘어가는 "릴레이" 패턴을 자동 감지한다.

핵심 로직:
  1. 슈퍼섹터 내 선행 섹터: 모멘텀 Top3 + RSI > 70 (과열)
  2. 형제 섹터 거래대금 +30% 증가 → 릴레이 후보
  3. Zone B 오버라이드: 릴레이 활성 시 Zone B → Zone A (HALF 사이즈)

입력:
  - data/sector_rotation/wics_mapping.csv       : WICS 3층 매핑
  - data/sector_rotation/wics_etf_bridge.csv    : WICS↔ETF 브릿지
  - data/sector_rotation/sector_momentum.json   : 섹터 모멘텀 순위
  - data/sector_rotation/etf_daily/*.parquet    : ETF 일별 시세

출력:
  - data/sector_rotation/relay_signal.json      : 릴레이 감지 결과

사용법:
  python scripts/sector_relay_engine.py             # 릴레이 감지
  python scripts/sector_relay_engine.py --verbose   # 상세 출력
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = DATA_DIR / "etf_daily"

# 릴레이 감지 파라미터
LEADER_TOP_N = 3          # 선행 섹터: 모멘텀 Top N
LEADER_RSI_MIN = 70       # 선행 섹터: RSI 최소 (과열 시작)
VOLUME_CHANGE_MIN = 0.30  # 형제 섹터: 거래대금 전일비 +30%
OVERRIDE_SIZE = "HALF"    # 오버라이드 시 포지션 크기
OVERRIDE_STOP = -3.0      # 오버라이드 시 손절%


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_wics_mapping() -> pd.DataFrame:
    path = DATA_DIR / "wics_mapping.csv"
    if not path.exists():
        logger.error("wics_mapping.csv 없음 — wics_sector_mapper.py 먼저 실행")
        sys.exit(1)
    return pd.read_csv(path, dtype={"stock_code": str})


def load_etf_bridge() -> dict[str, dict]:
    """WICS 중분류명 → {etf_code, etf_name, super_sector_code, super_sector_name}."""
    path = DATA_DIR / "wics_etf_bridge.csv"
    if not path.exists():
        logger.error("wics_etf_bridge.csv 없음")
        sys.exit(1)
    df = pd.read_csv(path)
    bridge = {}
    for _, row in df.iterrows():
        # ETF 코드 6자리 zero-pad (CSV에서 선행 0 유실 방지)
        etf_code = str(int(row["etf_code"])).zfill(6)
        bridge[row["wics_sector"]] = {
            "etf_code": etf_code,
            "etf_name": row.get("etf_name", ""),
            "super_sector_code": row.get("super_sector_code", ""),
            "super_sector_name": row.get("super_sector_name", ""),
        }
    return bridge


def load_momentum() -> list[dict]:
    path = DATA_DIR / "sector_momentum.json"
    if not path.exists():
        logger.error("sector_momentum.json 없음 — sector_momentum.py 먼저 실행")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sectors", [])


def load_etf_ohlcv(etf_code: str) -> pd.DataFrame | None:
    path = DAILY_DIR / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────
# ETF→WICS 중분류 역매핑 구축
# ─────────────────────────────────────────────

def build_etf_to_wics(bridge: dict, momentum_list: list[dict]) -> dict:
    """기존 ETF 섹터명 → WICS 중분류명 역매핑.

    sector_momentum.json의 섹터명(예: '증권', '은행')과
    wics_etf_bridge의 WICS 중분류명을 연결한다.
    """
    # ETF 코드 기준으로 매핑
    etf_to_wics = {}  # etf_code → wics_sector_name
    for wics_name, info in bridge.items():
        etf_to_wics[info["etf_code"]] = wics_name

    # sector_momentum.json의 sector명 → WICS 중분류명
    sector_to_wics = {}
    for m in momentum_list:
        etf_code = m.get("etf_code", "")
        sector_name = m["sector"]
        if etf_code in etf_to_wics:
            sector_to_wics[sector_name] = etf_to_wics[etf_code]

    return sector_to_wics


# ─────────────────────────────────────────────
# 릴레이 감지
# ─────────────────────────────────────────────

def detect_relays(
    momentum_list: list[dict],
    bridge: dict,
    wics_mapping: pd.DataFrame,
    verbose: bool = False,
) -> list[dict]:
    """슈퍼섹터 릴레이를 감지한다.

    Returns:
        list of relay signals:
        [{
            'supersector': '금융',
            'supersector_code': 'G40',
            'leader_sector': '증권',
            'leader_etf': '157500',
            'leader_rsi': 83.9,
            'leader_rank': 1,
            'relay_candidates': [{
                'sector': '보험',
                'etf_code': '140710',
                'volume_change_pct': 45.2,
                'override': True,
            }],
        }]
    """
    # 1. ETF→WICS 역매핑 구축
    sector_to_wics = build_etf_to_wics(bridge, momentum_list)

    # 2. WICS 중분류별 슈퍼섹터 정보
    wics_to_super = {}
    for wics_name, info in bridge.items():
        wics_to_super[wics_name] = {
            "super_code": info["super_sector_code"],
            "super_name": info["super_sector_name"],
            "etf_code": info["etf_code"],
        }

    # 3. 선행 섹터 식별: 모멘텀 Top N + RSI > threshold
    leaders = []
    for m in momentum_list:
        rank = m.get("rank", 99)
        rsi = m.get("rsi_14", 0)
        sector_name = m["sector"]
        wics_name = sector_to_wics.get(sector_name)

        if wics_name and rank <= LEADER_TOP_N and rsi >= LEADER_RSI_MIN:
            leaders.append({
                "sector": sector_name,
                "wics_name": wics_name,
                "etf_code": m["etf_code"],
                "rank": rank,
                "rsi": rsi,
                "ret_5": m.get("ret_5", 0),
                "ret_20": m.get("ret_20", 0),
                **wics_to_super.get(wics_name, {}),
            })

    if not leaders:
        logger.info("선행 섹터 없음 (Top%d + RSI>%d 조건)", LEADER_TOP_N, LEADER_RSI_MIN)
        return []

    if verbose:
        for l in leaders:
            logger.info("선행 섹터: %s (WICS: %s, RSI=%.1f, Rank=%d)",
                        l["sector"], l["wics_name"], l["rsi"], l["rank"])

    # 4. 선행 섹터 ETF 코드 세트 (형제 후보에서 제외용)
    leader_etf_codes = {l["etf_code"] for l in leaders}

    # 5. 슈퍼섹터별로 형제 섹터 릴레이 감지
    relays = []
    for leader in leaders:
        super_code = leader.get("super_code", "")
        if not super_code:
            continue

        # 같은 슈퍼섹터의 형제 섹터 (ETF 있는 것만)
        # 이미 선행 섹터인 형제는 제외 (이미 과열 중이므로 릴레이 대상 아님)
        siblings = []
        for wics_name, info in wics_to_super.items():
            if (info["super_code"] == super_code
                    and wics_name != leader["wics_name"]
                    and info["etf_code"] not in leader_etf_codes):
                siblings.append({
                    "wics_name": wics_name,
                    "etf_code": info["etf_code"],
                })

        if not siblings:
            continue

        # 5. 형제 섹터 거래대금 변화 확인
        relay_candidates = []
        for sib in siblings:
            etf_df = load_etf_ohlcv(sib["etf_code"])
            if etf_df is None or len(etf_df) < 5:
                continue

            # 거래대금 전일 대비 변화
            if "trading_value" in etf_df.columns:
                tv = etf_df["trading_value"].astype(float)
            elif "volume" in etf_df.columns:
                tv = etf_df["volume"].astype(float)
            else:
                continue

            today_tv = float(tv.iloc[-1])
            prev_avg = float(tv.iloc[-6:-1].mean())  # 직전 5일 평균

            if prev_avg <= 0:
                continue

            vol_change = (today_tv - prev_avg) / prev_avg

            # 형제 섹터의 모멘텀 정보 찾기
            sib_momentum = None
            for m in momentum_list:
                if m["etf_code"] == sib["etf_code"]:
                    sib_momentum = m
                    break

            sib_rsi = sib_momentum["rsi_14"] if sib_momentum else 0
            sib_rank = sib_momentum["rank"] if sib_momentum else 99

            candidate = {
                "sector": sib["wics_name"],
                "etf_code": sib["etf_code"],
                "volume_change_pct": round(vol_change * 100, 1),
                "rsi": round(sib_rsi, 1),
                "rank": sib_rank,
                "override": vol_change >= VOLUME_CHANGE_MIN,
            }
            relay_candidates.append(candidate)

            if verbose:
                mark = "→ RELAY!" if candidate["override"] else ""
                logger.info(
                    "  형제 %s: 거래대금 %+.1f%%, RSI=%.1f %s",
                    sib["wics_name"], vol_change * 100, sib_rsi, mark,
                )

        if relay_candidates:
            relays.append({
                "supersector": leader.get("super_name", ""),
                "supersector_code": super_code,
                "leader_sector": leader["wics_name"],
                "leader_etf": leader["etf_code"],
                "leader_rsi": round(leader["rsi"], 1),
                "leader_rank": leader["rank"],
                "leader_ret_5": leader.get("ret_5", 0),
                "leader_ret_20": leader.get("ret_20", 0),
                "relay_candidates": sorted(
                    relay_candidates,
                    key=lambda x: x["volume_change_pct"],
                    reverse=True,
                ),
            })

    return relays


# ─────────────────────────────────────────────
# Zone B 오버라이드 종목 리스트
# ─────────────────────────────────────────────

def get_override_stocks(
    relays: list[dict],
    wics_mapping: pd.DataFrame,
) -> list[dict]:
    """릴레이가 활성화된 형제 섹터의 종목 리스트를 반환.

    이 종목들은 Zone B여도 Zone A로 오버라이드 가능.
    """
    override_stocks = []

    for relay in relays:
        active_siblings = [c for c in relay["relay_candidates"] if c["override"]]
        if not active_siblings:
            continue

        for sib in active_siblings:
            sector_name = sib["sector"]
            # WICS 매핑에서 해당 중분류 종목들 추출
            sector_stocks = wics_mapping[
                wics_mapping["sector_name"] == sector_name
            ].sort_values("market_cap", ascending=False)

            for _, row in sector_stocks.iterrows():
                override_stocks.append({
                    "stock_code": row["stock_code"],
                    "stock_name": row["stock_name"],
                    "sector": sector_name,
                    "supersector": relay["supersector"],
                    "leader_sector": relay["leader_sector"],
                    "leader_rsi": relay["leader_rsi"],
                    "volume_change_pct": sib["volume_change_pct"],
                    "override_size": OVERRIDE_SIZE,
                    "override_stop": OVERRIDE_STOP,
                })

    return override_stocks


# ─────────────────────────────────────────────
# 저장 + 출력
# ─────────────────────────────────────────────

def save_relay_signal(relays: list[dict], override_stocks: list[dict]) -> Path:
    """relay_signal.json 저장."""
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "params": {
            "leader_top_n": LEADER_TOP_N,
            "leader_rsi_min": LEADER_RSI_MIN,
            "volume_change_min_pct": VOLUME_CHANGE_MIN * 100,
            "override_size": OVERRIDE_SIZE,
            "override_stop_pct": OVERRIDE_STOP,
        },
        "relays": relays,
        "override_stocks": override_stocks,
        "summary": {
            "active_relays": sum(
                1 for r in relays
                if any(c["override"] for c in r["relay_candidates"])
            ),
            "total_override_stocks": len(override_stocks),
        },
    }

    out_path = DATA_DIR / "relay_signal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("릴레이 시그널 저장: %s", out_path)
    return out_path


def print_relay_report(relays: list[dict], override_stocks: list[dict]):
    """릴레이 감지 결과를 출력."""
    active = [r for r in relays if any(c["override"] for c in r["relay_candidates"])]

    print(f"\n{'=' * 60}")
    print(f"  슈퍼섹터 릴레이 감지 결과")
    print(f"{'=' * 60}")

    if not relays:
        print("\n  릴레이 감지 없음")
        return

    for relay in relays:
        is_active = any(c["override"] for c in relay["relay_candidates"])
        status = "ACTIVE" if is_active else "WATCH"

        print(f"\n  [{relay['supersector']}] — {status}")
        print(f"  선행: {relay['leader_sector']} "
              f"(#{relay['leader_rank']}, RSI {relay['leader_rsi']}, "
              f"5일 {relay['leader_ret_5']:+.1f}%, 20일 {relay['leader_ret_20']:+.1f}%)")

        if relay["leader_rsi"] >= 75:
            print(f"  → 선행 섹터 과열 주의! 추격 금지")

        print(f"\n  릴레이 후보:")
        for c in relay["relay_candidates"]:
            mark = " ← RELAY!" if c["override"] else ""
            print(f"    {c['sector']}: 거래대금 {c['volume_change_pct']:+.1f}%, "
                  f"RSI {c['rsi']:.1f}, #{c['rank']}{mark}")

    if override_stocks:
        # 섹터별 그룹핑
        sectors = {}
        for s in override_stocks:
            key = s["sector"]
            if key not in sectors:
                sectors[key] = []
            sectors[key].append(s)

        print(f"\n{'─' * 60}")
        print(f"  Zone B → A 오버라이드 대상 ({len(override_stocks)}종목)")
        for sector, stocks in sectors.items():
            supersector = stocks[0]["supersector"]
            leader = stocks[0]["leader_sector"]
            vol_chg = stocks[0]["volume_change_pct"]
            print(f"\n  [{supersector}] {sector} (← {leader} 릴레이, 거래대금 {vol_chg:+.1f}%)")
            for s in stocks[:10]:  # 시총 상위 10개만 출력
                print(f"    {s['stock_name']} ({s['stock_code']}) — "
                      f"{OVERRIDE_SIZE}, 손절 {OVERRIDE_STOP}%")
            if len(stocks) > 10:
                print(f"    ... 외 {len(stocks) - 10}종목")

    print(f"\n{'─' * 60}")
    print(f"  활성 릴레이: {len(active)}건, 오버라이드 종목: {len(override_stocks)}개")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="슈퍼섹터 릴레이 감지 엔진")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 출력")
    args = parser.parse_args()

    # 데이터 로드
    wics_mapping = load_wics_mapping()
    bridge = load_etf_bridge()
    momentum_list = load_momentum()

    logger.info("WICS 매핑: %d종목, ETF 브릿지: %d개, 모멘텀: %d섹터",
                len(wics_mapping), len(bridge), len(momentum_list))

    # 릴레이 감지
    relays = detect_relays(momentum_list, bridge, wics_mapping, verbose=args.verbose)

    # 오버라이드 종목 추출
    override_stocks = get_override_stocks(relays, wics_mapping)

    # 저장 + 출력
    save_relay_signal(relays, override_stocks)
    print_relay_report(relays, override_stocks)


if __name__ == "__main__":
    main()
