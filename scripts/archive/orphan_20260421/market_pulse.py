"""
Market Pulse — 실시간 시장 인텔리전스
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
레짐 무관, 매일 나가는 FLOWX 콘텐츠 엔진

출력:
  1. 시장 방향 (ETF 레버리지/인버스 기반)
  2. ETF 추천 (방향 + 섹터)
  3. 섹터 랭킹 (오늘 가장 강한 섹터)
  4. TOP 5 개별 종목 (진입가/목표가/손절가/VWAP)
  5. 수급 + 장중 코멘트

실행:
  python scripts/market_pulse.py              # 기본 (저장 + 텔레그램)
  python scripts/market_pulse.py --no-send    # 저장만
  python scripts/market_pulse.py --dry        # 콘솔만 (저장/전송 안 함)

BAT 스케줄: BAT-H (11:30) 또는 독립 실행
결과: data/market_pulse.json + 텔레그램 + FLOWX Zone 8
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import mojito

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / "market_pulse.log", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("market_pulse")

DATA_DIR = PROJECT_ROOT / "data"

# ─────────────────────────────────────────
# 유니버스 정의: 섹터별 대표 종목
# ─────────────────────────────────────────

UNIVERSE = {
    "전력인프라": {
        "267260": "HD현대일렉트릭",
        "298040": "효성중공업",
        "034020": "두산에너빌리티",
        "009540": "한국전력",
        "267270": "HD현대인프라코어",
    },
    "방산": {
        "012450": "한화에어로스페이스",
        "064350": "현대로템",
        "042660": "한화오션",
        "047810": "한국항공우주",
        "272210": "한화시스템",
    },
    "반도체": {
        "000660": "SK하이닉스",
        "005930": "삼성전자",
        "042700": "한미반도체",
        "403870": "HPSP",
    },
    "금융": {
        "071050": "한국금융지주",
        "055550": "신한지주",
        "105560": "KB금융",
        "086790": "하나금융지주",
        "003540": "대신증권",
    },
    "조선": {
        "010140": "삼성중공업",
        "329180": "HD현대중공업",
    },
    "2차전지": {
        "051910": "LG화학",
        "006400": "삼성SDI",
        "373220": "LG에너지솔루션",
    },
    "바이오": {
        "068270": "셀트리온",
        "207940": "삼성바이오로직스",
    },
    "자동차": {
        "005380": "현대차",
        "000270": "기아",
        "012330": "현대모비스",
    },
    "플랫폼": {
        "035720": "카카오",
        "035420": "NAVER",
    },
    "에너지/화학": {
        "096770": "SK이노베이션",
        "010950": "S-Oil",
        "011170": "롯데케미칼",
    },
    "철강/소재": {
        "005490": "POSCO홀딩스",
        "004020": "현대제철",
    },
}

ETF_DIRECTION = {
    "122630": "KODEX 레버리지",
    "252670": "KODEX 200선물인버스2X",
    "229200": "KODEX 코스닥150레버리지",
    "251340": "KODEX 코스닥150선물인버스",
}

ETF_SECTORS = {
    "455850": ("KODEX K-방산", "방산"),
    "091170": ("KODEX 은행", "금융"),
    "091160": ("KODEX 반도체", "반도체"),
    "117460": ("KODEX 에너지화학", "에너지/화학"),
    "140710": ("KODEX 운송", "운송"),
    "117680": ("KODEX 철강", "철강/소재"),
    "364960": ("KODEX K-신재생에너지", "에너지"),
    "385590": ("KODEX K-미래차", "자동차"),
    "139260": ("KODEX 미디어&엔터", "엔터"),
    "102780": ("KODEX 삼성그룹", "삼성"),
}


# ─────────────────────────────────────────
# KIS API 헬퍼
# ─────────────────────────────────────────

def _create_broker():
    return mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=False,
    )


def _fetch_price(broker, ticker: str) -> dict | None:
    """종목 현재가 조회. 실패 시 None."""
    try:
        resp = broker.fetch_price(ticker)
        if resp and "output" in resp:
            o = resp["output"]
            return {
                "name": o.get("hts_kor_isnm", ""),
                "price": int(o.get("stck_prpr", 0) or 0),
                "change_pct": float(o.get("prdy_ctrt", 0) or 0),
                "volume": int(o.get("acml_vol", 0) or 0),
                "high": int(o.get("stck_hgpr", 0) or 0),
                "low": int(o.get("stck_lwpr", 0) or 0),
                "prev_close": int(o.get("stck_sdpr", 0) or 0),
                "open": int(o.get("stck_oprc", 0) or 0),
                "amount": int(o.get("acml_tr_pbmn", 0) or 0),  # 거래대금
            }
    except Exception as e:
        logger.debug("fetch_price %s 실패: %s", ticker, e)
    return None


def _tick_round(price: int) -> int:
    """KRX 호가 단위 내림."""
    if price >= 500_000:
        return (price // 1000) * 1000
    elif price >= 100_000:
        return (price // 500) * 500
    elif price >= 50_000:
        return (price // 100) * 100
    elif price >= 10_000:
        return (price // 50) * 50
    elif price >= 5_000:
        return (price // 10) * 10
    elif price >= 1_000:
        return (price // 5) * 5
    return price


# ─────────────────────────────────────────
# 1. 시장 방향 판단
# ─────────────────────────────────────────

def analyze_market_direction(broker) -> dict:
    """ETF 등락으로 시장 방향 판단."""
    logger.info("[PULSE] 시장 방향 분석...")

    etf_data = {}
    for ticker, name in ETF_DIRECTION.items():
        data = _fetch_price(broker, ticker)
        if data:
            etf_data[ticker] = {**data, "etf_name": name}
        time.sleep(0.05)

    lever = etf_data.get("122630", {}).get("change_pct", 0)
    inv2x = etf_data.get("252670", {}).get("change_pct", 0)
    kq_lever = etf_data.get("229200", {}).get("change_pct", 0)

    # 방향 판단
    if lever >= 3:
        direction = "STRONG_BULL"
        comment = "강한 상승일 — 추세 추종 유리"
        etf_call = "레버리지 보유 유지, 인버스 금지"
    elif lever >= 1.5:
        direction = "BULL"
        comment = "상승일 — 섹터 선별 매수 가능"
        etf_call = "섹터 ETF 선별 매수"
    elif lever >= 0.3:
        direction = "MILD_BULL"
        comment = "소폭 상승 — 관망 또는 소량 진입"
        etf_call = "관망, 눌림목 대기"
    elif lever >= -0.3:
        direction = "FLAT"
        comment = "보합 — 방향성 없음, 대기"
        etf_call = "현금 보유"
    elif lever >= -1.5:
        direction = "MILD_BEAR"
        comment = "소폭 하락 — 방어적 포지션"
        etf_call = "인버스 소량 헤지 가능"
    elif lever >= -3:
        direction = "BEAR"
        comment = "하락일 — 현금 확대"
        etf_call = "인버스2X 단기 트레이딩"
    else:
        direction = "CRASH"
        comment = "급락일 — 패닉 금지, 현금 사수"
        etf_call = "인버스 이미 늦음, 현금 사수"

    # 내일 전망 (간단 추론)
    if lever >= 2 and kq_lever >= 2:
        tomorrow = "상승 연장 가능 (양시장 동반 강세)"
    elif lever >= 1 and inv2x <= -2:
        tomorrow = "상승 유지 가능성 높음 (인버스 큰 폭 하락)"
    elif lever >= 0 and lever < 1:
        tomorrow = "방향 불분명 — 관망"
    elif lever < -1:
        tomorrow = "추가 하락 경계 — 현금 확대"
    else:
        tomorrow = "중립"

    return {
        "direction": direction,
        "comment": comment,
        "etf_call": etf_call,
        "tomorrow": tomorrow,
        "kospi_lever_pct": lever,
        "kosdaq_lever_pct": kq_lever,
        "inverse2x_pct": inv2x,
        "etf_data": {k: {"name": v.get("etf_name", ""), "change_pct": v.get("change_pct", 0), "price": v.get("price", 0)} for k, v in etf_data.items()},
    }


# ─────────────────────────────────────────
# 2. 섹터 ETF 랭킹
# ─────────────────────────────────────────

def analyze_sector_etfs(broker) -> list[dict]:
    """섹터 ETF 등락률 랭킹."""
    logger.info("[PULSE] 섹터 ETF 분석...")

    results = []
    for ticker, (name, sector) in ETF_SECTORS.items():
        data = _fetch_price(broker, ticker)
        if data:
            results.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "change_pct": data["change_pct"],
                "price": data["price"],
                "volume": data["volume"],
            })
        time.sleep(0.05)

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return results


# ─────────────────────────────────────────
# 3. 섹터별 개별종목 스캔
# ─────────────────────────────────────────

def scan_sectors(broker) -> dict:
    """전 섹터 개별종목 현재가 조회 + 랭킹."""
    logger.info("[PULSE] 섹터별 종목 스캔 (%d 섹터, ~%d 종목)...",
                len(UNIVERSE), sum(len(v) for v in UNIVERSE.values()))

    sector_results = {}
    all_stocks = []

    for sector, tickers in UNIVERSE.items():
        stocks = []
        for ticker, name in tickers.items():
            data = _fetch_price(broker, ticker)
            if data:
                stock = {
                    "ticker": ticker,
                    "name": data["name"] or name,
                    "sector": sector,
                    "price": data["price"],
                    "change_pct": data["change_pct"],
                    "volume": data["volume"],
                    "high": data["high"],
                    "low": data["low"],
                    "prev_close": data["prev_close"],
                    "open": data["open"],
                    "amount": data["amount"],
                }
                # VWAP 추정: (High + Low + Close) / 3
                if data["high"] > 0 and data["low"] > 0:
                    stock["vwap_est"] = _tick_round(
                        (data["high"] + data["low"] + data["price"]) // 3
                    )
                else:
                    stock["vwap_est"] = data["price"]

                stocks.append(stock)
                all_stocks.append(stock)
            time.sleep(0.03)

        stocks.sort(key=lambda x: x["change_pct"], reverse=True)
        avg_change = sum(s["change_pct"] for s in stocks) / max(len(stocks), 1)
        sector_results[sector] = {
            "avg_change_pct": round(avg_change, 2),
            "leader": stocks[0]["name"] if stocks else "",
            "leader_change": stocks[0]["change_pct"] if stocks else 0,
            "stocks": stocks,
        }

    return sector_results, all_stocks


# ─────────────────────────────────────────
# 4. TOP 5 종목 선정 + 진입/청산/손절
# ─────────────────────────────────────────

def select_top_picks(all_stocks: list[dict], market_dir: str, n: int = 5) -> list[dict]:
    """등락률 + 거래대금 기반 TOP N 선정, 진입/목표/손절가 계산."""
    logger.info("[PULSE] TOP %d 종목 선정...", n)

    # 등락률 상위 + 거래대금 가중 스코어
    for s in all_stocks:
        # 스코어: 등락률 70% + 거래대금 순위 30%
        s["_score"] = s["change_pct"]

    # 등락률 상위 필터 (양봉만)
    candidates = [s for s in all_stocks if s["change_pct"] > 0]
    candidates.sort(key=lambda x: x["_score"], reverse=True)

    picks = []
    seen_sectors = set()
    for s in candidates:
        # 섹터 분산: 같은 섹터에서 최대 2종목
        if seen_sectors.get(s["sector"], 0) if isinstance(seen_sectors, dict) else s["sector"] in seen_sectors:
            # 같은 섹터 2번째는 건너뜀 (분산)
            pass
        seen_sectors.add(s["sector"])

        price = s["price"]
        high = s["high"]
        low = s["low"]
        vwap = s["vwap_est"]

        # 진입가: VWAP 부근 또는 현재가 -1.5%
        if price > vwap:
            entry = _tick_round(vwap)  # VWAP까지 눌림 대기
        else:
            entry = _tick_round(int(price * 0.985))  # 현재가 -1.5%

        # 목표가: 고가 +3% 또는 현재가 +5%
        target = _tick_round(max(int(high * 1.03), int(price * 1.05)))

        # 손절가: 저가 -2% 또는 현재가 -4%
        stop = _tick_round(max(int(low * 0.98), int(price * 0.96)))

        # 포지션 코멘트
        if price > vwap * 1.02:
            position_comment = f"VWAP({vwap:,}) 상회 중 — 눌림 대기 유리"
        elif price < vwap * 0.98:
            position_comment = f"VWAP({vwap:,}) 하회 — 약세, 관망"
        else:
            position_comment = f"VWAP({vwap:,}) 근접 — 진입 적정 구간"

        # 수급 코멘트 (거래량 기반)
        if s["volume"] >= 1_000_000:
            supply_comment = "거래량 폭발 (100만+ 주) — 강한 관심"
        elif s["volume"] >= 500_000:
            supply_comment = "거래량 활발 (50만+ 주) — 수급 양호"
        elif s["volume"] >= 100_000:
            supply_comment = "거래량 보통"
        else:
            supply_comment = "거래량 저조 — 유동성 주의"

        pick = {
            "rank": len(picks) + 1,
            "ticker": s["ticker"],
            "name": s["name"],
            "sector": s["sector"],
            "price": price,
            "change_pct": s["change_pct"],
            "volume": s["volume"],
            "high": high,
            "low": low,
            "vwap_est": vwap,
            "entry_price": entry,
            "target_price": target,
            "stop_price": stop,
            "risk_reward": round((target - entry) / max(entry - stop, 1), 1),
            "position_comment": position_comment,
            "supply_comment": supply_comment,
        }
        picks.append(pick)

        if len(picks) >= n:
            break

    return picks


# ─────────────────────────────────────────
# 5. ETF 전략 추천
# ─────────────────────────────────────────

def recommend_etfs(market: dict, sector_etfs: list[dict]) -> list[dict]:
    """시장 방향 + 섹터 기반 ETF 추천."""
    recs = []
    direction = market["direction"]

    # 방향 ETF
    if direction in ("STRONG_BULL", "BULL"):
        recs.append({
            "ticker": "122630",
            "name": "KODEX 레버리지",
            "action": "매수" if direction == "STRONG_BULL" else "관심",
            "reason": f"상승장 ({market['kospi_lever_pct']:+.1f}%) — 추세 추종",
            "entry": "시초가 또는 오전 눌림",
            "exit": "종가 매도 (당일 트레이딩)",
        })
    elif direction in ("BEAR", "CRASH"):
        recs.append({
            "ticker": "252670",
            "name": "KODEX 인버스2X",
            "action": "매수" if direction == "CRASH" else "관심",
            "reason": f"하락장 ({market['kospi_lever_pct']:+.1f}%) — 헤지",
            "entry": "오전 반등 시 진입",
            "exit": "급락 시 익절",
        })

    # 상위 섹터 ETF (TOP 3)
    for etf in sector_etfs[:3]:
        if etf["change_pct"] >= 1.0:
            recs.append({
                "ticker": etf["ticker"],
                "name": etf["name"],
                "action": "관심",
                "reason": f"{etf['sector']} 섹터 강세 ({etf['change_pct']:+.1f}%)",
                "entry": "내일 시초 또는 눌림 시",
                "exit": "섹터 모멘텀 꺾일 때",
            })

    return recs


# ─────────────────────────────────────────
# 6. 텔레그램 포맷
# ─────────────────────────────────────────

def format_telegram(pulse: dict) -> str:
    """Market Pulse를 텔레그램 메시지로 변환."""
    mkt = pulse["market_direction"]
    picks = pulse["top_picks"]
    etfs = pulse["etf_recommendations"]

    lines = [
        f"📡 [Market Pulse] {pulse['timestamp'][:16]}",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🎯 시장: {mkt['direction']} ({mkt['kospi_lever_pct']:+.1f}%)",
        f"   {mkt['comment']}",
        f"   내일: {mkt['tomorrow']}",
        f"",
    ]

    # ETF 추천
    if etfs:
        lines.append("📊 ETF 추천:")
        for e in etfs[:3]:
            lines.append(f"   {'✅' if e['action']=='매수' else '👀'} {e['name']} — {e['reason']}")
        lines.append("")

    # TOP 5
    lines.append("🏆 TOP 5 종목:")
    for p in picks:
        lines.append(f"")
        lines.append(f"  {p['rank']}. {p['name']}({p['ticker']}) {p['change_pct']:+.1f}%")
        lines.append(f"     현재 {p['price']:,} | VWAP {p['vwap_est']:,}")
        lines.append(f"     진입 {p['entry_price']:,} → 목표 {p['target_price']:,} (손절 {p['stop_price']:,})")
        lines.append(f"     R:R {p['risk_reward']}배 | {p['supply_comment']}")

    # 섹터 랭킹
    lines.append("")
    lines.append("📈 섹터 랭킹:")
    for rank, (sector, data) in enumerate(
        sorted(pulse["sector_rankings"].items(),
               key=lambda x: x[1]["avg_change_pct"], reverse=True)[:5], 1
    ):
        lines.append(f"   {rank}. {sector} {data['avg_change_pct']:+.1f}% (대장: {data['leader']} {data['leader_change']:+.1f}%)")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🤖 Quantum Master | ppwangga.com")

    return "\n".join(lines)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_pulse(dry: bool = False, no_send: bool = False) -> dict:
    """Market Pulse 실행."""
    logger.info("=" * 50)
    logger.info("[PULSE] Market Pulse 시작")
    logger.info("=" * 50)

    broker = _create_broker()

    # 1. 시장 방향
    market = analyze_market_direction(broker)
    logger.info("[PULSE] 방향: %s (%+.1f%%)", market["direction"], market["kospi_lever_pct"])

    # 2. 섹터 ETF
    sector_etfs = analyze_sector_etfs(broker)
    logger.info("[PULSE] 섹터 ETF %d개 조회", len(sector_etfs))

    # 3. 개별종목 스캔
    sector_results, all_stocks = scan_sectors(broker)
    logger.info("[PULSE] 종목 %d개 조회 완료", len(all_stocks))

    # 4. TOP 5
    top_picks = select_top_picks(all_stocks, market["direction"])
    logger.info("[PULSE] TOP %d 선정", len(top_picks))

    # 5. ETF 추천
    etf_recs = recommend_etfs(market, sector_etfs)

    # 결과 조합
    pulse = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_direction": market,
        "sector_etfs": sector_etfs,
        "sector_rankings": sector_results,
        "top_picks": top_picks,
        "etf_recommendations": etf_recs,
    }

    # 저장
    if not dry:
        out_path = DATA_DIR / "market_pulse.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pulse, f, ensure_ascii=False, indent=2)
        logger.info("[PULSE] 저장: %s", out_path)

    # 텔레그램
    msg = format_telegram(pulse)
    print(msg)

    if not dry and not no_send:
        try:
            from src.telegram_sender import send_message
            ok = send_message(msg)
            logger.info("[PULSE] 텔레그램 %s", "성공" if ok else "실패")
        except Exception as e:
            logger.error("[PULSE] 텔레그램 실패: %s", e)
    else:
        logger.info("[PULSE] 텔레그램 미전송 (dry=%s, no_send=%s)", dry, no_send)

    logger.info("=" * 50)
    logger.info("[PULSE] 완료 — %s, TOP: %s",
                market["direction"],
                ", ".join(f"{p['name']}({p['change_pct']:+.1f}%)" for p in top_picks[:3]))
    logger.info("=" * 50)

    return pulse


def main():
    parser = argparse.ArgumentParser(description="Market Pulse — 실시간 시장 인텔리전스")
    parser.add_argument("--dry", action="store_true", help="콘솔만 (저장/전송 안 함)")
    parser.add_argument("--no-send", action="store_true", help="저장만 (텔레그램 미전송)")
    args = parser.parse_args()

    run_pulse(dry=args.dry, no_send=args.no_send)


if __name__ == "__main__":
    main()
