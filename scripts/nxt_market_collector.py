"""
NXT(넥스트레이드) 프리/애프터 마켓 데이터 수집기

수집 시간대:
  - 프리마켓 (--session pre):  08:00~08:55 / 5분 간격
  - 애프터마켓 (--session after): 15:35~19:55 / 10분 간격
  - 테스트 (--test): 현재 시각 기준 1회만 수집

KIS API의 fetch_price()는 NXT 시간에 호출하면 NXT 데이터를 반환.
KisIntradayAdapter._api_get() 재활용.

출력:
  data/nxt/nxt_{session}_{date}.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _load_nxt_tickers() -> list[str]:
    """NXT 마스터에서 모니터링 대상 종목 로드"""
    master_path = PROJECT_ROOT / "data" / "nxt" / "nxt_master.json"
    if not master_path.exists():
        logger.warning("[NXT수집] nxt_master.json 없음 → 유니버스 84종목 사용")
        # fallback: data/processed/*.parquet
        tickers = []
        proc_dir = PROJECT_ROOT / "data" / "processed"
        if proc_dir.exists():
            for f in sorted(proc_dir.glob("*.parquet")):
                t = f.stem
                if len(t) == 6 and t.isdigit():
                    tickers.append(t)
        return tickers

    data = json.loads(master_path.read_text(encoding="utf-8"))
    # 우리 유니버스 교차 종목만 (전체 800개 수집은 불필요)
    universe = set()
    proc_dir = PROJECT_ROOT / "data" / "processed"
    if proc_dir.exists():
        for f in proc_dir.glob("*.parquet"):
            t = f.stem
            if len(t) == 6 and t.isdigit():
                universe.add(t)

    nxt_set = set(data.get("tickers", []))

    # 유니버스 & NXT 교차
    if universe:
        tickers = sorted(universe & nxt_set) if nxt_set else sorted(universe)
    else:
        tickers = sorted(nxt_set)[:100]  # 최대 100종목

    logger.info("[NXT수집] 모니터링 대상: %d종목", len(tickers))
    return tickers


def _create_broker():
    """KIS 브로커 인스턴스 생성"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    import mojito
    is_mock = os.getenv("MODEL") != "REAL"
    broker = mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY", ""),
        api_secret=os.getenv("KIS_APP_SECRET", ""),
        acc_no=os.getenv("KIS_ACC_NO", ""),
        mock=is_mock,
    )
    return broker


def _fetch_ticker_snapshot(broker, ticker: str) -> dict | None:
    """단일 종목 현재가 스냅샷 수집"""
    try:
        data = broker.fetch_price(ticker)
        output = data.get("output", {})
        price = int(output.get("stck_prpr", 0))
        if price == 0:
            return None

        return {
            "price": price,
            "open": int(output.get("stck_oprc", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "change_pct": float(output.get("prdy_ctrt", 0)),
            "prev_close": int(output.get("stck_sdpr", 0)),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        logger.debug("[NXT수집] %s 조회 실패: %s", ticker, e)
        return None


def _fetch_orderbook(broker, ticker: str) -> dict | None:
    """호가 데이터 수집 (매수/매도 체결량 추정용)"""
    try:
        # KIS 호가 API: uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn
        from src.adapters.kis_intraday_adapter import _rate_limit
        _rate_limit()

        url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": broker.access_token,
            "appKey": broker.api_key,
            "appSecret": broker.api_secret,
            "tr_id": "FHKST01010200",
        }
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}

        import requests
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        output = data.get("output1", {})
        if not output:
            return None

        # 총 매도잔량 vs 총 매수잔량
        total_ask = int(output.get("total_askp_rsqn", 0))
        total_bid = int(output.get("total_bidp_rsqn", 0))

        return {
            "total_ask_volume": total_ask,
            "total_bid_volume": total_bid,
            "bid_ask_ratio": round(total_bid / total_ask, 3) if total_ask > 0 else 0,
        }
    except Exception as e:
        logger.debug("[NXT수집] %s 호가 조회 실패: %s", ticker, e)
        return None


def collect_session(session: str, test_mode: bool = False) -> dict:
    """
    한 세션(pre/after) 데이터 수집.

    test_mode=True면 1회만 수집하고 종료.
    """
    from src.adapters.kis_intraday_adapter import _rate_limit

    tickers = _load_nxt_tickers()
    if not tickers:
        print("[ERROR] 수집 대상 종목 없음")
        return {}

    broker = _create_broker()
    today = date.today().isoformat()

    # 세션별 설정
    if session == "pre":
        interval_sec = 300      # 5분
        start_hhmm = "0800"
        end_hhmm = "0855"
        session_label = "프리마켓"
    elif session == "after":
        interval_sec = 600      # 10분
        start_hhmm = "1535"
        end_hhmm = "1955"
        session_label = "애프터마켓"
    else:
        print(f"[ERROR] 지원하지 않는 세션: {session}")
        return {}

    output_path = PROJECT_ROOT / "data" / "nxt" / f"nxt_{session}_{today}.json"

    # 기존 데이터 로드 (추가 수집)
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "date": today,
            "session": session,
            "snapshots": [],       # 시간별 스냅샷 리스트
            "summary": {},         # 최종 요약 (세션 종료 후)
        }

    print(f"\n=== NXT {session_label} 수집 시작 ===")
    print(f"  대상: {len(tickers)}종목")
    print(f"  시간: {start_hhmm}~{end_hhmm}")
    print(f"  간격: {interval_sec}초")

    if test_mode:
        print("  [TEST MODE] 1회 수집 후 종료")

    round_num = len(result.get("snapshots", []))

    while True:
        now = datetime.now()
        now_hhmm = now.strftime("%H%M")

        # 시간 체크 (test_mode면 무시)
        if not test_mode:
            if now_hhmm < start_hhmm:
                wait = _calc_wait_seconds(now, start_hhmm)
                print(f"  {session_label} 시작 대기: {wait}초...")
                time.sleep(min(wait, 60))
                continue
            if now_hhmm > end_hhmm:
                print(f"  {session_label} 종료 시각 도달")
                break

        round_num += 1
        ts = now.strftime("%H:%M:%S")
        print(f"\n[Round {round_num}] {ts} 수집 시작...")

        snapshot = {
            "timestamp": ts,
            "tickers": {},
        }

        collected = 0
        for ticker in tickers:
            _rate_limit()
            tick = _fetch_ticker_snapshot(broker, ticker)
            if tick and tick["price"] > 0:
                # 호가 추가 (선택사항, rate limit 고려)
                if collected < 20:  # 상위 20종목만 호가 수집
                    ob = _fetch_orderbook(broker, ticker)
                    if ob:
                        tick.update(ob)

                snapshot["tickers"][ticker] = tick
                collected += 1

        result["snapshots"].append(snapshot)
        print(f"  수집: {collected}/{len(tickers)}종목 (가격 > 0)")

        # 저장 (매 라운드)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if test_mode:
            break

        # 다음 라운드 대기
        elapsed = time.time() - now.timestamp()
        sleep_time = max(0, interval_sec - elapsed)
        if sleep_time > 0:
            print(f"  다음 수집까지 {sleep_time:.0f}초 대기...")
            time.sleep(sleep_time)

    # 세션 종료 → 요약 생성
    result["summary"] = _build_summary(result, session)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== NXT {session_label} 수집 완료 ===")
    print(f"  라운드: {round_num}")
    print(f"  저장: {output_path}")

    return result


def _build_summary(result: dict, session: str) -> dict:
    """수집된 스냅샷에서 종목별 요약 생성"""
    snapshots = result.get("snapshots", [])
    if not snapshots:
        return {}

    # 모든 종목 수집
    all_tickers = set()
    for snap in snapshots:
        all_tickers.update(snap.get("tickers", {}).keys())

    summary = {}
    for ticker in sorted(all_tickers):
        prices = []
        volumes = []
        bid_volumes = []
        ask_volumes = []

        for snap in snapshots:
            td = snap.get("tickers", {}).get(ticker)
            if not td:
                continue
            prices.append(td["price"])
            volumes.append(td.get("volume", 0))
            if "total_bid_volume" in td:
                bid_volumes.append(td["total_bid_volume"])
                ask_volumes.append(td["total_ask_volume"])

        if not prices:
            continue

        first_price = prices[0]
        last_price = prices[-1]
        prev_close = 0
        # 첫 스냅샷에서 전일종가 가져오기
        for snap in snapshots:
            td = snap.get("tickers", {}).get(ticker)
            if td and td.get("prev_close", 0) > 0:
                prev_close = td["prev_close"]
                break

        last_vol = volumes[-1] if volumes else 0

        summary[ticker] = {
            "first_price": first_price,
            "last_price": last_price,
            "high": max(prices),
            "low": min(prices),
            "volume": last_vol,
            "prev_close": prev_close,
            "gap_pct": round((first_price / prev_close - 1) * 100, 2) if prev_close > 0 else 0,
            "session_change_pct": round((last_price / first_price - 1) * 100, 2) if first_price > 0 else 0,
        }

        # 수급 추정 (호가 기반)
        if bid_volumes and ask_volumes:
            avg_bid = sum(bid_volumes) / len(bid_volumes)
            avg_ask = sum(ask_volumes) / len(ask_volumes)
            summary[ticker]["avg_bid_volume"] = int(avg_bid)
            summary[ticker]["avg_ask_volume"] = int(avg_ask)
            summary[ticker]["net_buy_ratio"] = round(avg_bid / (avg_bid + avg_ask), 3) if (avg_bid + avg_ask) > 0 else 0.5

    return summary


def _calc_wait_seconds(now: datetime, target_hhmm: str) -> int:
    """현재시각 → 목표시각까지 초"""
    h, m = int(target_hhmm[:2]), int(target_hhmm[2:])
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    diff = (target - now).total_seconds()
    return max(0, int(diff))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="NXT 프리/애프터 마켓 데이터 수집")
    parser.add_argument("--session", required=True, choices=["pre", "after"],
                        help="수집 세션: pre(프리마켓) / after(애프터마켓)")
    parser.add_argument("--test", action="store_true",
                        help="테스트 모드: 1회 수집 후 종료")
    args = parser.parse_args()

    collect_session(session=args.session, test_mode=args.test)
