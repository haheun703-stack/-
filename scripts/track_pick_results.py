"""
추천 종목 성과 추적 (Picks Performance Tracker)

매일 17시 스케줄 실행 (scan_tomorrow_picks.py 이후):
  1단계: 오늘의 추천 → 히스토리에 아카이브
  2단계: 이전 추천 중 target_date == 오늘 → 실제 OHLCV 기록 → 성과 판정
  3단계: 아직 "보유중" 상태 → 최신 종가로 업데이트
  4단계: 전체 통계 재계산 → picks_history.json 저장

출력: data/picks_history.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORY_PATH = DATA_DIR / "picks_history.json"
PICKS_PATH = DATA_DIR / "tomorrow_picks.json"


def _sf(v) -> float:
    """NaN-safe float"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return round(float(v), 2)


def load_history() -> dict:
    """히스토리 로드 또는 초기화"""
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"records": [], "summary": {}}


def save_history(history: dict):
    """히스토리 저장"""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[저장] {HISTORY_PATH}")


def get_ohlcv(ticker: str, target_date: str) -> dict | None:
    """parquet에서 특정 날짜의 OHLCV 가져오기"""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path)
        # 날짜 인덱스 처리
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        target_dt = pd.Timestamp(target_date)
        if target_dt in df.index:
            row = df.loc[target_dt]
            return {
                "open": _sf(row.get("open", 0)),
                "high": _sf(row.get("high", 0)),
                "low": _sf(row.get("low", 0)),
                "close": _sf(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            }

        # 정확한 날짜가 없으면 ±1일 허용 (공휴일 등)
        nearby = df.loc[(df.index >= target_dt - timedelta(days=1)) &
                        (df.index <= target_dt + timedelta(days=1))]
        if len(nearby) > 0:
            row = nearby.iloc[-1]
            return {
                "open": _sf(row.get("open", 0)),
                "high": _sf(row.get("high", 0)),
                "low": _sf(row.get("low", 0)),
                "close": _sf(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            }
        return None
    except Exception as e:
        logger.warning("OHLCV 조회 실패 %s(%s): %s", ticker, target_date, e)
        return None


def get_latest_close(ticker: str) -> float:
    """최신 종가 가져오기"""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return 0
    try:
        df = pd.read_parquet(pq_path)
        return float(df.iloc[-1]["close"])
    except Exception:
        return 0


def judge_result(rec: dict, ohlcv: dict) -> dict:
    """1일차 성과 판정

    - 시가/종가/고가/저가 기록
    - 목표가 도달 여부 (고가 기준)
    - 손절가 도달 여부 (저가 기준)
    - 손절+목표 동시 히트 → 손절 우선 (보수적)
    """
    entry = rec.get("entry_price", 0) or rec.get("close_at_pick", 0)
    stop = rec.get("stop_loss", 0)
    target = rec.get("target_price", 0)

    open_p = ohlcv["open"]
    close_p = ohlcv["close"]
    high_p = ohlcv["high"]
    low_p = ohlcv["low"]

    rec["open_price"] = open_p
    rec["close_price"] = close_p
    rec["high_price"] = high_p
    rec["low_price"] = low_p

    # 수익률 (진입가 대비 종가)
    if entry > 0:
        rec["day1_return"] = round((close_p / entry - 1) * 100, 2)
    else:
        rec["day1_return"] = 0

    # 상태 판정
    hit_stop = stop > 0 and low_p <= stop
    hit_target = target > 0 and high_p >= target

    if hit_stop and hit_target:
        rec["status"] = "hit_stop"  # 보수적: 손절 우선
        rec["settled_price"] = stop
        rec["settled_return"] = round((stop / entry - 1) * 100, 2) if entry > 0 else 0
    elif hit_stop:
        rec["status"] = "hit_stop"
        rec["settled_price"] = stop
        rec["settled_return"] = round((stop / entry - 1) * 100, 2) if entry > 0 else 0
    elif hit_target:
        rec["status"] = "hit_target"
        rec["settled_price"] = target
        rec["settled_return"] = round((target / entry - 1) * 100, 2) if entry > 0 else 0
    else:
        rec["status"] = "holding"
        rec["settled_price"] = close_p
        rec["settled_return"] = rec["day1_return"]

    rec["settled_date"] = rec["target_date"]
    return rec


def update_holding(rec: dict) -> dict:
    """보유중 상태 → 최신 종가로 업데이트 (최대 5거래일)"""
    entry = rec.get("entry_price", 0) or rec.get("close_at_pick", 0)
    if entry <= 0:
        return rec

    latest = get_latest_close(rec["ticker"])
    if latest <= 0:
        return rec

    rec["latest_price"] = latest
    rec["latest_return"] = round((latest / entry - 1) * 100, 2)

    # 최신 종가로 목표/손절 체크
    stop = rec.get("stop_loss", 0)
    target = rec.get("target_price", 0)

    if stop > 0 and latest <= stop:
        rec["status"] = "hit_stop"
        rec["settled_price"] = stop
        rec["settled_return"] = round((stop / entry - 1) * 100, 2)
    elif target > 0 and latest >= target:
        rec["status"] = "hit_target"
        rec["settled_price"] = target
        rec["settled_return"] = round((target / entry - 1) * 100, 2)
    else:
        # 5거래일 경과 → 만기 처리
        pick_date = datetime.strptime(rec["pick_date"], "%Y-%m-%d")
        days_held = (datetime.now() - pick_date).days
        if days_held > 7:  # 약 5거래일
            rec["status"] = "expired"
            rec["settled_price"] = latest
            rec["settled_return"] = rec["latest_return"]

    return rec


def calc_summary(records: list[dict]) -> dict:
    """전체 통계 계산"""
    settled = [r for r in records if r.get("status") in ("hit_target", "hit_stop", "expired")]
    holding = [r for r in records if r.get("status") == "holding"]
    pending = [r for r in records if r.get("status") == "pending"]

    wins = [r for r in settled if (r.get("settled_return", 0) or 0) > 0]
    losses = [r for r in settled if (r.get("settled_return", 0) or 0) <= 0]

    total_settled = len(settled)
    win_rate = round(len(wins) / total_settled * 100, 1) if total_settled > 0 else 0
    avg_return = round(np.mean([r.get("settled_return", 0) or 0 for r in settled]), 2) if settled else 0
    avg_win = round(np.mean([r.get("settled_return", 0) or 0 for r in wins]), 2) if wins else 0
    avg_loss = round(np.mean([r.get("settled_return", 0) or 0 for r in losses]), 2) if losses else 0

    # PF = 총이익 / 총손실 (건수 반영)
    total_gain = sum(r.get("settled_return", 0) or 0 for r in wins)
    total_loss = abs(sum(r.get("settled_return", 0) or 0 for r in losses))

    return {
        "total_picks": len(records),
        "total_settled": total_settled,
        "hit_target": len([r for r in settled if r["status"] == "hit_target"]),
        "hit_stop": len([r for r in settled if r["status"] == "hit_stop"]),
        "expired": len([r for r in settled if r["status"] == "expired"]),
        "holding": len(holding),
        "pending": len(pending),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": round(total_gain / total_loss, 2) if total_loss > 0 else 0,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    history = load_history()
    records = history.get("records", [])

    today = datetime.now().strftime("%Y-%m-%d")

    # ── 1단계: 오늘의 추천을 히스토리에 아카이브 ──
    if PICKS_PATH.exists():
        with open(PICKS_PATH, encoding="utf-8") as f:
            picks_data = json.load(f)

        pick_date = picks_data.get("generated_at", "")[:10]  # YYYY-MM-DD
        target_date = picks_data.get("target_date", "")

        # 이미 아카이브된 pick_date인지 확인
        archived_dates = {r["pick_date"] for r in records}

        if pick_date and pick_date not in archived_dates:
            picks = picks_data.get("picks", [])
            # TOP5만 아카이브 (top5 배열 있으면 해당 종목만, 없으면 등급 기반)
            top5_tickers = set(picks_data.get("top5", []))
            if top5_tickers:
                top_picks = [p for p in picks if p["ticker"] in top5_tickers]
            else:
                top_picks = [p for p in picks if p.get("grade") in ("적극매수", "매수", "관심매수")][:5]

            new_count = 0
            for p in top_picks:
                rec = {
                    "pick_date": pick_date,
                    "target_date": target_date,
                    "ticker": p["ticker"],
                    "name": p["name"],
                    "grade": p["grade"],
                    "score": p["total_score"],
                    "n_sources": p.get("n_sources", 1),
                    "sources": p.get("sources", []),
                    "close_at_pick": p.get("close", 0),
                    "entry_price": p.get("entry_price", 0),
                    "stop_loss": p.get("stop_loss", 0),
                    "target_price": p.get("target_price", 0),
                    "entry_condition": p.get("entry_condition", ""),
                    "reasons": p.get("reasons", []),
                    "rsi": p.get("rsi", 0),
                    "stoch_k": p.get("stoch_k", 0),
                    # 결과 필드 (추후 기록)
                    "open_price": None,
                    "close_price": None,
                    "high_price": None,
                    "low_price": None,
                    "day1_return": None,
                    "status": "pending",
                    "settled_price": None,
                    "settled_return": None,
                    "settled_date": None,
                    "latest_price": None,
                    "latest_return": None,
                }
                records.append(rec)
                new_count += 1

            print(f"[아카이브] {pick_date} → {target_date} 추천 {new_count}건 저장")
        else:
            print(f"[아카이브] {pick_date} 이미 저장됨 (스킵)")
    else:
        print("[아카이브] tomorrow_picks.json 없음")

    # ── 2단계: target_date == 오늘인 추천 → 실제 결과 기록 ──
    updated_count = 0
    for rec in records:
        if rec["status"] != "pending":
            continue
        if rec["target_date"] != today:
            continue

        ohlcv = get_ohlcv(rec["ticker"], today)
        if ohlcv is None:
            # 공휴일이면 다음 영업일까지 대기
            continue

        rec = judge_result(rec, ohlcv)
        updated_count += 1
        status_emoji = {"hit_target": "🎯", "hit_stop": "🛑", "holding": "📊"}.get(rec["status"], "❓")
        print(f"  {status_emoji} {rec['name']}({rec['ticker']}) "
              f"진입:{rec['entry_price']:,} → 종가:{rec['close_price']:,} "
              f"수익:{rec['day1_return']:+.1f}% [{rec['status']}]")

    if updated_count > 0:
        print(f"[결과] {today} 대상 {updated_count}건 판정 완료")
    else:
        pending_today = [r for r in records if r["status"] == "pending" and r["target_date"] == today]
        if pending_today:
            print(f"[결과] {today} 대상 {len(pending_today)}건 — parquet 미업데이트 (장중이거나 공휴일)")
        else:
            print(f"[결과] {today} 대상 추천 없음")

    # ── 3단계: 보유중 → 최신 종가 업데이트 ──
    holding_count = 0
    for rec in records:
        if rec["status"] != "holding":
            continue
        rec = update_holding(rec)
        holding_count += 1

    if holding_count > 0:
        print(f"[보유] {holding_count}건 최신 종가 업데이트")

    # ── 4단계: 통계 재계산 ──
    summary = calc_summary(records)
    history["records"] = records
    history["summary"] = summary

    save_history(history)

    # 요약 출력
    print(f"\n{'='*50}")
    print(f"[전체 성과] 총 {summary['total_picks']}건")
    print(f"  완료: {summary['total_settled']}건 (목표달성:{summary['hit_target']} "
          f"손절:{summary['hit_stop']} 만기:{summary['expired']})")
    print(f"  보유중: {summary['holding']}건  대기: {summary['pending']}건")
    if summary["total_settled"] > 0:
        print(f"  승률: {summary['win_rate']}%  평균수익: {summary['avg_return']:+.2f}%")
        print(f"  평균이익: {summary['avg_win']:+.2f}%  평균손실: {summary['avg_loss']:+.2f}%")
        if summary["profit_factor"] > 0:
            print(f"  PF: {summary['profit_factor']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
