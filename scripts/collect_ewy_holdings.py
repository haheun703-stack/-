#!/usr/bin/env python3
"""EWY(iShares MSCI South Korea ETF) 보유종목 수집 + 비중 변화 분석.

외인 패시브 자금 흐름 선행 감지:
- 비중 증가 종목 = 패시브 외인 매수 유입 → 선취매 후보
- 비중 감소 종목 = 패시브 외인 매도 압력 → 매수 회피
- 신규 편입 = 강제 매수 발생 → 단기 수급 폭탄
- 편출 = 강제 매도 → 급락 리스크

실행:
    python scripts/collect_ewy_holdings.py
    python scripts/collect_ewy_holdings.py --upload   # Supabase 업로드 포함
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EWY_URL = (
    "https://www.ishares.com/us/products/239681/"
    "ishares-msci-south-korea-etf/1467271812596.ajax"
    "?fileType=csv&fileName=EWY_holdings&dataType=fund"
)
DATA_DIR = PROJECT_ROOT / "data" / "ewy"
UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe.csv"

# 비중 변동 분류 기준
LARGE_THRESHOLD = 0.30   # 0.3%p 이상
MEDIUM_THRESHOLD = 0.10  # 0.1%p 이상
STABLE_THRESHOLD = 0.05  # 0.05%p 미만 → 안정


# ─────────────────────────────────────────────
# 1. 한글 종목명 매핑
# ─────────────────────────────────────────────

def load_name_map() -> dict[str, str]:
    """universe.csv에서 ticker→한글명 매핑 로드."""
    name_map = {}
    if UNIVERSE_CSV.exists():
        with open(UNIVERSE_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name_map[row["ticker"]] = row["name"]
    logger.info("[EWY] 종목명 매핑: %d종목 로드", len(name_map))
    return name_map


# ─────────────────────────────────────────────
# 2. CSV 다운로드 + 파싱
# ─────────────────────────────────────────────

def download_ewy_csv() -> str:
    """iShares 공식 CSV 다운로드."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(EWY_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    logger.info("[EWY] CSV 다운로드 완료: %d chars", len(resp.text))
    return resp.text


def parse_ewy_csv(raw_csv: str, name_map: dict[str, str]) -> dict:
    """CSV 파싱 → 구조화된 dict 반환.

    Returns:
        {
            "as_of": "May 01, 2026",
            "holdings": [
                {"code": "000660", "name": "SK하이닉스", "name_en": "SK HYNIX INC",
                 "weight": 22.78, "sector": "Information Technology",
                 "quantity": 5382737, "market_value": 4667071050.43},
                ...
            ]
        }
    """
    lines = raw_csv.strip().split("\n")

    # 메타데이터에서 기준일 추출
    as_of = ""
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Fund Holdings as of"):
            parts = line.split(",", 1)
            if len(parts) >= 2:
                as_of = parts[1].strip().strip('"')
        if line.startswith("Ticker,Name"):
            header_idx = i
            break

    if header_idx < 0:
        logger.error("[EWY] CSV 헤더를 찾을 수 없음")
        return {"as_of": as_of, "holdings": []}

    # 데이터 행 파싱
    data_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(data_text))

    holdings = []
    for row in reader:
        ticker = (row.get("Ticker") or "").strip().strip('"')
        asset_class = (row.get("Asset Class") or "").strip().strip('"')

        # 주식만 필터 (현금, 선물, 기타 제외)
        if asset_class != "Equity":
            continue
        # 6자리 숫자 종목코드만
        if not ticker or not ticker.isdigit() or len(ticker) != 6:
            continue

        weight_str = (row.get("Weight (%)") or "0").strip().strip('"')
        quantity_str = (row.get("Quantity") or "0").strip().strip('"').replace(",", "")
        mv_str = (row.get("Market Value") or "0").strip().strip('"').replace(",", "")

        try:
            weight = float(weight_str)
            quantity = int(float(quantity_str))
            market_value = float(mv_str)
        except (ValueError, TypeError):
            continue

        name_en = (row.get("Name") or "").strip().strip('"')
        name_kr = name_map.get(ticker, name_en)
        sector = (row.get("Sector") or "").strip().strip('"')

        holdings.append({
            "code": ticker,
            "name": name_kr,
            "name_en": name_en,
            "weight": round(weight, 2),
            "sector": sector,
            "quantity": quantity,
            "market_value": round(market_value, 2),
        })

    # 비중 내림차순 정렬
    holdings.sort(key=lambda x: x["weight"], reverse=True)

    # 순위 부여
    for rank, h in enumerate(holdings, 1):
        h["rank"] = rank

    logger.info("[EWY] 파싱 완료: %d종목 (기준일: %s)", len(holdings), as_of)
    return {"as_of": as_of, "holdings": holdings}


# ─────────────────────────────────────────────
# 3. 전일 비교 → 변동 분석
# ─────────────────────────────────────────────

def load_previous(data_dir: Path) -> dict | None:
    """가장 최근 저장 데이터 로드 (비교용)."""
    prev_path = data_dir / "ewy_holdings_prev.json"
    if prev_path.exists():
        with open(prev_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def classify_change(delta: float) -> tuple[str, str]:
    """비중 변화 → (direction, magnitude) 분류."""
    abs_d = abs(delta)
    if abs_d < STABLE_THRESHOLD:
        return ("STABLE", "STABLE")
    direction = "UP" if delta > 0 else "DOWN"
    if abs_d >= LARGE_THRESHOLD:
        magnitude = "LARGE"
    elif abs_d >= MEDIUM_THRESHOLD:
        magnitude = "MEDIUM"
    else:
        magnitude = "SMALL"
    return (direction, magnitude)


def analyze_changes(
    current: list[dict], previous: list[dict] | None
) -> dict:
    """전일 대비 비중 변화, 편입/편출 분석.

    Returns:
        {
            "changes": [...],       # 비중 변동 (MEDIUM 이상만)
            "new_entries": [...],   # 신규 편입
            "removed": [...],      # 편출
        }
    """
    if not previous:
        return {"changes": [], "new_entries": [], "removed": []}

    prev_map = {h["code"]: h for h in previous}
    curr_map = {h["code"]: h for h in current}

    prev_codes = set(prev_map.keys())
    curr_codes = set(curr_map.keys())

    # 신규 편입
    new_entries = []
    for code in curr_codes - prev_codes:
        h = curr_map[code]
        new_entries.append({
            "code": code,
            "name": h["name"],
            "weight": h["weight"],
            "sector": h["sector"],
            "impact": "패시브 강제매수 예상",
        })
    new_entries.sort(key=lambda x: x["weight"], reverse=True)

    # 편출
    removed = []
    for code in prev_codes - curr_codes:
        h = prev_map[code]
        removed.append({
            "code": code,
            "name": h["name"],
            "weight": h.get("weight", 0),
            "sector": h.get("sector", ""),
            "impact": "패시브 강제매도 예상",
        })
    removed.sort(key=lambda x: x["weight"], reverse=True)

    # 비중 변동 (양쪽 모두 존재하는 종목)
    changes = []
    for code in curr_codes & prev_codes:
        c = curr_map[code]
        p = prev_map[code]
        delta = round(c["weight"] - p["weight"], 2)
        direction, magnitude = classify_change(delta)

        if magnitude in ("LARGE", "MEDIUM"):
            changes.append({
                "code": code,
                "name": c["name"],
                "weight": c["weight"],
                "weight_prev": p["weight"],
                "weight_change": delta,
                "direction": direction,
                "magnitude": magnitude,
            })
    changes.sort(key=lambda x: abs(x["weight_change"]), reverse=True)

    return {
        "changes": changes,
        "new_entries": new_entries,
        "removed": removed,
    }


# ─────────────────────────────────────────────
# 4. 저장 + 요약
# ─────────────────────────────────────────────

def build_summary(
    holdings: list[dict],
    changes: list[dict],
    new_entries: list[dict],
    removed: list[dict],
) -> str:
    """FLOWX 표시용 요약 문자열 생성."""
    parts = []

    # TOP 3 종목
    if holdings:
        top3 = [f"{h['name']} {h['weight']}%" for h in holdings[:3]]
        parts.append(f"TOP3: {', '.join(top3)}")

    # 주요 변동
    up_large = [c for c in changes if c["direction"] == "UP" and c["magnitude"] == "LARGE"]
    dn_large = [c for c in changes if c["direction"] == "DOWN" and c["magnitude"] == "LARGE"]
    if up_large:
        names = ", ".join(c["name"] for c in up_large[:3])
        parts.append(f"비중 급증: {names}")
    if dn_large:
        names = ", ".join(c["name"] for c in dn_large[:3])
        parts.append(f"비중 급감: {names}")

    # 편입/편출
    if new_entries:
        names = ", ".join(e["name"] for e in new_entries[:3])
        parts.append(f"신규편입: {names}")
    if removed:
        names = ", ".join(e["name"] for e in removed[:3])
        parts.append(f"편출: {names}")

    if not parts:
        parts.append("주요 변동 없음")

    # 섹터 비중 계산
    sector_weights: dict[str, float] = {}
    for h in holdings:
        sec = h["sector"]
        sector_weights[sec] = sector_weights.get(sec, 0) + h["weight"]
    top_sector = max(sector_weights, key=sector_weights.get) if sector_weights else ""
    if top_sector:
        parts.append(f"최대 섹터: {top_sector} {sector_weights[top_sector]:.1f}%")

    return " | ".join(parts)


def save_results(
    data_dir: Path,
    date_str: str,
    as_of: str,
    holdings: list[dict],
    analysis: dict,
    summary: str,
) -> Path:
    """JSON 저장 + prev 파일 갱신."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # top20 with prev weight
    prev_data = load_previous(data_dir)
    prev_map = {}
    if prev_data and "holdings" in prev_data:
        prev_map = {h["code"]: h for h in prev_data["holdings"]}

    top20 = []
    for h in holdings[:20]:
        prev = prev_map.get(h["code"], {})
        weight_prev = prev.get("weight", 0)
        weight_change = round(h["weight"] - weight_prev, 2) if prev else 0
        direction, _ = classify_change(weight_change) if prev else ("NEW", "NEW")

        top20.append({
            "rank": h["rank"],
            "code": h["code"],
            "name": h["name"],
            "name_en": h["name_en"],
            "weight": h["weight"],
            "weight_prev": weight_prev,
            "weight_change": weight_change,
            "quantity": h["quantity"],
            "sector": h["sector"],
            "signal": direction,
        })

    result = {
        "date": date_str,
        "as_of": as_of,
        "total_stocks": len(holdings),
        "top20": top20,
        "changes": analysis["changes"],
        "new_entries": analysis["new_entries"],
        "removed": analysis["removed"],
        "summary": summary,
        "holdings": holdings,  # 전체 보유종목 (로컬 보관용)
    }

    # 날짜별 파일 저장
    out_path = data_dir / f"ewy_holdings_{date_str.replace('-', '')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("[EWY] 저장: %s", out_path)

    # prev 파일 갱신 (다음 비교용)
    prev_path = data_dir / "ewy_holdings_prev.json"
    prev_save = {
        "date": date_str,
        "as_of": as_of,
        "holdings": holdings,
    }
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(prev_save, f, ensure_ascii=False, indent=2)

    return out_path


# ─────────────────────────────────────────────
# 5. Supabase 업로드
# ─────────────────────────────────────────────

def upload_to_supabase(date_str: str, as_of: str, total_stocks: int,
                       top20: list, changes: list, new_entries: list,
                       removed: list, summary: str, *,
                       monthly_summary: dict | None = None) -> bool:
    """quant_ewy_holdings 테이블에 UPSERT."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.warning("[EWY] SUPABASE_URL/KEY 미설정 — 업로드 스킵")
        return False

    try:
        from supabase import create_client
        client = create_client(url, key)
    except Exception as e:
        logger.error("[EWY] Supabase 연결 실패: %s", e)
        return False

    row = {
        "date": date_str,
        "as_of": as_of,
        "total_stocks": total_stocks,
        "top20": top20,
        "changes": changes,
        "new_entries": new_entries,
        "removed": removed,
        "summary": summary,
    }
    if monthly_summary:
        row["monthly_perf"] = monthly_summary

    try:
        result = client.table("quant_ewy_holdings").upsert(
            [row], on_conflict="date"
        ).execute()
        if result.data:
            logger.info("[EWY] Supabase 업로드 완료: %s (%d종목)", date_str, total_stocks)
            return True
        else:
            logger.warning("[EWY] Supabase 업로드 응답 비어있음")
            return False
    except Exception as e:
        logger.error("[EWY] Supabase 업로드 실패: %s", e)
        return False


# ─────────────────────────────────────────────
# 6. 멀티 월별 수익률 비교 (최근 3개월)
# ─────────────────────────────────────────────

def calc_multi_month_returns(
    holdings: list[dict],
    date_str: str,
    num_months: int = 3,
    prev_holdings: list[dict] | None = None,
) -> dict:
    """EWY 보유종목의 최근 N개월 수익률 비교.

    pykrx로 전체 기간 OHLCV를 한 번에 조회 후 월별 분리.
    Returns: {
        "months": [{"key": "2026-03", "label": "3월"}, ...],
        "stocks": [{"rank","code","name","weight","close","weight_change",
                     "returns":{"2026-03":-5.2, "2026-04":12.3, ...}}, ...],
        "summary": {"2026-03": {"avg":..,"wavg":..,"up":..,"dn":..,"total":..}, ...}
    }
    """
    try:
        from pykrx import stock as pykrx_mod
        import pandas as pd
    except ImportError:
        logger.warning("[EWY] pykrx/pandas 미설치 — 월별 수익률 스킵")
        return {}

    import calendar

    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # 전일 비중 매핑 (weight_change 계산용)
    prev_map = {}
    if prev_holdings:
        prev_map = {h["code"]: h.get("weight", 0) for h in prev_holdings}

    # 월 경계 계산
    months_info = []
    for i in range(num_months - 1, -1, -1):
        m = dt.month - i
        y = dt.year
        while m <= 0:
            m += 12
            y -= 1
        m_key = f"{y:04d}-{m:02d}"
        m_label = f"{m}월" + ("(MTD)" if i == 0 else "")
        m_start = datetime(y, m, 1)
        if i == 0:
            m_end = dt
        else:
            last_day = calendar.monthrange(y, m)[1]
            m_end = datetime(y, m, last_day)
        months_info.append({
            "key": m_key, "label": m_label,
            "start": m_start, "end": m_end,
        })

    # 전체 조회 범위
    range_start = months_info[0]["start"].strftime("%Y%m%d")
    range_end = months_info[-1]["end"].strftime("%Y%m%d")

    logger.info(
        "[EWY] 멀티월 수익률: %s ~ %s (%d개월, %d종목)",
        range_start, range_end, num_months, len(holdings),
    )

    # per-stock 타임아웃 (pykrx hang 방지)
    PER_STOCK_TIMEOUT = 30  # 초

    def _fetch_ohlcv_with_timeout(pykrx_fn, start, end, ticker, timeout_sec):
        """pykrx 호출에 타임아웃을 적용 (Linux signal / Windows threading)."""
        import platform
        if platform.system() != "Windows":
            import signal as _sig

            def _handler(signum, frame):
                raise TimeoutError(f"{ticker} pykrx 타임아웃 ({timeout_sec}s)")

            old = _sig.signal(_sig.SIGALRM, _handler)
            _sig.alarm(timeout_sec)
            try:
                result = pykrx_fn(start, end, ticker)
            finally:
                _sig.alarm(0)
                _sig.signal(_sig.SIGALRM, old)
            return result
        else:
            # Windows: threading 기반 폴백
            import threading
            result_box = [None]
            exc_box = [None]

            def _worker():
                try:
                    result_box[0] = pykrx_fn(start, end, ticker)
                except Exception as e:
                    exc_box[0] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=timeout_sec)
            if t.is_alive():
                raise TimeoutError(f"{ticker} pykrx 타임아웃 ({timeout_sec}s)")
            if exc_box[0]:
                raise exc_box[0]
            return result_box[0]

    stocks_result = []
    ok_count = 0
    skip_count = 0
    for idx, h in enumerate(holdings):
        ticker = h["code"]
        w_prev = prev_map.get(ticker, 0)
        w_change = round(h["weight"] - w_prev, 2) if w_prev else 0.0

        stock_entry = {
            "rank": h.get("rank", idx + 1),
            "code": ticker,
            "name": h["name"],
            "weight": h["weight"],
            "weight_change": w_change,
            "close": 0,
            "sector": h.get("sector", ""),
            "returns": {},
        }

        try:
            df = _fetch_ohlcv_with_timeout(
                pykrx_mod.get_market_ohlcv,
                range_start, range_end, ticker,
                PER_STOCK_TIMEOUT,
            )
            if df.empty:
                stocks_result.append(stock_entry)
                skip_count += 1
                time.sleep(0.03)
                continue

            # 월별 분리
            for m in months_info:
                m_start_ts = pd.Timestamp(m["start"])
                m_end_ts = pd.Timestamp(m["end"])
                mask = (df.index >= m_start_ts) & (df.index <= m_end_ts)
                m_df = df[mask]

                if m_df.empty or len(m_df) < 1:
                    stock_entry["returns"][m["key"]] = None
                    continue

                open_p = int(m_df.iloc[0]["종가"])
                close_p = int(m_df.iloc[-1]["종가"])
                if open_p == 0:
                    stock_entry["returns"][m["key"]] = None
                    continue

                stock_entry["returns"][m["key"]] = round(
                    (close_p - open_p) / open_p * 100, 2
                )

            # 최신 종가
            stock_entry["close"] = int(df.iloc[-1]["종가"])
            ok_count += 1

        except TimeoutError:
            logger.warning("[EWY] %s 타임아웃 (%ds) — 스킵", ticker, PER_STOCK_TIMEOUT)
            skip_count += 1
        except Exception as e:
            logger.debug("[EWY] %s 조회 실패: %s", ticker, e)
            skip_count += 1

        stocks_result.append(stock_entry)
        time.sleep(0.05)

        # 진행 로그 (20종목마다)
        if (idx + 1) % 20 == 0:
            logger.info("[EWY] 멀티월 진행: %d/%d (성공 %d, 스킵 %d)",
                        idx + 1, len(holdings), ok_count, skip_count)

    # 비중순 정렬
    stocks_result.sort(key=lambda x: x["weight"], reverse=True)

    # 월별 요약 통계
    summary = {}
    for m in months_info:
        key = m["key"]
        valid = [s for s in stocks_result if s["returns"].get(key) is not None]
        if valid:
            avg = round(sum(s["returns"][key] for s in valid) / len(valid), 2)
            total_w = sum(s["weight"] for s in valid)
            wavg = (
                round(sum(s["weight"] * s["returns"][key] for s in valid) / total_w, 2)
                if total_w > 0 else 0.0
            )
            up = len([s for s in valid if s["returns"][key] > 0])
            dn = len([s for s in valid if s["returns"][key] < 0])
            total = len(valid)
        else:
            avg, wavg, up, dn, total = 0.0, 0.0, 0, 0, 0
        summary[key] = {"avg": avg, "wavg": wavg, "up": up, "dn": dn, "total": total}

    logger.info(
        "[EWY] 멀티월 수익률 완료: %d종목 × %d개월", len(stocks_result), num_months
    )

    return {
        "months": [{"key": m["key"], "label": m["label"]} for m in months_info],
        "stocks": stocks_result,
        "summary": summary,
    }


# ─────────────────────────────────────────────
# 7. 텔레그램 보고
# ─────────────────────────────────────────────

def send_telegram_report(date_str: str, as_of: str, holdings: list[dict],
                         analysis: dict) -> None:
    """EWY 비중 변동 텔레그램 보고."""
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return

    lines = [f"[EWY] MSCI Korea 보유종목 ({as_of})"]
    lines.append(f"보유 {len(holdings)}종목\n")

    # TOP 10
    lines.append("--- TOP 10 ---")
    for h in holdings[:10]:
        arrow = ""
        if h.get("rank", 0) <= 10:
            # prev 비교는 간략히
            lines.append(f"{h['rank']:2d}. {h['name']} {h['weight']}%")

    # 주요 변동
    if analysis["changes"]:
        lines.append("\n--- 주요 변동 ---")
        for c in analysis["changes"][:5]:
            arrow = "+" if c["weight_change"] > 0 else ""
            lines.append(
                f"{'UP' if c['direction']=='UP' else 'DN'} {c['name']} "
                f"{c['weight']}% ({arrow}{c['weight_change']}%p)"
            )

    if analysis["new_entries"]:
        lines.append("\n--- 신규 편입 ---")
        for e in analysis["new_entries"]:
            lines.append(f"NEW {e['name']} {e['weight']}%")

    if analysis["removed"]:
        lines.append("\n--- 편출 ---")
        for e in analysis["removed"]:
            lines.append(f"OUT {e['name']} {e['weight']}%")

    text = "\n".join(lines)

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        logger.info("[EWY] 텔레그램 보고 완료")
    except Exception as e:
        logger.warning("[EWY] 텔레그램 전송 실패: %s", e)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EWY 보유종목 수집")
    parser.add_argument("--upload", action="store_true", help="Supabase 업로드")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 보고")
    parser.add_argument("--monthly", action="store_true", help="월별 수익률 계산")
    parser.add_argument("--date", type=str, default=None, help="기준일 (YYYY-MM-DD)")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info("[EWY] === 수집 시작: %s ===", date_str)

    # 1. 종목명 매핑 로드
    name_map = load_name_map()

    # 2. CSV 다운로드 + 파싱
    raw_csv = download_ewy_csv()
    parsed = parse_ewy_csv(raw_csv, name_map)
    holdings = parsed["holdings"]
    as_of = parsed["as_of"]

    if not holdings:
        logger.error("[EWY] 파싱 결과 0종목 — 중단")
        return

    # 3. 전일 비교
    prev_data = load_previous(DATA_DIR)
    prev_holdings = prev_data["holdings"] if prev_data else None
    analysis = analyze_changes(holdings, prev_holdings)

    logger.info(
        "[EWY] 분석: %d종목, 변동 %d건, 편입 %d건, 편출 %d건",
        len(holdings),
        len(analysis["changes"]),
        len(analysis["new_entries"]),
        len(analysis["removed"]),
    )

    # 4. 요약 생성
    summary = build_summary(
        holdings, analysis["changes"],
        analysis["new_entries"], analysis["removed"]
    )

    # 5. 저장
    out_path = save_results(
        DATA_DIR, date_str, as_of, holdings, analysis, summary
    )

    # 6. 콘솔 리포트
    print(f"\n{'='*60}")
    print(f"  EWY MSCI Korea 보유종목 ({as_of})")
    print(f"  보유 종목: {len(holdings)}개")
    print(f"{'='*60}")
    print("\n  [TOP 10]")
    for h in holdings[:10]:
        print(f"  {h['rank']:2d}. {h['name']:<12s} {h['weight']:6.2f}%  ({h['sector']})")

    if analysis["changes"]:
        print(f"\n  [주요 비중 변동 — 0.1%p 이상]")
        for c in analysis["changes"][:10]:
            arrow = "+" if c["weight_change"] > 0 else ""
            tag = "UP" if c["direction"] == "UP" else "DN"
            mag = f"[{c['magnitude']}]"
            print(
                f"  {tag} {c['name']:<12s} {c['weight']:6.2f}% "
                f"({arrow}{c['weight_change']:+.2f}%p) {mag}"
            )

    if analysis["new_entries"]:
        print(f"\n  [신규 편입]")
        for e in analysis["new_entries"]:
            print(f"  NEW {e['name']:<12s} {e['weight']:6.2f}%  {e['impact']}")

    if analysis["removed"]:
        print(f"\n  [편출]")
        for e in analysis["removed"]:
            print(f"  OUT {e['name']:<12s} {e['weight']:6.2f}%  {e['impact']}")

    print(f"\n  요약: {summary}")
    print(f"  저장: {out_path}")
    print(f"{'='*60}\n")

    # 7. 멀티 월별 수익률 비교 (최근 3개월)
    monthly_data = {}
    if args.monthly:
        logger.info("[EWY] 멀티월 수익률 계산 시작 (최근 3개월)...")
        monthly_data = calc_multi_month_returns(
            holdings, date_str, num_months=3, prev_holdings=prev_holdings,
        )

        if monthly_data and monthly_data.get("months"):
            # 콘솔 리포트 — 월별 요약
            print(f"\n{'='*74}")
            months_labels = [m["label"] for m in monthly_data["months"]]
            print(f"  월별 수익률 비교: {' / '.join(months_labels)}")
            for m in monthly_data["months"]:
                s = monthly_data["summary"].get(m["key"], {})
                print(
                    f"  {m['label']:>10s}: "
                    f"상승 {s.get('up',0):>2d} / 하락 {s.get('dn',0):>2d} "
                    f"| 단순 {s.get('avg',0):+.2f}% "
                    f"/ 비중가중 {s.get('wavg',0):+.2f}%"
                )
            print(f"{'='*74}")

            # TOP 30 테이블 (섹터발화 스타일)
            month_keys = [m["key"] for m in monthly_data["months"]]
            hdr = f"  {'#':>3s} {'종목':<12s} {'비중':>6s} {'변동':>6s} {'종가':>10s}"
            for mk in month_keys:
                hdr += f" {mk[-2:]+'월':>8s}"
            print(hdr)
            print(f"  {'-'*72}")

            for i, s in enumerate(monthly_data["stocks"][:30], 1):
                wc = s.get("weight_change", 0)
                wc_str = f"{wc:+.2f}" if wc else "   -"
                close_str = f"{s['close']:>10,d}" if s["close"] else "       N/A"
                line = (
                    f"  {i:>3d} {s['name']:<12s} "
                    f"{s['weight']:>5.2f}% {wc_str:>6s} {close_str}"
                )
                for mk in month_keys:
                    ret = s["returns"].get(mk)
                    if ret is not None:
                        line += f" {ret:>+7.2f}%"
                    else:
                        line += "      N/A"
                print(line)
            print()

        # 결과 JSON에 추가 저장
        result_data = json.loads(out_path.read_text(encoding="utf-8"))
        result_data["monthly_perf"] = monthly_data
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 8. Supabase 업로드
    if args.upload:
        result_for_upload = json.loads(out_path.read_text(encoding="utf-8"))
        upload_to_supabase(
            date_str, as_of, len(holdings),
            result_for_upload["top20"],
            result_for_upload["changes"],
            result_for_upload["new_entries"],
            result_for_upload["removed"],
            summary,
            monthly_summary=monthly_data or None,
        )

    # 8. 텔레그램
    if args.telegram:
        send_telegram_report(date_str, as_of, holdings, analysis)

    logger.info("[EWY] === 완료 ===")


if __name__ == "__main__":
    main()
