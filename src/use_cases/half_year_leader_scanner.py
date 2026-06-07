"""FLOWX Market OS v1 관측 레이어 — 반기 주도주 스캐너 (지시서 2단계).

반기 첫 거래일 시가(1~6월=1월 첫날, 7~12월=7월 첫날)를 기준으로 "반기 주도주"
후보를 점수화·분류만 한다. 매수 신호가 아니라 관측 라벨이다 — C60 hard gate 무변경,
전부 shadow / label.

점수(총 100, 단순 시작):
  +30 반기 시가 위 / +20 반기 신고가 근처 / +20 최근 20거래일 신고가 갱신 /
  +20 같은 섹터 동조 3개 이상 / +10 KOSPI 대비 상대강도 양수
분류: CORE>=80 / WATCH 60~79 / WEAK 40~59 / NOT_LEADER <40

★주문/매도/스케줄러/SAJANG 경로 import·호출 0. 입력 OHLCV·KOSPI·섹터맵, 출력 점수뿐.

설계: 진행 지시서 "주봉/반기/시가축 관측 레이어" §2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etf.c60_shadow import normalize_ohlcv
from src.use_cases.price_axis_regime import NEAR_HIGH_RATIO, NEW_HIGH_WINDOW, _pct

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KOSPI_INDEX_PATH = PROJECT_ROOT / "data" / "kospi_index.csv"
SECTOR_FIRE_MAP_PATH = PROJECT_ROOT / "config" / "sector_fire_map.yaml"

SCANNER_VERSION = "half_year_leader_scanner_v1"

# 점수 배점
PTS_ABOVE_HALF_OPEN = 30
PTS_NEAR_HALF_HIGH = 20
PTS_NEW_HIGH_20D = 20
PTS_SECTOR_SYNC = 20
PTS_RS_POSITIVE = 10
SECTOR_SYNC_MIN = 3

# 분류 임계
CORE_MIN = 80
WATCH_MIN = 60
WEAK_MIN = 40


def _current_half_mask(df: pd.DataFrame, last_ts) -> pd.Series:
    is_h1 = last_ts.month <= 6
    return (df.index.year == last_ts.year) & (
        (df.index.month <= 6) if is_h1 else (df.index.month >= 7)
    )


def kospi_half_return(kospi_df: pd.DataFrame | None, last_ts) -> float | None:
    """현재 반기 KOSPI 수익률(반기 첫날 시가→마지막 종가). 데이터 없으면 None."""
    if kospi_df is None or kospi_df.empty:
        return None
    seg = kospi_df.loc[_current_half_mask(kospi_df, last_ts)]
    if seg.empty:
        return None
    base = float(seg["open"].iloc[0]) if "open" in seg.columns else float(seg["close"].iloc[0])
    if base <= 0:
        return None
    return float(kospi_df["close"].iloc[-1]) / base - 1.0


def compute_half_year_metrics(df: pd.DataFrame, kospi_half_ret: float | None = None) -> dict:
    """단일 종목 OHLCV → 반기 주도주 원천 지표. 순수 함수."""
    if df is None or df.empty or "high" not in df.columns:
        return {"data_available": False}

    last_ts = df.index[-1]
    current_close = float(df["close"].iloc[-1])
    half_seg = df.loc[_current_half_mask(df, last_ts)]
    if half_seg.empty or "open" not in half_seg.columns:
        return {"data_available": False}

    half_year_open = float(half_seg["open"].iloc[0])
    half_year_high = float(half_seg["high"].max())
    if half_year_open <= 0:
        return {"data_available": False}

    above = current_close >= half_year_open
    near_high = half_year_high > 0 and current_close >= NEAR_HIGH_RATIO * half_year_high
    recent = df["close"].tail(NEW_HIGH_WINDOW)
    new_high_20d = bool(len(recent) >= 2 and current_close >= float(recent.max()))
    half_return = current_close / half_year_open - 1.0
    rs = None if kospi_half_ret is None else half_return - kospi_half_ret

    return {
        "data_available": True,
        "as_of_date": pd.Timestamp(last_ts).strftime("%Y-%m-%d"),
        "current_half": "H1" if last_ts.month <= 6 else "H2",
        "half_year_open": int(half_year_open),
        "current_close": int(current_close),
        "above_half_year_open": bool(above),
        "distance_from_half_year_open_pct": _pct(current_close, half_year_open),
        "half_year_high": int(half_year_high),
        "near_half_year_high": bool(near_high),
        "new_half_year_high_20d": new_high_20d,
        "half_year_return_pct": round(half_return * 100, 2),
        "rs_vs_kospi": None if rs is None else round(rs * 100, 2),
        "rs_positive": bool(rs is not None and rs > 0),
    }


def build_half_year_leader(
    ticker: str, name: str, metrics: dict, sector: str | None, sector_peer_sync_count: int
) -> dict:
    """반기 주도주 점수/분류. 순수 함수. metrics=compute_half_year_metrics 결과."""
    if not metrics.get("data_available"):
        return {
            "ticker": ticker, "name": name, "sector": sector,
            "data_available": False, "half_year_leader_score": 0,
            "half_year_leader_grade": "HY_NOT_LEADER",
        }

    score = 0
    if metrics["above_half_year_open"]:
        score += PTS_ABOVE_HALF_OPEN
    if metrics["near_half_year_high"]:
        score += PTS_NEAR_HALF_HIGH
    if metrics["new_half_year_high_20d"]:
        score += PTS_NEW_HIGH_20D
    if sector_peer_sync_count >= SECTOR_SYNC_MIN:
        score += PTS_SECTOR_SYNC
    if metrics["rs_positive"]:
        score += PTS_RS_POSITIVE

    if score >= CORE_MIN:
        grade = "HY_LEADER_CORE"
    elif score >= WATCH_MIN:
        grade = "HY_LEADER_WATCH"
    elif score >= WEAK_MIN:
        grade = "HY_LEADER_WEAK"
    else:
        grade = "HY_NOT_LEADER"

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "data_available": True,
        "half_year_open": metrics["half_year_open"],
        "current_close": metrics["current_close"],
        "above_half_year_open": metrics["above_half_year_open"],
        "distance_from_half_year_open_pct": metrics["distance_from_half_year_open_pct"],
        "half_year_high": metrics["half_year_high"],
        "near_half_year_high": metrics["near_half_year_high"],
        "new_half_year_high_20d": metrics["new_half_year_high_20d"],
        "sector_peer_sync_count": sector_peer_sync_count,
        "rs_vs_kospi": metrics["rs_vs_kospi"],
        "rs_positive": metrics["rs_positive"],
        "half_year_leader_score": score,
        "half_year_leader_grade": grade,
    }


def scan_half_year_leaders(
    items: list[dict], kospi_df: pd.DataFrame | None = None, sector_map: dict | None = None
) -> list[dict]:
    """여러 종목 → 반기 주도주 레코드 목록(점수 내림차순). 순수 함수.

    items: [{"ticker","name","df"(정규화 OHLCV),"sector"(선택)}].
    sector_peer_sync_count = 같은 섹터에서 반기 시가 위인 다른 후보 수(동조).
    """
    sector_map = sector_map or {}
    enriched = []
    for it in items:
        df = it.get("df")
        last_ts = df.index[-1] if df is not None and not df.empty else None
        khr = kospi_half_return(kospi_df, last_ts) if last_ts is not None else None
        metrics = compute_half_year_metrics(df, kospi_half_ret=khr)
        sector = it.get("sector") or sector_map.get(it.get("ticker"))
        enriched.append({"item": it, "sector": sector, "metrics": metrics})

    # 섹터별 "반기 시가 위" 카운트
    sector_above: dict[str, int] = {}
    for e in enriched:
        if e["sector"] and e["metrics"].get("above_half_year_open"):
            sector_above[e["sector"]] = sector_above.get(e["sector"], 0) + 1

    records = []
    for e in enriched:
        sector = e["sector"]
        sync = 0
        if sector and e["metrics"].get("above_half_year_open"):
            sync = max(0, sector_above.get(sector, 0) - 1)  # 자기 자신 제외
        records.append(
            build_half_year_leader(
                e["item"].get("ticker"), e["item"].get("name", e["item"].get("ticker")),
                e["metrics"], sector, sync,
            )
        )
    records.sort(key=lambda r: r["half_year_leader_score"], reverse=True)
    return records


def build_leader_board(records: list[dict], top_n: int = 20) -> list[dict]:
    """SHOW ME 표시용 TOP-N 보드(점수 내림차순, data_available만)."""
    avail = [r for r in records if r.get("data_available")]
    return avail[:top_n]


def load_kospi_index(path: Path = KOSPI_INDEX_PATH) -> pd.DataFrame | None:
    """IO: KOSPI 인덱스 CSV → 정규화 OHLCV. 실패 시 None(RS 미적용)."""
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        return normalize_ohlcv(df)
    except Exception:
        return None


def load_sector_map(path: Path = SECTOR_FIRE_MAP_PATH) -> dict:
    """IO: sector_fire_map.yaml → {ticker: 대표섹터}. 실패 시 빈 dict."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping: dict[str, str] = {}
        for sector_name, info in (data.get("sectors") or {}).items():
            for ticker in (info or {}).get("tickers", []) or []:
                mapping.setdefault(str(ticker), sector_name)  # 첫 섹터를 대표로
        return mapping
    except Exception:
        return {}
