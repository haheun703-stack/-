"""섹터 모멘텀 → FLOWX 후보 관측 라벨 (관측 전용).

sector_composite.json(섹터별 STRONG_ROTATION..EXODUS regime) + stock_to_sector.json
(종목→섹터 리스트)을 graceful 로드해, 후보 종목이 '강세 섹터'에 있는지 '약세/이탈
섹터'에 있는지 관측 라벨만 부착한다.

★중요(사장님 6/10 지시 — 올바른 개선 ①단계):
  classify_tier(SSOT) 무변경, 6/12 KEEP/DROP/TUNE 판정 무오염.
  - 이 라벨은 tier 선정·hard gate·진입·주문 어디에도 분기로 쓰지 않는다(관측뿐).
  - 6/11·12 daily_review에서 "강세섹터 종목 vs 약세섹터 종목" 성과를 누적 비교만 한다.
  - 검증(약세섹터 노출이 엇갈림을 키우는가)되면 6/12 판정 후 사장님 승인 하에
    classify_tier 보조 필터로 정식 연결(hard gate 금지, 보조점수로만).

설계 동형: src/use_cases/market_open_alignment.py (MARKET_OPEN_REGIME 보조 레이어와
동일한 graceful·tier-불변·shadow 부착 패턴).

graceful:
  - sector_composite 없거나 깨짐 → status=unavailable (라벨 없음, plan 생성은 계속)
  - 종목이 어느 섹터에도 매핑 안 됨 → status=unmapped
  - momentum_date != as_of_date → status=stale (라벨은 부착하되 stale 경고)
  - 정상 → status=ok

실주문/스케줄러/SAJANG/C60/tier 무관 — 순수 분석.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SECTOR_COMPOSITE_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "sector_composite.json"
STOCK_TO_SECTOR_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "stock_to_sector.json"

VERSION = "sector_momentum_label_v1"

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNMAPPED = "unmapped"

# sector_composite regime → 사람이 읽는 관측 라벨(매수 신호 아님)
REGIME_TO_LABEL = {
    "STRONG_ROTATION": "STRONG_SECTOR",
    "MODERATE_ROTATION": "MODERATE_SECTOR",
    "NEUTRAL": "NEUTRAL_SECTOR",
    "WEAK_ROTATION": "WEAK_SECTOR",
    "EXODUS": "EXODUS_SECTOR",
}
DEFAULT_LABEL = "NEUTRAL_SECTOR"

# 광범위 분류축(특정 섹터 강약과 무관) — 종목↔섹터 매칭에서 제외
BROAD_BUCKETS = {"KRX300", "KOSPI200", "KOSPI", "KRX", "KRX100"}


def load_sector_composite(path: Path = SECTOR_COMPOSITE_PATH) -> dict[str, Any] | None:
    """sector_composite.json graceful 로드. 없거나 깨지면 None(라벨 없음, plan 계속)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_stock_to_sector(path: Path = STOCK_TO_SECTOR_PATH) -> dict[str, list]:
    """stock_to_sector.json graceful 로드. 없거나 깨지면 {}(전 종목 unmapped degrade)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sector_regime_map(sector_composite: dict) -> dict[str, dict]:
    """sector_composite.sectors → {섹터명: {regime, composite_score}}."""
    out: dict[str, dict] = {}
    for s in sector_composite.get("sectors") or []:
        name = s.get("sector")
        if not name:
            continue
        out[str(name)] = {
            "regime": s.get("regime"),
            "composite_score": s.get("composite_score"),
        }
    return out


def assess_sector_momentum(
    ticker: str | None,
    sector_composite: dict[str, Any] | None,
    stock_to_sector: dict[str, list] | None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """후보 종목 ↔ 섹터 모멘텀 정렬 평가. ★tier 변경 없음 — 관측 라벨만.

    종목이 여러 섹터에 속하면(예: '증권'+'금융') composite_score 최고를 best(대표),
    최저를 worst(최악 노출)로 둘 다 기록한다. 6/12에 best/worst 중 어느 축이 성과
    예측력이 좋은지 비교하기 위함(사후비교 — 미리 한쪽을 정답으로 정하지 않는다).

    반환 키:
      status(ok/stale/unmapped/unavailable), momentum_date, matched_sectors,
      best_sector/best_regime/best_score, worst_sector/worst_regime/worst_score,
      in_strong_sector, in_exodus_sector, sector_strength_label(best 기준), note
    """
    base: dict[str, Any] = {
        "status": STATUS_UNAVAILABLE,
        "ticker": ticker,
        "momentum_date": None,
        "matched_sectors": [],
        "best_sector": None,
        "best_regime": None,
        "best_score": None,
        "worst_sector": None,
        "worst_regime": None,
        "worst_score": None,
        "in_strong_sector": None,
        "in_exodus_sector": None,
        "sector_strength_label": None,
        "note": None,
    }

    if not sector_composite or not isinstance(sector_composite, dict):
        base["note"] = "sector_composite.json 없음 → 라벨 없음(현행 유지)"
        return base

    momentum_date = sector_composite.get("momentum_date")
    base["momentum_date"] = momentum_date

    regime_map = _sector_regime_map(sector_composite)
    if not regime_map:
        base["note"] = "sector_composite.sectors 비어있음"
        return base

    s2s = stock_to_sector or {}
    stock_sectors = [
        str(x) for x in (s2s.get(str(ticker)) or []) if str(x) not in BROAD_BUCKETS
    ]

    matched: list[dict] = []
    for sec in stock_sectors:
        info = regime_map.get(sec)
        if info and info.get("composite_score") is not None:
            matched.append({
                "sector": sec,
                "regime": info.get("regime"),
                "composite_score": info.get("composite_score"),
            })

    if not matched:
        base["status"] = STATUS_UNMAPPED
        base["momentum_date"] = momentum_date
        base["note"] = "종목이 sector_composite 섹터에 매핑 안 됨(관측 제외)"
        return base

    matched.sort(key=lambda m: m["composite_score"], reverse=True)
    best, worst = matched[0], matched[-1]

    strong = {str(x) for x in (sector_composite.get("strong_sectors") or [])}
    exodus = {str(x) for x in (sector_composite.get("exodus_sectors") or [])}
    in_strong = any(m["sector"] in strong for m in matched)
    in_exodus = any(m["sector"] in exodus for m in matched)

    label = REGIME_TO_LABEL.get(best["regime"], DEFAULT_LABEL)

    status = STATUS_OK
    note = None
    if as_of_date and momentum_date and str(momentum_date) != str(as_of_date):
        status = STATUS_STALE
        note = f"momentum_date={momentum_date} != as_of={as_of_date} (stale, 관측만)"

    return {
        "status": status,
        "ticker": ticker,
        "momentum_date": momentum_date,
        "matched_sectors": matched,
        "best_sector": best["sector"],
        "best_regime": best["regime"],
        "best_score": best["composite_score"],
        "worst_sector": worst["sector"],
        "worst_regime": worst["regime"],
        "worst_score": worst["composite_score"],
        "in_strong_sector": in_strong,
        "in_exodus_sector": in_exodus,
        "sector_strength_label": label,
        "note": note,
    }


def sector_momentum_summary(sector_composite: dict[str, Any] | None) -> dict[str, Any]:
    """plan/SHOW ME에 박을 섹터 모멘텀 요약 메타(관측 표시용)."""
    if not sector_composite or not isinstance(sector_composite, dict):
        return {"available": False, "status": STATUS_UNAVAILABLE}
    return {
        "available": True,
        "status": STATUS_OK,
        "momentum_date": sector_composite.get("momentum_date"),
        "strong_sectors": sector_composite.get("strong_sectors") or [],
        "exodus_sectors": sector_composite.get("exodus_sectors") or [],
        "regime_summary": sector_composite.get("regime_summary") or {},
        "sector_count": sector_composite.get("sector_count"),
    }
