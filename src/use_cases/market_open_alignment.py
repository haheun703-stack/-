"""MARKET_OPEN_REGIME → FLOWX tier 보조 재심사 레이어 (관측 전용).

단타봇 build_market_open_regime이 생성하는 data_store/quant_market_regime.json을
graceful 로드해, 후보 섹터 ↔ 시장 주도축(leading/avoid_themes) 정렬을 평가한다.

★중요(사장님 지시): tier 자동 변경 없음. classify_tier(SSOT)는 그대로 두고,
재심사 라벨/점수만 부착한다.
  - RECHECK_CONTROL_TO_WATCH : CONTROL인데 주도축(leading)과 정렬 → WATCH 승격 재심사
  - CORE_WEAK_ALIGNMENT_RECHECK : CORE인데 주도축과 약정렬(회피축이거나 주도축 밖) → 재검토
ETF/US/EWY 수급은 tier 확정 신호가 아니라 "이 후보를 다시 봐야 한다"는 경고/가점이다.

graceful:
  - json 없거나 깨짐 → status=unavailable, action=None (현행 tier 유지)
  - freshness.ok == false → status=stale, action=None (정렬 보류, stale 경고만)
  - 정상 → status=ok, alignment_score + market_alignment_action

ETF 수급 단독으로 tier를 확정하지 않는다 — action은 '재심사' 신호일 뿐,
실제 tier 결정은 classify_tier(후보 가격/수급/브레드스 기반)가 SSOT다.
실주문/스케줄러/SAJANG/C60 무관 — 순수 분석.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MARKET_REGIME_PATH = PROJECT_ROOT / "data_store" / "quant_market_regime.json"

VERSION = "market_open_alignment_v1"

ACTION_RECHECK_CONTROL = "RECHECK_CONTROL_TO_WATCH"
ACTION_CORE_WEAK = "CORE_WEAK_ALIGNMENT_RECHECK"

STATUS_UNAVAILABLE = "unavailable"
STATUS_STALE = "stale"
STATUS_OK = "ok"

LEADING_BONUS = 2.0
AVOID_PENALTY = 2.0
WEIGHT_ALIGNMENT_MIN = 1.20

THEME_ALIASES: dict[str, tuple[str, ...]] = {
    "semiconductor": ("semiconductor", "semi", "chip", "hbm", "반도체", "ai반도체"),
    "battery": ("battery", "2차전지", "2차", "전지", "배터리", "ev"),
    "bio": ("bio", "health", "pharma", "바이오", "제약"),
    "it": ("it", "software", "platform", "전기전자", "전자"),
    "shipbuilding": ("shipbuilding", "ship", "조선", "lng"),
    "defense": ("defense", "방산", "aero", "항공"),
    "auto": ("auto", "car", "자동차"),
    "construction": ("construction", "건설"),
    "insurance": ("insurance", "보험"),
    "metals": ("metals", "metal", "steel", "금속", "철강"),
    "gold": ("gold", "금선물", "금현물", "귀금속"),
    "oil": ("oil", "energy", "원유", "에너지"),
}


def load_market_regime(path: Path = MARKET_REGIME_PATH) -> dict[str, Any] | None:
    """quant_market_regime.json graceful 로드. 없거나 깨지면 None(현행 tier 유지)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace(" ", "").replace("_", "")


def _raw_theme_tokens(value: Any) -> list[str]:
    """Return schema-flexible theme tokens.

    단타봇 산출물은 leading_themes가 문자열 리스트일 수도 있고,
    {key,label,score,...} dict 리스트일 수도 있다. 둘 다 받는다.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        keys = ("key", "label", "theme", "sector", "name", "alias")
        return [_norm(value.get(k)) for k in keys if value.get(k)]
    return [_norm(value)]


def _expand_aliases(token: str) -> set[str]:
    if not token:
        return set()
    out = {token}
    for key, aliases in THEME_ALIASES.items():
        key_n = _norm(key)
        alias_set = {_norm(a) for a in aliases}
        # sector='AI반도체'처럼 alias가 포함된 경우도 canonical set으로 확장.
        if token == key_n or token in alias_set or any(a and a in token for a in alias_set):
            out.add(key_n)
            out.update(alias_set)
    return {x for x in out if x}


def _theme_token_set(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.update(_theme_token_set(item))
        return tokens
    for raw in _raw_theme_tokens(value):
        tokens.update(_expand_aliases(raw))
    return tokens


def _token_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    # Avoid short-token accidents, e.g. "it" inside "battery".
    return (len(a) >= 3 and a in b) or (len(b) >= 3 and b in a)


def _theme_match(sector: Any, themes: Any) -> bool:
    """섹터가 테마 목록과 정렬되는지(대소문자 무시 + 양방향 부분 포함).

    예: sector='AI반도체' ↔ theme='반도체' / sector='semiconductor' ↔ theme='semiconductor'.
    """
    if not sector or not themes:
        return False
    sector_tokens = _theme_token_set(sector)
    theme_tokens = _theme_token_set(themes)
    if not sector_tokens or not theme_tokens:
        return False
    return any(_token_match(s, t) for s in sector_tokens for t in theme_tokens)


def _sector_weight_bonus(sector: Any, weights: Any) -> float:
    """sector_weights에 섹터가 매칭되면 그 가중치를 가점(있을 때만)."""
    if not sector or not isinstance(weights, dict):
        return 0.0
    for key, val in weights.items():
        if _theme_match(sector, [key]):
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _sector_weight_aligned(sector: Any, weights: Any) -> bool:
    return _sector_weight_bonus(sector, weights) >= WEIGHT_ALIGNMENT_MIN


def assess_alignment(
    tier: str | None, sector: Any, market_regime: dict[str, Any] | None
) -> dict[str, Any]:
    """후보 tier+섹터 ↔ 시장 주도축 정렬 평가. ★tier 변경 없음 — 재심사 라벨/점수만.

    반환 키:
      status(ok/stale/unavailable), alignment_score, in_leading, in_avoid,
      market_alignment_action, market_bias, note
    """
    base: dict[str, Any] = {
        "status": STATUS_UNAVAILABLE,
        "alignment_score": None,
        "in_leading": None,
        "in_avoid": None,
        "market_alignment_action": None,
        "market_bias": None,
        "note": None,
    }

    if not market_regime:
        base["note"] = "quant_market_regime.json 없음 → 현행 tier 유지"
        return base

    fresh = market_regime.get("freshness") or {}
    if fresh.get("ok") is False:
        base["status"] = STATUS_STALE
        base["market_bias"] = market_regime.get("market_bias")
        base["note"] = "freshness.ok=false → 정렬 보류(stale), tier 자동변경 없음"
        return base

    leading = market_regime.get("leading_themes") or []
    avoid = market_regime.get("avoid_themes") or []
    weights = market_regime.get("sector_weights") or {}

    in_leading = _theme_match(sector, leading)
    in_avoid = _theme_match(sector, avoid)
    weight_aligned = _sector_weight_aligned(sector, weights)

    score = 0.0
    if in_leading or weight_aligned:
        score += LEADING_BONUS
    if in_avoid:
        score -= AVOID_PENALTY
    score += _sector_weight_bonus(sector, weights)

    # ★재심사 라벨(tier 변경 아님). ETF/수급 정렬은 '다시 봐라' 신호일 뿐.
    action: str | None = None
    if tier == "CONTROL" and (in_leading or weight_aligned) and not in_avoid:
        action = ACTION_RECHECK_CONTROL
    elif tier == "CORE" and (in_avoid or not (in_leading or weight_aligned)):
        action = ACTION_CORE_WEAK

    return {
        "status": STATUS_OK,
        "alignment_score": round(score, 2),
        "in_leading": in_leading,
        "in_avoid": in_avoid,
        "weight_aligned": weight_aligned,
        "market_alignment_action": action,
        "market_bias": market_regime.get("market_bias"),
        "note": None,
    }


def regime_summary(market_regime: dict[str, Any] | None) -> dict[str, Any]:
    """plan/SHOW ME에 박을 시장 레짐 요약 메타(관측 표시용)."""
    if not market_regime:
        return {"available": False, "status": STATUS_UNAVAILABLE}
    fresh = market_regime.get("freshness") or {}
    return {
        "available": True,
        "status": STATUS_STALE if fresh.get("ok") is False else STATUS_OK,
        "market_bias": market_regime.get("market_bias"),
        "etf_dominant": market_regime.get("etf_dominant"),
        "leading_themes": market_regime.get("leading_themes") or [],
        "avoid_themes": market_regime.get("avoid_themes") or [],
        "freshness_ok": fresh.get("ok"),
    }
