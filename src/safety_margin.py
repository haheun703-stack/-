"""정상화 EPS 기반 안전마진 플래그 엔진.

버핏 철학: "확실한 것만으로 가격을 계산하고, 불확실한 것은 업사이드 보너스로 남긴다."
기존 100점 체계 변경 없음. 안전마진은 독립 플래그(GREEN/YELLOW/RED)로 운영.

데이터: consensus_screening.json (wisereport Forward EPS/PER/목표가)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONSENSUS_PATH = DATA_DIR / "consensus_screening.json"
HISTORY_DIR = DATA_DIR / "consensus_history"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


# ─── 결과 데이터 ───────────────────────────────────

@dataclass
class SafetyMarginResult:
    ticker: str = ""
    name: str = ""
    close: int = 0
    # Layer 1: 보수적 적정가
    forward_eps: float = 0.0
    conservative_per: float = 0.0
    conservative_value: int = 0       # EPS × 보수적PER
    margin_vs_conservative: float = 0.0
    # Layer 2: 애널리스트 할인
    target_price: int = 0
    discounted_target: int = 0        # target × 0.70
    margin_vs_target: float = 0.0
    # Layer 3: 이익 지속성
    eps_positive: bool = False
    analyst_sufficient: bool = False  # >= 3명
    opinion_sufficient: bool = False  # >= 3.5
    sustainability_pass: bool = False
    analyst_count: int = 0
    opinion_score: float = 0.0
    # 종합
    floor_price: int = 0              # min(보수적적정가, 할인목표가)
    floor_margin_pct: float = 0.0     # (바닥가/현재가-1)×100
    signal: str = ""                  # GREEN/YELLOW/RED/NO_DATA
    signal_label: str = ""
    reason: str = ""
    # 부속
    forward_per: float = 0.0
    forward_pbr: float = 0.0
    dividend_yield: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── 설정 로드 ─────────────────────────────────────

def _load_config() -> dict:
    """settings.yaml에서 safety_margin 설정 로드."""
    defaults = {
        "enabled": True,
        "per_discount_ratio": 0.70,
        "per_floor": 4.0,
        "per_cap": 12.0,
        "per_default": 7.0,
        "target_discount_ratio": 0.70,
        "min_analyst_count": 3,
        "min_opinion_score": 3.5,
        "yellow_threshold_pct": -20,
        "save_history": True,
    }
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        sm = cfg.get("safety_margin", {})
        for k, v in sm.items():
            if k in defaults:
                defaults[k] = v
    except Exception:
        pass
    return defaults


# ─── 컨센서스 풀 로드 ──────────────────────────────

def _load_consensus_pool() -> dict[str, dict]:
    """consensus_screening.json → ticker: {...} 매핑."""
    if not CONSENSUS_PATH.exists():
        logger.warning("[안전마진] consensus_screening.json 없음")
        return {}
    try:
        with open(CONSENSUS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        pool = {}
        for item in data.get("all_picks", []):
            t = item.get("ticker", "")
            if t:
                pool[t] = item
        return pool
    except Exception as e:
        logger.warning("[안전마진] 컨센서스 로드 실패: %s", e)
        return {}


# ─── 보수적 PER 산정 ──────────────────────────────

def _get_conservative_per(forward_per: float, cfg: dict) -> float:
    """Forward PER × 0.70, 클램프 4~12."""
    discount = cfg["per_discount_ratio"]
    floor = cfg["per_floor"]
    cap = cfg["per_cap"]
    default = cfg["per_default"]
    if forward_per <= 0:
        return default
    return max(floor, min(forward_per * discount, cap))


# ─── 메인 계산 ─────────────────────────────────────

def calc_safety_margin(
    ticker: str,
    name: str = "",
    close: int = 0,
    consensus: dict | None = None,
) -> SafetyMarginResult:
    """단일 종목 안전마진 판정.

    consensus: 외부에서 직접 전달 가능 (없으면 JSON에서 로드).
    """
    cfg = _load_config()
    result = SafetyMarginResult(ticker=ticker, name=name, close=close)

    # 컨센서스 데이터 확보
    if consensus is None:
        pool = _load_consensus_pool()
        consensus = pool.get(ticker)

    if not consensus:
        result.signal = "NO_DATA"
        result.signal_label = "데이터없음"
        result.reason = "컨센서스 데이터 없음"
        return result

    # 기본 데이터 추출
    fwd_eps = float(consensus.get("forward_eps", 0) or 0)
    fwd_per = float(consensus.get("forward_per", 0) or 0)
    target = int(consensus.get("target_price", 0) or 0)
    analyst_cnt = int(consensus.get("analyst_count", 0) or 0)
    opinion = float(consensus.get("opinion_score", 0) or 0)
    fwd_pbr = float(consensus.get("forward_pbr", 0) or 0)
    div_yield = float(consensus.get("dividend_yield", 0) or 0)

    result.forward_eps = fwd_eps
    result.forward_per = fwd_per
    result.target_price = target
    result.analyst_count = analyst_cnt
    result.opinion_score = opinion
    result.forward_pbr = fwd_pbr
    result.dividend_yield = div_yield

    # close가 0이면 consensus에서 가져오기
    if close <= 0:
        close = int(consensus.get("close", 0) or 0)
        result.close = close
    if close <= 0:
        result.signal = "NO_DATA"
        result.signal_label = "데이터없음"
        result.reason = "현재가 정보 없음"
        return result

    # ── Layer 1: 보수적 적정가 ──
    result.eps_positive = fwd_eps > 0
    if fwd_eps > 0:
        con_per = _get_conservative_per(fwd_per, cfg)
        result.conservative_per = round(con_per, 2)
        result.conservative_value = int(fwd_eps * con_per)
        result.margin_vs_conservative = round(
            (result.conservative_value / close - 1) * 100, 1
        )

    # ── Layer 2: 애널리스트 목표가 할인 ──
    if target > 0:
        discount_ratio = cfg["target_discount_ratio"]
        result.discounted_target = int(target * discount_ratio)
        result.margin_vs_target = round(
            (result.discounted_target / close - 1) * 100, 1
        )

    # ── Layer 3: 이익 지속성 ──
    result.analyst_sufficient = analyst_cnt >= cfg["min_analyst_count"]
    result.opinion_sufficient = opinion >= cfg["min_opinion_score"]
    result.sustainability_pass = (
        result.eps_positive
        and result.analyst_sufficient
        and result.opinion_sufficient
    )

    # ── 바닥가 산정 ──
    candidates = []
    if result.conservative_value > 0:
        candidates.append(result.conservative_value)
    if result.discounted_target > 0:
        candidates.append(result.discounted_target)

    if not candidates:
        result.signal = "RED"
        result.signal_label = "위험"
        result.reason = "적정가 산정 불가 (EPS 적자 + 목표가 없음)"
        return result

    result.floor_price = min(candidates)
    result.floor_margin_pct = round(
        (result.floor_price / close - 1) * 100, 1
    )

    # ── 신호 판정 ──
    if not result.eps_positive:
        result.signal = "RED"
        result.signal_label = "위험"
        result.reason = "EPS 적자 — 밸류에이션 앵커 없음"
    elif result.floor_margin_pct >= 0 and result.sustainability_pass:
        # 현재가 ≤ 바닥가 + 지속성OK → GREEN
        result.signal = "GREEN"
        result.signal_label = "안전"
        result.reason = (
            f"바닥가({result.floor_price:,}) 이하 "
            f"— 보수적으로 봐도 저평가"
        )
    elif result.floor_margin_pct >= 0 and not result.sustainability_pass:
        # 현재가 ≤ 바닥가 + 지속성NG → YELLOW
        result.signal = "YELLOW"
        result.signal_label = "주의"
        parts = []
        if not result.analyst_sufficient:
            parts.append(f"애널리스트 {analyst_cnt}명(<{cfg['min_analyst_count']})")
        if not result.opinion_sufficient:
            parts.append(f"투자의견 {opinion:.1f}(<{cfg['min_opinion_score']})")
        result.reason = f"싸보이지만 데이터 부족: {', '.join(parts)}"
    elif result.floor_margin_pct >= cfg["yellow_threshold_pct"]:
        # 바닥가 < 현재가 ≤ 바닥가×1.2 → YELLOW
        result.signal = "YELLOW"
        result.signal_label = "주의"
        result.reason = (
            f"바닥가 대비 {result.floor_margin_pct:+.1f}% "
            f"— 약간 비싸지만 허용범위"
        )
    else:
        # 현재가 > 바닥가×1.2 → RED
        result.signal = "RED"
        result.signal_label = "위험"
        result.reason = (
            f"바닥가 대비 {result.floor_margin_pct:+.1f}% "
            f"— 안전마진 없음"
        )

    return result


# ─── 배치 처리 ─────────────────────────────────────

def safety_margin_batch(
    picks: list[dict],
) -> list[SafetyMarginResult]:
    """추천 종목 배치 안전마진 판정. picks = [{ticker, name, close}, ...]"""
    pool = _load_consensus_pool()
    results = []
    for pick in picks:
        ticker = pick.get("ticker", "")
        if not ticker:
            continue
        name = pick.get("name", "")
        close = int(pick.get("close", 0) or 0)
        consensus = pool.get(ticker)
        r = calc_safety_margin(ticker, name, close, consensus)
        results.append(r)

    # GREEN → YELLOW → RED → NO_DATA 순 정렬
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2, "NO_DATA": 3}
    results.sort(key=lambda x: order.get(x.signal, 9))
    return results


# ─── 히스토리 스냅샷 ───────────────────────────────

def save_consensus_snapshot() -> Path | None:
    """당일 컨센서스 데이터 히스토리 저장."""
    cfg = _load_config()
    if not cfg.get("save_history", True):
        return None
    if not CONSENSUS_PATH.exists():
        return None

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    dest = HISTORY_DIR / f"{today}.json"

    if dest.exists():
        return dest  # 이미 저장됨

    try:
        with open(CONSENSUS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data["snapshot_date"] = today
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[안전마진] 히스토리 저장: %s", dest)
        return dest
    except Exception as e:
        logger.warning("[안전마진] 히스토리 저장 실패: %s", e)
        return None


def get_eps_revision(ticker: str, days: int = 30) -> dict:
    """EPS 리비전 방향 (히스토리 누적 후 활성화).

    Returns: {"direction": "UP"/"DOWN"/"FLAT"/"NO_DATA", "change_pct": float}
    """
    if not HISTORY_DIR.exists():
        return {"direction": "NO_DATA", "change_pct": 0.0}

    history_files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    if len(history_files) < 2:
        return {"direction": "NO_DATA", "change_pct": 0.0}

    try:
        # 최신 vs days일 전
        latest_path = history_files[0]
        oldest_path = history_files[-1]
        for f in history_files:
            date_str = f.stem
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if (datetime.now() - dt).days >= days:
                oldest_path = f
                break

        with open(latest_path, encoding="utf-8") as f:
            latest = json.load(f)
        with open(oldest_path, encoding="utf-8") as f:
            oldest = json.load(f)

        latest_eps = None
        for item in latest.get("all_picks", []):
            if item.get("ticker") == ticker:
                latest_eps = float(item.get("forward_eps", 0) or 0)
                break

        oldest_eps = None
        for item in oldest.get("all_picks", []):
            if item.get("ticker") == ticker:
                oldest_eps = float(item.get("forward_eps", 0) or 0)
                break

        if latest_eps is None or oldest_eps is None or oldest_eps == 0:
            return {"direction": "NO_DATA", "change_pct": 0.0}

        change_pct = round((latest_eps / oldest_eps - 1) * 100, 1)
        if change_pct > 2:
            direction = "UP"
        elif change_pct < -2:
            direction = "DOWN"
        else:
            direction = "FLAT"

        return {"direction": direction, "change_pct": change_pct}

    except Exception:
        return {"direction": "NO_DATA", "change_pct": 0.0}
