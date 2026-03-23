"""
MACRO CHAIN DETECTOR — 원자재/환율/금리 변동 → 종목/ETF 수혜/피해 자동 감지
===========================================================================
매크로 체인 맵(data/macro_chain_map.json)의 트리거 조건을 평가하여
현재 활성화된 체인을 판별.

데이터 소스:
  - overnight_signal.json (원자재: gold/oil/copper, VIX)
  - regime_macro_signal.json (환율, 금리)
  - 정보봇 sector_flow_jgis.json (WTI 가격, 금 가격)

출력:
  - data/active_macro_chains.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHARED_DATA_DIR = Path("D:/shared-bot-data/jgis_to_quant")


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("JSON 로드 실패 %s: %s", path.name, e)
    return None


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


class MacroChainDetector:
    """매크로 변동 → 수혜/피해 체인 자동 감지."""

    def __init__(self):
        self.chain_map = _load_json(DATA_DIR / "macro_chain_map.json") or {}
        self.macro_data = self._load_macro_data()

    def detect(self) -> list[dict]:
        """현재 활성화된 매크로 체인 판별."""
        chains = self.chain_map.get("chains", [])
        active_chains = []

        for chain in chains:
            trigger = chain.get("trigger", {})
            indicator = trigger.get("indicator", "")
            condition = trigger.get("condition", "")

            current = self.macro_data.get(indicator)
            if not current:
                continue

            triggered = self._evaluate_condition(condition, current)

            if triggered:
                active_chains.append({
                    "id": chain["id"],
                    "name": chain["name"],
                    "indicator": indicator,
                    "current_value": current,
                    "condition": condition,
                    "beneficiaries": chain.get("beneficiaries", []),
                    "victims": chain.get("victims", []),
                })
                logger.info("매크로 체인 활성: %s (%s)", chain["name"], indicator)

        # 저장
        result = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "active_count": len(active_chains),
            "active_chains": active_chains,
            "macro_snapshot": self.macro_data,
        }
        _save_json(DATA_DIR / "active_macro_chains.json", result)

        return active_chains

    def _load_macro_data(self) -> dict:
        """여러 소스에서 매크로 데이터 통합 로드."""
        data = {}

        # ── 1. overnight_signal.json (원자재 + VIX) ──
        overnight = _load_json(DATA_DIR / "us_market" / "overnight_signal.json")
        if overnight:
            commodities = overnight.get("commodities", {})

            # 원자재 (gold, oil, copper, silver, natgas, uranium)
            for name in ("gold", "oil", "copper", "silver", "natgas", "uranium"):
                item = commodities.get(name, {})
                if item:
                    data[name] = {
                        "ret_1d": item.get("ret_1d", 0),
                        "ret_5d": item.get("ret_5d", 0),
                        "signal": item.get("signal", ""),
                        "above_sma20": item.get("above_sma20", False),
                    }

            # VIX
            vix = overnight.get("vix", {})
            if vix:
                data["vix"] = {
                    "level": vix.get("level", 20),
                    "zscore": vix.get("zscore", 0),
                    "status": vix.get("status", ""),
                }

        # ── 2. 정보봇 공유 데이터 (WTI 절대가격 등) ──
        shared = _load_json(SHARED_DATA_DIR / "sector_flow_jgis.json")
        if shared:
            commodity_signals = shared.get("commodity_signals", {})

            # WTI 절대가격이 있으면 oil에 병합
            wti = commodity_signals.get("wti", {})
            if wti and "oil" in data:
                data["oil"]["price"] = wti.get("price", 0)
                data["oil"]["trend"] = wti.get("trend", "")
            elif wti:
                data["oil"] = {
                    "price": wti.get("price", 0),
                    "ret_1d": wti.get("change_1d_pct", 0),
                    "ret_5d": 0,
                    "trend": wti.get("trend", ""),
                }

            # 금 절대가격
            gold_shared = commodity_signals.get("gold", {})
            if gold_shared and "gold" in data:
                data["gold"]["price"] = gold_shared.get("price", 0)
            elif gold_shared:
                data["gold"] = {
                    "price": gold_shared.get("price", 0),
                    "ret_1d": gold_shared.get("change_1d_pct", 0),
                    "ret_5d": 0,
                }

            # 구리 절대가격
            copper_shared = commodity_signals.get("copper", {})
            if copper_shared and "copper" in data:
                data["copper"]["price"] = copper_shared.get("price", 0)

            # 글로벌 자금 흐름
            global_flow = shared.get("global_flow", {})
            if global_flow:
                data["global_flow"] = global_flow

        # ── 3. regime_macro_signal.json ──
        macro = _load_json(DATA_DIR / "regime_macro_signal.json")
        if macro:
            signals = macro.get("signals", {})

            # USD/KRW는 현재 직접 필드가 없지만, 추후 추가 가능
            # EWY를 프록시로 환율 방향 추정 (EWY 하락 = 원화 약세)
            ewy_5d = signals.get("ewy_5d", 0)
            data["usd_krw"] = {
                "level": 0,  # 절대가격 없음 — 추후 FDR에서 수집 가능
                "ret_5d": -ewy_5d * 0.3,  # EWY와 반대 방향 프록시
                "proxy": True,
            }

            # 미국 10년물 — VIX z-score로 프록시
            vix_data = data.get("vix", {})
            data["us_10y"] = {
                "level": 0,  # 절대가격 없음
                "ret_5d": 0,
                "proxy": True,
            }

        return data

    def _evaluate_condition(self, condition: str, current: dict) -> bool:
        """조건 문자열을 평가. 안전한 평가만 수행."""
        if not condition or not current:
            return False

        try:
            # OR로 분리 → 하나라도 True면 활성
            or_parts = [p.strip() for p in condition.split(" OR ")]
            for part in or_parts:
                if self._evaluate_single(part, current):
                    return True

            # AND로 분리 → 모두 True여야 활성
            if " AND " in condition and " OR " not in condition:
                and_parts = [p.strip() for p in condition.split(" AND ")]
                return all(self._evaluate_single(p, current) for p in and_parts)

            return False
        except Exception as e:
            logger.warning("조건 평가 실패: %s — %s", condition, e)
            return False

    def _evaluate_single(self, expr: str, current: dict) -> bool:
        """단일 비교 표현식 평가. ex: 'ret_5d >= 5'"""
        try:
            # 파싱: "field op value"
            for op in (">=", "<=", ">", "<", "=="):
                if op in expr:
                    parts = expr.split(op, 1)
                    field = parts[0].strip()
                    threshold = float(parts[1].strip())
                    actual = current.get(field, 0)
                    if actual is None:
                        return False
                    actual = float(actual)

                    if op == ">=":
                        return actual >= threshold
                    elif op == "<=":
                        return actual <= threshold
                    elif op == ">":
                        return actual > threshold
                    elif op == "<":
                        return actual < threshold
                    elif op == "==":
                        return actual == threshold
        except (ValueError, TypeError, IndexError):
            pass
        return False


# ============================================================
# 텔레그램 포맷
# ============================================================

def format_macro_telegram(active_chains: list[dict]) -> str:
    """활성 매크로 체인을 텔레그램 메시지로 포맷."""
    if not active_chains:
        return ""

    indicator_emoji = {
        "oil": "\U0001f6e2\ufe0f",     # 🛢️
        "gold": "\U0001f4b0",           # 💰
        "copper": "\U0001f527",         # 🔧
        "vix": "\U0001f628",            # 😨
        "usd_krw": "\U0001f4b1",        # 💱
        "us_10y": "\U0001f4c8",         # 📈
        "natgas": "\U0001f525",         # 🔥
        "silver": "\u26aa",             # ⚪
        "uranium": "\u2622\ufe0f",      # ☢️
    }

    lines = ["━━━ 매크로 시그널 ━━━", ""]

    for chain in active_chains:
        ind = chain.get("indicator", "")
        emoji = indicator_emoji.get(ind, "\U0001f4ca")  # 📊
        name = chain["name"]
        current = chain.get("current_value", {})

        # 값 표시
        val_parts = []
        if "price" in current:
            val_parts.append(f"${current['price']:,.0f}")
        if "level" in current and current["level"]:
            val_parts.append(f"{current['level']:.1f}")
        if "ret_1d" in current and current["ret_1d"]:
            val_parts.append(f"1d {current['ret_1d']:+.1f}%")
        if "ret_5d" in current and current["ret_5d"]:
            val_parts.append(f"5d {current['ret_5d']:+.1f}%")

        val_str = " / ".join(val_parts) if val_parts else ""
        lines.append(f"{emoji} {name}: {val_str}")

        # 수혜
        bene_parts = []
        for b in chain.get("beneficiaries", []):
            if b["type"] == "etf":
                bene_parts.append(b.get("name", b.get("ticker", "")))
            elif b["type"] == "sector":
                bene_parts.append(b.get("sector", ""))
            elif b["type"] == "stock":
                tickers = b.get("tickers", [])
                bene_parts.extend(t.split(":")[1] for t in tickers[:2])
        if bene_parts:
            lines.append(f"  \u2192 수혜: {', '.join(bene_parts[:4])}")

        # 피해
        victim_parts = []
        for v in chain.get("victims", []):
            if v["type"] == "sector":
                victim_parts.append(v.get("sector", ""))
            elif v["type"] == "warning":
                victim_parts.append(v.get("message", "")[:30])
        if victim_parts:
            lines.append(f"  \u2192 주의: {', '.join(victim_parts[:3])}")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def run_macro_detector() -> list[dict]:
    """매크로 체인 감지 실행."""
    print("=" * 50)
    print("매크로 체인 감지기 실행")
    print("=" * 50)

    detector = MacroChainDetector()
    active = detector.detect()

    print(f"\n매크로 데이터 소스: {len(detector.macro_data)}개 지표")
    for ind, val in detector.macro_data.items():
        if isinstance(val, dict) and not val.get("proxy"):
            parts = []
            if "price" in val:
                parts.append(f"${val['price']:,.0f}")
            if "level" in val:
                parts.append(f"{val['level']}")
            if "ret_1d" in val:
                parts.append(f"1d:{val['ret_1d']:+.1f}%")
            if "ret_5d" in val:
                parts.append(f"5d:{val['ret_5d']:+.1f}%")
            print(f"  {ind}: {' / '.join(parts)}")

    print(f"\n활성 매크로 체인: {len(active)}건")
    for chain in active:
        print(f"  [{chain['id']}] {chain['name']}")
        for b in chain.get("beneficiaries", []):
            if b["type"] == "etf":
                print(f"    수혜 ETF: {b['name']} — {b['reason']}")
            elif b["type"] == "sector":
                print(f"    수혜 섹터: {b.get('sector', '')} — {b['reason']}")

    # 텔레그램 프리뷰
    telegram_text = format_macro_telegram(active)
    if telegram_text:
        print(f"\n--- 텔레그램 프리뷰 ---\n{telegram_text}")

    return active


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_macro_detector()
