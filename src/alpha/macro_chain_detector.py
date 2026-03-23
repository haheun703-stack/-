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
        active_ids = set()

        for chain in chains:
            trigger = chain.get("trigger", {})
            indicator = trigger.get("indicator", "")
            indicators = trigger.get("indicators", [])  # 크로스 인디케이터
            condition = trigger.get("condition", "")

            if indicators:
                # 크로스 인디케이터: dot notation (gold.ret_5d, vix.level)
                all_present = all(self.macro_data.get(ind) for ind in indicators)
                if not all_present:
                    continue
                triggered = self._evaluate_condition(condition, self.macro_data,
                                                     cross_indicator=True)
                current = {ind: self.macro_data[ind] for ind in indicators}
                indicator = "+".join(indicators)
            else:
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
                active_ids.add(chain["id"])
                logger.info("매크로 체인 활성: %s (%s)", chain["name"], indicator)

        # overrides 처리: 상위 체인이 활성이면 하위 체인 제거
        override_targets = set()
        for chain in self.chain_map.get("chains", []):
            if chain["id"] in active_ids:
                for ov in chain.get("overrides", []):
                    override_targets.add(ov)
        if override_targets:
            active_chains = [c for c in active_chains if c["id"] not in override_targets]
            for ov_id in override_targets:
                logger.info("매크로 체인 오버라이드: %s 제거 (상위 체인 활성)", ov_id)

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

    def _evaluate_condition(self, condition: str, current: dict,
                            *, cross_indicator: bool = False) -> bool:
        """조건 문자열을 평가. 안전한 평가만 수행.

        cross_indicator=True 면 current는 {indicator: {field: val}} 형태이고
        조건은 'gold.ret_5d <= -5 AND vix.level >= 25' 같은 dot notation.
        """
        if not condition or not current:
            return False

        try:
            # OR로 분리 → 하나라도 True면 활성
            or_parts = [p.strip() for p in condition.split(" OR ")]
            for part in or_parts:
                if self._evaluate_single(part, current, cross_indicator=cross_indicator):
                    return True

            # AND로 분리 → 모두 True여야 활성
            if " AND " in condition and " OR " not in condition:
                and_parts = [p.strip() for p in condition.split(" AND ")]
                return all(self._evaluate_single(p, current,
                           cross_indicator=cross_indicator) for p in and_parts)

            return False
        except Exception as e:
            logger.warning("조건 평가 실패: %s — %s", condition, e)
            return False

    def _evaluate_single(self, expr: str, current: dict,
                          *, cross_indicator: bool = False) -> bool:
        """단일 비교 표현식 평가.

        일반: 'ret_5d >= 5' — current에서 field 직접 조회
        크로스: 'gold.ret_5d <= -5' — current[indicator][field] 조회
        """
        try:
            for op in (">=", "<=", ">", "<", "=="):
                if op in expr:
                    parts = expr.split(op, 1)
                    field_path = parts[0].strip()
                    threshold = float(parts[1].strip())

                    # dot notation: indicator.field
                    if cross_indicator and "." in field_path:
                        ind_name, field = field_path.split(".", 1)
                        ind_data = current.get(ind_name, {})
                        actual = ind_data.get(field, 0)
                    else:
                        actual = current.get(field_path, 0)

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
# 시나리오-수급 충돌 감지
# ============================================================

# 시나리오 → 관련 섹터 매핑
SCENARIO_SECTOR_MAP = {
    "WAR_MIDDLE_EAST": ["방산", "에너지화학"],
    "OIL_SPIKE": ["에너지화학"],
    "FED_RATE_CUT": ["IT", "바이오", "헬스케어"],
    "SEMICONDUCTOR_CYCLE_UP": ["반도체", "IT"],
    "CHINA_RECOVERY": ["철강소재", "2차전지"],
    "US_RECESSION": ["인터넷", "소프트웨어"],
    "KOREA_RATE_CUT": ["건설", "금융"],
}


def detect_scenario_supply_conflicts() -> list[dict]:
    """시나리오 활성(50+)인데 관련 섹터 수급이 COLD → 충돌 경고 생성.

    예: WAR_MIDDLE_EAST 70점인데 방산 -4,700억 순매도 = Phase 전환 가능성.
    """
    scenarios = _load_json(DATA_DIR / "scenarios" / "active_scenarios.json")
    flow = _load_json(DATA_DIR / "sector_rotation" / "investor_flow.json")
    if not scenarios or not flow:
        return []

    scenario_dict = scenarios.get("scenarios", {})
    sector_list = flow.get("sectors", [])

    # 섹터명 → 수급 데이터 매핑
    flow_by_sector = {}
    for s in sector_list:
        name = s.get("sector", "")
        total_flow = s.get("foreign_cum_bil", 0) + s.get("inst_cum_bil", 0)
        flow_by_sector[name] = {
            "total_flow_bil": round(total_flow, 1),
            "foreign_cum": s.get("foreign_cum_bil", 0),
            "inst_cum": s.get("inst_cum_bil", 0),
            "price_change_5": s.get("price_change_5", 0),
        }

    conflicts = []
    for sc_id, sc_data in scenario_dict.items():
        score = sc_data.get("score", 0)
        if score < 50:
            continue

        related_sectors = SCENARIO_SECTOR_MAP.get(sc_id, [])
        for sector_name in related_sectors:
            sf = flow_by_sector.get(sector_name)
            if not sf:
                continue

            total_flow = sf["total_flow_bil"]
            if total_flow < -500:  # 5일 순매도 500억 이상이면 충돌
                conflicts.append({
                    "scenario": sc_id,
                    "scenario_score": score,
                    "sector": sector_name,
                    "total_flow_bil": total_flow,
                    "price_change_5": sf["price_change_5"],
                    "warning": (
                        f"{sc_id}({score}점) 활성인데 "
                        f"{sector_name} 수급 {total_flow:+,.0f}억 (5일) — "
                        f"Phase 전환 가능성 주의"
                    ),
                })
                logger.warning("시나리오-수급 충돌: %s(%d점) vs %s 수급 %+.0f억",
                               sc_id, score, sector_name, total_flow)

    # 결과 저장
    if conflicts:
        result = {
            "timestamp": datetime.now().isoformat(),
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
        }
        _save_json(DATA_DIR / "scenario_supply_conflicts.json", result)

    return conflicts


def format_conflict_telegram(conflicts: list[dict]) -> str:
    """시나리오-수급 충돌을 텔레그램 메시지로 포맷."""
    if not conflicts:
        return ""

    lines = ["\u26a0\ufe0f 시나리오-수급 충돌 감지", ""]
    for c in conflicts:
        lines.append(
            f"\u2022 {c['scenario']}({c['scenario_score']}점) "
            f"vs {c['sector']} {c['total_flow_bil']:+,.0f}억"
        )
        if c["price_change_5"]:
            lines.append(f"  가격 5일: {c['price_change_5']:+.1f}%")
        lines.append(f"  \u2192 Phase 전환 가능성 주의")
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

    # 시나리오-수급 충돌 감지
    conflicts = detect_scenario_supply_conflicts()
    if conflicts:
        print(f"\n시나리오-수급 충돌: {len(conflicts)}건")
        for c in conflicts:
            print(f"  {c['warning']}")
    else:
        print("\n시나리오-수급 충돌: 없음")

    # 텔레그램 프리뷰
    telegram_text = format_macro_telegram(active)
    conflict_text = format_conflict_telegram(conflicts)
    full_text = telegram_text + ("\n" + conflict_text if conflict_text else "")
    if full_text.strip():
        print(f"\n--- 텔레그램 프리뷰 ---\n{full_text}")

    return active


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_macro_detector()
