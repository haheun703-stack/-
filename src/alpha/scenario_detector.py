"""STEP 10: SCENARIO ENGINE — 시나리오 기반 섹터 체인 자동 추적

이벤트(전쟁, 금리, 유가 등) 감지 시 인과 체인에 따라
다음 HOT/COLD 섹터를 LENS 2(FLOW MAP)에 자동 반영한다.

brain.py 수정 ❌, signal_engine.py 수정 ❌
LENS LAYER 확장만 사용.

JGIS 연동: D:/shared-bot-data/jgis_to_quant/ 에서 정보봇 데이터 읽기.
  - daily_intelligence.json → 키워드 매칭 우선 사용 (fallback: 자체 뉴스)
  - breaking_alerts.json → 긴급 뉴스 (INTRADAY EYE에서도 호출)

실행: BAT-D 11.23단계 (LENS 직전)
출력: data/scenarios/active_scenarios.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_SCENARIO_DIR = _DATA_DIR / "scenarios"
_CHAINS_PATH = _SCENARIO_DIR / "scenario_chains.json"
_ACTIVE_PATH = _SCENARIO_DIR / "active_scenarios.json"

# 데이터 소스 (자체)
_OVERNIGHT_PATH = _DATA_DIR / "us_market" / "overnight_signal.json"
_MACRO_PATH = _DATA_DIR / "regime_macro_signal.json"
_NEWS_PATH = _DATA_DIR / "market_news.json"
_DART_PATH = _DATA_DIR / "dart_event_signals.json"

# JGIS 공유 폴더 (정보봇 → 퀀트봇)
_JGIS_DEFAULT_PATH = Path("D:/shared-bot-data/jgis_to_quant")


def _load_jgis_config() -> dict:
    """settings.yaml에서 jgis_integration 설정 로드."""
    try:
        cfg_path = Path("config/settings.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("jgis_integration", {})
    except Exception:
        return {}


class ScenarioDetector:
    """시나리오 트리거 감지 + Phase 관리 + JGIS 연동 + 이벤트 캘린더."""

    def __init__(self):
        self.chains = self._load_json(_CHAINS_PATH)
        raw_active = self._load_json(_ACTIVE_PATH) or {}
        # active_scenarios.json은 {"updated": ..., "scenarios": {...}} 구조
        self.active = raw_active.get("scenarios", raw_active) if isinstance(raw_active, dict) else {}
        self.overnight = self._load_json(_OVERNIGHT_PATH) or {}
        self.macro = self._load_json(_MACRO_PATH) or {}
        self.news = self._load_json(_NEWS_PATH) or {}
        self.dart = self._load_json(_DART_PATH) or {}

        # 이벤트 캘린더 (STEP 11)
        try:
            from src.alpha.event_calendar import EventCalendar
            self.event_cal = EventCalendar()
        except Exception:
            self.event_cal = None

        # JGIS 연동
        self.jgis_cfg = _load_jgis_config()
        self.jgis_enabled = self.jgis_cfg.get("enabled", False)
        self.jgis_path = Path(self.jgis_cfg.get("shared_path", str(_JGIS_DEFAULT_PATH)))
        self.jgis_intel: dict | None = None  # lazy load

    def detect(self) -> dict:
        """모든 시나리오 스캔 → 활성/비활성 판별 → active_scenarios.json 저장.

        Returns:
            {scenario_id: {first_detected, current_phase, days_active, ...}, ...}
        """
        if not self.chains or "scenarios" not in self.chains:
            logger.warning("scenario_chains.json 로드 실패 또는 비어있음")
            return {}

        today = date.today().isoformat()
        changes = []  # (scenario_id, event_type, details) — 텔레그램용

        for scenario in self.chains["scenarios"]:
            sid = scenario["id"]
            score, reasons = self._evaluate_trigger(scenario)

            if score >= 40:
                if sid not in self.active:
                    # 신규 활성화
                    self.active[sid] = {
                        "first_detected": today,
                        "current_phase": 1,
                        "days_active": 0,
                        "score": score,
                        "reasons": reasons,
                        "last_phase_change": today,
                    }
                    changes.append((sid, "ACTIVATE", scenario["name"]))
                    logger.info("시나리오 활성화: %s (score=%d, %s)",
                                scenario["name"], score, reasons)
                else:
                    # 기존 활성 — Phase 업데이트
                    state = self.active[sid]
                    first = datetime.strptime(state["first_detected"], "%Y-%m-%d").date()
                    days_active = (date.today() - first).days
                    state["days_active"] = days_active
                    state["score"] = score
                    state["reasons"] = reasons

                    new_phase = self._determine_phase(scenario, days_active)
                    if new_phase != state["current_phase"]:
                        old_phase = state["current_phase"]
                        state["current_phase"] = new_phase
                        state["last_phase_change"] = today
                        changes.append((sid, "PHASE_CHANGE",
                                        f"{scenario['name']}: Phase {old_phase}→{new_phase}"))
                        logger.info("Phase 전환: %s P%d→P%d (D+%d)",
                                    scenario["name"], old_phase, new_phase, days_active)
            else:
                if sid in self.active:
                    # 비활성화 (트리거 조건 불충족)
                    state = self.active[sid]
                    first = datetime.strptime(state["first_detected"], "%Y-%m-%d").date()
                    days_active = (date.today() - first).days
                    # 최소 7일은 유지 (너무 빠른 비활성화 방지)
                    if days_active >= 7:
                        changes.append((sid, "DEACTIVATE", scenario["name"]))
                        logger.info("시나리오 비활성화: %s (D+%d, score=%d)",
                                    scenario["name"], days_active, score)
                        del self.active[sid]
                    else:
                        state["days_active"] = days_active

        # 상호 배타 시나리오 정리 (높은 점수만 유지)
        self._resolve_conflicts()

        # 저장
        self._save_active()

        # 변경 사항 반환 (텔레그램 알림용)
        return {"active": self.active, "changes": changes}

    def get_active_chains(self) -> list[dict]:
        """현재 활성 시나리오의 현재 Phase 체인 정보 반환.

        LENS flow_map이 이 결과를 읽어 가중치를 조정한다.

        Returns:
            [
                {
                    "scenario_id": "WAR_MIDDLE_EAST",
                    "scenario_name": "중동 전쟁",
                    "phase": 2,
                    "phase_name": "유가 전파 (D+2~7)",
                    "hot_sectors": ["정유", "에너지"],
                    "cold_sectors": ["항공", "화학"],
                    "hot_tickers": [...],
                    "next_phase_name": "안전자산 이동 (D+3~14)",
                    "next_hot": ["금", "달러", "채권"],
                    "days_active": 15,
                    "logic": "..."
                },
                ...
            ]
        """
        if not self.chains or "scenarios" not in self.chains:
            return []

        result = []
        scenario_map = {s["id"]: s for s in self.chains["scenarios"]}

        for sid, state in self.active.items():
            scenario = scenario_map.get(sid)
            if not scenario:
                continue

            phase_idx = state["current_phase"] - 1
            chain = scenario["chain"]
            if phase_idx >= len(chain):
                phase_idx = len(chain) - 1

            current = chain[phase_idx]
            next_phase = chain[phase_idx + 1] if phase_idx + 1 < len(chain) else None

            result.append({
                "scenario_id": sid,
                "scenario_name": scenario["name"],
                "phase": state["current_phase"],
                "phase_name": current["name"],
                "hot_sectors": current.get("hot_sectors", []),
                "cold_sectors": current.get("cold_sectors", []),
                "hot_tickers": current.get("hot_tickers", []),
                "etf": current.get("etf", []),
                "next_phase_name": next_phase["name"] if next_phase else None,
                "next_hot": next_phase.get("hot_sectors", []) if next_phase else [],
                "days_active": state["days_active"],
                "logic": current.get("logic", ""),
            })

        return result

    # ------------------------------------------------------------------
    # JGIS 긴급 뉴스 + 센티먼트
    # ------------------------------------------------------------------

    def check_breaking_alerts(self) -> list[dict]:
        """JGIS breaking_alerts.json 체크 — 미읽은 긴급 뉴스 반환.

        INTRADAY EYE에서도 호출 가능.
        읽은 후 read_by_quant=true 처리.
        """
        if not self.jgis_enabled:
            return []

        alert_file = self.jgis_cfg.get("files", {}).get(
            "breaking_alerts", "breaking_alerts.json")
        alert_path = self.jgis_path / alert_file
        data = self._load_json(alert_path)
        if not data:
            return []

        alerts = data.get("alerts", [])
        unread = [a for a in alerts if not a.get("read_by_quant", False)]

        if unread:
            # 읽음 처리
            for a in unread:
                a["read_by_quant"] = True
            try:
                with open(alert_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.warning("breaking_alerts 읽음 처리 실패: %s", e)

            logger.info("[JGIS] 긴급 뉴스 %d건 수신", len(unread))

        return unread

    def get_jgis_sector_sentiment(self) -> dict:
        """JGIS daily_intelligence.json에서 섹터 센티먼트 반환.

        flow_map.py에서 직접 호출하지 않고, 이 메서드를 통해 접근.
        Returns:
            {sector_name: {"score": int, "direction": str}, ...}
        """
        if not self.jgis_enabled:
            return {}

        if self.jgis_intel is None:
            intel_file = self.jgis_cfg.get("files", {}).get(
                "daily_intelligence", "daily_intelligence.json")
            intel_path = self.jgis_path / intel_file
            self.jgis_intel = self._load_json(intel_path) or {}

        return self.jgis_intel.get("sector_sentiment", {})

    # ------------------------------------------------------------------
    # 상호 배타 시나리오 정리
    # ------------------------------------------------------------------

    # 같은 그룹 내에서 높은 점수만 유지 (나머지 제거)
    _EXCLUSIVE_GROUPS = [
        ("FED_RATE_CUT", "FED_RATE_HIKE"),
    ]

    def _resolve_conflicts(self) -> None:
        """상호 배타적 시나리오 그룹에서 점수 낮은 쪽 제거."""
        for group in self._EXCLUSIVE_GROUPS:
            present = {sid: self.active[sid] for sid in group if sid in self.active}
            if len(present) <= 1:
                continue
            # 점수 높은 쪽만 유지
            winner = max(present, key=lambda sid: present[sid].get("score", 0))
            for sid in present:
                if sid != winner:
                    logger.info("상호배타 제거: %s (winner=%s)", sid, winner)
                    del self.active[sid]

    # ------------------------------------------------------------------
    # 트리거 평가
    # ------------------------------------------------------------------

    def _evaluate_trigger(self, scenario: dict) -> tuple[int, list[str]]:
        """시나리오 트리거 조건 평가. (score, reasons) 반환."""
        trigger = scenario.get("trigger_conditions", {})
        score = 0
        reasons = []

        # 1. 키워드 매칭 (JGIS 우선 → 자체 뉴스 fallback) — 최소 3건
        keywords = trigger.get("keywords", [])
        sid = scenario.get("id", "")
        if keywords:
            hits = self._check_keywords(keywords, scenario_id=sid)
            if hits >= 3:
                score += min(40 + (hits - 3) * 5, 60)  # 3건부터 40, 히트당 +5, 최대 60
                reasons.append(f"키워드 {hits}건")
            elif hits > 0:
                score += 10  # 1~2건은 약한 시그널 (단독으로는 활성화 안됨)
                reasons.append(f"키워드 {hits}건 (약)")

        # 2. 시장 시그널 매칭
        signals = trigger.get("market_signals", {})
        for signal_key, threshold in signals.items():
            matched, detail = self._check_signal(signal_key, threshold)
            if matched:
                score += 30
                reasons.append(detail)

        # 3. 이벤트 캘린더 부스트 (STEP 11)
        if self.event_cal and sid:
            event_boost = self.event_cal.get_scenario_boost(sid, days=7)
            if event_boost > 0:
                score += event_boost
                reasons.append(f"이벤트 임박 +{event_boost}")

        return score, reasons

    def _check_keywords(self, keywords: list[str], scenario_id: str = "") -> int:
        """키워드 매칭: JGIS 우선 → 자체 뉴스 fallback."""

        # 1) JGIS daily_intelligence.json 우선 사용
        if self.jgis_enabled:
            jgis_hits = self._check_keywords_jgis(keywords, scenario_id)
            if jgis_hits >= 0:  # -1이면 파일 없음 → fallback
                return jgis_hits

        # 2) fallback: 자체 뉴스/DART 데이터
        return self._check_keywords_local(keywords)

    def _check_keywords_jgis(self, keywords: list[str], scenario_id: str) -> int:
        """JGIS daily_intelligence.json에서 키워드 매칭.

        Returns:
            매칭 횟수. 파일 없으면 -1 (fallback 트리거).
        """
        if self.jgis_intel is None:
            intel_file = self.jgis_cfg.get("files", {}).get(
                "daily_intelligence", "daily_intelligence.json")
            intel_path = self.jgis_path / intel_file
            self.jgis_intel = self._load_json(intel_path)
            if self.jgis_intel is None:
                self.jgis_intel = {}  # 파일 없음 마킹

        if not self.jgis_intel:
            return -1  # fallback 트리거

        hits = 0

        # scenario_keywords에서 직접 매칭 (정보봇이 이미 계산)
        scenario_kw = self.jgis_intel.get("scenario_keywords", {})
        if scenario_id and scenario_id in scenario_kw:
            return scenario_kw[scenario_id].get("count", 0)

        # headline에서 키워드 서치
        all_headlines = (self.jgis_intel.get("global_headlines", [])
                         + self.jgis_intel.get("korea_headlines", []))
        for headline in all_headlines:
            title = headline.get("title", "")
            for kw in keywords:
                if kw in title:
                    hits += 1
                    break  # 같은 헤드라인에서 중복 방지

        return hits

    def _check_keywords_local(self, keywords: list[str]) -> int:
        """자체 뉴스 + DART 제목에서 키워드 매칭 횟수 (fallback)."""
        hits = 0
        # 뉴스 제목 검색
        articles = self.news.get("articles", [])
        for art in articles[:50]:  # 상위 50건만 (성능)
            title = art.get("title", "")
            for kw in keywords:
                if kw in title:
                    hits += 1
                    break  # 같은 기사에서 중복 카운트 방지

        # DART 공시 검색
        dart_signals = self.dart.get("signals", [])
        for sig in dart_signals[:30]:
            report = sig.get("report_nm", "") + sig.get("event", "")
            for kw in keywords:
                if kw in report:
                    hits += 1
                    break

        return hits

    def _check_signal(self, signal_key: str, threshold) -> tuple[bool, str]:
        """시장 시그널 단일 조건 체크."""
        # VIX
        if signal_key == "vix_above":
            vix = self.overnight.get("vix", {}).get("level", 0)
            if vix >= threshold:
                return True, f"VIX {vix:.1f} >= {threshold}"
            return False, ""

        # WTI (oil)
        if signal_key == "wti_change_1d_pct":
            ret = abs(self.overnight.get("commodities", {}).get("oil", {}).get("ret_1d", 0))
            if ret >= threshold:
                return True, f"WTI 1d {ret:+.1f}% (>={threshold}%)"
            return False, ""

        if signal_key == "wti_above":
            # overnight에 절대 가격은 없으므로, 1주 변동률로 대체
            ret_5d = self.overnight.get("commodities", {}).get("oil", {}).get("ret_5d", 0)
            if ret_5d >= 5.0:  # 5% 이상 급등이면 고유가 추정
                return True, f"WTI 5d +{ret_5d:.1f}% (급등)"
            return False, ""

        if signal_key == "wti_change_1w_pct":
            ret = abs(self.overnight.get("commodities", {}).get("oil", {}).get("ret_5d", 0))
            if ret >= threshold:
                return True, f"WTI 5d {ret:.1f}% (>={threshold}%)"
            return False, ""

        # Gold
        if signal_key == "gold_change_1d_pct":
            ret = abs(self.overnight.get("commodities", {}).get("gold", {}).get("ret_1d", 0))
            if ret >= threshold:
                return True, f"Gold 1d {ret:+.1f}% (>={threshold}%)"
            return False, ""

        # Natural Gas
        if signal_key == "natural_gas_change_1d_pct":
            ret = abs(self.overnight.get("commodities", {}).get("natgas", {}).get("ret_1d", 0))
            if ret >= threshold:
                return True, f"NatGas 1d {ret:+.1f}% (>={threshold}%)"
            return False, ""

        # USD/KRW
        if signal_key == "usd_krw_above":
            # macro signal에서 usdkrw_close 가져오기
            usdkrw = self.macro.get("signals", {}).get("usdkrw_close", 0)
            if not usdkrw:
                # overnight L4_fx_triangle에서 usdkrw_change_pct 참고 (절대값 없음)
                return False, ""
            if usdkrw >= threshold:
                return True, f"USD/KRW {usdkrw:.0f} >= {threshold}"
            return False, ""

        if signal_key == "usd_krw_change_1w_pct":
            pct = self.overnight.get("layer_scores", {}).get("L4_fx_triangle", {}).get("usdkrw_change_pct", 0)
            if abs(pct) >= threshold:
                return True, f"USD/KRW 1w {pct:+.1f}% (>={threshold}%)"
            return False, ""

        # US 10Y Treasury
        if signal_key == "us_10y_change_1w":
            # overnight에 직접 10Y 데이터 없음 → 스킵
            return False, ""

        # SOXX (반도체 ETF)
        if signal_key == "soxx_change_1m_pct":
            # overnight에 SOXX 없음 → 스킵
            return False, ""

        return False, ""

    # ------------------------------------------------------------------
    # Phase 결정
    # ------------------------------------------------------------------

    def _determine_phase(self, scenario: dict, days_active: int) -> int:
        """경과일 기반으로 현재 Phase 번호 반환."""
        chain = scenario.get("chain", [])
        current_phase = 1

        for link in chain:
            day_range = link.get("day_range", [0, 999])
            if days_active >= day_range[0]:
                current_phase = link["phase"]

        return current_phase

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug("JSON 로드 실패: %s — %s", path, e)
            return None

    def _save_active(self) -> None:
        _SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated": datetime.now().isoformat(),
            "scenarios": self.active,
        }
        with open(_ACTIVE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def format_scenario_alert(changes: list[tuple], active: dict,
                          chains_data: dict) -> str | None:
    """시나리오 변경 사항을 텔레그램 메시지로 포맷팅."""
    if not changes:
        return None

    scenario_map = {}
    if chains_data and "scenarios" in chains_data:
        scenario_map = {s["id"]: s for s in chains_data["scenarios"]}

    lines = []
    for sid, event_type, detail in changes:
        if event_type == "ACTIVATE":
            scenario = scenario_map.get(sid, {})
            chain = scenario.get("chain", [{}])
            phase1 = chain[0] if chain else {}
            lines.append(
                f"🎯 [시나리오 활성] {detail}\n"
                f"  Phase 1: {phase1.get('name', '?')}\n"
                f"  HOT: {', '.join(phase1.get('hot_sectors', []))}\n"
                f"  COLD: {', '.join(phase1.get('cold_sectors', []))}\n"
                f"  → {phase1.get('logic', '')}"
            )
        elif event_type == "PHASE_CHANGE":
            state = active.get(sid, {})
            scenario = scenario_map.get(sid, {})
            phase_idx = state.get("current_phase", 1) - 1
            chain = scenario.get("chain", [])
            current = chain[phase_idx] if phase_idx < len(chain) else {}
            next_p = chain[phase_idx + 1] if phase_idx + 1 < len(chain) else None

            msg = (
                f"🔄 [Phase 전환] {detail} (D+{state.get('days_active', 0)})\n"
                f"  현재: {current.get('name', '?')}\n"
                f"  HOT: {', '.join(current.get('hot_sectors', []))}\n"
                f"  COLD: {', '.join(current.get('cold_sectors', []))}"
            )
            if next_p:
                msg += f"\n  다음: {next_p['name']} → {', '.join(next_p.get('hot_sectors', []))}"
            lines.append(msg)
        elif event_type == "DEACTIVATE":
            lines.append(f"⬛ [시나리오 종료] {detail}")

    if not lines:
        return None

    return "📡 SCENARIO ENGINE\n\n" + "\n\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import sys
    from pathlib import Path as P

    PROJECT_ROOT = P(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dry_run = "--dry-run" in sys.argv

    detector = ScenarioDetector()
    result = detector.detect()

    active = result.get("active", {})
    changes = result.get("changes", [])

    # 상태 출력
    if active:
        print(f"\n활성 시나리오: {len(active)}개")
        for sid, state in active.items():
            print(f"  {sid}: Phase {state['current_phase']} "
                  f"(D+{state['days_active']}, score={state['score']}, "
                  f"{', '.join(state.get('reasons', []))})")
    else:
        print("\n활성 시나리오 없음")

    # LENS용 체인 정보
    chains = detector.get_active_chains()
    if chains:
        print(f"\nLENS 반영 체인: {len(chains)}개")
        for c in chains:
            print(f"  {c['scenario_name']} P{c['phase']}: "
                  f"HOT={c['hot_sectors']} COLD={c['cold_sectors']}")

    # JGIS 연동 상태
    if detector.jgis_enabled:
        breaking = detector.check_breaking_alerts()
        if breaking:
            print(f"\n[JGIS] 긴급 뉴스 {len(breaking)}건:")
            for a in breaking:
                print(f"  [{a.get('severity', '?')}] {a.get('title', '?')}")
        sentiment = detector.get_jgis_sector_sentiment()
        if sentiment:
            print(f"\n[JGIS] 섹터 센티먼트: {len(sentiment)}개 섹터")
            for sec, data in sorted(sentiment.items(),
                                     key=lambda x: x[1].get("score", 50),
                                     reverse=True)[:5]:
                print(f"  {sec}: {data.get('score', 0)} ({data.get('direction', '?')})")
    else:
        print("\n[JGIS] 미연동 (jgis_integration.enabled=false 또는 파일 없음)")

    # 이벤트 캘린더 (STEP 11)
    if detector.event_cal:
        cal_msg = detector.event_cal.format_weekly_briefing(days=7)
        if cal_msg:
            print(f"\n{cal_msg}")
        else:
            print("\n[캘린더] 향후 7일 이벤트 없음")
    else:
        print("\n[캘린더] 이벤트 캘린더 미로드")

    # 변경 사항
    if changes:
        print(f"\n변경: {len(changes)}건")
        for sid, etype, detail in changes:
            print(f"  [{etype}] {detail}")

    # 텔레그램 알림
    msg = format_scenario_alert(changes, active, detector.chains)
    if msg:
        if dry_run:
            print(f"\n[DRY-RUN] 텔레그램 메시지:\n{msg}")
        else:
            try:
                from src.telegram_sender import send_message
                send_message(msg)
                print("텔레그램 발송 완료")
            except Exception as e:
                print(f"텔레그램 발송 실패: {e}")
    else:
        print("\n시나리오 변경 없음 — 텔레그램 스킵")


if __name__ == "__main__":
    main()
