"""
4단계 릴레이 판정 엔진
========================
Phase 0: INACTIVE  — 조건 미충족
Phase 1: WATCH     — US 대장주 1개 강세 (예비 경보)
Phase 2: CONFIRM   — US 대장주 2개+ 강세 + US 2차 확산 (본 경보)
Phase 3: KR_READY  — 한국 대장주 전일 강세/시간외 강세 확인 (실행 준비)
Phase 4: EXECUTE   — 한국 대장주 레벨 회복 → 매수 가능 (실행 경보)

입력 데이터 (기존 시스템 재활용):
  - data/relay/us_leaders.json         ← us_tracker.py
  - data/us_market/overnight_signal.json ← 기존 US Overnight
  - data/ai_brain_judgment.json        ← sector_outlook
  - data/market_news.json              ← 키워드 매칭
  - data/sector_rotation/sector_momentum.json ← 한국 섹터 모멘텀

출력:
  - data/relay/relay_signal.json       ← 릴레이 경보 결과
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.relay.config import load_relay_config, get_sectors, get_common_rules, DATA_DIR
from src.relay.us_tracker import load_us_leaders
from src.relay.alert_classifier import classify_alert
from src.relay.execution_rules import (
    generate_kr_leader_signals,
    generate_kr_secondary_signals,
    format_execution_summary,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

SIGNAL_OUTPUT = DATA_DIR / "relay_signal.json"
HISTORY_OUTPUT = DATA_DIR / "relay_history.json"


class RelayEngine:
    """4단계 릴레이 판정 엔진."""

    def __init__(self, config: dict = None):
        self.config = config or load_relay_config()
        self.sectors = get_sectors(self.config)
        self.common_rules = get_common_rules(self.config)

    def run(self) -> dict:
        """전체 릴레이 판정 실행.

        Returns:
            relay_signal.json 형태의 딕셔너리
        """
        logger.info("=== 릴레이 엔진 시작 ===")

        # 1) 데이터 로드
        us_data = load_us_leaders()
        us_overnight = self._load_json("us_market/overnight_signal.json")
        ai_judgment = self._load_json("ai_brain_judgment.json")
        market_news = self._load_json("market_news.json")
        sector_momentum = self._load_json("sector_rotation/sector_momentum.json")

        # 2) 섹터별 판정
        results = {}
        active_alerts = []
        execution_ready = []

        for sec_key, sec_config in self.sectors.items():
            logger.info("── %s (%s) ──", sec_config.get("name", sec_key), sec_config.get("type"))

            # US 대장주 데이터
            us_sec = us_data.get(sec_key, {})

            # 뉴스 분석
            news_result = self._analyze_news(sec_key, sec_config, ai_judgment, market_news)

            # 한국 대장주 전일 강세 확인 (섹터 모멘텀에서)
            kr_strong = self._check_kr_leaders_strong(sec_config, sector_momentum)

            # 경보 분류
            alert = classify_alert(
                sector_key=sec_key,
                sector_config=sec_config,
                us_leaders_data=us_sec,
                us_overnight=us_overnight or {},
                news_score=news_result["score"],
                news_keywords_matched=news_result["keywords_matched"],
                kr_leaders_strong=kr_strong,
                negative_news_found=news_result.get("negative_found", False),
            )

            phase = alert["phase"]
            phase_name = alert["phase_name"]

            # 실행 신호 생성
            kr_leader_signals = generate_kr_leader_signals(
                sec_config, phase, self.common_rules,
            )
            kr_secondary_signals = generate_kr_secondary_signals(
                sec_config, phase,
                kr_leaders_active=(phase >= 3),
                common_rules=self.common_rules,
            )

            # 결과 조합
            sector_result = {
                "name": sec_config.get("name", sec_key),
                "type": sec_config.get("type", "persistent"),
                "phase": phase,
                "phase_name": phase_name,
                "alert_level": alert["alert_level"],
                "reasons": alert["reasons"],
                "kill_reason": alert.get("kill_reason"),
                "us_leaders_status": us_sec.get("leaders", {}),
                "us_strong_count": us_sec.get("strong_count", 0),
                "us_min_strong": us_sec.get("min_strong", 2),
                "us_all_strong_enough": us_sec.get("all_strong_enough", False),
                "news_score": news_result["score"],
                "news_keywords_matched": news_result["keywords_matched"],
                "kr_leaders_strong": kr_strong,
                "kr_leaders_action": format_execution_summary(kr_leader_signals),
                "kr_secondaries_action": format_execution_summary(kr_secondary_signals),
                "summary": self._build_summary(sec_key, sec_config, alert, us_sec, news_result, kr_strong),
            }

            results[sec_key] = sector_result

            if phase >= 1:
                active_alerts.append(sec_key)
            if phase >= 4:
                execution_ready.append(sec_key)

            logger.info(
                "  → Phase %d (%s), Alert %d, US %d/%d, News %d, KR강세=%s",
                phase, phase_name, alert["alert_level"],
                us_sec.get("strong_count", 0), len(us_sec.get("leaders", {})),
                news_result["score"], kr_strong,
            )

        # 3) 전체 결과 조합
        total_alert = sum(r["alert_level"] for r in results.values())

        output = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sectors": results,
            "active_alerts": active_alerts,
            "execution_ready": execution_ready,
            "total_alert_score": total_alert,
            "recommendation": self._build_recommendation(results, execution_ready),
            "telegram_summary": self._build_telegram(results),
        }

        logger.info("=== 릴레이 엔진 완료: 활성 %d, 실행준비 %d ===",
                     len(active_alerts), len(execution_ready))

        return output

    # ── 뉴스 분석 ──

    def _analyze_news(self, sec_key: str, sec_config: dict,
                      ai_judgment: dict | None, market_news: dict | None) -> dict:
        """기존 뉴스 데이터에서 키워드 매칭."""
        keywords = [k.lower() for k in sec_config.get("alert_keywords", [])]
        negative_kw = [k.lower() for k in sec_config.get("negative_keywords", [])]

        matched = []
        negative_found = False
        score = 0

        # 소스 1: AI Brain sector_outlook
        if ai_judgment:
            sector_outlook = ai_judgment.get("sector_outlook", {})
            for sector_name, outlook in sector_outlook.items():
                direction = outlook.get("direction", "")
                reason = (outlook.get("reason", "") or "").lower()
                # 키워드 매칭
                for kw in keywords:
                    if kw in reason or kw in sector_name.lower():
                        matched.append(kw)
                        if direction == "positive":
                            score += 2
                        elif direction == "neutral":
                            score += 1

            # 뉴스 감정
            key_themes = ai_judgment.get("key_themes", [])
            if isinstance(key_themes, list):
                themes_text = " ".join(str(t) for t in key_themes).lower()
                for kw in keywords:
                    if kw in themes_text and kw not in matched:
                        matched.append(kw)
                        score += 1

        # 소스 2: market_news (RSS)
        if market_news:
            news_items = market_news.get("items", market_news.get("news", []))
            if isinstance(news_items, list):
                for item in news_items[:50]:  # 최근 50건만
                    title = str(item.get("title", "")).lower()
                    for kw in keywords:
                        if kw in title and kw not in matched:
                            matched.append(kw)
                            impact = str(item.get("impact", "")).lower()
                            if impact == "high":
                                score += 2
                            else:
                                score += 1

        # 부정 키워드 체크
        if negative_kw:
            all_text = ""
            if ai_judgment:
                all_text += json.dumps(ai_judgment.get("sector_outlook", {}), ensure_ascii=False).lower()
            if market_news:
                for item in (market_news.get("items", market_news.get("news", [])) or [])[:30]:
                    all_text += str(item.get("title", "")).lower() + " "
            for nkw in negative_kw:
                if nkw in all_text:
                    negative_found = True
                    break

        # 점수 정규화 (0~10)
        score = min(score, 10)

        return {
            "score": score,
            "keywords_matched": list(set(matched)),
            "negative_found": negative_found,
        }

    # ── 한국 대장주 확인 ──

    def _check_kr_leaders_strong(self, sec_config: dict, momentum_data: dict | None) -> bool:
        """한국 대장주가 속한 섹터의 모멘텀이 양호한지 확인."""
        if not momentum_data:
            return False

        kr_leaders = sec_config.get("kr_leaders", [])
        if not kr_leaders:
            return False

        # 섹터 모멘텀에서 해당 섹터 확인
        sectors_data = momentum_data.get("sectors", [])
        if not isinstance(sectors_data, list):
            return False

        # 한국 대장주 이름으로 섹터 매핑
        sec_name = sec_config.get("name", "")

        # 직접 매핑
        name_to_sector = {
            "AI 반도체": "반도체",
            "방산": "방산",
            "정유/에너지": "에너지화학",
            "배터리/ESS": "2차전지",
            "조선/LNG": "조선",
        }
        target_sector = name_to_sector.get(sec_name, "")

        for s in sectors_data:
            if s.get("sector", "") == target_sector:
                ret_5 = s.get("ret_5", s.get("ret_5d", 0))
                rank = s.get("rank", 99)
                # 5일 수익률 양수 OR 순위 10위 이내면 강세
                return ret_5 > 0 or rank <= 10

        return False

    # ── 요약 빌더 ──

    def _build_summary(self, sec_key: str, sec_config: dict, alert: dict,
                       us_sec: dict, news_result: dict, kr_strong: bool) -> str:
        name = sec_config.get("name", sec_key)
        phase_name = alert["phase_name"]
        us_count = us_sec.get("strong_count", 0)
        us_total = len(us_sec.get("leaders", {}))
        news_kw = ", ".join(news_result["keywords_matched"][:3]) if news_result["keywords_matched"] else "없음"

        if alert["phase"] == 0:
            return f"{name} 비활성"
        elif alert["phase"] == 1:
            return f"{name} 예비경보 — US {us_count}/{us_total}, 뉴스: {news_kw}"
        elif alert["phase"] == 2:
            return f"{name} 본경보 — US {us_count}/{us_total} 강세, 뉴스: {news_kw}. KR 대장주 돌파 대기"
        elif alert["phase"] == 3:
            return f"{name} 실행준비 — US {us_count}/{us_total}↑, KR 대장주 전일 강세, 돌파 대기"
        else:
            return f"{name} 실행가능 — US {us_count}/{us_total}↑ + 뉴스 + KR 확인. 진입 가능"

    def _build_recommendation(self, results: dict, execution_ready: list) -> str:
        if execution_ready:
            names = [results[k]["name"] for k in execution_ready]
            return f"{', '.join(names)} 섹터 실행 준비 완료. 전일고가 돌파 시 진입."

        active = [k for k, v in results.items() if v["phase"] >= 2]
        if active:
            names = [results[k]["name"] for k in active]
            return f"{', '.join(names)} 본경보 활성. KR 대장주 확인 대기."

        watch = [k for k, v in results.items() if v["phase"] >= 1]
        if watch:
            names = [results[k]["name"] for k in watch]
            return f"{', '.join(names)} 예비경보. US 추가 확인 대기."

        return "전 섹터 비활성. 경보 없음."

    def _build_telegram(self, results: dict) -> str:
        """텔레그램 메시지 생성."""
        lines = [f"[릴레이 경보 {datetime.now().strftime('%m-%d %H:%M')}]", ""]

        stars_map = {0: "", 1: "*", 2: "**", 3: "***", 4: "****", 5: "*****"}

        # 활성 섹터 먼저
        for sec_key, r in sorted(results.items(), key=lambda x: -x[1]["alert_level"]):
            name = r["name"]
            phase_name = r["phase_name"]
            alert = r["alert_level"]
            stars = stars_map.get(alert, "")

            if r["phase"] == 0:
                lines.append(f"  {name}: 비활성")
                continue

            lines.append(f"  {name} [{phase_name} {stars}]")

            # US 대장주 상태
            us_parts = []
            for t, info in r.get("us_leaders_status", {}).items():
                arrow = "+" if info.get("is_strong") else ("-" if info.get("is_weak") else "=")
                us_parts.append(f"{t} ${info.get('close', 0)}{arrow}")
            if us_parts:
                lines.append(f"  US: {' | '.join(us_parts)}")

            # 뉴스
            kw = r.get("news_keywords_matched", [])
            if kw:
                lines.append(f"  News: {', '.join(kw[:3])}")

            # KR 액션
            kr_actions = []
            for name_kr, action_info in r.get("kr_leaders_action", {}).items():
                kr_actions.append(f"{name_kr}({action_info['action']})")
            for name_kr, action_info in r.get("kr_secondaries_action", {}).items():
                kr_actions.append(f"{name_kr}({action_info['action']})")
            if kr_actions:
                lines.append(f"  KR: {' | '.join(kr_actions)}")

            lines.append("")

        return "\n".join(lines)

    # ── 유틸리티 ──

    def _load_json(self, rel_path: str) -> dict | None:
        path = DATA_ROOT / rel_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("JSON 로드 실패 (%s): %s", rel_path, e)
            return None


def run_relay() -> dict:
    """릴레이 엔진 실행 + 결과 저장."""
    engine = RelayEngine()
    result = engine.run()

    # 결과 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("릴레이 시그널 저장: %s", SIGNAL_OUTPUT)

    # 히스토리 누적
    _append_history(result)

    return result


def _append_history(result: dict):
    """릴레이 경보 이력 누적."""
    history = []
    if HISTORY_OUTPUT.exists():
        try:
            history = json.loads(HISTORY_OUTPUT.read_text(encoding="utf-8"))
        except Exception:
            history = []

    entry = {
        "date": result["date"],
        "active_alerts": result["active_alerts"],
        "execution_ready": result["execution_ready"],
        "total_alert_score": result["total_alert_score"],
        "sector_phases": {
            k: {"phase": v["phase"], "alert_level": v["alert_level"]}
            for k, v in result.get("sectors", {}).items()
        },
    }
    history.append(entry)

    # 최근 90일만 유지
    history = history[-90:]

    HISTORY_OUTPUT.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
