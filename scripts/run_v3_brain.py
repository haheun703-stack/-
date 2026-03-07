"""v3 AI Brain 5단계 깔때기 러너

실행:
  python scripts/run_v3_brain.py [--dry-run] [--phase N] [--no-telegram]

5단계 깔때기:
  Phase 1: StrategicBrain (Opus) → ai_strategic_analysis.json
  Phase 2: SectorStrategist → ai_sector_focus.json
  Phase 3: scan_cache.json에서 후보 추출 + sector boost/suppress
  Phase 4: DeepAnalyst (Sonnet×N) → conviction 필터
  Phase 5: PortfolioBrain (Opus) → ai_v3_picks.json (최종 포트폴리오)

Phase 1~5 구현 완료.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# PYTHONPATH 안전장치 (BAT 스케줄 실행 시 필수)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.agents.strategic_brain import StrategicBrainAgent
from src.agents.sector_strategist import SectorStrategistAgent
from src.agents.deep_analyst import DeepAnalystAgent
from src.agents.portfolio_brain import PortfolioBrainAgent
from src.agents.o1_strategist import O1StrategistAgent
from src.agents.perplexity_verifier import PerplexityVerifier

# ─── 로깅 설정 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v3_brain")

# ─── 데이터 경로 ────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_json(path: Path) -> dict:
    """JSON 파일 로드 (없으면 빈 dict)"""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("JSON 로드 실패 %s: %s", path.name, e)
    else:
        logger.warning("파일 없음: %s", path)
    return {}


def _save_json(path: Path, data: dict) -> None:
    """JSON 파일 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장 완료: %s", path.name)


def _load_settings() -> dict:
    """settings.yaml의 ai_brain_v3 섹션 로드"""
    try:
        import yaml
        settings_path = CONFIG_DIR / "settings.yaml"
        with open(settings_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("ai_brain_v3", {})
    except Exception as e:
        logger.error("settings.yaml 로드 실패: %s", e)
        return {}


def _load_upgrade_settings() -> dict:
    """settings.yaml의 ai_upgrade 섹션 로드"""
    try:
        import yaml
        settings_path = CONFIG_DIR / "settings.yaml"
        with open(settings_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("ai_upgrade", {})
    except Exception:
        return {}


# ─── BRAIN 지시서 로드 (v13.9: 매크로→종목 연동) ─────────────────

def _load_brain_cap() -> dict:
    """brain_decision.json에서 BRAIN 캡 정보를 로드.

    Returns:
        dict: {regime, swing_pct, confidence, slot_cap, min_conviction, capped}
        실패 시 기본값 반환 (기존 로직 그대로 동작).
    """
    import yaml
    try:
        settings_path = CONFIG_DIR / "settings.yaml"
        with open(settings_path, encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
    except Exception:
        full_cfg = {}

    integ = full_cfg.get("brain_v3_integration", {})
    if not integ.get("enabled", False):
        return {"capped": False, "reason": "brain_v3_integration disabled"}

    brain = _load_json(DATA_DIR / "brain_decision.json")
    if not brain or not brain.get("arms"):
        logger.warning("BRAIN 지시서 없음 — 기존 로직으로 독립 실행")
        return {"capped": False, "reason": "brain_decision.json 없음"}

    # stale 체크
    stale_hours = integ.get("stale_hours", 24)
    ts = brain.get("timestamp", "")
    regime = brain.get("effective_regime", "NEUTRAL")
    try:
        decision_time = datetime.fromisoformat(ts)
        age_hours = (datetime.now() - decision_time).total_seconds() / 3600
        if age_hours > stale_hours:
            logger.warning("BRAIN 지시서 %.1f시간 경과 — 보수적 기본값", age_hours)
            regime = integ.get("stale_default_regime", "CAUTION")
    except Exception:
        pass

    # swing_pct 추출
    swing_pct = 30.0  # 기본값
    for arm in brain.get("arms", []):
        if arm.get("name") == "swing":
            swing_pct = arm.get("adjusted_pct", 30.0)
            break

    confidence = brain.get("confidence", 0.5)

    # 레짐별 슬롯 캡
    slot_caps = integ.get("regime_slot_cap", {})
    slot_cap = slot_caps.get(regime, 99)  # 미정의 레짐은 제한 없음

    # confidence → min_conviction
    conv_map = integ.get("confidence_conviction_map", [])
    min_conviction = 4  # 기본값
    for entry in conv_map:
        if confidence < entry.get("below", 1.01):
            min_conviction = entry.get("min_conviction", 4)
            break

    logger.info(
        "BRAIN 캡 로드: 레짐=%s, swing=%.1f%%, confidence=%.2f → 슬롯캡=%d, min_conviction=%d",
        regime, swing_pct, confidence, slot_cap, min_conviction,
    )

    return {
        "capped": True,
        "regime": regime,
        "swing_pct": swing_pct,
        "confidence": confidence,
        "slot_cap": slot_cap,
        "min_conviction": min_conviction,
    }


# ─── Phase 0: o1 Deep Thinking (NEW) ─────────────────────────────

async def run_phase0() -> dict:
    """Phase 0: GPT o1 Deep Thinking 거시/미시 분석.

    o1의 deep reasoning으로 거시적+미시적 분석을 수행하여
    Phase 1 StrategicBrain에 컨텍스트로 주입.

    Returns:
        o1_deep_analysis.json 내용 (실패 시 빈 dict)
    """
    upgrade = _load_upgrade_settings()
    if not upgrade.get("o1_enabled", False):
        logger.info("Phase 0 비활성화 (ai_upgrade.o1_enabled=false)")
        return {}

    logger.info("=" * 60)
    logger.info("Phase 0: o1 Deep Thinking (거시/미시 분석)")
    logger.info("=" * 60)

    try:
        agent = O1StrategistAgent()
        result = await agent.deep_analyze()

        # 저장
        output_path = DATA_DIR / "o1_deep_analysis.json"
        _save_json(output_path, result)

        if result.get("error"):
            logger.warning("Phase 0 fallback: %s", result["error"])
        else:
            macro = result.get("macro_analysis", {})
            logger.info(
                "Phase 0 완료: regime=%s, confidence=%.0f%%, 미시 %d섹터",
                macro.get("macro_regime", "?"),
                macro.get("confidence", 0) * 100,
                len(result.get("micro_analysis", [])),
            )

        return result
    except Exception as e:
        logger.error("Phase 0 실패 (Phase 1 독립 동작): %s", e)
        return {}


# ─── Phase 1: Strategic Brain ───────────────────────────────────

async def run_phase1(dry_run: bool = False, o1_context: dict | None = None) -> dict:
    """Phase 1: 전략 두뇌 (Agent 2A) 실행

    5개 소스를 로드하여 StrategicBrainAgent에 전달.
    결과를 ai_strategic_analysis.json에 저장.

    Args:
        dry_run: True이면 저장만 하고 실매매 안 함 (기본 동작)

    Returns:
        ai_strategic_analysis.json 내용
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Strategic Brain (v3 Agent 2A) 시작")
    logger.info("=" * 60)

    # ── 5개 소스 로드 ──
    overnight = _load_json(DATA_DIR / "us_market" / "overnight_signal.json")
    news = _load_json(DATA_DIR / "ai_brain_judgment.json")
    sector_flow = _load_json(
        DATA_DIR / "sector_rotation" / "etf_trading_signal.json"
    )
    relay_patterns = _load_json(DATA_DIR / "relay_pattern_db.json")
    portfolio = _load_json(DATA_DIR / "positions.json")

    # 소스 유효성 로그
    sources_loaded = sum([
        bool(overnight),
        bool(news),
        bool(sector_flow),
        bool(relay_patterns),
        bool(portfolio),
    ])
    logger.info(
        "소스 로드: %d/5 (overnight=%s, news=%s, sector=%s, relay=%s, portfolio=%s)",
        sources_loaded,
        "O" if overnight else "X",
        "O" if news else "X",
        "O" if sector_flow else "X",
        "O" if relay_patterns else "X",
        "O" if portfolio else "X",
    )

    if sources_loaded < 2:
        logger.warning("유효 소스 2개 미만 — 방어 모드로 전환")
        result = StrategicBrainAgent._fallback_result("소스 데이터 부족")
        output_path = DATA_DIR / "ai_strategic_analysis.json"
        _save_json(output_path, result)
        return result

    # ── 피드백 로드 (있으면) ──
    feedback = ""
    feedback_path = DATA_DIR / "weekly_accuracy.json"
    if feedback_path.exists():
        try:
            fb = _load_json(feedback_path)
            if fb.get("feedback_text"):
                feedback = fb["feedback_text"]
                logger.info("주간 피드백 로드 완료 (%d자)", len(feedback))
        except Exception:
            pass

    # ── Agent 2A 실행 ──
    context = {
        "global_market": overnight,
        "news": news,
        "sector_flow": sector_flow,
        "relay_patterns": relay_patterns,
        "portfolio": portfolio,
        "feedback": feedback,
    }

    # o1 Deep Thinking 컨텍스트 주입 (Phase 0 결과)
    if o1_context and not o1_context.get("error"):
        context["o1_deep_analysis"] = o1_context
        logger.info("o1 Deep Thinking 컨텍스트 주입 완료")

    agent = StrategicBrainAgent()
    result = await agent.analyze(context)

    # ── 결과 저장 ──
    settings = _load_settings()
    output_path = Path(settings.get("regime_output", "data/ai_strategic_analysis.json"))
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    _save_json(output_path, result)

    # ── 결과 요약 로그 ──
    logger.info("-" * 60)
    logger.info("Phase 1 결과 요약:")
    logger.info("  레짐: %s (신뢰도: %.0f%%)", result.get("regime"), result.get("regime_confidence", 0) * 100)
    logger.info("  레짐 근거: %s", result.get("regime_reasoning", "")[:100])

    theses = result.get("industry_thesis", [])
    if theses:
        logger.info("  산업 Thesis (%d개):", len(theses))
        for t in theses[:3]:
            logger.info(
                "    - %s: %s (확신 %d/10)",
                t.get("sector", "?"),
                t.get("thesis", "?")[:60],
                t.get("confidence", 0),
            )

    priority = result.get("sector_priority", {})
    logger.info("  공격: %s", ", ".join(priority.get("attack", [])) or "없음")
    logger.info("  관찰: %s", ", ".join(priority.get("watch", [])) or "없음")
    logger.info("  회피: %s", ", ".join(priority.get("avoid", [])) or "없음")
    logger.info("  최대 신규매수: %d종목", result.get("max_new_buys", 0))
    logger.info("  현금 비중 권고: %d%%", result.get("cash_reserve_suggestion", 20))

    alerts = result.get("relay_alerts", [])
    if alerts:
        logger.info("  릴레이 알림:")
        for a in alerts:
            logger.info("    - %s: %s → %s", a.get("pattern"), a.get("status"), a.get("action"))

    logger.info("-" * 60)

    return result


# ─── Phase 2: Sector Strategist ────────────────────────────────

async def run_phase2(strategic_result: dict) -> dict:
    """Phase 2: 섹터 전략가 (Agent 2B) 실행

    Phase 1의 산업 thesis와 ETF 모멘텀 데이터를 교차 검증하여
    집중 섹터 + boost/suppress 목록 생성.

    Args:
        strategic_result: Phase 1 결과 (ai_strategic_analysis.json)

    Returns:
        ai_sector_focus.json 내용
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Sector Strategist (v3 Agent 2B) 시작")
    logger.info("=" * 60)

    # ── 소스 로드 ──
    sector_flow = _load_json(
        DATA_DIR / "sector_rotation" / "etf_trading_signal.json"
    )
    relay_db = _load_json(DATA_DIR / "relay_pattern_db.json")

    if not sector_flow:
        logger.warning("ETF 모멘텀 데이터 없음 — Phase 2 스킵")
        return SectorStrategistAgent._fallback_result("ETF 데이터 없음")

    # ── Agent 2B 실행 ──
    agent = SectorStrategistAgent()
    result = await agent.focus(strategic_result, sector_flow, relay_db)

    # ── 결과 저장 ──
    settings = _load_settings()
    output_path = Path(settings.get("sector_output", "data/ai_sector_focus.json"))
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    _save_json(output_path, result)

    # ── 결과 요약 로그 ──
    logger.info("-" * 60)
    logger.info("Phase 2 결과 요약:")
    focus = result.get("focus_sectors", [])
    logger.info("  집중 섹터: %d개", len(focus))
    for s in focus[:5]:
        logger.info(
            "    - %s: 타이밍=%s, 정합=%s, 비중=%.1fx (순위 %s)",
            s.get("sector", "?"),
            s.get("entry_timing", "?"),
            s.get("thesis_alignment", "?"),
            s.get("size_weight", 1.0),
            s.get("momentum_rank", "?"),
        )

    boost = result.get("screening_boost", [])
    suppress = result.get("screening_suppress", [])
    logger.info("  부스트: %s", ", ".join(boost) or "없음")
    logger.info("  억제: %s", ", ".join(suppress) or "없음")

    warnings = result.get("sector_warnings", [])
    if warnings:
        logger.info("  경고:")
        for w in warnings:
            logger.info("    - [%s] %s: %s", w.get("severity", "?"), w.get("sector", "?"), w.get("warning", "?"))

    logger.info("-" * 60)

    return result


# ─── Phase 3+4: 후보 추출 + Deep Analyst ──────────────────────

async def run_phase3_4(
    strategic_result: dict,
    sector_focus: dict,
    brain_cap: dict | None = None,
) -> list[dict]:
    """Phase 3+4: 후보 추출 → 정밀 분석

    Phase 3: scan_cache.json에서 파이프라인 통과 종목 추출 + sector boost/suppress
    Phase 4: DeepAnalyst 배치 분석 → conviction 필터

    Args:
        strategic_result: Phase 1 결과
        sector_focus: Phase 2 결과
        brain_cap: BRAIN 캡 정보 (None이면 기존 로직)

    Returns:
        conviction 기준 통과한 종목 리스트
    """
    logger.info("=" * 60)
    logger.info("Phase 3+4: 후보 추출 + Deep Analyst (v3 Agent 2D) 시작")
    logger.info("=" * 60)

    # ── Phase 3: scan_cache에서 후보 추출 ──
    scan_cache = _load_json(DATA_DIR / "scan_cache.json")

    # v9 생존자 + v9에서 제거된 종목 합산 (v8 파이프라인 통과 전체)
    candidates = list(scan_cache.get("candidates", []))
    killed = scan_cache.get("stats", {}).get("v9_killed_list", [])
    if killed:
        existing_tickers = {c.get("ticker") for c in candidates}
        for k in killed:
            if k.get("ticker") not in existing_tickers:
                candidates.append(k)
        logger.info("v9 killed 종목 %d개 복원 (AI가 재판단)", len(killed))

    if not candidates:
        logger.warning("scan_cache 후보 0 — Phase 3+4 스킵")
        return []

    # sector boost/suppress 적용
    boost_sectors = set(sector_focus.get("screening_boost", []))
    suppress_sectors = set(sector_focus.get("screening_suppress", []))

    # 공격 섹터 목록 (Phase 1)
    attack_sectors = set(
        strategic_result.get("sector_priority", {}).get("attack", [])
    )

    # boost 섹터에 해당하는 종목 우선 포함
    boosted = []
    normal = []
    suppressed = []

    for c in candidates:
        sector = c.get("sector", "")
        if sector in suppress_sectors:
            suppressed.append(c)
        elif sector in boost_sectors or sector in attack_sectors:
            boosted.append(c)
        else:
            normal.append(c)

    # 부스트 우선 + 일반 + 억제(제외)
    filtered = boosted + normal
    logger.info(
        "Phase 3 후보: %d종목 (부스트 %d + 일반 %d, 억제 %d 제외)",
        len(filtered), len(boosted), len(normal), len(suppressed),
    )

    if not filtered:
        logger.warning("후보 종목 0 — Phase 4 스킵")
        return []

    # 배치 상한
    settings = _load_settings()
    max_batch = settings.get("deep_analyst", {}).get("max_batch_size", 30)
    if len(filtered) > max_batch:
        logger.info("후보 %d → %d로 제한", len(filtered), max_batch)
        filtered = filtered[:max_batch]

    # ── Phase 4: DeepAnalyst 배치 분석 ──
    industry_thesis = strategic_result.get("industry_thesis", [])
    min_conviction = settings.get("deep_analyst", {}).get("min_conviction", 5)

    # BRAIN 캡: confidence 낮으면 conviction 상향
    if brain_cap and brain_cap.get("capped"):
        brain_min = brain_cap.get("min_conviction", min_conviction)
        if brain_min > min_conviction:
            logger.info(
                "BRAIN 캡 적용: min_conviction %d → %d (confidence=%.2f)",
                min_conviction, brain_min, brain_cap.get("confidence", 0),
            )
            min_conviction = brain_min

    # Vision 활성화 여부
    upgrade = _load_upgrade_settings()
    enable_vision = upgrade.get("vision_enabled", False)

    agent = DeepAnalystAgent()
    passed = await agent.analyze_batch(
        candidates=filtered,
        industry_thesis=industry_thesis,
        sector_focus=sector_focus,
        min_conviction=min_conviction,
        enable_vision=enable_vision,
    )

    # ── 결과 로그 ──
    logger.info("-" * 60)
    logger.info("Phase 4 결과 요약:")
    logger.info("  분석 대상: %d종목 → 통과: %d종목 (conviction>=%d)", len(filtered), len(passed), min_conviction)

    for p in passed[:10]:
        logger.info(
            "    - %s(%s): conviction=%d, strategy=%s, thesis=%s",
            p.get("name", "?"),
            p.get("ticker", "?"),
            p.get("conviction", 0),
            p.get("strategy", "?"),
            p.get("thesis_alignment", "?"),
        )

    logger.info("-" * 60)

    # ── 결과 저장 (임시 — Phase 5에서 최종 picks로 변환) ──
    total_scanned = scan_cache.get("stats", {}).get("total", len(candidates))
    picks_data = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "phase": "phase4_deep_analyst",
        "total_scanned": total_scanned,
        "pipeline_passed": len(candidates),
        "boost_filtered": len(filtered),
        "conviction_passed": len(passed),
        "min_conviction": min_conviction,
        "picks": passed,
    }

    picks_path = Path(settings.get("picks_output", "data/ai_v3_picks.json"))
    if not picks_path.is_absolute():
        picks_path = PROJECT_ROOT / picks_path
    _save_json(picks_path, picks_data)

    return passed


# ─── Phase 5: Portfolio Brain ─────────────────────────────────

async def run_phase5(
    deep_picks: list[dict],
    strategic_result: dict,
    brain_cap: dict | None = None,
) -> dict:
    """Phase 5: 최종 포트폴리오 결정 (Agent 2E).

    Deep Analyst 통과 종목 + 현재 포지션 + 레짐 → 최종 매수 결정.

    Args:
        deep_picks: Phase 4 통과 종목
        strategic_result: Phase 1 결과
        brain_cap: BRAIN 캡 정보 (None이면 기존 로직)

    Returns:
        ai_v3_picks.json 최종 형식
    """
    logger.info("=" * 60)
    logger.info("Phase 5: Portfolio Brain (v3 Agent 2E) 시작")
    logger.info("=" * 60)

    # 현재 포지션 로드
    positions = _load_json(DATA_DIR / "positions.json")
    if isinstance(positions, dict):
        positions = positions.get("positions", [])

    logger.info("현재 보유: %d종목, Phase 4 후보: %d종목", len(positions), len(deep_picks))

    # ── BRAIN 슬롯 캡 적용 ──
    if brain_cap and brain_cap.get("capped"):
        slot_cap = brain_cap["slot_cap"]
        original_max = strategic_result.get("max_new_buys", 99)
        if slot_cap < original_max:
            strategic_result["max_new_buys"] = slot_cap
            logger.info(
                "BRAIN 캡 적용됨: 레짐=%s, 슬롯캡=%d, 원래요청=%d → 캡적용=%d",
                brain_cap.get("regime", "?"), slot_cap, original_max, slot_cap,
            )
        else:
            logger.info(
                "BRAIN 캡 불필요: 레짐=%s, 슬롯캡=%d, 원래요청=%d (캡 이내)",
                brain_cap.get("regime", "?"), slot_cap, original_max,
            )
        # BRAIN 레짐 정보를 strategic_result에 주입 (Opus 참조용)
        strategic_result["brain_regime"] = brain_cap.get("regime", "NEUTRAL")
        strategic_result["brain_swing_pct"] = brain_cap.get("swing_pct", 30)

    if not deep_picks:
        logger.info("Phase 4 통과 종목 없음 — Phase 5 결과: 매수 0")
        result = PortfolioBrainAgent._fallback_result("후보 없음", strategic_result)
        result.pop("error", None)
        result["reasoning"] = "Deep Analyst 통과 종목이 없어 매수 보류"
    else:
        agent = PortfolioBrainAgent()
        result = await agent.decide(deep_picks, positions, strategic_result)

    # ── BRAIN 종목당 배분 상한 적용 ──
    if brain_cap and brain_cap.get("capped") and result.get("buys"):
        swing_pct = brain_cap.get("swing_pct", 30)
        n_buys = len(result["buys"])
        if n_buys > 0:
            max_per_stock = swing_pct / n_buys
            capped_any = False
            for buy in result["buys"]:
                original = buy.get("size_pct", 0)
                if original > max_per_stock:
                    buy["size_pct"] = round(max_per_stock, 1)
                    buy["brain_size_capped"] = True
                    capped_any = True
            if capped_any:
                logger.info(
                    "BRAIN 배분캡: swing %.1f%% / %d종목 = 종목당 최대 %.1f%%",
                    swing_pct, n_buys, max_per_stock,
                )
        result["brain_cap_applied"] = {
            "regime": brain_cap.get("regime"),
            "slot_cap": brain_cap.get("slot_cap"),
            "swing_pct": swing_pct,
            "min_conviction": brain_cap.get("min_conviction"),
        }

    # 결과 저장
    settings = _load_settings()
    picks_path = Path(settings.get("picks_output", "data/ai_v3_picks.json"))
    if not picks_path.is_absolute():
        picks_path = PROJECT_ROOT / picks_path
    _save_json(picks_path, result)

    # 결과 로그
    logger.info("-" * 60)
    logger.info("Phase 5 결과 요약:")
    buys = result.get("buys", [])
    logger.info("  매수 결정: %d종목", len(buys))
    for b in buys:
        logger.info(
            "    - %s(%s): conviction=%d, 비중=%.1f%%, strategy=%s",
            b.get("name", "?"),
            b.get("ticker", "?"),
            b.get("conviction", 0),
            b.get("size_pct", 0),
            b.get("strategy", "?"),
        )

    skipped = result.get("skipped", [])
    if skipped:
        logger.info("  스킵: %d종목", len(skipped))
        for s in skipped:
            logger.info("    - %s: %s", s.get("name", "?"), s.get("skip_reason", "?"))

    warnings = result.get("portfolio_warnings", [])
    if warnings:
        for w in warnings:
            logger.info("  경고: %s", w)

    logger.info("-" * 60)
    return result


# ─── Phase 6: Perplexity 교차검증 (NEW) ──────────────────────────

async def run_phase6(final_picks: dict, strategic_result: dict) -> dict:
    """Phase 6: Perplexity 교차검증.

    Phase 5 결과의 매수 종목 촉매/리스크를 웹검색으로 팩트체크.
    Phase 1의 산업 thesis도 교차검증.

    Args:
        final_picks: Phase 5 결과 (ai_v3_picks.json)
        strategic_result: Phase 1 결과 (ai_strategic_analysis.json)

    Returns:
        perplexity_verification.json 내용 (실패 시 빈 dict)
    """
    upgrade = _load_upgrade_settings()
    if not upgrade.get("perplexity_verify_enabled", False):
        logger.info("Phase 6 비활성화 (ai_upgrade.perplexity_verify_enabled=false)")
        return {}

    logger.info("=" * 60)
    logger.info("Phase 6: Perplexity 교차검증")
    logger.info("=" * 60)

    buys = final_picks.get("buys", [])
    if not buys:
        logger.info("매수 종목 없음 — Phase 6 스킵")
        return {}

    try:
        max_v = upgrade.get("perplexity_max_verifications", 5)
        verifier = PerplexityVerifier(max_verifications=max_v)
        result = verifier.verify_picks(final_picks, strategic_result)

        # 저장
        output_path = DATA_DIR / "perplexity_verification.json"
        _save_json(output_path, result)

        # 로그
        logger.info("-" * 60)
        logger.info("Phase 6 결과 요약:")
        logger.info("  종합 신뢰도: %.0f%%", result.get("overall_confidence", 0) * 100)
        logger.info(
            "  종목 검증: %d건, thesis 검증: %d건",
            len(result.get("stock_verifications", [])),
            len(result.get("thesis_verifications", [])),
        )

        warnings = result.get("warnings", [])
        if warnings:
            for w in warnings:
                logger.warning("  ⚠️ %s", w)

        hallucinations = result.get("hallucination_flags", [])
        if hallucinations:
            logger.warning("  🚨 환각 감지 %d건!", len(hallucinations))
            for h in hallucinations:
                logger.warning("    - %s: confidence=%.2f", h.get("name", "?"), h.get("confidence_score", 0))

        logger.info("-" * 60)
        return result

    except Exception as e:
        logger.error("Phase 6 실패 (검증 없이 진행): %s", e)
        return {}


# ─── 텔레그램 알림 ──────────────────────────────────────────────

async def _send_telegram_summary(
    result: dict,
    sector_focus: dict | None = None,
    picks: list[dict] | None = None,
    final_picks: dict | None = None,
) -> None:
    """v3 Brain 전체 결과를 텔레그램으로 전송"""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        logger.warning("telegram_sender 임포트 실패 — 알림 생략")
        return

    regime = result.get("regime", "?")
    confidence = result.get("regime_confidence", 0)
    max_buys = result.get("max_new_buys", 0)
    cash = result.get("cash_reserve_suggestion", 20)

    regime_emoji = {"공격": "🟢", "중립": "🟡", "방어": "🟠", "회피": "🔴"}.get(regime, "⚪")

    lines = [
        f"🧠 v3 AI Brain 판단",
        f"",
        f"{regime_emoji} 레짐: {regime} ({confidence:.0%})",
        f"📊 최대 매수: {max_buys}종목 | 현금: {cash}%",
    ]

    # 산업 Thesis
    theses = result.get("industry_thesis", [])
    if theses:
        lines.append(f"\n📋 산업 Thesis ({len(theses)}개):")
        for t in theses[:3]:
            sector = t.get("sector", "?")
            conf = t.get("confidence", 0)
            thesis_text = t.get("thesis", "?")[:40]
            lines.append(f"  • {sector} ({conf}/10): {thesis_text}")

    # 섹터 우선순위
    priority = result.get("sector_priority", {})
    attack = priority.get("attack", [])
    avoid = priority.get("avoid", [])
    if attack:
        lines.append(f"\n🎯 공격: {', '.join(attack)}")
    if avoid:
        lines.append(f"⛔ 회피: {', '.join(avoid)}")

    # Phase 2 — 섹터 포커스
    if sector_focus:
        boost = sector_focus.get("screening_boost", [])
        suppress = sector_focus.get("screening_suppress", [])
        focus_sectors = sector_focus.get("focus_sectors", [])

        if focus_sectors:
            lines.append(f"\n🔍 섹터 포커스 ({len(focus_sectors)}개):")
            for s in focus_sectors[:3]:
                lines.append(
                    f"  • {s.get('sector', '?')}: "
                    f"{s.get('entry_timing', '?')} "
                    f"({s.get('thesis_alignment', '?')})"
                )
        if boost:
            lines.append(f"⬆️ 부스트: {', '.join(boost)}")
        if suppress:
            lines.append(f"⬇️ 억제: {', '.join(suppress)}")

    # Phase 4 — Deep Analyst picks
    if picks:
        lines.append(f"\n🎯 Deep Analyst 통과: {len(picks)}종목")
        for p in picks[:5]:
            lines.append(
                f"  • {p.get('name', '?')}({p.get('ticker', '?')}): "
                f"확신 {p.get('conviction', 0)}/10 "
                f"[{p.get('strategy', '?')}]"
            )

    # Phase 5 — 최종 매수 결정
    if final_picks:
        buys = final_picks.get("buys", [])
        if buys:
            lines.append(f"\n💰 최종 매수 결정: {len(buys)}종목")
            for b in buys[:5]:
                lines.append(
                    f"  • {b.get('name', '?')}({b.get('ticker', '?')}): "
                    f"비중 {b.get('size_pct', 0):.0f}% "
                    f"[{b.get('strategy', '?')}]"
                )
        else:
            lines.append(f"\n💰 최종 매수: 없음")
        if final_picks.get("reasoning"):
            lines.append(f"  → {final_picks['reasoning'][:60]}")

    # 릴레이 알림
    alerts = result.get("relay_alerts", [])
    if alerts:
        lines.append(f"\n🔗 릴레이:")
        for a in alerts[:2]:
            lines.append(f"  • {a.get('pattern')}: {a.get('action')}")

    # 리스크
    risks = result.get("risk_factors", [])
    if risks:
        lines.append(f"\n⚠️ 리스크: {', '.join(risks[:3])}")

    text = "\n".join(lines)

    try:
        await send_message(text)
        logger.info("텔레그램 전송 완료")
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


# ─── Phase 7: 학습 루프 ──────────────────────────────────────────

def daily_review() -> dict:
    """Agent 7A — 일일 성과 리뷰

    당일 v3 Brain 판단 vs 실제 시장 결과를 비교하여
    data/daily_performance.json에 누적.

    비교 항목:
      - regime 적중 여부 (KOSPI 방향 vs 판단)
      - sector_priority 적중 여부 (공격 섹터 수익률)
      - 릴레이 예측 적중 여부
      - v3 picks 수익률

    Returns:
        오늘자 리뷰 결과
    """
    logger.info("=" * 60)
    logger.info("Agent 7A: 일일 성과 리뷰")
    logger.info("=" * 60)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── v3 판단 로드 ──
    strategic = _load_json(DATA_DIR / "ai_strategic_analysis.json")
    sector_focus = _load_json(DATA_DIR / "ai_sector_focus.json")
    v3_picks = _load_json(DATA_DIR / "ai_v3_picks.json")

    if not strategic:
        logger.warning("ai_strategic_analysis.json 없음 — 리뷰 스킵")
        return {}

    analysis_date = strategic.get("analysis_date", "")
    if analysis_date != today_str:
        logger.info("오늘(%s) 분석 아님(%s) — 리뷰 스킵", today_str, analysis_date)
        return {}

    # ── 실제 시장 데이터 확인 ──
    review = {
        "date": today_str,
        "regime_predicted": strategic.get("regime", "?"),
        "regime_confidence": strategic.get("regime_confidence", 0),
        "attack_sectors": strategic.get("sector_priority", {}).get("attack", []),
        "avoid_sectors": strategic.get("sector_priority", {}).get("avoid", []),
    }

    # KOSPI 실제 방향 (ETF 데이터에서 추출)
    try:
        import pandas as pd
        kospi_path = DATA_DIR / "kospi_index.csv"
        if kospi_path.exists():
            df = pd.read_csv(kospi_path)
            if len(df) >= 2:
                col = "Close" if "Close" in df.columns else "close"
                last_close = float(df.iloc[-1][col])
                prev_close = float(df.iloc[-2][col])
                kospi_chg = (last_close / prev_close - 1) * 100
                review["kospi_change_pct"] = round(kospi_chg, 2)

                # 레짐 판단 적중 여부
                regime = strategic.get("regime", "")
                if regime == "공격" and kospi_chg > 0:
                    review["regime_hit"] = True
                elif regime == "방어" and kospi_chg < 0:
                    review["regime_hit"] = True
                elif regime == "회피" and kospi_chg < -1:
                    review["regime_hit"] = True
                elif regime == "중립" and abs(kospi_chg) < 1:
                    review["regime_hit"] = True
                else:
                    review["regime_hit"] = False
    except Exception as e:
        logger.warning("KOSPI 데이터 확인 실패: %s", e)

    # 섹터 모멘텀 결과 (ETF 시그널에서 추출)
    etf_signal = _load_json(DATA_DIR / "sector_rotation" / "etf_trading_signal.json")
    if etf_signal and "sectors" in etf_signal:
        attack_returns = []
        avoid_returns = []
        for sec in etf_signal.get("sectors", []):
            sec_name = sec.get("sector", "")
            day_return = sec.get("return_1d", 0)
            if sec_name in review.get("attack_sectors", []):
                attack_returns.append({"sector": sec_name, "return_1d": day_return})
            if sec_name in review.get("avoid_sectors", []):
                avoid_returns.append({"sector": sec_name, "return_1d": day_return})

        if attack_returns:
            avg_attack = sum(r["return_1d"] for r in attack_returns) / len(attack_returns)
            review["attack_sector_avg_return"] = round(avg_attack, 2)
            review["attack_sector_details"] = attack_returns
        if avoid_returns:
            avg_avoid = sum(r["return_1d"] for r in avoid_returns) / len(avoid_returns)
            review["avoid_sector_avg_return"] = round(avg_avoid, 2)
            # 회피 섹터가 실제로 하락했으면 적중
            review["avoid_hit"] = avg_avoid < 0

    # v3 picks 결과
    if v3_picks and "buys" in v3_picks:
        buy_results = []
        for buy in v3_picks.get("buys", []):
            ticker = buy.get("ticker", "")
            entry_price = buy.get("entry_price", 0)
            if ticker and entry_price:
                # processed parquet에서 당일 종가 확인
                try:
                    pq_path = DATA_DIR / "processed" / f"{ticker}.parquet"
                    if pq_path.exists():
                        df = pd.read_parquet(pq_path)
                        if len(df) > 0:
                            last_close = float(df.iloc[-1]["close"])
                            pnl = (last_close / entry_price - 1) * 100
                            buy_results.append({
                                "ticker": ticker,
                                "name": buy.get("name", ""),
                                "entry_price": entry_price,
                                "current_close": last_close,
                                "pnl_pct": round(pnl, 2),
                                "conviction": buy.get("conviction", 0),
                            })
                except Exception:
                    pass
        if buy_results:
            review["v3_picks_results"] = buy_results
            review["v3_picks_avg_pnl"] = round(
                sum(r["pnl_pct"] for r in buy_results) / len(buy_results), 2
            )

    # ── 누적 저장 ──
    perf_path = DATA_DIR / "daily_performance.json"
    history = []
    if perf_path.exists():
        try:
            with open(perf_path, encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []

    # 같은 날짜 중복 방지
    history = [h for h in history if h.get("date") != today_str]
    history.append(review)

    # 최근 30일만 유지
    history = history[-30:]

    _save_json(perf_path, history)

    # 로그
    logger.info("일일 리뷰 결과:")
    logger.info("  레짐 판단: %s (적중: %s)", review.get("regime_predicted"), review.get("regime_hit", "N/A"))
    if "kospi_change_pct" in review:
        logger.info("  KOSPI 변동: %.2f%%", review["kospi_change_pct"])
    if "attack_sector_avg_return" in review:
        logger.info("  공격 섹터 평균: %.2f%%", review["attack_sector_avg_return"])
    if "v3_picks_avg_pnl" in review:
        logger.info("  v3 picks 평균 수익: %.2f%%", review["v3_picks_avg_pnl"])

    return review


def weekly_review() -> dict:
    """Agent 7B — 주간 정확도 분석

    daily_performance.json 누적 데이터를 집계하여
    data/weekly_accuracy.json에 피드백 텍스트 생성.
    → 다음 주 StrategicBrain 프롬프트에 자동 주입.

    금요일 장후 자동 실행 (BAT-D 22단계에서 DOW==4 체크).

    Returns:
        주간 분석 결과 + feedback_text
    """
    logger.info("=" * 60)
    logger.info("Agent 7B: 주간 정확도 분석")
    logger.info("=" * 60)

    perf_path = DATA_DIR / "daily_performance.json"
    if not perf_path.exists():
        logger.warning("daily_performance.json 없음 — 주간 리뷰 스킵")
        return {}

    try:
        with open(perf_path, encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        logger.warning("daily_performance.json 로드 실패: %s", e)
        return {}

    if not history or len(history) < 3:
        logger.info("데이터 %d일분 — 최소 3일 필요, 스킵", len(history))
        return {}

    # ── 최근 5일(영업일) 데이터 집계 ──
    recent = history[-5:]
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 레짐 적중률
    regime_hits = [r for r in recent if r.get("regime_hit") is not None]
    regime_accuracy = (
        sum(1 for r in regime_hits if r["regime_hit"]) / len(regime_hits)
        if regime_hits else 0
    )

    # 공격 섹터 평균 수익률
    attack_returns = [
        r["attack_sector_avg_return"]
        for r in recent
        if "attack_sector_avg_return" in r
    ]
    avg_attack_return = sum(attack_returns) / len(attack_returns) if attack_returns else 0

    # 회피 섹터 적중률
    avoid_hits = [r for r in recent if r.get("avoid_hit") is not None]
    avoid_accuracy = (
        sum(1 for r in avoid_hits if r["avoid_hit"]) / len(avoid_hits)
        if avoid_hits else 0
    )

    # v3 picks 평균 수익률
    picks_pnls = [
        r["v3_picks_avg_pnl"]
        for r in recent
        if "v3_picks_avg_pnl" in r
    ]
    avg_picks_pnl = sum(picks_pnls) / len(picks_pnls) if picks_pnls else 0

    result = {
        "review_date": today_str,
        "period_days": len(recent),
        "regime_accuracy": round(regime_accuracy, 2),
        "regime_details": [
            {"date": r["date"], "predicted": r.get("regime_predicted"), "hit": r.get("regime_hit")}
            for r in regime_hits
        ],
        "attack_sector_avg_return": round(avg_attack_return, 2),
        "avoid_accuracy": round(avoid_accuracy, 2),
        "v3_picks_avg_pnl": round(avg_picks_pnl, 2),
    }

    # ── 피드백 텍스트 생성 (Phase 1 프롬프트에 주입) ──
    feedback_lines = [
        f"[주간 피드백 {today_str}] 최근 {len(recent)}일 성과:",
        f"- 레짐 판단 적중률: {regime_accuracy:.0%} ({sum(1 for r in regime_hits if r['regime_hit'])}/{len(regime_hits)})" if regime_hits else "- 레짐 적중률: 데이터 부족",
        f"- 공격 섹터 평균 수익: {avg_attack_return:+.2f}%" if attack_returns else "- 공격 섹터: 데이터 부족",
        f"- 회피 섹터 적중률: {avoid_accuracy:.0%}" if avoid_hits else "- 회피 섹터: 데이터 부족",
        f"- v3 picks 평균 수익: {avg_picks_pnl:+.2f}%" if picks_pnls else "- v3 picks: 데이터 부족",
    ]

    # 개선 제안
    if regime_accuracy < 0.5 and regime_hits:
        feedback_lines.append("⚠️ 레짐 판단 정확도 낮음 — 거시 소스 가중치 재검토 필요")
    if avg_attack_return < 0 and attack_returns:
        feedback_lines.append("⚠️ 공격 섹터 수익 마이너스 — 모멘텀 과열 종목 회피 필요")
    if avg_picks_pnl < -2 and picks_pnls:
        feedback_lines.append("⚠️ v3 picks 평균 손실 — conviction 기준 상향 검토")

    result["feedback_text"] = "\n".join(feedback_lines)

    # 저장
    _save_json(DATA_DIR / "weekly_accuracy.json", result)

    # 로그
    logger.info("주간 리뷰 결과:")
    logger.info("  레짐 적중률: %.0f%%", regime_accuracy * 100)
    logger.info("  공격 섹터 평균: %+.2f%%", avg_attack_return)
    logger.info("  회피 적중률: %.0f%%", avoid_accuracy * 100)
    logger.info("  v3 picks 평균: %+.2f%%", avg_picks_pnl)
    logger.info("  피드백 텍스트 (%d자) → weekly_accuracy.json", len(result["feedback_text"]))

    return result


# ─── Phase 4.5: ICT 전술 필터 ─────────────────────────────────

def _apply_ict_filter(picks: list[dict]) -> list[dict]:
    """ICT 프리미엄 레벨 + OR/IR bias로 conviction 보정.

    - confidence_adjust(-0.2~+0.1) × 10 → conviction 보정(-2~+1)
    - 보정 후 min_conviction 이하 → 자동 필터링
    - data/ict_log/{date}.json에 로그 누적

    Args:
        picks: Phase 4 통과 종목 리스트

    Returns:
        ICT 보정 적용된 picks (필터링 포함)
    """
    try:
        from src.ict.ict_filter import ict_check
    except ImportError:
        logger.warning("ICT 모듈 미설치 — Phase 4.5 스킵")
        return picks

    if not picks:
        return picks

    settings = _load_settings()
    min_conviction = settings.get("deep_analyst", {}).get("min_conviction", 5)
    today_str = datetime.now().strftime("%Y-%m-%d")

    log_records = []
    adjusted_picks = []
    filtered_count = 0

    for pick in picks:
        ticker = pick.get("ticker", "")
        name = pick.get("name", ticker)
        original_conviction = pick.get("conviction", 0)

        try:
            result = ict_check(ticker, "buy", today_str)
        except Exception as e:
            logger.debug("ICT 체크 실패 %s: %s", ticker, e)
            adjusted_picks.append(pick)
            continue

        adj = result.get("confidence_adjust", 0.0)
        conviction_delta = int(round(adj * 10))  # -2 ~ +1
        new_conviction = max(0, min(10, original_conviction + conviction_delta))

        # 로그 레코드
        record = {
            "symbol": ticker,
            "name": name,
            "signal_type": "buy",
            "daily_bias": result.get("daily_bias", "unknown"),
            "confidence_adjust": adj,
            "conviction_before": original_conviction,
            "conviction_delta": conviction_delta,
            "conviction_after": new_conviction,
            "tags": result.get("tags", []),
            "reason": result.get("reason", ""),
            "suggested_target": result.get("suggested_target"),
            "suggested_stop": result.get("suggested_stop"),
            "filtered": False,
        }

        if conviction_delta != 0:
            pick["conviction"] = new_conviction
            pick["ict_adjust"] = conviction_delta
            pick["ict_tags"] = result.get("tags", [])
            pick["ict_reason"] = result.get("reason", "")

        # 필터링 체크
        if new_conviction < min_conviction:
            record["filtered"] = True
            filtered_count += 1
            logger.info(
                "[ICT] %s(%s) FILTERED: conviction %d→%d (<%d) | bias=%s | %s",
                name, ticker, original_conviction, new_conviction,
                min_conviction, result.get("daily_bias", "?"), result.get("reason", ""),
            )
        else:
            adjusted_picks.append(pick)
            if conviction_delta != 0:
                logger.info(
                    "[ICT] %s(%s) adj=%+d: conviction %d→%d | bias=%s | %s",
                    name, ticker, conviction_delta, original_conviction,
                    new_conviction, result.get("daily_bias", "?"), result.get("reason", ""),
                )

        log_records.append(record)

    # ICT 로그 저장
    _save_ict_log(today_str, log_records, filtered_count)

    if filtered_count > 0:
        logger.info(
            "Phase 4.5 ICT 필터: %d종목 → %d종목 (-%d)",
            len(picks), len(adjusted_picks), filtered_count,
        )
    else:
        adj_count = sum(1 for r in log_records if r["conviction_delta"] != 0)
        logger.info(
            "Phase 4.5 ICT: %d종목, %d종목 conviction 보정, 필터링 0",
            len(picks), adj_count,
        )

    return adjusted_picks


def _save_ict_log(date_str: str, records: list[dict], filtered_count: int) -> None:
    """ICT 체크 로그를 data/ict_log/{date}.json에 저장."""
    log_dir = DATA_DIR / "ict_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{date_str}.json"

    positive = sum(1 for r in records if r["conviction_delta"] > 0)
    negative = sum(1 for r in records if r["conviction_delta"] < 0)
    neutral = sum(1 for r in records if r["conviction_delta"] == 0)
    avg_adj = (
        round(sum(r["confidence_adjust"] for r in records) / len(records), 4)
        if records else 0
    )

    log_data = {
        "date": date_str,
        "total_checks": len(records),
        "adjustments": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
        "filtered_out": filtered_count,
        "avg_adjustment": avg_adj,
        "records": records,
    }

    _save_json(log_path, log_data)
    logger.info("ICT 로그 저장: %s (%d건, +%d/-%d/=%d, filtered=%d)",
                log_path.name, len(records), positive, negative, neutral, filtered_count)


# ─── 메인 ───────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="v3 AI Brain 5단계 깔때기 러너")
    parser.add_argument("--dry-run", action="store_true", help="분석만 수행, 매수 없음")
    parser.add_argument("--phase", type=int, default=5, help="실행할 최대 Phase (기본: 5)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 알림 생략")
    parser.add_argument("--skip-phase1", action="store_true", help="Phase 1 생략 (기존 ai_strategic_analysis.json 사용)")
    parser.add_argument("--review", action="store_true", help="일일 성과 리뷰만 실행 (Phase 7A)")
    parser.add_argument("--weekly-review", action="store_true", help="주간 정확도 분석 실행 (Phase 7B)")
    args = parser.parse_args()

    settings = _load_settings()
    if not settings.get("enabled", True):
        logger.info("ai_brain_v3.enabled=false — 스킵")
        return

    # ── 리뷰 전용 모드 ──
    if args.review:
        daily_review()
        return
    if args.weekly_review:
        daily_review()  # 주간 리뷰 전에 당일 리뷰도 실행
        weekly_review()
        return

    start_time = datetime.now()
    logger.info("v3 AI Brain 러너 시작 (phase=%d, dry_run=%s)", args.phase, args.dry_run)

    # ── BRAIN 지시서 로드 (매크로→종목 연동) ──
    brain_cap = _load_brain_cap()

    # Phase 0: o1 Deep Thinking (NEW)
    o1_context = {}
    if not args.skip_phase1:
        try:
            o1_context = await run_phase0()
        except Exception as e:
            logger.error("Phase 0 예외 (Phase 1 독립 진행): %s", e)

    # Phase 1: Strategic Brain
    if args.skip_phase1:
        logger.info("Phase 1 생략 — 기존 ai_strategic_analysis.json 로드")
        output_path = Path(settings.get("regime_output", "data/ai_strategic_analysis.json"))
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        result = _load_json(output_path)
        if not result:
            logger.error("ai_strategic_analysis.json 없음 — 종료")
            return
    else:
        result = await run_phase1(dry_run=args.dry_run, o1_context=o1_context)

    sector_focus = None
    picks = None

    # Phase 2: Sector Strategist
    if args.phase >= 2 and not result.get("error"):
        try:
            sector_focus = await run_phase2(result)
        except Exception as e:
            logger.error("Phase 2 실패 (계속 진행): %s", e)
            sector_focus = SectorStrategistAgent._fallback_result(str(e))

    # Phase 3+4: 후보 추출 + Deep Analyst
    if args.phase >= 4 and sector_focus and not sector_focus.get("error"):
        try:
            picks = await run_phase3_4(result, sector_focus, brain_cap=brain_cap)
        except Exception as e:
            logger.error("Phase 3+4 실패: %s", e)
            picks = []

    # Phase 4.5: ICT 전술 필터 (Phase 4 conviction 보정)
    if picks:
        picks = _apply_ict_filter(picks)

    # Phase 5: Portfolio Brain
    final_picks = None
    if args.phase >= 5 and picks is not None:
        try:
            final_picks = await run_phase5(picks, result, brain_cap=brain_cap)
        except Exception as e:
            logger.error("Phase 5 실패: %s", e)
            final_picks = None

    # Phase 6: Perplexity 교차검증 (NEW)
    if final_picks and final_picks.get("buys"):
        try:
            verification = await run_phase6(final_picks, result)

            # HALLUCINATION 판정 종목 자동 제거
            if verification:
                hallucination_tickers = set()
                for v in verification.get("stock_verifications", []):
                    if v.get("verdict") == "HALLUCINATION_DETECTED":
                        hallucination_tickers.add(v.get("ticker", ""))
                for h in verification.get("hallucination_flags", []):
                    hallucination_tickers.add(h.get("ticker", ""))
                hallucination_tickers.discard("")

                if hallucination_tickers:
                    before_count = len(final_picks["buys"])
                    removed = [b for b in final_picks["buys"] if b.get("ticker") in hallucination_tickers]
                    final_picks["buys"] = [b for b in final_picks["buys"] if b.get("ticker") not in hallucination_tickers]
                    for r in removed:
                        final_picks.setdefault("skipped", []).append({
                            "ticker": r.get("ticker", ""),
                            "name": r.get("name", ""),
                            "skip_reason": "Perplexity HALLUCINATION_DETECTED — 촉매 팩트체크 실패",
                        })
                    logger.warning(
                        "HALLUCINATION 자동 제거: %d종목 (%s) → 최종 %d종목",
                        len(removed),
                        ", ".join(r.get("name", "?") for r in removed),
                        len(final_picks["buys"]),
                    )

                    # 변경된 최종 picks 재저장
                    settings = _load_settings()
                    picks_path = Path(settings.get("picks_output", "data/ai_v3_picks.json"))
                    if not picks_path.is_absolute():
                        picks_path = PROJECT_ROOT / picks_path
                    _save_json(picks_path, final_picks)

        except Exception as e:
            logger.error("Phase 6 예외 (검증 없이 진행): %s", e)

    # 텔레그램
    if not args.no_telegram and not result.get("error"):
        await _send_telegram_summary(result, sector_focus, picks, final_picks)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("v3 AI Brain 러너 완료 (%.1f초)", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
