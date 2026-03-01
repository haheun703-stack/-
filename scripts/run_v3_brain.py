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


# ─── Phase 1: Strategic Brain ───────────────────────────────────

async def run_phase1(dry_run: bool = False) -> dict:
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
    overnight = _load_json(DATA_DIR / "overnight_signal.json")
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
) -> list[dict]:
    """Phase 3+4: 후보 추출 → 정밀 분석

    Phase 3: scan_cache.json에서 파이프라인 통과 종목 추출 + sector boost/suppress
    Phase 4: DeepAnalyst 배치 분석 → conviction 필터

    Args:
        strategic_result: Phase 1 결과
        sector_focus: Phase 2 결과

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

    agent = DeepAnalystAgent()
    passed = await agent.analyze_batch(
        candidates=filtered,
        industry_thesis=industry_thesis,
        sector_focus=sector_focus,
        min_conviction=min_conviction,
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
) -> dict:
    """Phase 5: 최종 포트폴리오 결정 (Agent 2E).

    Deep Analyst 통과 종목 + 현재 포지션 + 레짐 → 최종 매수 결정.

    Args:
        deep_picks: Phase 4 통과 종목
        strategic_result: Phase 1 결과

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

    if not deep_picks:
        logger.info("Phase 4 통과 종목 없음 — Phase 5 결과: 매수 0")
        result = PortfolioBrainAgent._fallback_result("후보 없음", strategic_result)
        result.pop("error", None)
        result["reasoning"] = "Deep Analyst 통과 종목이 없어 매수 보류"
    else:
        agent = PortfolioBrainAgent()
        result = await agent.decide(deep_picks, positions, strategic_result)

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


# ─── 메인 ───────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="v3 AI Brain 5단계 깔때기 러너")
    parser.add_argument("--dry-run", action="store_true", help="분석만 수행, 매수 없음")
    parser.add_argument("--phase", type=int, default=5, help="실행할 최대 Phase (기본: 5)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 알림 생략")
    parser.add_argument("--skip-phase1", action="store_true", help="Phase 1 생략 (기존 ai_strategic_analysis.json 사용)")
    args = parser.parse_args()

    settings = _load_settings()
    if not settings.get("enabled", True):
        logger.info("ai_brain_v3.enabled=false — 스킵")
        return

    start_time = datetime.now()
    logger.info("v3 AI Brain 러너 시작 (phase=%d, dry_run=%s)", args.phase, args.dry_run)

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
        result = await run_phase1(dry_run=args.dry_run)

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
            picks = await run_phase3_4(result, sector_focus)
        except Exception as e:
            logger.error("Phase 3+4 실패: %s", e)
            picks = []

    # Phase 5: Portfolio Brain
    final_picks = None
    if args.phase >= 5 and picks is not None:
        try:
            final_picks = await run_phase5(picks, result)
        except Exception as e:
            logger.error("Phase 5 실패: %s", e)
            final_picks = None

    # 텔레그램
    if not args.no_telegram and not result.get("error"):
        await _send_telegram_summary(result, sector_focus, picks, final_picks)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("v3 AI Brain 러너 완료 (%.1f초)", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
