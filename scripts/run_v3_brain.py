"""v3 AI Brain 5단계 깔때기 러너

실행:
  python scripts/run_v3_brain.py [--dry-run] [--phase N] [--no-telegram]

5단계 깔때기:
  Phase 1: StrategicBrain (Opus) → ai_strategic_analysis.json
  Phase 2: SectorStrategist → ai_sector_focus.json       (미구현)
  Phase 3: SignalEngine 기존 스캔 + sector boost → 후보   (미구현)
  Phase 4: DeepAnalyst (Sonnet×N) → conviction 필터       (미구현)
  Phase 5: PortfolioBrain (Opus) → ai_v3_picks.json       (미구현)

현재 Phase 1만 구현. Phase 2~5는 후속 커밋에서 추가.
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


# ─── 텔레그램 알림 ──────────────────────────────────────────────

async def _send_telegram_summary(result: dict) -> None:
    """Phase 1 결과를 텔레그램으로 전송"""
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
        f"🧠 v3 Strategic Brain 판단",
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
    parser.add_argument("--phase", type=int, default=1, help="실행할 최대 Phase (기본: 1)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 알림 생략")
    args = parser.parse_args()

    settings = _load_settings()
    if not settings.get("enabled", True):
        logger.info("ai_brain_v3.enabled=false — 스킵")
        return

    start_time = datetime.now()
    logger.info("v3 AI Brain 러너 시작 (phase=%d, dry_run=%s)", args.phase, args.dry_run)

    # Phase 1
    result = await run_phase1(dry_run=args.dry_run)

    # Phase 2~5는 미구현 — 후속 커밋에서 추가
    if args.phase >= 2:
        logger.info("Phase 2~5는 미구현 — Phase 1 결과만 사용")

    # 텔레그램
    if not args.no_telegram and not result.get("error"):
        await _send_telegram_summary(result)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("v3 AI Brain 러너 완료 (%.1f초)", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
