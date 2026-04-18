"""
COO Orchestrator — Quantum Master 파이프라인 지휘관

BAT-D 31단계를 그룹별로 추적하고, 실패 시 폴백을 적용하며,
전체 결과를 coo_run_log.json에 기록합니다.

무슨 일이 있어도 FLOWX에 무언가 발행됩니다.

사용법:
  python coo_orchestrator.py              # 전체 파이프라인 실행
  python coo_orchestrator.py --dry-run    # 실행 없이 로그만
  python coo_orchestrator.py --group 데이터수집  # 특정 그룹만
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [COO] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "coo.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("coo")

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
VENV_PYTHON = str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")


# ═══════════════════════════════════════════════════════════════
#  파이프라인 그룹 정의 (BAT-D 실제 구조 기반)
# ═══════════════════════════════════════════════════════════════

PIPELINE_GROUPS = [
    {
        "group": "데이터수집",
        "description": "CSV/Parquet 종가 + KOSPI 인덱스 + 분봉",
        "critical": True,
        "fallback": "use_cached_data",
        "steps": [
            {"name": "CSV 전종목 업데이트", "cmd": "scripts/update_daily_data.py"},
            {"name": "Parquet 증분 업데이트", "cmd": "scripts/extend_parquet_data.py --skip-supply"},
            {"name": "KOSPI 인덱스", "cmd": "scripts/update_kospi_index.py"},
            {"name": "투자자수급", "cmd": "scripts/collect_investor_flow.py"},
            {"name": "ECOS 매크로 수집", "cmd": "scripts/fetch_ecos_macro.py"},
        ],
    },
    {
        "group": "지표계산",
        "description": "기술지표 + US 시그널 + 파생 + 공매도 + 패턴DB",
        "critical": True,
        "fallback": "use_previous_indicators",
        "steps": [
            {"name": "기술지표 재계산", "cmd": "scripts/rebuild_indicators.py"},
            {"name": "ICT 레벨+OR/IR", "cmd": "scripts/run_ict_levels.py"},
            {"name": "US Overnight Signal", "cmd": "scripts/us_overnight_signal.py --update"},
            {"name": "파생 시그널 수집", "cmd": "scripts/derivatives_collector.py"},
            # {"name": "공매도 시그널", "cmd": "scripts/collect_short_selling.py"},  # KRX 데이터 제공 중단
            {"name": "US-KR 패턴DB", "cmd": "scripts/update_us_kr_daily.py"},
        ],
    },
    {
        "group": "섹터ETF분석",
        "description": "섹터 모멘텀 + 수급 + ETF 마스터 + 시그널",
        "critical": False,
        "fallback": "use_previous_sector_data",
        "steps": [
            {"name": "섹터 ETF 시세", "cmd": "scripts/sector_etf_builder.py --daily"},
            {"name": "섹터 모멘텀", "cmd": "scripts/sector_momentum.py --history"},
            {"name": "섹터 z-score", "cmd": "scripts/sector_zscore.py --top 5"},
            {"name": "섹터 수급", "cmd": "scripts/sector_investor_flow.py --days 5"},
            {"name": "차이나머니 수급", "cmd": "scripts/crawl_china_money.py"},
            {"name": "ETF 투자자 수급", "cmd": "scripts/collect_etf_investor_flow.py"},
            {"name": "KRX 국적별 외국인", "cmd": "scripts/scan_nationality.py --send"},
            {"name": "외인소진율 수집", "cmd": "scripts/collect_foreign_exhaustion.py"},
            {"name": "섹터 일일 리포트", "cmd": "scripts/sector_daily_report.py"},
            {"name": "TIER2 기관수급 수집", "cmd": "scripts/institutional_flow_collector.py"},
            {"name": "TIER2 섹터 컴포짓", "cmd": "src/sector_composite.py"},
            {"name": "섹터 ETF 거래량", "cmd": "scripts/collect_etf_volume.py"},
            {"name": "ETF 마스터 빌드", "cmd": "scripts/update_etf_master.py"},
            {"name": "ETF 매매 시그널", "cmd": "scripts/etf_trading_signal.py"},
        ],
    },
    {
        "group": "BRAIN_SHIELD",
        "description": "SHIELD 방어 → BRAIN 배분 → LENS 맥락",
        "critical": False,
        "fallback": "use_previous_brain_decision",
        "steps": [
            {"name": "SHIELD 방어 점검", "cmd": "scripts/run_shield.py --send"},
            {"name": "COT Slow Eye", "cmd": "scripts/run_cot_tracker.py"},
            {"name": "매크로 레짐 분석", "cmd": "-m src.use_cases.macro_regime"},
            {"name": "BRAIN 자본배분", "cmd": "scripts/run_brain.py"},
            {"name": "SCENARIO ENGINE", "cmd": "-m src.alpha.scenario_detector"},
            {"name": "LENS LAYER", "cmd": "scripts/run_lens.py"},
            {"name": "ETF 3축 로테이션", "cmd": "scripts/run_etf_rotation.py --blind-test --no-telegram"},
        ],
    },
    {
        "group": "종목스캔",
        "description": "각종 스캔 + AI분석 + 내일 추천",
        "critical": False,
        "fallback": "use_previous_candidates",
        "steps": [
            {"name": "레버리지 ETF 스캔", "cmd": "scripts/leverage_etf_scanner.py"},
            {"name": "눌림목 스캔", "cmd": "scripts/scan_pullback.py"},
            {"name": "수급 폭발 스캐너", "cmd": "scripts/scan_volume_spike.py"},
            {"name": "소형주 급등 포착", "cmd": "scripts/scan_smallcap_explosion.py"},
            {"name": "밸류체인 스캔", "cmd": "scripts/scan_value_chain.py"},
            {"name": "DART 공시 크롤링", "cmd": "scripts/crawl_dart_disclosure.py"},
            {"name": "레짐 매크로 시그널", "cmd": "scripts/regime_macro_signal.py"},
            {"name": "시장 뉴스 크롤링", "cmd": "scripts/crawl_market_news.py"},
            {"name": "세력감지 스캔", "cmd": "scripts/scan_whale_detect.py"},
            {"name": "세력감지 하이브리드", "cmd": "scripts/scan_force_hybrid.py"},
            {"name": "동반매수 스캔", "cmd": "scripts/scan_dual_buying.py"},
            {"name": "섹터 릴레이 시그널", "cmd": "scripts/relay_report.py"},
            {"name": "그룹 릴레이 감지", "cmd": "scripts/group_relay_detector.py"},
            {"name": "매집 추적 스캔", "cmd": "scripts/scan_accumulation_tracker.py"},
            {"name": "실적 가속도 분석", "cmd": "scripts/scan_earnings_acceleration.py"},
            {"name": "턴어라운드 스크리닝", "cmd": "scripts/scan_turnaround.py"},
        ],
    },
    {
        "group": "추천생성",
        "description": "성과추적 + AI두뇌 + 내일 추천 통합",
        "critical": False,
        "fallback": "use_previous_candidates",
        "steps": [
            {"name": "추천 성과 추적", "cmd": "scripts/track_pick_results.py"},
            {"name": "DART 이벤트 시그널", "cmd": "scripts/dart_event_signal.py"},
            {"name": "기관 목표가 계산", "cmd": "scripts/calc_institutional_targets.py"},
            {"name": "보유종목 재판정", "cmd": "scripts/position_monitor.py"},
            {"name": "CPI 트래커", "cmd": "scripts/update_cpi_data.py"},
            {"name": "Perplexity 인텔리전스", "cmd": "scripts/perplexity_market_intel.py"},
            {"name": "AI 두뇌 뉴스 분석", "cmd": "scripts/ai_news_brain.py"},
            {"name": "v3 AI Brain", "cmd": "scripts/run_v3_brain.py --no-telegram"},
            {"name": "컨센서스 스크리닝", "cmd": "scripts/scan_consensus.py --top 15 --send"},
            {"name": "내일 추천 종목 스캔", "cmd": "scripts/scan_tomorrow_picks.py"},
            {"name": "일일 시장 학습", "cmd": "scripts/daily_market_learner.py"},
            {"name": "포트폴리오 배분", "cmd": "scripts/portfolio_allocator.py"},
            {"name": "3단 예측 체인", "cmd": "scripts/run_predict_chain.py --send --blind"},
            {"name": "Master Brain 통합", "cmd": "scripts/run_master_brain.py"},
            {"name": "매크로 체인 감지", "cmd": "-m src.alpha.macro_chain_detector"},
            {"name": "ETF 추천 엔진", "cmd": "-m src.alpha.etf_engine"},
            {"name": "킬러픽 생성", "cmd": "scripts/build_killer_picks.py"},
            {"name": "타입1 수급릴레이 스캔", "cmd": "scripts/scan_type1_relay.py"},
            {"name": "타입2 바닥반등 스캔", "cmd": "scripts/scan_type2_bottom.py"},
            {"name": "급락반등 스캔", "cmd": "scripts/scan_crash_bounce.py"},
            {"name": "피보나치 스캔", "cmd": "scripts/scan_fibonacci.py"},
        ],
    },
    {
        "group": "FLOWX발행",
        "description": "아카이브 + 대시보드 + 텔레그램 + 페이퍼트레이딩",
        "critical": False,
        "fallback": "publish_minimal_flowx",
        "steps": [
            {"name": "일일 아카이브", "cmd": "src/daily_archive.py"},
            {"name": "v3 일일 리뷰", "cmd": "scripts/run_v3_brain.py --review"},
            {"name": "JARVIS 업로드", "cmd": "-c \"from src.adapters.jarvis_uploader import main; main()\""},
            {"name": "Brain 대시보드 빌드", "cmd": "scripts/build_brain_upload.py"},
            {"name": "FLOWX Supabase 업로드", "cmd": "scripts/upload_flowx.py"},
            {"name": "Market Journal", "cmd": "scripts/market_journal.py"},
            {"name": "저녁 통합 리포트", "cmd": "scripts/send_evening_summary.py --send"},
            {"name": "페이퍼 트레이딩", "cmd": "scripts/paper_trading_unified.py"},
            {"name": "데이터 건강검진", "cmd": "scripts/data_health_check.py"},
            {"name": "CFO 포트폴리오 리포트", "cmd": "src/use_cases/portfolio_cfo.py"},
            {"name": "CTO 시스템 리포트", "cmd": "scripts/run_cto.py"},
        ],
    },
    {
        "group": "분봉아카이브",
        "description": "장마감 분봉 수집 (non-critical, 타임아웃 허용)",
        "critical": False,
        "fallback": "skip_silently",
        "steps": [
            {"name": "분봉 데이터 수집", "cmd": "scripts/collect_intraday_candles.py --top-n 200"},
        ],
    },
]


class COOOrchestrator:
    """Quantum Master 파이프라인 지휘관."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.log_data = {
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "groups": [],
            "summary": {},
        }

    # ─── 메인 실행 ───

    def run(self, target_group: str | None = None):
        """전체 파이프라인 또는 특정 그룹 실행."""
        logger.info("=" * 60)
        logger.info("COO 파이프라인 시작 (dry_run=%s)", self.dry_run)
        logger.info("=" * 60)

        groups = PIPELINE_GROUPS
        if target_group:
            groups = [g for g in groups if g["group"] == target_group]
            if not groups:
                logger.error("그룹 '%s' 없음. 가능: %s",
                             target_group, [g["group"] for g in PIPELINE_GROUPS])
                return

        aborted = False
        for group in groups:
            result = self._run_group(group)
            self.log_data["groups"].append(result)

            if not result["success"] and group["critical"]:
                logger.error("CRITICAL 그룹 '%s' 실패 → 파이프라인 중단", group["group"])
                aborted = True
                break

        # 무슨 일이 있어도 FLOWX는 발행
        self._ensure_flowx_published(aborted)

        # 요약
        self._build_summary()
        self._save_log()

        logger.info("=" * 60)
        logger.info("COO 파이프라인 완료: %s", self.log_data["summary"])
        logger.info("=" * 60)

    # ─── 그룹 실행 ───

    def _run_group(self, group: dict) -> dict:
        """하나의 그룹(여러 step)을 실행."""
        group_name = group["group"]
        logger.info("─── 그룹 [%s] 시작 (%d단계) ───", group_name, len(group["steps"]))

        result = {
            "group": group_name,
            "description": group["description"],
            "critical": group["critical"],
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "success": True,
            "failed_count": 0,
            "fallback_used": False,
        }

        for step in group["steps"]:
            step_result = self._run_step(step)
            result["steps"].append(step_result)

            if not step_result["success"]:
                result["failed_count"] += 1

        # 그룹 내 전체 실패 시 폴백
        total = len(group["steps"])
        if result["failed_count"] == total:
            result["success"] = False
            fallback_name = group["fallback"]
            logger.warning("그룹 [%s] 전체 실패 → 폴백: %s", group_name, fallback_name)
            self._execute_fallback(fallback_name)
            result["fallback_used"] = True
        elif result["failed_count"] > 0:
            # 부분 실패는 경고만 (그룹은 성공 처리)
            logger.warning("그룹 [%s] 부분 실패: %d/%d", group_name, result["failed_count"], total)

        result["finished_at"] = datetime.now().isoformat()
        elapsed = (datetime.fromisoformat(result["finished_at"])
                   - datetime.fromisoformat(result["started_at"])).total_seconds()
        result["elapsed_seconds"] = round(elapsed, 1)

        logger.info("─── 그룹 [%s] 완료 (%.0f초, 실패 %d/%d) ───",
                     group_name, elapsed, result["failed_count"], total)
        return result

    # ─── 단일 스텝 실행 ───

    def _run_step(self, step: dict) -> dict:
        """하나의 스크립트를 subprocess로 실행."""
        name = step["name"]
        cmd = step["cmd"]

        step_result = {
            "name": name,
            "cmd": cmd,
            "started_at": datetime.now().isoformat(),
            "success": False,
            "return_code": None,
            "error": None,
        }

        if self.dry_run:
            logger.info("  [DRY] %s → %s", name, cmd)
            step_result["success"] = True
            step_result["return_code"] = 0
            return step_result

        # 명령어 조립
        if cmd.startswith("-m ") or cmd.startswith("-c "):
            flag = cmd[:2]
            rest = cmd[3:]
            full_cmd = [VENV_PYTHON, "-u", "-X", "utf8", flag, rest]
        else:
            script_path = str(PROJECT_ROOT / cmd.split()[0])
            args = cmd.split()[1:]
            full_cmd = [VENV_PYTHON, "-u", "-X", "utf8", script_path] + args

        logger.info("  [RUN] %s", name)

        try:
            proc = subprocess.run(
                full_cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=2400,  # 40분 타임아웃 (CSV 업데이트 25분 소요)
                env={
                    **__import__("os").environ,
                    "PYTHONPATH": str(PROJECT_ROOT),
                },
            )
            step_result["return_code"] = proc.returncode
            step_result["success"] = proc.returncode == 0

            if proc.returncode != 0:
                # 에러 출력 마지막 3줄만 기록
                stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-3:]) if proc.stderr else ""
                step_result["error"] = stderr_tail or f"exit code {proc.returncode}"
                logger.warning("  [FAIL] %s (exit %d): %s", name, proc.returncode, stderr_tail[:200])
            else:
                logger.info("  [OK] %s", name)

        except subprocess.TimeoutExpired:
            step_result["error"] = f"TIMEOUT ({2400}s)"
            logger.error("  [TIMEOUT] %s", name)
        except Exception as e:
            step_result["error"] = str(e)
            logger.error("  [ERROR] %s: %s", name, e)

        step_result["finished_at"] = datetime.now().isoformat()
        return step_result

    # ─── 폴백 핸들러 ───

    def _execute_fallback(self, fallback_name: str):
        """폴백 전략 실행. 이전 데이터를 유지하거나 최소 출력 생성."""
        logger.info("  [FALLBACK] %s 실행", fallback_name)

        if fallback_name == "use_cached_data":
            logger.info("  → 캐시된 parquet/CSV 데이터 유지 (오늘 수집 스킵)")

        elif fallback_name == "use_previous_indicators":
            logger.info("  → 이전 지표 데이터 유지 (재계산 스킵)")

        elif fallback_name == "use_previous_sector_data":
            logger.info("  → 이전 섹터/ETF 데이터 유지")

        elif fallback_name == "use_previous_brain_decision":
            brain_path = DATA_DIR / "brain_decision.json"
            if brain_path.exists():
                logger.info("  → 이전 BRAIN 결정 유지: %s", brain_path)
            else:
                logger.warning("  → BRAIN 결정 파일 없음 — 보수적 모드 적용")

        elif fallback_name == "use_previous_candidates":
            picks_path = DATA_DIR / "tomorrow_picks.json"
            if picks_path.exists():
                logger.info("  → 이전 추천 종목 유지: %s", picks_path)
            else:
                logger.warning("  → 추천 종목 파일 없음")

        elif fallback_name == "publish_minimal_flowx":
            self._publish_emergency_flowx("FLOWX 발행 그룹 실패 — 최소 발행")

    # ─── FLOWX 보장 ───

    def _ensure_flowx_published(self, aborted: bool):
        """무슨 일이 있어도 FLOWX에 무언가 나가야 한다."""
        if self.dry_run:
            logger.info("[FLOWX] dry-run 모드 — 발행 스킵")
            return

        if aborted:
            self._publish_emergency_flowx("파이프라인 중단 — 데이터 수집 실패")
            return

        # 정상 완료 시: dashboard_data.py가 이미 FLOWX 그룹에서 실행됨
        # 추가로 flowx_message를 확인
        brain_upload = PROJECT_ROOT / "website" / "data" / "brain_data_upload.json"
        if brain_upload.exists():
            mtime = datetime.fromtimestamp(brain_upload.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < 2:
                logger.info("[FLOWX] 정상 발행 확인 (%.0f분 전 생성)", age_hours * 60)
            else:
                logger.warning("[FLOWX] brain_data_upload.json 이 %.1f시간 전 — 갱신 필요", age_hours)
        else:
            logger.warning("[FLOWX] brain_data_upload.json 없음 — 최소 발행")
            self._publish_emergency_flowx("대시보드 데이터 미생성")

    def _publish_emergency_flowx(self, reason: str):
        """긴급 FLOWX 메시지 발행."""
        logger.warning("[FLOWX-EMERGENCY] %s", reason)

        emergency = {
            "situation": "점검",
            "headline": "시스템 점검 중 — 수동 판단 권고",
            "reason": reason,
            "action": "오늘은 자동 분석 결과를 제공할 수 없습니다. 수동으로 판단해 주세요.",
            "generated_at": datetime.now().isoformat(),
        }

        emergency_path = DATA_DIR / "flowx_emergency.json"
        with open(emergency_path, "w", encoding="utf-8") as f:
            json.dump(emergency, f, ensure_ascii=False, indent=2)
        logger.info("[FLOWX-EMERGENCY] 저장: %s", emergency_path)

        # 텔레그램 알림 시도
        try:
            from src.telegram_sender import send_message
            msg = (
                f"⚠️ [COO] 파이프라인 이상\n"
                f"사유: {reason}\n"
                f"시각: {datetime.now().strftime('%H:%M')}\n"
                f"조치: 수동 판단 권고"
            )
            send_message(msg)
        except Exception as e:
            logger.error("[FLOWX-EMERGENCY] 텔레그램 전송 실패: %s", e)

    # ─── 요약 + 저장 ───

    def _build_summary(self):
        """실행 결과 요약."""
        groups = self.log_data["groups"]
        total_steps = sum(len(g["steps"]) for g in groups)
        failed_steps = sum(g["failed_count"] for g in groups)
        success_groups = sum(1 for g in groups if g["success"])

        self.log_data["finished_at"] = datetime.now().isoformat()
        elapsed = (datetime.fromisoformat(self.log_data["finished_at"])
                   - datetime.fromisoformat(self.log_data["started_at"])).total_seconds()

        self.log_data["summary"] = {
            "total_groups": len(groups),
            "success_groups": success_groups,
            "total_steps": total_steps,
            "failed_steps": failed_steps,
            "success_rate": round((total_steps - failed_steps) / max(total_steps, 1) * 100, 1),
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "fallbacks_used": sum(1 for g in groups if g.get("fallback_used")),
        }

    def _save_log(self):
        """실행 로그 저장."""
        log_path = DATA_DIR / "coo_run_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        logger.info("실행 로그 저장: %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="COO Orchestrator — 파이프라인 지휘관")
    parser.add_argument("--dry-run", action="store_true",
                        help="실행 없이 로그만 생성")
    parser.add_argument("--group", type=str, default=None,
                        help="특정 그룹만 실행 (예: 데이터수집, BRAIN_SHIELD)")
    args = parser.parse_args()

    coo = COOOrchestrator(dry_run=args.dry_run)
    coo.run(target_group=args.group)


if __name__ == "__main__":
    main()
