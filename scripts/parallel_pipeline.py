"""BAT-D 병렬 파이프라인 — 의존성 DAG 기반 실행기

기존 BAT-D의 31단계를 의존성 그래프(DAG)로 재구성하여
독립적인 단계를 동시에 실행한다.

예상 효과: 50분 → 25~30분 (20~25분 절감)

실행:
  python -u -X utf8 scripts/parallel_pipeline.py
  python -u -X utf8 scripts/parallel_pipeline.py --dry-run   # 실행 없이 그래프만 표시
  python -u -X utf8 scripts/parallel_pipeline.py --max-workers 4  # 동시 실행 수 제한
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("parallel_pipeline")

# ──────────────────────────────────────────────
# 단계 정의
# ──────────────────────────────────────────────

@dataclass
class Step:
    """파이프라인 단계."""
    id: str                          # "1", "5.5", "9a" 등
    name: str                        # 한글 설명
    cmd: str                         # 실행 명령어
    depends: list[str] = field(default_factory=list)  # 선행 단계 ID
    friday_only: bool = False        # 금요일만 실행
    optional: bool = False           # 실패해도 전체 중단 안 함
    timeout: int = 600               # 초 단위 타임아웃 (기본 10분)


def build_pipeline() -> list[Step]:
    """전체 BAT-D 파이프라인을 DAG로 정의."""
    py = "python -u -X utf8"
    s = py + " scripts/"
    m = py + " -m "

    return [
        # ═══ LEVEL 0: 독립 데이터 수집 (동시 12개) ═══
        Step("1",    "CSV 전종목 종가",         s + "update_daily_data.py",    timeout=900),
        Step("2",    "Parquet 증분",           s + "extend_parquet_data.py",  timeout=2400),
        Step("3",    "수급 데이터",             s + "collect_supply_data.py"),
        Step("4",    "KOSPI 인덱스",           s + "update_kospi_index.py"),
        Step("4.5",  "분봉 데이터",            s + "collect_intraday_candles.py", timeout=1800),
        Step("7",    "US-KR 패턴DB",          s + "update_us_kr_daily.py"),
        Step("9c.5", "차이나머니",             s + "crawl_china_money.py"),
        Step("9c.7", "ETF 투자자 수급",        s + "collect_etf_investor_flow.py"),
        Step("9c.9", "KRX 국적별",            s + "scan_nationality.py --send"),
        Step("13",   "DART 공시",             s + "crawl_dart_disclosure.py"),
        Step("14",   "시장 뉴스",              s + "crawl_market_news.py"),
        Step("19.65","CPI 트래커",            s + "update_cpi_data.py"),

        # ═══ LEVEL 1: 기초 데이터 완료 후 지표 계산 ═══
        Step("5",    "기술지표 35개",          s + "rebuild_indicators.py",
             depends=["1", "2", "3", "4"]),
        Step("5.5",  "ICT 레벨+OR/IR",       s + "run_ict_levels.py",
             depends=["1", "4.5"]),
        Step("6",    "US Overnight Signal",  s + "us_overnight_signal.py --update",
             depends=["1"]),

        # ═══ LEVEL 2: 지표 완료 후 대량 병렬 스캔 ═══
        Step("8",    "섹터 ETF 시세",          s + "sector_etf_builder.py --daily",
             depends=["5"]),
        Step("9a",   "섹터 모멘텀",            s + "sector_momentum.py --history",
             depends=["5"]),
        Step("9b",   "섹터 z-score",          s + "sector_zscore.py --top 5",
             depends=["5"]),
        Step("9c",   "섹터 수급",              s + "sector_investor_flow.py --days 5",
             depends=["5"]),
        Step("12",   "눌림목 스캔",            s + "scan_pullback.py",
             depends=["5"]),
        Step("12.5", "수급 폭발 스캔",         s + "scan_volume_spike.py",
             depends=["5"]),
        Step("12.6", "소형주 급등",            s + "scan_smallcap_explosion.py",
             depends=["5"]),
        Step("12.7", "밸류체인 릴레이",         s + "scan_value_chain.py",
             depends=["5"]),
        Step("13.5", "레짐 매크로",            s + "regime_macro_signal.py",
             depends=["5", "6"]),
        Step("15",   "세력감지",               s + "scan_whale_detect.py",
             depends=["5"]),
        Step("17",   "동반매수",               s + "scan_dual_buying.py",
             depends=["5"]),
        Step("18.5", "매집 추적",              s + "scan_accumulation_tracker.py",
             depends=["5"]),
        Step("19",   "추천 성과 추적",          s + "track_pick_results.py",
             depends=["5"]),
        Step("19.5", "기관 목표가",            s + "calc_institutional_targets.py",
             depends=["5"]),
        Step("19.6", "보유종목 재판정",         s + "position_monitor.py",
             depends=["5"]),
        Step("19.95","컨센서스 스크리닝",       s + "scan_consensus.py --top 15 --send",
             depends=["5"]),
        Step("20.8", "3단 예측 체인",          s + "run_predict_chain.py --send --blind",
             depends=["5"]),

        # ═══ LEVEL 3: 섹터 분석 완료 후 ═══
        Step("9d",   "섹터 종합 리포트",        s + "sector_daily_report.py",
             depends=["9a", "9b", "9c"]),
        Step("16",   "세력감지 하이브리드",      s + "scan_force_hybrid.py",
             depends=["15"]),
        Step("19.3", "DART 이벤트 시그널",     s + "dart_event_signal.py",
             depends=["13"]),
        Step("17.5", "섹터 릴레이",            s + "relay_report.py",
             depends=["9d"]),
        Step("18",   "그룹 릴레이",            s + "group_relay_detector.py",
             depends=["9d"]),

        # ═══ LEVEL 4: ETF + SHIELD + COT ═══
        Step("10",   "ETF 마스터",            s + "update_etf_master.py",
             depends=["8", "9d"]),
        Step("11.1", "SHIELD 방어",           s + "run_shield.py --send",
             depends=["9d", "13.5"]),
        Step("11.15","COT Slow Eye",         s + "run_cot_tracker.py",
             depends=["5"]),

        # ═══ LEVEL 5: ETF 시그널 + BRAIN ═══
        Step("11",   "ETF 매매 시그널",        s + "etf_trading_signal.py",
             depends=["10"]),
        Step("11.2", "BRAIN 자본배분",         s + "run_brain.py",
             depends=["11.1", "11.15"]),

        # ═══ LEVEL 6: SCENARIO → LENS (순차 필수) ═══
        Step("11.23","SCENARIO ENGINE",      m + "src.alpha.scenario_detector",
             depends=["11.2"]),
        Step("11.25","LENS LAYER",           s + "run_lens.py",
             depends=["11.23"]),

        # ═══ LEVEL 7: LENS 완료 후 병렬 ═══
        Step("11.3", "ETF 로테이션",           s + "run_etf_rotation.py --blind-test --no-telegram",
             depends=["11.2"]),
        Step("11.5", "레버리지 ETF",           s + "leverage_etf_scanner.py",
             depends=["11.2"]),
        Step("19.7", "Perplexity 인텔",       s + "perplexity_market_intel.py",
             depends=["5"]),
        Step("19.8", "AI 뉴스 분석",          s + "ai_news_brain.py",
             depends=["14"]),
        Step("19.9", "v3 AI Brain",          s + "run_v3_brain.py --no-telegram",
             depends=["11.25"]),

        # ═══ LEVEL 8: 추천 통합 (critical path) ═══
        Step("20",   "내일 추천 종목",         s + "scan_tomorrow_picks.py",
             depends=["19", "19.3", "19.9"]),

        # ═══ LEVEL 9: 추천 완료 후 병렬 ═══
        Step("20.5", "일일 시장 학습",         s + "daily_market_learner.py",
             depends=["20"]),
        Step("20.7", "포트폴리오 배분",        s + "portfolio_allocator.py",
             depends=["20"]),
        Step("20.9", "Master Brain",         s + "run_master_brain.py",
             depends=["20", "19.9"]),

        # ═══ LEVEL 10: 아카이브 + 업로드 (병렬) ═══
        Step("21",   "일일 아카이브",          py + " src/daily_archive.py",
             depends=["20.9"]),
        Step("23",   "JARVIS 업로드",
             py + ' -c "from src.adapters.jarvis_uploader import main; main()"',
             depends=["20.9"]),
        Step("23.5", "Brain 대시보드",         s + "build_brain_upload.py",
             depends=["20.9"]),
        Step("23.7", "FLOWX Supabase",       s + "upload_flowx.py",
             depends=["20.9"]),
        Step("23.9", "Market Journal",       s + "market_journal.py",
             depends=["20.9"]),
        Step("25",   "페이퍼 트레이딩",        s + "paper_trader.py",
             depends=["20.9"]),

        # ═══ LEVEL 11: 최종 ═══
        Step("21.5", "v3 일일 리뷰",          s + "run_v3_brain.py --review",
             depends=["21"]),
        Step("22",   "주간 보고서",            py + " src/daily_archive.py --weekly",
             depends=["21"], friday_only=True),
        Step("22.5", "v3 주간 정확도",         s + "run_v3_brain.py --weekly-review",
             depends=["22"], friday_only=True),
        Step("24",   "저녁 통합 리포트",        s + "send_evening_summary.py --send",
             depends=["23.9"]),
        Step("25.5", "역발상 주간",            s + "paper_trader.py --weekly",
             depends=["25"], friday_only=True),
        Step("26",   "네 마녀의 날",
             py + ' -c "'
             "from src.use_cases.market_calendar import check_witching_proximity; "
             "w=check_witching_proximity(); print(w['warning_level'], w['message']); "
             "exec('from src.telegram_sender import send_message; send_message(w[\\\"message\\\"])') "
             "if w['warning_level']=='HIGH' else None\"",
             depends=["20.9"], optional=True),
    ]


# ──────────────────────────────────────────────
# DAG 실행기
# ──────────────────────────────────────────────

class PipelineRunner:
    """의존성 DAG 기반 병렬 실행기."""

    def __init__(self, steps: list[Step], max_workers: int = 6,
                 dry_run: bool = False):
        self.steps = {s.id: s for s in steps}
        self.max_workers = max_workers
        self.dry_run = dry_run

        # 상태 추적
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()
        self.timings: dict[str, float] = {}

        # 금요일 체크
        self.is_friday = datetime.now().weekday() == 4

    def _is_ready(self, step: Step) -> bool:
        """선행 단계가 모두 완료(또는 실패+optional)되었는지."""
        for dep_id in step.depends:
            if dep_id not in self.completed and dep_id not in self.skipped:
                return False
        return True

    def _should_skip(self, step: Step) -> bool:
        """금요일 전용 + 필수 선행 실패 체크."""
        if step.friday_only and not self.is_friday:
            return True
        # 필수 선행 단계가 실패했으면 스킵
        for dep_id in step.depends:
            if dep_id in self.failed:
                dep = self.steps.get(dep_id)
                if dep and not dep.optional:
                    # 선행 실패 → 이 단계도 스킵 (단, 다른 선행이 다 성공이면 시도)
                    pass
        return False

    def _kill_proc_tree(self, pid: int):
        """Windows에서 프로세스 트리 전체 종료 (shell=True로 생긴 자식까지)."""
        try:
            subprocess.run(
                f"taskkill /F /T /PID {pid}",
                shell=True, capture_output=True, timeout=10,
            )
        except Exception:
            pass

    def _run_step(self, step: Step) -> tuple[str, bool, float, str]:
        """단일 단계 실행. (id, success, elapsed, output)"""
        start = time.time()

        if self.dry_run:
            logger.info("[DRY] %s: %s", step.id, step.name)
            time.sleep(0.1)
            return step.id, True, 0.1, "(dry run)"

        logger.info("[START] %s: %s", step.id, step.name)
        step_timeout = step.timeout  # 단계별 타임아웃

        try:
            proc = subprocess.Popen(
                step.cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
            try:
                stdout, stderr = proc.communicate(timeout=step_timeout)
            except subprocess.TimeoutExpired:
                self._kill_proc_tree(proc.pid)
                proc.kill()
                proc.wait(timeout=5)
                elapsed = time.time() - start
                logger.error("[TIMEOUT] %s: %s (%.1fs, limit=%ds)",
                             step.id, step.name, elapsed, step_timeout)
                return step.id, False, elapsed, f"TIMEOUT ({step_timeout}s 초과)"

            elapsed = time.time() - start
            success = proc.returncode == 0
            output = stdout[-500:] if stdout else ""
            if not success:
                output += f"\nSTDERR: {stderr[-300:]}" if stderr else ""
                logger.warning("[FAIL] %s: %s (%.1fs)\n%s",
                               step.id, step.name, elapsed, output)
            else:
                logger.info("[DONE] %s: %s (%.1fs)", step.id, step.name, elapsed)

            return step.id, success, elapsed, output

        except Exception as e:
            elapsed = time.time() - start
            logger.error("[ERROR] %s: %s — %s", step.id, step.name, e)
            return step.id, False, elapsed, str(e)

    def run(self) -> dict:
        """전체 파이프라인 실행."""
        total_start = time.time()
        pending = set(self.steps.keys())

        # 금요일 전용 단계 스킵
        for sid, step in self.steps.items():
            if self._should_skip(step):
                pending.discard(sid)
                self.skipped.add(sid)
                logger.info("[SKIP] %s: %s (금요일 전용)", sid, step.name)

        logger.info("=" * 60)
        logger.info("파이프라인 시작: %d개 단계 (%d개 스킵)",
                     len(pending), len(self.skipped))
        logger.info("병렬 워커: %d개", self.max_workers)
        logger.info("=" * 60)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending:
                # 실행 가능한 단계 찾기
                ready = []
                for sid in list(pending):
                    step = self.steps[sid]
                    if self._is_ready(step):
                        ready.append(step)

                if not ready:
                    # 데드락 방지: 남은 단계가 있는데 실행 가능한 게 없으면
                    blocked = pending - self.completed - self.failed - self.skipped
                    if blocked:
                        logger.error("데드락 감지: %s (의존성 미충족)", blocked)
                        for sid in blocked:
                            self.failed.add(sid)
                            pending.discard(sid)
                    break

                # 병렬 제출
                futures = {}
                for step in ready:
                    pending.discard(step.id)
                    future = executor.submit(self._run_step, step)
                    futures[future] = step

                # 결과 수집
                for future in as_completed(futures):
                    sid, success, elapsed, output = future.result()
                    self.timings[sid] = elapsed
                    if success:
                        self.completed.add(sid)
                    else:
                        self.failed.add(sid)
                        # optional이 아닌 단계 실패 → 의존 단계도 스킵
                        step = self.steps[sid]
                        if not step.optional:
                            self._cascade_skip(sid, pending)

        total_elapsed = time.time() - total_start
        return self._build_report(total_elapsed)

    def _cascade_skip(self, failed_id: str, pending: set[str]):
        """실패한 단계에 의존하는 모든 후속 단계를 스킵."""
        for sid in list(pending):
            step = self.steps[sid]
            if failed_id in step.depends:
                pending.discard(sid)
                self.skipped.add(sid)
                logger.warning("[CASCADE-SKIP] %s: %s (선행 %s 실패)",
                               sid, step.name, failed_id)
                self._cascade_skip(sid, pending)

    def _build_report(self, total_elapsed: float) -> dict:
        """실행 보고서 생성."""
        report = {
            "total_seconds": round(total_elapsed, 1),
            "total_minutes": round(total_elapsed / 60, 1),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "total": len(self.steps),
            "failed_steps": sorted(self.failed),
            "skipped_steps": sorted(self.skipped),
            "slowest": sorted(
                self.timings.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

        # 로그 출력
        logger.info("=" * 60)
        logger.info("파이프라인 완료: %.1f분", report["total_minutes"])
        logger.info("  성공: %d / 실패: %d / 스킵: %d",
                     report["completed"], report["failed"], report["skipped"])
        if report["failed_steps"]:
            logger.warning("  실패 목록: %s", report["failed_steps"])
        logger.info("  가장 느린 단계:")
        for sid, elapsed in report["slowest"]:
            logger.info("    %s (%s): %.1fs",
                         sid, self.steps[sid].name, elapsed)
        logger.info("=" * 60)

        return report


# ──────────────────────────────────────────────
# 텔레그램 보고
# ──────────────────────────────────────────────

def send_pipeline_report(report: dict):
    """파이프라인 결과를 텔레그램으로 전송."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        logger.warning("텔레그램 모듈 없음 — 보고서 스킵")
        return

    grade = "A" if report["failed"] == 0 else "B" if report["failed"] <= 2 else "C"
    icon = {"A": "🟢", "B": "🟡", "C": "🔴"}[grade]

    lines = [
        f"{icon} BAT-D 파이프라인 완료 ({grade}등급)",
        f"⏱ {report['total_minutes']}분 "
        f"({report['completed']}✅ {report['failed']}❌ {report['skipped']}⏭)",
    ]

    if report["failed_steps"]:
        lines.append(f"❌ 실패: {', '.join(report['failed_steps'])}")

    if report["slowest"]:
        top3 = report["slowest"][:3]
        slow_str = ", ".join(f"{sid}({t:.0f}s)" for sid, t in top3)
        lines.append(f"🐢 느린 단계: {slow_str}")

    try:
        send_message("\n".join(lines))
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="BAT-D 병렬 파이프라인")
    parser.add_argument("--dry-run", action="store_true",
                        help="실행 없이 DAG 구조만 표시")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="동시 실행 최대 수 (기본 6)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="텔레그램 보고 비활성화")
    parser.add_argument("--graph", action="store_true",
                        help="의존성 그래프를 텍스트로 출력")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / "logs" / "parallel_pipeline.log",
                encoding="utf-8",
            ),
        ],
    )

    steps = build_pipeline()

    if args.graph:
        _print_graph(steps)
        return

    runner = PipelineRunner(
        steps, max_workers=args.max_workers, dry_run=args.dry_run
    )
    report = runner.run()

    if not args.no_telegram and not args.dry_run:
        send_pipeline_report(report)


def _print_graph(steps: list[Step]):
    """의존성 그래프를 레벨별로 출력."""
    step_map = {s.id: s for s in steps}
    assigned: dict[str, int] = {}

    def get_level(sid: str) -> int:
        if sid in assigned:
            return assigned[sid]
        step = step_map[sid]
        if not step.depends:
            assigned[sid] = 0
            return 0
        lvl = max(get_level(d) for d in step.depends) + 1
        assigned[sid] = lvl
        return lvl

    for s in steps:
        get_level(s.id)

    max_lvl = max(assigned.values()) if assigned else 0
    for lvl in range(max_lvl + 1):
        members = [s for s in steps if assigned[s.id] == lvl]
        print(f"\n═══ Level {lvl} ({len(members)}개 병렬) ═══")
        for s in members:
            deps = f" ← {s.depends}" if s.depends else ""
            fri = " [금]" if s.friday_only else ""
            print(f"  [{s.id}] {s.name}{fri}{deps}")


if __name__ == "__main__":
    main()
