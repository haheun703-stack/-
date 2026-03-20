"""데이터 무결성 체크 시스템 — 사일런트 실패 감지

매일 BAT-D 완료 후 (또는 독립 실행) 모든 데이터 소스를 점검하고
결과를 텔레그램으로 보낸다.

14개 항목:
  1. 종가 데이터 (Parquet)
  2. 수급 데이터 (종목별)
  3. 수급 데이터 (섹터별)
  4. 국적별 수급
  5. 원자재 가격
  6. FRED 매크로
  7. CFTC COT
  8. DART 공시
  9. 레짐 매크로
  10. BRAIN 결과
  11. SHIELD 방어
  12. 학습 기록
  13. Supabase 업로드
  14. 스케줄러 실행 로그

실행:
  python -u -X utf8 scripts/data_health_check.py
  python -u -X utf8 scripts/data_health_check.py --date 2026-03-19
  python -u -X utf8 scripts/data_health_check.py --no-telegram
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("data_health_check")


# ──────────────────────────────────────────────
# 체크 결과 모델
# ──────────────────────────────────────────────

class CheckResult:
    """단일 체크 결과."""

    def __init__(self, name: str, passed: bool, detail: str,
                 count: int = 0, total: int = 0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.count = count
        self.total = total

    @property
    def icon(self) -> str:
        return "✅" if self.passed else "❌"

    def __str__(self) -> str:
        return f"{self.icon} {self.name}: {self.detail}"


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class DataHealthCheck:
    """14개 데이터 소스 무결성 검진."""

    def __init__(self, check_date: date | None = None):
        self.today = check_date or date.today()
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_compact = self.today.strftime("%Y%m%d")
        self.data_dir = PROJECT_ROOT / "data"
        self.is_weekday = self.today.weekday() < 5

    def run_full_check(self) -> list[CheckResult]:
        """전체 14개 항목 점검."""
        checks = [
            self._check_price_data,         # 1. 종가
            self._check_supply_stocks,       # 2. 수급(종목)
            self._check_supply_sectors,      # 3. 수급(섹터)
            self._check_nationality,         # 4. 국적별
            self._check_commodities,         # 5. 원자재
            self._check_fred,               # 6. FRED
            self._check_cot,                # 7. COT
            self._check_dart,               # 8. DART
            self._check_regime,             # 9. 레짐
            self._check_brain,              # 10. BRAIN
            self._check_shield,             # 11. SHIELD
            self._check_learning,           # 12. 학습
            self._check_uploads,            # 13. 업로드
            self._check_scheduler,          # 14. 스케줄러
        ]

        results = []
        for check_fn in checks:
            try:
                result = check_fn()
                results.append(result)
            except Exception as e:
                name = check_fn.__name__.replace("_check_", "")
                results.append(CheckResult(name, False, f"체크 오류: {e}"))
                logger.exception("체크 실패: %s", name)

        return results

    # ─── 1. 종가 데이터 (Parquet) ───

    def _check_price_data(self) -> CheckResult:
        """processed/ parquet에서 오늘 종가 존재 여부.
        인덱스가 datetime64 타입이므로 Timestamp로 비교."""
        processed = self.data_dir / "processed"
        if not processed.exists():
            return CheckResult("종가", False, "processed/ 폴더 없음")

        parquets = list(processed.glob("*.parquet"))
        total = len(parquets)
        if total == 0:
            return CheckResult("종가", False, "parquet 파일 없음")

        # 샘플링: 전체 중 50개만 체크 (속도)
        import random
        samples = random.sample(parquets, min(50, total))

        has_today = 0
        try:
            import pandas as pd
            target_ts = pd.Timestamp(self.today)
            for pf in samples:
                try:
                    df = pd.read_parquet(pf, columns=["close"])
                    # 인덱스가 datetime64 타입
                    if target_ts in df.index:
                        has_today += 1
                except Exception:
                    continue
        except ImportError:
            return CheckResult("종가", False, "pandas 없음")

        ratio = has_today / len(samples) if samples else 0
        estimated_total = int(ratio * total)
        pct = ratio * 100

        passed = pct >= 95
        return CheckResult(
            "종가", passed,
            f"{estimated_total:,}/{total:,} ({pct:.1f}%)",
            count=estimated_total, total=total,
        )

    # ─── 2. 수급 데이터 (종목별) ───

    def _check_supply_stocks(self) -> CheckResult:
        """processed/ parquet에서 수급 컬럼 존재 여부.
        인덱스가 datetime64 타입."""
        processed = self.data_dir / "processed"
        parquets = list(processed.glob("*.parquet"))
        if not parquets:
            return CheckResult("수급(종목)", False, "parquet 없음")

        import random
        samples = random.sample(parquets, min(30, len(parquets)))

        has_supply = 0
        supply_cols = {"외국인합계", "기관합계", "체결강도"}

        try:
            import pandas as pd
            target_ts = pd.Timestamp(self.today)
            for pf in samples:
                try:
                    df = pd.read_parquet(pf)
                    cols = set(df.columns)
                    if supply_cols & cols:
                        if target_ts in df.index:
                            row = df.loc[target_ts]
                            if any(not pd.isna(row.get(c))
                                   for c in supply_cols if c in df.columns):
                                has_supply += 1
                except Exception:
                    continue
        except ImportError:
            return CheckResult("수급(종목)", False, "pandas 없음")

        ratio = has_supply / len(samples) if samples else 0
        estimated = int(ratio * len(parquets))

        # 수급 데이터는 유니버스 종목(~100개)에만 있으므로 10% 이상이면 정상
        passed = estimated >= 80
        return CheckResult(
            "수급(종목)", passed,
            f"~{estimated}/{len(parquets)} (유니버스 {estimated}종목 확인)",
            count=estimated, total=len(parquets),
        )

    # ─── 3. 수급 데이터 (섹터별) ───

    def _check_supply_sectors(self) -> CheckResult:
        """섹터 수급 JSON 존재 + 오늘 날짜."""
        files = [
            ("sector_momentum", self.data_dir / "sector_rotation" / "sector_momentum.json"),
            ("sector_zscore", self.data_dir / "sector_rotation" / "sector_zscore.json"),
            ("investor_flow", self.data_dir / "sector_rotation" / "investor_flow.json"),
        ]

        found = 0
        details = []
        for name, path in files:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    # 날짜 필드 확인
                    d = data.get("date", data.get("generated_at", ""))
                    if self.today_str in str(d) or self.today_compact in str(d):
                        found += 1
                        details.append(f"{name}✅")
                    else:
                        details.append(f"{name}⚠️({d[:10] if d else '날짜없음'})")
                except Exception:
                    details.append(f"{name}❌")
            else:
                details.append(f"{name}❌(없음)")

        passed = found >= 2
        return CheckResult(
            "수급(섹터)", passed,
            f"{found}/{len(files)} — {', '.join(details)}",
            count=found, total=len(files),
        )

    # ─── 4. 국적별 수급 ───

    def _check_nationality(self) -> CheckResult:
        """nationality DB 또는 JSON 확인."""
        json_path = self.data_dir / "krx_nationality" / "nationality_signal.json"
        db_path = self.data_dir / "krx_nationality" / "nationality.db"

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                # analyzed_at 또는 date 필드
                d = str(data.get("analyzed_at", data.get("date",
                        data.get("trade_date", ""))))
                yesterday = (self.today - timedelta(days=1)).strftime("%Y-%m-%d")
                if self.today_str in d or yesterday in d or self.today_compact in d:
                    count = data.get("total_stocks", len(data.get("signals", [])))
                    return CheckResult("국적별수급", True,
                                       f"{count}종목 ({d[:10]})")
                return CheckResult("국적별수급", False,
                                   f"오래된 데이터: {d[:10]}")
            except Exception as e:
                return CheckResult("국적별수급", False, f"JSON 파싱 오류: {e}")

        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT MAX(trade_date) FROM nationality_data"
                )
                row = cursor.fetchone()
                conn.close()
                if row and row[0]:
                    return CheckResult("국적별수급", True, f"DB 최신: {row[0]}")
            except Exception:
                pass

        return CheckResult("국적별수급", False, "데이터 없음")

    # ─── 5. 원자재 가격 ───

    def _check_commodities(self) -> CheckResult:
        """JGIS 또는 liquidity_signal에서 원자재 확인."""
        signal_path = self.data_dir / "liquidity_cycle" / "liquidity_signal.json"

        if signal_path.exists():
            try:
                data = json.loads(signal_path.read_text(encoding="utf-8"))
                d = str(data.get("date", data.get("generated_at", "")))
                commodities = data.get("commodities", data.get("asset_prices", {}))

                if not commodities:
                    # 다른 구조로 시도
                    details = []
                    for key in ["wti", "gold", "copper", "dxy"]:
                        val = data.get(key)
                        if val is not None:
                            details.append(f"{key}={val}")
                    if details:
                        return CheckResult("원자재", True,
                                           f"{', '.join(details[:4])}")

                count = len(commodities) if isinstance(commodities, dict) else 0
                recent = self.today_str in str(d) or (
                    self.today - timedelta(days=2)
                ).strftime("%Y-%m-%d") <= str(d)[:10]

                if recent and count > 0:
                    return CheckResult("원자재", True,
                                       f"{count}종 (날짜: {str(d)[:10]})")
                return CheckResult("원자재", False,
                                   f"{count}종 (날짜: {str(d)[:10]})")
            except Exception as e:
                return CheckResult("원자재", False, f"파싱 오류: {e}")

        return CheckResult("원자재", False, "liquidity_signal.json 없음")

    # ─── 6. FRED 매크로 ───

    def _check_fred(self) -> CheckResult:
        """regime_macro_signal에서 FRED 관련 데이터 확인."""
        path = self.data_dir / "regime_macro_signal.json"
        if not path.exists():
            return CheckResult("FRED", False, "regime_macro_signal.json 없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            d = str(data.get("date", ""))
            signals = data.get("signals", {})

            recent = self.today_str in d or (
                self.today - timedelta(days=2)
            ).strftime("%Y-%m-%d") <= d[:10]

            # FRED 관련 시그널 확인
            fred_keys = ["vix_level", "us_grade", "rv_current"]
            found = sum(1 for k in fred_keys if k in signals)

            if recent and found >= 2:
                vix = signals.get("vix_level", "?")
                us = signals.get("us_grade", "?")
                return CheckResult("FRED/매크로", True,
                                   f"VIX={vix}, US={us} ({d})")
            return CheckResult("FRED/매크로", False,
                               f"오래된 데이터: {d}")
        except Exception as e:
            return CheckResult("FRED/매크로", False, f"파싱 오류: {e}")

    # ─── 7. CFTC COT ───

    def _check_cot(self) -> CheckResult:
        """COT 포지션 데이터 비어있지 않은지."""
        path = self.data_dir / "cot" / "cot_signal.json"
        if not path.exists():
            return CheckResult("COT", False, "cot_signal.json 없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            contracts = data.get("contracts", data.get("positions", {}))

            if not contracts:
                return CheckResult("COT", False, "빈 딕셔너리 — 사일런트 실패!")

            count = len(contracts)
            report_date = str(data.get("report_date", data.get("date", "?")))
            stale = data.get("stale_days", 0)

            # COT는 주간 데이터라 14일 이내면 OK, 최소 3개 포지션
            if count >= 3:
                stale_str = f", {stale}일 전" if stale else ""
                passed = stale <= 14 if stale else True
                return CheckResult("COT", passed,
                                   f"{count}개 포지션 ({report_date}{stale_str})")
            return CheckResult("COT", False,
                               f"불충분: {count}개 (최소 3개 필요)")
        except Exception as e:
            return CheckResult("COT", False, f"파싱 오류: {e}")

    # ─── 8. DART 공시 ───

    def _check_dart(self) -> CheckResult:
        """오늘 DART 공시 수집 건수."""
        path = self.data_dir / "dart_disclosures.json"
        if not path.exists():
            return CheckResult("DART", False, "dart_disclosures.json 없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            crawled_at = str(data.get("crawled_at", ""))
            total = data.get("total_count", 0)
            tier1 = data.get("tier1_count", 0)

            recent = self.today_str in crawled_at or self.today_compact in crawled_at

            if recent and total > 0:
                return CheckResult("DART", True,
                                   f"{total}건 (Tier1: {tier1}건)")
            elif recent and total == 0 and self.is_weekday:
                return CheckResult("DART", False,
                                   "0건 수집 — 비정상 (장 운영일)")
            elif not recent:
                return CheckResult("DART", False,
                                   f"오래된 데이터: {crawled_at}")
            return CheckResult("DART", True, f"{total}건")
        except Exception as e:
            return CheckResult("DART", False, f"파싱 오류: {e}")

    # ─── 9. 레짐 매크로 ───

    def _check_regime(self) -> CheckResult:
        """regime_macro_signal 날짜 + 레짐 값."""
        path = self.data_dir / "regime_macro_signal.json"
        if not path.exists():
            return CheckResult("레짐", False, "없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            d = str(data.get("date", ""))
            regime = data.get("current_regime", "?")
            score = data.get("macro_score", "?")

            recent = self.today_str in d
            if recent:
                return CheckResult("레짐", True,
                                   f"{regime} (점수:{score}, {d})")
            return CheckResult("레짐", False, f"오래된: {d}")
        except Exception as e:
            return CheckResult("레짐", False, f"오류: {e}")

    # ─── 10. BRAIN 결과 ───

    def _check_brain(self) -> CheckResult:
        """BRAIN 자본배분 결과 확인."""
        path = self.data_dir / "brain_decision.json"
        if not path.exists():
            return CheckResult("BRAIN", False, "brain_decision.json 없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = str(data.get("timestamp", ""))
            regime = data.get("effective_regime", "?")
            confidence = data.get("confidence", 0)
            arms = data.get("arms", [])

            recent = self.today_str in ts
            if recent:
                return CheckResult("BRAIN", True,
                                   f"{regime}({confidence:.0%}), ARM {len(arms)}개")
            return CheckResult("BRAIN", False, f"오래된: {ts[:10]}")
        except Exception as e:
            return CheckResult("BRAIN", False, f"오류: {e}")

    # ─── 11. SHIELD 방어 ───

    def _check_shield(self) -> CheckResult:
        """SHIELD 방어 보고서 확인."""
        path = self.data_dir / "shield_report.json"
        if not path.exists():
            return CheckResult("SHIELD", False, "shield_report.json 없음")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            d = str(data.get("timestamp", data.get("date",
                    data.get("checked_at", ""))))
            status = data.get("overall_level", data.get("overall_status",
                     data.get("defense_level", "?")))

            recent = self.today_str in d
            if recent:
                return CheckResult("SHIELD", True, f"{status} ({d[:10]})")
            return CheckResult("SHIELD", False, f"오래된: {d[:10]}")
        except Exception as e:
            return CheckResult("SHIELD", False, f"오류: {e}")

    # ─── 12. 학습 기록 ───

    def _check_learning(self) -> CheckResult:
        """market_learning 오늘자 기록 존재."""
        learning_dir = self.data_dir / "market_learning"

        if not learning_dir.exists():
            return CheckResult("학습기록", False, "market_learning/ 없음")

        today_file = learning_dir / f"{self.today_str}.json"
        index_file = learning_dir / "_index.json"

        has_daily = today_file.exists()
        has_index = index_file.exists()

        if has_daily and has_index:
            try:
                data = json.loads(today_file.read_text(encoding="utf-8"))
                signals = data.get("signal_count", data.get("total_signals", "?"))
                return CheckResult("학습기록", True,
                                   f"daily({self.today_str}) + index 있음 (시그널: {signals})")
            except Exception:
                return CheckResult("학습기록", True,
                                   f"daily({self.today_str}) + index 있음")

        parts = []
        if has_daily:
            parts.append("daily✅")
        else:
            parts.append(f"daily❌({self.today_str} 없음)")
        if has_index:
            parts.append("index✅")
        else:
            parts.append("index❌")

        return CheckResult("학습기록", has_daily, ", ".join(parts))

    # ─── 13. 업로드 확인 ───

    def _check_uploads(self) -> CheckResult:
        """JARVIS + Supabase 업로드 간접 확인 (파일 수정 시각)."""
        upload_files = {
            "brain_upload": PROJECT_ROOT / "website" / "data" / "brain_data_upload.json",
            "journal": self.data_dir / "market_journal" / f"{self.today_str}.json",
        }

        found = 0
        details = []
        for name, path in upload_files.items():
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime.date() == self.today:
                    found += 1
                    details.append(f"{name}✅")
                else:
                    details.append(f"{name}⚠️({mtime.date()})")
            else:
                details.append(f"{name}❌")

        passed = found >= 1
        return CheckResult("업로드", passed,
                           f"{found}/{len(upload_files)} — {', '.join(details)}",
                           count=found, total=len(upload_files))

    # ─── 14. 스케줄러 로그 ───

    def _check_scheduler(self) -> CheckResult:
        """오늘 스케줄러 로그에서 실행/실패 단계 수 파싱."""
        log_path = PROJECT_ROOT / "logs" / "schedule.log"
        if not log_path.exists():
            return CheckResult("스케줄러", False, "schedule.log 없음")

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # 오늘 날짜 로그만 필터
            today_lines = [
                l for l in lines
                if self.today.strftime("%Y-%m-%d") in l
                or self.today.strftime("%m/%d") in l  # Windows 날짜 포맷
                or self.today.strftime("%Y. %m. %d") in l  # 한국어 포맷
            ]

            if not today_lines:
                # 날짜 포맷이 다를 수 있으니 마지막 200줄에서 "BAT-D" 검색
                recent = lines[-200:] if len(lines) > 200 else lines
                bat_d_lines = [l for l in recent if "BAT-D" in l]
                if bat_d_lines:
                    return CheckResult("스케줄러", True,
                                       f"최근 BAT-D 로그 {len(bat_d_lines)}줄")
                return CheckResult("스케줄러", False, "오늘 로그 없음")

            # FAILED 카운트
            failed = [l for l in today_lines if "FAILED" in l]
            # 단계별 실행 카운트
            executed = [l for l in today_lines if re.search(r"\[\d+", l)]

            total_steps = len(executed)
            failed_count = len(failed)

            if failed_count == 0:
                return CheckResult("스케줄러", True,
                                   f"{total_steps}단계 실행, 실패 0건")
            else:
                failed_ids = []
                for l in failed:
                    m = re.search(r"\[([^\]]+)\].*FAILED", l)
                    if m:
                        failed_ids.append(m.group(1))
                return CheckResult("스케줄러", failed_count <= 2,
                                   f"{total_steps}단계 중 {failed_count}건 실패: "
                                   f"{', '.join(failed_ids[:5])}")
        except Exception as e:
            return CheckResult("스케줄러", False, f"로그 파싱 오류: {e}")


# ──────────────────────────────────────────────
# 보고서 생성
# ──────────────────────────────────────────────

def generate_health_report(results: list[CheckResult], check_date: date) -> str:
    """체크 결과 → 등급 + 텔레그램 메시지."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if passed == total:
        grade, icon = "A", "🟢"
    elif passed >= total - 2:
        grade, icon = "B", "🟡"
    elif passed >= total - 4:
        grade, icon = "C", "🟠"
    else:
        grade, icon = "D", "🚨"

    lines = [
        f"{icon} 데이터 건강검진 [{grade}등급] ({check_date})",
        f"합계: {passed}/{total} 통과",
        "",
    ]

    # 실패 항목 먼저
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append("── 실패 항목 ──")
        for r in failed:
            lines.append(f"  {r}")
        lines.append("")

    # 성공 항목
    passed_items = [r for r in results if r.passed]
    if passed_items:
        lines.append("── 정상 항목 ──")
        for r in passed_items:
            lines.append(f"  {r}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="데이터 무결성 체크")
    parser.add_argument("--date", type=str, default=None,
                        help="체크 날짜 (YYYY-MM-DD, 기본: 오늘)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="텔레그램 전송 안 함")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    check_date = None
    if args.date:
        check_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    checker = DataHealthCheck(check_date)
    results = checker.run_full_check()

    if args.json:
        output = {
            "date": checker.today_str,
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in results
            ],
            "passed_count": sum(1 for r in results if r.passed),
            "total_count": len(results),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    report = generate_health_report(results, checker.today)
    print(report)

    # 텔레그램 전송
    if not args.no_telegram:
        try:
            from src.telegram_sender import send_message
            send_message(report)
            logger.info("텔레그램 전송 완료")
        except Exception as e:
            logger.warning("텔레그램 전송 실패: %s", e)

    # 로그 저장
    log_path = PROJECT_ROOT / "logs" / "health_check.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"{datetime.now().isoformat()}\n")
        f.write(report)
        f.write("\n")


if __name__ == "__main__":
    main()
