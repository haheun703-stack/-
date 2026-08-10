"""데이터 무결성 체크 시스템 — 사일런트 실패 감지

매일 BAT-D 완료 후 (또는 독립 실행) 모든 데이터 소스를 점검하고
결과를 텔레그램으로 보낸다.

17개 항목:
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
  14. 스케줄러(cron) 실행 로그
  15. 파이프라인 에러율
  16. KOSPI 인덱스 신선도
  17. 투자자수급 신선도
  (18. 수급이면분석 — 폐기)
  (19. 수급스냅샷 — 폐기)

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
                 count: int = 0, total: int = 0, skipped: bool = False):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.count = count
        self.total = total
        self.skipped = skipped  # 휴장일 등 판정 대상이 아님 → 등급 계산 제외

    @property
    def icon(self) -> str:
        if self.skipped:
            return "⏸️"
        return "✅" if self.passed else "❌"

    def __str__(self) -> str:
        return f"{self.icon} {self.name}: {self.detail}"


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class DataHealthCheck:
    """17개 데이터 소스 무결성 검진."""

    def __init__(self, check_date: date | None = None):
        self.today = check_date or date.today()
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_compact = self.today.strftime("%Y%m%d")
        self.data_dir = PROJECT_ROOT / "data"
        self.is_weekday = self.today.weekday() < 5
        # 휴장일 판정 (7/20 B-13): 평일 공휴일에도 cron이 도는데 국내시장 데이터가
        # 없는 걸 "장 운영일 비정상"으로 오판해 가짜 경보를 보냈다(7/17 D등급 13/18).
        # 판정 불가 시 True(장날)로 폴백 — 경보를 잠재우는 쪽이 더 위험하기 때문.
        try:
            from src.adapters.kis_holiday_adapter import is_trading_day
            self.is_trading_day = is_trading_day(self.today)
        except Exception as e:
            logger.warning("휴장일 판정 실패 — 장날로 간주: %s", e)
            self.is_trading_day = True

    # 휴장일에 판정 대상에서 제외할 체크 (국내 장중 거래 산출물에 의존)
    MARKET_DEPENDENT = {
        "종가", "수급(종목)", "수급(섹터)", "DART",
        "KOSPI인덱스", "투자자수급", "수급커버리지",
    }

    def run_full_check(self) -> list[CheckResult]:
        """전체 19개 항목 점검."""
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
            self._check_pipeline_errors,    # 15. 파이프라인 에러율
            self._check_kospi_index,        # 16. KOSPI 인덱스
            self._check_investor_flow,      # 17. 투자자수급
            self._check_supply_coverage,    # 18. 수급 커버리지 급감 (대량 수집장애 방어)
            # 구 18/19(supply_demand/snapshots) 폐기: BAT에서 제거됨 (2026-04-06)
        ]

        results = []
        for check_fn in checks:
            try:
                result = check_fn()
            except Exception as e:
                name = check_fn.__name__.replace("_check_", "")
                result = CheckResult(name, False, f"체크 오류: {e}")
                logger.exception("체크 실패: %s", name)

            # 휴장일: 국내시장 의존 체크는 실패로 세지 않고 SKIP (7/20 B-13)
            if not self.is_trading_day and result.name in self.MARKET_DEPENDENT:
                result.skipped = True
                result.passed = True  # 등급 계산 오염 방지
                result.detail = f"휴장일 SKIP — {result.detail}"

            results.append(result)

        return results

    # ─── 1. 종가 데이터 (Parquet) ───

    def _check_price_data(
        self, frozen_alert_pct: float = 10.0, lookback: int = 5
    ) -> CheckResult:
        """processed/ parquet에서 오늘 종가 존재 여부 + 값이 살아있는지.
        인덱스가 datetime64 타입이므로 Timestamp로 비교.

        ★2026-08-05 정체 검사 추가. 이 검사는 오래 "당일 행이 있는가"만 봤다.
          그래서 거래정지로 종가가 멈춘 종목과, **수집기가 고장나 전일 값을
          그대로 재사용하는 상태를 구분하지 못했다** — 화면상 둘 다 ✅다.
          7/31 `dashboard_smart_money price=0`이 6주간 ✅로 보였던 것과 같은 구조
          ("적재됐는가"만 보는 감시는 값 오류를 영원히 못 잡는다).
          실측(8/5): 1,182종목 중 15종목이 5일 고정 + 거래량 0 + 수급 DB 부재로
          3중 일치 = 거래정지 확실, 정상 변동 1,161종목."""
        processed = self.data_dir / "processed"
        if not processed.exists():
            return CheckResult("종가", False, "processed/ 폴더 없음")

        parquets = list(processed.glob("*.parquet"))
        total = len(parquets)
        if total == 0:
            return CheckResult("종가", False, "parquet 파일 없음")

        # 전수 검사 (7/23 전환). 구 50개 샘플링은 비율을 전체에 곱해
        #   "1,180/1,180 (100.0%)"처럼 **추정치를 실측처럼** 표기했다.
        #   실측은 1,174/1,180(99.49%)이고 차이 6종목은 전부 상폐/거래정지였으나,
        #   샘플링으로는 그것과 진짜 수집 실패를 구분할 수 없다(임계 95% 경계에서 오판 위험).
        #   비용 실측 15.5초(샘플 0.6초의 26배)로 BAT-D 143분 대비 무시 가능.
        try:
            import pandas as pd
        except ImportError:
            return CheckResult("종가", False, "pandas 없음")

        target_ts = pd.Timestamp(self.today)
        has_today = 0
        missing: list[str] = []
        buckets: dict[str, list[str]] = {
            "halted": [], "reused": [], "corrupt": [], "short": [], "unjudged": [],
        }
        for pf in parquets:
            # ★판정까지 파일별 try 안에 둔다 — 판정을 밖에 두면 파일 하나의 예외가
            #   1,182개 전체를 날려 검사 결과가 통째로 소실된다(구 코드는 파일 단위
            #   격리였는데 정체 검사를 넣으며 예외 표면을 넓혔다).
            try:
                df = self._read_close_volume(pf)
                if target_ts not in df.index:  # 인덱스가 datetime64 타입
                    missing.append(pf.stem)
                    continue
                kind = self._frozen_kind(df, target_ts, lookback)
            except Exception:  # noqa: BLE001  (개별 파일 파손 — 미달로 격리)
                missing.append(pf.stem)
                continue
            has_today += 1
            if kind:
                buckets[kind].append(pf.stem)

        halted, reused = buckets["halted"], buckets["reused"]
        corrupt, short, unjudged = buckets["corrupt"], buckets["short"], buckets["unjudged"]
        pct = has_today / total * 100 if total else 0.0
        # ★live는 "판정했는데 정상"만 센다. 판정 불가(unjudged/short/corrupt)를
        #   여기 합산하면 검사가 실명했을 때 오히려 가장 건강한 숫자가 나온다.
        live = has_today - sum(len(v) for v in buckets.values())
        halted_pct = len(halted) / total * 100 if total else 0.0

        # ★passed 기준은 "당일 행이 있는가"를 유지한다 — 거래정지는 종목의 정상
        #   상태이므로 실패로 세면 매일 오탐이 된다(8/1 const_sanity와 같은 원칙:
        #   알려진 지속 상태는 억제, 급변만 경고).
        #   실패로 보는 것은 넷이고, 셋은 기저선이 0이라 1건도 급변이다:
        #   ⑴ 거래정지 추정이 임계 초과 → 개별 종목 사정이 아니라 수집 이상
        #      (실측: VPS 8/4 1.27%, 로컬 전 이력 최대 1.54% → 임계 3%면 2배 여유)
        #   ⑵ 값 재사용 의심(기저 0 — 103만 stock-day 전수에서 0건)
        #   ⑶ 값 파손: 종가가 0/NaN 고정(기저 0)
        #   ⑷ 판정 불가: 거래량을 못 읽음(기저 0) — 검사 실명을 침묵시키지 않는다
        #   `short`(이력 부족)만 정상 사유라 실패에서 뺀다.
        passed = (
            pct >= 95
            and halted_pct < frozen_alert_pct
            and not reused
            and not corrupt
            and not unjudged
        )

        detail = f"{has_today:,}/{total:,} ({pct:.1f}%)"
        detail += f" | 실동 {live:,}·정지 {len(halted):,}({halted_pct:.1f}%)·재사용 {len(reused):,}"
        for label, items in (("🚨파손", corrupt), ("🚨판정불가", unjudged)):
            if items:
                detail += f"·{label} {len(items):,}"
        if short:
            detail += f"·이력부족 {len(short):,}"
        for label, items in (("재사용", reused), ("파손", corrupt), ("판정불가", unjudged)):
            if items:
                # 기저선 0에서의 발생이라 종목코드를 아끼지 않고 남긴다
                detail += f" | 🚨{label}: " + ",".join(sorted(items)[:12])
                if len(items) > 12:
                    detail += f" 외 {len(items) - 12}건"
        if halted_pct >= frozen_alert_pct:
            detail += f" ⚠️정지 급증(임계 {frozen_alert_pct:g}%) — 수집 이상 의심"
        if halted:
            detail += " | 정지: " + ",".join(sorted(halted)[:8])
            if len(halted) > 8:
                detail += f" 외 {len(halted) - 8}건"
        if missing:
            # ★미달 종목 코드를 남긴다 — 거래정지/상폐인지 진짜 실패인지 즉시 판별하기 위함
            shown = sorted(missing)[:10]
            detail += " | 미달: " + ",".join(shown)
            if len(missing) > 10:
                detail += f" 외 {len(missing) - 10}건"

        return CheckResult(
            "종가", passed, detail,
            count=has_today, total=total,
        )

    @staticmethod
    def _read_close_volume(pf):
        """close + volume을 읽되, volume이 없는 스키마는 close만으로 폴백."""
        import pandas as pd
        try:
            return pd.read_parquet(pf, columns=["close", "volume"])
        except Exception:  # noqa: BLE001  (컬럼 부재 — close만으로 재시도)
            return pd.read_parquet(pf, columns=["close"])

    @staticmethod
    def _frozen_kind(df, target_ts, lookback: int = 3):
        """종가 정체 유형 판정. 반환값:

        - `None`      : 값이 살아있다(정상)
        - `"halted"`  : 종가 고정 + 당일 거래량 0/결측 → 거래정지 추정(종목의 정상 상태)
        - `"reused"`  : 종가 고정 + 거래량까지 고정인데 0이 아님 → 값 재사용 지문
        - `"corrupt"` : 고정된 종가가 0/NaN → 값 자체가 파손
        - `"short"`   : 이력이 lookback 미만 → 판정 불가하나 신규상장 등 정상 사유
        - `"unjudged"`: 거래량을 못 읽음(컬럼 부재·형변환 실패·인덱스 이상) → 판정 불가

        ★설계 자기수정 2회 (같은 실수를 두 번 했다):
          ⑴ 1차 = "종가 고정 AND 거래량 0" 하나뿐이었다. 그런데 **수집기가 전일 행을
             통째로 복사하면 거래량도 함께 고정되면서 0이 아니게 되어 안 걸린다** →
             `reused` 신설.
          ⑵ 2차 = 판정 불가를 전부 `None`으로 돌려보냈다. 호출부는 `None`을 "실동"으로
             세므로 **거래량 컬럼이 통째로 사라지면 전 종목이 '실동'으로 집계되어
             가장 건강한 표기와 함께 통과**한다(합성 실증: 전 종목 close 고정인데
             `실동 200·정지 0` PASS). 또 종가가 0/NaN으로 고정돼도 `halted`
             = 종목의 정상 상태로 접혔다 — **7/31 `price=0`을 "거래정지"라 부르는 꼴**.
             → `unjudged`/`short`/`corrupt`를 분리해 "판정했는데 정상"과
               "판정을 못 했다"를 다른 값으로 만든다.
          7/30 F1(판별실패를 '타봇'으로 접어 미탐)·F4(확인불가를 정상으로 접음)와
          정확히 같은 형태가 새 코드에 다시 들어왔던 것이다.

        ★lookback 기본값 5→3: 로컬 전 이력 1,904거래일 × 1,166종목(약 103만 stock-day)
          전수에서 `reused` 후보가 lookback 3·4·5 모두 **0건**이고, 2에서만 6건
          (거래량 2~1,009주 초저유동주)이 나온다. 즉 3이면 역사적 오탐 증가 없이
          탐지 지연만 4거래일 → 2거래일로 줄어든다.
        """
        import pandas as pd
        try:
            tail = df.loc[:target_ts].tail(lookback)
        except Exception:  # noqa: BLE001  (인덱스 비정렬·중복 등)
            return "unjudged"
        if len(tail) < lookback:
            return "short"
        if tail["close"].nunique(dropna=False) != 1:
            return None
        # ★고정값이 0/NaN이면 거래정지가 아니라 값 파손이다. 이걸 halted로 접으면
        #   임계(수 %) 안에서 영구 침묵한다 — 이 검사가 막으려던 바로 그 사고다.
        close_v = tail["close"].iloc[-1]
        try:
            if pd.isna(close_v) or float(close_v) == 0:
                return "corrupt"
        except (TypeError, ValueError):
            return "corrupt"
        if "volume" not in df.columns:
            return "unjudged"
        vol = tail["volume"].iloc[-1]
        try:
            is_zero = bool(pd.isna(vol) or float(vol) == 0)
        except (TypeError, ValueError):  # 거래량 파손 — 판정 불가
            return "unjudged"
        if is_zero:
            return "halted"
        return "reused" if tail["volume"].nunique(dropna=False) == 1 else None

    def _price_has_today(self, sample: int = 20) -> bool:
        """종가 parquet에 당일 데이터가 있는지 (수급 당일결손 판정의 독립 소스).

        전수 검사(_check_price_data)와 달리 표본 20개로 "당일 수집이 돌았는가"만 본다.
        시각 조건 하드코딩 대신 이 판정을 쓰는 이유는 _check_supply_coverage 주석 참조.
        """
        processed = self.data_dir / "processed"
        if not processed.exists():
            return False
        try:
            import pandas as pd
        except ImportError:
            return False
        target_ts = pd.Timestamp(self.today)
        hit = 0
        for pf in sorted(processed.glob("*.parquet"))[:sample]:
            try:
                if target_ts in pd.read_parquet(pf, columns=["close"]).index:
                    hit += 1
            except Exception:  # noqa: BLE001  (개별 파일 파손은 무시 — 표본 판정)
                continue
        # 표본 과반이면 당일 수집이 돈 것으로 본다(상폐·거래정지 소수 혼입 방어)
        return hit >= max(1, sample // 2)

    @staticmethod
    def _dead_row_cols(df, date_col: str) -> bool:
        """마지막 행의 숫자 컬럼이 **전량** 0/결측인가 (B-53 ⑤, 8/10).

        "수집은 됐는데 값이 안 들어왔다"를 잡는다 — 날짜·행수만 보는 신선도
        검사는 7/31 `price=0`형 사고를 통과시킨다.

        ★개별 컬럼의 0은 정상일 수 있다(순매수가 정확히 0인 주체, 거래정지주의
        거래량 0). 전량 0일 때만 결손으로 본다 — 8/1 `overheat_penalty`
        오탐 교훈("0이 정상 하한인 지표를 잡지 않는다")을 그대로 적용.

        ※numpy 스칼라는 `isinstance(v, int)`로 안 걸리는 경우가 있어
        `float()` 변환 성공 여부로 숫자를 판별한다.
        """
        try:
            last = df.iloc[-1]
        except Exception:  # noqa: BLE001
            return False  # 판정 불가는 결손으로 세지 않는다
        numeric = []
        for c in df.columns:
            if c == date_col:
                continue
            try:
                v = float(last[c])
            except (TypeError, ValueError):
                continue
            if v == v:  # NaN 제외 (NaN != NaN)
                numeric.append(v)
        if not numeric:
            return True
        return all(v == 0 for v in numeric)

    # ─── 2. 수급 데이터 (종목별) ───

    #: parquet 경유 소비처(v8 스코어러 등)가 실제로 읽는 핵심 수급 컬럼.
    #: ★8/10 교정 전에는 {"외국인합계","기관합계","체결강도"}였는데 **`체결강도`는
    #: parquet에 존재하지 않는 컬럼**이었다(실측 확인). 3개 중 1개가 헛다리였고,
    #: 기타법인·연기금은 애초에 검사 대상이 아니었다.
    SUPPLY_VALUE_COLS = ("기관합계", "외국인합계", "기타법인", "pension_net_5d")

    #: 이미 등재된 미해결 결손 — 새로 생긴 결손과 구분한다.
    #: ★매일 ❌로 세면 B-44에서 막 걷어낸 "등급 영구 고착"이 그대로 재현된다.
    #: 8/1에 채택한 판별 기준("이미 알려진 지속 상태인가")을 그대로 적용한다.
    KNOWN_ZERO_COLS = {
        "pension_net_5d": "B-46",  # 401일 전구간 0. supply_cols 미포함이 원인
    }

    #: 전량 0 판정 관측 창(거래일).
    SUPPLY_ZERO_WINDOW = 20

    def _check_supply_stocks(self) -> CheckResult:
        """processed/ parquet 수급 컬럼 — 존재 + **값이 살아 있는지**.

        ★★8/10 교정(B-53 ①). 구 로직은 `not pd.isna(row.get(c))`로 **NaN만**
        보고 0을 통과시켰다. 그래서 B-39(기타법인 3.5개월 0)·B-46(연기금 401일
        0)·B-47(공매도 4컬럼 5.5개월 0)이 **전부 이 검사를 통과**했다. 같은
        검사가 세 번 뚫린 것이라 "존재하는가"에 "값이 0으로 죽어 있지 않은가"를
        더한다.

        판정 단위는 **컬럼 × 표본 전체**다. 개별 종목의 특정일 0은 거래가 없었을
        뿐이라 정상이고, **표본 전 종목이 관측창 내내 0인 컬럼**만 결손으로 본다
        (8/1 `overheat_penalty` 오탐 교훈 — 0이 정상 하한인 지표를 잡지 않도록).
        """
        processed = self.data_dir / "processed"
        parquets = list(processed.glob("*.parquet"))
        if not parquets:
            return CheckResult("수급(종목)", False, "parquet 없음")

        import random
        samples = random.sample(parquets, min(30, len(parquets)))

        has_supply = 0
        # 컬럼별 비영값 누적 — 표본 전체를 합산해 "통째로 0인가"를 본다.
        nonzero: dict[str, int] = {c: 0 for c in self.SUPPLY_VALUE_COLS}
        present: dict[str, int] = {c: 0 for c in self.SUPPLY_VALUE_COLS}

        try:
            import pandas as pd
            target_ts = pd.Timestamp(self.today)
            for pf in samples:
                try:
                    df = pd.read_parquet(pf)
                    cols = set(df.columns)
                    if not (set(self.SUPPLY_VALUE_COLS) & cols):
                        continue

                    if target_ts in df.index:
                        row = df.loc[target_ts]
                        if any(not pd.isna(row.get(c))
                               for c in self.SUPPLY_VALUE_COLS if c in cols):
                            has_supply += 1

                    tail = df.tail(self.SUPPLY_ZERO_WINDOW)
                    for c in self.SUPPLY_VALUE_COLS:
                        if c in cols:
                            present[c] += 1
                            nonzero[c] += int((tail[c].fillna(0) != 0).sum())
                except Exception:
                    continue
        except ImportError:
            return CheckResult("수급(종목)", False, "pandas 없음")

        ratio = has_supply / len(samples) if samples else 0
        estimated = int(ratio * len(parquets))

        # ① 커버리지 — 수급 데이터는 유니버스 종목에만 있으므로 80종목 이상이면 정상.
        #   ★당일 행 자체가 없으면(BAT-D·rebuild_indicators 전 실행) 판정을 보류한다.
        #   구 로직은 이 경우에도 estimated=0으로 FAIL이었다 — HEALTH가 18:45 cron
        #   으로만 돌아 드러나지 않았을 뿐이고, 장중 수동 실행은 늘 가짜 실패였다.
        #   ★시각 하드코딩 대신 종가 존재를 독립 소스로 쓴다(#18과 동일 원칙, B-19①
        #   교훈). 종가가 당일치인데 수급만 비었으면 시각과 무관하게 진짜 이상이다.
        coverage_hold = False
        if has_supply == 0 and not self._price_has_today():
            coverage_ok, coverage_hold = True, True
        else:
            coverage_ok = estimated >= 80

        # ② 값 생존 — 표본 어디에도 비영값이 없는 컬럼 = 통째로 죽은 컬럼
        dead_new, dead_known, missing = [], [], []
        for c in self.SUPPLY_VALUE_COLS:
            if present[c] == 0:
                missing.append(c)          # 컬럼 자체가 표본에 없음
            elif nonzero[c] == 0:
                (dead_known if c in self.KNOWN_ZERO_COLS else dead_new).append(c)

        passed = coverage_ok and not dead_new and not missing

        parts = [f"~{estimated}/{len(parquets)}종목"]
        if dead_new:
            parts.append("🚨전량0(신규): " + ", ".join(dead_new))
        if missing:
            parts.append("🚨컬럼없음: " + ", ".join(missing))
        if dead_known:
            parts.append("알려진결손 "
                         + ", ".join(f"{c}({self.KNOWN_ZERO_COLS[c]})"
                                     for c in dead_known))
        live = [c for c in self.SUPPLY_VALUE_COLS if nonzero[c] > 0]
        if live:
            parts.append("실값 " + "·".join(live))
        if coverage_hold:
            parts[0] = f"당일 미수집 — 커버리지 판정 보류 (종가도 당일 없음, BAT-D 전)"
        elif not coverage_ok:
            parts.append("커버리지 미달!")

        return CheckResult(
            "수급(종목)", passed, " | ".join(parts),
            count=estimated, total=len(parquets),
        )

    # ─── 3. 수급 데이터 (섹터별) ───

    @staticmethod
    def _json_asof(data: dict) -> str:
        """산출물 JSON의 기준일 추출 — 최상위 → 항목 레벨 순으로 본다.

        ★8/10 실측(B-53 ②): `investor_flow.json`은 최상위에 날짜가 없고
        `sectors[].date`에 항목마다 들어 있다. 구 검사는 최상위만 봐서 **매일
        `⚠️(날짜없음)`**이었는데, `found>=2` 임계에 가려 3/3이 아니어도 통과라
        아무도 보지 않았다. 파일은 매일 17:57에 정상 생성되고 있었고 19개 항목
        전부 당일자였다 — **데이터가 아니라 검사가 틀린 것**이었다.
        """
        d = data.get("date") or data.get("generated_at") or ""
        if d:
            return str(d)
        items = data.get("sectors") or data.get("items") or []
        if isinstance(items, dict):
            items = list(items.values())
        dates = {str(it.get("date")) for it in items
                 if isinstance(it, dict) and it.get("date")}
        # 항목마다 날짜가 갈리면 기준일을 특정할 수 없으므로 판정하지 않는다.
        return dates.pop() if len(dates) == 1 else ""

    def _check_supply_sectors(self) -> CheckResult:
        """섹터 수급 JSON 존재 + 오늘 날짜.

        ★8/10 교정 2건(B-53 ②): ⑴기준일 추출을 항목 레벨까지 폴백 ⑵임계
        `found>=2` → **전량 요구**. 3개 중 1개가 죽어도 통과하던 것은 감시
        구멍이고, 실제로 investor_flow가 매일 ⚠️인 채 통과하고 있었다.
        """
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
                    d = self._json_asof(data)
                    if self.today_str in str(d) or self.today_compact in str(d):
                        found += 1
                        details.append(f"{name}✅")
                    else:
                        details.append(f"{name}⚠️({d[:10] if d else '날짜없음'})")
                except Exception:
                    details.append(f"{name}❌")
            else:
                details.append(f"{name}❌(없음)")

        # ★당일 산출물이 아직 하나도 없으면 BAT-D 전 실행 — 판정 보류.
        #   판단 소스는 종가 존재(#18·#2와 동일 원칙). 섹터 JSON은 BAT-D 안에서
        #   종가 재계산(17:43~17:56) 다음에 생성되므로, 종가가 당일치인데 섹터만
        #   비어 있으면 시각과 무관하게 진짜 이상이다.
        if found == 0 and not self._price_has_today():
            return CheckResult(
                "수급(섹터)", True,
                f"당일 미생성 — 판정 보류(BAT-D 전) | {', '.join(details)}",
                count=0, total=len(files),
            )

        passed = found == len(files)
        return CheckResult(
            "수급(섹터)", passed,
            f"{found}/{len(files)} — {', '.join(details)}",
            count=found, total=len(files),
        )

    # ─── 4. 국적별 수급 ───

    def _check_nationality(self) -> CheckResult:
        """nationality signal JSON + DB 신선도 동시 확인.

        재발방지: signal은 정상인데 DB 수집이 403으로 중단된 사고 (2026-03-25~26).
        """
        json_path = self.data_dir / "krx_nationality" / "nationality_signal.json"
        db_path = self.data_dir / "krx_nationality" / "nationality.db"

        signal_ok = False
        signal_info = ""
        db_ok = False
        db_info = ""

        # Signal JSON 체크
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                d = str(data.get("analyzed_at", data.get("date",
                        data.get("trade_date", ""))))
                yesterday = (self.today - timedelta(days=1)).strftime("%Y-%m-%d")
                if self.today_str in d or yesterday in d or self.today_compact in d:
                    count = data.get("total_stocks", len(data.get("signals", [])))
                    signal_ok = True
                    signal_info = f"signal={count}종목({d[:10]})"
                else:
                    signal_info = f"signal STALE({d[:10]})"
            except Exception as e:
                signal_info = f"signal 오류: {e}"

        # DB 신선도 체크
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT date, stock_count, status FROM collect_log "
                    "WHERE stock_count > 0 "
                    "ORDER BY date DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    last_date, stock_count, status = row
                    # date 형식: YYYYMMDD 또는 YYYY-MM-DD 모두 지원
                    fmt = "%Y%m%d" if "-" not in str(last_date) else "%Y-%m-%d"
                    gap = (self.today - datetime.strptime(str(last_date), fmt).date()).days
                    if gap <= 5 and stock_count > 0:
                        db_ok = True
                        db_info = f"db={last_date}({stock_count}종목)"
                    elif stock_count == 0:
                        db_info = f"db={last_date}(0종목! API오류)"
                    else:
                        db_info = f"db STALE({last_date}, {gap}일전)"
            except Exception:
                db_info = "db 쿼리 실패"

        passed = signal_ok and db_ok
        msg = f"{signal_info} | {db_info}" if db_info else signal_info
        if not signal_ok and not db_ok:
            msg = msg or "데이터 없음"
        return CheckResult("국적별수급", passed, msg)

    # ─── 5. 원자재 가격 ───

    COMMODITY_STALE_DAYS = 7  # 원자재는 일일 시세 — 7일 넘으면 STALE

    def _days_old(self, date_str: str) -> int | None:
        """YYYY-MM-DD(또는 YYYYMMDD) 문자열 → 오늘 대비 경과일. 파싱 불가면 None."""
        s = str(date_str)[:10].strip()
        if not s:
            return None
        fmt = "%Y%m%d" if "-" not in s else "%Y-%m-%d"
        try:
            return (self.today - datetime.strptime(s, fmt).date()).days
        except ValueError:
            return None

    def _check_commodities(self) -> CheckResult:
        """원자재 가격 — 소스 + 신선도 + **값 생존**.

        신선도 게이트 (7/16): 날짜가 7일 초과 과거면 FAIL.

        ★★8/10 실측 정정(B-53 ④) — 1·2차 순서가 사실과 반대였다. 구 로직은
        `liquidity_signal.json`을 1차로 보고 `commodity_prices.json`을
        "Alpha Vantage, 수집기 3/31 아카이브, 유물 가능성 높음"으로 취급해
        폴백에 두었다. 실측하니 정반대다:
          - `liquidity_signal.json`에는 원자재 키가 **아예 없다**
            (date·data_date·stale_days·indicators·regime·signals·composite_*).
            유동성 지표 파일이지 원자재 파일이 아니다.
          - `commodity_prices.json`은 **오늘 06:10 갱신**, `source`가
            `"yfinance futures (v2, 7/17 재구축)"` — 7/17 B-12로 부활한 정본이다.
        즉 검사는 매번 1차에서 헛돈 뒤 폴백으로 통과하고 있었고, **주석만 낡은
        채 남아 있었다.** 틀린 주석을 방치하면 언젠가 "유물이니 지우자"는 판단으로
        진짜 소스가 사라진다 — 순서를 사실에 맞추고 근거를 실측으로 갱신한다.
        """
        # 1차: commodity_prices.json (정본 — yfinance futures v2, 7/17 재구축)
        commodity_path = self.data_dir / "commodity_prices.json"
        if commodity_path.exists():
            try:
                data = json.loads(commodity_path.read_text(encoding="utf-8"))
                commodities = data.get("commodities", {})
                d = str(data.get("date", ""))[:10]
                if commodities:
                    # ★값 생존 — 파일이 있고 날짜가 최신이어도 가격이 0이면 죽은 것.
                    #   구 로직은 `v.get('price','?')`로 표시만 하고 판정엔 안 썼다.
                    prices = {k: (v.get("price") if isinstance(v, dict) else v)
                              for k, v in commodities.items()}
                    live = {k: p for k, p in prices.items()
                            if isinstance(p, (int, float)) and p > 0}
                    dead = [k for k in prices if k not in live]
                    names = [f"{k}={prices[k]}" for k in list(commodities)[:4]]

                    gap = self._days_old(d)
                    if gap is not None and gap > self.COMMODITY_STALE_DAYS:
                        return CheckResult("원자재", False,
                                           f"STALE({d}, {gap}일전): {', '.join(names)}")
                    if not live:
                        return CheckResult("원자재", False,
                                           f"🚨{len(prices)}종 전량 0/결측 ({d})")
                    detail = f"{', '.join(names)} ({d})"
                    if dead:
                        detail += f" | 🚨값없음: {', '.join(dead)}"
                    return CheckResult("원자재", not dead, detail,
                                       count=len(live), total=len(prices))
            except Exception:
                pass  # fallthrough

        # 2차: liquidity_signal.json — 현재는 원자재 키가 없으나 부활 대비로 남긴다
        signal_path = self.data_dir / "liquidity_cycle" / "liquidity_signal.json"
        if signal_path.exists():
            try:
                data = json.loads(signal_path.read_text(encoding="utf-8"))
                commodities = data.get("commodities", data.get("asset_prices", {}))
                if commodities:
                    d = str(data.get("date", ""))[:10]
                    gap = self._days_old(d)
                    if gap is not None and gap > self.COMMODITY_STALE_DAYS:
                        return CheckResult("원자재", False,
                                           f"{len(commodities)}종 STALE({d}, {gap}일전)")
                    return CheckResult("원자재", True,
                                       f"{len(commodities)}종 (날짜: {d}, 폴백소스)")
            except Exception:
                pass

        return CheckResult("원자재", False, "commodity/liquidity 데이터 없음")

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

            # ★8/10(B-53 ⑥): 구 로직은 `k in signals`로 **키 존재만** 봤다.
            #   키가 남아 있고 값만 0/빈값으로 죽으면 그대로 ✅였다 — 오늘 ①~④와
            #   같은 형태다. 키가 아니라 **값이 유효한지**를 센다.
            fred_keys = ["vix_level", "us_grade", "rv_current"]

            def _valid(key: str) -> bool:
                v = signals.get(key)
                if v is None:
                    return False
                if isinstance(v, str):
                    return bool(v.strip()) and v not in ("?", "N/A")
                try:
                    return float(v) != 0  # VIX·RV가 0인 상태는 실재하지 않는다
                except (TypeError, ValueError):
                    return False

            found = sum(1 for k in fred_keys if _valid(k))
            dead = [k for k in fred_keys if k in signals and not _valid(k)]

            if recent and found >= 2:
                vix = signals.get("vix_level", "?")
                us = signals.get("us_grade", "?")
                detail = f"VIX={vix}, US={us} ({d})"
                if dead:
                    detail += f" | 🚨값죽음: {', '.join(dead)}"
                return CheckResult("FRED/매크로", not dead, detail)
            if recent:
                return CheckResult("FRED/매크로", False,
                                   f"🚨유효 신호 {found}/3 ({d})"
                                   + (f" — 값죽음: {', '.join(dead)}" if dead else ""))
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
                # ★8/10(B-53 ⑥): 구 로직은 계약 **개수만** 셌다. 4개가 들어와도
                #   net/long/short가 전부 0이면 죽은 데이터인데 ✅였다.
                #   계약 단위로 하나라도 비영값이 있으면 살아 있는 것으로 본다
                #   (특정 계약의 net이 0인 것은 실재하므로 전량 0만 잡는다).
                live = 0
                for v in contracts.values():
                    if not isinstance(v, dict):
                        live += 1 if v else 0
                        continue
                    nums = []
                    for f in ("net", "long", "short"):
                        try:
                            nums.append(float(v.get(f)))
                        except (TypeError, ValueError):
                            pass
                    if any(n != 0 for n in nums):
                        live += 1
                if live == 0:
                    return CheckResult("COT", False,
                                       f"🚨{count}개 계약 전량 0 — 수집은 됐으나"
                                       f" 값이 비었다 ({report_date}{stale_str})")
                return CheckResult("COT", passed,
                                   f"{live}/{count}개 포지션 실값"
                                   f" ({report_date}{stale_str})",
                                   count=live, total=count)
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

        # ★당일 파일이 없고 종가도 당일치가 없으면 BAT-D 전 실행 — 판정 보류
        #   (#2·#3·#18과 동일 원칙). 학습기록은 BAT-D 후반(≈18:49)에 생성된다.
        if not has_daily and not self._price_has_today():
            return CheckResult("학습기록", True,
                               "당일 미생성 — 판정 보류(BAT-D 전)")

        if has_daily and has_index:
            try:
                data = json.loads(today_file.read_text(encoding="utf-8"))
                # ★8/10 교정(B-53 ③): 구 검사는 `signal_count`/`total_signals`를
                #   찾았으나 **둘 다 존재하지 않는 키**라 매일 `(시그널: ?)`를
                #   찍고 있었다. 게다가 파일 존재만으로 통과시켜 **내용이 비어도
                #   ✅**였다. 실제 산출물 키는 summary.total(분석 종목 수)과
                #   signal_accuracy(평가 엔진 dict)다.
                summary = data.get("summary") or {}
                total = summary.get("total", 0)
                engines = len(data.get("signal_accuracy") or {})
                ok = total > 0 and engines > 0
                detail = (f"daily({self.today_str}) + index"
                          f" | 분석 {total}종목 · 평가엔진 {engines}개")
                if not ok:
                    detail += " — 🚨내용 비어 있음(학습이 헛돌았다)"
                return CheckResult("학습기록", ok, detail,
                                   count=total, total=engines)
            except Exception as e:  # noqa: BLE001
                # 파싱 실패를 통과로 세지 않는다 — 구 로직은 except에서 True를
                # 반환해 파손 파일도 ✅였다.
                return CheckResult("학습기록", False,
                                   f"daily({self.today_str}) 파싱 오류: {e}")

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
            "journal": self.data_dir / "market_journal" / "daily" / f"{self.today_str}.json",
        }

        found = 0
        details = []
        for name, path in upload_files.items():
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                # ★8/10(B-53 ⑥): 구 로직은 **mtime만** 봤다. 업로드 스크립트가
                #   빈 JSON을 매일 새로 쓰면 mtime은 갱신되므로 영원히 ✅다 —
                #   7/31 "적재는 됐는데 값이 0"과 정확히 같은 형태.
                #   내용이 실제로 들어 있는지(파싱 성공 + 비어 있지 않음)를 본다.
                size = path.stat().st_size
                try:
                    body = json.loads(path.read_text(encoding="utf-8"))
                    empty = not body or (isinstance(body, (dict, list)) and len(body) == 0)
                except Exception:  # noqa: BLE001
                    body, empty = None, True
                if mtime.date() != self.today:
                    details.append(f"{name}⚠️({mtime.date()})")
                elif empty:
                    details.append(f"{name}🚨빈내용({size}B)")
                else:
                    found += 1
                    details.append(f"{name}✅({size // 1024}KB)")
            else:
                details.append(f"{name}❌")

        passed = found >= 1
        return CheckResult("업로드", passed,
                           f"{found}/{len(upload_files)} — {', '.join(details)}",
                           count=found, total=len(upload_files))

    # ─── 15. 파이프라인 에러율 ───

    def _check_pipeline_errors(self) -> CheckResult:
        """최근 24시간 파이프라인 에러율 확인."""
        try:
            from src.pipeline_alert import get_recent_error_rate
            stats = get_recent_error_rate(hours=24)

            if stats["total_runs"] == 0:
                return CheckResult("파이프라인", True, "기록 없음 (정상)")

            rate = stats["error_rate_pct"]
            errors = stats["total_errors"]
            scripts = stats["scripts_with_errors"]

            if rate < 5:
                return CheckResult(
                    "파이프라인", True,
                    f"에러율 {rate}% ({errors}건) — 정상",
                    count=errors, total=stats["total_runs"],
                )
            else:
                return CheckResult(
                    "파이프라인", False,
                    f"에러율 {rate}% ({errors}건) — {', '.join(scripts)}",
                    count=errors, total=stats["total_runs"],
                )
        except Exception as e:
            return CheckResult("파이프라인", True, f"체크 불가: {e}")

    # ─── 16. KOSPI 인덱스 신선도 ───

    def _check_kospi_index(self) -> CheckResult:
        """kospi_index.csv 마지막 날짜가 3영업일(≈5달력일) 이내인지."""
        csv_path = self.data_dir / "kospi_index.csv"
        if not csv_path.exists():
            return CheckResult("KOSPI인덱스", False, "kospi_index.csv 없음")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if df.empty:
                return CheckResult("KOSPI인덱스", False, "CSV 비어있음")

            date_col = "Date" if "Date" in df.columns else df.columns[0]
            last_date_str = str(df[date_col].iloc[-1])[:10]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            gap = (self.today - last_date).days

            # ★8/10(B-53 ⑤): 날짜·행수만 보던 것에 **마지막 행 값 생존**을 더한다.
            #   지수가 최신 날짜로 들어와도 close가 0이면 레짐·알파 계산이 통째로
            #   무너지는데 구 검사는 ✅였다(7/31 `price=0` 6주와 같은 형태).
            #   개별 컬럼 0은 정상일 수 있으므로 **전량 0/NaN**만 잡는다.
            dead = self._dead_row_cols(df, date_col)
            if dead is True:
                return CheckResult("KOSPI인덱스", False,
                                   f"🚨{last_date_str} 마지막 행 값 전량 0/결측"
                                   f" — 지수 계산 붕괴 위험!")

            if gap <= 5:
                close = df["close"].iloc[-1] if "close" in df.columns else "?"
                return CheckResult("KOSPI인덱스", True,
                                   f"최신: {last_date_str} ({gap}일 전, {len(df)}행,"
                                   f" close={close})")
            return CheckResult("KOSPI인덱스", False,
                               f"STALE: {last_date_str} ({gap}일 전) — 갱신 필요!")
        except Exception as e:
            return CheckResult("KOSPI인덱스", False, f"파싱 오류: {e}")

    # ─── 17. 투자자수급 신선도 ───

    def _check_investor_flow(self) -> CheckResult:
        """kospi_investor_flow.csv 마지막 날짜가 3영업일(≈5달력일) 이내인지."""
        csv_path = self.data_dir / "kospi_investor_flow.csv"
        if not csv_path.exists():
            return CheckResult("투자자수급", False, "kospi_investor_flow.csv 없음")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if df.empty:
                return CheckResult("투자자수급", False, "CSV 비어있음")

            date_col = "Date" if "Date" in df.columns else df.columns[0]
            last_date_str = str(df[date_col].iloc[-1])[:10]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            gap = (self.today - last_date).days

            # ★8/10(B-53 ⑤): 값 생존 추가. 순매수가 정확히 0인 주체는 드물게
            #   있을 수 있으므로 개별 0은 통과시키고 **전량 0/NaN**만 잡는다.
            dead = self._dead_row_cols(df, date_col)
            if dead is True:
                return CheckResult("투자자수급", False,
                                   f"🚨{last_date_str} 마지막 행 값 전량 0/결측"
                                   f" — 수집은 됐으나 값이 비었다!")

            if gap <= 5:
                fn = df["foreign_net"].iloc[-1] if "foreign_net" in df.columns else "?"
                return CheckResult("투자자수급", True,
                                   f"최신: {last_date_str} ({gap}일 전, {len(df)}행,"
                                   f" 외인 {fn})")
            return CheckResult("투자자수급", False,
                               f"STALE: {last_date_str} ({gap}일 전) — 수집 파이프라인 확인!")
        except Exception as e:
            return CheckResult("투자자수급", False, f"파싱 오류: {e}")

    # ─── 18. 투자자수급 커버리지 급감 감지 ───

    def _check_supply_coverage(self) -> CheckResult:
        """investor_daily.db 종목수 커버리지 — 당일 급감 + D-1 익일채움 검증.

        _check_supply_stocks(샘플30·임계80)가 못 잡는 KIS 대량 수집장애를 방어한다.
        일상 변동(거래정지 등락 ±수%)은 통과, 15%p 초과 급감만 경고(오탐 최소).
        신선도는 _check_investor_flow가 별도로 보므로 여기선 커버리지만 본다.

        ★7/30 확정 규칙(껍데기 가설, 7/27 발견→7/28 반증→7/30 확정): 껍데기(실거래0)
        종목은 당일 BAT-D 안이 아니라 **익일 수집(KIS 30일 재조회)에서 채워진다**.
        당일 종목수 ~2,64x는 정상(부분)이고 종목수 정본은 D+1.

        ★★8/10 교정 — 총 종목수는 완전성 지표가 아니다. 7/30 규칙은 "채워진다"는
        사실만 맞았고 **무엇이 채워지는지**를 안 봤다. 8/6↔8/7 실측: D+1 추가분
        204종목 중 **197종목(96.6%)이 직전일에도 실거래값 0**인 무거래 종목
        (우선주·거래정지주)이고, 종가 parquet이 있는 나머지 3종도 당일 거래량이
        실제 0이었다. 즉 **D+1에 채워지는 것은 실값이 아니라 무거래 0행이며,
        실질 데이터는 당일 수집에서 이미 완비된다.**

        그래서 구 ①은 성격이 다른 둘을 비교하고 있었다 — today는 실거래 종목만
        (~2,62x), prev는 0행 포함 완전판(~2,827). **상시 7.2%의 가짜 감소가 깔려
        임계 15% 중 실질 여유가 7.8%뿐이었고, 실값 종목 200개가 통째로 누락돼도
        통과했다.** 이제 급감 판정을 **실거래값 있는 종목수끼리**(동종 비교)로
        바꾼다. 직전 12거래일 실측 변동폭이 1.3%(2,608~2,641)라 임계 5%는 4배
        여유이면서 130종목 누락부터 잡는다.

        ① 실값 급감 5% — 주 판정. baseline drift 0.
        ② D-1 총 종목수 채움 95% — 보조. 실값과 무관하나 0행 충전 파이프라인
           (KIS 30일 재조회) 자체의 고장 신호라 유지한다.
        """
        db = self.data_dir / "investor_flow" / "investor_daily.db"
        if not db.exists():
            return CheckResult("수급커버리지", False, "investor_daily.db 없음")
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
            try:
                dates = [r[0] for r in con.execute(
                    "SELECT DISTINCT date FROM investor_daily ORDER BY date DESC LIMIT 3")]
                if not dates:
                    return CheckResult("수급커버리지", False, "데이터 없음")
                counts = [con.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM investor_daily WHERE date=?",
                    (d,)).fetchone()[0] for d in dates]
                # ★8/10 신설: 실거래값 있는 종목수 — 급감 판정의 주 소스.
                # 총 종목수와 달리 무거래 0행에 영향받지 않아 당일↔전일이 동종 비교가 된다.
                real_counts = [con.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM investor_daily WHERE date=?"
                    " AND (COALESCE(buy_val,0)>0 OR COALESCE(sell_val,0)>0)",
                    (d,)).fetchone()[0] for d in dates]
            finally:
                con.close()
        except Exception as e:  # noqa: BLE001
            return CheckResult("수급커버리지", False, f"DB 조회 오류: {e}")

        today_n = counts[0]

        # ★당일 결손 감지(7/30 검수 F2): DB에 존재하는 날짜들끼리만 비교하면, 당일 수집이
        # 0행으로 끝날 때(collect_investor_kis가 거래일 확정 실패 시 조용히 종료하는 경로
        # 실재) 어제↔그제를 비교해 "정상"을 찍고, 결손일은 다음날 DISTINCT 목록에서 사라져
        # 영구 불가시였다.
        # ★판정 소스는 시각이 아니라 **종가 parquet**(B-19 ① 교훈: 당일검증을 시각 조건으로
        # 넣으면 아침 수동 실행에서 오탐이고 시각 하드코딩은 더 나쁘다). 종가가 이미 당일치인데
        # 수급만 없으면 시각과 무관하게 진짜 이상 — 두 파이프라인은 서로 독립이다(B-13 원칙).
        if self.is_trading_day and dates[0] != self.today_compact:
            if self._price_has_today():
                return CheckResult(
                    "수급커버리지", False,
                    f"당일({self.today_compact}) 수급 없음 — DB 최신 {dates[0]} {today_n}종목"
                    f" (종가는 당일 존재) KIS 수급 수집 전면 실패 의심!",
                    count=today_n)
            # 종가도 아직 없음 = 수집 전 시각(아침 수동 실행 등) → 당일 결손 판정 보류

        if len(dates) < 2:
            return CheckResult("수급커버리지", True,
                               f"{dates[0]} {today_n}종목 (직전일 없음)",
                               count=today_n)
        prev_n = counts[1]
        real_today, real_prev = real_counts[0], real_counts[1]

        # ① 당일 급감 — ★8/10부터 실거래값 있는 종목수끼리 비교(동종). 총 종목수 비교는
        #   무거래 0행 때문에 상시 7.2% 가짜 감소가 깔려 200종목 누락을 못 잡았다.
        drop = (real_prev - real_today) / real_prev if real_prev else 0.0
        drop_ok = drop <= 0.05

        # ② D-1 익일채움 (D-2 대비 — 둘 다 완전판이어야 정상)
        fill_note = ""
        fill_ok = True
        if len(dates) >= 3 and counts[2]:
            fill_ratio = prev_n / counts[2]
            fill_ok = fill_ratio >= 0.95
            fill_note = "·채움OK" if fill_ok else (
                f"·채움미달 {fill_ratio * 100:.1f}% (D-2 {dates[2]} {counts[2]})")

        passed = drop_ok and fill_ok
        if not drop_ok:
            state = (f"실값 급감 {drop * 100:.1f}%"
                     f" ({real_prev}→{real_today}) — KIS 수집장애 의심!")
        elif not fill_ok:
            state = "D-1 익일채움 실패 — KIS 30일 재조회 확인!"
        else:
            state = "정상"
        return CheckResult(
            "수급커버리지", passed,
            f"{dates[0]} 실값 {real_today}종목/총 {today_n}"
            f" (직전 {dates[1]} 실값 {real_prev}/총 {prev_n}{fill_note}) {state}",
            count=real_today, total=real_prev)

    # ─── (폐기) 수급이면분석 (supply_demand/) 신선도 ───

    def _check_supply_demand(self) -> CheckResult:
        """supply_demand/ JSON의 최신 날짜가 3영업일(≈5달력일) 이내인지.

        재발방지: config tickers 비어 21일간 수집 중단 사고 (2026-03-05~26).
        """
        sd_dir = self.data_dir / "supply_demand"
        if not sd_dir.exists():
            return CheckResult("수급이면분석", False, "supply_demand/ 디렉토리 없음")

        json_files = sorted(sd_dir.glob("*.json"))
        if not json_files:
            return CheckResult("수급이면분석", False, "JSON 파일 0개 — 수집 중단 의심!")

        latest = json_files[-1].stem  # e.g. "20260326"
        try:
            latest_date = datetime.strptime(latest, "%Y%m%d").date()
        except ValueError:
            return CheckResult("수급이면분석", False, f"파일명 파싱 실패: {latest}")

        gap = (self.today - latest_date).days

        # 파일 크기로 빈 파일 감지
        size_kb = json_files[-1].stat().st_size / 1024

        if gap <= 5 and size_kb > 1:
            return CheckResult(
                "수급이면분석", True,
                f"최신: {latest} ({gap}일 전, {size_kb:.0f}KB, {len(json_files)}일치)",
            )
        elif size_kb <= 1:
            return CheckResult(
                "수급이면분석", False,
                f"빈 파일: {latest} ({size_kb:.1f}KB) — collect_investor_kis 확인!",
            )
        else:
            return CheckResult(
                "수급이면분석", False,
                f"STALE: {latest} ({gap}일 전) — BAT-D2 스케줄 또는 load_tickers 확인!",
            )

    # ─── 19. 수급 스냅샷 (supply_snapshots/) 신선도 ───

    def _check_supply_snapshots(self) -> CheckResult:
        """supply_snapshots/ 장중 스냅샷이 오늘(평일) 생성되었는지.

        재발방지: 데몬 비활성화 후 20일간 스냅샷 미수집 사고 (2026-03-06~26).
        """
        snap_dir = self.data_dir / "supply_snapshots"
        if not snap_dir.exists():
            return CheckResult("수급스냅샷", False, "supply_snapshots/ 디렉토리 없음")

        # 주말이면 스킵 (장중 스냅샷은 평일에만)
        if not self.is_weekday:
            return CheckResult("수급스냅샷", True, "주말 — 스냅샷 불필요")

        today_compact = self.today.strftime("%Y%m%d")
        today_snaps = list(snap_dir.glob(f"{today_compact}_snap*.json"))

        if today_snaps:
            # 빈 파일 감지
            non_empty = 0
            for sf in today_snaps:
                try:
                    data = json.loads(sf.read_text(encoding="utf-8"))
                    if data.get("stocks"):
                        non_empty += 1
                except Exception:
                    pass
            if non_empty > 0:
                return CheckResult(
                    "수급스냅샷", True,
                    f"오늘 {len(today_snaps)}개 스냅샷 ({non_empty}개 데이터有)",
                )
            return CheckResult(
                "수급스냅샷", False,
                f"오늘 {len(today_snaps)}개 스냅샷 — 전부 stocks 비어있음!",
            )

        # 오늘 스냅샷 없으면 최신 파일 확인
        all_snaps = sorted(snap_dir.glob("*_snap*.json"))
        if all_snaps:
            latest = all_snaps[-1].stem.split("_")[0]
            return CheckResult(
                "수급스냅샷", False,
                f"오늘 스냅샷 없음 — 최신: {latest} (BAT-SNAP 스케줄 확인!)",
            )
        return CheckResult("수급스냅샷", False, "스냅샷 파일 0개")

    # ─── 14. 스케줄러 로그 ───

    def _check_scheduler(self) -> CheckResult:
        """오늘 cron 로그에서 BAT-D 실행 여부 + 실패 건수 파싱."""
        # VPS cron 기반: logs/cron_YYYYMMDD.log
        log_name = f"cron_{self.today.strftime('%Y%m%d')}.log"
        log_path = PROJECT_ROOT / "logs" / log_name
        if not log_path.exists():
            # 주말이면 OK
            if self.today.weekday() >= 5:
                return CheckResult("스케줄러", True, "주말 — cron 미실행 정상")
            return CheckResult("스케줄러", False, f"{log_name} 없음")

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # BAT-D 실행 여부
            bat_d_start = [l for l in lines if "BAT-D 시작" in l]
            bat_d_end = [l for l in lines if "BAT-D 완료" in l]

            if not bat_d_start:
                return CheckResult("스케줄러", False, "BAT-D 시작 로그 없음")

            # 실패 건수 (WARN 라인)
            warn_lines = [l for l in lines if "[WARN]" in l]
            fail_count = len(warn_lines)

            if bat_d_end:
                # "=== BAT-D 완료 (실패: 2건) ===" 파싱
                m = re.search(r"실패:\s*(\d+)건", bat_d_end[-1])
                if m:
                    fail_count = int(m.group(1))

            if fail_count == 0:
                return CheckResult("스케줄러", True,
                                   f"BAT-D 정상 완료, 실패 0건")
            else:
                return CheckResult("스케줄러", fail_count <= 3,
                                   f"BAT-D 완료, 실패 {fail_count}건")
        except Exception as e:
            return CheckResult("스케줄러", False, f"로그 파싱 오류: {e}")


# ──────────────────────────────────────────────
# 보고서 생성
# ──────────────────────────────────────────────

#: 의도적으로 중단한 소스 — 실패해도 등급 계산에서 제외한다(B-44, 8/10).
#: ★반드시 명시적 등재로만 동작한다. "실패하면 자동 retired"로 만들면 일시적
#: 장애까지 등급에서 빠져 진짜 고장을 숨긴다(8/1 "억제 설계가 뚫림"과 같은 계열).
#: ★감시를 끄는 것이 아니다 — 데이터가 다시 오면 passed=True가 되어 아래 조건
#: (not r.passed)에서 자동으로 빠져 정상 판정 대상으로 복귀한다.
RETIRED_CHECKS = {
    "국적별수급",  # 단타봇 재연결 대기. 8/5 `2ddf8332`로 부활 경로까지 차단된 상태
}


def generate_health_report(results: list[CheckResult], check_date: date) -> str:
    """체크 결과 → 등급 + 텔레그램 메시지.

    휴장일 SKIP 항목(7/20 B-13)은 분모·분자 모두에서 제외 — 판정 대상이 아닌 것을
    '통과'로 세면 등급이 부풀고, '실패'로 세면 가짜 경보가 된다.

    ★8/10 B-44: 의도적 중단 소스(RETIRED_CHECKS)도 같은 이유로 분모에서 뺀다.
    국적별수급이 매일 ❌로 세어져 **17/18 = B등급이 영구 고정**돼 있었고, 그러면
    진짜 이상이 생겨도 "B등급이 늘 그렇지"로 무뎌진다. 리포트엔 경과일수와 함께
    남겨 방치도 함께 막는다.
    """
    skipped = [r for r in results if r.skipped]
    retired = [r for r in results
               if not r.skipped and not r.passed and r.name in RETIRED_CHECKS]
    judged = [r for r in results if not r.skipped and r not in retired]
    passed = sum(1 for r in judged if r.passed)
    total = len(judged)

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
        f"합계: {passed}/{total} 통과"
        + (f" (휴장일 SKIP {len(skipped)}건 제외)" if skipped else "")
        + (f" (의도적 중단 {len(retired)}건 제외)" if retired else ""),
        "",
    ]

    # 실패 항목 먼저
    failed = [r for r in judged if not r.passed]
    if failed:
        lines.append("── 실패 항목 ──")
        for r in failed:
            lines.append(f"  {r}")
        lines.append("")

    if skipped:
        lines.append("── 휴장일 SKIP (판정 제외) ──")
        for r in skipped:
            lines.append(f"  {r}")
        lines.append("")

    if retired:
        # 존재는 남긴다 — 등급에서 뺐다고 잊히면 복구해야 할 것이 방치된다.
        # detail에 경과일수가 들어 있어 숫자가 커지면 눈에 띈다.
        lines.append("── 의도적 중단 (판정 제외 · 복구 시 자동 복귀) ──")
        for r in retired:
            lines.append(f"  ⏸️ {r.name}: {r.detail}")
        lines.append("")

    # 성공 항목
    passed_items = [r for r in judged if r.passed]
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

    # 텔레그램 전송 — [HEALTH] 태그로 QUIET 모드 통과
    if not args.no_telegram:
        try:
            from src.telegram_sender import send_message
            send_message(f"[HEALTH] {report}")
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
