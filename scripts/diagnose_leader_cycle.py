"""주도주 사이클 진단 CLI — 종목코드 입력 → 사이클 시계 + 액션 신호 JSON 산출.

데이터: data/raw/{code}.parquet (일봉 2019~, 수급·qoq_oi_growth 포함) → 주봉 리샘플.
엔진:   src/use_cases/leader_cycle_diagnosis.diagnose_leader_cycle (순수 함수).

사용:
    python -u -X utf8 scripts/diagnose_leader_cycle.py 086520 247540
    python -u -X utf8 scripts/diagnose_leader_cycle.py 086520 --as-of 2023-07-31
    python -u -X utf8 scripts/diagnose_leader_cycle.py 086520 --no-delta --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# PYTHONPATH 안전장치 (BAT/cron 호출 대비)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")   # DART_API_KEY 주입 (없으면 parquet 폴백)

import pandas as pd  # noqa: E402

from src.adapters.dart_adapter import DartAdapter  # noqa: E402
from src.use_cases.leader_cycle_diagnosis import diagnose_leader_cycle  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
US_DIR = PROJECT_ROOT / "data" / "us_market" / "leader_cycle"   # fetch_us_leader_data.py 산출
UNIVERSE = PROJECT_ROOT / "data" / "universe.csv"

_NAME_CACHE: dict | None = None
_DART: DartAdapter | None = None


def _get_dart() -> DartAdapter | None:
    """DartAdapter 지연 싱글턴 (CLI 1회 실행 내 캐시·corp_code 재사용)."""
    global _DART
    if _DART is None:
        try:
            _DART = DartAdapter()
        except Exception:   # noqa: BLE001
            _DART = None
    return _DART


def op_growth_from_dart(code: str, *, as_of: str | None = None) -> pd.Series | None:
    """DART 분기 영업이익 TTM-YoY 시계열 (KR 델타 게이트 1순위 소스).

    as_of(백테스트)면 그 연도까지만 조회해 미래 실적 누설 방지.
    """
    dart = _get_dart()
    if dart is None or not dart.is_available:
        return None
    end_year = None
    if as_of:
        try:
            end_year = pd.to_datetime(as_of).year
        except (ValueError, TypeError):
            end_year = None
    try:
        return dart.get_op_growth_series(code, end_year=end_year)
    except Exception:   # noqa: BLE001
        return None


def op_growth_from_us(code: str) -> pd.Series | None:
    """US 연간 영업이익 YoY 성장률 시계열 (US 델타 게이트 소스).

    yfinance 분기는 ~5개뿐이라 단일분기 YoY 델타 불가 → 연간 영업이익(4년) YoY 사용.
    다년 성장둔화가 '주도주 기간 가설'의 핵심이라 연간 granularity도 의미 있음.
    10-K 공시지연 +60일로 가용일 색인(백테스트 누설 차단).
    """
    p = US_DIR / f"{code}_opinc.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    annual = data.get("annual", {})
    if len(annual) < 3:   # YoY 2점(델타 1점) 위해 연간 3개 필요
        return None
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in annual.items()}).sort_index()
    yoy: dict[pd.Timestamp, float] = {}
    prev_val: float | None = None
    for dt, val in s.items():
        if prev_val is not None and prev_val > 0:
            g = (val - prev_val) / prev_val * 100
            yoy[dt + pd.Timedelta(days=60)] = round(max(-999.0, min(999.0, g)), 1)   # 노이즈 클램프
        prev_val = val
    return pd.Series(yoy).sort_index() if len(yoy) >= 2 else None


def _name_of(code: str) -> str:
    global _NAME_CACHE
    if _NAME_CACHE is None:
        try:
            df = pd.read_csv(UNIVERSE, dtype={"ticker": str})
            _NAME_CACHE = dict(zip(df["ticker"], df["name"]))
        except Exception:
            _NAME_CACHE = {}
    return _NAME_CACHE.get(code, code)


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉(금요일 마감). 모든 사이클 판정은 주봉 종가 기준(일봉 노이즈 금지)."""
    d = daily.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    return d.resample("W-FRI").agg(agg).dropna(subset=["close"])


def extract_op_growth(daily: pd.DataFrame) -> pd.Series | None:
    """parquet의 qoq_oi_growth → 분기별 영업이익 성장률 시계열(델타 게이트용).

    일별로 동일 분기값이 반복되므로 분기(Q) 마지막 값으로 압축. 값이 모두 결측이면 None.
    """
    if "qoq_oi_growth" not in daily.columns:
        return None
    if not isinstance(daily.index, pd.DatetimeIndex):
        daily = daily.copy()
        daily.index = pd.to_datetime(daily.index)
    s = daily["qoq_oi_growth"].dropna()
    # 죽은 컬럼 가드: 전부 0 또는 단일값이면 미충전 → 데이터 없음(델타 None, 신뢰도 하향)
    if s.empty or s.nunique() <= 1:
        return None
    q = s.resample("QE").last().dropna()
    return q if len(q) >= 2 else None


def load_and_diagnose(code: str, *, market: str, as_of: str | None,
                      use_delta: bool, params: dict | None) -> dict:
    # ── US: data/us_market/leader_cycle/{code}.parquet (이미 주봉) + 연간 영업이익 YoY ──
    if market == "US":
        path = US_DIR / f"{code}.parquet"
        if not path.exists():
            return {"ticker": code, "data_available": False,
                    "error": f"US 주봉 없음(먼저 fetch_us_leader_data.py 실행): {path.name}"}
        weekly = pd.read_parquet(path)   # 이미 주봉 OHLCV
        op_growth, delta_source = None, "off"
        if use_delta:
            op_growth = op_growth_from_us(code)
            delta_source = "US-annual" if op_growth is not None else "none"
        res = diagnose_leader_cycle(
            weekly, market="US", op_growth=op_growth, as_of=as_of,
            params=params, ticker=code, name=code,
        )
        res["delta_source"] = delta_source
        return res

    # ── KR: data/raw/{code}.parquet (일봉) → 주봉 리샘플 + DART TTM-YoY ──
    path = RAW_DIR / f"{code}.parquet"
    if not path.exists():
        return {"ticker": code, "data_available": False, "error": f"parquet 없음: {path.name}"}
    daily = pd.read_parquet(path)
    weekly = resample_weekly(daily)

    op_growth = None
    delta_source = "off"
    if use_delta:
        op_growth = op_growth_from_dart(code, as_of=as_of)       # DART TTM-YoY 우선
        if op_growth is not None:
            delta_source = "DART"
        if op_growth is None:
            op_growth = extract_op_growth(daily)                 # parquet 폴백(현 죽은 컬럼→None)
            if op_growth is not None:
                delta_source = "parquet"
        if op_growth is None:
            delta_source = "none"

    res = diagnose_leader_cycle(
        weekly, market=market, op_growth=op_growth, as_of=as_of,
        params=params, ticker=code, name=_name_of(code),
    )
    res["delta_source"] = delta_source
    return res


def _fmt(res: dict) -> str:
    """사람이 읽는 요약."""
    if not res.get("data_available"):
        return f"  [{res.get('ticker')}] {res.get('name', '')}  ✗ {res.get('error')}"
    icon = {"오전": "🌅", "정오": "☀", "오후": "🌆", "마감": "🌙", "사이클없음": "·"}.get(res["clock"], "?")
    sig = res["signal"]
    sig_icon = {"매수적기": "🟢", "보유": "🔵", "경계": "🟡", "청산": "🔴", "해당없음": "⚪"}.get(sig, "")
    age = res.get("age_months")
    age_s = f"{age:>4.1f}m" if age is not None else "  - "
    surv = res.get("survival_pct")
    surv_s = f"{surv*100:>3.0f}%" if surv is not None else " - "
    delta = res.get("delta_value")
    src = res.get("delta_source", "")
    src_tag = {"DART": "ᴰ", "parquet": "ᴾ", "US-annual": "ᵁ", "none": "·", "off": ""}.get(src, "")
    delta_s = f"Δ{delta:+.1f}{src_tag}" if delta is not None else f"Δ없음{src_tag}"
    head = (f"  {res['ticker']} {res.get('name',''):14} {icon}{res['clock']:5} "
            f"age={age_s} 생존={surv_s} {delta_s:>8} → {sig_icon}{sig}")
    lines = [head]
    for r in res.get("reasons", []):
        lines.append(f"        · {r}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="주도주 사이클 진단")
    ap.add_argument("codes", nargs="+", help="종목코드 (6자리)")
    ap.add_argument("--market", default="KR", choices=["KR", "US"])
    ap.add_argument("--as-of", default=None, help="진단 기준일 YYYY-MM-DD (백테스트)")
    ap.add_argument("--no-delta", action="store_true", help="델타(영업이익) 게이트 끄기")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    ap.add_argument("--tolerance", type=int, default=None, help="역추적 잔파동 허용 주수")
    args = ap.parse_args()

    params = {}
    if args.tolerance is not None:
        params["tolerance_weeks"] = args.tolerance
    elif args.market == "US":
        # US 캘리브레이션(backtest_leader_cycle_us): 메가캡은 리딩 중 깊은조정(NVDA 2025 -37%)이
        # 잦아 KR=6주로는 사이클이 파편화("방금 시작" 오독)됨. 10주가 파편화/과병합 절충.
        params["tolerance_weeks"] = 10

    results = [
        load_and_diagnose(c, market=args.market, as_of=args.as_of,
                          use_delta=not args.no_delta, params=params or None)
        for c in args.codes
    ]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        asof_s = args.as_of or "최신"
        print(f"=== 주도주 사이클 진단 (기준일={asof_s}, market={args.market}, "
              f"tolerance={params.get('tolerance_weeks', 'default')}) ===")
        for res in results:
            print(_fmt(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
