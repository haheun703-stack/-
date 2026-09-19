"""정보봇 원천 CSV → raw parquet 공매도·대차 컬럼 배선 (B-47).

배경
  `short_volume`·`short_ratio`·`short_balance`·`lending_balance` 4컬럼이
  2026-02-19부터 전량 0이다. 원 수집기(pykrx 경로)가 제거된 뒤 **대체 소스가
  배선되지 않은 채 컬럼만 0으로 남아** 계속 계산에 들어갔다(B-45 §1 위반).
  3봇 분업상 공매도·신용·대차는 정보봇 담당이고, 원천이 매일 도착하고 있다:
    /home/ubuntu/jgis/data/supply_tracker/{ticker}.csv

매핑 (2026-08-21 실측으로 확정)
  short_volume    ← short_selling_qty          ✅
  lending_balance ← loan_balance_qty           ✅
  short_ratio     ← short_selling_qty / volume × 100   ✅ (정의: 공매도 비중 %)
  short_balance   ← short_balance_qty  ✅ **2026-09-07 배선**
                    8/21에는 원천에 이 필드가 없어 비워 뒀다. `loan_balance_qty`(대차잔고)로
                    대체하지 않은 판단이 맞았고(정의가 다르다), 퀀트봇 요청으로 정보봇이
                    CSV에 컬럼을 추가하자(`f62790e`, 9/7 16:50 실행분부터) 그대로 배선됐다.
                    ⚠️대상은 아직 **시총 상위 30종목**이다 — 확대(300→889)는 정보봇 실측·결재 중.
                    `short_cover_signal`은 그 커버리지 위에서만 유효하므로 확대 전까지 신호로 쓰지 않는다.

★가장 중요한 규칙 — 0을 채우지 않는다
  원천 CSV의 공매도/대차 필드는 빈칸이 아니라 **명시적 `0`**으로 들어온다.
  그런데 삼성전자 90행 중 73행(81%)이 0이다 — 삼성전자가 공매도 0인 날이
  81%일 수는 없으므로 그 0은 **미수집**이다. 미수집과 진짜 0이 파싱 단계에서
  구분되지 않는다(B-80이 지적한 것과 같은 구조).
  → **비0 값만 기록하고 나머지는 건드리지 않는다.** 0으로 덮으면 이 스크립트가
    B-47을 고치면서 같은 결함을 새로 심는 꼴이 된다.

★실효 시점 주의
  원천이 실제로 채워지는 구간은 **2026-08부터**다(3~7월은 loan/credit 전량 0,
  short도 간헐 0~4일). 즉 지금 배선해도 확보되는 건 **약 11거래일치**이고,
  `short_ratio_ma40`(40일)은 40거래일이 쌓인 뒤에야 유효하다.
  그래도 지금 배선하지 않으면 영영 쌓이지 않는다.

실행
  python -u -X utf8 scripts/fill_short_from_jgis.py --dry-run
  python -u -X utf8 scripts/fill_short_from_jgis.py
  python -u -X utf8 scripts/fill_short_from_jgis.py --ticker 005930
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.jgis_short_adapter import JgisShortAdapter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [B-47] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fill_short")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

#: 원천에 대응 필드가 없어 복구할 수 없는 컬럼 — 명시적으로 남긴다.
UNMAPPABLE: dict[str, str] = {
    # ★9/7 해소: 정보봇이 `short_balance_qty`·`short_balance_ratio` 컬럼을 supply_tracker
    #   CSV에 추가(`f62790e`, 2026-09-07 16:50 실행분부터). 8/21에는 "원천에 필드 없음"이라
    #   비워 뒀던 자리다 — 대차잔고로 대체하지 않은 판단이 맞았고, 원천이 생기자 그대로 배선된다.
    #   ※대상은 아직 시총 상위 30종목이다(확대는 정보봇 실측·결재 진행 중).
}



# ──────────────────────────────────────────────
# 정의 검산 게이트 (B-105, 2026-09-19 신설)
# ──────────────────────────────────────────────
#
# 왜 필요한가
#   9/7에 정보봇이 `short_balance_qty`를 추가해 주자 이름만 보고 `short_balance`에
#   배선했다. 9/19 실측에서 그 값이 **전일 `short_selling_qty`와 완전일치**함이
#   드러났다 — 표본 300종목 중 값이 있는 48종목의 **43개(89.6%)**, 005930은
#   9/8~9/14 7거래일 전부. 이름은 「잔고」인데 내용은 **한 칸 밀린 거래량**이다.
#
#   배선이 동작하지 않아 오염은 없었지만, 동작했다면 「잔고」라는 이름으로 거래량이
#   들어가 `short_balance_chg_5d`·`short_cover_signal`을 **정반대로** 흔들었을 것이다.
#   8/21 `short_spike`가 정확히 그 형태였다 — **0으로 죽은 컬럼보다 왜곡된 컬럼이
#   훨씬 찾기 어렵다**(살아 있는 것처럼 보이기 때문이다).
#
# 규칙
#   **남이 준 컬럼은 이름이 아니라 분포로 받는다.** 채우기 **전에** 검산하고,
#   실패하면 그 컬럼만 건너뛴다(다른 컬럼은 정상 진행한다).
#
#   ※ 9/7에 나는 정보봇 `nationality_flow`를 «필드명이 `nation`이니 국적»이라
#     단정했다가 정정받았다. 같은 오류를 **주는 쪽·받는 쪽 양쪽에서** 했다.

#: 컬럼별 정의 검산. (검사명, 판정함수) — True면 위반.
#  판정함수는 (src_df) -> bool.
def _violates_shift(src: pd.DataFrame, field: str, other: str,
                    min_pairs: int = 3, thresh: float = 0.7) -> bool:
    """field[t] 가 other[t-1] 과 사실상 같은가 — 「한 칸 밀린 다른 지표」 탐지."""
    if field not in src.columns or other not in src.columns:
        return False
    a = pd.to_numeric(src[field], errors="coerce").to_numpy()
    b = pd.to_numeric(src[other], errors="coerce").to_numpy()
    pairs = [(a[i], b[i - 1]) for i in range(1, len(a))
             if pd.notna(a[i]) and a[i] > 0 and pd.notna(b[i - 1])]
    if len(pairs) < min_pairs:
        return False                      # 표본 부족 — 판정하지 않는다
    same = sum(1 for x, y in pairs if abs(x - y) < 1)
    return (same / len(pairs)) >= thresh


def _violates_constant(src: pd.DataFrame, field: str, min_n: int = 5) -> bool:
    """전 기간 한 값으로 고정 — 갱신이 멈춘 컬럼."""
    if field not in src.columns:
        return False
    v = pd.to_numeric(src[field], errors="coerce").dropna()
    v = v[v != 0]
    return len(v) >= min_n and v.nunique() == 1


#: {대상 parquet 컬럼: [(검사명, 검사함수)]}
DEFINITION_GATES: dict[str, list] = {
    "short_balance": [
        ("전일 공매도량과 shift 일치",
         lambda s: _violates_shift(s, "short_balance_qty", "short_selling_qty")),
        ("전 기간 상수", lambda s: _violates_constant(s, "short_balance_qty")),
    ],
}


def check_definition(col: str, src: pd.DataFrame) -> str | None:
    """위반 시 사유 문자열, 통과 시 None."""
    for name, fn in DEFINITION_GATES.get(col, []):
        try:
            if fn(src):
                return name
        except Exception:                 # 검산 자체의 고장을 통과로 위장하지 않는다
            return f"{name}(검산 실패)"
    return None


def fill_one(ticker: str, rows: list[dict], dry: bool) -> dict:
    """단일 종목 raw parquet에 공매도·대차 채움. 0은 채우지 않는다."""
    out = {"ticker": ticker, "status": "skip", "short": 0, "lend": 0,
           "ratio": 0, "sbal": 0}
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        out["status"] = "no_parquet"
        return out

    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        out["status"] = f"read_err:{e}"
        return out

    src = pd.DataFrame(rows)
    if src.empty or "date" not in src.columns:
        out["status"] = "no_src"
        return out
    src["date"] = pd.to_datetime(src["date"], errors="coerce")
    src = src.dropna(subset=["date"]).set_index("date")

    idx = pd.to_datetime(df.index)
    common = idx.intersection(src.index)
    if len(common) == 0:
        out["status"] = "no_overlap"
        return out

    changed = False
    for col, field in (("short_volume", "short_selling_qty"),
                       ("lending_balance", "loan_balance_qty"),
                       # ★9/7: 정보봇 CSV에 추가된 공매도 잔고(수량). 원천에 컬럼이 없으면
                       #   아래 존재 검사에서 걸러진다 — 없는 날에도 안전하다.
                       ("short_balance", "short_balance_qty")):
        if field not in src.columns:
            # ★B-105: 여기서 조용히 넘어간 것이 12거래일 무동작의 원인이었다.
            #   매핑에 적어 둔 필드가 원천에 없으면 «배선했다»는 선언이 사실과
            #   어긋난 상태다. 침묵하지 않고 남긴다.
            out.setdefault("missing_field", []).append(f"{col}←{field}")
            continue
        # ★B-105 정의 검산 — 이름이 맞아도 내용이 다르면 채우지 않는다.
        bad = check_definition(col, src)
        if bad:
            out.setdefault("gated", []).append(f"{col}:{bad}")
            continue
        if col not in df.columns:
            df[col] = pd.NA
        vals = pd.to_numeric(src.loc[common, field], errors="coerce")
        vals = vals[vals > 0]                      # ★0은 채우지 않는다
        if len(vals) == 0:
            continue
        cur = pd.to_numeric(df.loc[vals.index, col], errors="coerce").fillna(0)
        target = vals.index[cur.values == 0]       # 기존 실값은 보존
        if len(target):
            df.loc[target, col] = vals.loc[target].values
            out[{"short_volume": "short",
                 "lending_balance": "lend",
                 "short_balance": "sbal"}[col]] = len(target)
            changed = True

    # short_ratio = 공매도 비중(%) — 공매도량과 거래량이 **둘 다 비0**일 때만
    if "volume" in df.columns:
        if "short_ratio" not in df.columns:
            df["short_ratio"] = pd.NA
        sv = pd.to_numeric(src.loc[common, "short_selling_qty"], errors="coerce")
        vol = pd.to_numeric(df.loc[common, "volume"], errors="coerce")
        ok = (sv > 0) & (vol > 0)
        if ok.any():
            ratio = (sv[ok] / vol[ok] * 100).round(4)
            cur = pd.to_numeric(df.loc[ratio.index, "short_ratio"],
                                errors="coerce").fillna(0)
            target = ratio.index[cur.values == 0]
            if len(target):
                df.loc[target, "short_ratio"] = ratio.loc[target].values
                out["ratio"] = len(target)
                changed = True

    if changed and not dry:
        df.to_parquet(path)
    out["status"] = "filled" if changed else "nothing_to_fill"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="정보봇 원천 → raw parquet 공매도 배선")
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 집계만")
    ap.add_argument("--ticker", help="단일 종목만")
    ap.add_argument("--lookback", type=int, default=400, help="원천 조회 일수")
    args = ap.parse_args()

    adapter = JgisShortAdapter()
    avail = adapter.list_available_tickers()
    if not avail:
        logger.error("원천 CSV 디렉토리가 비어 있다 — 정보봇 경로 확인 필요")
        return 1

    if args.ticker:
        targets = [args.ticker]
    else:
        have_parquet = {p.stem for p in RAW_DIR.glob("*.parquet")}
        targets = sorted(set(avail) & have_parquet)

    logger.info("원천 %d종목 · raw parquet 교집합 %d종목%s",
                len(avail), len(targets), " (DRY-RUN)" if args.dry_run else "")
    for col, why in UNMAPPABLE.items():
        logger.warning("복구 불가 컬럼 %s — %s", col, why)

    agg = {"filled": 0, "nothing_to_fill": 0, "no_overlap": 0,
           "no_parquet": 0, "no_src": 0, "err": 0}
    tot = {"short": 0, "lend": 0, "ratio": 0, "sbal": 0}
    gated: dict[str, int] = {}          # B-105 정의 검산에 걸린 컬럼별 종목 수
    missing: dict[str, int] = {}        # 매핑에 있으나 원천에 없는 필드
    for i, t in enumerate(targets, 1):
        rows = adapter.load_ticker_csv(t, lookback_days=args.lookback)
        if not rows:
            agg["no_src"] += 1
            continue
        r = fill_one(t, rows, args.dry_run)
        st = r["status"]
        agg[st] = agg.get(st, 0) + 1 if st in agg else agg.setdefault("err", 0) + 1
        for k in tot:
            tot[k] += r[k]
        for g in r.get("gated", []):
            gated[g] = gated.get(g, 0) + 1
        for m in r.get("missing_field", []):
            missing[m] = missing.get(m, 0) + 1
        if i % 200 == 0:
            logger.info("  진행 %d/%d — 채움 %d종목", i, len(targets), agg["filled"])

    logger.info("완료: %s", agg)
    logger.info("채운 셀 — short_volume %d · lending_balance %d · short_ratio %d"
                " · short_balance %d",
                tot["short"], tot["lend"], tot["ratio"], tot.get("sbal", 0))
    if missing:
        for m, n in sorted(missing.items(), key=lambda x: -x[1]):
            logger.warning("⚠️ 매핑 필드가 원천에 없다 — %s (%d종목). "
                           "「배선 완료」 선언과 실제가 어긋난 상태다.", m, n)
    if gated:
        # ★스킵을 조용히 넘기지 않는다 — 「성공 위장」 차단(단타봇 9/7 설계).
        for g, n in sorted(gated.items(), key=lambda x: -x[1]):
            logger.warning("🚧 정의 검산 차단 — %s (%d종목). 원천 정의가 확정될 "
                           "때까지 채우지 않는다.", g, n)
    if args.dry_run:
        logger.info("DRY-RUN이라 저장하지 않았다. 실제 반영은 --dry-run 없이 실행.")
    else:
        logger.info("★raw만 갱신했다. 소비처(processed)에 반영하려면 "
                    "rebuild_indicators가 돌아야 한다(8/7 교훈: 쓰기 성공 ≠ 소비처 도달).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
