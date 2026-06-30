"""주도주 사이클 진단 — shadow 관측 러너 (BAT-D G5, 매매 미반영).

config/global_leaders.yaml(섹터별 글로벌 대장주 + KR 하위종목)을 워치리스트로
전 종목 사이클 진단 → data/shadow/leader_cycle.json 산출(관측 전용).

freeze 준수: 실주문·시그널·텔레그램 무접촉. 순수 관측 데이터만 생성.
대시보드 노출은 정보봇(FLOWX UI)이 이 JSON을 읽어 처리(global_leaders.yaml 주석 규약).

데이터:
  KR — data/raw/{code}.parquet + DART TTM-YoY 델타 (tolerance=6)
  US — data/us_market/leader_cycle/{tk}.parquet + 연간영업이익 YoY (tolerance=10)
       (US 가격/재무는 fetch_us_leader_data.py가 선행 갱신 — 로컬 Yahoo 차단, VPS 수집)

사용:
    python -u -X utf8 scripts/run_leader_cycle.py
    python -u -X utf8 scripts/run_leader_cycle.py --quiet   # 요약 출력 생략
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from scripts.diagnose_leader_cycle import load_and_diagnose  # noqa: E402

LEADERS_YAML = PROJECT_ROOT / "config" / "global_leaders.yaml"
OUT = PROJECT_ROOT / "data" / "shadow" / "leader_cycle.json"

US_TOL = {"tolerance_weeks": 10}   # US 메가캡 깊은조정 보정(backtest_leader_cycle_us)

# 결과 dict에서 관측에 남길 핵심 필드만 추출
_KEEP = ("ticker", "name", "signal", "clock", "age_months", "survival_pct",
         "cycle_start", "delta_value", "delta_source", "confidence",
         "trend_intact", "mdd_from_high", "data_available", "error")


def _load_watchlist() -> tuple[list[dict], list[dict]]:
    """global_leaders.yaml → (US 대장주, KR 종목) 유니크 리스트(메타 포함)."""
    d = yaml.safe_load(LEADERS_YAML.read_text(encoding="utf-8"))
    us: dict[str, dict] = {}
    kr: dict[str, dict] = {}
    for sec, body in (d.get("sectors") or {}).items():
        for gl in body.get("global_leaders", []) or []:
            tk = gl.get("ticker")
            if tk and tk not in us:
                us[tk] = {"ticker": tk, "name": gl.get("name", ""), "sector": sec}
            for ks in gl.get("kr_stocks", []) or []:
                code = ks.get("ticker")
                if code and code not in kr:
                    kr[code] = {"ticker": code, "name": ks.get("name", ""), "sector": sec,
                                "revenue_dependency": ks.get("revenue_dependency", "")}
    return list(us.values()), list(kr.values())


def _slim(res: dict, meta: dict) -> dict:
    out = {k: res.get(k) for k in _KEEP if k in res}
    out["market"] = res.get("market")
    out["sector"] = meta.get("sector", "")
    if meta.get("revenue_dependency"):
        out["revenue_dependency"] = meta["revenue_dependency"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="주도주 사이클 shadow 관측 러너")
    ap.add_argument("--quiet", action="store_true", help="요약 출력 생략")
    args = ap.parse_args()

    us_list, kr_list = _load_watchlist()
    leaders: list[dict] = []

    # US 대장주 (market=US, tol=10)
    for m in us_list:
        try:
            res = load_and_diagnose(m["ticker"], market="US", as_of=None,
                                    use_delta=True, params=US_TOL)
        except Exception as e:  # noqa: BLE001
            res = {"ticker": m["ticker"], "market": "US", "data_available": False,
                   "error": f"{type(e).__name__}: {e}"}
        leaders.append(_slim(res, m))

    # KR 종목 (market=KR, tol=6 기본, DART 델타)
    for m in kr_list:
        try:
            res = load_and_diagnose(m["ticker"], market="KR", as_of=None,
                                    use_delta=True, params=None)
        except Exception as e:  # noqa: BLE001
            res = {"ticker": m["ticker"], "market": "KR", "data_available": False,
                   "error": f"{type(e).__name__}: {e}"}
        leaders.append(_slim(res, m))

    # 집계
    by_signal: dict[str, int] = {}
    for r in leaders:
        sig = r.get("signal") or ("데이터없음" if not r.get("data_available") else "?")
        by_signal[sig] = by_signal.get(sig, 0) + 1

    from datetime import datetime
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "config/global_leaders.yaml",
        "params": {"kr_tolerance_weeks": 6, "us_tolerance_weeks": 10},
        "note": "관측 전용(shadow) — 매매 미반영. 신호는 별도 펀더멘털/수급 확인 필요(생존편향).",
        "summary": {"total": len(leaders), "us": len(us_list), "kr": len(kr_list),
                    "by_signal": by_signal},
        "leaders": leaders,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"=== 주도주 사이클 shadow 관측 ({len(leaders)}종목) → {OUT.relative_to(PROJECT_ROOT)} ===")
        order = ["청산", "경계", "매수적기", "보유", "해당없음", "데이터없음", "?"]
        print("  신호분포:", " · ".join(f"{k} {by_signal[k]}" for k in order if k in by_signal))
        # 주목 신호(청산/경계/매수적기)만 사이클 활성 종목 노출
        icon = {"매수적기": "🟢", "보유": "🔵", "경계": "🟡", "청산": "🔴"}
        for sig in ("청산", "경계", "매수적기"):
            picks = [r for r in leaders if r.get("signal") == sig and r.get("age_months") is not None]
            if not picks:
                continue
            picks.sort(key=lambda r: r.get("age_months") or 0, reverse=(sig != "매수적기"))
            print(f"\n  {icon[sig]} {sig} ({len(picks)}):")
            for r in picks:
                d = r.get("delta_value")
                ds = f"Δ{d:+.0f}" if d is not None else "Δ-"
                print(f"    [{r['market']}] {r['ticker']:7}{r.get('name',''):12} "
                      f"{r.get('clock',''):4} age={r.get('age_months'):>5.1f}m {ds:>7} "
                      f"({r.get('sector','')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
