"""scripts/scan_holding_nav.py — 지주사 NAV(순자산가치) 디스카운트 스캔 (POC).

config/holding_nav.yaml의 지주·관계사에 대해 현재 시총 기준 NAV·할인율을 계산해 출력한다.
시총 조달: data/universe.csv 우선 → 부족분 pykrx 최근 영업일 보강.
★관측 전용 — 실주문 0, 매수신호 아님(밸류트랩 회피 위해 핵심자회사 반등 동반은 백테스트가 판정).

실행: python -u -X utf8 scripts/scan_holding_nav.py
"""
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.use_cases.holding_nav import EOK, Holding, compute_nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE = os.path.join(ROOT, "data", "universe.csv")
YAML_PATH = os.path.join(ROOT, "config", "holding_nav.yaml")


def load_universe_caps() -> dict[str, float]:
    caps: dict[str, float] = {}
    if not os.path.exists(UNIVERSE):
        return caps
    with open(UNIVERSE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                caps[str(row["ticker"]).zfill(6)] = float(row["market_cap"])
            except (ValueError, KeyError):
                continue
    return caps


def pykrx_caps(tickers: set[str]) -> tuple[dict[str, float], str | None]:
    """pykrx 최근 영업일 시가총액(원). 장중 불안정 회피 위해 어제부터 뒤로 탐색."""
    out: dict[str, float] = {}
    asof: str | None = None
    try:
        from pykrx import stock
    except ImportError:
        print("  [경고] pykrx 미설치 — universe.csv만 사용")
        return out, None
    today = datetime.date.today()
    for back in range(1, 9):  # 어제부터(당일 장중 확정 전이라 제외)
        d = (today - datetime.timedelta(days=back)).strftime("%Y%m%d")
        try:
            df = stock.get_market_cap_by_ticker(d)
        except Exception:  # noqa: BLE001
            continue
        if df is not None and len(df) > 0:
            asof = d
            for tk in tickers:
                if tk in df.index:
                    out[tk] = float(df.loc[tk, "시가총액"])
            break
    return out, asof


def fmt_eok(won: float | None) -> str:
    if won is None:
        return "—"
    return f"{won / EOK:,.0f}억"


def fmt_jo(won: float | None) -> str:
    if won is None:
        return "—"
    return f"{won / 1e12:,.2f}조"


def main() -> None:
    with open(YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    holdings = [Holding.from_dict(tk, d) for tk, d in cfg.get("holdings", {}).items()]

    # 필요한 모든 티커(지주사 + 자회사) 시총 조달
    need: set[str] = set()
    for h in holdings:
        need.add(h.ticker)
        for s in h.listed_stakes:
            need.add(s.ticker)

    caps = load_universe_caps()
    miss = {tk for tk in need if tk not in caps}
    px, asof = pykrx_caps(miss) if miss else ({}, None)
    caps.update(px)

    print("=" * 64)
    print(f"지주사 NAV 디스카운트 스캔 (POC) — pykrx 보강일: {asof or 'N/A'}")
    print("=" * 64)

    for h in holdings:
        r = compute_nav(h, caps)
        print(f"\n■ {r.name} ({r.ticker})  [{r.kind}]  기준 {r.as_of}")
        print(f"  지주사 시총 : {fmt_jo(r.holding_market_cap)}")
        print(f"  ── NAV 구성 ──")
        for sv in r.stake_values:
            tag = " ⚠️시총결측" if sv.missing else ""
            w = (sv.stake_value / r.nav * 100) if r.nav > 0 else 0
            print(f"    {sv.name:12s} {sv.pct:5.2f}% × {fmt_jo(sv.sub_market_cap):>8s}"
                  f" = {fmt_eok(sv.stake_value):>10s}  (NAV {w:4.1f}%){tag}")
        if r.own_business:
            print(f"    {'자체사업':12s} {'':16s}   {fmt_eok(r.own_business):>10s}")
        if r.other_nav:
            print(f"    {'비상장':12s} {'':16s}   {fmt_eok(r.other_nav):>10s}")
        print(f"    {'(−)순부채':12s} {'':16s}   {fmt_eok(r.net_debt):>10s}")
        print(f"  ── 결과 ──")
        print(f"  NAV         : {fmt_jo(r.nav)}  ({fmt_eok(r.nav)})")
        dp = r.discount_pct
        if dp is not None:
            state = "할인거래" if dp < 0 else "할증거래"
            print(f"  할인율      : {dp:+.1f}%  [{state}]  (NAV/시총 {r.nav_per_market_cap}배)")
        else:
            print(f"  할인율      : 계산불가(NAV≤0 또는 시총결측)")
        if r.missing_subs:
            print(f"  ⚠️ 시총결측 자회사: {', '.join(r.missing_subs)} → NAV 과소평가(할인율 보수적)")

    print("\n" + "=" * 64)
    print("※ 관측 전용. 할인율 깊다=즉시 매수 아님. 역사적 밴드 하단+핵심자회사 반등")
    print("  동반 시에만 엣지(다음 단계: 백테스트). 지분율·순부채는 ⚠️검증 후 확정.")
    print("=" * 64)


if __name__ == "__main__":
    main()
