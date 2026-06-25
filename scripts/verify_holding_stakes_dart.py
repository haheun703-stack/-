"""scripts/verify_holding_stakes_dart.py — holding_nav.yaml 지분율을 DART 실측과 대조.

DART '타법인 출자현황'(otrCprInvstmntSttus)으로 각 지주사의 상장 자회사 기말 지분율을
가져와, config/holding_nav.yaml의 (추정) 지분율과 대조한다. 출자 법인명(inv_prm, 한글음차)
↔ corp_codes.csv(영문약자)는 별칭 정규화로 종목코드 매칭한다.

실행:  python -u -X utf8 scripts/verify_holding_stakes_dart.py
       python -u -X utf8 scripts/verify_holding_stakes_dart.py --update   # YAML 자동 갱신
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import yaml  # noqa: E402

from src.adapters.dart_adapter import DartAdapter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(ROOT, "config", "holding_nav.yaml")
CORP_CSV = os.path.join(ROOT, "data", "dart_cache", "corp_codes.csv")

# 한글음차 → 영문약자 (corp_codes.csv는 영문, DART inv_prm은 한글음차)
ALIASES = {
    "엘지": "lg", "에스케이": "sk", "씨제이": "cj", "지에스": "gs",
    "에이치디현대": "hd현대", "에이치디": "hd", "케이티앤지": "kt&g", "케이티": "kt",
    "에스디아이": "sdi", "에스디에스": "sds", "이엔에이": "e&a", "이앤에이": "e&a",
    "이엔엠": "enm", "생명보험": "생명",
}
_DROP = ["(주)", "㈜", "주식회사", "(유)", "(株)", " ", ".", ",", "(", ")",
         "co", "ltd", "inc", "corporation", "gmbh", "limited", "lp", "llc"]


def norm(s: str) -> str:
    s = (s or "").lower()
    for k, v in ALIASES.items():
        s = s.replace(k, v)
    for t in _DROP:
        s = s.replace(t.lower(), "")
    return s


def build_name2stock() -> dict[str, str]:
    m: dict[str, str] = {}
    with open(CORP_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sc = str(r.get("stock_code", "")).strip()
            if sc and sc.lower() != "nan" and len(sc) >= 5:
                m[norm(r.get("corp_name", ""))] = sc.zfill(6)
    return m


def main() -> None:
    update = "--update" in sys.argv
    with open(YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name2stock = build_name2stock()
    dart = DartAdapter()
    if not dart.is_available:
        print("DART_API_KEY 미설정 — 중단")
        return

    print("=" * 78)
    print("holding_nav.yaml 지분율  vs  DART 실측 (타법인 출자현황)")
    print("=" * 78)

    total_diff, total_n = 0.0, 0
    for tk, hd in cfg["holdings"].items():
        inv = dart.fetch_other_corp_investments(tk)
        # DART 상장 자회사 지분율 맵: stock_code -> pct
        dart_map: dict[str, float] = {}
        for x in inv:
            if x["stake_pct"] is None:
                continue
            sc = name2stock.get(norm(x["inv_name"]))
            if sc:
                dart_map[sc] = x["stake_pct"]
        print(f"\n■ {hd['name']} ({tk})  DART 출자 {len(inv)}건 / 상장매칭 {len(dart_map)}건")
        for s in hd.get("listed_stakes", []):
            sc = str(s["ticker"]).zfill(6)
            ours = float(s["pct"])
            dval = dart_map.get(sc)
            if dval is None:
                print(f"    {s['name']:14s} {sc} | 우리 {ours:6.2f}% | DART 매칭실패 ⚠️(표기/비상장 분할)")
                continue
            diff = dval - ours
            total_diff += abs(diff)
            total_n += 1
            flag = "  ✅" if abs(diff) < 1.0 else ("  ⚠️" if abs(diff) < 5 else "  ❗")
            print(f"    {s['name']:14s} {sc} | 우리 {ours:6.2f}% | DART {dval:6.2f}% | Δ{diff:+6.2f}{flag}")
            if update:
                s["pct"] = round(dval, 2)

    print("\n" + "-" * 78)
    if total_n:
        print(f"매칭 {total_n}건 평균 절대오차 {total_diff / total_n:.2f}%p")
    if update:
        with open(YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print("→ holding_nav.yaml 지분율을 DART 실측으로 갱신 완료")
    else:
        print("※ --update 플래그로 YAML 지분율을 DART 실측으로 갱신 가능")
    print("=" * 78)


if __name__ == "__main__":
    main()
