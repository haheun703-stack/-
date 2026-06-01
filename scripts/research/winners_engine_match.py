"""오늘(6/1) +10% 종목 × 수급 × 우리 엔진(5/29 지표) 대조. 사장님 6/1 장마감 지시.

KIS 등락률순위(market_ranking_20260601.json)에서 +10% 추출 →
  ① 수급: 외인/기관 순매수(억) 매칭
  ② 우리 엔진이 미리 잡았나: processed parquet 5/29 마지막행 지표
     - supply_divergence(외인매도+기관매수) / 1개월 모멘텀 / 거래량비(vol/ma20)
     - 추세(close>ma20) / 박스돌파 임박(close/high_20) / 저점높이기
  ③ picks_history 최근 등장 여부
판정: 우리 엔진이 폭등 전 신호를 줬는가(포착) / 놓쳤나.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RANK = json.load(open(PROJECT_ROOT / "data" / "market_ranking_20260601.json", encoding="utf-8"))
# 상한가(아까 stdout) 보강
LIMIT_UP = {"LG헬로비전", "크레오에스지", "LG전자", "LG전자우", "로보스타",
            "한국비티비", "팸텍", "크라우드웍스", "오브젠", "두산로보틱스"}


def supply_map():
    m = {}
    fi = RANK["foreign_institution"]
    for x in fi["foreign_buy"] + fi["foreign_sell"]:
        m.setdefault(x["code"], {})["frgn"] = x.get("frgn_amt_억")
    for x in fi["inst_buy"] + fi["inst_sell"]:
        m.setdefault(x["code"], {})["inst"] = x.get("inst_amt_억")
    return m


def engine_signal(code):
    """5/29까지 processed 지표 (look-ahead 0, 6/1 폭등 전 상태)."""
    f = PROJECT_ROOT / "data" / "processed" / f"{code}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f).sort_index()
        df = df[df["close"] > 0]
    except Exception:
        return None
    if len(df) < 25:
        return None
    last = df.iloc[-1]
    c = df["close"]
    mom20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else None
    ma20 = c.rolling(20).mean().iloc[-1]
    volr = (last["volume"] / last["volume_ma20"]) if last.get("volume_ma20", 0) else None
    h20 = last.get("high_20")
    return dict(
        date=str(df.index[-1])[:10],
        supply_div=int(last.get("supply_divergence", 0)),
        mom20=mom20,
        volr=volr,
        trend_up=bool(c.iloc[-1] > ma20) if not pd.isna(ma20) else None,
        near_break=(c.iloc[-1] / h20) if h20 else None,
        hl5=int(last.get("higher_low_5d", 0)) if "higher_low_5d" in df.columns else None,
    )


def picks_recent(code, name):
    try:
        ph = json.load(open(PROJECT_ROOT / "data" / "picks_history.json", encoding="utf-8"))
    except Exception:
        return False
    s = json.dumps(ph, ensure_ascii=False)
    return code in s or (name and name in s)


def main() -> int:
    fl = RANK["fluctuation_rank"]
    sup = supply_map()
    # +10% 종목 (등락률순) — 데이터 이상치(음수) 제외
    ups = sorted([x for x in fl if x.get("change_pct", 0) >= 10.0],
                 key=lambda z: z["change_pct"], reverse=True)
    codes_seen = {x["code"] for x in ups}

    print(f"=== 오늘(6/1) +10% 종목 — KIS 등락률순위 기준 {len(ups)}건 (상위30 한계, 저녁 KRX확정 보강필요) ===\n")
    print(f'{"종목":<16}{"등락률":>7}{"외인억":>8}{"기관억":>8} | 우리엔진(5/29)')
    hit = 0
    for x in ups:
        code, name, cp = x["code"], x["name"], x["change_pct"]
        sg = engine_signal(code)
        sm = sup.get(code, {})
        fr = sm.get("frgn"); ins = sm.get("inst")
        frs = f'{fr:>+7.0f}' if isinstance(fr, (int, float)) else f'{"-":>7}'
        ins_s = f'{ins:>+7.0f}' if isinstance(ins, (int, float)) else f'{"-":>7}'
        if sg:
            tags = []
            if sg["supply_div"] > 0: tags.append("수급다이버전스")
            if sg["mom20"] is not None and sg["mom20"] > 0.10: tags.append(f"모멘텀+{sg['mom20']*100:.0f}%")
            if sg["volr"] and sg["volr"] >= 1.5: tags.append(f"거래량x{sg['volr']:.1f}")
            if sg["near_break"] and sg["near_break"] >= 0.97: tags.append("박스돌파임박")
            if sg["trend_up"]: tags.append("추세상승")
            eng = "  ".join(tags) if tags else "신호없음"
            if tags: hit += 1
        else:
            eng = "데이터없음(신규/소형)"
        pk = " ★picks기록" if picks_recent(code, name) else ""
        lim = "🔥상한" if name in LIMIT_UP else ""
        print(f'{name[:15]:<16}{cp:>+6.1f}%{frs}{ins_s} | {eng}{pk} {lim}')

    print(f"\n=== 상한가(아까 KIS stdout) ===\n  {', '.join(LIMIT_UP)}")
    print(f"\n우리 엔진이 5/29 시점에 '미리 신호 준' 종목: {hit}/{len(ups)}")
    print("★ hit 높으면 = 엔진이 폭등 전 포착. 0~낮으면 = 폭등을 미리 못 봄(5/30 조기발굴 연구 과제).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
