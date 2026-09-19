# -*- coding: utf-8 -*-
"""α_barrier — 우리 픽이 배리어 기저선을 이기는가 (2026-09-19)

  α_barrier = p_실측(픽) − p_기저선(W,L,H)

기저선은 config/barrier_baseline_kr.json (전수 16~18만 진입 실측).
픽은 forward_paper 원장의 BUY FILL — 실제로 집행된 진입만 본다.
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path("/home/ubuntu/quantum-master")
BASE = json.loads((ROOT / "config/barrier_baseline_kr.json").read_text(encoding="utf-8"))
COST = 0.0023

# ── 진입 수집 ──
buys = []
for line in (ROOT / "data/forward_paper/events.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        p = json.loads(line).get("payload", {})
    except Exception:
        continue
    if p.get("event_type") != "FILL" or p.get("side") != "BUY":
        continue
    if not p.get("ticker") or not p.get("event_at"):
        continue
    buys.append({"ticker": str(p["ticker"]).zfill(6),
                 "date": str(p["event_at"])[:10],
                 "strategy": p.get("strategy_id") or "?",
                 "grade": (p.get("metadata") or {}).get("grade")})
print(f"BUY FILL {len(buys)}건 · 종목 {len({b['ticker'] for b in buys})} · "
      f"기간 {min(b['date'] for b in buys)} ~ {max(b['date'] for b in buys)}")

# ── 가격 캐시 ──
cache = {}
def px(t):
    if t not in cache:
        f = ROOT / f"data/processed/{t}.parquet"
        if not f.exists():
            cache[t] = None
        else:
            try:
                d = pd.read_parquet(f, columns=["high", "low", "close"])
                d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
                cache[t] = d
            except Exception:
                cache[t] = None
    return cache[t]

def judge(t, day, W, L, H):
    d = px(t)
    if d is None or day not in d.index:
        return None
    i = list(d.index).index(day)
    if i + H >= len(d):
        return None                       # 관측창 미완 — 판정하지 않는다
    e = float(d["close"].iloc[i])
    if not (e > 0):
        return None
    hs = d["high"].iloc[i+1:i+1+H].to_numpy(float)
    ls = d["low"].iloc[i+1:i+1+H].to_numpy(float)
    up, dn = e*(1+W/100), e*(1-L/100)
    iu = int(np.argmax(hs >= up)) if (hs >= up).any() else -1
    il = int(np.argmax(ls <= dn)) if (ls <= dn).any() else -1
    if iu >= 0 and (il < 0 or iu < il): return ("win", W/100)
    if il >= 0: return ("loss", -L/100)
    return ("timeout", float(d["close"].iloc[i+H])/e - 1)

for W, L, H in ((15,5,60),(15,5,120),(20,5,120),(10,5,60)):
    key = f"W{W}_L{L}_H{H}"
    if key not in BASE: continue
    p_base = BASE[key]["p_win"]; r_base = BASE[key]["mean_ret"]
    tot = defaultdict(lambda: {"win":0,"n":0,"rets":[]})
    for b in buys:
        r = judge(b["ticker"], b["date"], W, L, H)
        if r is None: continue
        for k in ("__ALL__", b["strategy"]):
            tot[k]["n"] += 1
            tot[k]["win"] += (r[0] == "win")
            tot[k]["rets"].append(r[1])
    print(f"\n━━━━ +{W}/−{L}/H{H}  기저선 p={p_base:.4f} 평균 {r_base*100:+.3f}%")
    rows = sorted(tot.items(), key=lambda x: -x[1]["n"])
    for name, v in rows:
        if v["n"] < 20: continue
        p = v["win"]/v["n"]; mu = float(np.mean(v["rets"]))
        a_p = p - p_base; a_r = mu - r_base
        mark = "✅" if a_p > 0 else "🚨"
        print(f"  {mark} {name[:38]:38s} n={v['n']:4d} p={p:.4f} "
              f"α_p={a_p*100:+6.2f}%p  평균 {mu*100:+7.3f}% α_r={a_r*100:+6.2f}%p")
