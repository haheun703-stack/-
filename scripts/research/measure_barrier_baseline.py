# -*- coding: utf-8 -*-
"""삼중배리어 도달확률 전수 측정 — 한국시장 기저선
주장: -5% 손절 / +15% 익절 / 10개 중 3개면 이득
검증: 무작위 진입의 실제 도달확률 p 와 기대값 E
"""
import numpy as np, pandas as pd, glob, sys, json

FILES = sorted(glob.glob("/home/ubuntu/quantum-master/data/processed/*.parquet"))
STEP = 2                      # 진입일 샘플 간격(거래일)
GRIDS = [(15,5),(10,5),(20,5),(15,7.5),(9,3),(30,10),(6,2)]
HORIZONS = [20,60,120]
COST = 0.0023                 # 왕복 거래비용(세금0.18%+수수료) 근사

def run(files, tag):
    out = {}
    for W,L in GRIDS:
        for H in HORIZONS:
            win=loss=timeout=0; rets=[]; tt_win=[]; tt_loss=[]
            for f in files:
                try:
                    df = pd.read_parquet(f, columns=["high","low","close"])
                except Exception:
                    continue
                h=df["high"].to_numpy(float); lo=df["low"].to_numpy(float); c=df["close"].to_numpy(float)
                n=len(c)
                if n < H+30: continue
                for i in range(20, n-H, STEP):
                    e=c[i]
                    if not (e>0): continue
                    up=e*(1+W/100.0); dn=e*(1-L/100.0)
                    hs=h[i+1:i+1+H]; ls=lo[i+1:i+1+H]
                    iu=np.argmax(hs>=up) if (hs>=up).any() else -1
                    il=np.argmax(ls<=dn) if (ls<=dn).any() else -1
                    if iu>=0 and (il<0 or iu<il):
                        win+=1; rets.append(W/100.0); tt_win.append(iu+1)
                    elif il>=0:
                        loss+=1; rets.append(-L/100.0); tt_loss.append(il+1)
                        # 같은 날 양쪽 터치는 손절 우선(보수적)
                    else:
                        timeout+=1; rets.append(c[i+H]/e-1.0)
            tot=win+loss+timeout
            if not tot: continue
            r=np.array(rets)
            p_up = win/tot
            p_theo = L/(W+L)          # 랜덤워크 이론 도달확률
            out[f"W{W}_L{L}_H{H}"] = dict(
                n=tot, win=win, loss=loss, timeout=timeout,
                p_win=round(p_up,4), p_theory=round(p_theo,4),
                mean_ret=round(float(r.mean()),5),
                mean_ret_after_cost=round(float(r.mean()-COST),5),
                med_ret=round(float(np.median(r)),5),
                std=round(float(r.std()),5),
                days_win=round(float(np.mean(tt_win)),1) if tt_win else None,
                days_loss=round(float(np.mean(tt_loss)),1) if tt_loss else None,
            )
            print(f"[{tag}] W+{W}/L-{L}/H{H}: n={tot} p_win={p_up:.4f} (이론 {p_theo:.4f}) "
                  f"평균 {r.mean()*100:+.3f}% 비용후 {(r.mean()-COST)*100:+.3f}% "
                  f"승일수 {np.mean(tt_win) if tt_win else 0:.1f} 패일수 {np.mean(tt_loss) if tt_loss else 0:.1f}", flush=True)
    return out

sub = FILES[::3]     # 1/3 표본으로 격자 전체
print(f"대상 parquet {len(FILES)}개 중 {len(sub)}개 사용 (격자 {len(GRIDS)}×{len(HORIZONS)})", flush=True)
res = run(sub, "KR")
json.dump(res, open("/tmp/barrier_result.json","w"), ensure_ascii=False, indent=1)
print("\n저장: /tmp/barrier_result.json")
