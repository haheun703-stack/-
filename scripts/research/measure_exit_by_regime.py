# -*- coding: utf-8 -*-
"""레짐 조건부 청산규칙 — 진입시점 KOSPI 200일선 위/아래로 분할
※ 레짐 라벨은 진입시점 정보만 사용(미래정보 없음)
"""
import numpy as np, pandas as pd, glob

k=pd.read_csv("/home/ubuntu/quantum-master/data/kospi_index.csv",parse_dates=["Date"]).set_index("Date")
k["ma200"]=k["close"].rolling(200).mean()
k["bull"]=(k["close"]>k["ma200"])
k["ret120"]=k["close"].pct_change(120)
bull=k["bull"].to_dict(); 
FILES=sorted(glob.glob("/home/ubuntu/quantum-master/data/processed/*.parquet"))[::4]
STEP=3; COST=0.0023
CASES=[(15,5,60),(15,5,120),(30,10,120)]

for W,L,H in CASES:
    acc={True:{"A":[],"B":[],"C":[],"D":[]},False:{"A":[],"B":[],"C":[],"D":[]}}
    for f in FILES:
        try: df=pd.read_parquet(f,columns=["high","low","close"])
        except Exception: continue
        h=df["high"].to_numpy(float); lo=df["low"].to_numpy(float); c=df["close"].to_numpy(float)
        idx=df.index; n=len(c)
        if n<H+30: continue
        for i in range(20,n-H,STEP):
            e=c[i]
            if not (e>0): continue
            rg=bull.get(pd.Timestamp(idx[i]))
            if rg is None or (isinstance(rg,float) and np.isnan(rg)): continue
            rg=bool(rg)
            up=e*(1+W/100.); dn=e*(1-L/100.)
            hs=h[i+1:i+1+H]; ls=lo[i+1:i+1+H]; cs=c[i+1:i+1+H]
            iu=int(np.argmax(hs>=up)) if (hs>=up).any() else -1
            il=int(np.argmax(ls<=dn)) if (ls<=dn).any() else -1
            a=acc[rg]
            a["A"].append(cs[-1]/e-1.0)
            a["B"].append(-L/100. if il>=0 else cs[-1]/e-1.0)
            a["C"].append(W/100. if iu>=0 else cs[-1]/e-1.0)
            if iu>=0 and (il<0 or iu<il): a["D"].append(W/100.)
            elif il>=0: a["D"].append(-L/100.)
            else: a["D"].append(cs[-1]/e-1.0)
    print(f"\n━━━━ W+{W} / L-{L} / H{H}")
    for rg,name in ((True,"BULL (KOSPI>MA200)"),(False,"BEAR (KOSPI<MA200)")):
        a=acc[rg]
        if not a["A"]: print(f"  {name}: 표본 0"); continue
        print(f"  [{name}]  n={len(a['A']):,}")
        base=np.array(a["A"]); bm=base.mean(); bs=base.std()
        for tag in ("A","B","C","D"):
            X=np.array(a[tag]); mu=X.mean(); sd=X.std()
            lbl={"A":"단순보유","B":"손절만","C":"익절만","D":"삼중배리어"}[tag]
            d="" if tag=="A" else f" | 기저선차 {(mu-bm)*100:+7.3f}%p"
            print(f"    {lbl:10s} 평균 {mu*100:+8.3f}%  σ {sd*100:6.2f}%  "
                  f"mu/σ {mu/sd if sd else 0:+.4f}{d}")
