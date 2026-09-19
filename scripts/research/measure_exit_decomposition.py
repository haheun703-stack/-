# -*- coding: utf-8 -*-
"""손절·익절의 한계기여 분해 — 같은 진입점에서 4가지 청산규칙 비교
A 단순보유(H일)   B 손절만   C 익절만   D 삼중배리어
※ 동일 진입·동일 표본이라 차이는 청산규칙만의 효과
"""
import numpy as np, pandas as pd, glob, json

FILES = sorted(glob.glob("/home/ubuntu/quantum-master/data/processed/*.parquet"))[::3]
STEP=2; COST=0.0023
CASES=[(15,5,60),(15,5,120),(20,5,120),(30,10,120),(15,7.5,120)]

for W,L,H in CASES:
    A=[];B=[];C=[];D=[];dw=[];dl=[];bd=[];cd=[]
    for f in FILES:
        try: df=pd.read_parquet(f,columns=["high","low","close"])
        except Exception: continue
        h=df["high"].to_numpy(float); lo=df["low"].to_numpy(float); c=df["close"].to_numpy(float)
        n=len(c)
        if n<H+30: continue
        for i in range(20,n-H,STEP):
            e=c[i]
            if not (e>0): continue
            up=e*(1+W/100.); dn=e*(1-L/100.)
            hs=h[i+1:i+1+H]; ls=lo[i+1:i+1+H]; cs=c[i+1:i+1+H]
            iu=int(np.argmax(hs>=up)) if (hs>=up).any() else -1
            il=int(np.argmax(ls<=dn)) if (ls<=dn).any() else -1
            # A 단순보유
            A.append(cs[-1]/e-1.0)
            # B 손절만 (닿으면 -L, 아니면 만기종가)
            if il>=0: B.append(-L/100.); bd.append(il+1)
            else: B.append(cs[-1]/e-1.0); bd.append(H)
            # C 익절만
            if iu>=0: C.append(W/100.); cd.append(iu+1)
            else: C.append(cs[-1]/e-1.0); cd.append(H)
            # D 삼중배리어 (동시 터치는 손절 우선)
            if iu>=0 and (il<0 or iu<il): D.append(W/100.); dw.append(iu+1)
            elif il>=0: D.append(-L/100.); dl.append(il+1)
            else: D.append(cs[-1]/e-1.0)
    A=np.array(A);B=np.array(B);C=np.array(C);D=np.array(D)
    n=len(A)
    hold_D = (len(dw)*np.mean(dw) if dw else 0)+(len(dl)*np.mean(dl) if dl else 0)+((n-len(dw)-len(dl))*H)
    hold_D/= n
    print(f"\n━━ W+{W} / L-{L} / H{H}  (n={n:,})")
    for tag,X,hold in (("A 단순보유",A,H),("B 손절만",B,float(np.mean(bd))),
                       ("C 익절만",C,float(np.mean(cd))),("D 삼중배리어",D,hold_D)):
        mu=X.mean(); sd=X.std()
        print(f"  {tag:12s} 평균 {mu*100:+7.3f}%  비용후 {(mu-COST)*100:+7.3f}%  "
              f"표준편차 {sd*100:6.2f}%  평균보유 {hold:5.1f}일  "
              f"일당 {(mu-COST)/hold*100:+.4f}%  Sharpe유사 {mu/sd:+.4f}")
    print(f"  → 손절의 한계기여(B-A) {(B.mean()-A.mean())*100:+.3f}%p | "
          f"익절의 한계기여(C-A) {(C.mean()-A.mean())*100:+.3f}%p | "
          f"둘다(D-A) {(D.mean()-A.mean())*100:+.3f}%p")
