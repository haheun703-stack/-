"""수급 알파 실현가능(포트폴리오) 백테스트 — 관문 #1 (5/31 코워크).

신호-수준 평균 초과수익(+3.56%p)이 '동시보유 제한 + 현금 재활용 + 슬리피지'를
반영한 실제 운용에서도 살아남는지 검증. paper 직전 최종 게이트.

룰 = scan_supply_alpha.py 동일:
  supply_divergence>0 + 거래대금≥10억 + KOSPI 과열회피(120일선+15%) + D+40 보유.

A. 신호수준(무제한 자본, 매 신호 독립, 0.2% 왕복) — '+3.56%p' 앵커/재현
B. 실현가능 포트폴리오(N슬롯, 현금 재활용, 슬리피지+세금) — 진짜 알파 가치
벤치마크 = KOSPI buy&hold 동일기간.

★ 생존자편향(data/processed=현재 생존종목, 상폐 미포함) 여전 → 절대수익은 상방편향.
  '시장 대비 초과'와 '슬롯 제약 시 잔존 여부'로 해석할 것.
"""
from __future__ import annotations

import glob
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
KOSPI = PROJECT_ROOT / "data" / "kospi_index.csv"

HOLD = 40
MIN_TV = 1e9
OVERHEAT_MULT = 1.15
START, END = pd.Timestamp("2023-06-01"), pd.Timestamp("2026-05-29")
CAP = 100_000_000

# 비용 모델
COST_RT = 0.002          # A: 왕복 0.2% (앵커 재현용)
BUY_SLIP = 0.0015        # B: 매수 슬리피지(편도)
SELL_SLIP = 0.0015       # B: 매도 슬리피지(편도)
SELL_TAX = 0.0018        # B: 매도 거래세(약 0.18%)


def load_kospi():
    k = pd.read_csv(KOSPI)
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    k = k.set_index("date").sort_index()
    k["ma120"] = k["close"].rolling(120).mean()
    k["overheat"] = k["close"] > k["ma120"] * OVERHEAT_MULT
    return k


def main() -> int:
    kospi = load_kospi()
    cal = list(kospi.index)
    cal_pos = {d: i for i, d in enumerate(cal)}
    overheat = {d: bool(v) for d, v in kospi["overheat"].items()}
    kclose = {d: float(v) for d, v in kospi["close"].items()}

    files = glob.glob(str(PROCESSED / "*.parquet"))
    sig_by_date = defaultdict(list)   # date -> [(code, supply_div, dual)]
    closes_raw, opens_raw = {}, {}
    n_sig = 0
    for f in files:
        code = Path(f).stem.zfill(6)
        try:
            df = pd.read_parquet(f).sort_index()
        except Exception:
            continue
        if "supply_divergence" not in df.columns or "trading_value" not in df.columns:
            continue
        df = df[(df.index >= START) & (df.index <= END)]
        if len(df) < 2:
            continue
        closes_raw[code] = df["close"]
        opens_raw[code] = df["open"]
        sd = df["supply_divergence"]
        tv = df["trading_value"]
        fc = df["foreign_consecutive_buy"] if "foreign_consecutive_buy" in df.columns else None
        ic = df["inst_consecutive_buy"] if "inst_consecutive_buy" in df.columns else None
        for t in df.index:
            if float(sd.get(t, 0) or 0) > 0 and float(tv.get(t, 0) or 0) >= MIN_TV:
                dual = (fc is not None and ic is not None
                        and int(fc.get(t, 0) or 0) >= 3 and int(ic.get(t, 0) or 0) >= 3)
                sig_by_date[t].append((code, float(sd.get(t, 0)), bool(dual)))
                n_sig += 1

    # 가격 매트릭스 (캘린더 정렬). close는 ffill(평가/청산), open은 raw(진입은 실제가만)
    C = pd.DataFrame(closes_raw).reindex(cal).ffill()
    O = pd.DataFrame(opens_raw).reindex(cal)
    print(f"신호 총 {n_sig:,}건 / 종목 {len(closes_raw)}개 / 캘린더 {len(cal)}일 "
          f"(과열일 {sum(overheat.values())}일)\n")

    # ---------- A. 신호수준 (무제한 자본) ----------
    exret, exc = [], []
    for d, lst in sig_by_date.items():
        if overheat.get(d, False):
            continue
        i = cal_pos.get(d)
        if i is None or i + 1 + HOLD >= len(cal):
            continue
        ed, xd = cal[i + 1], cal[i + 1 + HOLD]
        mkt = (kclose.get(xd, 0) / kclose.get(ed, 1) - 1) if kclose.get(ed) else 0.0
        for code, _sd, _dual in lst:
            eo = O.at[ed, code] if code in O.columns else np.nan
            xc = C.at[xd, code] if code in C.columns else np.nan
            if not eo or eo <= 0 or pd.isna(eo) or pd.isna(xc):
                continue
            r = xc / eo - 1 - COST_RT
            exret.append(r)
            exc.append(r - mkt)
    if exc:
        print(f"[A] 신호수준 (무제한자본·과열제외·D+40·왕복0.2%): "
              f"표본 {len(exc):,} / 평균 {np.mean(exret) * 100:+.2f}% / "
              f"시장초과 {np.mean(exc) * 100:+.2f}%p  ← '+3.56%p' 앵커")

    # ---------- B. 실현가능 포트폴리오 ----------
    idxs = [i for i, d in enumerate(cal) if START <= d <= END]
    kos_ret = kclose[cal[idxs[-1]]] / kclose[cal[idxs[0]]] - 1
    print(f"\n[B] 실현가능 포트폴리오 (현금재활용 + 슬리피지 {BUY_SLIP*100:.2f}%/편도 + 세금 {SELL_TAX*100:.2f}%):")
    print(f'{"슬롯":>4}{"최종수익":>10}{"KOSPI":>9}{"초과":>9}{"진입":>7}{"MaxDD":>8}{"평균보유":>9}')
    for N in (3, 5, 10, 20):
        cash = float(CAP)
        pos = {}            # code -> (shares, exit_idx, entry_idx)
        entries = 0
        hold_sum = 0
        peak, mdd, mv = CAP, 0.0, CAP
        for i in idxs:
            d = cal[i]
            # 청산: 보유 HOLD일 경과
            for code in list(pos):
                shares, xi, ei = pos[code]
                if i >= xi:
                    px = C.at[d, code] if code in C.columns else np.nan
                    if pd.isna(px):
                        continue
                    cash += shares * px * (1 - SELL_SLIP - SELL_TAX)
                    hold_sum += (i - ei)
                    del pos[code]
            # 진입: 전일 신호 + 과열 아님
            if i > 0:
                dp = cal[i - 1]
                if not overheat.get(dp, False):
                    cands = sorted(sig_by_date.get(dp, []), key=lambda x: (not x[2], -x[1]))
                    free = N - len(pos)
                    for code, _sd, _dual in cands:
                        if free <= 0:
                            break
                        if code in pos:
                            continue
                        eo = O.at[d, code] if code in O.columns else np.nan
                        if pd.isna(eo) or eo <= 0:
                            continue
                        alloc = cash / free
                        if alloc < 300_000:
                            continue
                        shares = int(alloc / (eo * (1 + BUY_SLIP)))
                        if shares <= 0:
                            continue
                        cash -= shares * eo * (1 + BUY_SLIP)
                        pos[code] = (shares, i + HOLD, i)
                        free -= 1
                        entries += 1
            # 평가
            held = 0.0
            for code, (shares, _xi, _ei) in pos.items():
                px = C.at[d, code] if code in C.columns else np.nan
                if not pd.isna(px):
                    held += shares * px
            mv = cash + held
            peak = max(peak, mv)
            mdd = max(mdd, (peak - mv) / peak)
        strat = mv / CAP - 1
        avg_hold = hold_sum / max(entries, 1)
        print(f'{N:>4}{strat*100:>+9.1f}%{kos_ret*100:>+8.1f}%'
              f'{(strat-kos_ret)*100:>+8.1f}%p{entries:>7}{mdd*100:>7.1f}%{avg_hold:>8.0f}일')

    print("\n★ 해석: B(실현가능) 초과가 +면 알파 잔존 / 0~음수면 신호수준 +3.56%p는 "
          "무제한자본 환상. 생존자편향 미보정이라 여전히 상방편향 가설.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
