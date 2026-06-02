"""① 극단신호 봇 탈출 — 적대적 재검증 (사장님 6/2, "봇=MDD 관리자" 결론 검증).

원본(extreme_crash_signal.py): 반도체레버 buyhold +1768%/-41%/샤프3.42, extreme_exit +1502%/-26%/3.76.
"봇=MDD 관리자" 메모리 확정 전 4가지 적대 검증 (6/1 모멘텀 착시 기각 교훈 적용):
 1. 전반/후반 분할 — 양쪽 모두 MDD 개선 유지? (전체로만 이기면 복리 신기루)
 2. MDD 분해 — 회피한 손실이 1~2일에 몰리면 = 운(표본빈약 착시)
 3. 임계 민감도 — VIX/SOXX/SPY 임계 흔들면 MDD 개선 사라지나
 4. 정적 de-leverage 벤치마크 ★핵심 — 봇 타이밍이 '멍청한 비중축소(w·lev 매일)'보다 나은가
    (de-leverage 전환비용 0 = 봇이 이기기 더 어려운 적대 조건)
look-ahead 0 동일. 봇 전환비용 0.1%. ★2025.6~2026.5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pykrx import stock

SWITCH = 0.001


def load():
    us = pd.read_parquet(PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet").sort_index()
    us["vix_z"] = us.get("vix_zscore", (us["vix_close"] - us["vix_close"].rolling(20).mean()) / us["vix_close"].rolling(20).std())
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")
    kdays = [d for d in k.index if S <= d <= E]
    lev = stock.get_market_ohlcv("20250601", "20260529", "488080")["종가"].astype(float)
    lev.index = pd.to_datetime(lev.index)
    lev_r = lev.pct_change().fillna(0)
    return us, k, kdays, lev_r, S, E


def extreme_map(us, kdays, vix_thr=2.0, spy_thr=-0.02, soxx_thr=-0.03):
    sig = pd.DataFrame(index=us.index)
    ext = (us["vix_z"] >= vix_thr) | (us["spy_ret_1d"] <= spy_thr) | (us["soxx_ret_1d"] <= soxx_thr)
    sdf = pd.DataFrame({"usdate": us.index, "extreme": ext.values}).sort_values("usdate")
    m = pd.merge_asof(pd.DataFrame({"kdate": kdays}), sdf, left_on="kdate", right_on="usdate", direction="backward")
    return m.set_index("kdate")["extreme"].fillna(False)


def run_pos(kdays, lev_r, inpos):
    v = 1.0; prev = None; curve = []
    for d in kdays:
        ip = bool(inpos[d])
        r = lev_r.get(d, 0) if ip else 0
        v *= (1 + r)
        if prev is not None and ip != prev:
            v *= (1 - SWITCH)
        curve.append(v); prev = ip
    return pd.Series(curve, index=kdays)


def run_weight(kdays, lev_r, w):
    v = 1.0; curve = []
    for d in kdays:
        v *= (1 + w * lev_r.get(d, 0)); curve.append(v)
    return pd.Series(curve, index=kdays)


def stats(eq):
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    dr = eq.pct_change().dropna(); sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return ret, mdd, sh


def main() -> int:
    us, k, kdays, lev_r, S, E = load()
    ext = extreme_map(us, kdays)
    allin = pd.Series(True, index=kdays)
    exitpos = ~ext  # 극단일 현금

    print("=" * 64)
    print("① 극단신호 봇 탈출 — 적대적 재검증 (반도체레버 488080)")
    print("=" * 64)

    # ── 검증 1: 전반/후반 분할 ──────────────────────────────
    print("\n[검증1] 전반/후반 분할 — 양쪽 모두 MDD 개선 유지하나?")
    half = len(kdays) // 2
    segs = [("전체", kdays), ("전반", kdays[:half]), ("후반", kdays[half:])]
    print(f'  {"구간":<6}{"buyhold(수익/MDD/샤프)":>30}{"extreme_exit":>30}')
    seg_verdict = []
    for nm, days in segs:
        d2 = [d for d in days]
        bh = stats(run_pos(d2, lev_r, allin.reindex(d2)))
        ee = stats(run_pos(d2, lev_r, exitpos.reindex(d2)))
        mdd_better = ee[1] > bh[1]  # MDD는 음수, 큰 값(0에 가까움)이 좋음
        seg_verdict.append((nm, mdd_better, ee[1] - bh[1], ee[0] - bh[0]))
        print(f'  {nm:<6}{bh[0]:>+9.0f}% {bh[1]:>5.0f}% {bh[2]:>5.2f}{ee[0]:>+13.0f}% {ee[1]:>5.0f}% {ee[2]:>5.2f}')
    print("  → 판정:")
    for nm, better, dmdd, dret in seg_verdict[1:]:
        tag = "MDD개선 유지" if better else "★MDD개선 실패(착시 의심)"
        print(f"     {nm}: MDD차 {dmdd:+.0f}%p / 수익차 {dret:+.0f}%p → {tag}")

    # ── 검증 2: MDD 분해 (회피 손실 집중도) ──────────────────
    print("\n[검증2] MDD 분해 — 봇이 회피한 손실이 며칠에 몰렸나? (1~2일 집중=운)")
    avoided = []
    for d in kdays:
        if ext[d]:  # 이날 현금 → lev 손실 회피(또는 이익 포기)
            avoided.append((d, lev_r.get(d, 0) * 100))
    avoided.sort(key=lambda x: x[1])  # 가장 큰 손실 회피 먼저
    tot_avoid_loss = sum(r for _, r in avoided if r < 0)
    print(f"  극단 신호로 포지션 아웃된 날: {len(avoided)}일")
    print(f"  그 중 lev 하락일(손실 회피): {sum(1 for _,r in avoided if r<0)}일 / 상승일(이익 포기): {sum(1 for _,r in avoided if r>0)}일")
    print(f"  회피한 총 하락폭 합계: {tot_avoid_loss:+.1f}%p")
    print("  큰 손실 회피 TOP5 (이 며칠이 MDD 개선의 대부분이면 = 운):")
    for d, r in avoided[:5]:
        share = r / tot_avoid_loss * 100 if tot_avoid_loss < 0 else 0
        print(f"     {d.date()}  lev {r:+6.2f}%  (회피손실 기여 {share:4.0f}%)")
    top2 = sum(r for _, r in avoided[:2] if r < 0)
    print(f"  ▶ 상위 2일이 전체 회피손실의 {top2/tot_avoid_loss*100:.0f}% 차지"
          + ("  → ★1~2일 운 의존(착시 위험)" if tot_avoid_loss < 0 and top2 / tot_avoid_loss > 0.6 else "  → 분산됨(체계적)"))

    # ── 검증 3: 임계 민감도 ─────────────────────────────────
    print("\n[검증3] 임계 민감도 — 신호 기준 흔들어도 MDD 개선 유지하나?")
    print(f'  {"VIX/SPY/SOXX 임계":<22}{"신호일":>6}{"수익":>9}{"MDD":>7}{"샤프":>6}')
    bh_full = stats(run_pos(kdays, lev_r, allin))
    print(f'  {"(buyhold 기준)":<22}{"-":>6}{bh_full[0]:>+8.0f}%{bh_full[1]:>6.0f}%{bh_full[2]:>6.2f}')
    grid = [
        (1.5, -0.015, -0.02), (2.0, -0.02, -0.03), (2.5, -0.025, -0.04),
        (1.5, -0.02, -0.03), (2.5, -0.02, -0.03),
    ]
    sens_mdd = []
    for vx, sp, so in grid:
        em = extreme_map(us, kdays, vx, sp, so)
        ee = stats(run_pos(kdays, lev_r, ~em))
        sens_mdd.append(ee[1])
        n = int(em.sum())
        print(f'  VIX{vx} SPY{sp*100:.0f}% SOXX{so*100:.0f}%{"":<2}{n:>6}{ee[0]:>+8.0f}%{ee[1]:>6.0f}%{ee[2]:>6.2f}')
    spread = max(sens_mdd) - min(sens_mdd)
    print(f"  ▶ MDD 범위 {min(sens_mdd):.0f}% ~ {max(sens_mdd):.0f}% (폭 {spread:.0f}%p)"
          + ("  → 안정적" if spread < 8 else "  → ★임계 민감(파라미터 운)"))

    # ── 검증 4: 정적 de-leverage 벤치마크 ★핵심 ─────────────
    print("\n[검증4] ★봇 vs 멍청한 비중축소(de-leverage, 신호 무관 매일 w·lev)")
    bh = stats(run_pos(kdays, lev_r, allin))
    ee = stats(run_pos(kdays, lev_r, exitpos))
    print(f'  {"전략":<28}{"수익":>9}{"MDD":>7}{"샤프":>6}')
    print(f'  {"buyhold(레버100%)":<28}{bh[0]:>+8.0f}%{bh[1]:>6.0f}%{bh[2]:>6.2f}')
    print(f'  {"★봇 extreme_exit":<28}{ee[0]:>+8.0f}%{ee[1]:>6.0f}%{ee[2]:>6.2f}')
    # 같은 MDD(-26%)를 만드는 정적 비중 w 탐색
    best_same_mdd = None
    for w in np.arange(0.50, 1.001, 0.01):
        s = stats(run_weight(kdays, lev_r, w))
        if best_same_mdd is None or abs(s[1] - ee[1]) < abs(best_same_mdd[1] - ee[1]):
            best_same_mdd = (w, s[1], s[0], s[2])
    w, wmdd, wret, wsh = best_same_mdd
    print(f'  {f"정적 비중 w={w:.2f} (MDD매칭)":<28}{wret:>+8.0f}%{wmdd:>6.0f}%{wsh:>6.2f}')
    print(f"  ▶ 같은 MDD({ee[1]:.0f}%)에서 — 봇 수익 {ee[0]:+.0f}% vs 멍청한 비중축소 {wret:+.0f}%")
    if wret > ee[0]:
        print(f"     → ★멍청한 비중축소가 봇보다 {wret-ee[0]:+.0f}%p 우위 = 봇 '타이밍' 무가치(착시)")
    else:
        print(f"     → 봇이 비중축소보다 {ee[0]-wret:+.0f}%p 우위 = 봇 타이밍에 실제 가치")
    # 같은 수익(+1502%)을 만드는 정적 비중에서 MDD 비교
    best_same_ret = None
    for w in np.arange(0.50, 1.001, 0.005):
        s = stats(run_weight(kdays, lev_r, w))
        if best_same_ret is None or abs(s[0] - ee[0]) < abs(best_same_ret[0] - ee[0]):
            best_same_ret = (w, s[1], s[0], s[2])
    w2, w2mdd, w2ret, w2sh = best_same_ret
    print(f"  ▶ 같은 수익(~{ee[0]:.0f}%)에서 — 봇 MDD {ee[1]:.0f}% vs 멍청한 비중축소 MDD {w2mdd:.0f}% (w={w2:.2f})")
    if w2mdd >= ee[1]:
        print(f"     → 멍청한 비중축소 MDD가 더 얕음 = 봇 무가치")
    else:
        print(f"     → 봇 MDD가 {ee[1]-w2mdd:+.0f}%p 더 얕음 = 봇 타이밍 가치")

    print("\n" + "=" * 64)
    print("종합: 4검증 중 하나라도 '착시' = 메모리 '확정' 보류, shadow 추적으로 forward 재검증")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
