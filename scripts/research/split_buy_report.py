"""분할 매수표 SHOW ME 리포트 — B/C equity·drawdown·현재위치·V자 급락 시각화.

사장님 지시(2026-06-03): 숫자만 남기지 말고 그림으로 박아 다음 세션이 눈으로 판단.
- B 일괄+C60 vs C 조정분할+C60 equity / drawdown (강세장 / 2022 약세장 분리)
- 현재 위치(현재가·60선·괴리율·C 분할 예정가) / V자 급락(2026-03-31) 취약 구간

출력: docs/02-design/assets/ (data/는 gitignore라 docs에 둠) — PNG + 숫자표 stdout.
read-only / 실주문 0. 사용: python -u -X utf8 scripts/research/split_buy_report.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import FinanceDataReader as fdr

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from scripts.research.split_buy_backtest import prepare, sim  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent.parent / "docs" / "02-design" / "assets"
BULL = ("2025-06-01", "2026-06-02")
BEAR = ("2022-01-01", "2022-12-31")
STRAT_COLOR = {"A": "#999999", "B": "#1f77b4", "C": "#d62728"}
STRAT_LABEL = {"A": "A 일괄buyhold", "B": "B 일괄+C60", "C": "C 조정분할+C60"}
VZONE = pd.Timestamp("2026-03-31")  # 강세장 중 60선 1일 이탈 = -45% 바닥


def _curves(lev_price, bull_sig, strat, start, end):
    r = sim(lev_price, bull_sig, strat, start, end, return_curves=True)
    eq = pd.Series(r["equity"], index=pd.to_datetime(r["dates"]))
    dd = (eq / eq.cummax() - 1.0) * 100
    return eq, dd, r


def fig_backtest():
    """2x2: equity(강세/약세) + drawdown(강세/약세). B vs C 핵심, A는 기준선."""
    lev_s, bull_s = prepare("005930", 2.0)
    s_b, e_b = pd.Timestamp(BULL[0]), pd.Timestamp(BULL[1])
    s_r, e_r = pd.Timestamp(BEAR[0]), pd.Timestamp(BEAR[1])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("분할 매수표 백테스트 — 삼성×2 합성 (B 일괄+C60 vs C 조정분할+C60)", fontsize=14, fontweight="bold")

    # (0,0) 강세장 equity (log)
    for st in ("A", "B", "C"):
        eq, _, r = _curves(lev_s, bull_s, st, s_b, e_b)
        ax[0, 0].plot(eq.index, eq.values, color=STRAT_COLOR[st],
                      label=f"{STRAT_LABEL[st]} ({r['final_return_pct']:+.0f}%)",
                      lw=1.8 if st != "A" else 1.0, alpha=0.9 if st != "A" else 0.6)
    ax[0, 0].axvline(VZONE, color="purple", ls="--", lw=1, alpha=0.7)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("강세장 25.6~26.5 — equity (로그)  ※보라선=2026-03-31 V자")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # (0,1) 약세장 equity
    for st in ("A", "B", "C"):
        eq, _, r = _curves(lev_s, bull_s, st, s_r, e_r)
        ax[0, 1].plot(eq.index, eq.values, color=STRAT_COLOR[st],
                      label=f"{STRAT_LABEL[st]} ({r['final_return_pct']:+.0f}%)",
                      lw=1.8 if st != "A" else 1.0, alpha=0.9 if st != "A" else 0.6)
    ax[0, 1].set_title("2022 약세장 (실데이터) — equity  ★C가 최고 방어")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # (1,0) 강세장 drawdown B vs C
    for st in ("B", "C"):
        _, dd, _ = _curves(lev_s, bull_s, st, s_b, e_b)
        ax[1, 0].fill_between(dd.index, dd.values, 0, color=STRAT_COLOR[st], alpha=0.25)
        ax[1, 0].plot(dd.index, dd.values, color=STRAT_COLOR[st], lw=1.5, label=STRAT_LABEL[st])
    ax[1, 0].axvline(VZONE, color="purple", ls="--", lw=1, alpha=0.7)
    ax[1, 0].annotate("V자: B·C 모두 -45% 동일\n(C60 바닥청산, 분할도 풀투입)",
                      xy=(VZONE, -45), xytext=(0.05, 0.15), textcoords="axes fraction",
                      fontsize=8, color="purple", arrowprops=dict(arrowstyle="->", color="purple"))
    ax[1, 0].set_title("강세장 drawdown(%) — 하루 급락엔 분할/C60 무력")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (1,1) 약세장 drawdown B vs C
    for st in ("B", "C"):
        _, dd, _ = _curves(lev_s, bull_s, st, s_r, e_r)
        ax[1, 1].fill_between(dd.index, dd.values, 0, color=STRAT_COLOR[st], alpha=0.25)
        ax[1, 1].plot(dd.index, dd.values, color=STRAT_COLOR[st], lw=1.5, label=STRAT_LABEL[st])
    ax[1, 1].set_title("2022 약세장 drawdown(%) — 추세하락엔 C가 -9.8%로 방어")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS / "split_buy_backtest.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_position():
    """1x2: 현재 위치(삼성/488080 가격 vs 60선 + C 분할 예정가) / V자 확대."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("현재 위치 & V자 급락 취약 구간 (실주문 0 / HOLD)", fontsize=14, fontweight="bold")

    pos_txt = []
    for i, (tic, name, col) in enumerate(
        [("005930", "삼성전자", "#d62728"), ("488080", "반도체레버", "#1f77b4")]
    ):
        df = fdr.DataReader(tic, "2025-01-01", "2026-06-02")
        close = df["Close"].astype(float)
        ma60 = close.rolling(60).mean()
        cur = float(close.iloc[-1]); m60 = float(ma60.iloc[-1])
        gap = (cur / m60 - 1.0) * 100
        ax[0].plot(close.index, close.values, color=col, lw=1.4, label=f"{name} 종가")
        ax[0].plot(ma60.index, ma60.values, color=col, lw=1.0, ls="--", alpha=0.6)
        # C 분할 예정가 = 현재가 기준 -5/-10/-15% (삼성만 표시, 가독성)
        if tic == "005930":
            for d in (0.05, 0.10, 0.15):
                ax[0].axhline(cur * (1 - d), color=col, lw=0.6, ls=":", alpha=0.4)
        pos_txt.append(f"{name}: 현재 {cur:,.0f} / 60선 {m60:,.0f} / 괴리 +{gap:.0f}%")
    ax[0].set_title("삼성·488080 종가 vs 60일선 (점선)  ※현재 60선 +57~62% 과열")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].text(0.02, 0.02, "\n".join(pos_txt) + "\nC 분할예정: 직전고점 -5/-10/-15%",
               transform=ax[0].transAxes, fontsize=8, va="bottom",
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # V자 확대: 삼성 2026-02~04 가격 vs 60선
    df = fdr.DataReader("005930", "2026-01-15", "2026-05-15")
    close = df["Close"].astype(float)
    ma60 = close.rolling(60).mean()
    full = fdr.DataReader("005930", "2025-08-01", "2026-05-15")["Close"].astype(float)
    ma60f = full.rolling(60).mean().reindex(close.index)
    ax[1].plot(close.index, close.values, color="#d62728", lw=1.6, label="삼성 종가")
    ax[1].plot(ma60f.index, ma60f.values, color="black", lw=1.2, ls="--", label="60일선")
    ax[1].axvline(VZONE, color="purple", ls="--", lw=1.2)
    ax[1].annotate("2026-03-31\n60선 1일 이탈 = -45% 바닥\n→C60 청산→다음날 V자 반등 놓침",
                   xy=(VZONE, float(close.loc[:VZONE].iloc[-1])), xytext=(0.30, 0.12),
                   textcoords="axes fraction", fontsize=8, color="purple",
                   arrowprops=dict(arrowstyle="->", color="purple"))
    ax[1].set_title("V자 급락 취약 — 하루 급락은 C60/분할이 못 막음 (buyhold 회복)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ASSETS / "split_buy_position.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out, pos_txt


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("분할 매수표 SHOW ME 리포트 생성")
    print("=" * 60)
    p1 = fig_backtest()
    p2, pos_txt = fig_position()
    print("\n[현재 위치]")
    for t in pos_txt:
        print("  " + t)
    print("\n[생성 PNG]")
    print("  " + str(p1.relative_to(ASSETS.parent.parent.parent)))
    print("  " + str(p2.relative_to(ASSETS.parent.parent.parent)))
    print("\n실주문 0 / HOLD / read-only")


if __name__ == "__main__":
    main()
