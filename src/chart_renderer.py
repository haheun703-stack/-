"""주식 차트 렌더러 — parquet → matplotlib → base64 PNG

Phase 4 DeepAnalyst에 차트 이미지를 첨부하기 위한 유틸리티.
3-panel 구성: 가격+MA+볼린저, RSI, 거래량
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 렌더링
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def render_stock_chart(
    ticker: str,
    days: int = 60,
    width: int = 800,
    height: int = 600,
) -> str | None:
    """종목 차트를 base64 PNG 문자열로 렌더링.

    Args:
        ticker: 종목코드 (예: "005930")
        days: 표시할 최근 거래일 수
        width: 이미지 너비 (px)
        height: 이미지 높이 (px)

    Returns:
        base64 인코딩된 PNG 문자열. 실패 시 None.
    """
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        logger.debug("parquet 없음: %s", ticker)
        return None

    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        logger.warning("parquet 로드 실패 %s: %s", ticker, e)
        return None

    df = df.tail(days).copy().reset_index(drop=True)
    if len(df) < 10:
        return None

    try:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(width / 100, height / 100),
            gridspec_kw={"height_ratios": [3, 1, 1]},
            sharex=True,
        )

        x = np.arange(len(df))

        # ── Panel 1: 가격 + MA + 볼린저 ──
        _draw_candlestick(ax1, df, x)
        _draw_moving_averages(ax1, df, x)
        _draw_bollinger(ax1, df, x)
        _draw_sar(ax1, df, x)

        ax1.set_title(f"{ticker} ({days}일)", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7, loc="upper left", ncol=3, framealpha=0.5)
        ax1.grid(True, alpha=0.2)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # ── Panel 2: RSI ──
        _draw_rsi(ax2, df, x)

        # ── Panel 3: 거래량 ──
        _draw_volume(ax3, df, x)

        # x축 날짜 레이블 (10개 간격)
        if "date" in df.columns:
            dates = df["date"].astype(str).tolist()
            step = max(1, len(dates) // 8)
            ax3.set_xticks(x[::step])
            ax3.set_xticklabels([d[5:] for d in dates[::step]], fontsize=7, rotation=45)

        plt.tight_layout(pad=0.5)

        # base64 변환
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.warning("차트 렌더링 실패 %s: %s", ticker, e)
        plt.close("all")
        return None


def render_batch(tickers: list[str], days: int = 60) -> dict[str, str | None]:
    """여러 종목 차트를 일괄 렌더링.

    Returns:
        {ticker: base64_png_or_None, ...}
    """
    results = {}
    for t in tickers:
        results[t] = render_stock_chart(t, days)
    success = sum(1 for v in results.values() if v)
    logger.info("차트 배치 렌더링: %d/%d 성공", success, len(tickers))
    return results


# ─── 내부 드로잉 함수 ────────────────────────────────────────────


def _draw_candlestick(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """간이 캔들스틱 (matplotlib bar)"""
    if not all(c in df.columns for c in ("open", "high", "low", "close")):
        ax.plot(x, df.get("close", []), color="#2196F3", linewidth=1.5)
        return

    up = df["close"] >= df["open"]
    dn = ~up

    # 몸통
    ax.bar(x[up], (df["close"] - df["open"])[up], bottom=df["open"][up],
           width=0.6, color="#EF5350", alpha=0.9, linewidth=0)
    ax.bar(x[dn], (df["open"] - df["close"])[dn], bottom=df["close"][dn],
           width=0.6, color="#42A5F5", alpha=0.9, linewidth=0)

    # 꼬리
    ax.vlines(x[up], df["low"][up], df["high"][up], color="#EF5350", linewidth=0.5)
    ax.vlines(x[dn], df["low"][dn], df["high"][dn], color="#42A5F5", linewidth=0.5)


def _draw_moving_averages(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """이동평균선"""
    ma_config = [
        ("ma5", "#FFB74D", "MA5"),
        ("ma20", "#FF9800", "MA20"),
        ("ma60", "#4CAF50", "MA60"),
        ("ma120", "#9C27B0", "MA120"),
    ]
    for col, color, label in ma_config:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                ax.plot(x[:len(vals)], vals.values[-len(x):],
                        color=color, linewidth=0.8, label=label, alpha=0.7)


def _draw_bollinger(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """볼린저 밴드"""
    upper_col = "bb_upper" if "bb_upper" in df.columns else "bollinger_upper"
    lower_col = "bb_lower" if "bb_lower" in df.columns else "bollinger_lower"

    if upper_col in df.columns and lower_col in df.columns:
        upper = df[upper_col].values
        lower = df[lower_col].values
        ax.fill_between(x, upper, lower, alpha=0.08, color="gray", label="BB")


def _draw_sar(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """Parabolic SAR 점"""
    if "sar" not in df.columns or "sar_trend" not in df.columns:
        return

    up_mask = df["sar_trend"] == 1
    dn_mask = df["sar_trend"] == -1

    if up_mask.any():
        ax.scatter(x[up_mask], df["sar"][up_mask],
                   marker=".", s=8, color="#4CAF50", alpha=0.6, zorder=3)
    if dn_mask.any():
        ax.scatter(x[dn_mask], df["sar"][dn_mask],
                   marker=".", s=8, color="#F44336", alpha=0.6, zorder=3)


def _draw_rsi(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """RSI 패널"""
    if "rsi" not in df.columns:
        ax.text(0.5, 0.5, "RSI N/A", transform=ax.transAxes, ha="center", fontsize=8)
        return

    ax.plot(x, df["rsi"], color="#E91E63", linewidth=1)
    ax.axhline(70, color="red", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.axhline(30, color="green", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.fill_between(x, 30, 70, alpha=0.05, color="gray")
    ax.set_ylabel("RSI", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)


def _draw_volume(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    """거래량 패널"""
    if "volume" not in df.columns:
        ax.text(0.5, 0.5, "Volume N/A", transform=ax.transAxes, ha="center", fontsize=8)
        return

    if "open" in df.columns and "close" in df.columns:
        colors = np.where(df["close"] >= df["open"], "#EF5350", "#42A5F5")
    else:
        colors = "#90A4AE"

    ax.bar(x, df["volume"], color=colors, alpha=0.7, width=0.6)

    # 거래량 이동평균 (있으면)
    vol_ma_col = None
    for candidate in ("volume_ma20", "vol_ma20", "vol_ma"):
        if candidate in df.columns:
            vol_ma_col = candidate
            break
    if vol_ma_col:
        ax.plot(x, df[vol_ma_col], color="#FF9800", linewidth=0.8)

    ax.set_ylabel("거래량", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
    ))
