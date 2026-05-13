"""
ETF 전파 모델 (ETF Transmission Model)
═══════════════════════════════════════════════════════
미국 ETF 급등/급락 → 한국 섹터 ETF 기대수익 → 개별종목 기대수익 계산

핵심 공식:
    stock_expected = US_ETF_ret × beta_us_kr × (weight / avg_weight) × vol_factor

구조:
    1. US overnight signal에서 US ETF 수익률 로드
    2. beta_us_kr 적용 → KR 섹터 ETF 기대수익률
    3. 구성종목 비중 기반 분배 → 개별종목 기대수익률
    4. surge_pullback_engine 연동 → 시그널 우선순위/임계값 조정

사용법:
    python scripts/etf_transmission.py [--date YYYY-MM-DD]

    # 모듈로 사용:
    from scripts.etf_transmission import ETFTransmissionModel
    model = ETFTransmissionModel()
    results = model.compute()  # 최신 overnight signal 기반 계산
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 경로 ───
DATA_DIR = project_root / "data"
HOLDINGS_PATH = DATA_DIR / "etf_holdings" / "sector_etf_constituents.json"
OVERNIGHT_SIGNAL_PATH = DATA_DIR / "us_market" / "overnight_signal.json"
PARQUET_PATH = DATA_DIR / "us_market" / "us_daily.parquet"
OUTPUT_PATH = DATA_DIR / "etf_transmission_result.json"


class ETFTransmissionModel:
    """미국 ETF → 한국 개별종목 기대수익률 전파 모델."""

    def __init__(self, holdings_path: Path = HOLDINGS_PATH):
        self.holdings = self._load_holdings(holdings_path)
        self.us_signal = None
        self.parquet_df = None

    def _load_holdings(self, path: Path) -> dict:
        """구성종목 비중 데이터 로드."""
        if not path.exists():
            logger.error(f"Holdings 데이터 없음: {path}")
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        # _metadata 키 제거
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def _load_overnight_signal(self) -> dict:
        """최신 overnight signal 로드."""
        if not OVERNIGHT_SIGNAL_PATH.exists():
            logger.warning(f"Overnight signal 없음: {OVERNIGHT_SIGNAL_PATH}")
            return {}
        return json.loads(OVERNIGHT_SIGNAL_PATH.read_text(encoding="utf-8"))

    def _get_us_etf_returns(self) -> dict:
        """US ETF 수익률 추출 (1D + 5D).

        Returns:
            {"SOXX": {"ret_1d": 0.057, "ret_5d": 0.117}, ...}
        """
        returns = {}

        # 1. Overnight signal에서 추출 (우선)
        if self.us_signal:
            sector_sigs = self.us_signal.get("sector_signals", {})
            # sector_signals의 driver에서 파싱하기보다 parquet 직접 사용

        # 2. Parquet에서 직접 읽기 (정확한 수치)
        if PARQUET_PATH.exists():
            try:
                df = pd.read_parquet(PARQUET_PATH)
                self.parquet_df = df
                latest = df.iloc[-1]

                for us_etf in self.holdings.keys():
                    prefix = us_etf.lower()
                    ret_1d_col = f"{prefix}_ret_1d"
                    ret_5d_col = f"{prefix}_ret_5d"

                    ret_1d = float(latest.get(ret_1d_col, 0) or 0)
                    ret_5d = float(latest.get(ret_5d_col, 0) or 0)

                    if ret_1d != 0 or ret_5d != 0:
                        returns[us_etf] = {
                            "ret_1d": ret_1d,
                            "ret_5d": ret_5d,
                        }
            except Exception as e:
                logger.warning(f"Parquet 로드 실패: {e}")

        return returns

    def _compute_historical_beta(self, us_prefix: str) -> Optional[float]:
        """US ETF vs KR(EWY proxy) 히스토리컬 베타 계산 (parquet 데이터 기반).

        120일 covariance → beta 추정.
        데이터 부족 시 None 반환 (정적 beta 사용).
        """
        if self.parquet_df is None or self.parquet_df.empty:
            return None

        us_col = f"{us_prefix.lower()}_ret_1d"
        # KR ETF 수익률은 EWY로 프록시 (직접 KR ETF 일간수익률 없음)
        kr_col = "ewy_ret_1d"

        if us_col not in self.parquet_df.columns or kr_col not in self.parquet_df.columns:
            return None

        us_rets = self.parquet_df[us_col].dropna().tail(120)
        kr_rets = self.parquet_df[kr_col].dropna().tail(120)

        # 인덱스 정렬
        common = us_rets.index.intersection(kr_rets.index)
        if len(common) < 30:
            return None

        us_aligned = us_rets.loc[common]
        kr_aligned = kr_rets.loc[common]

        # beta = cov(US, KR) / var(US)
        cov = np.cov(us_aligned.values, kr_aligned.values)
        if cov[0, 0] == 0:
            return None

        beta = cov[0, 1] / cov[0, 0]
        return round(max(0.1, min(1.5, beta)), 3)

    def compute(self, min_expected_pct: float = 1.0) -> dict:
        """전파 모델 계산 메인.

        Args:
            min_expected_pct: 결과에 포함할 최소 기대수익률 (%) — 잡음 필터

        Returns:
            {
                "computed_at": "2026-05-10 08:30",
                "us_close_date": "2026-05-09",
                "transmissions": {
                    "SOXX": {
                        "us_ret_1d": 5.7,
                        "us_ret_5d": 11.7,
                        "kr_etf": "TIGER 반도체",
                        "kr_etf_expected": 3.7,
                        "beta_used": 0.65,
                        "stocks": [
                            {"ticker": "000660", "name": "SK하이닉스",
                             "weight": 28.5, "expected_ret": 7.4,
                             "concentration_factor": 2.0},
                            ...
                        ]
                    }
                },
                "top_picks": [...],  # 기대수익률 상위 종목
                "summary": "..."
            }
        """
        self.us_signal = self._load_overnight_signal()
        us_returns = self._get_us_etf_returns()

        if not us_returns:
            logger.warning("US ETF 수익률 데이터 없음 — 계산 중단")
            return {"error": "no_us_data", "transmissions": {}}

        result = {
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "us_close_date": self.us_signal.get("us_close_date", "unknown"),
            "transmissions": {},
            "top_picks": [],
            "all_stocks": [],
        }

        for us_etf, etf_config in self.holdings.items():
            if us_etf not in us_returns:
                continue

            us_ret = us_returns[us_etf]
            ret_1d = us_ret["ret_1d"]
            ret_5d = us_ret["ret_5d"]

            # 전파 의미 있는 움직임만 처리 (1D > 1% 또는 5D > 3%)
            if abs(ret_1d) < 0.01 and abs(ret_5d) < 0.03:
                continue

            # 주요 수익률 선택: 1D가 크면 1D, 아니면 5D/5 사용
            # 로직: 급등일(1D 큰 날)은 1D 사용, 지속 상승은 5D 평균 사용
            if abs(ret_1d) >= 0.03:
                primary_ret = ret_1d
                ret_type = "1D_surge"
            elif abs(ret_5d) >= 0.05:
                primary_ret = ret_5d / 3  # 5일 누적의 1/3을 다음날 기대치로
                ret_type = "5D_momentum"
            else:
                primary_ret = ret_1d
                ret_type = "1D_normal"

            # Beta 계산 (동적 vs 정적)
            dynamic_beta = self._compute_historical_beta(us_etf)
            static_beta = etf_config["beta_us_kr"]
            beta_used = dynamic_beta if dynamic_beta else static_beta

            # KR 섹터 ETF 기대수익률
            kr_etf_expected = primary_ret * beta_used

            # 구성종목별 기대수익률 분배
            holdings = [h for h in etf_config["holdings"] if h["weight"] > 0]
            if not holdings:
                continue

            avg_weight = sum(h["weight"] for h in holdings) / len(holdings)
            stocks = []

            for holding in holdings:
                weight = holding["weight"]
                # 집중도 팩터: 비중이 평균 대비 높을수록 더 많이 움직임
                concentration = weight / avg_weight if avg_weight > 0 else 1.0

                # 개별종목 기대수익률
                # 공식: KR_ETF_expected × concentration_factor
                # concentration cap: 최대 2.5배 (과도한 예측 방지)
                concentration_capped = min(2.5, concentration)
                stock_expected = kr_etf_expected * concentration_capped

                # 변동성 팩터 (소형주 증폭) — AUM 기반 근사
                # 대형주(비중 >15%)는 ETF와 거의 동행, 중소형은 증폭
                if weight >= 15:
                    vol_factor = 1.0
                elif weight >= 8:
                    vol_factor = 1.15
                else:
                    vol_factor = 1.3

                stock_expected *= vol_factor

                stock_entry = {
                    "ticker": holding["ticker"],
                    "name": holding["name"],
                    "weight": weight,
                    "expected_ret_pct": round(stock_expected * 100, 2),
                    "concentration_factor": round(concentration_capped, 2),
                    "vol_factor": vol_factor,
                }

                # 최소 기대수익률 필터
                if abs(stock_expected * 100) >= min_expected_pct:
                    stocks.append(stock_entry)

            # 기대수익률 내림차순 정렬
            stocks.sort(key=lambda x: x["expected_ret_pct"], reverse=True)

            transmission_entry = {
                "us_ret_1d_pct": round(ret_1d * 100, 2),
                "us_ret_5d_pct": round(ret_5d * 100, 2),
                "ret_type": ret_type,
                "kr_etf": etf_config["kr_etf"]["name"],
                "kr_etf_ticker": etf_config["kr_etf"]["ticker"],
                "kr_etf_expected_pct": round(kr_etf_expected * 100, 2),
                "beta_used": beta_used,
                "beta_source": "dynamic" if dynamic_beta else "static",
                "stock_count": len(stocks),
                "stocks": stocks,
            }

            result["transmissions"][us_etf] = transmission_entry

            # all_stocks에 추가 (중복 제거용)
            for s in stocks:
                s_copy = dict(s)
                s_copy["source_etf"] = us_etf
                s_copy["kr_etf"] = etf_config["kr_etf"]["name"]
                result["all_stocks"].append(s_copy)

        # ─── Top Picks 생성 (전체 종목 중 기대수익률 상위) ───
        # 같은 종목이 여러 ETF에서 나올 수 있음 → 최대값 채택
        stock_map = {}
        for s in result["all_stocks"]:
            ticker = s["ticker"]
            if ticker not in stock_map or s["expected_ret_pct"] > stock_map[ticker]["expected_ret_pct"]:
                stock_map[ticker] = s

        top_picks = sorted(stock_map.values(), key=lambda x: x["expected_ret_pct"], reverse=True)
        result["top_picks"] = top_picks[:15]  # 상위 15종목

        # ─── 요약 ───
        active_etfs = list(result["transmissions"].keys())
        if active_etfs:
            etf_parts = []
            for etf in active_etfs:
                t = result["transmissions"][etf]
                etf_parts.append(f"{etf} {t['us_ret_1d_pct']:+.1f}%→{t['kr_etf']} {t['kr_etf_expected_pct']:+.1f}%")
            summary = f"ETF전파 {len(active_etfs)}개: " + " | ".join(etf_parts)

            if result["top_picks"]:
                top3 = result["top_picks"][:3]
                picks_str = ", ".join(
                    f"{p['name']}({p['expected_ret_pct']:+.1f}%)" for p in top3
                )
                summary += f" | TOP3: {picks_str}"
        else:
            summary = "ETF전파: 유의미한 US ETF 변동 없음"

        result["summary"] = summary

        # all_stocks는 내부용이므로 저장 시 제거
        del result["all_stocks"]

        return result



def save_result(result: dict):
    """결과를 JSON으로 저장."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"저장: {OUTPUT_PATH}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ETF 전파 모델 계산")
    parser.add_argument("--min-pct", type=float, default=1.0, help="최소 기대수익률 %%")
    args = parser.parse_args()

    model = ETFTransmissionModel()
    result = model.compute(min_expected_pct=args.min_pct)

    if "error" in result:
        logger.error(f"계산 실패: {result['error']}")
        return

    save_result(result)

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("  ETF 전파 모델 — 개별종목 기대수익률")
    print("=" * 60)
    print(f"  미국 마감일: {result['us_close_date']}")
    print(f"  계산 시각: {result['computed_at']}")
    print("-" * 60)

    for etf, trans in result["transmissions"].items():
        print(f"\n  [{etf}] {trans['us_ret_1d_pct']:+.1f}% (5D: {trans['us_ret_5d_pct']:+.1f}%)")
        print(f"    → {trans['kr_etf']} 기대: {trans['kr_etf_expected_pct']:+.1f}% (β={trans['beta_used']})")
        print(f"    구성종목 ({trans['stock_count']}개):")
        for s in trans["stocks"][:5]:
            print(
                f"      {s['name']:12s} | 비중 {s['weight']:5.1f}% | "
                f"기대 {s['expected_ret_pct']:+5.1f}% | "
                f"집중도 ×{s['concentration_factor']:.1f}"
            )

    if result["top_picks"]:
        print("\n" + "-" * 60)
        print("  [TOP PICKS] 기대수익률 상위")
        print("-" * 60)
        for i, p in enumerate(result["top_picks"][:10], 1):
            print(
                f"  {i:2d}. {p['name']:12s} ({p['ticker']}) | "
                f"기대 {p['expected_ret_pct']:+5.1f}% | "
                f"{p['source_etf']} → {p['kr_etf']}"
            )

    print("\n" + "=" * 60)
    print(f"  {result['summary']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
