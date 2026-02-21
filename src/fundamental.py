"""
Step 3: fundamental.py — 재무 데이터 관리

v6.5: DART OpenAPI 연동으로 실제 매출/영업이익 데이터 사용.
DART_API_KEY가 없으면 기존 sector_map 기반 fallback 유지.

백테스트에서는 Look-ahead bias 방지를 위해:
- Forward PER → pykrx Trailing PER로 대체 (과거 시점 Forward PER 확보 어려움)
- EPS 리비전 → EPS 3개월 변화율로 근사
- 매출액/영업이익 → DART API (실시간) 또는 가장 최근 공시 기준

실시간에서는 DART API로 분기별 재무제표 직접 조회.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FundamentalEngine:
    """재무 데이터 로딩 + 밸류에이션 점수 계산"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.sector_per = self.config.get("sector_per", {})
        self.universe_dir = Path("data/universe")

        # 종목-업종 매핑 로드
        self.sector_map = self._load_sector_map()

        # DART 재무 캐시 (fundamentals_all.csv)
        self._fund_cache = self._load_fundamentals_cache()

        # DART 어댑터 (Lazy 초기화)
        self._dart = None
        self._dart_initialized = False

    def _load_fundamentals_cache(self) -> dict:
        """
        data/dart_cache/fundamentals_all.csv 로딩.
        ticker → {revenue_억, op_income_억, net_income_억, op_margin_pct, profitable}
        """
        cache_file = Path("data/dart_cache/fundamentals_all.csv")
        if not cache_file.exists():
            return {}

        try:
            df = pd.read_csv(cache_file, dtype={"ticker": str})
            df["ticker"] = df["ticker"].str.zfill(6)
            cache = {}
            for _, row in df.iterrows():
                cache[row["ticker"]] = {
                    "revenue": row.get("revenue_억"),
                    "operating_income": row.get("op_income_억"),
                    "net_income": row.get("net_income_억"),
                    "operating_margin": row.get("op_margin_pct"),
                    "profitable": row.get("profitable"),
                }
            logger.info(f"DART 재무 캐시 로드: {len(cache)}종목 (fundamentals_all.csv)")
            return cache
        except Exception as e:
            logger.warning(f"DART 재무 캐시 로드 실패: {e}")
            return {}

    @property
    def dart(self):
        """DART 어댑터 Lazy 로딩 (API 키 없으면 None)"""
        if not self._dart_initialized:
            self._dart_initialized = True
            try:
                from src.adapters.dart_adapter import DartAdapter
                adapter = DartAdapter()
                if adapter.is_available:
                    self._dart = adapter
                    logger.info("DART API 어댑터 초기화 완료")
                else:
                    logger.info("DART_API_KEY 미설정 — sector_map fallback 사용")
            except Exception as e:
                logger.warning(f"DART 어댑터 로드 실패: {e}")
        return self._dart

    def _load_sector_map(self) -> dict:
        """ticker → sector 매핑"""
        sector_file = self.universe_dir / "sector_map.csv"
        if sector_file.exists():
            df = pd.read_csv(sector_file)
            if "sector" in df.columns:
                return dict(zip(df["ticker"].astype(str).str.zfill(6), df["sector"]))
        return {}

    def get_sector(self, ticker: str) -> str:
        """종목의 업종 반환"""
        return self.sector_map.get(ticker, "기타")

    def get_sector_avg_per(self, ticker: str) -> float:
        """업종 평균 PER 반환"""
        sector = self.get_sector(ticker)
        return self.sector_per.get(sector, 11.0)  # 기본값 11.0 (KOSPI 평균)

    def calc_trailing_value_score(self, current_per: float, sector_avg_per: float) -> float:
        """
        Trailing PER 기반 밸류 점수 (Forward PER 대용)
        value_ratio = current_per / sector_avg_per
        """
        if pd.isna(current_per) or current_per <= 0 or pd.isna(sector_avg_per):
            return 0.3  # 데이터 없으면 중립

        ratio = current_per / sector_avg_per

        ranges = self.config["strategy"]["forward_value_ranges"]
        if ratio <= ranges["deep_discount"][1] or ratio <= ranges["discount"][1]:
            return 1.0
        elif ratio <= ranges["fair"][1]:
            return 0.5
        elif ratio <= ranges["premium"][1]:
            return 0.2
        else:
            return 0.0

    def calc_eps_revision_score(self, df: pd.DataFrame, idx: int,
                                lookback_days: int = 60) -> float:
        """
        EPS 리비전 점수 (3개월 전 대비 현재 EPS 변화율)

        상향(>5%):  1.0
        유지(-5~5%): 0.7
        하향(<-5%): 0.3
        적자 전환:   0.0
        """
        if "fund_EPS" not in df.columns:
            return 0.5  # 데이터 없으면 중립

        current_eps = df["fund_EPS"].iloc[idx]
        prev_idx = max(0, idx - lookback_days)
        prev_eps = df["fund_EPS"].iloc[prev_idx]

        if pd.isna(current_eps) or pd.isna(prev_eps):
            return 0.5

        # 적자 전환
        if current_eps <= 0 and prev_eps > 0:
            return 0.0
        if current_eps <= 0:
            return 0.1

        # 리비전 계산
        if prev_eps != 0:
            revision_pct = (current_eps - prev_eps) / abs(prev_eps) * 100
        else:
            revision_pct = 0

        if revision_pct > 5:
            return 1.0   # 상향
        elif revision_pct >= -5:
            return 0.7   # 유지
        else:
            return 0.3   # 하향

    def calc_combined_value_score(self, per_score: float, eps_score: float) -> float:
        """밸류 종합 점수 = PER 비율 × 0.6 + EPS 리비전 × 0.4"""
        return per_score * 0.6 + eps_score * 0.4

    # ──────────────────────────────────────────────
    # Pre-screening 필터 (DART 연동)
    # ──────────────────────────────────────────────

    def check_revenue_filter(self, ticker: str,
                              min_revenue_억: float = 1000) -> bool:
        """
        매출 필터: 매출 >= min_revenue_억 (기본 1,000억원)

        우선순위:
        1. fundamentals_all.csv 캐시 (API 호출 없이 즉시)
        2. DART API 실시간 조회
        3. Fallback: sector_map에 있으면 통과 (기존 동작)
        """
        # 1. CSV 캐시에서 조회 (가장 빠름)
        cached = self._fund_cache.get(ticker.zfill(6))
        if cached and pd.notna(cached.get("revenue")):
            revenue = cached["revenue"]
            passed = revenue >= min_revenue_억
            if not passed:
                logger.debug(
                    f"{ticker}: 매출 {revenue:.0f}억 < {min_revenue_억}억 → 필터 차단 (캐시)"
                )
            return passed

        # 2. DART API 실시간
        if self.dart is not None:
            year = datetime.now().year
            financials = self.dart.get_key_financials(ticker, year)
            revenue = financials.get("revenue")
            if revenue is not None:
                passed = revenue >= min_revenue_억
                if not passed:
                    logger.debug(
                        f"{ticker}: 매출 {revenue:.0f}억 < {min_revenue_억}억 → 필터 차단"
                    )
                return passed

        # 3. Fallback: sector_map 기반
        if ticker in self.sector_map:
            return True
        return False

    def check_profitability(self, df: pd.DataFrame, idx: int,
                            ticker: str | None = None) -> bool:
        """
        수익성 필터: 영업이익 > 0

        우선순위:
        1. CSV 캐시 profitable 필드
        2. Fallback: EPS > 0으로 근사 (기존 백테스트 동작)
        """
        # 1. CSV 캐시 (ticker가 넘어온 경우)
        if ticker:
            cached = self._fund_cache.get(ticker.zfill(6))
            if cached and cached.get("profitable") is not None:
                return bool(cached["profitable"])

        # 2. Fallback: EPS 기반 (백테스트 및 DART 없을 때)
        if "fund_EPS" not in df.columns:
            return True

        current_eps = df["fund_EPS"].iloc[idx]
        prev_eps = df["fund_EPS"].iloc[max(0, idx - 60)]

        if pd.isna(current_eps) or pd.isna(prev_eps):
            return True

        return current_eps > 0 and prev_eps > 0

    def check_profitability_dart(self, ticker: str, quarters: int = 2) -> bool | None:
        """
        DART API 기반 연속 흑자 확인 (실시간 스캐너용).

        Returns:
            True/False: 판정 결과
            None: DART 데이터 없음 (fallback 필요)
        """
        if self.dart is None:
            return None

        year = datetime.now().year
        return self.dart.check_consecutive_profit(ticker, year, quarters)

    # ──────────────────────────────────────────────
    # DART 재무 데이터 직접 접근 (스캐너/리포트용)
    # ──────────────────────────────────────────────

    def get_financials(self, ticker: str, year: int | None = None) -> dict:
        """
        종목 핵심 재무지표 조회.

        우선순위: CSV 캐시 → DART API → fallback

        Returns:
            {
                "revenue": 매출(억원),
                "operating_income": 영업이익(억원),
                "net_income": 순이익(억원),
                "operating_margin": 영업이익률(%),
                "profitable": 흑자 여부,
                "source": "cache" | "dart" | "fallback",
            }
        """
        if year is None:
            year = datetime.now().year

        # 1. CSV 캐시
        cached = self._fund_cache.get(ticker.zfill(6))
        if cached and pd.notna(cached.get("revenue")):
            return {
                "revenue": cached["revenue"],
                "operating_income": cached["operating_income"],
                "net_income": cached["net_income"],
                "operating_margin": cached["operating_margin"],
                "profitable": cached["profitable"],
                "source": "cache",
            }

        # 2. DART API 실시간
        if self.dart is not None:
            result = self.dart.get_key_financials(ticker, year)
            if result.get("revenue") is not None:
                result["source"] = "dart"
                return result

        # 3. Fallback
        return {
            "revenue": None,
            "operating_income": None,
            "net_income": None,
            "operating_margin": None,
            "profitable": ticker in self.sector_map,
            "source": "fallback",
        }


    # ──────────────────────────────────────────────
    # 분기 실적 변화율 + 방향성 판정 (Earnings Momentum)
    # ──────────────────────────────────────────────

    def _load_historical_quarters(self) -> pd.DataFrame:
        """fundamentals_historical.csv → 개별 분기 DataFrame 반환.

        원본 형식: Q1/Q2/Q3 = 개별 분기, Q4 = 연간 누적
        → Q4를 개별(=연간 - Q1 - Q2 - Q3)로 변환하여 반환.
        """
        if hasattr(self, "_hist_quarters_cache"):
            return self._hist_quarters_cache

        hist_path = Path("data/dart_cache/fundamentals_historical.csv")
        if not hist_path.exists():
            self._hist_quarters_cache = pd.DataFrame()
            return self._hist_quarters_cache

        df = pd.read_csv(hist_path, dtype={"ticker": str})
        df["ticker"] = df["ticker"].str.zfill(6)
        for col in ["revenue_억", "op_income_억", "net_income_억"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["year"] = df["year"].astype(int)
        df["quarter"] = df["quarter"].astype(int)

        rows = []
        for (ticker, year), grp in df.groupby(["ticker", "year"]):
            grp = grp.sort_values("quarter")
            q_map = {int(r["quarter"]): r for _, r in grp.iterrows()}

            for q in [1, 2, 3]:
                if q in q_map:
                    r = q_map[q]
                    rows.append({
                        "ticker": ticker,
                        "year": year,
                        "quarter": q,
                        "revenue": float(r["revenue_억"]) if pd.notna(r["revenue_억"]) else 0,
                        "op_income": float(r["op_income_억"]) if pd.notna(r["op_income_억"]) else 0,
                        "net_income": float(r["net_income_억"]) if pd.notna(r["net_income_억"]) else 0,
                    })

            if 4 in q_map:
                ann = q_map[4]
                ann_rev = float(ann["revenue_억"]) if pd.notna(ann["revenue_억"]) else 0
                ann_op = float(ann["op_income_억"]) if pd.notna(ann["op_income_억"]) else 0
                ann_net = float(ann["net_income_억"]) if pd.notna(ann["net_income_억"]) else 0

                sum_q123 = lambda col: sum(
                    float(q_map[q][col]) if q in q_map and pd.notna(q_map[q][col]) else 0
                    for q in [1, 2, 3]
                )
                rows.append({
                    "ticker": ticker,
                    "year": year,
                    "quarter": 4,
                    "revenue": ann_rev - sum_q123("revenue_억"),
                    "op_income": ann_op - sum_q123("op_income_억"),
                    "net_income": ann_net - sum_q123("net_income_억"),
                })

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)
        self._hist_quarters_cache = result
        logger.info("분기 실적 히스토리 로드: %d행 (%d종목)",
                     len(result), result["ticker"].nunique() if not result.empty else 0)
        return result

    def calc_earnings_momentum(self, ticker: str) -> dict:
        """분기 실적 변화율 + 방향성 판정.

        Returns:
            {
                "verdict": "ACCELERATING" | "DECELERATING" | ... ,
                "score": 0~20 (Consensus축 점수),
                "op_yoy_pct": YoY 영업이익 변화율,
                "net_yoy_pct": YoY 순이익 변화율,
                "opm_trend": OPM 4분기 추세 기울기,
                "is_holding_co": 지주사 여부,
                "quarters": [...],  # 최근 8분기 데이터
                "detail": "판정 근거 설명",
            }
        """
        ticker = ticker.zfill(6)
        hist = self._load_historical_quarters()
        empty = {
            "verdict": "NO_DATA", "score": 0, "op_yoy_pct": None,
            "net_yoy_pct": None, "opm_trend": None, "is_holding_co": False,
            "quarters": [], "detail": "DART 분기 데이터 없음",
        }

        if hist.empty:
            return empty

        sub = hist[hist["ticker"] == ticker].copy()
        if len(sub) < 5:
            return empty

        # 최근 8분기
        sub = sub.tail(8).reset_index(drop=True)

        # 지주사 판별: 순이익 >> 영업이익 (3배 이상)이 2분기 이상
        holding_co_count = sum(
            1 for _, r in sub.iterrows()
            if r["op_income"] > 0 and r["net_income"] > r["op_income"] * 3
        )
        is_holding = holding_co_count >= 2

        # 기준 지표 선택: 지주사는 순이익, 일반은 영업이익
        metric_col = "net_income" if is_holding else "op_income"

        # 영업이익률
        sub["opm"] = sub.apply(
            lambda r: (r["op_income"] / r["revenue"] * 100) if r["revenue"] > 10 else 0,
            axis=1,
        )

        # ── YoY 계산 (같은 분기 전년 대비) ──
        sub["metric_yoy"] = None
        for i in range(len(sub)):
            if i >= 4:
                prev = sub.iloc[i - 4][metric_col]
                curr = sub.iloc[i][metric_col]
                if prev != 0:
                    sub.at[sub.index[i], "metric_yoy"] = (curr / abs(prev) - 1) * 100

        # ── QoQ 계산 ──
        sub["metric_qoq"] = sub[metric_col].pct_change() * 100

        # ── 방향성 판정 (YoY 우선) ──
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        metric_now = last[metric_col]
        metric_prev = prev[metric_col]
        yoy = last["metric_yoy"]
        qoq = last["metric_qoq"]

        # 2분기 연속 YoY 방향
        prev_yoy = sub.iloc[-2]["metric_yoy"] if len(sub) >= 2 else None
        consecutive_yoy_up = (
            yoy is not None and prev_yoy is not None
            and float(yoy) > 0 and float(prev_yoy) > 0
        )
        consecutive_yoy_down = (
            yoy is not None and prev_yoy is not None
            and float(yoy) < 0 and float(prev_yoy) < 0
        )

        # OPM 4분기 추세 (선형회귀 기울기)
        recent_4_opm = sub["opm"].tail(4).values
        if len(recent_4_opm) == 4 and not any(np.isnan(recent_4_opm)):
            opm_slope = float(np.polyfit(np.arange(4), recent_4_opm, 1)[0])
        else:
            opm_slope = 0.0

        # ── 판정 ──
        if metric_prev < 0 and metric_now > 0:
            verdict = "TURNAROUND_STRONG"
            detail = f"적자→흑자 전환! ({'순이익' if is_holding else '영업이익'} {metric_prev:.1f}→{metric_now:.1f}억)"
        elif metric_prev < 0 and metric_now < 0 and abs(metric_now) < abs(metric_prev) * 0.7:
            verdict = "TURNAROUND_EARLY"
            detail = f"적자 30%+ 축소 ({metric_prev:.1f}→{metric_now:.1f}억)"
        elif metric_now > 0 and yoy is not None and float(yoy) > 20 and consecutive_yoy_up:
            verdict = "ACCELERATING"
            detail = f"2분기 연속 YoY 성장 (최근 {float(yoy):+.1f}%)"
        elif metric_now > 0 and yoy is not None and float(yoy) > 0:
            verdict = "GROWING"
            detail = f"YoY 개선 ({float(yoy):+.1f}%)"
        elif metric_now > 0 and yoy is not None and float(yoy) > -15:
            verdict = "STABLE"
            detail = f"흑자 유지, YoY 소폭 변동 ({float(yoy):+.1f}%)"
        elif metric_now > 0 and consecutive_yoy_down:
            verdict = "DECELERATING"
            detail = f"2분기 연속 YoY 악화 ({float(yoy):+.1f}%)"
        elif metric_now <= 0:
            verdict = "DETERIORATING"
            detail = f"적자 전환/지속 ({metric_now:.1f}억)"
        else:
            verdict = "UNCERTAIN"
            detail = f"판정 불충분 (YoY={yoy})"

        # OPM 추세 보정
        if opm_slope > 1.0 and verdict in ("STABLE", "GROWING"):
            detail += f" + OPM 개선(+{opm_slope:.1f}%p/Q)"
        elif opm_slope < -1.0 and verdict in ("GROWING", "ACCELERATING"):
            detail += f" ⚠️ OPM 악화({opm_slope:+.1f}%p/Q)"

        # ── 점수 매기기 (0~20) ──
        score_map = {
            "TURNAROUND_STRONG": 18,
            "ACCELERATING": 16,
            "TURNAROUND_EARLY": 14,
            "GROWING": 12,
            "STABLE": 8,
            "UNCERTAIN": 5,
            "DECELERATING": 3,
            "DETERIORATING": 1,
            "NO_DATA": 0,
        }
        base_score = score_map.get(verdict, 5)

        # OPM 보정: 개선 +2, 악화 -2
        if opm_slope > 1.0:
            base_score = min(base_score + 2, 20)
        elif opm_slope < -1.0:
            base_score = max(base_score - 2, 0)

        # 분기 데이터 정리
        q_list = []
        for _, r in sub.iterrows():
            q_list.append({
                "period": f"{int(r['year'])}Q{int(r['quarter'])}",
                "revenue": round(r["revenue"], 1),
                "op_income": round(r["op_income"], 1),
                "net_income": round(r["net_income"], 1),
                "opm": round(r["opm"], 1),
            })

        yoy_val = float(yoy) if yoy is not None and not pd.isna(yoy) else None
        net_yoy = None
        if len(sub) >= 5:
            n_prev = sub.iloc[-5]["net_income"]
            n_curr = sub.iloc[-1]["net_income"]
            if n_prev != 0:
                net_yoy = round((n_curr / abs(n_prev) - 1) * 100, 1)

        return {
            "verdict": verdict,
            "score": base_score,
            "op_yoy_pct": round(yoy_val, 1) if yoy_val is not None else None,
            "net_yoy_pct": net_yoy,
            "opm_trend": round(opm_slope, 2),
            "is_holding_co": is_holding,
            "quarters": q_list,
            "detail": detail,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = FundamentalEngine()
    print(f"업종 PER 맵: {engine.sector_per}")
    print(f"종목-업종 매핑: {len(engine.sector_map)}종목")
    print(f"DART API: {'연결됨' if engine.dart else '미연결'}")

    # Earnings Momentum 테스트
    test_tickers = {"036640": "HRS", "107590": "미원홀딩스", "036560": "KZ정밀"}
    for t, name in test_tickers.items():
        result = engine.calc_earnings_momentum(t)
        print(f"\n{'='*60}")
        print(f"  {name} ({t})")
        print(f"  판정: {result['verdict']} ({result['score']}/20점)")
        print(f"  사유: {result['detail']}")
        if result["op_yoy_pct"] is not None:
            print(f"  영업이익 YoY: {result['op_yoy_pct']:+.1f}%")
        if result["net_yoy_pct"] is not None:
            print(f"  순이익 YoY: {result['net_yoy_pct']:+.1f}%")
        if result["opm_trend"] is not None:
            print(f"  OPM 추세: {result['opm_trend']:+.2f}%p/Q")
        if result["is_holding_co"]:
            print(f"  ⚠️ 지주사 감지 → 순이익 기준 판정")
        print(f"{'='*60}")
