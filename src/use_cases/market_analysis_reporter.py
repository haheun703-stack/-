"""
장시작/장마감 통합 분석 보고서 생성기

시장 체제, 글로벌 매크로, 매수 후보, 보유 포지션, 시장 시그널을
수집하여 텔레그램 보고서용 dict를 조립.

사용법:
    reporter = MarketAnalysisReporter(config)
    data = reporter.generate("morning")   # 장시작
    data = reporter.generate("closing")   # 장마감
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MACRO_PATH = Path("data/macro/global_indices.parquet")
SECTOR_SCAN_PATH = Path("data/sector_rotation/krx_sector_scan.json")


class MarketAnalysisReporter:
    """장시작/장마감 통합 분석 보고서 데이터 생성"""

    def __init__(self, config: dict):
        self.config = config

    def generate(self, report_type: str = "morning") -> dict:
        """
        분석 보고서 데이터를 수집하여 dict로 반환.

        Args:
            report_type: "morning" (장시작) / "closing" (장마감)

        Returns:
            보고서 데이터 dict
        """
        now = datetime.now()

        data = {
            "report_type": report_type,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "regime": self._collect_regime(),
            "macro": self._collect_macro(),
            "candidates": self._collect_candidates(),
            "positions": [],
            "portfolio_summary": {},
            "market_signals": [],
        }

        # 보유 포지션
        pos_data = self._collect_positions()
        data["positions"] = pos_data["positions"]
        data["portfolio_summary"] = pos_data["summary"]

        # 시장 시그널 (장마감 only)
        if report_type == "closing":
            data["market_signals"] = self._collect_market_signals()

        return data

    # ──────────────────────────────────────────
    # Section 1: 시장 체제
    # ──────────────────────────────────────────

    def _collect_regime(self) -> dict:
        """RegimeGate에서 현재 시장 체제 판정"""
        default = {
            "state": "unknown",
            "composite": 0.0,
            "position_scale": 0.0,
            "breadth": 0.0,
            "foreign": 0.0,
            "volatility": 0.0,
            "macro": 0.0,
        }
        try:
            from src.regime_gate import RegimeGate

            gate = RegimeGate(self.config)

            # 유니버스 데이터 로드 (processed parquet)
            data_dict = {}
            parquet_dir = PROCESSED_DIR if PROCESSED_DIR.exists() else RAW_DIR
            for pq in sorted(parquet_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(pq)
                    if not df.empty:
                        data_dict[pq.stem] = df
                except Exception:
                    continue

            if not data_dict:
                logger.warning("[보고서] parquet 데이터 없음 — regime 기본값 사용")
                return default

            # 마지막 인덱스로 detect
            sample_df = next(iter(data_dict.values()))
            idx = len(sample_df) - 1

            result = gate.detect(data_dict, idx)

            # macro score 추출 (details 파싱)
            macro_score = 0.0
            if result.details:
                for part in result.details.split():
                    if part.startswith("macro="):
                        try:
                            macro_score = float(part.split("=")[1])
                        except ValueError:
                            pass

            return {
                "state": result.regime,
                "composite": round(result.composite_score, 2),
                "position_scale": result.position_scale,
                "breadth": round(result.breadth_score * 100, 0),
                "foreign": round(result.foreign_score * 100, 0),
                "volatility": round(result.volatility_score * 100, 0),
                "macro": round(macro_score * 100, 0),
            }
        except Exception as e:
            logger.error("[보고서] regime 수집 실패: %s", e)
            return default

    # ──────────────────────────────────────────
    # Section 2: 글로벌 매크로
    # ──────────────────────────────────────────

    def _collect_macro(self) -> dict:
        """매크로 데이터 (VIX, USD/KRW, KOSPI, SOXX) 최신 값 + 전일 대비 변화"""
        default = {
            "vix": 0.0, "vix_change": 0.0,
            "usdkrw": 0.0, "usdkrw_change": 0.0,
            "kospi": 0.0, "kospi_change": 0.0,
            "soxx": 0.0, "soxx_change": 0.0,
        }
        try:
            if not MACRO_PATH.exists():
                logger.warning("[보고서] 매크로 데이터 없음: %s", MACRO_PATH)
                return default

            df = pd.read_parquet(MACRO_PATH)
            if df.empty or len(df) < 2:
                return default

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            def _pct_change(curr, prev_val):
                if pd.isna(curr) or pd.isna(prev_val) or prev_val == 0:
                    return 0.0
                return round((curr - prev_val) / prev_val * 100, 2)

            result = {}
            col_map = {
                "vix": "vix_close",
                "usdkrw": "usdkrw_close",
                "kospi": "kospi_close",
                "soxx": "soxx_close",
            }

            for key, col in col_map.items():
                curr = latest.get(col, 0.0)
                prev_val = prev.get(col, 0.0)
                if pd.isna(curr):
                    curr = 0.0
                if pd.isna(prev_val):
                    prev_val = 0.0
                result[key] = round(float(curr), 2)
                result[f"{key}_change"] = _pct_change(curr, prev_val)

            return result
        except Exception as e:
            logger.error("[보고서] 매크로 수집 실패: %s", e)
            return default

    # ──────────────────────────────────────────
    # Section 3: 매수 후보
    # ──────────────────────────────────────────

    def _collect_candidates(self) -> list[dict]:
        """섹터 로테이션 스캔 결과에서 매수 후보 로드"""
        try:
            if not SECTOR_SCAN_PATH.exists():
                return []

            import json
            with open(SECTOR_SCAN_PATH, encoding="utf-8") as f:
                data = json.load(f)

            candidates = []
            for item in data.get("smart_money", []):
                candidates.append({
                    "ticker": str(item.get("ticker", "")).zfill(6),
                    "grade": "SMART",
                    "trigger": f"BB{item.get('bb_pct', 0):.0f}%",
                    "zone_score": 0,
                    "entry": 0,
                    "stop": item.get("stop_pct", -7),
                    "target": 0,
                })
            for item in data.get("theme_money", []):
                candidates.append({
                    "ticker": str(item.get("ticker", "")).zfill(6),
                    "grade": "THEME",
                    "trigger": f"ADX{item.get('adx', 0):.0f}",
                    "zone_score": 0,
                    "entry": 0,
                    "stop": item.get("stop_pct", -7),
                    "target": 0,
                })

            return candidates[:10]
        except Exception as e:
            logger.error("[보고서] 매수 후보 수집 실패: %s", e)
            return []

    # ──────────────────────────────────────────
    # Section 4: 보유 포지션
    # ──────────────────────────────────────────

    def _collect_positions(self) -> dict:
        """PositionTracker에서 보유 포지션 조회"""
        empty = {
            "positions": [],
            "summary": {
                "count": 0,
                "total_invested": 0,
                "total_eval": 0,
                "total_pnl_pct": 0.0,
            },
        }
        try:
            from src.use_cases.position_tracker import PositionTracker

            tracker = PositionTracker(self.config)
            if not tracker.positions:
                return empty

            today = datetime.now().strftime("%Y-%m-%d")
            positions = []

            for p in tracker.positions:
                try:
                    hold_days = (
                        datetime.strptime(today, "%Y-%m-%d")
                        - datetime.strptime(p.entry_date, "%Y-%m-%d")
                    ).days
                except Exception:
                    hold_days = 0

                positions.append({
                    "ticker": p.ticker,
                    "name": p.name,
                    "shares": p.shares,
                    "entry_price": int(p.entry_price),
                    "current_price": int(p.current_price),
                    "pnl_pct": round(p.unrealized_pnl_pct, 1),
                    "hold_days": hold_days,
                    "partial_exits": p.partial_exits_done,
                })

            summary = tracker.get_summary()

            return {
                "positions": positions,
                "summary": {
                    "count": summary["count"],
                    "total_invested": summary["total_investment"],
                    "total_eval": summary["total_eval"],
                    "total_pnl_pct": summary["total_pnl_pct"],
                },
            }
        except Exception as e:
            logger.error("[보고서] 포지션 수집 실패: %s", e)
            return empty

    # ──────────────────────────────────────────
    # Section 5: 시장 시그널 (장마감 only)
    # ──────────────────────────────────────────

    def _collect_market_signals(self) -> list[dict]:
        """MarketSignalScanner로 주요 시장 시그널 수집"""
        try:
            from src.market_signal_scanner import MarketSignalScanner

            scanner = MarketSignalScanner()
            parquet_dir = PROCESSED_DIR if PROCESSED_DIR.exists() else RAW_DIR

            all_signals = []
            for pq in sorted(parquet_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(pq)
                    if df.empty or len(df) < 60:
                        continue
                    signals = scanner.scan_all(df)
                    for sig in signals:
                        all_signals.append({
                            "category": sig.title,
                            "ticker": pq.stem,
                            "importance": sig.importance,
                            "confidence": sig.confidence,
                            "detail": sig.description,
                        })
                except Exception:
                    continue

            # 중요도/신뢰도 순 정렬, 상위 10개
            importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            all_signals.sort(
                key=lambda s: (importance_order.get(s["importance"], 9), -s["confidence"])
            )
            return all_signals[:10]
        except Exception as e:
            logger.error("[보고서] 시장 시그널 수집 실패: %s", e)
            return []
