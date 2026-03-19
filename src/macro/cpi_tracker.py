"""CPI/PCE/실업률 트래커 — FRED API 기반 스태그플레이션 감지

FRED 시리즈:
  - CPIAUCSL: CPI (All Urban Consumers, 월간)
  - PCEPILFE: Core PCE (식품/에너지 제외, 월간)
  - UNRATE: 실업률 (월간)
  - FEDFUNDS: 연방기금금리 (월간)

출력: data/macro/cpi_data.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/macro")
OUTPUT_PATH = DATA_DIR / "cpi_data.json"

# FRED 시리즈 ID
SERIES = {
    "cpi": "CPIAUCSL",
    "core_pce": "PCEPILFE",
    "unemployment": "UNRATE",
    "fed_funds": "FEDFUNDS",
}


class CPITracker:
    """FRED API로 CPI/PCE/실업률 수집 + 스태그플레이션 판단."""

    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            from dotenv import load_dotenv
            load_dotenv()
            self.api_key = os.environ.get("FRED_API_KEY", "")

    def update(self) -> dict:
        """FRED에서 최신 데이터 가져와서 저장."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("FRED_API_KEY 없음 — 기존 데이터 유지")
            return self._load_existing()

        try:
            from fredapi import Fred
        except ImportError:
            logger.warning("fredapi 미설치 — pip install fredapi 필요")
            return self._load_existing()

        fred = Fred(api_key=self.api_key)
        result = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        # 각 시리즈 최신값 가져오기
        for key, series_id in SERIES.items():
            try:
                data = fred.get_series(series_id, observation_start="2020-01-01")
                if data is not None and not data.empty:
                    latest_val = float(data.dropna().iloc[-1])
                    latest_date = str(data.dropna().index[-1].date())
                    result[f"{key}_value"] = round(latest_val, 2)
                    result[f"{key}_date"] = latest_date

                    # 전년대비 변화율 계산 (CPI, PCE는 지수 → YoY%)
                    if key in ("cpi", "core_pce") and len(data.dropna()) >= 13:
                        vals = data.dropna()
                        current = float(vals.iloc[-1])
                        year_ago = float(vals.iloc[-13])  # 12개월 전
                        if year_ago > 0:
                            yoy_pct = ((current / year_ago) - 1) * 100
                            result[f"{key}_yoy"] = round(yoy_pct, 2)
                    elif key == "unemployment":
                        result["unemployment_rate"] = round(latest_val, 1)
                    elif key == "fed_funds":
                        result["fed_funds_rate"] = round(latest_val, 2)
                else:
                    logger.warning(f"FRED {series_id}: 데이터 없음")
            except Exception as e:
                logger.warning(f"FRED {series_id} 실패: {e}")

        # CPI 추세 판단
        result["cpi_trend"] = self._detect_trend(fred, "CPIAUCSL")

        # 실업률 추세 판단
        result["unemployment_trend"] = self._detect_trend(fred, "UNRATE")

        # 스태그플레이션 판단
        stagflation = self._detect_stagflation(result)
        result.update(stagflation)

        # 저장
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"CPI 데이터 저장: {OUTPUT_PATH}")

        return result

    def _detect_trend(self, fred, series_id: str) -> str:
        """3개월 추세 판단: RISING / STABLE / FALLING."""
        try:
            data = fred.get_series(series_id, observation_start="2023-01-01")
            if data is None or len(data.dropna()) < 4:
                return "UNKNOWN"

            vals = data.dropna()
            recent_3 = vals.iloc[-3:].values

            if len(recent_3) < 3:
                return "UNKNOWN"

            # 3개월간 변화
            diff = recent_3[-1] - recent_3[0]

            if series_id == "CPIAUCSL":
                # CPI 지수는 절대값이 크므로 변화율로 판단
                pct_change = (diff / recent_3[0]) * 100 if recent_3[0] > 0 else 0
                if pct_change > 0.3:
                    return "RISING"
                elif pct_change < -0.1:
                    return "FALLING"
                return "STABLE"
            else:
                # 실업률 등 %포인트 기준
                if diff > 0.2:
                    return "RISING"
                elif diff < -0.2:
                    return "FALLING"
                return "STABLE"
        except Exception:
            return "UNKNOWN"

    def _detect_stagflation(self, data: dict) -> dict:
        """스태그플레이션 시그널 판단."""
        cpi_yoy = data.get("cpi_yoy", 0)
        unemp_trend = data.get("unemployment_trend", "UNKNOWN")
        unemp_rate = data.get("unemployment_rate", 0)

        signal = "NONE"
        description = ""

        if cpi_yoy >= 5.0 and unemp_rate >= 5.0:
            signal = "ALERT"
            description = f"CPI {cpi_yoy}% + 실업률 {unemp_rate}% → 스태그플레이션 경보"
        elif cpi_yoy >= 4.0 and unemp_trend == "RISING":
            signal = "WARNING"
            description = f"CPI {cpi_yoy}% + 실업률 상승 추세 → 스태그플레이션 경고"
        elif cpi_yoy >= 4.0:
            signal = "INFLATION_HIGH"
            description = f"CPI {cpi_yoy}% → 고인플레이션 (실업률 안정)"
        elif cpi_yoy <= 2.5 and unemp_trend != "RISING":
            signal = "DISINFLATION"
            description = f"CPI {cpi_yoy}% 안정 + 실업률 안정 → 긍정적"

        return {
            "stagflation_signal": signal,
            "stagflation_description": description,
        }

    def _load_existing(self) -> dict:
        """기존 저장된 데이터 로드."""
        if OUTPUT_PATH.exists():
            try:
                return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"updated_at": "N/A", "stagflation_signal": "UNKNOWN"}
