"""JARVIS 대시보드 연동 — ppwangga.com 자동 업로드

BAT-D/BAT-A에서 호출하여 일일 리포트 + 메트릭스를 대시보드에 업로드.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class JarvisUploader:
    def __init__(self):
        self.base_url = os.getenv("JARVIS_URL", "https://www.ppwangga.com")
        self.api_key = os.getenv("JARVIS_API_KEY", "")
        self.headers = {"X-API-Key": self.api_key}

        if not self.api_key:
            logger.warning("JARVIS_API_KEY 미설정 — 대시보드 업로드 비활성")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def upload_report(self, date: str = None, filepath: str = None, html: str = None) -> dict:
        """일별 HTML 리포트 업로드."""
        if not self.is_available:
            return {"error": "API key not set"}

        date = date or datetime.now().strftime("%Y-%m-%d")
        url = f"{self.base_url}/api/upload"

        try:
            if filepath:
                with open(filepath, "rb") as f:
                    resp = requests.post(
                        url, headers=self.headers,
                        files={"file": f}, data={"date": date}, timeout=30,
                    )
            elif html:
                resp = requests.post(
                    url, headers=self.headers,
                    json={"date": date, "html": html}, timeout=30,
                )
            else:
                return {"error": "filepath 또는 html 필요"}

            resp.raise_for_status()
            logger.info(f"[JARVIS] {date} 리포트 업로드 완료")
            return resp.json()

        except Exception as e:
            logger.warning(f"[JARVIS] 리포트 업로드 실패: {e}")
            return {"error": str(e)}

    def update_metrics(self, **kwargs) -> dict:
        """대시보드 메트릭스 업데이트."""
        if not self.is_available:
            return {"error": "API key not set"}

        url = f"{self.base_url}/api/metrics"

        try:
            resp = requests.post(url, headers=self.headers, json=kwargs, timeout=30)
            resp.raise_for_status()
            logger.info("[JARVIS] 메트릭스 업데이트 완료")
            return resp.json()

        except Exception as e:
            logger.warning(f"[JARVIS] 메트릭스 업데이트 실패: {e}")
            return {"error": str(e)}

    def upload_daily_auto(self):
        """BAT-D/BAT-A 자동 호출용 — HTML 보고서 + 메트릭스 일괄 업로드."""
        today = datetime.now().strftime("%Y-%m-%d")
        results = []

        # 1) HTML 보고서 업로드 (장시작전 보고서)
        html_dir = Path(r"D:\클로드 HTML 보고서")
        html_files = sorted(html_dir.glob(f"*{today}*.html")) if html_dir.exists() else []
        if not html_files:
            # 날짜 없는 최신 파일
            html_files = sorted(html_dir.glob("*.html"), key=lambda p: p.stat().st_mtime)

        if html_files:
            latest = html_files[-1]
            r = self.upload_report(date=today, filepath=str(latest))
            results.append(f"HTML: {r.get('status', r.get('error', '?'))}")
            logger.info(f"[JARVIS] HTML 업로드: {latest.name}")

        # 2) 메트릭스 업데이트 (picks_history + 포트폴리오 데이터)
        metrics = self._build_metrics()
        if metrics:
            r = self.update_metrics(**metrics)
            results.append(f"Metrics: {r.get('status', r.get('error', '?'))}")

        return results

    def _build_metrics(self) -> dict:
        """현재 시스템 데이터에서 메트릭스 추출."""
        metrics = {}

        # KIS 잔고에서 포트폴리오 가치
        balance_path = DATA_DIR / "kis_balance.json"
        if balance_path.exists():
            try:
                bal = json.loads(balance_path.read_text(encoding="utf-8"))
                total_eval = sum(
                    int(h.get("eval_amount", 0))
                    for h in bal.get("holdings", [])
                )
                cash = int(bal.get("available_cash", 0))
                metrics["portfolio_value"] = total_eval + cash
            except Exception:
                pass

        # 추천 성과에서 승률/PF
        perf_path = DATA_DIR / "daily_performance.json"
        if perf_path.exists():
            try:
                perf = json.loads(perf_path.read_text(encoding="utf-8"))
                if isinstance(perf, list) and perf:
                    latest = perf[-1]
                    metrics["win_rate"] = latest.get("win_rate", 0)
                    metrics["pf"] = latest.get("profit_factor", 0)
                    metrics["total_trades"] = latest.get("total_trades", 0)
            except Exception:
                pass

        # 전략 정보
        metrics["strategy_a"] = {
            "name": "Quantum v10.3 + v3 Brain",
            "alloc": 60,
        }
        metrics["strategy_b"] = {
            "name": "ETF 3축 로테이션",
            "alloc": 40,
        }

        return metrics


def main():
    """CLI 호출용."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    jarvis = JarvisUploader()

    if not jarvis.is_available:
        print("[JARVIS] API 키 미설정 — .env의 JARVIS_API_KEY 확인")
        return

    results = jarvis.upload_daily_auto()
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
