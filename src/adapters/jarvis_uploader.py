"""JARVIS 대시보드 연동 — ppwangga.com 자동 업로드

BAT-D/BAT-A에서 호출하여 일일 리포트 + 메트릭스 + 시장 데이터를 대시보드에 업로드.

v2.0: 추천종목 + US Overnight + KOSPI 레짐 업로드 추가
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

    def update_market_data(self, data: dict) -> dict:
        """대시보드 시장 데이터 업데이트 (추천종목, US시그널, 레짐)."""
        if not self.is_available:
            return {"error": "API key not set"}

        url = f"{self.base_url}/api/market"

        try:
            resp = requests.post(url, headers=self.headers, json=data, timeout=30)
            resp.raise_for_status()
            logger.info("[JARVIS] 시장 데이터 업데이트 완료")
            return resp.json()

        except Exception as e:
            logger.warning(f"[JARVIS] 시장 데이터 업데이트 실패: {e}")
            return {"error": str(e)}

    def update_brain(self, data: dict) -> dict:
        """대시보드 AI Brain 데이터 업데이트."""
        if not self.is_available:
            return {"error": "API key not set"}

        url = f"{self.base_url}/api/brain"

        try:
            resp = requests.post(url, headers=self.headers, json=data, timeout=30)
            resp.raise_for_status()
            logger.info("[JARVIS] AI Brain 데이터 업데이트 완료")
            return resp.json()

        except Exception as e:
            logger.warning(f"[JARVIS] AI Brain 업데이트 실패: {e}")
            return {"error": str(e)}

    def update_holdings(self, data: dict) -> dict:
        """대시보드 보유주식 업데이트."""
        if not self.is_available:
            return {"error": "API key not set"}

        url = f"{self.base_url}/api/holdings"

        try:
            resp = requests.post(url, headers=self.headers, json=data, timeout=30)
            resp.raise_for_status()
            logger.info(f"[JARVIS] 보유주식 업데이트 완료 ({len(data.get('holdings', []))}종목)")
            return resp.json()

        except Exception as e:
            logger.warning(f"[JARVIS] 보유주식 업데이트 실패: {e}")
            return {"error": str(e)}

    def upload_daily_auto(self):
        """BAT-D/BAT-A 자동 호출용 — HTML 보고서 + 메트릭스 + 시장 데이터 + 보유주식 일괄 업로드."""
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

        # 2) 메트릭스 업데이트 (포트폴리오 + 성과)
        metrics = self._build_metrics()
        if metrics:
            r = self.update_metrics(**metrics)
            results.append(f"Metrics: {r.get('status', r.get('error', '?'))}")

        # 3) 시장 데이터 업데이트 (추천종목 + US시그널 + 레짐)
        market = self._build_market_data()
        if market:
            r = self.update_market_data(market)
            results.append(f"Market: {r.get('status', r.get('error', '?'))}")

        # 4) 보유주식 업데이트 (KIS 잔고)
        holdings = self._build_holdings()
        if holdings:
            r = self.update_holdings(holdings)
            results.append(f"Holdings: {r.get('status', r.get('error', '?'))}")

        # 5) AI Brain 데이터 업데이트 (전략/섹터/릴레이/뉴스/v3)
        brain = self._build_brain_data()
        if brain:
            r = self.update_brain(brain)
            results.append(f"Brain: {r.get('status', r.get('error', '?'))}")

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

    def _build_market_data(self) -> dict:
        """추천종목 + US Overnight + KOSPI 레짐 데이터 빌드."""
        market = {}

        # 1) 추천종목 (tomorrow_picks.json) — raw 전체 필드 + 호환 alias
        picks_path = DATA_DIR / "tomorrow_picks.json"
        if picks_path.exists():
            try:
                raw = json.loads(picks_path.read_text(encoding="utf-8"))
                picks_list = raw if isinstance(raw, list) else raw.get("picks", [])
                picks_stats = raw.get("stats", {}) if isinstance(raw, dict) else {}
                enriched = []
                for p in picks_list[:10]:
                    pick = dict(p)  # raw 전체 복사
                    # 템플릿 호환 alias 추가
                    pick.setdefault("code", pick.get("ticker", ""))
                    pick.setdefault("score", pick.get("total_score", 0))
                    pick.setdefault("signals", pick.get("sources", []))
                    enriched.append(pick)
                market["picks"] = enriched
                if picks_stats:
                    market["picks_stats"] = picks_stats
                market["picks_date_label"] = raw.get("date", "") if isinstance(raw, dict) else ""
            except Exception as e:
                logger.warning(f"[JARVIS] 추천종목 로드 실패: {e}")

        # 2) US Overnight Signal (data/us_market/overnight_signal.json)
        signal_path = DATA_DIR / "us_market" / "overnight_signal.json"
        if signal_path.exists():
            try:
                sig = json.loads(signal_path.read_text(encoding="utf-8"))
                idx = sig.get("index_direction", {})
                vix = sig.get("vix", {})
                kills = sig.get("sector_kills", {})
                killed_sectors = [k for k, v in kills.items() if v.get("killed")]
                market["us_signal"] = {
                    "grade": sig.get("grade", "-"),
                    "score": sig.get("combined_score_100", sig.get("l1_score_100", 0)),
                    "spy_return": idx.get("SPY", {}).get("ret_1d", 0),
                    "qqq_return": idx.get("QQQ", {}).get("ret_1d", 0),
                    "vix_return": vix.get("level", 0),
                    "special_rules": [
                        r.get("name", str(r)) if isinstance(r, dict) else str(r)
                        for r in sig.get("special_rules", [])
                    ],
                    "kill_sectors": killed_sectors,
                }
            except Exception as e:
                logger.warning(f"[JARVIS] US 시그널 로드 실패: {e}")

        # 3) KOSPI 레짐
        regime_map = {"BULL": 5, "CAUTION": 3, "BEAR": 2, "CRISIS": 0}
        kospi_path = DATA_DIR / "kospi_index.csv"
        if kospi_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(kospi_path, parse_dates=["Date"])
                # 컬럼명 대소문자 호환 (close / Close)
                col = "Close" if "Close" in df.columns else "close"
                if len(df) >= 60:
                    close = df[col].iloc[-1]
                    ma20 = df[col].iloc[-20:].mean()
                    ma60 = df[col].iloc[-60:].mean()
                    rv20 = df[col].pct_change().iloc[-20:].std() * (252 ** 0.5)
                    rv_median = df[col].pct_change().rolling(20).std().iloc[-252:].median() * (252 ** 0.5) if len(df) >= 272 else 0.2

                    if close > ma20 and rv20 < rv_median:
                        regime = "BULL"
                    elif close > ma20:
                        regime = "CAUTION"
                    elif close > ma60:
                        regime = "BEAR"
                    else:
                        regime = "CRISIS"

                    market["regime"] = regime
                    market["regime_slots"] = regime_map.get(regime, 0)
            except Exception as e:
                logger.warning(f"[JARVIS] KOSPI 레짐 계산 실패: {e}")

        return market

    def _build_brain_data(self) -> dict:
        """AI Brain 관련 데이터 통합 빌드."""
        brain = {}

        # 1) AI 전략 분석
        path = DATA_DIR / "ai_strategic_analysis.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                brain["strategic"] = {
                    "regime": d.get("regime", "-"),
                    "confidence": d.get("regime_confidence", 0),
                    "thesis": [
                        {"sector": t["sector"], "thesis": t["thesis"][:80], "confidence": t.get("confidence", 0)}
                        for t in d.get("industry_thesis", [])[:5]
                    ],
                    "sector_priority": d.get("sector_priority", {}),
                    "risk_factors": d.get("risk_factors", []),
                    "max_new_buys": d.get("max_new_buys", 0),
                }
            except Exception as e:
                logger.warning(f"[JARVIS] AI 전략 로드 실패: {e}")

        # 2) 섹터 집중
        path = DATA_DIR / "ai_sector_focus.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                brain["sector_focus"] = {
                    "focus_sectors": [f["sector"] for f in d.get("focus_sectors", [])[:6]],
                    "boost": d.get("screening_boost", [])[:6],
                    "suppress": d.get("screening_suppress", [])[:6],
                    "warnings": d.get("sector_warnings", []),
                }
            except Exception as e:
                logger.warning(f"[JARVIS] 섹터 집중 로드 실패: {e}")

        # 3) 그룹 릴레이
        path = DATA_DIR / "group_relay" / "group_relay_today.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                fired = []
                for g in d.get("fired_groups", [])[:3]:
                    subs = [
                        {"name": s["name"], "change": s.get("change_pct", 0), "score": s.get("score", 0)}
                        for s in g.get("waiting_subsidiaries", [])[:5]
                    ]
                    fired.append({
                        "group": g.get("group", ""),
                        "leader": g.get("leader_name", ""),
                        "leader_change": g.get("leader_change", 0),
                        "subsidiaries": subs,
                    })
                brain["group_relay"] = {
                    "fired": fired,
                    "summary": d.get("summary", ""),
                }
            except Exception as e:
                logger.warning(f"[JARVIS] 그룹 릴레이 로드 실패: {e}")

        # 4) AI 뉴스 판단
        path = DATA_DIR / "ai_brain_judgment.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                outlook = {}
                for sec, info in d.get("sector_outlook", {}).items():
                    if isinstance(info, dict):
                        outlook[sec] = {"dir": info.get("direction", ""), "reason": info.get("reason", "")[:60]}
                    else:
                        outlook[sec] = {"dir": "", "reason": str(info)[:60]}
                brain["news"] = {
                    "sentiment": d.get("market_sentiment", "-"),
                    "themes": d.get("key_themes", [])[:5],
                    "sector_outlook": dict(list(outlook.items())[:8]),
                    "date": d.get("date", ""),
                }
            except Exception as e:
                logger.warning(f"[JARVIS] 뉴스 판단 로드 실패: {e}")

        # 5) v3 AI Picks
        path = DATA_DIR / "ai_v3_picks.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                buys = []
                for b in d.get("buys", [])[:5]:
                    buys.append({
                        "name": b.get("name", b.get("stock_name", "?")),
                        "ticker": b.get("ticker", ""),
                        "conviction": b.get("conviction", 0),
                        "strategy": b.get("strategy", ""),
                        "thesis_alignment": b.get("thesis_alignment", ""),
                    })
                brain["v3_picks"] = {
                    "regime": d.get("regime", ""),
                    "slots": d.get("available_slots", 0),
                    "buys": buys,
                }
            except Exception as e:
                logger.warning(f"[JARVIS] v3 Picks 로드 실패: {e}")

        return brain

    def _build_holdings(self) -> dict:
        """KIS 잔고에서 보유주식 데이터 빌드."""
        balance_path = DATA_DIR / "kis_balance.json"
        if not balance_path.exists():
            return {}

        try:
            bal = json.loads(balance_path.read_text(encoding="utf-8"))
            return {
                "holdings": bal.get("holdings", []),
                "total_eval": bal.get("total_eval", 0),
                "total_pnl": bal.get("total_pnl", 0),
                "available_cash": bal.get("available_cash", 0),
                "fetched_at": bal.get("fetched_at", ""),
            }
        except Exception as e:
            logger.warning(f"[JARVIS] 보유주식 로드 실패: {e}")
            return {}


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
