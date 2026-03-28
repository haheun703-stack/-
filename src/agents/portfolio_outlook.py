"""PortfolioOutlook — 보유 종목 방향성 판단 에이전트

장마감 후(15:30~) 보유 종목마다 "내일 오를까? 내릴까?" 종합 판단.
  - 기술적 추세 (SAR, RSI, MACD, MA, 볼린저)
  - 촉매 상태 (Perplexity 실시간 + AI Brain 캐시)
  - 시장 컨텍스트 (US Overnight, KOSPI 레짐, VIX)
  - 수급 (외인·기관 흐름)
  - 현재 수익률 + thesis 진행도

결과:
  방향: ↑상승 / →횡보 / ↓하락
  판단: HOLD / ADD(추가매수) / TRIM(일부매도) / SELL(전량매도)
  근거: 1줄 핵심 사유

사용:
  import asyncio
  from src.agents.portfolio_outlook import PortfolioOutlook
  outlook = PortfolioOutlook()
  results = asyncio.run(outlook.analyze_all())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# ── 종목-섹터 매핑 (보유 종목용, 수동) ──
TICKER_SECTOR = {
    "005380": "자동차", "006800": "증권", "010140": "조선",
    "010950": "정유", "024060": "정유", "042700": "반도체",
    "064350": "방산", "068270": "바이오", "218410": "방산",
    "272210": "방산",  # 한화시스템
}


def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("portfolio_outlook", {})
    except (OSError, yaml.YAMLError) as e:
        logger.warning("portfolio_outlook 설정 로드 실패: %s", e)
        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시스템 프롬프트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """\
당신은 한국 주식 포트폴리오 방향성 판단 전문가입니다.
보유 종목의 **향후 1~3일** 방향을 종합 판단합니다.

## 판단 프레임워크

### 1. 방향성 (direction)
- "↑" : 상승 전망 (기술적 상승추세 + 촉매 유효 + 수급 양호)
- "→" : 횡보 예상 (혼재된 신호, 관망)
- "↓" : 하락 우려 (기술적 하락추세 + 촉매 약화 + 수급 이탈)

### 2. 행동 (action)
- HOLD : 현재 포지션 유지 (thesis 유효, 추세 안정)
- ADD  : 추가 매수 검토 (눌림목 + 촉매 강화)
- TRIM : 일부 매도 (50%) — 불확실성 증가, 목표가 근접
- SELL : 전량 매도 — thesis 소멸 + 기술적 이탈

### 3. 핵심 원칙 (S-Oil 교훈)
- **촉매가 살아있으면 기계적 매도 금지**
- 손실 중이라도 thesis 유효하면 HOLD 가능
- 단기 기술적 하락 ≠ thesis 소멸 (구분 필수)
- 수익 실현은 thesis 진행도 기준 (목표가 대비 %)

### 3.5. 강제 매도 규칙 (HARD RULE — 반드시 적용)
- **손절가 근접 + 수급 약 → HOLD 금지**
  - 현재가가 손절가까지 -5% 이내 AND 섹터 수급 순매도 → TRIM 또는 SELL
  - 촉매 DEAD + 손절가 근접 → SELL (예외 없음)
  - 촉매 ALIVE라도 수급이 5일 연속 순매도(-1000억+) → TRIM
- **수급 패턴 F(물림) → SELL 강력 검토**
  - F패턴 = 스마트머니(외인+기관) 이탈 + 개인이 받는 중 → 가장 위험한 수급 형태
  - F + 촉매 DEAD → SELL 확정
  - F + 촉매 ALIVE → TRIM (스마트머니가 빠지는 이유가 있음)
- **수급 패턴별 행동 가이드**
  - A(스텔스매집): 외인 조용히 매집 + 가격 횡보 → ADD 검토 (최고 기회)
  - B(스마트머니합류): 외인+기관 동반 매수 → HOLD/ADD
  - C(추세확인): 추세 매수세 → HOLD
  - D(초기전환): 방향 전환 초기 → HOLD (관찰)
  - F(물림): 스마트머니 이탈 → TRIM/SELL (위 HARD RULE 참조)
- **수급 점수 판단 기준**
  - 섹터 외인+기관 합산 5일 순매수 > 0 → 수급 양호
  - 섹터 외인+기관 합산 5일 순매도 < -500억 → 수급 약화
  - 섹터 외인+기관 합산 5일 순매도 < -2000억 → 수급 이탈

### 4. 시장 컨텍스트 반영
- US Overnight bearish + VIX 스파이크 → 전체적 보수적 판단
- KOSPI 레짐 BEAR/CRISIS → ADD 자제, TRIM 적극
- 섹터별 US 시그널 참고 (해당 섹터 약세면 주의)

반드시 JSON 배열로 응답하세요. 각 원소:
{
  "ticker": "005380",
  "direction": "↑",
  "confidence": 70,
  "action": "HOLD",
  "reason": "1줄 핵심 사유",
  "catalyst_status": "ALIVE|FADING|DEAD|UNKNOWN",
  "key_risk": "핵심 리스크 1줄",
  "target_note": "목표가 대비 진행도 또는 지지선"
}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PortfolioOutlook Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PortfolioOutlook(BaseAgent):
    """보유 종목 전체 방향성 판단."""

    def __init__(self):
        settings = _load_settings()
        model = settings.get("model", "claude-sonnet-4-5-20250929")
        super().__init__(model=model)
        self.settings = settings
        self.perplexity_enabled = settings.get("perplexity_enabled", True)

    # ── 메인 분석 ──

    async def analyze_all(self) -> list[dict]:
        """보유 종목 전체 분석. [{ticker, name, direction, action, ...}]"""
        holdings = self._load_holdings()
        if not holdings:
            logger.warning("[Outlook] 보유 종목 없음")
            return []

        logger.info("[Outlook] %d종목 방향성 분석 시작", len(holdings))

        # 데이터 수집
        market_ctx = self._gather_market_context()
        stock_data = []
        for h in holdings:
            ticker = h["ticker"]
            data = self._gather_stock_data(ticker, h)
            stock_data.append(data)

        # Claude 종합 판단 (전 종목 한 번에)
        prompt = self._build_prompt(stock_data, market_ctx)

        try:
            text = await self._ask_claude(SYSTEM_PROMPT, prompt, max_tokens=8000)
            results = self._parse_results(text)
        except Exception as e:
            logger.error("[Outlook] Claude 분석 실패: %s", e)
            results = [{"ticker": h["ticker"], "name": h["name"],
                         "direction": "→", "action": "HOLD",
                         "reason": f"AI 분석 실패: {e}",
                         "confidence": 0} for h in holdings]

        # 보유 정보 + SD V2 병합
        sd_map = {s["ticker"]: s.get("sd_v2") for s in stock_data}
        for r in results:
            ticker = r.get("ticker", "")
            for h in holdings:
                if h["ticker"] == ticker:
                    r["name"] = h["name"]
                    r["shares"] = h["shares"]
                    r["entry_price"] = h["entry_price"]
                    r["current_price"] = h["current_price"]
                    r["pnl_pct"] = round(
                        (h["current_price"] - h["entry_price"]) / h["entry_price"] * 100, 1
                    ) if h["entry_price"] > 0 else 0
                    break
            # SD V2 수급 패턴 병합
            if ticker in sd_map and sd_map[ticker]:
                r["sd_v2"] = sd_map[ticker]

        logger.info("[Outlook] 분석 완료: %d종목", len(results))
        return results

    # ── 데이터 수집 ──

    def _load_holdings(self) -> list[dict]:
        """positions.json에서 보유 종목 로드."""
        pos_path = DATA_DIR / "positions.json"
        if not pos_path.exists():
            return []
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        # positions.json은 리스트 or {"positions": [...]}
        if isinstance(data, list):
            return data
        return data.get("positions", [])

    def _gather_stock_data(self, ticker: str, holding: dict) -> dict:
        """개별 종목 분석 데이터 수집."""
        data = {
            "ticker": ticker,
            "name": holding.get("name", ticker),
            "entry_price": holding.get("entry_price", 0),
            "current_price": holding.get("current_price", 0),
            "shares": holding.get("shares", 0),
            "entry_date": holding.get("entry_date", ""),
            "target_price": holding.get("target_price", 0),
            "stop_loss": holding.get("stop_loss", 0),
            "sector": TICKER_SECTOR.get(ticker, "기타"),
        }

        # 수익률
        if data["entry_price"] > 0:
            data["pnl_pct"] = round(
                (data["current_price"] - data["entry_price"]) / data["entry_price"] * 100, 1
            )
        else:
            data["pnl_pct"] = 0

        # 목표가 진행도
        if data["target_price"] > data["entry_price"] > 0:
            total_move = data["target_price"] - data["entry_price"]
            current_move = data["current_price"] - data["entry_price"]
            data["target_progress_pct"] = round(current_move / total_move * 100, 0)
        else:
            data["target_progress_pct"] = 0

        # 기술적 지표 (parquet)
        data.update(self._get_technicals(ticker))

        # 촉매 (Perplexity)
        if self.perplexity_enabled:
            data["catalyst"] = self._get_catalyst(ticker, data["name"])
        else:
            data["catalyst"] = None

        # AI Brain 캐시
        data["ai_brain"] = self._get_ai_brain(ticker)

        # 섹터 수급
        data["sector_flow"] = self._get_sector_flow(data["sector"])

        # SD V2 개별 종목 수급 패턴
        data["sd_v2"] = self._get_sd_v2(ticker)

        # 손절가 근접도
        if data["stop_loss"] > 0 and data["current_price"] > 0:
            dist_to_sl = (data["current_price"] - data["stop_loss"]) / data["current_price"] * 100
            data["stop_loss_distance_pct"] = round(dist_to_sl, 1)
        else:
            data["stop_loss_distance_pct"] = 999

        return data

    def _get_technicals(self, ticker: str) -> dict:
        """parquet에서 최근 기술 지표."""
        try:
            import pandas as pd
            parquet_path = DATA_DIR / "processed" / f"{ticker}.parquet"
            if not parquet_path.exists():
                return {"tech_available": False}

            df = pd.read_parquet(parquet_path)
            if df.empty:
                return {"tech_available": False}

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            # 최근 5일 추세
            recent_5 = df.tail(5)
            trend_5d = "상승" if recent_5["close"].iloc[-1] > recent_5["close"].iloc[0] else "하락"
            change_5d = round(
                (recent_5["close"].iloc[-1] - recent_5["close"].iloc[0]) / recent_5["close"].iloc[0] * 100, 1
            )

            # MACD 방향
            macd_val = float(last.get("macd", 0))
            macd_sig = float(last.get("macd_signal", 0))
            macd_prev = float(prev.get("macd", 0))
            if macd_val > macd_sig:
                macd_dir = "골든크로스" if macd_prev <= float(prev.get("macd_signal", 0)) else "상향"
            else:
                macd_dir = "데드크로스" if macd_prev >= float(prev.get("macd_signal", 0)) else "하향"

            return {
                "tech_available": True,
                "close": float(last.get("close", 0)),
                "rsi": round(float(last.get("rsi_14", 50)), 1),
                "adx": round(float(last.get("adx", 0)), 1),
                "macd": round(macd_val, 2),
                "macd_signal": round(macd_sig, 2),
                "macd_direction": macd_dir,
                "sar_trend": int(last.get("sar_trend", 0)),
                "sar_trend_str": "상승" if int(last.get("sar_trend", 0)) == 1 else "하락",
                "ma_20": round(float(last.get("ma_20", 0)), 0),
                "ma_60": round(float(last.get("ma_60", 0)), 0),
                "bb_pct": round(float(last.get("bb_pct", 50)), 1),
                "vol_ratio": round(float(last.get("vol_ratio", 1.0)), 2),
                "trend_5d": trend_5d,
                "change_5d": change_5d,
            }
        except Exception as e:
            logger.warning("[Tech] %s 지표 실패: %s", ticker, e)
            return {"tech_available": False}

    def _get_catalyst(self, ticker: str, name: str) -> dict | None:
        """Perplexity로 촉매 상태 조회."""
        try:
            from src.agents.perplexity_verifier import PerplexityVerifier
            verifier = PerplexityVerifier(max_verifications=1, max_thesis_checks=0)
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = f"""오늘 날짜: {today}
{name}({ticker}) 종목의 최신 뉴스, 촉매, 리스크를 검색하세요.
JSON으로 응답:
{{
  "catalyst_status": "ALIVE|FADING|DEAD",
  "catalyst_summary": "핵심 촉매 1줄",
  "risk_summary": "핵심 리스크 1줄",
  "sentiment": "POSITIVE|NEUTRAL|NEGATIVE",
  "upcoming_event": "가까운 이벤트 (있으면)"
}}"""
            return verifier._query(prompt)
        except Exception as e:
            logger.warning("[Catalyst] %s 조회 실패: %s", name, e)
            return None

    def _get_ai_brain(self, ticker: str) -> dict | None:
        """ai_brain_judgment.json에서 기존 판정."""
        path = DATA_DIR / "ai_brain_judgment.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for buy in data.get("buys", []):
                if buy.get("ticker") == ticker:
                    return {
                        "action": buy.get("action"),
                        "confidence": buy.get("confidence"),
                        "reasoning": buy.get("reasoning"),
                    }
        except Exception:
            pass
        return None

    def _get_sd_v2(self, ticker: str) -> dict | None:
        """parquet에서 SD V2 수급 패턴 계산."""
        try:
            import pandas as pd
            from src.alpha.factors.sd_score_v2 import compute_sd_features, format_sd_summary

            parquet_path = DATA_DIR / "processed" / f"{ticker}.parquet"
            if not parquet_path.exists():
                return None

            df = pd.read_parquet(parquet_path)
            if len(df) < 20:
                return None

            idx = len(df) - 1
            feat = compute_sd_features(df, idx, ticker)
            return {
                "pattern": feat.pattern,
                "pattern_name": feat.pattern_name,
                "sd_score": round(feat.sd_score, 2),
                "foreign_net_20d": round(feat.foreign_net_20d, 0),
                "inst_net_20d": round(feat.inst_net_20d, 0),
                "individual_net_20d": round(feat.individual_net_20d, 0),
                "summary": format_sd_summary(feat),
            }
        except Exception as e:
            logger.warning("[SD V2] %s 계산 실패: %s", ticker, e)
            return None

    def _get_sector_flow(self, sector: str) -> dict | None:
        """investor_flow.json에서 섹터별 수급 데이터."""
        try:
            path = DATA_DIR / "sector_rotation" / "investor_flow.json"
            if not path.exists():
                return None
            with open(path, encoding="utf-8") as f:
                flow = json.load(f)

            # 섹터명 매핑 (보유 종목 섹터 → investor_flow 섹터명)
            sector_alias = {
                "자동차": "현대차그룹", "증권": "증권", "조선": "조선",
                "정유": "에너지화학", "반도체": "반도체", "방산": "방산",
                "바이오": "바이오",
            }
            flow_sector = sector_alias.get(sector, sector)

            for s in flow.get("sectors", []):
                if s.get("sector") == flow_sector:
                    total = s.get("foreign_cum_bil", 0) + s.get("inst_cum_bil", 0)
                    status = "양호" if total > 0 else "약화" if total > -2000 else "이탈"
                    return {
                        "sector": flow_sector,
                        "total_flow_bil": round(total, 0),
                        "foreign_cum": round(s.get("foreign_cum_bil", 0), 0),
                        "inst_cum": round(s.get("inst_cum_bil", 0), 0),
                        "status": status,
                    }
        except Exception as e:
            logger.warning("[Flow] %s 수급 실패: %s", sector, e)
        return None

    def _gather_market_context(self) -> dict:
        """시장 전체 컨텍스트 수집."""
        ctx = {}

        # 1) US Overnight Signal
        try:
            path = DATA_DIR / "us_market" / "overnight_signal.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    us = json.load(f)
                ctx["us_overnight"] = {
                    "composite": us.get("composite", "unknown"),
                    "score": us.get("score", 0),
                    "vix_level": us.get("vix", {}).get("level", 0),
                    "vix_status": us.get("vix", {}).get("status", ""),
                    "ewy_5d": us.get("index_direction", {}).get("EWY", {}).get("ret_5d", 0),
                    "special_rules": [r.get("rule") for r in us.get("special_rules_triggered", [])],
                }
                # 섹터별 시그널
                sector_signals = {}
                for sec, val in us.get("sector_signals", {}).items():
                    if val.get("signal") != "neutral":
                        sector_signals[sec] = val
                ctx["us_sector_alerts"] = sector_signals
        except Exception as e:
            logger.warning("[Market] US Overnight 로드 실패: %s", e)

        # 2) KOSPI 레짐
        try:
            path = DATA_DIR / "regime_macro_signal.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    regime = json.load(f)
                ctx["kospi_regime"] = regime.get("regime", "UNKNOWN")
                ctx["kospi_regime_detail"] = regime.get("detail", "")
        except Exception:
            pass

        # 3) AI Brain 시장 판단
        try:
            path = DATA_DIR / "ai_brain_judgment.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    brain = json.load(f)
                ctx["market_sentiment"] = brain.get("market_sentiment", "")
                ctx["sector_outlook"] = brain.get("sector_outlook", {})
        except Exception:
            pass

        return ctx

    # ── 프롬프트 구성 ──

    def _build_prompt(self, stocks: list[dict], market: dict) -> str:
        """Claude 분석 프롬프트 구성."""
        lines = [
            f"## 포트폴리오 방향성 분석 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            "",
            "### 시장 컨텍스트",
        ]

        # US Overnight
        us = market.get("us_overnight", {})
        if us:
            lines.append(f"- US Overnight: {us.get('composite', '?')} (점수 {us.get('score', 0):.2f})")
            lines.append(f"- VIX: {us.get('vix_level', 0)} ({us.get('vix_status', '')})")
            lines.append(f"- EWY 5일: {us.get('ewy_5d', 0):+.1f}%")
            rules = us.get("special_rules", [])
            if rules:
                lines.append(f"- 특수 룰: {', '.join(rules)}")

        # KOSPI 레짐
        regime = market.get("kospi_regime", "")
        if regime:
            lines.append(f"- KOSPI 레짐: {regime}")

        # 시장 심리
        sentiment = market.get("market_sentiment", "")
        if sentiment:
            lines.append(f"- AI Brain 시장 심리: {sentiment}")

        # 섹터 전망
        outlook = market.get("sector_outlook", {})
        if outlook:
            outlook_str = ", ".join(f"{k}:{v}" for k, v in list(outlook.items())[:5])
            lines.append(f"- 섹터 전망: {outlook_str}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # 개별 종목
        for s in stocks:
            lines.append(f"### {s['name']}({s['ticker']}) — {s.get('sector', '기타')}")
            lines.append(f"- 보유: {s['shares']}주, 평단 {s['entry_price']:,.0f}원")
            lines.append(f"- 현재가: {s['current_price']:,.0f}원 ({s['pnl_pct']:+.1f}%)")
            lines.append(f"- 목표가: {s['target_price']:,.0f}원 (진행도 {s['target_progress_pct']:.0f}%)")
            lines.append(f"- 손절가: {s['stop_loss']:,.0f}원")

            if s.get("tech_available"):
                lines.append(f"- 기술: SAR {s.get('sar_trend_str', '?')} | RSI {s.get('rsi', 50)} | "
                             f"ADX {s.get('adx', 0)} | MACD {s.get('macd_direction', '?')}")
                lines.append(f"  BB% {s.get('bb_pct', 50)} | MA20 {s.get('ma_20', 0):,.0f} | "
                             f"거래량비 {s.get('vol_ratio', 1.0)}")
                lines.append(f"  5일 추세: {s.get('trend_5d', '?')} ({s.get('change_5d', 0):+.1f}%)")
            else:
                lines.append("- 기술: parquet 데이터 없음")

            # 손절가 근접도 + 수급
            sl_dist = s.get("stop_loss_distance_pct", 999)
            if sl_dist < 5:
                lines.append(f"  ⚠ 손절가 근접: 현재가→손절가 {sl_dist:+.1f}%")

            # SD V2 개별 종목 수급 패턴 (핵심)
            sd = s.get("sd_v2")
            if sd:
                lines.append(f"- 수급 패턴: {sd['pattern']}({sd['pattern_name']}) SD={sd['sd_score']}")
                lines.append(f"  외인 {sd['foreign_net_20d']:+,.0f}억 | "
                             f"기관 {sd['inst_net_20d']:+,.0f}억 | "
                             f"개인 {sd['individual_net_20d']:+,.0f}억 (20일)")
                if sd["pattern"] == "F":
                    lines.append(f"  ⚠⚠ 물림 패턴(F) — 스마트머니 이탈, 개인이 받는 중! SELL 강력 검토!")

            sf = s.get("sector_flow")
            if sf:
                lines.append(f"- 섹터 수급({sf['sector']}): {sf['total_flow_bil']:+,.0f}억 "
                             f"(외인 {sf['foreign_cum']:+,.0f}, 기관 {sf['inst_cum']:+,.0f}) "
                             f"[{sf['status']}]")

            # 촉매
            cat = s.get("catalyst")
            if cat:
                lines.append(f"- 촉매: [{cat.get('catalyst_status', '?')}] {cat.get('catalyst_summary', '')}")
                if cat.get("risk_summary"):
                    lines.append(f"  리스크: {cat.get('risk_summary', '')}")
                if cat.get("upcoming_event"):
                    lines.append(f"  이벤트: {cat.get('upcoming_event', '')}")
            else:
                lines.append("- 촉매: 미확인")

            # AI Brain
            brain = s.get("ai_brain")
            if brain:
                lines.append(f"- AI Brain: {brain.get('action', '')} — {brain.get('reasoning', '')[:80]}")

            # 섹터 US 시그널
            sec = s.get("sector", "")
            us_sec = market.get("us_sector_alerts", {}).get(sec)
            if us_sec:
                lines.append(f"- US 섹터 시그널: {us_sec.get('signal', '')} ({us_sec.get('driver', '')})")

            lines.append("")

        lines.append("---")
        lines.append("위 데이터를 종합하여 각 종목의 1~3일 방향성을 판단하세요.")
        lines.append("JSON 배열로 응답하세요.")

        return "\n".join(lines)

    # ── 결과 파싱 ──

    def _parse_results(self, text: str) -> list[dict]:
        """Claude 응답에서 JSON 배열 추출."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning("[Outlook] JSON 파싱 실패: %s — 원문: %s", e, text[:200])
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed
