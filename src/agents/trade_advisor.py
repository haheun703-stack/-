"""TradeAdvisor — 텔레그램 매수/매도 AI 판단 에이전트

텔레그램에서 "매수 삼성전자 10" / "매도 현대차" 입력 시
기존 즉시 주문 대신 AI 분석 → 판단 제시 → 사용자 최종 결정.

매수: SignalEngine + parquet 지표 + Perplexity 촉매 → BUY_OK / WAIT / SKIP
매도: SellBrain 캐시 + 촉매 상태 + 기술 지표 → SELL_OK / PARTIAL / HOLD
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _load_settings() -> dict:
    """settings.yaml의 trade_advisor 섹션 로드."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("trade_advisor", {})
    except Exception:
        return {}


# ─── 매수 분석 시스템 프롬프트 ───

SYSTEM_BUY = """\
당신은 한국 주식 매수 판단 전문가입니다.
사용자가 텔레그램으로 수동 매수 명령을 내렸습니다.
아래 데이터를 종합하여 매수 적합성을 빠르게 판단하세요.

## 판단 기준
- BUY_OK: 시그널 B등급 이상 + 트리거 1개+ + 촉매 확인 → "매수 괜찮음"
- WAIT: 조건 부분 충족 (과열 근접, 촉매 불확실) → "눌림 대기 추천"
- SKIP: 시그널 C등급 OR RSI>75 과열 OR 촉매 없음 → "매수 비추"

## 중요
- 과열 종목(RSI>75, BB>90)은 반드시 SKIP 또는 WAIT
- 촉매가 확인된 종목은 기술적 지표가 다소 애매해도 BUY_OK 가능
- 이미 보유 중이면 추가매수 리스크 경고
- 52주 고점 근접(3% 이내)은 WAIT 추천

반드시 JSON으로만 응답하세요."""

SYSTEM_SELL = """\
당신은 한국 주식 매도 판단 전문가입니다.
사용자가 텔레그램으로 수동 매도 명령을 내렸습니다.
아래 데이터를 종합하여 매도 적합성을 빠르게 판단하세요.

## 핵심 원칙 — S-Oil 교훈
촉매(catalyst)가 살아있는 종목을 기계적으로 매도하면
"내일 10% 더 오를 종목"을 놓칩니다. 촉매 우선 원칙.

## 판단 기준
- SELL_OK: thesis 소멸 + 기술적 이탈 + 수급 악화 → "매도 추천"
- PARTIAL: 불확실 OR 목표가 근접 → "50% 분할 매도 추천"
- HOLD: thesis 유효 + 촉매 ALIVE → "홀딩 추천"

## 중요
- 촉매 ALIVE + 수급 양호 → 함부로 매도하지 마라
- 촉매 DEAD + 기술적 이탈 → 적극 매도
- 촉매 FADING → 분할 매도(50%) 후 관찰
- 손실 중이라도 thesis 살아있으면 HOLD 가능

반드시 JSON으로만 응답하세요."""


class TradeAdvisor(BaseAgent):
    """텔레그램 매수/매도 AI 판단 에이전트."""

    def __init__(self):
        settings = _load_settings()
        model = settings.get("model", "claude-sonnet-4-5-20250929")
        super().__init__(model=model)
        self.settings = settings
        self.buy_thresholds = settings.get("buy_thresholds", {})
        self.sell_thresholds = settings.get("sell_thresholds", {})
        self.perplexity_enabled = settings.get("perplexity_enabled", True)
        self._signal_engine = None  # 지연 초기화 캐시

    # ═══════════════════════════════════════════
    # 매수 분석
    # ═══════════════════════════════════════════

    async def analyze_buy(
        self, ticker: str, qty: int, current_price: int = 0
    ) -> dict:
        """매수 AI 분석. BUY_OK / WAIT / SKIP 판정."""
        try:
            data = self._gather_buy_data(ticker, current_price)
            prompt = self._build_buy_prompt(ticker, qty, data)
            result = await self._ask_claude_json(SYSTEM_BUY, prompt)
            # 필수 필드 보정
            result.setdefault("verdict", "SKIP")
            result.setdefault("confidence", 50)
            result.setdefault("signal_grade", data.get("grade", "N/A"))
            result.setdefault("signal_score", data.get("score", 0))
            result.setdefault("technical_summary", "")
            result.setdefault("catalyst", "미확인")
            result.setdefault("risk_warning", None)
            result.setdefault("suggestion", "")
            return result
        except Exception as e:
            logger.error("TradeAdvisor.analyze_buy 실패: %s", e)
            return {"verdict": "ERROR", "error": str(e)}

    def _gather_buy_data(self, ticker: str, current_price: int) -> dict:
        """매수 분석에 필요한 데이터 수집 (동기)."""
        data: dict = {"ticker": ticker}

        # 1) SignalEngine — v8 파이프라인 결과
        try:
            data.update(self._get_signal_data(ticker))
        except Exception as e:
            logger.warning("SignalEngine 조회 실패 (%s): %s", ticker, e)
            data["grade"] = "N/A"
            data["score"] = 0

        # 2) parquet 최신 지표
        try:
            data.update(self._get_parquet_indicators(ticker))
        except Exception as e:
            logger.warning("parquet 지표 실패 (%s): %s", ticker, e)

        # 3) 보유 여부
        try:
            data["already_held"] = self._check_holdings(ticker)
        except Exception:
            data["already_held"] = False

        # 4) AI Brain 기존 판정
        try:
            data["ai_brain"] = self._get_ai_brain_judgment(ticker)
        except Exception:
            data["ai_brain"] = None

        # 5) Perplexity 촉매 검색
        if self.perplexity_enabled:
            try:
                data["perplexity"] = self._query_perplexity_catalyst(
                    ticker, data.get("stock_name", ticker)
                )
            except Exception as e:
                logger.warning("Perplexity 실패 (%s): %s", ticker, e)
                data["perplexity"] = None

        if current_price:
            data["current_price"] = current_price

        return data

    def _get_signal_data(self, ticker: str) -> dict:
        """SignalEngine으로 v8 파이프라인 시그널 계산."""
        import pandas as pd
        from src.signal_engine import SignalEngine

        parquet_path = DATA_DIR / "processed" / f"{ticker}.parquet"
        if not parquet_path.exists():
            return {"grade": "N/A", "score": 0, "signal": False}

        df = pd.read_parquet(parquet_path)
        if df.empty:
            return {"grade": "N/A", "score": 0, "signal": False}

        if self._signal_engine is None:
            self._signal_engine = SignalEngine()
        engine = self._signal_engine
        result = engine.calculate_signal(ticker, df, len(df) - 1)
        return {
            "grade": result.get("grade", "F"),
            "score": round(result.get("zone_score", 0) * 100, 1),
            "signal": result.get("signal", False),
            "trigger_type": result.get("trigger_type", "none"),
            "v8_action": result.get("v8_action", "SKIP"),
            "entry_price": result.get("entry_price", 0),
            "stop_loss": result.get("stop_loss", 0),
            "target_price": result.get("target_price", 0),
        }

    def _get_parquet_indicators(self, ticker: str) -> dict:
        """parquet에서 최신 기술 지표 추출."""
        import pandas as pd

        parquet_path = DATA_DIR / "processed" / f"{ticker}.parquet"
        if not parquet_path.exists():
            return {}

        df = pd.read_parquet(parquet_path)
        if df.empty:
            return {}

        last = df.iloc[-1]
        close = float(last.get("close", 0))

        # 52주 고점 대비
        high_52w = float(df["close"].tail(252).max()) if len(df) >= 10 else close
        near_high_pct = ((high_52w - close) / high_52w * 100) if high_52w > 0 else 0

        # 종목명 추출
        stock_name = ticker
        try:
            from src.stock_name_resolver import ticker_to_name
            stock_name = ticker_to_name(ticker) or ticker
        except Exception:
            pass

        return {
            "close": close,
            "rsi": round(float(last.get("rsi_14", 50)), 1),
            "adx": round(float(last.get("adx", 0)), 1),
            "macd": round(float(last.get("macd", 0)), 2),
            "macd_signal": round(float(last.get("macd_signal", 0)), 2),
            "ma_20": round(float(last.get("ma_20", 0)), 0),
            "ma_60": round(float(last.get("ma_60", 0)), 0),
            "bb_pct": round(float(last.get("bb_pct", 50)), 1),
            "sar_trend": int(last.get("sar_trend", 0)),
            "vol_ratio": round(float(last.get("vol_ratio", 1.0)), 2),
            "near_high_pct": round(near_high_pct, 1),
            "stock_name": stock_name,
        }

    def _check_holdings(self, ticker: str) -> bool:
        """positions.json에서 보유 여부 확인."""
        pos_path = DATA_DIR / "positions.json"
        if not pos_path.exists():
            return False
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("positions", []):
            if p.get("ticker") == ticker:
                return True
        return False

    def _get_ai_brain_judgment(self, ticker: str) -> dict | None:
        """ai_brain_judgment.json에서 해당 종목 판정 가져오기."""
        path = DATA_DIR / "ai_brain_judgment.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # buys 리스트에서 ticker 매칭
        for buy in data.get("buys", []):
            if buy.get("ticker") == ticker:
                return {
                    "action": buy.get("action", ""),
                    "reasoning": buy.get("reasoning", ""),
                    "catalysts": buy.get("catalysts", []),
                }
        return None

    def _query_perplexity_catalyst(self, ticker: str, name: str) -> dict | None:
        """Perplexity로 실시간 촉매/리스크 검색 (동기)."""
        from src.agents.perplexity_verifier import PerplexityVerifier

        verifier = PerplexityVerifier(max_verifications=1, max_thesis_checks=0)
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""오늘 날짜: {today}

{name}({ticker}) 종목의 최신 뉴스와 촉매를 검색하세요.

다음 JSON으로 응답:
{{
  "catalyst_found": true,
  "catalyst_summary": "주요 촉매 1줄 요약",
  "risk_found": false,
  "risk_summary": "주요 리스크 1줄 요약",
  "sentiment": "POSITIVE"
}}

sentiment: POSITIVE / NEUTRAL / NEGATIVE"""
        return verifier._query(prompt)

    def _build_buy_prompt(self, ticker: str, qty: int, data: dict) -> str:
        """매수 분석용 Claude 프롬프트 구성."""
        name = data.get("stock_name", ticker)
        lines = [
            f"## 매수 분석 요청: {name}({ticker}) {qty}주",
            "",
            "### 시그널 엔진 결과",
            f"- Grade: {data.get('grade', 'N/A')}",
            f"- Score: {data.get('score', 0)}/100",
            f"- Signal: {data.get('signal', False)}",
            f"- Trigger: {data.get('trigger_type', 'none')}",
            f"- V8 Action: {data.get('v8_action', 'N/A')}",
            "",
            "### 기술적 지표",
            f"- 종가: {data.get('close', 0):,.0f}원",
            f"- RSI(14): {data.get('rsi', 50)}",
            f"- ADX: {data.get('adx', 0)}",
            f"- MACD: {data.get('macd', 0)} (sig: {data.get('macd_signal', 0)})",
            f"- BB%: {data.get('bb_pct', 50)}",
            f"- SAR 추세: {'상승' if data.get('sar_trend') == 1 else '하락'}",
            f"- MA20: {data.get('ma_20', 0):,.0f} | MA60: {data.get('ma_60', 0):,.0f}",
            f"- 거래량 비율: {data.get('vol_ratio', 1.0)}배",
            f"- 52주 고점 대비: {data.get('near_high_pct', 0):.1f}% 아래",
        ]

        if data.get("already_held"):
            lines.append("\n⚠️ 이미 보유 중인 종목 (추가매수)")

        if data.get("ai_brain"):
            brain = data["ai_brain"]
            lines.append(f"\n### AI Brain 기존 판정")
            lines.append(f"- Action: {brain.get('action', '?')}")
            lines.append(f"- Reasoning: {brain.get('reasoning', '')[:200]}")
            cats = brain.get("catalysts", [])
            if cats:
                lines.append(f"- Catalysts: {', '.join(cats[:3])}")

        if data.get("perplexity"):
            pp = data["perplexity"]
            lines.append(f"\n### Perplexity 실시간 검색")
            lines.append(f"- 촉매 발견: {pp.get('catalyst_found', False)}")
            lines.append(f"- 촉매: {pp.get('catalyst_summary', '미확인')}")
            lines.append(f"- 리스크: {pp.get('risk_summary', '없음')}")
            lines.append(f"- 감성: {pp.get('sentiment', 'NEUTRAL')}")
        else:
            lines.append("\n### Perplexity: 사용 불가")

        lines.append("""
다음 JSON으로 응답:
{
  "verdict": "BUY_OK|WAIT|SKIP",
  "confidence": 75,
  "signal_grade": "B",
  "signal_score": 62,
  "technical_summary": "SAR↑ + RSI 55 + ADX 22 — 추세 초기",
  "catalyst": "HBM 수주 확대 (confirmed)",
  "risk_warning": "52주 고점 근접 (3% 이내)",
  "suggestion": "현재가 대비 -2% 눌림에서 진입 추천"
}""")
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 매도 분석
    # ═══════════════════════════════════════════

    async def analyze_sell(
        self, ticker: str, qty: int, holding_info: dict = None
    ) -> dict:
        """매도 AI 분석. SELL_OK / PARTIAL / HOLD 판정."""
        try:
            data = self._gather_sell_data(ticker, holding_info)
            prompt = self._build_sell_prompt(ticker, qty, data)
            result = await self._ask_claude_json(SYSTEM_SELL, prompt)
            # 필수 필드 보정
            result.setdefault("verdict", "HOLD")
            result.setdefault("confidence", 50)
            result.setdefault("catalyst_status", "UNKNOWN")
            result.setdefault("thesis_intact", True)
            result.setdefault("technical_signal", "")
            result.setdefault("supply_demand", "")
            result.setdefault("risk_warning", None)
            result.setdefault("suggestion", "")
            result.setdefault("partial_pct", 50)
            return result
        except Exception as e:
            logger.error("TradeAdvisor.analyze_sell 실패: %s", e)
            return {"verdict": "ERROR", "error": str(e)}

    def _gather_sell_data(self, ticker: str, holding_info: dict = None) -> dict:
        """매도 분석 데이터 수집."""
        data: dict = {"ticker": ticker}

        # 1) parquet 지표
        try:
            data.update(self._get_parquet_indicators(ticker))
        except Exception:
            pass

        # 2) ai_sell_cache.json (35분 TTL)
        try:
            data["sell_cache"] = self._get_sell_cache(ticker)
        except Exception:
            data["sell_cache"] = None

        # 3) ai_sell_consensus.json
        try:
            data["consensus"] = self._get_sell_consensus(ticker)
        except Exception:
            data["consensus"] = None

        # 4) perplexity_verification.json
        try:
            data["perplexity_v"] = self._get_perplexity_verification(ticker)
        except Exception:
            data["perplexity_v"] = None

        # 5) 보유 정보
        if holding_info:
            data["holding"] = holding_info
        else:
            data["holding"] = self._get_position_info(ticker)

        # 6) Perplexity 실시간 (캐시 없으면)
        if not data.get("sell_cache") and self.perplexity_enabled:
            try:
                data["perplexity_live"] = self._query_perplexity_catalyst(
                    ticker, data.get("stock_name", ticker)
                )
            except Exception:
                data["perplexity_live"] = None

        return data

    def _get_sell_cache(self, ticker: str) -> dict | None:
        """ai_sell_cache.json에서 35분 이내 캐시 조회."""
        cache_path = DATA_DIR / "ai_sell_cache.json"
        if not cache_path.exists():
            return None
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        entry = cache.get(ticker)
        if not entry:
            return None
        # TTL 확인
        cached_at = entry.get("cached_at", "")
        ttl_min = self.settings.get("sell_cache_ttl_min", 35)
        if cached_at:
            try:
                ts = datetime.fromisoformat(cached_at)
                if datetime.now() - ts > timedelta(minutes=ttl_min):
                    return None  # 만료
            except Exception:
                pass
        return entry

    def _get_sell_consensus(self, ticker: str) -> dict | None:
        """ai_sell_consensus.json에서 해당 종목 합의 결과."""
        path = DATA_DIR / "ai_sell_consensus.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get(ticker)

    def _get_perplexity_verification(self, ticker: str) -> dict | None:
        """perplexity_verification.json에서 해당 종목 검증 결과."""
        path = DATA_DIR / "perplexity_verification.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for v in data.get("stock_verifications", []):
            if v.get("ticker") == ticker:
                return v
        return None

    def _get_position_info(self, ticker: str) -> dict | None:
        """positions.json에서 보유 상세."""
        pos_path = DATA_DIR / "positions.json"
        if not pos_path.exists():
            return None
        with open(pos_path, encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("positions", []):
            if p.get("ticker") == ticker:
                entry = p.get("entry_price", 0)
                current = p.get("current_price", entry)
                pnl = ((current - entry) / entry * 100) if entry else 0
                return {
                    "entry_price": entry,
                    "current_price": current,
                    "pnl_pct": round(pnl, 1),
                    "shares": p.get("shares", 0),
                    "entry_date": p.get("entry_date", ""),
                    "stop_loss": p.get("stop_loss", 0),
                    "target_price": p.get("target_price", 0),
                    "name": p.get("name", ticker),
                }
        return None

    def _build_sell_prompt(self, ticker: str, qty: int, data: dict) -> str:
        """매도 분석용 Claude 프롬프트 구성."""
        name = data.get("stock_name", ticker)
        holding = data.get("holding", {}) or {}

        lines = [
            f"## 매도 분석 요청: {name}({ticker}) {qty}주",
        ]

        if holding:
            lines.extend([
                "",
                "### 보유 현황",
                f"- 평균단가: {holding.get('entry_price', 0):,.0f}원",
                f"- 현재가: {holding.get('current_price', 0):,.0f}원",
                f"- 손익률: {holding.get('pnl_pct', 0):+.1f}%",
                f"- 보유주수: {holding.get('shares', 0)}주",
                f"- 진입일: {holding.get('entry_date', '?')}",
                f"- 손절가: {holding.get('stop_loss', 0):,.0f}",
                f"- 목표가: {holding.get('target_price', 0):,.0f}",
            ])

        lines.extend([
            "",
            "### 기술적 지표",
            f"- RSI(14): {data.get('rsi', 50)}",
            f"- ADX: {data.get('adx', 0)}",
            f"- SAR 추세: {'상승' if data.get('sar_trend') == 1 else '하락'}",
            f"- MA20: {data.get('ma_20', 0):,.0f} | MA60: {data.get('ma_60', 0):,.0f}",
            f"- BB%: {data.get('bb_pct', 50)}",
            f"- 거래량 비율: {data.get('vol_ratio', 1.0)}배",
        ])

        # SellBrain 캐시
        sell_cache = data.get("sell_cache")
        if sell_cache:
            lines.extend([
                "",
                "### SellBrain AI 판정 (캐시)",
                f"- Action: {sell_cache.get('action', '?')}",
                f"- Confidence: {sell_cache.get('confidence', 0)}%",
                f"- Thesis: {sell_cache.get('thesis_status', '?')}",
                f"- Reasoning: {sell_cache.get('reasoning', '')[:200]}",
            ])
            top_reasons = sell_cache.get("top_reasons", [])
            if top_reasons:
                lines.append(f"- Top Reasons: {', '.join(top_reasons[:3])}")

        # Dual AI 합의
        consensus = data.get("consensus")
        if consensus:
            lines.extend([
                "",
                "### Dual AI 합의",
                f"- Claude Action: {consensus.get('claude_action', '?')}",
                f"- GPT Catalyst: {consensus.get('gpt_catalyst', '?')}",
                f"- Final Action: {consensus.get('final_action', '?')}",
                f"- Combined Score: {consensus.get('combined_score', 0)}",
            ])
            if consensus.get("override_reason"):
                lines.append(f"- Override: {consensus['override_reason']}")

        # Perplexity 검증
        perp_v = data.get("perplexity_v")
        if perp_v:
            lines.extend([
                "",
                "### Perplexity 검증",
                f"- Verdict: {perp_v.get('verdict', '?')}",
                f"- Confidence: {perp_v.get('confidence_score', 0):.0%}",
            ])
            findings = perp_v.get("additional_findings", [])
            if findings:
                lines.append(f"- 추가 발견: {', '.join(findings[:2])}")

        # Perplexity 실시간
        perp_live = data.get("perplexity_live")
        if perp_live:
            lines.extend([
                "",
                "### Perplexity 실시간",
                f"- 촉매: {perp_live.get('catalyst_summary', '미확인')}",
                f"- 리스크: {perp_live.get('risk_summary', '없음')}",
                f"- 감성: {perp_live.get('sentiment', 'NEUTRAL')}",
            ])

        lines.append("""
다음 JSON으로 응답:
{
  "verdict": "SELL_OK|PARTIAL|HOLD",
  "confidence": 80,
  "catalyst_status": "ALIVE|FADING|DEAD|UNKNOWN",
  "thesis_intact": true,
  "technical_signal": "추세 유지 — SAR↑, MA20 위",
  "supply_demand": "외인 3일 연속 순매수",
  "risk_warning": null,
  "suggestion": "촉매 살아있음 — 홀딩 추천",
  "partial_pct": 50,
  "reason": "thesis 유효 + 촉매 ALIVE → 기계적 매도 금지"
}""")
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 텔레그램 메시지 포맷
    # ═══════════════════════════════════════════

    @staticmethod
    def format_buy_message(ticker: str, name: str, qty: int, result: dict) -> str:
        """매수 분석 결과를 텔레그램 메시지로 포맷."""
        verdict = result.get("verdict", "ERROR")
        confidence = result.get("confidence", 0)
        grade = result.get("signal_grade", "N/A")
        score = result.get("signal_score", 0)
        tech = result.get("technical_summary", "")
        catalyst = result.get("catalyst", "미확인")
        risk = result.get("risk_warning")
        suggestion = result.get("suggestion", "")

        if verdict == "ERROR":
            return f"❌ AI 분석 실패: {result.get('error', '알 수 없는 오류')}"

        icon = {"BUY_OK": "🟢", "WAIT": "🟡", "SKIP": "🔴"}.get(verdict, "⚪")
        verdict_text = {
            "BUY_OK": "매수 OK",
            "WAIT": "눌림 대기",
            "SKIP": "매수 비추",
        }.get(verdict, verdict)

        lines = [
            f"🔍 [AI 매수 분석] {name}({ticker})",
            "━" * 22,
            f"📊 시그널: {grade}등급 ({score}/100점)",
            f"📈 기술: {tech}",
            f"📰 촉매: {catalyst}",
        ]
        if risk:
            lines.append(f"⚠️ 주의: {risk}")
        lines.extend([
            f"💡 AI 판단: {icon} {verdict_text} (신뢰도 {confidence}%)",
        ])
        if suggestion:
            lines.append(f"   → {suggestion}")
        lines.append("━" * 22)
        return "\n".join(lines)

    @staticmethod
    def format_sell_message(
        ticker: str, name: str, qty: int, result: dict, holding: dict = None
    ) -> str:
        """매도 분석 결과를 텔레그램 메시지로 포맷."""
        verdict = result.get("verdict", "ERROR")
        confidence = result.get("confidence", 0)
        catalyst_status = result.get("catalyst_status", "UNKNOWN")
        tech = result.get("technical_signal", "")
        supply = result.get("supply_demand", "")
        risk = result.get("risk_warning")
        suggestion = result.get("suggestion", "")
        reason = result.get("reason", "")

        if verdict == "ERROR":
            return f"❌ AI 분석 실패: {result.get('error', '알 수 없는 오류')}"

        icon = {"SELL_OK": "🔴", "PARTIAL": "🟡", "HOLD": "🟢"}.get(verdict, "⚪")
        verdict_text = {
            "SELL_OK": "매도 추천",
            "PARTIAL": "분할 매도",
            "HOLD": "홀딩 추천",
        }.get(verdict, verdict)

        lines = [
            f"🔍 [AI 매도 분석] {name}({ticker})",
            "━" * 22,
        ]
        if holding:
            entry = holding.get("entry_price", 0)
            current = holding.get("current_price", 0)
            pnl = holding.get("pnl_pct", 0)
            shares = holding.get("shares", 0)
            lines.append(
                f"💰 보유: {shares}주 (평단 {entry:,.0f}원 → 현재 {current:,.0f}원, {pnl:+.1f}%)"
            )
        lines.extend([
            f"📈 기술: {tech}",
            f"📰 촉매: {catalyst_status}",
        ])
        if supply:
            lines.append(f"🏢 수급: {supply}")
        if risk:
            lines.append(f"⚠️ 주의: {risk}")
        lines.append(f"💡 AI 판단: {icon} {verdict_text} (신뢰도 {confidence}%)")
        if suggestion:
            lines.append(f"   → {suggestion}")
        if reason:
            lines.append(f"   📝 {reason}")
        lines.append("━" * 22)
        return "\n".join(lines)
