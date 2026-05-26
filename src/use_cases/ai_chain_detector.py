"""AI 밸류체인 동조화 검출 — 5/26 (서브에이전트 5종 폭등주 분석 기반).

배경 (서브에이전트 stock-analyzer 보고, 5/26 11:00 분석):
- 5/26 ISC+16.9%/인텍플러스+18.9%/코리아써키트+12.4%/두산+10.1%/동진쎄미켐+10.2% 동시 폭등
- 공통 패턴: AI 데이터센터 인프라 공급망 (검사/PCB/소재) 동조화
- 우리 시스템 못 잡은 이유: AI 밸류체인 동조화 검출 로직 부재 (단일 종목 시그널만 봄)

룰:
- AI 관련 4개 세부 섹터 (AI반도체검사 / AI반도체PCB / AI반도체소재 / AI산업소재)
- 4개 중 3개 이상에서 "거래량 1.5배 이상 + 양봉 50%+ 상승 종목 1개+" 동시 발화 시
  → "AI 밸류체인 동조 시그널" 발화
- 발화 시 → 워치리스트 추가 (오늘 핫한 AI 종목들)
- 매수는 H4~H9 게이트 정상 통과 시에만

활용 위치:
- run_adaptive_cycle.py 또는 scan_sector_fire.py에서 매 사이클 호출
- 시그널 발화 시 텔레그램 알림 + 학습 로그
"""

from __future__ import annotations

import logging
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# AI 밸류체인 세부 섹터 4개 (sector_fire_map.yaml 정의)
AI_CHAIN_SECTORS = ["AI반도체검사", "AI반도체PCB", "AI반도체소재", "AI산업소재"]
AI_CORE_SECTOR = "AI반도체"  # 주력 섹터

MIN_FIRE_SECTORS = int(os.getenv("AI_CHAIN_MIN_SECTORS", "3"))  # 4개 중 3개 발화 시
SURGE_THRESHOLD_PCT = float(os.getenv("AI_CHAIN_SURGE_PCT", "5.0"))  # 종목 +5% 이상
VOLUME_RATIO_MIN = float(os.getenv("AI_CHAIN_VOL_RATIO", "1.5"))  # 평균 대비 1.5배

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "sector_fire_map.yaml"


@dataclass
class AIChainSignal:
    """AI 밸류체인 동조 시그널."""
    triggered: bool                         # 발화 여부
    fire_sectors: list[str] = field(default_factory=list)   # 발화된 섹터 명
    surge_stocks: list[dict] = field(default_factory=list)  # 폭등 종목 [{ticker, name, sector, change_pct}]
    fire_sector_count: int = 0
    reason: str = ""


def load_ai_chain_tickers() -> dict[str, list[str]]:
    """sector_fire_map.yaml에서 AI 밸류체인 4섹터의 종목 로드.

    Returns:
        {sector_name: [ticker, ...]}
    """
    if not CONFIG_PATH.exists():
        logger.warning("[AI chain] sector_fire_map.yaml 없음")
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        sectors = cfg.get("sectors", {})
        result = {}
        for s in AI_CHAIN_SECTORS:
            if s in sectors:
                result[s] = sectors[s].get("tickers", [])
        return result
    except Exception as e:
        logger.warning("[AI chain] sector_fire_map.yaml 로드 실패: %s", e)
        return {}


def detect_ai_chain_sync(
    broker,
    min_fire_sectors: int = MIN_FIRE_SECTORS,
    surge_pct: float = SURGE_THRESHOLD_PCT,
) -> AIChainSignal:
    """AI 밸류체인 동조화 검출.

    매 사이클 (30분) 호출:
    1. 4개 세부 섹터 각 종목 현재가 조회
    2. 섹터별 폭등 종목 (≥ surge_pct%) 카운트
    3. 폭등 종목 1개 이상 발화한 섹터 수 >= min_fire_sectors → AI 동조 시그널

    Args:
        broker: KisOrderAdapter (fetch_price)
        min_fire_sectors: 최소 발화 섹터 수 (기본 3 — 4개 중 3개)
        surge_pct: 종목 폭등 임계 (기본 +5%)

    Returns:
        AIChainSignal — triggered=True면 동조 시그널 발화.
    """
    chain_tickers = load_ai_chain_tickers()
    if not chain_tickers:
        return AIChainSignal(triggered=False, reason="섹터 로드 실패")

    surge_stocks = []
    sector_fire = {}  # {sector: surge_count}

    for sector, tickers in chain_tickers.items():
        sector_fire[sector] = 0
        for tk in tickers:
            try:
                res = broker.fetch_price(tk)
                out = res.get("output", {}) if res else {}
                p = int(out.get("stck_prpr", 0))
                chg = float(out.get("prdy_ctrt", 0))
                if p > 0 and chg >= surge_pct:
                    surge_stocks.append({
                        "ticker": tk,
                        "name": out.get("hts_kor_isnm", "") or out.get("prdt_name", ""),
                        "sector": sector,
                        "current_price": p,
                        "change_pct": chg,
                    })
                    sector_fire[sector] += 1
            except Exception as e:
                logger.debug("[AI chain] %s fetch 실패: %s", tk, e)

    # 발화 섹터 (1개 이상 폭등 종목 있는 섹터)
    fire_sectors = [s for s, c in sector_fire.items() if c > 0]
    triggered = len(fire_sectors) >= min_fire_sectors

    reason = (
        f"{len(fire_sectors)}/{len(chain_tickers)} 섹터 발화"
        f" (임계 {min_fire_sectors}+)"
    )
    if triggered:
        reason = f"⚠️ AI 밸류체인 동조 — {reason}, 폭등 종목 {len(surge_stocks)}건"

    sig = AIChainSignal(
        triggered=triggered,
        fire_sectors=fire_sectors,
        surge_stocks=sorted(surge_stocks, key=lambda x: -x["change_pct"]),
        fire_sector_count=len(fire_sectors),
        reason=reason,
    )
    if triggered:
        logger.warning(
            "[AI chain] 동조 발화: 섹터 %s, 폭등 %d종목",
            fire_sectors, len(surge_stocks),
        )
    return sig


def format_ai_chain_signal_for_telegram(sig: AIChainSignal) -> str:
    """텔레그램 알림 포맷."""
    if not sig.triggered:
        return f"⏸️ AI 밸류체인 — {sig.reason}"

    lines = [
        f"⚠️ [AI 밸류체인 동조 발화]",
        f"  발화 섹터: {', '.join(sig.fire_sectors)}",
        f"  폭등 종목 ({len(sig.surge_stocks)}건):",
    ]
    for s in sig.surge_stocks[:10]:
        nm = s.get("name") or s.get("ticker")
        lines.append(
            f"    {s['sector'][:10]:10s} {nm}({s['ticker']}) "
            f"{s['current_price']:,}원 (+{s['change_pct']:.2f}%)"
        )
    lines.append(f"  → 워치리스트 자동 추가 권장 + H4~H7 게이트 통과 시 매수")
    return "\n".join(lines)
