"""
🎯 포물선의 초점 v9.0 — C+E 하이브리드 파이프라인
=================================================
"Kill로 죽이고, R:R×Zone으로 줄 세우고, 태그로 참고한다."

Architecture:
  Layer 0: 7D 레짐 Gate (시장 환경 파악 → Kill 기준값 세팅)
  Layer 1: Kill Filters (하나라도 걸리면 즉시 탈락)
  Layer 2: 6D 함정 필터 (기술적 근거 없이 수급+뉴스만 좋은 종목 제거)
  Layer 3: 최종 순위 = R:R × (Zone/15)
  Layer 4: 정보 태그 (순위 안 바꿈, tiebreaker만)

1D~8D 프레임워크 정합성:
  1D(현상)=Zone, 2D(관계)=R:R, 3D(구조)=Kill, 4D(변화)=Trigger,
  5D(관찰자)=SD태그, 6D(설계자)=함정필터, 7D(조류)=레짐Gate, 8D(한계)=R:R기반정렬+태그겸손

Author: 퐝가 × 자비스
Version: 9.0 (C+E Hybrid)
Date: 2026-02-16
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Data Models
# ============================================================

class RegimeType(Enum):
    """7D 레짐 분류"""
    NORMAL = "normal"                    # 공매도 없음, 금리 중립
    SHORT_SELLING = "short_selling"      # 공매도 허용
    HIGH_VOLATILITY = "high_volatility"  # VIX 30+
    RATE_HIKING = "rate_hiking"          # 금리 인상기
    RATE_CUTTING = "rate_cutting"        # 금리 인하기


@dataclass
class Regime:
    """Layer 0: 7D 레짐 상태"""
    short_selling_allowed: bool = True
    interest_rate_cycle: str = "neutral"   # hiking / cutting / neutral
    vix_level: str = "mid"                 # low(<15) / mid(15~25) / high(>25)
    market_trend: str = "neutral"          # bull / bear / neutral
    
    # 레짐에 따른 Kill 기준값 (핵심)
    zone_threshold: int = 5
    rr_threshold: float = 1.5
    
    def __post_init__(self):
        """레짐에 따라 Kill 기준값 자동 조정"""
        if self.short_selling_allowed:
            self.zone_threshold = 7    # 공매도 있으면 더 엄격
            self.rr_threshold = 2.0
        else:
            self.zone_threshold = 5
            self.rr_threshold = 1.5
        
        if self.vix_level == "high":
            self.rr_threshold += 0.5   # 고변동성이면 R:R 기준 상향
        
        if self.interest_rate_cycle == "hiking":
            self.zone_threshold += 1   # 금리 인상기엔 위치 기준 상향


@dataclass
class StockData:
    """종목 데이터"""
    ticker: str
    name: str
    
    # Quant 관련
    zone_score: float = 0.0          # 0~15
    trigger_grade: str = "D"          # T1_high, T1_low, T2, T3, confirm, D
    trigger_confidence: float = 0.0   # 0~1
    rr_ratio: float = 0.0            # 손익비
    trend_score: float = 0.0         # 0~4
    quant_score: float = 0.0         # 0~30 (합산)
    
    # Supply/Demand
    sd_score: float = 0.0            # 0~25
    foreign_net_buy_days: int = 0
    institution_net_buy_days: int = 0
    
    # News
    news_score: float = 0.0          # 0~25
    
    # Consensus
    consensus_score: float = 0.0     # 0~20
    
    # 유동성/시장 데이터
    avg_trading_value_20d: float = 0  # 20일 평균 거래대금
    pct_from_52w_high: float = 0.0    # 52주 고점 대비 % (음수)
    short_selling_ratio: float = 0.0  # 공매도 비율
    
    # 파이프라인 결과
    status: str = "PENDING"           # PENDING / KILLED / TRAPPED / CANDIDATE
    kill_reasons: List[str] = field(default_factory=list)
    trap_reason: Optional[str] = None
    rank_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    tag_count: int = 0


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    regime: Regime
    candidates: List[StockData]
    killed: List[StockData]
    trapped: List[StockData]
    total_universe: int = 0
    
    @property
    def summary(self) -> str:
        return (
            f"Universe: {self.total_universe} → "
            f"Killed: {len(self.killed)} | "
            f"Trapped: {len(self.trapped)} | "
            f"Candidates: {len(self.candidates)}"
        )


# ============================================================
# Layer 0: 7D 레짐 Gate
# ============================================================

def detect_regime(market_data: Dict[str, Any]) -> Regime:
    """
    7D 레짐 감지.
    시장 환경을 파악하고 Kill 기준값을 세팅한다.
    향후 슈퍼사이클 모듈이 여기에 연결된다.
    """
    regime = Regime(
        short_selling_allowed=market_data.get('short_selling_allowed', True),
        interest_rate_cycle=_classify_rate_cycle(market_data),
        vix_level=_classify_vix(market_data.get('vix', 20)),
        market_trend=_classify_market_trend(market_data),
    )
    
    logger.info(
        f"[L0 Regime] short_sell={regime.short_selling_allowed}, "
        f"rate={regime.interest_rate_cycle}, vix={regime.vix_level}, "
        f"zone_th={regime.zone_threshold}, rr_th={regime.rr_threshold}"
    )
    
    return regime


def _classify_rate_cycle(market_data: Dict) -> str:
    rate_change = market_data.get('rate_change_3m', 0)
    if rate_change > 0.25:
        return "hiking"
    elif rate_change < -0.25:
        return "cutting"
    return "neutral"


def _classify_vix(vix: float) -> str:
    if vix < 15:
        return "low"
    elif vix > 25:
        return "high"
    return "mid"


def _classify_market_trend(market_data: Dict) -> str:
    kospi_ma60_pct = market_data.get('kospi_vs_ma60_pct', 0)
    if kospi_ma60_pct > 3:
        return "bull"
    elif kospi_ma60_pct < -3:
        return "bear"
    return "neutral"


# ============================================================
# Layer 1: Kill Filters
# ============================================================

def kill_filter(stock: StockData, regime: Regime) -> bool:
    """
    Kill Filters. 하나라도 걸리면 즉시 탈락.
    
    8D 철학: "맞추기보다 거르기"
    튜닝 포인트 = 5개 (과적합 방지)
    
    Returns:
        True if passed (alive), False if killed
    """
    kills = []
    
    # K1: 기술적 위치 불량 (1D~2D)
    if stock.zone_score < regime.zone_threshold:
        kills.append(
            f"K1: Zone {stock.zone_score:.1f} < {regime.zone_threshold}"
        )
    
    # K2: 반취약성 미달 (8D 핵심)
    if stock.rr_ratio < regime.rr_threshold:
        kills.append(
            f"K2: R:R {stock.rr_ratio:.2f} < {regime.rr_threshold}"
        )
    
    # K3: 시작 신호 없음 (4D)
    invalid_triggers = {'D', None, ''}
    if stock.trigger_grade in invalid_triggers:
        kills.append("K3: No valid trigger")
    
    # K4: 유동성 부족 (8D 놓친것#4 반영)
    min_liquidity = 1_000_000_000  # 10억원
    if stock.avg_trading_value_20d < min_liquidity:
        liquidity_billions = stock.avg_trading_value_20d / 1e8
        kills.append(
            f"K4: Liquidity {liquidity_billions:.0f}억 < 10억"
        )
    
    # K5: 이미 올라간 놈 차단
    if stock.pct_from_52w_high > -5:
        kills.append(
            f"K5: Too close to 52w high ({stock.pct_from_52w_high:.1f}%)"
        )
    
    if kills:
        stock.status = "KILLED"
        stock.kill_reasons = kills
        logger.debug(f"[L1 Kill] {stock.ticker} {stock.name}: {kills}")
        return False
    
    return True


# ============================================================
# Layer 2: 6D 함정 필터
# ============================================================

def trap_filter_6d(stock: StockData) -> bool:
    """
    6D 함정 필터.
    
    "기술적 근거 없이 수급+뉴스만 좋은 건 설계된 종목이다."
    Quant가 약한데 SD/News가 강하면 → 누군가 의도적으로 만든 신호.
    
    Returns:
        True if passed (clean), False if trapped
    """
    quant_weak = stock.quant_score < 18
    sd_strong = stock.sd_score >= 20
    news_strong = stock.news_score >= 15
    
    is_trap = quant_weak and sd_strong and news_strong
    
    if is_trap:
        stock.status = "TRAPPED"
        stock.trap_reason = (
            f"6D TRAP: Quant({stock.quant_score:.0f}) weak "
            f"but SD({stock.sd_score:.0f})+News({stock.news_score:.0f}) strong "
            f"→ designed stock suspected"
        )
        logger.debug(f"[L2 Trap] {stock.ticker} {stock.name}: {stock.trap_reason}")
        return False
    
    return True


# ============================================================
# Layer 3: 최종 순위 = R:R × (Zone/15)
# ============================================================

def calculate_rank_score(stock: StockData) -> float:
    """
    최종 순위 점수.
    
    R:R = 8D 반취약성 ("틀려도 적게 잃는 종목이 항상 위")
    Zone = 기술적 위치 ("좋은 자리에 있는 종목 우선")
    곱셈 = 둘 다 높아야 상위
    
    E 구조에서 차용한 유일한 요소.
    """
    rank = stock.rr_ratio * (stock.zone_score / 15.0)
    stock.rank_score = round(rank, 4)
    return stock.rank_score


# ============================================================
# Layer 4: 정보 태그
# ============================================================

def generate_tags(stock: StockData) -> List[str]:
    """
    정보 태그 생성.
    
    순위를 바꾸지 않는다. tiebreaker에서만 역할.
    8D 겸손: "모르는 건 모른다" → 없어도 패널티 없음, 있으면 참고만.
    
    향후 LLM 태그가 여기에 플러그인으로 연결된다.
    """
    tags = []
    
    # === 수급 태그 (5D: 관찰자의 행동) ===
    if stock.sd_score >= 20:
        tags.append("🟢 SD: 수급전환 확인")
    elif stock.sd_score >= 10:
        tags.append("🟡 SD: 수급 초기전환")
    elif stock.sd_score >= 5:
        tags.append("⚪ SD: 수급 미약")
    
    # === 뉴스 태그 ===
    if stock.news_score >= 15:
        tags.append("🟢 News: 이슈+실적")
    elif stock.news_score >= 5:
        tags.append("🟡 News: 이슈 존재")
    
    # === 컨센서스 태그 ===
    if stock.consensus_score >= 15:
        tags.append("🟢 Consensus: 강한 상향")
    elif stock.consensus_score >= 10:
        tags.append("🟡 Consensus: 상향")
    
    # === 공매도 경고 태그 (7D) ===
    if stock.short_selling_ratio > 5:
        tags.append("🔴 공매도 과열 경고")
    elif stock.short_selling_ratio > 3:
        tags.append("🟠 공매도 주의")
    
    # === 수급 디테일 태그 ===
    if stock.foreign_net_buy_days >= 5:
        tags.append("🐋 외국인 5일+ 연속 순매수")
    if stock.institution_net_buy_days >= 5:
        tags.append("🏛️ 기관 5일+ 연속 순매수")
    
    # ============================================
    # 🔸 향후 LLM 플러그인 자리 (Layer 4 확장)
    # ============================================
    # tags.append("🧠 LLM: 공시 해석 - 호재 확률 85%")
    # tags.append("🧠 LLM: 뉴스 신선도 - 새로운 정보 (아직 미반영)")
    # tags.append("🧠 LLM: 6D 판정 - 자연발생 뉴스로 보임")
    # tags.append("🧠 LLM: 섹터 맥락 - 동종업계 대비 저평가")
    
    stock.tags = tags
    stock.tag_count = len(tags)
    return tags


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    universe: List[StockData],
    market_data: Dict[str, Any]
) -> PipelineResult:
    """
    C+E 하이브리드 파이프라인 실행.
    
    Flow:
        Layer 0 (7D Regime) → Layer 1 (Kill) → Layer 2 (6D Trap) 
        → Layer 3 (Rank) → Layer 4 (Tags) → Sort → Output
    """
    # Layer 0: 레짐 파악
    regime = detect_regime(market_data)
    
    candidates = []
    killed = []
    trapped = []
    
    for stock in universe:
        # Layer 1: Kill Filter
        if not kill_filter(stock, regime):
            killed.append(stock)
            continue
        
        # Layer 2: 6D 함정 필터
        if not trap_filter_6d(stock):
            trapped.append(stock)
            continue
        
        # Layer 3: 순위 스코어
        calculate_rank_score(stock)
        
        # Layer 4: 정보 태그
        generate_tags(stock)
        
        stock.status = "CANDIDATE"
        candidates.append(stock)
    
    # 정렬: rank_score 내림차순, 동점이면 tag_count 내림차순
    candidates.sort(
        key=lambda x: (x.rank_score, x.tag_count),
        reverse=True
    )
    
    result = PipelineResult(
        regime=regime,
        candidates=candidates,
        killed=killed,
        trapped=trapped,
        total_universe=len(universe),
    )
    
    logger.info(f"[Pipeline] {result.summary}")
    
    return result


# ============================================================
# 검증: 가상 종목 7개 시뮬레이션
# ============================================================

def _run_validation():
    """
    노션 문서의 가상 종목 7개로 파이프라인 검증.
    정답 순위: S5 > S1 > S7 > S3 > S4 > S6 > S2
    """
    print("=" * 60)
    print("🎯 포물선의 초점 v9.0 — C+E 하이브리드 검증")
    print("=" * 60)
    
    # 가상 종목 생성
    stocks = [
        StockData(
            ticker="S1", name="이상적 시작점",
            zone_score=10, trigger_grade="T1_low", trigger_confidence=0.6,
            rr_ratio=2.5, trend_score=3, quant_score=22,
            sd_score=10, news_score=2, consensus_score=6,
            avg_trading_value_20d=5e9, pct_from_52w_high=-25,
            foreign_net_buy_days=2, institution_net_buy_days=1,
        ),
        StockData(
            ticker="S2", name="이미 올라간 놈",
            zone_score=3, trigger_grade="D", trigger_confidence=0,
            rr_ratio=1.2, trend_score=1, quant_score=8,
            sd_score=25, news_score=25, consensus_score=19,
            avg_trading_value_20d=20e9, pct_from_52w_high=-2,
        ),
        StockData(
            ticker="S3", name="수급 전환 초기",
            zone_score=8, trigger_grade="T2", trigger_confidence=0.65,
            rr_ratio=3.5, trend_score=2, quant_score=18,
            sd_score=10, news_score=0, consensus_score=4,
            avg_trading_value_20d=3e9, pct_from_52w_high=-30,
            foreign_net_buy_days=1, institution_net_buy_days=2,
        ),
        StockData(
            ticker="S4", name="뉴스 드리븐 급등",
            zone_score=5, trigger_grade="confirm", trigger_confidence=0.5,
            rr_ratio=2.0, trend_score=2, quant_score=12,
            sd_score=24, news_score=25, consensus_score=14,
            avg_trading_value_20d=15e9, pct_from_52w_high=-10,
        ),
        StockData(
            ticker="S5", name="진짜 포물선 직전",
            zone_score=13, trigger_grade="T1_high", trigger_confidence=0.85,
            rr_ratio=8.5, trend_score=4, quant_score=28,
            sd_score=16, news_score=6, consensus_score=5,
            avg_trading_value_20d=8e9, pct_from_52w_high=-35,
            foreign_net_buy_days=3, institution_net_buy_days=2,
        ),
        StockData(
            ticker="S6", name="6D 함정 종목",
            zone_score=6, trigger_grade="T2", trigger_confidence=0.6,
            rr_ratio=2.2, trend_score=2, quant_score=14,
            sd_score=25, news_score=23, consensus_score=15,
            avg_trading_value_20d=12e9, pct_from_52w_high=-8,
        ),
        StockData(
            ticker="S7", name="조용한 보석",
            zone_score=12, trigger_grade="T1_low", trigger_confidence=0.65,
            rr_ratio=9.0, trend_score=3, quant_score=26,
            sd_score=1, news_score=0, consensus_score=1,
            avg_trading_value_20d=2e9, pct_from_52w_high=-40,
        ),
    ]
    
    # 레짐: 공매도 허용 상태 (현재 한국시장)
    market_data = {
        'short_selling_allowed': True,
        'vix': 18,
        'rate_change_3m': 0,
        'kospi_vs_ma60_pct': 1.5,
    }
    
    # 파이프라인 실행
    result = run_pipeline(stocks, market_data)
    
    # 결과 출력
    print(f"\n📊 레짐: short_sell={result.regime.short_selling_allowed}, "
          f"zone_th={result.regime.zone_threshold}, rr_th={result.regime.rr_threshold}")
    print(f"\n{result.summary}\n")
    
    print("─" * 60)
    print("❌ KILLED 종목")
    print("─" * 60)
    for s in result.killed:
        print(f"  {s.ticker} {s.name}")
        for reason in s.kill_reasons:
            print(f"    └── {reason}")
    
    print(f"\n{'─' * 60}")
    print("🚫 TRAPPED 종목 (6D 함정)")
    print("─" * 60)
    for s in result.trapped:
        print(f"  {s.ticker} {s.name}")
        print(f"    └── {s.trap_reason}")
    
    print(f"\n{'─' * 60}")
    print("✅ CANDIDATES (최종 순위)")
    print("─" * 60)
    for i, s in enumerate(result.candidates, 1):
        tag_str = " | ".join(s.tags) if s.tags else "(태그 없음)"
        print(f"  {i}위: {s.ticker} {s.name}")
        print(f"       Rank Score: {s.rank_score:.4f} "
              f"(R:R={s.rr_ratio} × Zone={s.zone_score}/15)")
        print(f"       Tags: {tag_str}")
        print()
    
    # 정답 비교
    print("─" * 60)
    print("📋 정답 비교")
    print("─" * 60)
    expected = ["S5", "S1", "S7", "S3"]  # Kill/Trap 된 S2, S4, S6 제외
    actual = [s.ticker for s in result.candidates]
    
    print(f"  정답 (생존자): {expected}")
    print(f"  실제:         {actual}")
    
    matches = sum(1 for e, a in zip(expected, actual) if e == a)
    print(f"  일치율: {matches}/{len(expected)} ({matches/len(expected)*100:.0f}%)")
    
    # S5가 1위인지 확인
    if actual and actual[0] == "S5":
        print("  ✅ S5(포물선 직전)이 1위 — 정확!")
    else:
        print(f"  ⚠️ 1위가 {actual[0] if actual else 'N/A'} — S5가 아님")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_validation()
