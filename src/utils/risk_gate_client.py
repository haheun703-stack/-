# -*- coding: utf-8 -*-
"""
위험감지시스템 클라이언트 (RiskGateClient)

목적:
    퀀트봇/단타봇이 매매 직전에 정보봇이 산출한 한국시장 위험점수를 조회하여
    자동으로 매수금액/진입 여부를 결정할 수 있도록 제공하는 경량 SDK.

데이터 출처:
    Supabase macro_risk_daily 테이블 (정보봇이 매일 16:49 갱신)

봇 측 사용 패턴:

    # 1. 초기화 (봇 시작 시 1회)
    from src.infrastructure.adapters.risk_gate_client import RiskGateClient
    risk_gate = RiskGateClient(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_KEY"],
        bot_name="quant"  # "quant" or "daytrading"
    )

    # 2. 매수 직전 호출 (퀀트봇 예시)
    base_amount = 1_000_000  # 평소 매수금액 100만원
    multiplier = risk_gate.get_position_multiplier()
    actual_amount = int(base_amount * multiplier)
    # CRISIS 구간: 1,000,000 × 0.2 = 200,000원으로 자동 축소

    # 3. 단타봇 신규 진입 차단 체크
    if risk_gate.should_block_new_entry():
        return  # 위험 70+ 구간 → 신규 매수 안 함

    # 4. 상세 정보가 필요할 때 (로그/알림용)
    status = risk_gate.get_full_status()
    print(f"위험점수 {status['total_score']}점 ({status['level_kr']})")
    print(f"권장 행동: {status['recommended_action']}")
    for sig in status['key_signals']:
        print(f"  • {sig}")

설계 원칙:
    - 캐시: 같은 날짜는 메모리 캐시 (10분 TTL) — 매수 1000회 호출해도 Supabase 1회만
    - Fail-safe: Supabase 조회 실패 시 NORMAL(보수적 동작 X) — 운영 중단 방지
    - 의존성 최소: supabase 라이브러리만 필요
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class RiskGateClient:
    """위험감지시스템 SDK — 퀀트봇/단타봇이 매매 직전 호출"""

    # ─── 등급별 매수금액 곱하기 계수 ───
    # 퀀트봇: 0.2~1.0 (위기 시 평소의 20%만 매수)
    POSITION_SIZE_QUANT = {
        "NORMAL":  1.0,
        "CAUTION": 0.8,
        "WARNING": 0.6,
        "DANGER":  0.4,
        "CRISIS":  0.2,
    }

    # 단타봇: 위기 시 평소의 10%만 (더 보수적)
    POSITION_SIZE_DAYTRADING = {
        "NORMAL":  1.0,
        "CAUTION": 0.7,
        "WARNING": 0.5,
        "DANGER":  0.0,    # 위험 진입 시 신규 매수 차단
        "CRISIS":  0.0,    # 위기 진입 시 신규 매수 차단
    }

    # ─── 신규 진입 차단 기준 ───
    DAYTRADING_BLOCK_LEVELS = {"DANGER", "CRISIS"}   # 단타봇: 위험·위기 차단
    QUANT_BLOCK_LEVELS      = {"CRISIS"}             # 퀀트봇: 위기만 차단

    # 캐시 TTL (분)
    DEFAULT_CACHE_MINUTES = 10

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        bot_name: str = "quant",
        cache_minutes: int = DEFAULT_CACHE_MINUTES,
    ):
        """
        Args:
            supabase_url: Supabase 프로젝트 URL
            supabase_key: Supabase anon key 또는 service_role key
            bot_name: "quant" 또는 "daytrading"
            cache_minutes: 메모리 캐시 유효시간 (기본 10분)
        """
        if bot_name not in ("quant", "daytrading"):
            raise ValueError("bot_name must be 'quant' or 'daytrading'")

        self.bot_name = bot_name
        self.cache_minutes = cache_minutes
        self._cache: Optional[dict] = None
        self._cache_at: Optional[datetime] = None

        # supabase 라이브러리 lazy import
        try:
            from supabase import create_client
            self.client = create_client(supabase_url, supabase_key)
        except Exception as e:
            logger.error("[위험감지] Supabase 클라이언트 초기화 실패: %s", e)
            self.client = None

    # ════════════════════════════════════════════════════════════════
    # 공개 API
    # ════════════════════════════════════════════════════════════════

    def get_current_level(self) -> str:
        """현재 위험 등급 반환 (NORMAL/CAUTION/WARNING/DANGER/CRISIS)

        Supabase 조회 실패 시 "NORMAL" 반환 (운영 중단 방지)
        """
        status = self._get_cached_status()
        if not status:
            return "NORMAL"
        return status["level"] if "level" in status else "NORMAL"

    def get_current_level_kr(self) -> str:
        """현재 위험 등급 한글 (정상/주의/경고/위험/위기)"""
        status = self._get_cached_status()
        if not status:
            return "정상"
        return status["level_kr"] if "level_kr" in status else "정상"

    def get_position_multiplier(self) -> float:
        """매수금액 곱하기 계수 반환 (0.0 ~ 1.0)

        퀀트봇: CRISIS=0.2, DANGER=0.4, WARNING=0.6, CAUTION=0.8, NORMAL=1.0
        단타봇: DANGER/CRISIS=0.0 (차단), WARNING=0.5, CAUTION=0.7, NORMAL=1.0

        사용 예:
            actual_amount = int(base_amount * client.get_position_multiplier())
        """
        level = self.get_current_level()
        table = (
            self.POSITION_SIZE_DAYTRADING
            if self.bot_name == "daytrading"
            else self.POSITION_SIZE_QUANT
        )
        return table[level] if level in table else 1.0

    def should_block_new_entry(self) -> bool:
        """신규 매수를 차단해야 하는가?

        퀀트봇: 위기(CRISIS)만 차단
        단타봇: 위험(DANGER) 이상 차단

        Returns:
            True = 차단해야 함, False = 매수 진행 가능
        """
        level = self.get_current_level()
        block_levels = (
            self.DAYTRADING_BLOCK_LEVELS
            if self.bot_name == "daytrading"
            else self.QUANT_BLOCK_LEVELS
        )
        return level in block_levels

    def should_force_close_positions(self) -> bool:
        """보유 포지션 강제 청산해야 하는가? (위기 구간만)

        Returns:
            True = 위기 구간이라 모든 포지션 청산 검토
        """
        return self.get_current_level() == "CRISIS"

    def get_full_status(self) -> Optional[dict]:
        """현재 위험 상태 전체 조회 (로그/알림용)

        Returns:
            {
              'date': '2026-05-15',
              'total_score': 78.0,
              'level': 'DANGER',
              'level_kr': '위험',
              'external_score': 15.0,
              'foreign_flow_score': 25.0,
              'event_score': 14.0,
              'decoupling_score': 24.0,
              'key_signals': [...],
              'recommended_action': '...',
              'components': {...},
            }
            또는 None (조회 실패 시)
        """
        return self._get_cached_status()

    def get_key_signals(self) -> list:
        """핵심 위험 시그널 리스트 (텍스트)"""
        status = self._get_cached_status()
        if not status:
            return []
        signals = status["key_signals"] if "key_signals" in status else []
        return signals if isinstance(signals, list) else []

    def get_recommended_action(self) -> str:
        """등급별 권장 행동 문구"""
        status = self._get_cached_status()
        if not status:
            return "정상 운영. 모든 봇 평소대로."
        return status["recommended_action"] if "recommended_action" in status else ""

    def force_refresh(self) -> None:
        """캐시 강제 무효화 — 다음 호출 시 Supabase 재조회"""
        self._cache = None
        self._cache_at = None

    # ════════════════════════════════════════════════════════════════
    # P0-5: 신뢰 분리 — 봇이 자체 검증할 수 있는 도구
    # ════════════════════════════════════════════════════════════════

    def get_verification_payload(self) -> dict:
        """봇이 자체 검증에 사용할 raw evidence + 출처 + 산출 근거

        Returns:
            {
              'evidence': {외부환경/외인수급/이벤트/디커플링 raw값들},
              'data_sources': {각 데이터 출처/수집시각/나이},
              'score_explanations': {점수 산출 근거 문자열들},
              'data_freshness': {신선도 + 경고},
            }
            또는 정보 없으면 빈 dict
        """
        status = self._get_cached_status()
        if not status:
            return {}
        return {
            "evidence":           status["evidence"]           if "evidence"           in status else {},
            "data_sources":       status["data_sources"]       if "data_sources"       in status else {},
            "score_explanations": status["score_explanations"] if "score_explanations" in status else {},
            "data_freshness":     status["data_freshness"]     if "data_freshness"     in status else {},
        }

    def verify_with_local_data(
        self,
        local_usd_krw: Optional[float] = None,
        local_foreign_flow_oku: Optional[float] = None,
        local_kospi_change_pct: Optional[float] = None,
        local_vix: Optional[float] = None,
    ) -> dict:
        """봇 자체 데이터와 정보봇 데이터 일치 여부 검증

        Args (모두 선택 — 봇이 수집 가능한 것만 전달):
            local_usd_krw: 환율 (원)
            local_foreign_flow_oku: 외국인 일일 수급 (KOSPI+KOSDAQ 합계, 억원)
            local_kospi_change_pct: KOSPI 변동률 (%)
            local_vix: VIX

        Returns:
            {
              'agreement_score': 0.0~1.0 (1.0=완전 일치),
              'verified_count': int (일치한 항목 수),
              'total_checked': int (검증한 항목 수),
              'discrepancies': list[str] (불일치 설명),
              'recommendation': 'trust' / 'verify_further' / 'use_fallback',
              'data_age_hours': float (정보봇 데이터 나이),
            }

        봇 측 사용 패턴:
            v = risk_gate.verify_with_local_data(
                local_usd_krw=my_data['usd_krw'],
                local_foreign_flow_oku=my_data['foreign'],
            )
            if v['recommendation'] != 'trust':
                # 불일치 → 보수적 동작
                multiplier *= 0.5
        """
        info = self.get_full_status()
        if not info or "evidence" not in info:
            return {
                "agreement_score": 0.0,
                "verified_count": 0,
                "total_checked": 0,
                "discrepancies": ["정보봇 evidence 데이터 없음"],
                "recommendation": "use_fallback",
                "data_age_hours": None,
            }

        evidence = info.get("evidence") if isinstance(info.get("evidence"), dict) else {}
        ext = evidence["external"]     if "external"     in evidence else {}
        flow = evidence["foreign_flow"] if "foreign_flow" in evidence else {}
        deco = evidence["decoupling"]  if "decoupling"  in evidence else {}

        verified = 0
        total = 0
        discrepancies = []

        # 환율 비교 (±0.5% 허용 — 환율은 변동성 작음)
        if local_usd_krw is not None and ext.get("usd_krw"):
            total += 1
            ib = ext["usd_krw"]
            diff_pct = abs(local_usd_krw - ib) / ib * 100
            if diff_pct <= 0.5:
                verified += 1
            else:
                discrepancies.append(
                    f"환율 불일치: 봇={local_usd_krw:.2f} vs 정보봇={ib:.2f} ({diff_pct:.2f}% 차이)"
                )

        # 외인수급 비교 (±10% 허용 — 데이터 소스에 따라 약간 다름)
        if local_foreign_flow_oku is not None and flow.get("daily_won_oku") is not None:
            total += 1
            ib = flow["daily_won_oku"]
            denom = max(abs(ib), 1)
            diff_pct = abs(local_foreign_flow_oku - ib) / denom * 100
            if diff_pct <= 10:
                verified += 1
            else:
                discrepancies.append(
                    f"외인수급 불일치: 봇={local_foreign_flow_oku:.0f}억 "
                    f"vs 정보봇={ib:.0f}억 ({diff_pct:.1f}% 차이)"
                )

        # KOSPI 변동률 (±0.1%p 허용)
        if local_kospi_change_pct is not None and deco.get("kospi_chg") is not None:
            total += 1
            ib = deco["kospi_chg"]
            diff = abs(local_kospi_change_pct - ib)
            if diff <= 0.1:
                verified += 1
            else:
                discrepancies.append(
                    f"KOSPI 변동률 불일치: 봇={local_kospi_change_pct:+.2f}% "
                    f"vs 정보봇={ib:+.2f}% ({diff:.2f}%p 차이)"
                )

        # VIX (±5% 허용)
        if local_vix is not None and ext.get("vix"):
            total += 1
            ib = ext["vix"]
            diff_pct = abs(local_vix - ib) / ib * 100
            if diff_pct <= 5:
                verified += 1
            else:
                discrepancies.append(
                    f"VIX 불일치: 봇={local_vix:.2f} vs 정보봇={ib:.2f} ({diff_pct:.1f}% 차이)"
                )

        agreement = (verified / total) if total > 0 else 0.0

        # 신선도 가산
        freshness = info["data_freshness"] if isinstance(info.get("data_freshness"), dict) else {}
        max_age = max(
            freshness["macro_age_hours"]  if freshness.get("macro_age_hours")  is not None else 0,
            freshness["supply_age_hours"] if freshness.get("supply_age_hours") is not None else 0,
        )
        if freshness.get("has_warnings"):
            for w in (freshness["warnings"] if "warnings" in freshness else []):
                discrepancies.append(f"신선도 경고: {w}")

        # 추천 등급
        if total == 0:
            recommendation = "verify_further"  # 봇이 전달한 데이터 없음
        elif agreement >= 0.8 and not freshness.get("has_warnings"):
            recommendation = "trust"
        elif agreement >= 0.5:
            recommendation = "verify_further"
        else:
            recommendation = "use_fallback"

        return {
            "agreement_score": round(agreement, 2),
            "verified_count": verified,
            "total_checked": total,
            "discrepancies": discrepancies,
            "recommendation": recommendation,
            "data_age_hours": round(max_age, 1) if max_age else 0,
        }

    def is_data_fresh(self, max_age_hours: float = 36.0) -> bool:
        """정보봇 데이터가 충분히 신선한가? (기본 36시간 이내)

        36시간으로 잡은 이유:
            - 정보봇은 매일 16:49 갱신 (24시간 주기)
            - 다음날 아침 매매 직전엔 ~15시간 경과
            - 36시간 = 1.5일 = 정보봇 1회 누락 허용

        Returns:
            True = 신선 (사용 가능), False = 오래됨 (보수 모드 또는 fallback)
        """
        status = self._get_cached_status()
        if not status:
            return False
        freshness = status["data_freshness"] if isinstance(status.get("data_freshness"), dict) else {}
        macro_age = freshness["macro_age_hours"]  if freshness.get("macro_age_hours")  is not None else 999
        supply_age = freshness["supply_age_hours"] if freshness.get("supply_age_hours") is not None else 999
        return macro_age <= max_age_hours and supply_age <= max_age_hours

    def get_data_age_hours(self) -> dict:
        """각 데이터 소스의 나이 (시간 단위)"""
        status = self._get_cached_status()
        if not status:
            return {}
        return status["data_freshness"] if isinstance(status.get("data_freshness"), dict) else {}

    # ════════════════════════════════════════════════════════════════
    # MSCI 차단목록 (msci_blacklist 테이블)
    # ════════════════════════════════════════════════════════════════

    def is_msci_blacklisted(self, ticker: str) -> bool:
        """ticker가 현재 활성 MSCI 편출 차단목록에 있는가?

        봇이 매수 직전 호출:
            if risk_gate.is_msci_blacklisted(ticker):
                return  # 5/29 강제매도 종목 회피

        Returns:
            True = 차단, False = 매수 가능
        """
        blacklist = self._get_cached_msci_exclusion()
        return ticker in blacklist

    def get_msci_exclusion_list(self) -> list[str]:
        """현재 활성 편출 종목 ticker 리스트"""
        return list(self._get_cached_msci_exclusion())

    def get_msci_inclusion_list(self) -> list[str]:
        """현재 활성 편입 종목 ticker 리스트 (강제 매수 수혜 → 보유 유리)"""
        return list(self._get_cached_msci_inclusion())

    def _get_cached_msci_exclusion(self) -> set:
        """편출 ticker 집합 (10분 캐시)"""
        return self._get_cached_msci("exclusion")

    def _get_cached_msci_inclusion(self) -> set:
        return self._get_cached_msci("inclusion")

    def _get_cached_msci(self, type_: str) -> set:
        """MSCI 차단/편입 ticker 집합 조회 (10분 캐시)"""
        now = datetime.utcnow()
        cache_attr = f"_msci_{type_}_cache"
        cache_at_attr = f"_msci_{type_}_cache_at"
        cached = getattr(self, cache_attr, None)
        cached_at = getattr(self, cache_at_attr, None)
        if (
            cached is not None
            and cached_at is not None
            and (now - cached_at) < timedelta(minutes=self.cache_minutes)
        ):
            return cached

        tickers = set()
        if self.client is not None:
            try:
                today_str = now.date().isoformat()
                r = (
                    self.client.table("msci_blacklist")
                    .select("ticker")
                    .eq("type", type_)
                    .eq("is_active", True)
                    .lte("ann_date", today_str)
                    .gte("expires_at", today_str)
                    .execute()
                )
                for row in (r.data or []):
                    if "ticker" in row and row["ticker"]:
                        tickers.add(row["ticker"])
            except Exception as e:
                logger.warning("[위험감지] MSCI 차단목록 조회 실패: %s", e)

        setattr(self, cache_attr, tickers)
        setattr(self, cache_at_attr, now)
        return tickers

    def is_blacklisted_for_exhaustion(self, foreign_exhaustion_rate: float) -> bool:
        """외인소진율 기반 종목 차단 여부

        위험 구간(70+)에서 외인소진율 50%+ 종목은 자동 차단:
        5/15 유진테크 -19.15%, 티씨케이 -12.48% 같은 패닉셀 회피

        Args:
            foreign_exhaustion_rate: 외인소진율 (%, 0~100)

        Returns:
            True = 차단해야 함
        """
        level = self.get_current_level()
        if level == "NORMAL":
            return False
        # 등급별 외인소진율 차단 임계값
        threshold = {
            "CAUTION": 80,   # 매우 높은 종목만
            "WARNING": 60,
            "DANGER":  50,
            "CRISIS":  30,   # 위기 구간은 보수적
        }
        return foreign_exhaustion_rate >= (threshold[level] if level in threshold else 100)

    # ════════════════════════════════════════════════════════════════
    # 내부: 캐시 + Supabase 조회
    # ════════════════════════════════════════════════════════════════

    def _get_cached_status(self) -> Optional[dict]:
        """캐시된 상태 반환 (10분 TTL)"""
        now = datetime.utcnow()
        if (
            self._cache is not None
            and self._cache_at is not None
            and (now - self._cache_at) < timedelta(minutes=self.cache_minutes)
        ):
            return self._cache

        # 캐시 미스 또는 만료 → Supabase 조회
        status = self._fetch_latest_from_supabase()
        if status:
            self._cache = status
            self._cache_at = now
        return status

    def _fetch_latest_from_supabase(self) -> Optional[dict]:
        """Supabase에서 가장 최근 위험점수 조회"""
        if self.client is None:
            logger.warning("[위험감지] Supabase 클라이언트 없음 — NORMAL fallback")
            return None
        try:
            r = (
                self.client.table("macro_risk_daily")
                .select("*")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if r.data:
                row = r.data[0]
                logger.debug(
                    "[위험감지] %s %s점 (%s) 조회",
                    row["date"] if "date" in row else "?",
                    row["total_score"] if "total_score" in row else "?",
                    row["level_kr"] if "level_kr" in row else "?",
                )
                return row
            logger.warning("[위험감지] macro_risk_daily 데이터 없음")
            return None
        except Exception as e:
            logger.error("[위험감지] Supabase 조회 실패: %s", e)
            return None


# ════════════════════════════════════════════════════════════════
# 편의 함수 — 인스턴스 없이 1회성 조회
# ════════════════════════════════════════════════════════════════

def quick_check(bot_name: str = "quant") -> dict:
    """환경변수에서 Supabase URL/KEY 읽어 1회성 상태 조회

    환경변수: SUPABASE_URL, SUPABASE_KEY

    Returns:
        {
          'level': str, 'level_kr': str, 'multiplier': float,
          'block_new_entry': bool, 'status': dict
        }
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "") or os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return {
            "level": "NORMAL",
            "level_kr": "정상",
            "multiplier": 1.0,
            "block_new_entry": False,
            "status": None,
            "error": "SUPABASE_URL or SUPABASE_KEY 환경변수 없음",
        }
    client = RiskGateClient(url, key, bot_name=bot_name)
    return {
        "level": client.get_current_level(),
        "level_kr": client.get_current_level_kr(),
        "multiplier": client.get_position_multiplier(),
        "block_new_entry": client.should_block_new_entry(),
        "status": client.get_full_status(),
    }
