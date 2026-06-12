"""L2 사전 게이트 — 엔진의 심장 (docs/01-plan/RISK_ENGINE_SPEC_v2.md §3.5, Phase 1a).

설계 철학 (§0 — 협상 불가):
1. 수익은 결과, 생존이 목표 — 이 게이트의 목적은 "계좌가 죽지 않는 구조"다.
2. 리스크 관리는 사후 보고서가 아니라 사전 게이트 — 주문은 통과해야만 나간다.
3. 모든 모델은 틀린다 — 마지막 층은 모델이 아니라 하드 룰이다.
4. 노출(exposure)이 아니라 리스크(risk)를 본다.

Phase 1a 범위 = 정적 한도 G3~G6만 활성:
  G3 단일 종목 비중 ≤ max_single_weight(12%)   → 위반 시 RESIZE(비례 축소 재검사)
  G4 단일 섹터 비중 ≤ max_sector_weight(30%)   → 위반 시 REJECT
  G5 상관 클러스터(ρ ≥ 0.8 보유 합산 = 유효 단일 포지션, G3 한도 적용) → 위반 시 REJECT
  G6 유동성(주문금액 ≤ ADV20 × 5%, 데이터 없으면 fail-closed REJECT) → 위반 시 REJECT
  G1/G2(VaR)=Phase 2, G8(드로다운 사다리)=Phase 3, G7(Component VaR)=Phase 4
  — checks dict에 'not_active'로만 명시(자리 선등록).

쓰기 3원칙 (6/11 레포 표준):
  ① 안전한 기본값 — log_dir=None(기본)이면 순수 평가, 파일쓰기 등 부작용 0.
  ② 기록은 KST ISO 타임스탬프로 자기상태 선언 — JSONL 1줄 append.
  ③ fail-closed — 모르면 차단(유동성 데이터 없음 / HMAC 키 없음 / 감사 로그 기록 실패
    / 잘못된 입력 / 토큰 만료·미래 발급 전부 차단 방향).

★이 모듈은 판정 + 토큰 발급만 한다. 실주문 경로 접촉 0 — execution 배선은 Phase 1b이며,
  배선 측은 verify_gate_token() 통과 없이는 주문을 낼 수 없도록 구조적으로 강제된다.
  '게이트 로그 없는 체결 = Critical 버그' — 따라서 log_dir이 지정됐는데 기록에 실패하면
  결과 자체를 REJECT로 강등한다(감사 추적 없는 PASS는 존재할 수 없다).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import numbers
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from risk.config import KST, RISK_CONFIG, RiskConfig

# ── 상수 ──────────────────────────────────────────────────────────────────────
_HMAC_ENV_KEY = "ORDER_INTENTS_HMAC_KEY"   # 토큰 서명 키 환경변수 (Phase 1b 배선과 공유)
_UNKNOWN_SECTOR = "UNKNOWN"                # 섹터 미상 버킷 — fail-closed로 한 버킷에 합산
_W_EPS = 1e-12                             # 비중 비교 부동소수 여유
_KRW_EPS = 1e-6                            # 금액 비교 부동소수 여유 (KRW)

# 비활성 게이트(자리 선등록) — Phase 도래 전까지 checks에 명시만 한다.
_NOT_ACTIVE_GATES: dict[str, str] = {
    "G1": "not_active (Phase 2: 포트폴리오 1D VaR95)",
    "G2": "not_active (Phase 2: 스트레스 VaR95)",
    "G7": "not_active (Phase 4: Component VaR 기여)",
    "G8": "not_active (Phase 3: 드로다운 사다리)",
}

_TOKEN_OK_VERDICTS = ("PASS", "RESIZE")


# ── 공개 데이터 계약 ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Holding:
    """기존 보유 1건. corr_with_new=신규 종목과의 상관(None=클러스터 판정 제외)."""

    ticker: str
    value_krw: float
    sector: str | None = None
    corr_with_new: float | None = None


@dataclass(frozen=True)
class GateRequest:
    """신규 주문 의도 1건에 대한 게이트 평가 요청."""

    ticker: str
    sector: str | None
    proposed_size_krw: float
    equity_krw: float
    adv20_krw: float | None


@dataclass
class GateResult:
    """게이트 판정 결과. verdict ∈ {'PASS','RESIZE','REJECT'}.

    ticker는 계약 외 보조 필드 — verify_gate_token()이 결과 객체 단독으로
    서명을 재계산할 수 있도록 보존한다(자기완결 감사 단위).
    """

    verdict: str
    final_size_krw: float
    original_size_krw: float
    violations: list[dict] = field(default_factory=list)
    resize_iterations: int = 0
    checks: dict = field(default_factory=dict)
    issued_at: str = ""
    token: str | None = None
    signed: bool = False
    ticker: str = ""
    nonce: str = ""   # 토큰 1회성(replay 방지)용 고유 식별자 — Phase 1b가 seen_nonces로 추적


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────
def _ensure_kst(dt: datetime) -> datetime:
    """주입된 datetime을 KST aware로 정규화. naive면 KST로 간주(테스트 주입 편의)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def _is_finite(x) -> bool:
    """유한한 실수인지 (None/bool/NaN/inf 차단 — fail-closed 입력 위생).

    np.int64/np.float64(pandas 산출) 허용을 위해 numbers.Real로 본다(정상 입력 오거부 방지).
    bool은 금액이 아니므로 명시 제외.
    """
    if x is None or isinstance(x, bool):
        return False
    return isinstance(x, numbers.Real) and math.isfinite(float(x))


def _validate_inputs(request: GateRequest, holdings: list[Holding]) -> str | None:
    """입력 위생 검사. 문제 있으면 사유 문자열 반환(없으면 None). fail-closed."""
    if not request.ticker or not isinstance(request.ticker, str):
        return "ticker_empty"
    if not _is_finite(request.equity_krw) or request.equity_krw <= 0:
        return "equity_krw_not_positive"
    if not _is_finite(request.proposed_size_krw) or request.proposed_size_krw <= 0:
        return "proposed_size_krw_not_positive"
    for h in holdings:
        # 음수/비유한 보유금액은 섹터·클러스터 합산을 왜곡(리스크 과소평가)하므로 차단.
        if not _is_finite(h.value_krw) or h.value_krw < 0:
            return f"holding_value_invalid:{h.ticker}"
        if h.corr_with_new is not None and not _is_finite(h.corr_with_new):
            return f"holding_corr_invalid:{h.ticker}"
    return None


def _inactive_checks() -> dict:
    """비활성 게이트(G1/G2/G7/G8) 자리만 채운 checks 골격."""
    return {g: {"status": _NOT_ACTIVE_GATES[g]} for g in ("G1", "G2", "G7", "G8")}


def _run_static_checks(
    size_krw: float,
    request: GateRequest,
    holdings: list[Holding],
    cfg: RiskConfig,
) -> tuple[dict, list[dict], dict | None]:
    """현재 평가 사이즈로 가상 포트폴리오(기존+신규)를 재계산해 G3~G6 검사.

    반환: (checks, reject급 위반 목록[G4/G5/G6], G3 위반 dict 또는 None).
    G3만 RESIZE 대상이므로 분리해서 돌려준다.
    """
    equity = float(request.equity_krw)
    checks: dict = {}
    # 게이트 번호 순서대로 기록 (감사 로그 가독성)
    checks["G1"] = {"status": _NOT_ACTIVE_GATES["G1"]}
    checks["G2"] = {"status": _NOT_ACTIVE_GATES["G2"]}

    reject_violations: list[dict] = []
    g3_violation: dict | None = None

    # G3 — 단일 종목 비중 (같은 ticker 기보유분 합산: 가상 포트폴리오 전체 재계산)
    same_ticker_value = sum(h.value_krw for h in holdings if h.ticker == request.ticker)
    g3_weight = (same_ticker_value + size_krw) / equity
    g3_ok = g3_weight <= cfg.max_single_weight + _W_EPS
    checks["G3"] = {
        "status": "pass" if g3_ok else "violation",
        "weight": g3_weight,
        "limit": cfg.max_single_weight,
        "existing_same_ticker_krw": same_ticker_value,
    }
    if not g3_ok:
        g3_violation = {
            "gate": "G3",
            "reason": "single_weight_limit",
            "weight": g3_weight,
            "limit": cfg.max_single_weight,
        }

    # G4 — 단일 섹터 비중 (sector None → 'UNKNOWN' 버킷으로 fail-closed 합산)
    bucket = request.sector if request.sector else _UNKNOWN_SECTOR
    sector_value = sum(
        h.value_krw for h in holdings if (h.sector if h.sector else _UNKNOWN_SECTOR) == bucket
    )
    g4_weight = (sector_value + size_krw) / equity
    g4_ok = g4_weight <= cfg.max_sector_weight + _W_EPS
    checks["G4"] = {
        "status": "pass" if g4_ok else "violation",
        "sector": bucket,
        "weight": g4_weight,
        "limit": cfg.max_sector_weight,
    }
    if not g4_ok:
        reject_violations.append({
            "gate": "G4",
            "reason": "sector_weight_limit",
            "sector": bucket,
            "weight": g4_weight,
            "limit": cfg.max_sector_weight,
        })

    # G5 — 상관 클러스터: ρ ≥ threshold 보유 합산 + 신규 = '유효 단일 포지션' → G3 한도 적용
    cluster = [
        h for h in holdings
        if h.corr_with_new is not None and h.corr_with_new >= cfg.corr_cluster_threshold
    ]
    cluster_value = sum(h.value_krw for h in cluster)
    g5_weight = (cluster_value + size_krw) / equity
    g5_ok = (not cluster) or g5_weight <= cfg.max_single_weight + _W_EPS
    # 상관 미상(corr_with_new=None) 보유는 클러스터 판정에서 빠진다(fail-open 소지) —
    # 그 수를 감사 로그에 투명하게 노출(적대리뷰 P2). Phase 1b가 이 수를 보고 상관 계산을 강제.
    unknown_corr_count = sum(1 for h in holdings if h.corr_with_new is None)
    checks["G5"] = {
        "status": "pass" if g5_ok else "violation",
        "cluster_tickers": [h.ticker for h in cluster],
        "cluster_value_krw": cluster_value,
        "weight": g5_weight if cluster else None,
        "limit": cfg.max_single_weight,
        "threshold": cfg.corr_cluster_threshold,
        "unknown_corr_count": unknown_corr_count,
    }
    if not g5_ok:
        reject_violations.append({
            "gate": "G5",
            "reason": "corr_cluster_limit",
            "cluster_tickers": [h.ticker for h in cluster],
            "weight": g5_weight,
            "limit": cfg.max_single_weight,
        })

    # G6 — 유동성: 주문금액 ≤ ADV20 × adv_limit_ratio. 데이터 없음/비정상 → fail-closed REJECT
    adv = request.adv20_krw
    if not _is_finite(adv) or float(adv) <= 0:  # type: ignore[arg-type]
        checks["G6"] = {"status": "violation", "reason": "no_liquidity_data", "adv20_krw": adv}
        reject_violations.append({"gate": "G6", "reason": "no_liquidity_data", "adv20_krw": adv})
    else:
        adv_cap_krw = float(adv) * cfg.adv_limit_ratio
        g6_ok = size_krw <= adv_cap_krw + _KRW_EPS
        checks["G6"] = {
            "status": "pass" if g6_ok else "violation",
            "size_krw": size_krw,
            "adv_cap_krw": adv_cap_krw,
            "adv20_krw": float(adv),
        }
        if not g6_ok:
            reject_violations.append({
                "gate": "G6",
                "reason": "adv_liquidity_limit",
                "size_krw": size_krw,
                "adv_cap_krw": adv_cap_krw,
            })

    checks["G7"] = {"status": _NOT_ACTIVE_GATES["G7"]}
    checks["G8"] = {"status": _NOT_ACTIVE_GATES["G8"]}
    return checks, reject_violations, g3_violation


def _sanitize_json(obj):
    """NaN/inf를 None으로 정제(재귀) — JSONL이 RFC 준수 라인이 되게 한다(적대리뷰 P2).

    json.dumps 기본은 NaN→'NaN'(RFC 비준수)을 내보내 엄격 파서가 라인 전체를 못 읽는다.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _sign_token(key: str, ticker: str, final_size_krw: float, issued_at: str,
                verdict: str, nonce: str) -> str:
    """'ticker|round(final_size)|issued_at|verdict|nonce' HMAC-SHA256 hex (Phase 1b 검증용).

    nonce 포함 → 같은 (ticker,size,time,verdict)라도 매 발급 토큰이 달라지고,
    verify가 nonce로 1회성(replay)을 추적할 수 있다(적대리뷰 P1: replay 방지).
    """
    msg = f"{ticker}|{round(final_size_krw)}|{issued_at}|{verdict}|{nonce}"
    return hmac.new(key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()


def _append_audit_log(
    log_dir: Path,
    result: GateResult,
    request: GateRequest,
    holdings: list[Holding],
    now: datetime,
) -> None:
    """감사 로그 1줄 append — {log_dir}/gate_log_YYYYMMDD.jsonl (KST 날짜·타임스탬프).

    토큰 자체는 기록하지 않는다(서명 비밀 유사물) — signed 플래그만 남긴다.
    """
    record = {
        "ts": result.issued_at,  # KST ISO — 기록의 자기상태 선언
        "request": {
            "ticker": request.ticker,
            "sector": request.sector,
            "proposed_size_krw": request.proposed_size_krw,
            "equity_krw": request.equity_krw,
            "adv20_krw": request.adv20_krw,
        },
        "holdings_summary": {
            "count": len(holdings),
            "total_value_krw": sum(h.value_krw for h in holdings),
        },
        "checks": result.checks,
        "verdict": result.verdict,
        "original_size_krw": result.original_size_krw,
        "final_size_krw": result.final_size_krw,
        "violations": result.violations,
        "resize_iterations": result.resize_iterations,
        "signed": result.signed,
    }
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"gate_log_{now.strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        # NaN/inf → None 정제 후 allow_nan=False — RFC 비준수 'NaN' 토큰이 엄격 파서를 깨는 것 방지.
        f.write(json.dumps(_sanitize_json(record), ensure_ascii=False, allow_nan=False, default=str) + "\n")


# ── 공개 API ─────────────────────────────────────────────────────────────────
def evaluate_pre_trade(
    request: GateRequest,
    holdings: list[Holding],
    cfg: RiskConfig = RISK_CONFIG,
    log_dir: Path | None = None,
    hmac_key: str | None = None,
    now_kst: datetime | None = None,
) -> GateResult:
    """신규 주문 의도를 가상 포트폴리오(기존+신규) 기준으로 G3~G6 검사해 판정한다.

    판정:
      PASS   — 전 항목 통과 (원래 사이즈 그대로)
      RESIZE — G3 위반 → 한도 내로 비례 축소 후 재검사 통과 (최대 resize_max_iter회,
               축소 결과가 min_trade_krw 미만이면 REJECT)
      REJECT — G4/G5/G6 위반, RESIZE 실패, 잘못된 입력, 감사 로그 기록 실패 (fail-closed)

    부작용: log_dir=None(기본)이면 0. log_dir 지정 시 JSONL 감사 로그 1줄 append —
    기록 실패 시 verdict를 REJECT로 강등한다('게이트 로그 없는 체결=Critical 버그').
    토큰: ★감사 로그 기록 성공 후에만 발급된다 — log_dir 지정 + 기록 성공 + verdict PASS/RESIZE
    + hmac_key(기본=환경변수 ORDER_INTENTS_HMAC_KEY) 존재. log_dir=None(순수 평가)이면 토큰
    없음(무감사 PASS 차단). 매 평가 고유 nonce 부여 — verify가 seen_nonces로 replay 추적.
    now_kst: 테스트 주입용. None이면 datetime.now(KST). naive 주입 시 KST로 간주.
    """
    now = _ensure_kst(now_kst) if now_kst is not None else datetime.now(KST)
    issued_at = now.isoformat()
    original = float(request.proposed_size_krw) if _is_finite(request.proposed_size_krw) else 0.0

    # 0) 입력 위생 — fail-closed 즉시 REJECT
    invalid_reason = _validate_inputs(request, holdings)
    if invalid_reason is not None:
        result = GateResult(
            verdict="REJECT",
            final_size_krw=0.0,
            original_size_krw=original,
            violations=[{"gate": "G0", "reason": "invalid_input", "detail": invalid_reason}],
            resize_iterations=0,
            checks=_inactive_checks(),
            issued_at=issued_at,
            token=None,
            signed=False,
            ticker=str(request.ticker) if request.ticker else "",
        )
        return _finalize(result, request, holdings, log_dir, now, hmac_key)

    # 1) G3 RESIZE 루프 (G4/G5/G6 위반은 즉시 REJECT — 축소로 구제하지 않는다, 스펙 §3.5)
    equity = float(request.equity_krw)
    same_ticker_value = sum(h.value_krw for h in holdings if h.ticker == request.ticker)
    size = original
    iterations = 0
    resize_trail: list[dict] = []  # 축소 이력 (감사 추적)

    while True:
        checks, reject_viols, g3_viol = _run_static_checks(size, request, holdings, cfg)

        if reject_viols:
            violations = resize_trail + reject_viols
            if g3_viol is not None:
                violations.append(g3_viol)  # 사유 전부 기록
            result = _build_result("REJECT", 0.0, original, violations, iterations,
                                   checks, issued_at, request)
            return _finalize(result, request, holdings, log_dir, now, hmac_key)

        if g3_viol is None:
            verdict = "PASS" if iterations == 0 else "RESIZE"
            result = _build_result(verdict, size, original, list(resize_trail), iterations,
                                   checks, issued_at, request)
            return _finalize(result, request, holdings, log_dir, now, hmac_key)

        # G3 위반 → 비례 축소 시도
        if iterations >= cfg.resize_max_iter:
            violations = resize_trail + [dict(g3_viol, reason="resize_failed")]
            result = _build_result("REJECT", 0.0, original, violations, iterations,
                                   checks, issued_at, request)
            return _finalize(result, request, holdings, log_dir, now, hmac_key)

        # 한도 내 허용 사이즈 = 한도비중×자본 − 같은 종목 기보유분
        allowed = cfg.max_single_weight * equity - same_ticker_value
        new_size = min(size, allowed)
        if new_size < cfg.min_trade_krw:
            violations = resize_trail + [dict(
                g3_viol,
                reason="resize_below_min_trade",
                resized_to_krw=new_size,
                min_trade_krw=cfg.min_trade_krw,
            )]
            result = _build_result("REJECT", 0.0, original, violations, iterations,
                                   checks, issued_at, request)
            return _finalize(result, request, holdings, log_dir, now, hmac_key)
        if new_size >= size - _KRW_EPS:
            # 축소가 진행되지 않으면(수치 안정성) 무의미 반복 대신 즉시 실패 — fail-closed
            violations = resize_trail + [dict(g3_viol, reason="resize_failed")]
            result = _build_result("REJECT", 0.0, original, violations, iterations,
                                   checks, issued_at, request)
            return _finalize(result, request, holdings, log_dir, now, hmac_key)

        resize_trail.append(dict(g3_viol, action="resized",
                                 from_krw=size, to_krw=new_size))
        size = new_size
        iterations += 1


def verify_gate_token(
    result: GateResult,
    hmac_key: str | None = None,
    max_age_sec: int = 300,
    now_kst: datetime | None = None,
    seen_nonces: set | None = None,
) -> bool:
    """Phase 1b가 주문 직전 호출 — 결과 객체 단독으로 토큰 유효성을 재검증한다.

    False 조건(전부 fail-closed): verdict가 PASS/RESIZE가 아님 / token 없음 / 키 없음 /
    서명 불일치 / issued_at 파싱 불가·naive / 발급 후 max_age_sec 초과 / 미래 발급(시계 역행) /
    nonce 비어있음 / seen_nonces에 이미 있음(replay).
    ★seen_nonces(set) 주입 시 토큰 1회성(replay)을 강제한다 — Phase 1b 배선이 영속 set을
      전달해 '동일 토큰으로 max_age 내 중복 주문'(적대리뷰 P1)을 막는다.
    """
    try:
        if result.verdict not in _TOKEN_OK_VERDICTS:
            return False
        if not result.token or not result.signed:
            return False
        key = hmac_key if hmac_key is not None else os.environ.get(_HMAC_ENV_KEY)
        if not key:
            return False
        expected = _sign_token(key, result.ticker, result.final_size_krw,
                               result.issued_at, result.verdict, result.nonce)
        if not hmac.compare_digest(expected, result.token):
            return False
        issued = datetime.fromisoformat(result.issued_at)
        if issued.tzinfo is None:
            return False  # naive 타임스탬프는 신뢰하지 않는다
        now = _ensure_kst(now_kst) if now_kst is not None else datetime.now(KST)
        age_sec = (now - issued).total_seconds()
        if not (0.0 <= age_sec <= max_age_sec):
            return False
        # ★replay 방지: seen_nonces 주입 시 1회성 강제(같은 토큰 두 번째 검증은 거부).
        if seen_nonces is not None:
            if not result.nonce or result.nonce in seen_nonces:
                return False
            seen_nonces.add(result.nonce)
        return True
    except Exception:
        return False  # 어떤 예외도 통과로 이어지지 않는다


# ── 내부: 결과 조립 + 마무리(감사 로그) ──────────────────────────────────────
def _build_result(
    verdict: str,
    final_size_krw: float,
    original_size_krw: float,
    violations: list[dict],
    resize_iterations: int,
    checks: dict,
    issued_at: str,
    request: GateRequest,
) -> GateResult:
    """판정 → GateResult 조립. ★토큰은 여기서 발급하지 않는다 — 감사 로그 기록 성공 후
    _finalize에서만 발급한다(감사 추적 없는 PASS는 존재할 수 없다, 적대리뷰 P1).
    여기선 매 평가마다 고유 nonce만 부여한다(replay 추적용)."""
    return GateResult(
        verdict=verdict,
        final_size_krw=final_size_krw,
        original_size_krw=original_size_krw,
        violations=violations,
        resize_iterations=resize_iterations,
        checks=checks,
        issued_at=issued_at,
        token=None,
        signed=False,
        ticker=request.ticker,
        nonce=uuid.uuid4().hex,
    )


def _finalize(
    result: GateResult,
    request: GateRequest,
    holdings: list[Holding],
    log_dir: Path | None,
    now: datetime,
    hmac_key: str | None,
) -> GateResult:
    """쓰기 3원칙 + 토큰 발급 지점.

    ★log_dir=None(순수 평가)이면 토큰을 발급하지 않는다 — 감사 추적 없는 PASS는
      존재할 수 없다(적대리뷰 P1: 무감사 서명 토큰 차단). 순수 평가 결과로는 주문 불가.
    log_dir 지정 시: 감사 로그 1줄 기록 → 성공하면 PASS/RESIZE에 토큰 발급, 실패하면
      REJECT로 강등('게이트 로그 없는 체결=Critical 버그').
    """
    if log_dir is None:
        return result  # 순수 평가 — 토큰 없음(주문 경로로 갈 수 없다)
    try:
        _append_audit_log(Path(log_dir), result, request, holdings, now)
    except Exception as exc:  # 감사 추적 없는 PASS는 존재할 수 없다
        result.verdict = "REJECT"
        result.final_size_krw = 0.0
        result.token = None
        result.signed = False
        result.violations.append({
            "gate": "AUDIT",
            "reason": "audit_log_write_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        })
        return result
    # ★감사 로그 기록 성공 후에만 토큰 발급 (PASS/RESIZE + 키 존재 시) — 무감사 PASS 차단
    if result.verdict in _TOKEN_OK_VERDICTS:
        key = hmac_key if hmac_key is not None else os.environ.get(_HMAC_ENV_KEY)
        if key:
            result.token = _sign_token(key, result.ticker, result.final_size_krw,
                                       result.issued_at, result.verdict, result.nonce)
            result.signed = True
    return result
