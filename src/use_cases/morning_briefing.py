"""장시작전 마켓 브리핑 텔레그램 발송 모듈.

origin: scripts/archive/legacy_wrappers/send_market_briefing.py (2026-05-17 이전)
이전 사유: CLAUDE.md LOCK 규칙(`scripts/archive/` 참조 금지) 위반 해소.
run_morning_briefing.py가 importlib 동적 로딩으로 우회하던 패턴을
정규 import로 전환하기 위해 정식 위치(src/use_cases/)로 이전.

데이터 소스:
  - data/us_market/overnight_signal.json (US overnight signal)
  - data/sector_rotation/relay_trading_signal.json (릴레이 순환매수)
  - data/sector_rotation/etf_trading_signal.json (ETF 매매 시그널)
  - data/sector_rotation/krx_sector_scan.json (스마트머니/테마머니)

수동 실행: python -m src.use_cases.morning_briefing [--send]
BAT-B 호출: scripts/run_morning_briefing.py
"""

import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime

logger = logging.getLogger(__name__)

SIGNAL_PATH = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
PRED_MODEL_PATH = PROJECT_ROOT / "data" / "us_market" / "kospi_pred_model.json"
RELAY_SIGNAL_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "relay_trading_signal.json"
ETF_SIGNAL_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "etf_trading_signal.json"
SECTOR_SCAN_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "krx_sector_scan.json"
STOCK_DATA_DIR = PROJECT_ROOT / "stock_data_daily"


# ─────────────────────────────────────────────
# 확률 계산 (로지스틱 회귀 모델)
# ─────────────────────────────────────────────

def _load_pred_model() -> dict | None:
    """kospi_pred_model.json 로드. 없으면 None."""
    if PRED_MODEL_PATH.exists():
        with open(PRED_MODEL_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def calc_market_probability(
    ewy_ret: float,
    vix: float,
    spy_ret: float,
    qqq_ret: float,
    soxx_ret: float = 0.0,
    vix_chg: float = 0.0,
    stoxx50_ret: float = 0.0,
) -> dict:
    """로지스틱 회귀 기반 KOSPI 상승/하락 확률.

    v2: 449일 학습, STOXX50 추가 / EWY 제거
    Features: soxx, vix_chg, vix_level, spy, qqq, stoxx50
    """
    # ── 비거래일 데이터 감지: 모든 주요 지수 0% → 50/50 반환 ──
    all_zero = (
        abs(ewy_ret) < 0.01 and abs(spy_ret) < 0.01
        and abs(qqq_ret) < 0.01 and abs(soxx_ret) < 0.01
    )
    if all_zero:
        return {
            "up_prob": 50,
            "down_prob": 50,
            "est_low": -0.3,
            "est_high": 0.3,
            "stale_data": True,
        }

    model = _load_pred_model()

    if model and model.get("coef"):
        features = model["features"]
        mean = model["scaler_mean"]
        scale = model["scaler_scale"]
        coef = model["coef"]
        intercept = model["intercept"]

        raw = {
            "ewy": ewy_ret, "soxx": soxx_ret, "vix_chg": vix_chg,
            "vix_level": vix, "spy": spy_ret, "qqq": qqq_ret,
            "stoxx50": stoxx50_ret,
        }

        logit = intercept
        for i, feat in enumerate(features):
            val = raw.get(feat, 0.0)
            scaled = (val - mean[i]) / scale[i] if scale[i] != 0 else 0
            logit += coef[i] * scaled

        up_prob = round(_sigmoid(logit) * 100)
    else:
        # 폴백: 단순 룰 기반
        up_prob = 50
        if soxx_ret > 2:
            up_prob += 12
        elif soxx_ret < -2:
            up_prob -= 12
        if spy_ret > 1:
            up_prob += 8
        elif spy_ret < -1:
            up_prob -= 8
        if vix > 25:
            up_prob -= 8
        up_prob = max(5, min(95, up_prob))

    down_prob = 100 - up_prob

    # 예상 레인지 (SOXX + SPY + STOXX50 가중)
    base = soxx_ret * 0.35 + spy_ret * 0.35 + stoxx50_ret * 0.3
    if abs(base) > 0.1:
        est_low = round(base * 0.5, 1)
        est_high = round(base * 1.1, 1)
    else:
        est_low, est_high = -0.3, 0.3

    # 정합성 보정: up_prob와 est 방향 모순 방지
    est_lo = min(est_low, est_high)
    est_hi = max(est_low, est_high)

    if up_prob >= 60 and est_hi < 0:
        est_lo = round(base * 0.3, 1) if base > 0 else -0.2
        est_hi = round(base * 1.0, 1) if base > 0 else 0.5
    elif down_prob >= 60 and est_lo > 0:
        est_lo = round(base * 1.0, 1) if base < 0 else -0.5
        est_hi = round(base * 0.3, 1) if base < 0 else 0.2

    return {
        "up_prob": up_prob,
        "down_prob": down_prob,
        "est_low": min(est_lo, est_hi),
        "est_high": max(est_lo, est_hi),
    }


def make_prob_bar(up_pct: int, width: int = 20) -> str:
    """상승 확률 시각화 바. 채운 부분=상승."""
    filled = round(up_pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_overnight_signal() -> dict:
    return _load_json(SIGNAL_PATH)


def load_relay_signals() -> list:
    """릴레이 시그널 로드. HIGH confidence만 진입적기."""
    data = _load_json(RELAY_SIGNAL_PATH)
    signals = data.get("signals", [])
    return [s for s in signals if s.get("confidence") == "HIGH"]


def load_etf_signals() -> dict:
    """ETF 매매 시그널 로드."""
    data = _load_json(ETF_SIGNAL_PATH)
    return {
        "buy": data.get("smart_money_etf", []) + data.get("theme_money_etf", []),
        "watch": data.get("watch_list", []),
    }


def load_scan_results() -> dict:
    """섹터 로테이션 스캔 결과 로드."""
    data = _load_json(SECTOR_SCAN_PATH)
    return {
        "smart_money": data.get("smart_money", []),
        "theme_money": data.get("theme_money", []),
    }


def _grade_stock(item: dict, money_type: str) -> str:
    """종목 등급 판정."""
    bb = item.get("bb_pct", 50)
    rsi = item.get("rsi", 50)
    adx = item.get("adx", 0)
    gx = item.get("stoch_golden_recent", False)

    if money_type == "SMART":
        if bb < 30 and rsi < 45:
            return "S"
        elif bb < 50 and rsi < 55:
            return "A"
        return "B"
    else:
        if adx > 50:
            return "S"
        elif adx > 40 or gx:
            return "A"
        return "B"


# ─────────────────────────────────────────────
# 메시지 빌더
# ─────────────────────────────────────────────

def _fetch_stoxx50_ret() -> float:
    """전일 STOXX50 수익률 가져오기 (yfinance)."""
    try:
        import yfinance as yf
        d = yf.download("^STOXX50E", period="5d", progress=False)
        if len(d) >= 2:
            d.columns = d.columns.droplevel(1) if d.columns.nlevels > 1 else d.columns
            closes = d["Close"].dropna()
            if len(closes) >= 2:
                ret = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
                return round(float(ret), 2)
    except Exception as e:
        logger.warning("STOXX50 수집 실패: %s", e)
    return 0.0


def build_briefing_message() -> str:
    """노션 스타일 장전 마켓 브리핑 메시지."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === 데이터 로드 ===
    sig = load_overnight_signal()
    idx = sig.get("index_direction", {})

    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)
    ewy_5d = idx.get("EWY", {}).get("ret_5d", 0)
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    dia_ret = idx.get("DIA", {}).get("ret_1d", 0)
    vix_data = sig.get("vix", {})
    vix = vix_data.get("level", 20)
    vix_status = vix_data.get("status", "정상")
    us_grade = sig.get("grade", "NEUTRAL")
    combined = sig.get("combined_score_100", 0)

    l2 = sig.get("l2_pattern", {})
    us_vec = l2.get("today_us_vector", {})
    sox_ret = us_vec.get("us_soxx_chg", 0)
    vix_chg = us_vec.get("us_vix_chg", 0)

    # STOXX50 (유럽 전일 종가)
    stoxx50_ret = _fetch_stoxx50_ret()

    prob = calc_market_probability(
        ewy_ret, vix, spy_ret, qqq_ret,
        soxx_ret=sox_ret, vix_chg=vix_chg,
        stoxx50_ret=stoxx50_ret,
    )

    relay = load_relay_signals()
    etf = load_etf_signals()
    scan = load_scan_results()

    L = []

    # ── Header ──
    L.append(f"\U0001f4ca 장전 마켓 브리핑 | {now}")
    L.append(f"Quantum Master v10.3")
    L.append("━" * 28)

    # ══════════════════════════════
    # 1. KOSPI 예측
    # ══════════════════════════════
    if prob["down_prob"] >= 60:
        dir_icon = "\U0001f534"
        dir_label = "하락 우세"
    elif prob["up_prob"] >= 60:
        dir_icon = "\U0001f7e2"
        dir_label = "상승 우세"
    elif prob["down_prob"] >= 55:
        dir_icon = "\U0001f7e0"
        dir_label = "하락 소폭"
    elif prob["up_prob"] >= 55:
        dir_icon = "\U0001f7e2"
        dir_label = "상승 소폭"
    else:
        dir_icon = "⚪"
        dir_label = "보합권"

    bar = make_prob_bar(prob["up_prob"])
    L.append("")
    L.append(f"{dir_icon} KOSPI 예측 — {dir_label}")
    L.append(f"  상승 {prob['up_prob']}% │ 하락 {prob['down_prob']}%")
    L.append(f"  {bar}")
    L.append(f"  예상 레인지: {prob['est_low']:+.1f}% ~ {prob['est_high']:+.1f}%")

    if prob["down_prob"] >= 60:
        L.append("  ➜ 갭다운 예상 — 우량주 저가매수 기회")
    elif prob["up_prob"] >= 60:
        L.append("  ➜ 갭업 예상 — 목표가 부근 분할매도")
    else:
        L.append(f"  ➜ 혼조 예상 — 시가 확인 후 대응")

    # ══════════════════════════════
    # 2. 판단 근거 (모델 기여도 순)
    # ══════════════════════════════
    L.append("")
    L.append("\U0001f4cc 판단 근거")

    # SPY (모델 기여도 1위: coef +0.855)
    us_tag = "강세" if spy_ret > 0.5 else ("약세" if spy_ret < -0.5 else "보합")
    L.append(f"  • SPY {spy_ret:+.1f}% | QQQ {qqq_ret:+.1f}% | DIA {dia_ret:+.1f}% — {us_tag}")

    # SOXX (모델 기여도 2위: coef +0.656)
    sox_tag = "반도체 급락" if sox_ret < -2 else (
        "반도체 약세" if sox_ret < -0.5 else (
            "반도체 강세" if sox_ret > 0.5 else "반도체 중립"))
    L.append(f"  • SOXX {sox_ret:+.1f}% — {sox_tag}")

    # VIX (모델 기여도 3위: coef +0.124)
    vix_icon = "⚠️" if vix >= 20 else "✅"
    L.append(f"  • VIX {vix:.1f} {vix_icon} {vix_status} (변화 {vix_chg:+.1f}%)")

    # STOXX50 (모델 기여도 4위: 유럽)
    if stoxx50_ret != 0:
        stoxx_tag = "유럽 강세" if stoxx50_ret > 0.5 else (
            "유럽 약세" if stoxx50_ret < -0.5 else "유럽 보합")
        L.append(f"  • STOXX50 {stoxx50_ret:+.1f}% — {stoxx_tag}")

    # EWY (참고: 모델에서 제거, 한국 ADR 참고용)
    ewy_tag = "강한 상방" if ewy_ret > 2 else (
        "강한 하방" if ewy_ret < -2 else (
            "약세" if ewy_ret < 0 else "양호"))
    L.append(f"  • EWY {ewy_ret:+.2f}% (5d {ewy_5d:+.1f}%) — {ewy_tag}")

    # US Signal
    L.append(f"  • US Signal: {us_grade} ({combined:+.1f})")

    # 섹터 Kill (있을 때만)
    kills = sig.get("sector_kills", {})
    killed_sectors = [s for s, v in kills.items() if v.get("killed")]
    if killed_sectors:
        L.append("")
        L.append("\U0001f6a8 섹터 Kill")
        for s in killed_sectors:
            info = kills[s]
            L.append(f"  ❌ {s}: {info['driver_ret']:+.1f}% (임계 {info['threshold_pct']:+.1f}%)")

    # ══════════════════════════════
    # 3. 패턴매칭
    # ══════════════════════════════
    l2_kospi = l2.get("kospi", {})
    if l2_kospi:
        pos_rate = l2_kospi.get("positive_rate", 50)
        mean_chg = l2_kospi.get("mean_chg", 0)
        sample = l2.get("sample_count", 0)
        if sample > 30:
            L.append("")
            L.append(f"\U0001f50d 패턴매칭 (유사 {sample}건)")
            L.append(f"  KOSPI 상승확률 {pos_rate:.0f}% (평균 {mean_chg:+.2f}%)")
            sec_l2 = l2.get("sectors", {})
            if sec_l2:
                sorted_sec = sorted(
                    sec_l2.items(),
                    key=lambda x: x[1].get("mean_chg", 0),
                    reverse=True,
                )
                top3 = sorted_sec[:3]
                bot1 = sorted_sec[-1]
                names = ", ".join(
                    f"{s[0]}({s[1]['positive_rate']:.0f}%)"
                    for s in top3
                )
                L.append(f"  ▲ 강세: {names}")
                L.append(
                    f"  ▼ 약세: {bot1[0]}"
                    f"({bot1[1]['positive_rate']:.0f}%)"
                )

    # ══════════════════════════════
    # 4. 매수 후보 — 4섹션
    # ══════════════════════════════
    L.append("")
    L.append("━" * 28)
    L.append("\U0001f4b0 매수 후보")
    L.append("━" * 28)

    # ── 4-1. 릴레이 시그널 ──
    L.append("")
    L.append(f"\U0001f504 릴레이 시그널 ({len(relay)}건 진입적기)")
    if relay:
        for i, r in enumerate(relay[:5], 1):
            lead = r.get("lead", "?")
            follow = r.get("follow", "?")
            wr = r.get("win_rate", 0)
            lag = r.get("best_lag", 0)
            picks = r.get("picks", [])
            pick_names = ", ".join(p.get("name", "?") for p in picks[:2])
            extra = f" 외{len(picks)-2}" if len(picks) > 2 else ""
            L.append(
                f"  {i}. {lead}→{follow} | "
                f"승률{wr:.0f}% lag{lag} | "
                f"{pick_names}{extra}"
            )
    else:
        L.append("  해당 없음")

    # ── 4-2. ETF 매매 시그널 ──
    etf_buy = etf.get("buy", [])
    etf_watch = etf.get("watch", [])
    L.append("")
    L.append(f"\U0001f4e6 ETF 매매 시그널")
    if etf_buy:
        for e in etf_buy[:3]:
            sector = e.get("sector", "?")
            code = e.get("etf_code", "")
            bb = e.get("bb_pct", 0)
            rsi = e.get("rsi", 0)
            sizing = e.get("sizing", "FULL")
            L.append(f"  • [BUY] {sector}({code}) BB{bb:.0f}% RSI{rsi:.0f} {sizing}")
    if etf_watch:
        for e in etf_watch[:3]:
            sector = e.get("sector", "?")
            code = e.get("etf_code", "")
            bb = e.get("bb_pct", 0)
            rsi = e.get("rsi", 0)
            sizing = e.get("sizing", "HALF")
            L.append(f"  • [관망] {sector}({code}) BB{bb:.0f}% RSI{rsi:.0f} {sizing}")
    if not etf_buy and not etf_watch:
        L.append("  해당 없음")

    # ── 4-3. 테마머니 ──
    theme = scan.get("theme_money", [])
    L.append("")
    L.append(f"\U0001f525 테마머니 (모멘텀)")
    if theme:
        for t in theme[:3]:
            g = _grade_stock(t, "THEME")
            name = t.get("name", t.get("ticker", "?"))
            ticker = str(t.get("ticker", "")).zfill(6)
            sector = t.get("etf_sector", t.get("krx_sector", ""))
            bb = t.get("bb_pct", 0)
            rsi = t.get("rsi", 0)
            adx = t.get("adx", 0)
            L.append(f"  • [{g}] {name}({ticker}) {sector}")
            L.append(f"       BB{bb:.0f}% RSI{rsi:.0f} ADX{adx:.0f}")
    else:
        L.append("  해당 없음")

    # ── 4-4. 스마트머니 ──
    smart = scan.get("smart_money", [])
    L.append("")
    L.append(f"\U0001f48e 스마트머니 (외인+기관)")
    if smart:
        for s in smart[:3]:
            g = _grade_stock(s, "SMART")
            name = s.get("name", s.get("ticker", "?"))
            ticker = str(s.get("ticker", "")).zfill(6)
            sector = s.get("etf_sector", s.get("krx_sector", ""))
            bb = s.get("bb_pct", 0)
            rsi = s.get("rsi", 0)
            stop = s.get("stop_pct", -7)
            L.append(f"  • [{g}] {name}({ticker}) {sector}")
            L.append(f"       BB{bb:.0f}% RSI{rsi:.0f} 손절{stop}%")
    else:
        L.append("  해당 없음")

    # ── Footer ──
    L.append("")
    L.append("⚠️ 투자 판단은 본인 책임 | QM v10.3")

    return "\n".join(L)


# ─────────────────────────────────────────────
# 통합 아침 브리핑 (1건으로 압축)
# ─────────────────────────────────────────────

MORNING_REPORTS_PATH = PROJECT_ROOT / "data" / "morning_reports.json"
THEME_ALERTS_PATH = PROJECT_ROOT / "data" / "theme_alerts.json"


def build_unified_morning() -> str:
    """아침 통합 브리핑 — KOSPI예측+US+증권사+테마+ETF → 1건.

    기존 build_briefing_message()의 핵심만 추출하고,
    증권사 리포트, 테마 알림, ETF 시그널을 한 메시지에 통합.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === 데이터 로드 ===
    sig = load_overnight_signal()
    idx = sig.get("index_direction", {})

    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    dia_ret = idx.get("DIA", {}).get("ret_1d", 0)
    vix_data = sig.get("vix", {})
    vix = vix_data.get("level", 20)
    vix_status = vix_data.get("status", "정상")
    us_grade = sig.get("grade", "NEUTRAL")

    us_vec = sig.get("l2_pattern", {}).get("today_us_vector", {})
    sox_ret = us_vec.get("us_soxx_chg", 0)
    vix_chg = us_vec.get("us_vix_chg", 0)
    stoxx50_ret = _fetch_stoxx50_ret()

    prob = calc_market_probability(
        ewy_ret, vix, spy_ret, qqq_ret,
        soxx_ret=sox_ret, vix_chg=vix_chg,
        stoxx50_ret=stoxx50_ret,
    )

    L = []

    # ── Header ──
    L.append(f"\U0001f4ca 장전 브리핑 | {now}")
    L.append("━" * 24)

    # ── 비거래일 데이터 경고 ──
    if prob.get("stale_data"):
        L.append("\n⚠️ US 데이터 없음 (주말/공휴일)")
        L.append("  KOSPI 예측 불가 — 보합 가정")

    # ── 1. KOSPI 예측 ──
    if prob.get("stale_data"):
        dir_icon, dir_label = "⚪", "보합 (데이터 없음)"
    elif prob["down_prob"] >= 60:
        dir_icon, dir_label = "\U0001f534", "하락 우세"
    elif prob["up_prob"] >= 60:
        dir_icon, dir_label = "\U0001f7e2", "상승 우세"
    elif prob["down_prob"] >= 55:
        dir_icon, dir_label = "\U0001f7e0", "하락 소폭"
    elif prob["up_prob"] >= 55:
        dir_icon, dir_label = "\U0001f7e2", "상승 소폭"
    else:
        dir_icon, dir_label = "⚪", "보합권"

    bar = make_prob_bar(prob["up_prob"], width=15)
    L.append(f"\n{dir_icon} KOSPI {dir_label}")
    L.append(f"  상승 {prob['up_prob']}% {bar} 하락 {prob['down_prob']}%")
    L.append(f"  레인지: {prob['est_low']:+.1f}% ~ {prob['est_high']:+.1f}%")

    # ── 2. US 야간 (1줄 요약) ──
    L.append(f"\n\U0001f30d US: SPY{spy_ret:+.1f}% QQQ{qqq_ret:+.1f}% SOXX{sox_ret:+.1f}%")
    vix_icon = "⚠️" if vix >= 20 else "✅"
    L.append(f"  VIX {vix:.1f}{vix_icon} EWY{ewy_ret:+.1f}% Signal:{us_grade}")

    # 섹터 Kill
    kills = sig.get("sector_kills", {})
    killed = [s for s, v in kills.items() if v.get("killed")]
    if killed:
        L.append(f"  \U0001f6a8 Kill: {', '.join(killed)}")

    # ── 3. 증권사 리포트 ──
    morning = _load_json(MORNING_REPORTS_PATH)
    reports = morning.get("reports", [])
    positive = [r for r in reports if r.get("opinion_type") == "매수"]
    if positive:
        L.append(f"\n\U0001f4dd 증권사 매수 ({len(positive)}건)")
        for r in positive[:4]:
            corp = r.get("company", "?")
            broker = r.get("broker", "")
            target = r.get("target_price", "")
            target_str = f" 목표{target}" if target else ""
            L.append(f"  • {corp}{target_str} ({broker})")

    # ── 4. 테마 동향 ──
    theme_data = _load_json(THEME_ALERTS_PATH)
    if isinstance(theme_data, list) and theme_data:
        themes = [t.get("theme_name", "") for t in theme_data if t.get("theme_name")][:4]
        if themes:
            L.append(f"\n\U0001f525 테마: {' | '.join(themes)}")

    # ── 5. ETF 시그널 (BAT-C 흡수) ──
    etf = load_etf_signals()
    etf_buy = etf.get("buy", [])
    if etf_buy:
        parts = []
        for e in etf_buy[:3]:
            sector = e.get("sector", "?")
            sizing = e.get("sizing", "FULL")
            parts.append(f"{sector}({sizing})")
        L.append(f"\n\U0001f4e6 ETF: {' | '.join(parts)}")

    # ── 6. 네 마녀의 날 경고 ──
    try:
        from src.use_cases.market_calendar import check_witching_proximity
        witching = check_witching_proximity()
        if witching["warning_level"] in ("CRITICAL", "HIGH"):
            L.append(f"\n{witching['message']}")
        elif witching["warning_level"] == "MODERATE":
            L.append(f"\n{witching['message']}")
    except Exception:
        pass  # 캘린더 모듈 실패 시 무시

    # ── 7. 전략 요약 ──
    L.append("")
    if prob["down_prob"] >= 60:
        L.append("➜ 갭다운 예상 — 우량주 저가매수 기회")
    elif prob["up_prob"] >= 60:
        L.append("➜ 갭업 예상 — 목표가 부근 분할매도")
    else:
        L.append("➜ 혼조 예상 — 시가 확인 후 대응")

    L.append("\n⚠️ 투자 판단은 본인 책임 | QM v10.3")

    return "\n".join(L)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    msg = build_briefing_message()

    print("=" * 50)
    print("[미리보기]")
    print("=" * 50)
    print(msg)
    print("=" * 50)
    print(f"총 {len(msg)}자 (텔레그램 제한: 4096자)")
    print()

    if "--send" in sys.argv:
        from src.telegram_sender import send_message
        ok = send_message(msg)
        if ok:
            print("텔레그램 발송 완료!")
        else:
            print("텔레그램 발송 실패")
    else:
        print("실제 발송: python -m src.use_cases.morning_briefing --send")
