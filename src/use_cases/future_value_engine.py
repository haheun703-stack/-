"""미래가치 통합 엔진 (Future Value Engine) v0 — 관측 전용(shadow).

설계: docs/02-design/future-value-engine_2026-07-04.md
근거(7/4 백테스트, valuation_gap 46스냅샷 5/1~7/3 cross-sectional D+20):
  - 저PER 분위 +3.98%p·승률 67%·중앙값 +5.44%p (유일하게 셋 다 양호)
  - 실적YoY 고분위 -2.77%p·승률 28% (모멘텀 피크 추격 = 독)
  - 같은 고실적도 사이클 초입/후기에 따라 약↔독 → 사이클 게이트 필수

원칙:
  - 계수 튜닝 금지(과최적화 방지) — 분위(상/중/하)와 온/오프 게이트, 고정 가점만.
  - 모든 입력은 graceful: 파일/DB 없으면 해당 축 0점 처리, 엔진은 죽지 않는다.
  - 매매 미배선. 산출물은 data/shadow/future_value.json 관측 전용.

6축 스코어카드 (유니버스 = consensus_screening.all_picks: forward 데이터 보유 종목):
  V 밸류에이션 갭(장기) : 컨센서스 목표가 괴리(upside_pct) + forward_per 분위
                        + ★역사 PER 밴드(자기 5년 대비 위치, v1-2번). 근거=7/5
                        백테스트(안정흑자주 13k관측·진짜 2021~2025·자기시장 벤치마크·
                        드롭0): 밴드 저평가(Q1/Q2 +0.3~0.5%p) vs 고평가(Q5 -0.58%p)
                        스프레드 +0.87%p·횡단면PER과 상관 +0.23(독립). 밸류트랩
                        (자기역사상 비쌈=이익둔화) 회피가 핵심.
  E 실적 가속(중기)    : leader_cycle delta_value(TTM-YoY 델타) 재사용
  L 사이클 위치(중기)   : leader_cycle signal — 후기(경계/청산)는 중기 차단
  O 수주 정보태그(무가점): contract_history.jsonl 매출대비 50%+ 계약 → 태그만.
                        ★7/5 이벤트 스터디 v2(576건·3지표): raw D+1 +5.34%는
                        전부 체결불가 오버나이트 갭 — 실행가능(D+1 시가 진입)은
                        D+1 -2.98%·승률 19%로 오히려 손실 → 가점 제거,
                        '수주팝직후(추격주의)' 정보 태그만 유지. 이력은
                        v2 '수주잔고 누적/시총' 가설 검증용으로 계속 축적.
  S 스마트머니(단기)    : investor_daily.db 금투+연기금 5D 순매수 (5/14 우선순위)
  T 테마 촉매(단기)     : theme_alerts_today.json 뉴스 발화

국면 스위치(층3): brain 레짐 — BEAR/PRE_BEAR=장기(가치)만 권장,
  CAUTION=가치 가중, PRE_BULL/BULL=모멘텀 축 가중 (7/4 레짐 검증: 방어 적중 81%).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CONSENSUS_PATH = DATA_DIR / "consensus_screening.json"
VALGAP_GLOB = "valuation_gap_*.json"
LEADER_PATH = DATA_DIR / "shadow" / "leader_cycle.json"
CONTRACT_PATH = DATA_DIR / "contract_history.jsonl"
THEME_PATH = DATA_DIR / "theme_alerts_today.json"
BRAIN_PATH = DATA_DIR / "brain_history.json"
INVESTOR_DB = DATA_DIR / "investor_flow" / "investor_daily.db"
OUTPUT_PATH = DATA_DIR / "shadow" / "future_value.json"

# 사이클 후기 신호 = 중기 차단 (7/4: 고실적×후기 = 최악 -21%p의 정체)
CYCLE_LATE_SIGNALS = {"경계", "청산"}
CYCLE_EARLY_SIGNALS = {"매수적기"}


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_valgap() -> dict[str, dict]:
    """최신 valuation_gap 스냅샷 → {ticker: item}."""
    try:
        files = sorted(DATA_DIR.glob(VALGAP_GLOB))
        if not files:
            return {}
        items = _load_json(files[-1]) or []
        return {it["ticker"]: it for it in items if it.get("ticker")}
    except Exception:
        return {}


def _leader_map() -> dict[str, dict]:
    d = _load_json(LEADER_PATH) or {}
    return {l["ticker"]: l for l in d.get("leaders", []) if l.get("market") == "KR"}


def _big_contract_tickers(days: int = 3, min_ratio: float = 50.0) -> set[str]:
    """최근 N일(달력, 주말 커버) 내 매출대비 min_ratio%+ 공급계약 종목 — 정보 태그용.

    ★스코어 가점 없음: 이벤트 스터디 v2(7/5, 576건) 실행가능 지표(D+1 시가 진입)가
    D+1 -2.98%·승률 19% — 팝은 체결불가 갭이고 추격은 손실. 태그는 '이 종목이
    수주 팝 직후 구간(추격주의)'이라는 경고 정보로만 쓴다.
    """
    out: set[str] = set()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        with open(CONTRACT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if (str(r.get("date", "")) >= cutoff
                        and (r.get("revenue_ratio_pct") or 0) >= min_ratio):
                    out.add(r.get("ticker"))
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("[FV] 계약이력 로드 실패(축 0점 처리): %s", e)
    return out


def _theme_tickers() -> set[str]:
    d = _load_json(THEME_PATH) or {}
    return set(d.get("ticker_map", {}).keys())


def _current_regime() -> str:
    hist = _load_json(BRAIN_PATH) or []
    if isinstance(hist, list) and hist:
        return hist[-1].get("regime", "CAUTION")
    return "CAUTION"


def _market_map() -> dict[str, str]:
    """ticker → KOSPI/KOSDAQ (v1-3번). 부재 시 빈 dict(전부 KOSPI 취급)."""
    return _load_json(DATA_DIR / "market_map.json") or {}


def _regime_to_horizon(regime: str) -> str:
    """레짐(BRAIN PRE_BULL/… 또는 지수 BULL/CRISIS) → 권장 타임프레임.

    7/4 검증: 레짐=방어 유능. 하락 국면일수록 단기 공세 비권장·장기 가치 위주.
    """
    if regime in ("CRISIS",):
        return "장기(가치·방어)"
    if regime in ("BEAR", "PRE_BEAR"):
        return "장기(가치적립)"
    if regime in ("BULL", "PRE_BULL"):
        return "단기+중기(공세)"
    return "중기+장기(선별)"  # CAUTION 등


def _smart_money_5d(tickers: list[str]) -> dict[str, dict]:
    """금투/연기금 최근 5거래일 순매수(원). 퀀트봇 담당 수급 = 스마트머니 1단계."""
    out: dict[str, dict] = {}
    try:
        con = sqlite3.connect(str(INVESTOR_DB))
        cur = con.cursor()
        cur.execute("SELECT DISTINCT date FROM investor_daily ORDER BY date DESC LIMIT 5")
        days = [r[0] for r in cur.fetchall()]
        if not days:
            return out
        ph_days = ",".join("?" * len(days))
        for tk in tickers:
            cur.execute(
                f"SELECT investor, SUM(net_val) FROM investor_daily "
                f"WHERE ticker=? AND date IN ({ph_days}) "
                f"AND investor IN ('금융투자','연기금') GROUP BY investor",
                [tk, *days],
            )
            row = {inv: (val or 0) for inv, val in cur.fetchall()}
            out[tk] = {"finance_5d": row.get("금융투자", 0), "pension_5d": row.get("연기금", 0)}
        con.close()
    except Exception as e:  # noqa: BLE001 — 축 결측 graceful
        logger.warning("[FV] 수급 조인 실패(축 0점 처리): %s", e)
    return out


def _per_bands(tickers: list[str]) -> dict[str, dict]:
    """종목별 역사 PER 밴드 (v1-2번). 모듈/데이터 부재 시 빈 dict(graceful)."""
    out: dict[str, dict] = {}
    try:
        from src.use_cases.valuation_band_history import compute_per_band
    except Exception as e:  # noqa: BLE001
        logger.warning("[FV] PER 밴드 모듈 로드 실패(V축 밴드 생략): %s", e)
        return out
    for tk in tickers:
        try:
            b = compute_per_band(tk)
            if b:
                out[tk] = b
        except Exception:  # noqa: BLE001 — 개별 종목 실패는 스킵
            continue
    return out


def _tercile_bounds(values: list[float]) -> tuple[float, float]:
    """(하위 경계, 상위 경계) — 33/67 분위."""
    s = sorted(values)
    if not s:
        return 0.0, 0.0
    return s[int(len(s) * 0.33)], s[min(int(len(s) * 0.67), len(s) - 1)]


def _is_preferred(name: str, ticker: str) -> bool:
    """KRX 우선주 판별(이름 끝 '우'[+등급자]·티커 비-0). 성우·에코글로우 등 보존."""
    return bool(re.search(r"우[A-Z]?$", name or "")) and not ticker.endswith("0")


def build_scorecards() -> dict:
    """6축 결합 → 종목별 단/중/장기 FV 스코어. 반환: 산출물 dict(JSON 직렬화 가능)."""
    cons = _load_json(CONSENSUS_PATH) or {}
    picks = cons.get("all_picks", [])
    if not picks:
        return {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "error": "consensus_screening 없음", "scorecards": []}
    # 우선주 방어 제거(v1-4번): 우선주가 보통주 목표가/forward_eps를 복사받아 가짜
    #   상승여력 발생(현대차우 +283%). scan_consensus 생산자 픽스의 이중방어이자
    #   consensus_screening 재생성(BAT-D) 전까지 shadow 즉시 정확성 보장.
    n_before = len(picks)
    picks = [p for p in picks if not _is_preferred(p.get("name", ""), p.get("ticker", ""))]
    if n_before != len(picks):
        logger.info("[FV] 우선주 %d종 제외(보통주 목표가 복사 왜곡 방지)", n_before - len(picks))

    valgap = _latest_valgap()
    leaders = _leader_map()
    contracts = _big_contract_tickers()  # 50%+/최근 3일 — 정보 태그 전용(가점 0, 스터디 v2)
    themes = _theme_tickers()
    regime = _current_regime()  # BRAIN(KOSPI) 헤드라인 레짐
    smart = _smart_money_5d([p["ticker"] for p in picks])
    bands = _per_bands([p["ticker"] for p in picks])  # 역사 PER 밴드 (v1-2번)
    mkt_map = _market_map()  # v1-3번: 종목별 시장(KOSPI/KOSDAQ)
    try:
        from src.use_cases.index_regime import kosdaq_regime
        kosdaq_reg = kosdaq_regime().get("regime", "CAUTION")
    except Exception:  # noqa: BLE001
        kosdaq_reg = "CAUTION"

    # 분위 경계 (유니버스 내 상대 — 계수 튜닝 없음)
    fwd_pers = [p["forward_per"] for p in picks if p.get("forward_per") and p["forward_per"] > 0]
    per_lo, per_hi = _tercile_bounds(fwd_pers)
    oiys = [valgap[p["ticker"]].get("oi_yoy") for p in picks
            if p["ticker"] in valgap and valgap[p["ticker"]].get("oi_yoy") is not None]
    oiy_lo, oiy_hi = _tercile_bounds(oiys)

    cards = []
    for p in picks:
        tk = p["ticker"]
        vg = valgap.get(tk, {})
        ld = leaders.get(tk, {})
        sm = smart.get(tk, {})
        tags: list[str] = []

        upside = p.get("upside_pct") or 0
        fwd_per = p.get("forward_per") or 0
        oi_yoy = vg.get("oi_yoy")
        delta = ld.get("delta_value")
        cyc_signal = ld.get("signal", "")
        has_contract = tk in contracts
        has_theme = tk in themes
        fin5 = sm.get("finance_5d", 0)
        pen5 = sm.get("pension_5d", 0)
        band = bands.get(tk)

        # ── V 장기: 컨센서스 괴리 + 저PER 분위 (백테스트: 저PER 승률 67%) ──
        fv_long = 50.0
        if upside >= 30:
            fv_long += 15
            tags.append(f"목표가괴리+{upside:.0f}%")
        elif upside >= 15:
            fv_long += 8
        if fwd_per and fwd_per > 0:
            if fwd_per <= per_lo:
                fv_long += 15
                tags.append(f"저PER({fwd_per:.1f})")
            elif fwd_per >= per_hi:
                fv_long -= 10
        if (p.get("dividend_yield") or 0) >= 4:
            fv_long += 5
            tags.append("배당4%+")
        # 역사 PER 밴드 (안정흑자주만·reliable) — 7/5 백테스트: 밴드 상단=밸류트랩 회피
        if band and band.get("reliable"):
            pr = band["pct_rank"]
            if pr <= 0.25:
                fv_long += 8
                tags.append(f"역사적저평가(PER밴드 {pr*100:.0f}%ile)")
            elif pr >= 0.75:
                fv_long -= 10
                tags.append(f"밸류트랩주의(PER밴드 {pr*100:.0f}%ile)")

        # ── E+L 중기: 실적 가속 × 사이클 게이트 ──
        fv_mid = 50.0
        cycle_late = cyc_signal in CYCLE_LATE_SIGNALS
        if cycle_late:
            fv_mid = 0.0  # 후기 차단 (7/4 근거: 고실적×후기 = 최악)
            tags.append(f"사이클후기차단({cyc_signal})")
        else:
            if delta is not None and delta > 0:
                fv_mid += 15
                tags.append(f"실적가속Δ+{delta:.0f}")
            if cyc_signal in CYCLE_EARLY_SIGNALS:
                fv_mid += 15
                tags.append("사이클초입")
            # 모멘텀 피크 감점: oi_yoy 상위분위인데 사이클 정보 없음/초입 아님
            if (oi_yoy is not None and oiys and oi_yoy >= oiy_hi
                    and cyc_signal not in CYCLE_EARLY_SIGNALS):
                fv_mid -= 15
                tags.append("모멘텀피크주의")

        # ── S+T 단기: 스마트머니 + 테마 ──
        fv_short = 50.0
        if fin5 > 0 and pen5 > 0:
            fv_short += 20
            tags.append("금투·연기금 동시매수")
        elif fin5 > 0 or pen5 > 0:
            fv_short += 10
        if has_theme:
            fv_short += 10
            tags.append("테마발화")
        if has_contract:
            # 가점 없음 — 스터디 v2(7/5): 팝=체결불가 갭, D+1 시가 추격은 -2.98%(승률 19%)
            tags.append("수주팝직후(추격주의)")

        # ── 층3 국면 스위치(시장인지 v1-3번): KOSPI 종목=BRAIN 레짐(검증),
        #    KOSDAQ 종목=KOSDAQ 지수 레짐(BRAIN은 KOSPI 전용이라 오배정 방지) ──
        market = mkt_map.get(tk, "KOSPI")
        mkt_regime = kosdaq_reg if market == "KOSDAQ" else regime
        horizon = _regime_to_horizon(mkt_regime)

        best = max(("short", fv_short), ("mid", fv_mid), ("long", fv_long), key=lambda x: x[1])
        cards.append({
            "ticker": tk, "name": p.get("name", tk), "close": p.get("close"),
            "target_price": p.get("target_price"), "fair_gap_pct": round(upside, 1),
            "forward_per": fwd_per or None,
            "fv_short": round(fv_short, 1), "fv_mid": round(fv_mid, 1),
            "fv_long": round(fv_long, 1), "fv_best": best[0],
            "cycle_signal": cyc_signal or None,
            "market": market, "market_regime": mkt_regime, "horizon": horizon,
            "finance_5d_억": round(fin5 / 1e8, 1), "pension_5d_억": round(pen5 / 1e8, 1),
            "per_band": ({"pct_rank": band["pct_rank"], "median": band["band_median"],
                          "current": band["current_per"], "signal": band["signal"],
                          "reliable": band["reliable"]} if band else None),
            "tags": tags,
        })

    cards.sort(key=lambda c: max(c["fv_short"], c["fv_mid"], c["fv_long"]), reverse=True)
    n_kosdaq = sum(1 for c in cards if c["market"] == "KOSDAQ")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regime": regime,                       # BRAIN(KOSPI) 헤드라인
        "kospi_regime": regime, "kosdaq_regime": kosdaq_reg,
        "recommended_horizon": _regime_to_horizon(regime),  # KOSPI 기준
        "universe_size": len(cards),
        "market_split": {"KOSPI": len(cards) - n_kosdaq, "KOSDAQ": n_kosdaq},
        "axes_coverage": {
            "valuation_gap": len(valgap), "leader_cycle": len(leaders),
            "supply_contract": len(contracts), "theme": len(themes),
            "smart_money": len(smart),
            "per_band": sum(1 for b in bands.values() if b.get("reliable")),
            "market_map": len(mkt_map),
        },
        "scorecards": cards,
    }
