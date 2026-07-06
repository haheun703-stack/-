"""미국판 미래가치 통합 엔진 (US Future Value Engine) v1 — 관측 전용(shadow).

한국 future_value_engine.py의 이식판. 미국엔 한국식 세분수급(금투/연기금)·DART 공시·
RSS 테마·수주 파서가 없으므로, **정직하게 V(밸류·장기) + E·L(실적·사이클·중기) 2축**으로 시작.
S(스마트머니)·T(테마)·O(수주) 단기축은 데이터(옵션흐름·13F·ETF flow)가 쌓이는 v2로 미룸.

축 구성 (전부 기존 자산 재사용 · 계수 튜닝 금지 · graceful):
  V 밸류(장기) : consensus_screening_us(컨센서스 목표가 괴리 + forward_per 분위)
               + 역사 PER 밴드(valuation_band_history_us, 자기 5년 대비 위치·basis가드).
  E 실적(중기) : leader_cycle US delta_value(연간 영업이익 YoY 델타) 재사용.
  L 사이클(중기): leader_cycle US signal — 후기(경계/청산)는 중기 차단, 초입(매수적기) 가점.
  국면 스위치  : index_regime S&P500(spy) 헤드라인 레짐 → 권장 타임프레임.
               나스닥(qqq) 레짐도 관측 병기(divergence 정보). ★v1은 S&P 헤드라인 일괄 —
               종목별 S&P/나스닥 소속 배선은 거래소 매핑 확보 후 v2.

★★ 산출물 data/shadow/future_value_us.json 관측 전용 · 매매 미배선 · 실주문 0.
   backtest_first: 가점 확정은 백테스트(scripts/research/) 예측력 검증 후. `./venv/bin/python3.11`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

CONSENSUS_US_PATH = DATA_DIR / "consensus_screening_us.json"
LEADER_PATH = DATA_DIR / "shadow" / "leader_cycle.json"
OUTPUT_PATH = DATA_DIR / "shadow" / "future_value_us.json"

# 한국 엔진과 동일 신호 어휘(leader_cycle 공용)
CYCLE_LATE_SIGNALS = {"경계", "청산"}
CYCLE_EARLY_SIGNALS = {"매수적기"}


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _leader_map() -> dict[str, dict]:
    """leader_cycle US 리더 → {ticker: {signal, delta_value}}."""
    d = _load_json(LEADER_PATH) or {}
    return {l["ticker"]: l for l in d.get("leaders", []) if l.get("market") == "US"}


def _per_bands(tickers: list[str]) -> dict[str, dict]:
    """종목별 US 역사 PER 밴드. 모듈/캐시 부재 시 빈 dict(graceful)."""
    out: dict[str, dict] = {}
    try:
        from src.use_cases.valuation_band_history_us import compute_per_band_us
    except Exception as e:  # noqa: BLE001
        logger.warning("[FV-US] PER 밴드 모듈 로드 실패(V축 밴드 생략): %s", e)
        return out
    for tk in tickers:
        try:
            b = compute_per_band_us(tk)
            if b:
                out[tk] = b
        except Exception:  # noqa: BLE001
            continue
    return out


def _tercile_bounds(values: list[float]) -> tuple[float, float]:
    """(하위 경계, 상위 경계) — 33/67 분위 (한국 엔진 동일)."""
    s = sorted(values)
    if not s:
        return 0.0, 0.0
    return s[int(len(s) * 0.33)], s[min(int(len(s) * 0.67), len(s) - 1)]


def _regime_to_horizon(regime: str) -> str:
    """지수 레짐(BULL/CAUTION/BEAR/CRISIS) → 권장 타임프레임 (한국 엔진 동일 규칙)."""
    if regime == "CRISIS":
        return "장기(가치·방어)"
    if regime in ("BEAR", "PRE_BEAR"):
        return "장기(가치적립)"
    if regime in ("BULL", "PRE_BULL"):
        return "단기+중기(공세)"
    return "중기+장기(선별)"


def build_scorecards() -> dict:
    """V + E·L 결합 → 종목별 중/장기 FV 스코어. 반환: JSON 직렬화 dict."""
    cons = _load_json(CONSENSUS_US_PATH) or {}
    picks = cons.get("all_picks", [])
    if not picks:
        return {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "market": "US", "error": "consensus_screening_us 없음", "scorecards": []}

    leaders = _leader_map()
    bands = _per_bands([p["ticker"] for p in picks])

    # 국면(S&P 헤드라인 + 나스닥 관측)
    try:
        from src.use_cases.index_regime import nasdaq_regime, sp500_regime
        sp_reg = sp500_regime().get("regime", "CAUTION")
        nq_reg = nasdaq_regime().get("regime", "CAUTION")
    except Exception:  # noqa: BLE001
        sp_reg = nq_reg = "CAUTION"

    # 분위 경계 (유니버스 내 상대 — 계수 튜닝 없음)
    fwd_pers = [p["forward_per"] for p in picks if p.get("forward_per") and p["forward_per"] > 0]
    per_lo, per_hi = _tercile_bounds(fwd_pers)
    deltas = [leaders[p["ticker"]].get("delta_value") for p in picks
              if p["ticker"] in leaders and leaders[p["ticker"]].get("delta_value") is not None]
    delta_lo, delta_hi = _tercile_bounds(deltas)

    cards = []
    for p in picks:
        tk = p["ticker"]
        ld = leaders.get(tk, {})
        tags: list[str] = []

        upside = p.get("upside_pct") or 0
        fwd_per = p.get("forward_per") or 0
        delta = ld.get("delta_value")
        cyc_signal = ld.get("signal", "")
        band = bands.get(tk)

        # ── V 장기: 컨센서스 괴리 + 저PER 분위 + 역사 PER 밴드 ──
        fv_long = 50.0
        if upside >= 30:
            fv_long += 15
            tags.append(f"목표가괴리+{upside:.0f}%")
        elif upside >= 15:
            fv_long += 8
        # ★④ 백테스트: 횡단면 저PER은 US 성장장서 역효과(고PER +1.1%p 우세)·유의성 없음.
        #   밴드와 동일 규율(④-검증 약/역행 팩터 무가점) 적용 → 태그만. US forward-PER 백테스트로
        #   부호 재확정 전까지 무가점(렌즈3 규율일관성 지적 반영).
        if fwd_per and fwd_per > 0:
            if fwd_per <= per_lo:
                tags.append(f"저PER({fwd_per:.1f})·가점보류:④역행")
            elif fwd_per >= per_hi:
                tags.append(f"고PER({fwd_per:.1f})")
        if (p.get("dividend_yield") or 0) >= 4:
            fv_long += 5
            tags.append("배당4%+")
        if band and band.get("reliable"):
            pr = band["pct_rank"]
            # ★④ 백테스트: 월간롱숏 t=0.61(유의성 없음)·승률~48%. 게다가 프로덕션정합(reliable
            #   coverage≥3) 서브셋(16%)선 스프레드 -0.61%p로 역전(Q1 저평가 -1.70%p) → 어느 방향도
            #   예측력 없음. 저평가 가점·고평가 감점 전부 무가점, 관측 태그만. 현 창=2023-26 단일
            #   성장국면 → 다국면 백테스트 확보 후 재평가(7/5 O축 선례=백테스트 실패시 무가점).
            if pr <= 0.25:
                tags.append(f"역사적저평가(PER밴드 {pr*100:.0f}%ile·관측)")
            elif pr >= 0.75:
                tags.append(f"밸류트랩주의(PER밴드 {pr*100:.0f}%ile·관측)")

        # ── E+L 중기: 실적 가속 × 사이클 게이트 ──
        fv_mid = 50.0
        cycle_late = cyc_signal in CYCLE_LATE_SIGNALS
        if cycle_late:
            fv_mid = 0.0  # 후기 차단 (한국 근거: 고실적×후기 = 최악)
            tags.append(f"사이클후기차단({cyc_signal})")
        else:
            if delta is not None and delta > 0:
                fv_mid += 15
                tags.append(f"실적가속Δ+{delta:.0f}")
            if cyc_signal in CYCLE_EARLY_SIGNALS:
                fv_mid += 15
                tags.append("사이클초입")
            # 모멘텀 피크 감점(추격 위험): 델타 양수 & 상위분위 & 초입 아님. ★소표본(consensus∩
            #   US리더 겹침 적음)에선 delta_hi=최댓값이라 실적가속 +15를 상시 오상쇄 → len>=5 가드.
            #   둔화주(delta<=0)는 '피크' 아니므로 delta>0 조건으로 오태깅 차단(렌즈1 #2).
            if (delta is not None and delta > 0 and len(deltas) >= 5 and delta >= delta_hi
                    and cyc_signal not in CYCLE_EARLY_SIGNALS):
                fv_mid -= 15
                tags.append("모멘텀피크주의")

        # ── 국면 스위치: S&P 헤드라인 레짐 → 타임프레임 (v1 일괄) ──
        horizon = _regime_to_horizon(sp_reg)

        best = max(("mid", fv_mid), ("long", fv_long), key=lambda x: x[1])
        cards.append({
            "ticker": tk, "name": p.get("name", tk), "close": p.get("close"),
            "target_price": p.get("target_price"), "fair_gap_pct": round(upside, 1),
            "forward_per": fwd_per or None,
            "fv_mid": round(fv_mid, 1), "fv_long": round(fv_long, 1), "fv_best": best[0],
            "cycle_signal": cyc_signal or None, "delta_value": delta,
            "market_regime": sp_reg, "horizon": horizon,
            "per_band": ({"pct_rank": band["pct_rank"], "median": band["band_median"],
                          "current": band["current_per"], "signal": band["signal"],
                          "reliable": band["reliable"]} if band else None),
            "tags": tags,
        })

    cards.sort(key=lambda c: max(c["fv_mid"], c["fv_long"]), reverse=True)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market": "US",
        "validation": "shadow_unvalidated",  # ★관측 전용. 가중치=한국 이식(US 미검증).
        "validation_note": (
            "④ PER밴드 백테스트(44종·1198관측·32개월·2023-09~2026-06): 헤드라인 Q1-Q5 +0.83%p이나 "
            "월간 롱숏 t=0.61(유의성 없음). ★프로덕션정합(reliable coverage≥3) 서브셋(16%)선 스프레드 "
            "-0.61%p로 역전 → 어느 방향도 예측력 없음. 규율(④-검증 실패/역행 팩터 무가점): 밴드·forward_per "
            "전부 무가점(관측 태그만). fv_long=컨센서스 목표가 괴리+배당(백테스트 불가/미검증 hypothesis). "
            "E·L=leader_cycle 사이클게이트. 20거래일 forward shadow로 검증, 승격은 게이트 통과 후."),
        "regime": sp_reg,                       # S&P 헤드라인
        "sp500_regime": sp_reg, "nasdaq_regime": nq_reg,
        "recommended_horizon": _regime_to_horizon(sp_reg),
        "universe_size": len(cards),
        "axes_coverage": {
            "consensus": len(picks), "leader_cycle": len(leaders),
            "per_band": sum(1 for b in bands.values() if b.get("reliable")),
        },
        "scorecards": cards,
    }


def run() -> dict:
    result = build_scorecards()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("[FV-US] 저장: %s (%d종목·레짐 S&P=%s/NQ=%s)",
                OUTPUT_PATH, result.get("universe_size", 0),
                result.get("sp500_regime"), result.get("nasdaq_regime"))
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    r = run()
    cards = r.get("scorecards", [])
    print(f"\n=== US 미래가치 스코어카드 (레짐 S&P={r.get('sp500_regime')} / "
          f"NQ={r.get('nasdaq_regime')} · {r.get('recommended_horizon')}) ===")
    print(f"{'#':>2} {'종목':>6} {'종가':>9} {'괴리':>7} {'F-PER':>6} "
          f"{'FV장기':>6} {'FV중기':>6} {'best':>4} {'사이클':>8}  태그")
    for i, c in enumerate(cards[:25], 1):
        fp = f"{c['forward_per']:.1f}" if c.get("forward_per") else "-"
        print(f"{i:>2} {c['ticker']:>6} {c['close']:>9.2f} {c['fair_gap_pct']:>+6.1f}% "
              f"{fp:>6} {c['fv_long']:>6.1f} {c['fv_mid']:>6.1f} {c['fv_best']:>4} "
              f"{c.get('cycle_signal') or '-':>8}  {', '.join(c['tags'][:4])}")


if __name__ == "__main__":
    main()
