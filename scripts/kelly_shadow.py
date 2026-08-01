"""켈리 섀도 관측 — 매매 경로를 건드리지 않고 "켈리라면 얼마로 줄였을까"만 기록한다.

★왜 섀도인가(8\1 사장님 지시 "켜서 돌려보면서 진행"):
멈춰서 기다릴 이유는 없다. 다만 지금 그냥 켜면 두 가지를 잃는다 —
⑴켈리가 수량을 깎기 시작하면 이후 페이퍼 성적이 "종목 탓"과 "켈리 탓"으로 섞이고
⑵승률 재료가 틀린 상태(B-35)라 그 구간 기록이 나중에 해석 불가가 된다.
섀도는 둘 다 피하면서 "코드가 도는가"와 "무슨 값이 나오는가"를 오늘부터 쌓는다.
이 저장소가 한미충격·주도주 사이클·국면신호에 써온 방식과 같다.

★코덱스 작업분(B-34)을 import하지 않고 산식을 자체 구현한다:
그 7파일은 커밋 보류 상태(의미론·집계 편향 미해소)라 코드 상태에 의존하면
섀도가 같이 멈춘다. 여기서는 **관측만** 하므로 독립이 맞다.

★★두 값을 나란히 낸다 — **하나만 켈리다.**
  ⑴ `mult_abs` = 절대 승률 기반 켈리(코덱스 코드가 실제로 쓸 값)
  ⑵ `selection_edge_proxy` = 시장 대비 초과분을 같은 식에 넣어본 **진단용 대리지표**
     ★켈리가 아니다(코덱스 8\1 판정): 롱온리 켈리의 p는 "현금 기준으로 이겼을 확률"인데
     초과 승률은 "시장보다 나았을 확률"이라 둘 사이에 변환식이 없다. 종목이 늘 -1%,
     시장이 늘 -2%면 초과 승률은 100%지만 계좌는 손실이라 켈리 비중은 0이다.
     **주문 수량·성과 감산의 근거로 쓰지 않는다.**

p·b는 **엄격한 승/패 표본**에서 온 값을 쓴다(B-35 ③) — flat(정확히 0)과
invalid(평가 불가)를 패배로 세면 승률도 손익비도 왜곡된다.
손익비가 없으면 1:1 같은 가정을 넣지 않고 **켈리 계산 자체를 건너뛴다**.

실행:
    python -u -X utf8 scripts/kelly_shadow.py              # 관측 1회 + 기록
    python -u -X utf8 scripts/kelly_shadow.py --no-save    # 화면 출력만
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACCURACY_FILE = ROOT / "data" / "market_learning" / "signal_accuracy.json"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_FILE = ROOT / "data" / "metrics" / "kelly_shadow.jsonl"

# 코덱스 기본값(config/settings.yaml alpha_v2.sizing)과 동일하게 맞춘다.
KELLY_FRACTION = 0.50
KELLY_FLOOR = 0.00
KELLY_CAP = 1.00
KELLY_FALLBACK_ODDS = 1.00
KELLY_DEFAULT = 0.50
KELLY_MIN_SAMPLES = 30


def _kelly(p: float, b: float) -> float:
    """이산(Bernoulli) 켈리. 코덱스 `_calc_half_kelly`와 같은 산식.

    ★이 값은 '자본 중 몇 %를 걸어라'인데 코덱스 코드는 기존 ATR 수량에 곱하는
    배수로 쓴다(Q2 의미론 쟁점, 미해소). 여기서는 그 코드가 낼 값을 그대로
    재현해 관측하는 것이 목적이므로 같은 방식으로 계산만 한다.
    """
    if b <= 0:
        return KELLY_DEFAULT
    k = (p * b - (1.0 - p)) / b
    if k <= 0:
        return KELLY_FLOOR
    return max(KELLY_FLOOR, min(KELLY_CAP, k * KELLY_FRACTION))


def _market_win_rate() -> tuple[float, int]:
    """폴백 전용 — 당일 유니버스 상승 비율.

    ⚠️**이건 D+1 기준이라 성과파일(D+3)과 잣대가 다르다.** 1차 구현에서 이걸
    기저선으로 썼다가 "D+3 승률을 D+1 기저선과 비교"하는 오류를 냈다 —
    7/31(KOSPI +17.91%)에 89.7%가 나와 모든 보정값이 0으로 눌렸다.
    지금은 성과파일이 같은 규칙(next_open → D+3_close)의 기저선을 직접 주므로
    그 값을 쓴다. 이 함수는 구버전 파일을 만났을 때의 참고값일 뿐이다.
    """
    import pandas as pd

    rets = []
    for pq in PROCESSED_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq, columns=["close"])
            if len(df) >= 2:
                prev = float(df["close"].iloc[-2])
                if prev > 0:
                    rets.append(float(df["close"].iloc[-1]) / prev - 1)
        except Exception:  # noqa: BLE001
            continue
    if not rets:
        return 0.0, 0
    up = sum(1 for r in rets if r > 0)
    return up / len(rets), len(rets)


def _p_from(hit_rate) -> float | None:
    """저장된 hit_rate → 확률. **백분율 계약으로 고정**(추측하지 않는다).

    코덱스 코드는 `hit_rate > 1`로 단위를 추측하는데, 생산자 2곳 모두 백분율로
    확정 저장한다(Q5). 경계에서 뒤집히므로(1.0을 100%로 읽음) 여기서는 계약을
    고정하고 범위를 벗어나면 판정하지 않는다.
    """
    try:
        v = float(hit_rate)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= v <= 100.0:
        return None
    return v / 100.0


def observe() -> dict:
    if not ACCURACY_FILE.exists():
        return {"error": f"{ACCURACY_FILE} 없음"}
    data = json.loads(ACCURACY_FILE.read_text(encoding="utf-8"))
    signals = data.get("signals") or {}

    # 기저선: 성과파일이 **같은 규칙**(next_open → D+3_close)으로 계산해 둔 것을 쓴다(B-35 ②).
    # 없으면(구버전 파일) 당일 기준 폴백 — 잣대가 달라 참고용임을 표기한다.
    base = data.get("baseline") or {}
    if base.get("hit_rate") is not None:
        mkt_p, mkt_n = base["hit_rate"] / 100.0, base.get("samples", 0)
        base_rule = base.get("rule", "")
    else:
        mkt_p, mkt_n = _market_win_rate()
        base_rule = "⚠️당일 D+1 폴백 — 성과파일(D+3)과 잣대 불일치"

    rows = []
    for eng, acc in sorted(signals.items(), key=lambda x: -x[1].get("total", 0)):
        total = int(acc.get("total", 0) or 0)
        p = _p_from(acc.get("hit_rate"))
        if p is None:
            rows.append({"engine": eng, "total": total, "verdict": "단위 계약 위반 — 미판정"})
            continue

        # p·b는 **엄격한 승/패 표본**에서 온 값을 우선한다(B-35 ③).
        # flat(정확히 0)과 invalid(평가 불가)를 패배로 세면 둘 다 왜곡된다.
        p_strict = _p_from(acc.get("hit_rate_strict"))
        if p_strict is not None:
            p = p_strict
        wl_n = acc.get("win_loss_n")
        n_for_sample = int(wl_n) if wl_n is not None else total

        b = acc.get("odds_b")
        if b is None:
            aw, al = acc.get("avg_win"), acc.get("avg_loss")
            if aw and al:
                b = abs(float(aw)) / abs(float(al))
        # ★손익비가 없으면 **가정하지 않는다**(코덱스 권고). 1:1을 값으로 박으면
        #   실측으로 오인된다 — 이 엔진은 켈리 계산 자체를 하지 않는다.
        has_odds = b is not None and float(b) > 0

        # 시장 대비 선택력 — ★켈리 확률이 아니다(코덱스 8/1 후속 판정).
        # 롱온리 켈리의 p는 "현금 기준으로 이겼을 확률"인데 초과 승률은
        # "시장보다 나았을 확률"이라 변환식이 없다. 종목이 늘 -1%, 시장이 늘 -2%면
        # 초과 승률은 100%지만 계좌는 손실이라 켈리 비중은 0이다.
        # 따라서 아래 값은 **진단·시각화 전용**이며 주문 수량 근거로 쓰지 않는다.
        exc = acc.get("hit_rate_excess")
        if exc is not None:
            p_adj = max(0.0, min(1.0, 0.5 + float(exc) / 100.0))
        else:
            p_adj = max(0.0, min(1.0, 0.5 + (p - mkt_p)))

        enough = n_for_sample >= KELLY_MIN_SAMPLES
        row = {
            "engine": eng,
            "total": total,
            "win_loss_n": n_for_sample,
            "flat": acc.get("flat"),
            "invalid": acc.get("invalid"),
            "sample_ok": enough,
            "p_abs": round(p, 4),
            "p_excess": round(p_adj, 4),
            "b": round(float(b), 3) if has_odds else None,
            "b_measured": has_odds,
        }
        if has_odds:
            row["mult_abs"] = round(_kelly(p, float(b)) if enough else KELLY_DEFAULT, 4)
            # 이름에서 Kelly를 뺀다 — 켈리가 아니기 때문(코덱스 권고).
            row["selection_edge_proxy"] = round(
                _kelly(p_adj, float(b)) if enough else KELLY_DEFAULT, 4)
        else:
            row["mult_abs"] = None
            row["selection_edge_proxy"] = None
        rows.append(row)

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "observed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_updated_at": data.get("updated_at"),
        "source_window_days": data.get("window_days"),
        "market_win_rate": round(mkt_p, 4),
        "market_n": mkt_n,
        "baseline_rule": base_rule,
        "engines": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true", help="파일 기록 없이 출력만")
    args = ap.parse_args()

    r = observe()
    if r.get("error"):
        print(f"[kelly-shadow] {r['error']}")
        return 1

    print(f"=== 켈리 섀도 관측 {r['observed_at']} ===")
    print(f"성과파일 updated_at={r['source_updated_at']} window_days={r['source_window_days']}")
    print(f"기저선 {r['market_win_rate'] * 100:.1f}%  (n={r['market_n']:,})  {r.get('baseline_rule', '')}")
    print()
    print(f"{'엔진':<22}{'승패표본':>9}{'flat':>6}{'무효':>6}{'승률':>8}"
          f"{'손익비':>8}{'켈리배수':>10}{'선택력':>9}  비고")
    for e in r["engines"]:
        if "verdict" in e:
            print(f"{e['engine']:<22}{e['total']:>9}  {e['verdict']}")
            continue
        note = []
        if not e["sample_ok"]:
            note.append(f"승패표본<{KELLY_MIN_SAMPLES}")
        if not e["b_measured"]:
            note.append("손익비 미실측 → 켈리 미산출")
        km = f"{e['mult_abs']:.3f}" if e.get("mult_abs") is not None else "—"
        se = f"{e['selection_edge_proxy']:.3f}" if e.get("selection_edge_proxy") is not None else "—"
        bs = f"{e['b']:.2f}" if e.get("b") is not None else "—"
        print(f"{e['engine']:<22}{e['win_loss_n']:>9}{e.get('flat') or 0:>6}"
              f"{e.get('invalid') or 0:>6}{e['p_abs'] * 100:>7.1f}%"
              f"{bs:>8}{km:>10}{se:>9}  {' / '.join(note)}")

    print()
    print("※ '켈리배수' = 절대 승률 기반. '선택력' = 시장 대비 초과분 기반으로 같은 식에")
    print("   넣어본 **진단용 대리지표**이며 켈리 확률이 아니다 — 롱온리 계좌의 p는")
    print("   '현금 기준으로 이겼을 확률'인데 초과 승률은 '시장보다 나았을 확률'이라")
    print("   변환식이 없다(코덱스 8/1 판정). 주문 수량 근거로 쓰지 않는다.")

    if not args.no_save:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[save] {OUT_FILE} (누적)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
