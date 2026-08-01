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

★★핵심 산출 = 같은 신호에 대해 두 가지 켈리를 나란히 낸다.
  ⑴ 절대 승률 기반 — 지금 파일에 저장된 hit_rate를 그대로 쓴 값(코덱스 코드가 쓸 값)
  ⑵ 기저선 보정 승률 기반 — 같은 날 시장 전체 상승비율을 뺀 초과분만 실력으로 본 값
둘의 격차가 **B-35(승률이 시장 베타를 실력으로 기록하고 있다)의 크기**다.
7/31 실측 예: 시장 상승비율 89.7%인 날 신호 승률 92.8% → 초과 +3.1%p.

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
    """당일 유니버스 동일가중 상승 비율 — 승률의 기저선.

    ★7/24 교훈("비교 대상의 기저선을 맞춰라")을 승률에 적용한다. 그때 αEW가
    노출도를 안 맞춰 "덜 투자한 것"을 능력으로 둔갑시켰던 것과 같은 구조로,
    절대 승률은 **시장이 좋았던 날을 신호 실력으로 기록**한다.
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

    mkt_p, mkt_n = _market_win_rate()
    rows = []
    for eng, acc in sorted(signals.items(), key=lambda x: -x[1].get("total", 0)):
        total = int(acc.get("total", 0) or 0)
        p = _p_from(acc.get("hit_rate"))
        if p is None:
            rows.append({"engine": eng, "total": total, "verdict": "단위 계약 위반 — 미판정"})
            continue

        avg_win = abs(float(acc.get("avg_win", 0) or 0))
        avg_loss = abs(float(acc.get("avg_loss", 0) or 0))
        b = avg_win / avg_loss if (avg_win > 0 and avg_loss > 0) else KELLY_FALLBACK_ODDS
        has_odds = avg_win > 0 and avg_loss > 0

        # 기저선 보정: 시장 중립을 0.5로 놓고 초과 승률만 실력으로 본다.
        p_adj = max(0.0, min(1.0, 0.5 + (p - mkt_p)))

        enough = total >= KELLY_MIN_SAMPLES
        rows.append({
            "engine": eng,
            "total": total,
            "sample_ok": enough,
            "p_abs": round(p, 4),
            "p_excess": round(p_adj, 4),
            "b": round(b, 3),
            "b_measured": has_odds,          # False = 손익비 실측 없음(1:1 가정)
            "mult_abs": round(_kelly(p, b) if enough else KELLY_DEFAULT, 4),
            "mult_excess": round(_kelly(p_adj, b) if enough else KELLY_DEFAULT, 4),
        })

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "observed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_updated_at": data.get("updated_at"),
        "source_window_days": data.get("window_days"),
        "market_win_rate": round(mkt_p, 4),
        "market_n": mkt_n,
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
    print(f"시장 기저선(당일 상승비율) {r['market_win_rate'] * 100:.1f}%  (n={r['market_n']})")
    print()
    print(f"{'엔진':<22}{'표본':>6}{'승률':>8}{'초과승률':>9}{'손익비':>8}"
          f"{'배수(절대)':>11}{'배수(보정)':>11}  비고")
    for e in r["engines"]:
        if "verdict" in e:
            print(f"{e['engine']:<22}{e['total']:>6}  {e['verdict']}")
            continue
        note = []
        if not e["sample_ok"]:
            note.append(f"표본<{KELLY_MIN_SAMPLES}→기본값")
        if not e["b_measured"]:
            note.append("손익비 미실측(1:1 가정)")
        print(f"{e['engine']:<22}{e['total']:>6}{e['p_abs'] * 100:>7.1f}%"
              f"{e['p_excess'] * 100:>8.1f}%{e['b']:>8.2f}"
              f"{e['mult_abs']:>11.3f}{e['mult_excess']:>11.3f}  {' / '.join(note)}")

    # 격차 요약 — B-35(승률이 시장 베타를 실력으로 기록)의 크기
    graded = [e for e in r["engines"] if e.get("sample_ok") and "verdict" not in e]
    if graded:
        print()
        print("★ 절대 vs 보정 격차 (= 시장 베타를 실력으로 적은 몫)")
        for e in sorted(graded, key=lambda x: -(x["mult_abs"] - x["mult_excess"]))[:5]:
            gap = e["mult_abs"] - e["mult_excess"]
            ratio = (e["mult_abs"] / e["mult_excess"]) if e["mult_excess"] > 0 else float("inf")
            rs = f"{ratio:.1f}배" if ratio != float("inf") else "∞(보정 시 0)"
            print(f"  {e['engine']:<22} {e['mult_abs']:.3f} → {e['mult_excess']:.3f}"
                  f"  (차 {gap:+.3f}, {rs})")

    if not args.no_save:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[save] {OUT_FILE} (누적)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
