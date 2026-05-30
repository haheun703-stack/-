"""source_weight 학습 모듈 (A 청사진 — 엔진별 성과 → 가중치).

코덱스 청사진 핵심: "엔진별 성과를 계속 기록해서 가중치를 학습".
signal_accuracy.json(엔진별 hit_rate/avg_ret)을 읽어 selector가 쓸 가중치를 산출.
- 음수 평균수익 엔진 → weight 0 (live 신호 비활성, 손실 회피)
- 양수 엔진 → avg_ret × hit_rate 기반 가중치 → 정규화
- 표본 부족(<MIN_SAMPLE) → 보류(None)

출력: data/market_learning/source_weights.json
selector는 신호 점수에 해당 엔진 norm_weight를 곱해 최종 후보를 정렬한다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ACCURACY_FILE = PROJECT_ROOT / "data" / "market_learning" / "signal_accuracy.json"
WEIGHTS_FILE = PROJECT_ROOT / "data" / "market_learning" / "source_weights.json"
MIN_SAMPLE = 50  # 표본 부족 임계 (이하면 보류) — 5/30 상향: 신뢰표본 누적 후에만 학습
STALE_DAYS = 14  # signal_accuracy 갱신 경과 경보


def learn_weights(accuracy: dict) -> dict:
    signals = accuracy.get("signals", {})
    weights: dict[str, dict] = {}
    for eng, s in signals.items():
        total = int(s.get("total", 0) or 0)
        hr = float(s.get("hit_rate", 0) or 0)
        ar = float(s.get("avg_ret", 0) or 0)
        if total < MIN_SAMPLE:
            weights[eng] = {"weight": None, "reason": "insufficient_sample",
                            "total": total, "hit_rate": hr, "avg_ret": ar}
        elif ar <= 0:
            # 음수 평균수익 → 비활성 (손실 회피 = source_weight 학습의 가장 확실한 가치)
            weights[eng] = {"weight": 0.0, "reason": "negative_return_disabled",
                            "total": total, "hit_rate": hr, "avg_ret": ar}
        else:
            # 양수 → avg_ret × (hit_rate/100). 수익률·승률 동시 반영
            w = round(ar * (hr / 100.0), 4)
            weights[eng] = {"weight": w, "reason": "active",
                            "total": total, "hit_rate": hr, "avg_ret": ar}
    # 정규화 (활성 엔진 가중치 합 = 1)
    active = {k: v["weight"] for k, v in weights.items() if v.get("weight")}
    tot = sum(active.values()) or 1.0
    for k in active:
        weights[k]["norm_weight"] = round(active[k] / tot, 4)
    return weights


def main() -> int:
    if not ACCURACY_FILE.exists():
        print(f"[FAIL] {ACCURACY_FILE} 없음 — 성과추적 먼저 가동 필요")
        return 1
    acc = json.loads(ACCURACY_FILE.read_text(encoding="utf-8"))
    updated = acc.get("updated_at", "?")
    weights = learn_weights(acc)

    out = {
        "learned_from": str(ACCURACY_FILE.relative_to(PROJECT_ROOT)),
        "accuracy_updated_at": updated,
        "min_sample": MIN_SAMPLE,
        "weights": weights,
    }
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 보고
    print(f"=== source_weight 학습 (signal_accuracy updated={updated}) ===")
    # 신선도 경보
    try:
        u = datetime.fromisoformat(str(updated)).date()
        age = (datetime.now(tz=timezone(timedelta(hours=9))).date() - u).days
        if age > STALE_DAYS:
            print(f"⚠️ 경보: signal_accuracy {age}일 미갱신 — 성과추적 재가동 필요(A-2)")
    except Exception:
        pass
    order = sorted(weights.items(),
                   key=lambda x: (x[1].get("weight") if x[1].get("weight") is not None else -1),
                   reverse=True)
    print(f"{'엔진':<22}{'가중치':>9}{'정규':>8}{'승률':>8}{'평균수익':>9}{'표본':>7}  판정")
    for eng, v in order:
        w = v.get("weight")
        nw = v.get("norm_weight", "-")
        ws = f"{w:.4f}" if isinstance(w, float) else "보류"
        print(f"{eng:<22}{ws:>9}{str(nw):>8}{v['hit_rate']:>7.1f}%"
              f"{v['avg_ret']:>8.2f}%{v['total']:>7}  {v['reason']}")
    print(f"\n저장: {WEIGHTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
