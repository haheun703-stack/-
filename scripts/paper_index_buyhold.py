#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""지수 Buy&Hold 페이퍼 (6번째 트랙) — 영구 벤치마크 기준선

config/benchmarks.yaml의 17종(국내외 지수·ETF·레버리지)을 baseline_start(6/22)에
매수 후 영구보유했다고 가정 → 각 지수의 정규화(6/22=자본금) 수익률 시계열 계산.
5개 전략 페이퍼의 공통 벤치마크("시장/레버리지를 그냥 들고 있었다면").

데이터: data/benchmark/{key}.csv (update_benchmarks.py 수집)
출력:   data/paper_portfolio_indexbh.json
실행(BAT-D 장후): python -X utf8 scripts/paper_index_buyhold.py

Buy&Hold는 보유상태 변화가 없어(6/22 entry 고정) 매번 CSV에서 전체 재계산한다
(idempotent) — 중복매매·주말가드 불필요. 통화 무관(수익률 % 비교).
"""
import json
import os
import sys
from datetime import datetime

ROOT = os.environ.get("QM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import yaml

CONFIG = os.path.join(ROOT, "config", "benchmarks.yaml")
BM_DIR = os.path.join(ROOT, "data", "benchmark")
PF_PATH = os.path.join(ROOT, "data", "paper_portfolio_indexbh.json")
INITIAL = 100_000_000


def load_config():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["benchmarks"], cfg.get("baseline_start", "2026-06-22")


def compute_one(key: str, meta: dict, baseline: str):
    """단일 벤치마크의 baseline 대비 정규화 수익률 시계열."""
    path = os.path.join(BM_DIR, f"{key}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Date" not in df.columns or "close" not in df.columns or df.empty:
        return None

    df = df.dropna(subset=["close"]).sort_values("Date").reset_index(drop=True)
    # baseline 이후(포함) 첫 거래일부터 = entry
    sub = df[df["Date"] >= baseline].reset_index(drop=True)
    if len(sub) < 1:
        return None

    entry_px = float(sub["close"].iloc[0])
    if entry_px <= 0:
        return None
    entry_date = str(sub["Date"].iloc[0])
    last_px = float(sub["close"].iloc[-1])
    last_date = str(sub["Date"].iloc[-1])

    daily, peak, mdd = [], INITIAL, 0.0
    for _, r in sub.iterrows():
        px = float(r["close"])
        eq = round(INITIAL * px / entry_px)
        peak = max(peak, eq)
        mdd = min(mdd, (eq / peak - 1) * 100)
        daily.append({
            "date": str(r["Date"]), "px": round(px, 4),
            "return_pct": round((px / entry_px - 1) * 100, 2), "equity": eq,
        })

    return {
        "symbol": meta["symbol"], "name": meta["name"], "mult": meta["mult"],
        "market": meta["market"], "group": meta["group"],
        "entry_date": entry_date, "entry_px": round(entry_px, 4),
        "last_date": last_date, "last_px": round(last_px, 4),
        "return_pct": round((last_px / entry_px - 1) * 100, 2),
        "equity": round(INITIAL * last_px / entry_px),
        "mdd_pct": round(mdd, 2), "days": len(daily), "daily": daily,
    }


def run():
    benchmarks, baseline = load_config()
    result = {}
    for key, meta in benchmarks.items():
        r = compute_one(key, meta, baseline)
        if r:
            result[key] = r

    if not result:
        print("[index_bh] 벤치마크 데이터 없음 — update_benchmarks.py 먼저 실행")
        sys.exit(1)

    ranking = sorted(
        [{"key": k, "name": v["name"], "mult": v["mult"], "market": v["market"],
          "return_pct": v["return_pct"]} for k, v in result.items()],
        key=lambda x: x["return_pct"], reverse=True,
    )

    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    if os.path.exists(PF_PATH):
        try:
            created = json.load(open(PF_PATH, encoding="utf-8")).get("created", created)
        except Exception:
            pass

    pf = {
        "created": created,
        "baseline_start": baseline,
        "initial_capital": INITIAL,
        "benchmarks": result,
        "ranking": ranking,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(PF_PATH), exist_ok=True)
    with open(PF_PATH, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

    # 요약 (수익률 순위)
    print(f"=== 지수 Buy&Hold 벤치마크 [{baseline}~{ranking and result[ranking[0]['key']]['last_date']}] {len(result)}종 ===")
    for row in ranking:
        v = result[row["key"]]
        print(f"  {v['name']:<20} {v['mult']}x {v['return_pct']:>+7.2f}%  "
              f"MDD{v['mdd_pct']:>+6.1f}%  ({v['days']}일)")
    print(f"[저장] {PF_PATH}")

    # FLOWX(Supabase) 관측 업로드 — leader_cycle 패턴(산출 직후 자기결과 전송).
    #   paper_index_benchmark 테이블 미생성 시 graceful(정보봇 DDL 대기). 매매 미반영.
    try:
        from src.adapters.flowx_uploader import FlowxUploader
        up = FlowxUploader()
        if up.is_active:
            ok = up.upload_index_benchmark(datetime.now().strftime("%Y-%m-%d"))
            print(f"[FLOWX] 지수벤치마크 업로드: {'OK' if ok else 'FAIL(테이블 대기?)'}")
        else:
            print("[FLOWX] Supabase 미연결 — 업로드 skip")
    except Exception as e:  # noqa: BLE001
        print(f"[FLOWX] 업로드 skip: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run()
