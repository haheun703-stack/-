"""A-2: 신호 원장 적재 + 성과추적 재가동 (3/27 멈춤 복구).

코덱스 청사진 데이터 누적 파이프라인의 코어. 매일 장 마감 후 cron 실행:
  1. snapshot — 오늘 tomorrow_picks를 data/signal_ledger/{date}.json 으로 보존(원장 적재)
  2. refresh  — 충분히 지난(>=eval 최대일) 원장 신호의 D+1/D+3/D+5 성과를 FDR 종가로
                계산 → 엔진(sources)별 hit_rate/avg_ret 집계 → signal_accuracy.json 갱신
  3. (별도) source_weight_learner.py 가 갱신된 accuracy로 가중치 재학습

과거 백필 불가(원장 2일치뿐)이나, 본 파이프라인 가동 후부터 매일 누적되어
1~2개월 뒤 source_weight 학습이 데이터 기반으로 작동한다. 지금은 추측, 그때는 데이터.
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEDGER_DIR = PROJECT_ROOT / "data" / "signal_ledger"
PICKS_FILE = PROJECT_ROOT / "data" / "tomorrow_picks.json"
ACCURACY_FILE = PROJECT_ROOT / "data" / "market_learning" / "signal_accuracy.json"
WINDOW_DAYS = 60  # 성과 집계 롤링 윈도우
EVAL_DAYS = (1, 3, 5)
KST = timezone(timedelta(hours=9))

# tomorrow_picks sources(한글) → signal_accuracy 엔진명 매핑
SOURCE_MAP = {
    "눌림목": "pullback_scan", "반등임박": "pullback_scan",
    "매집": "accumulation_tracker", "누적": "accumulation_tracker", "매집추적": "accumulation_tracker",
    "세력": "whale_detect", "웨일": "whale_detect", "세력감지": "whale_detect",
    "퀀텀": "tomorrow_picks", "퀀트바닥": "tomorrow_picks",
    "공시": "dart_event", "DART": "dart_event",
    "거래량": "volume_spike", "거래량폭발": "volume_spike",
    "수급": "dual_buying", "외인기관": "dual_buying", "쌍끌이": "dual_buying",
}


def _today() -> str:
    return datetime.now(tz=KST).strftime("%Y-%m-%d")


def _map_engine(src: str) -> str:
    for key, eng in SOURCE_MAP.items():
        if key in str(src):
            return eng
    return str(src)


def snapshot_today() -> int:
    if not PICKS_FILE.exists():
        print(f"[snapshot] {PICKS_FILE} 없음 — 스킵")
        return 0
    picks = json.loads(PICKS_FILE.read_text(encoding="utf-8"))
    items = picks.get("picks", picks if isinstance(picks, list) else [])
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    out = LEDGER_DIR / f"{_today()}.json"
    out.write_text(
        json.dumps({"date": _today(), "picks": items}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[snapshot] {_today()} 신호 {len(items)}건 원장 적재 → {out.name}")
    return len(items)


def refresh_performance() -> dict:
    import FinanceDataReader as fdr

    today = datetime.now(tz=KST).date()
    cutoff = today - timedelta(days=WINDOW_DAYS)
    # engine -> {total, hit(d1>0), ret_sum(d1)}
    stats: dict[str, dict] = {}
    ledger_files = sorted(glob.glob(str(LEDGER_DIR / "*.json")))
    evaluated = 0
    for lf in ledger_files:
        try:
            data = json.loads(Path(lf).read_text(encoding="utf-8"))
        except Exception:
            continue
        sig_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        # 평가 가능: 신호일 + 최대 EVAL_DAYS 영업일 경과 + 윈도우 내
        if sig_date < cutoff or (today - sig_date).days < max(EVAL_DAYS) + 1:
            continue
        for p in data.get("picks", []):
            tk = str(p.get("ticker", "")).zfill(6)
            srcs = p.get("sources", []) or ["unknown"]
            try:
                df = fdr.DataReader(tk, data["date"],
                                    (sig_date + timedelta(days=15)).strftime("%Y-%m-%d"))
            except Exception:
                continue
            if df is None or len(df) < 5:  # entry(다음날 시가) + D+3 종가 필요
                continue
            entry = float(df["Open"].iloc[1])  # 신호 다음 영업일 시가 (가짜 진입 방지, 백테스트 동일 기준)
            if entry <= 0:
                continue
            d3 = df["Close"].iloc[3] / entry - 1  # D+3 종가 대표 보유
            for src in srcs:
                eng = _map_engine(src)
                st = stats.setdefault(eng, {"total": 0, "hit": 0, "ret": 0.0})
                st["total"] += 1
                st["hit"] += int(d3 > 0)
                st["ret"] += d3
            evaluated += 1
    # signal_accuracy 신규 누적 (★ 기존 병합 제거 — 3/27 데이터 실측 불일치 오염 차단)
    acc = {
        "updated_at": _today(),
        "window_days": WINDOW_DAYS,
        "entry_rule": "next_open",
        "eval_rule": "D+3_close",
        "signals": {},
    }
    signals = acc["signals"]
    for eng, st in stats.items():
        if st["total"] == 0:
            continue
        signals[eng] = {
            "total": st["total"],
            "hit": st["hit"],
            "hit_rate": round(st["hit"] / st["total"] * 100, 1),
            "avg_ret": round(st["ret"] / st["total"] * 100, 2),
            "days_tracked": len(ledger_files),
        }
    ACCURACY_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCURACY_FILE.write_text(
        json.dumps(acc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[refresh] {evaluated}개 신호 성과 평가 → signal_accuracy 갱신 "
          f"(엔진 {len(stats)}개)")
    return stats


def main() -> int:
    print(f"=== 신호 성과추적 파이프라인 ({_today()}) ===")
    snapshot_today()
    refresh_performance()
    print("→ 다음: source_weight_learner.py 로 가중치 재학습")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
