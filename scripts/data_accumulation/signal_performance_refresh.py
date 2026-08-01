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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"   # 기저선 계산용(B-35 ②)
# ★이 파일이 **전방 평가 정본**이다(8\1 B-35에서 생산자 분리).
# 신호 원장을 남겨 두고 **다음 영업일 시가 진입 → D+3 종가**로 채점한다
# = 신호가 나온 뒤의 성과만 본다. 소비자 6곳이 이 파일을 읽는다
# (dashboard_data · build_killer_picks · source_weight_learner · market_journal ·
#  run_v3_brain · flowx_uploader).
# ⚠️`daily_market_learner.py`는 **당일 종가/전일 종가**라 평가 성격이 다르다.
#   예전엔 둘이 이 파일을 같이 써서 ⑴여기 결과가 매일 18:53에 덮어써지고
#   ⑵`daily_log`가 초기화돼 "20일 누적"이 하루치가 됐다(days_tracked=1 실측).
#   그쪽은 이제 `signal_accuracy_daily.json`을 쓴다 — **다시 합치지 말 것.**
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


def _new_stat() -> dict:
    """엔진별 집계 그릇. win/loss/flat/invalid를 **분리**해서 센다(B-35 ③).

    ★왜 분리인가: 예전엔 `d3 <= 0`을 전부 패배로 세면서 손실 크기는 `abs(0)=0`을
    더했다. 그러면 `avg_loss = loss_ret / loss`에서 분모만 늘어 평균 손실이 과소
    평가되고, 손익비 `b = avg_win / avg_loss`가 부풀려진다 —
    실증: 승6(+5%)·무변동3·패1(-5%)일 때 b가 1.00 → 4.00으로 **4배 과대**,
    켈리가 0.200 → 0.500으로 뛴다(위험한 방향).
    `invalid`(조회 실패·봉 부족)는 애초에 평가가 성립하지 않은 건이라
    승/패/무변동 어디에도 넣으면 안 되고, 사유별로 세어 진단에 남긴다.
    """
    return {"total": 0, "hit": 0, "loss": 0, "flat": 0, "ret": 0.0,
            "win_ret": 0.0, "loss_ret": 0.0,
            "invalid": 0, "invalid_reasons": {},
            "base_n": 0, "base_hit": 0.0, "base_ret": 0.0}


def _baseline_by_date(dates: list[str]) -> dict[str, dict]:
    """신호일별 유니버스 기저선 — 신호와 **완전히 같은 규칙**으로 잰다. (B-35 ②)

    ★왜 필요한가(8\1): 전방 평가로 뽑은 승률 40.7%·평균 -1~-3%가 **"신호가 나쁘다"인지
    "그 46일이 나빴다"인지 절대값만으로는 구분되지 않는다.** 실제로 이 구간엔
    7/28 -10.84%, 7/29 -5.98% 같은 날이 끼어 있다.
    7/23에 "KOSPI 차감 알파는 이 국면에서 판별력 0"(유니버스 동일가중조차 KOSPI 대비
    D+20 -9.9%p)임을 확인하고 기저선을 동일가중으로 바꿨는데, 그 잣대가 승률에도
    그대로 필요하다.

    ★잣대를 맞추는 것이 핵심이다 — 진입도 청산도 신호와 동일하게 **다음 거래일 시가
    진입 → 3거래일 뒤 종가**로 계산한다. 기저선을 당일 등락률 같은 다른 기준으로 재면
    비교 자체가 성립하지 않는다(그 함정에 이미 한 번 빠졌다: kelly_shadow 1차).

    로컬 parquet을 쓴다 — FDR로 유니버스×기간을 다시 조회하면 5만 건이 넘는다.
    반환: {신호일: {"win_rate": 0~1, "avg_ret": 소수, "n": 종목수}}
    """
    import pandas as pd

    want = sorted(set(dates))
    if not want:
        return {}
    agg: dict[str, list[float]] = {d: [] for d in want}
    for pq in PROCESSED_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq, columns=["open", "close"])
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        idx = df.index
        for d in want:
            try:
                # 신호일 위치 → +1 시가 진입, +3 종가 청산 (신호 평가와 동일)
                pos = idx.searchsorted(pd.Timestamp(d))
                if pos >= len(idx) or idx[pos] != pd.Timestamp(d):
                    continue  # 그날 거래 없음(정지·상장전) → 기저선에서 제외
                if pos + 3 >= len(idx):
                    continue
                entry = float(df["open"].iloc[pos + 1])
                if entry <= 0:
                    continue
                agg[d].append(float(df["close"].iloc[pos + 3]) / entry - 1)
            except Exception:  # noqa: BLE001
                continue

    out: dict[str, dict] = {}
    for d, rets in agg.items():
        if not rets:
            continue
        out[d] = {
            "win_rate": sum(1 for r in rets if r > 0) / len(rets),
            "avg_ret": sum(rets) / len(rets),
            "n": len(rets),
        }
    return out


def refresh_performance() -> dict:
    import FinanceDataReader as fdr

    today = datetime.now(tz=KST).date()
    cutoff = today - timedelta(days=WINDOW_DAYS)
    # engine -> {total, hit(d3>0), ret_sum(d3), 기저선 누적}
    stats: dict[str, dict] = {}
    ledger_files = sorted(glob.glob(str(LEDGER_DIR / "*.json")))

    # 평가 대상 신호일을 먼저 모아 기저선을 한 번에 계산한다 (B-35 ②).
    eval_dates: list[str] = []
    for lf in ledger_files:
        try:
            d = json.loads(Path(lf).read_text(encoding="utf-8"))["date"]
        except Exception:  # noqa: BLE001
            continue
        sd = datetime.strptime(d, "%Y-%m-%d").date()
        if sd >= cutoff and (today - sd).days >= max(EVAL_DAYS) + 1:
            eval_dates.append(d)
    baseline = _baseline_by_date(eval_dates)
    print(f"[baseline] {len(baseline)}일 유니버스 기저선 산출 "
          f"(동일 규칙: next_open → D+3_close)")

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
        base = baseline.get(data["date"])
        for p in data.get("picks", []):
            tk = str(p.get("ticker", "")).zfill(6)
            srcs = p.get("sources", []) or ["unknown"]
            # ★invalid는 "평가 불가"이지 "패배"가 아니다 — 사유별로 세어 진단에 남긴다(B-35 ③).
            #   예전엔 조용히 `continue`라 얼마나 빠졌는지 알 수 없었다.
            reason = ""
            df = None
            try:
                df = fdr.DataReader(tk, data["date"],
                                    (sig_date + timedelta(days=15)).strftime("%Y-%m-%d"))
            except Exception:  # noqa: BLE001
                reason = "fetch_error"
            if not reason and (df is None or len(df) < 5):
                reason = "insufficient_bars"   # entry(다음날 시가) + D+3 종가 필요
            entry = 0.0
            if not reason:
                entry = float(df["Open"].iloc[1])  # 신호 다음 영업일 시가 (가짜 진입 방지)
                if entry <= 0:
                    reason = "bad_entry_price"
            if reason:
                for src in srcs:
                    eng = _map_engine(src)
                    st = stats.setdefault(eng, _new_stat())
                    st["invalid"] += 1
                    st["invalid_reasons"][reason] = st["invalid_reasons"].get(reason, 0) + 1
                continue

            d3 = df["Close"].iloc[3] / entry - 1  # D+3 종가 대표 보유
            for src in srcs:
                eng = _map_engine(src)
                st = stats.setdefault(eng, _new_stat())
                st["total"] += 1
                st["ret"] += d3
                # ★win/loss/flat 3분 (B-35 ③, 코덱스 Q3 권고 ③):
                #   예전엔 `d3 <= 0`을 전부 loss로 세면서 손실 크기는 0을 더해
                #   avg_loss가 과소평가되고 손익비 b가 부풀었다(실증 4배).
                #   p와 b는 **엄격한 win/loss 표본에서만** 계산한다.
                if d3 > 0:
                    st["hit"] += 1
                    st["win_ret"] += d3
                elif d3 < 0:
                    st["loss"] += 1
                    st["loss_ret"] += -d3      # 양수 크기로 누적
                else:
                    st["flat"] += 1            # 정확히 0 — 승도 패도 아니다
                # 기저선은 **그 신호가 난 날**의 것을 쌓는다 — 엔진마다 신호일 분포가
                # 다르므로 전체 평균을 쓰면 비교가 어긋난다.
                if base:
                    st["base_n"] += 1
                    st["base_hit"] += base["win_rate"]
                    st["base_ret"] += base["avg_ret"]
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
        row = {
            "total": st["total"],
            "hit": st["hit"],
            "hit_rate": round(st["hit"] / st["total"] * 100, 1),
            "avg_ret": round(st["ret"] / st["total"] * 100, 2),
            "days_tracked": len(ledger_files),
        }
        # ★win/loss/flat/invalid 분리 결과 (B-35 ③). 기존 두 필드는 그대로 두므로
        #   소비자 6곳은 영향 없다 — 아래는 전부 신규 필드다.
        #   단위는 **백분율 계약**으로 고정한다(소비자가 값 크기로 추측하지 않도록).
        wl = st["hit"] + st["loss"]
        row.update({
            "loss": st["loss"],
            "flat": st["flat"],          # 정확히 0 — 승도 패도 아님
            "invalid": st["invalid"],    # 평가 불가(조회 실패·봉 부족) — 패배가 아님
            "win_loss_n": wl,            # 켈리 최소표본은 total이 아니라 이 값에 건다
            # p·b는 **엄격한 승/패 표본에서만** 계산한다(코덱스 Q3·Q4)
            "hit_rate_strict": round(st["hit"] / wl * 100, 1) if wl else None,
            "avg_win": round(st["win_ret"] / st["hit"] * 100, 2) if st["hit"] else None,
            "avg_loss": round(st["loss_ret"] / st["loss"] * 100, 2) if st["loss"] else None,
        })
        if st["invalid_reasons"]:
            row["invalid_reasons"] = st["invalid_reasons"]
        # 손익비 b — 양쪽 표본이 다 있을 때만. 없으면 **가정하지 않고 null**
        # (1:1 같은 임의 가정을 값으로 박으면 소비자가 실측으로 오인한다).
        if row["avg_win"] and row["avg_loss"]:
            row["odds_b"] = round(row["avg_win"] / row["avg_loss"], 3)
        # ★기저선 대비(B-35 ②) — 절대 승률만으로는 "신호가 나쁜지 시장이 나빴는지"를
        #   가릴 수 없다. 같은 날·같은 규칙의 유니버스 평균을 빼서 **실력분만** 남긴다.
        #   기존 필드는 그대로 두므로 소비자 6곳은 영향 없다(신규 필드 추가만).
        if st["base_n"] > 0:
            b_hit = st["base_hit"] / st["base_n"] * 100
            b_ret = st["base_ret"] / st["base_n"] * 100
            row["baseline_hit_rate"] = round(b_hit, 1)
            row["baseline_avg_ret"] = round(b_ret, 2)
            row["hit_rate_excess"] = round(row["hit_rate"] - b_hit, 1)
            row["avg_ret_excess"] = round(row["avg_ret"] - b_ret, 2)
        signals[eng] = row

    # 전체 기저선 요약 — 그 기간 시장이 어땠는지 한 줄로 남긴다.
    if baseline:
        tot_n = sum(v["n"] for v in baseline.values())
        acc["baseline"] = {
            "rule": "universe equal-weight, next_open → D+3_close",
            "days": len(baseline),
            "samples": tot_n,
            "hit_rate": round(sum(v["win_rate"] * v["n"] for v in baseline.values())
                              / tot_n * 100, 1),
            "avg_ret": round(sum(v["avg_ret"] * v["n"] for v in baseline.values())
                             / tot_n * 100, 2),
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
