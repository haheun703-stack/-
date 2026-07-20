#!/usr/bin/env python3
"""BULL 전환 감시 + 진입 후보 즉시 제시 (B-15, 7/20 신설).

★문제 정의 (7/20 A-0 실측): 메인A 알파 -29.1%p. 방어는 실증됐으나 공세 경로가 없다.
  기계적 원인 = 레짐 전환이 JSON에 조용히 기록될 뿐 **아무에게도 통보되지 않는다**
  (`run_scenario_v1.py`에 알림 코드 0줄). 반등이 와도 다음 세션에 로그를 뒤져야 안다.

★이 스크립트가 하지 않는 것 — 반등 '타이밍 예측'.
  지수 레벨 타이밍은 우리 시스템에서 반복 기각됐다:
    - BULL 전환일의 타이밍 우월성: 이벤트 D+20 +1.49% vs 비이벤트 +1.36% → 엣지 t≈0.2 (7/7 철회)
    - 레짐 게이트(SCAN CAUTION 차단): 엣지 -0.62%p ns = 레짐 베타 오귀속 (7/9 기각)
    - 인버스 타이밍 t=-3.11 / 주도주 로테이션 t=-2.06 역방향 (7/7·7/10 기각)
    - 하방변동 진정(drv 90+→<60) 진입: edge -1.88%p 역방향 (7/20 본 과제 사전검증에서 기각)
    - 지수 수급(외인·기관 5일, 쌍끌이·전환) 진입: 전 조건 |t|<1.1 무효 (7/20 동)

★그래서 이 스크립트가 하는 것 — **반응 속도**. 타이밍은 못 맞히지만 무엇을 살지는 안다.
  생존 근거 = BULL 전환 이벤트일의 **종목 축(저PER)**: D+10 +0.91%p(t=3.37)·D+20 +0.90%p(t=3.13),
  승률 76%(26/34). 비이벤트일(+0.03%p) 대조 완료 → 이벤트일 한정 효과.
  (한계: 현존종목 생존편향·fund_PER 정적 오염·19~24년 표본. `data/backtest/bull_entry_playbook_report.md`)
  보조 축 = 수급 3단계 PHASE5_STAGE3(D+1 +2.02%·승률 59.1%) — ★표본 n=22·20일로 매우 작음,
  D+5는 -0.35% 음전. 확인용 태그로만 쓰고 단독 진입 근거로 삼지 않는다.

실주문 0 (freeze). 알림·후보 제시까지만.

실행:
    python scripts/bull_entry_gate.py
    python scripts/bull_entry_gate.py --simulate BULL   # 발동 경로 검증용
cron: BAT-D 후반 (run_scenario_v1 이후) — run_bat.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SCENARIO = DATA_DIR / "shadow" / "scenario_v1.json"
HISTORY = DATA_DIR / "shadow" / "scenario_v1_history.jsonl"
SIGNALS_DIR = DATA_DIR / "signals"
OUT_PATH = DATA_DIR / "metrics" / "bull_entry_gate.json"
LOG_JSONL = DATA_DIR / "metrics" / "bull_entry_gate.jsonl"

RISK_ON = {"BULL"}  # 진입 창이 열리는 레짐


def load_scenario() -> dict:
    if not SCENARIO.exists():
        raise FileNotFoundError(f"{SCENARIO} 없음 — run_scenario_v1.py 선행 필요")
    return json.load(open(SCENARIO, encoding="utf-8"))


def load_prev_regime(asof: str) -> str | None:
    """history에서 asof 직전 거래일의 regime_v0."""
    if not HISTORY.exists():
        return None
    rows = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("asof", "") < asof:
            rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("asof", ""))
    return rows[-1].get("regime_v0")


def load_stage3_tickers() -> set[str]:
    """오늘 signals에서 수급 3단계(PHASE5_STAGE3) 확인된 종목."""
    files = sorted(SIGNALS_DIR.glob("signals_*.json"))
    if not files:
        return set()
    try:
        d = json.load(open(files[-1], encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    for p in d.get("picks", []):
        sigs = p.get("signals", []) or p.get("tags", [])
        if any("PHASE5_STAGE3" in str(s) for s in sigs):
            t = str(p.get("ticker", "")).zfill(6)
            if t.isdigit():
                out.add(t)
    return out


def pick_candidates(watchlist: list[dict], stage3: set[str]) -> list[dict]:
    """검증된 종목축(저PER) 우선 + 수급 3단계 확인 태깅.

    저PER 태그가 이벤트일 초과수익의 유일한 생존 축이므로 1순위 정렬키로 쓴다.
    """
    out = []
    for w in watchlist:
        tags = w.get("tags", []) or []
        has_low_per = any("저PER" in str(t) for t in tags)
        ticker = str(w.get("ticker", "")).zfill(6)
        out.append({
            "ticker": ticker,
            "name": w.get("name", ""),
            "fv_long": w.get("fv_long", 0),
            "low_per": has_low_per,
            "stage3_confirmed": ticker in stage3,
            "tags": tags[:4],
        })
    # 저PER > 수급확인 > FV점수 순
    out.sort(key=lambda x: (x["low_per"], x["stage3_confirmed"], x["fv_long"]), reverse=True)
    return out


def build_report(sc: dict, prev_regime: str | None, candidates: list[dict],
                 triggered: bool) -> str:
    regime = sc.get("regime_v0", "?")
    lines = []
    if triggered:
        lines.append(f"🚀 *BULL 전환 감지* ({sc.get('asof')})")
        lines.append(f"레짐: {prev_regime} → {regime} | mode={sc.get('mode')}")
        lines.append("")
        low_per = [c for c in candidates if c["low_per"]]
        lines.append(f"진입 후보 (저PER 축 {len(low_per)}종 — 검증된 유일 축, D+20 +0.90%p t=3.13)")
        for c in low_per[:7]:
            mark = " ✅수급3단계" if c["stage3_confirmed"] else ""
            lines.append(f"  {c['name']}({c['ticker']}) FV{c['fv_long']:.0f}{mark}")
        if not low_per:
            lines.append("  (저PER 태그 후보 없음 — 무리한 진입 금지)")
        lines.append("")
        lines.append("※ 타이밍 우월성 근거 없음(기각) — 이 알림은 '반응 속도'용")
        lines.append("※ 낙폭과대 바닥주·고PER 추격 금지 / 실주문 0(freeze)")
    else:
        v3b = sc.get("regime_v3b", "?")
        div = " ⚠️V3b 선행" if sc.get("v3b_divergence") else ""
        lines.append(
            f"[BULL 게이트] {sc.get('asof')} {regime}"
            f"(v3b {v3b}){div} — 대기 · 후보 {len(candidates)}종 관측"
        )
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    import requests

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[BULL] 텔레그램 미설정")
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        return True
    except Exception as e:
        logger.warning("[BULL] 텔레그램 실패: %s", e)
        return False


def main():
    ap = argparse.ArgumentParser(description="BULL 전환 감시 + 진입 후보 제시")
    ap.add_argument("--simulate", type=str, default=None,
                    help="레짐 강제 지정 (발동 경로 검증용, 예: BULL)")
    ap.add_argument("--no-send", action="store_true", help="텔레그램 생략")
    args = ap.parse_args()

    sc = load_scenario()
    if args.simulate:
        sc = dict(sc)
        sc["regime_v0"] = args.simulate
        logger.info("[BULL] 시뮬레이션 모드 — 레짐 강제 %s", args.simulate)

    regime = sc.get("regime_v0", "?")
    asof = sc.get("asof", datetime.now().strftime("%Y-%m-%d"))
    prev = load_prev_regime(asof)
    if args.simulate:
        prev = prev or "CRISIS"

    # 전환 발동: 직전이 non-BULL이고 오늘 BULL (첫 진입일에만 알림 — 매일 스팸 방지)
    triggered = regime in RISK_ON and (prev is None or prev not in RISK_ON)

    stage3 = load_stage3_tickers()
    candidates = pick_candidates(sc.get("watchlist", []), stage3)

    report = build_report(sc, prev, candidates, triggered)
    print(report)

    result = {
        "asof": asof,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regime_v0": regime,
        "regime_v3b": sc.get("regime_v3b"),
        "prev_regime": prev,
        "triggered": triggered,
        "mode": sc.get("mode"),
        "stage3_count": len(stage3),
        "candidates": candidates,
        "validation": "observation_only_no_live_order",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    json.dump(result, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tmp.replace(OUT_PATH)
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in result.items() if k != "candidates"},
                           ensure_ascii=False) + "\n")

    # 발동 시에만 텔레그램 (평시엔 조용히 — 알림 피로 방지)
    if triggered and not args.no_send:
        ok = send_telegram(report)
        logger.info("[BULL] 전환 알림 %s", "발송 OK" if ok else "발송 SKIP")
    logger.info("[BULL] 저장: %s (발동=%s, 후보 %d종)", OUT_PATH, triggered, len(candidates))


if __name__ == "__main__":
    main()
