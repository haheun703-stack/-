"""
3단 예측 체인 — 실행 스크립트

사용법:
    python -u -X utf8 scripts/run_predict_chain.py              # 시그널 생성 (stdout)
    python -u -X utf8 scripts/run_predict_chain.py --send       # 텔레그램 발송
    python -u -X utf8 scripts/run_predict_chain.py --blind      # 블라인드 테스트 로그

실행 시점: 16:35 KST (유럽장 오픈 30분 후, BAT 스케줄)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SIGNAL_DIR = Path("data/us_market")
SIGNAL_PATH = SIGNAL_DIR / "predict_chain_signal.json"
BLIND_DIR = Path("data/us_market/predict_chain_blind")


def _load_settings() -> dict:
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def format_telegram_message(result: dict) -> str:
    """텔레그램 메시지 포맷."""
    s1 = result.get("stage1_asian_risk", {})
    s2 = result.get("stage2_europe_open", {})
    s3 = result.get("stage3_divergence", {})

    signal = result.get("final_signal", "?")
    score = result.get("final_score", 0)
    conf = result.get("confidence", 0)
    match = result.get("direction_match", False)

    # 시그널 이모지
    if signal == "BULL":
        sig_icon = "\U0001f7e2"  # green circle
    elif signal == "BEAR":
        sig_icon = "\U0001f534"  # red circle
    else:
        sig_icon = "\u26aa"  # white circle

    match_mark = "\u2705" if match else "\u274c"

    lines = [
        f"{sig_icon} [3\ub2e8 \uc608\uce21 \uccb4\uc778] {signal}",
        f"Score: {score:+.2f}% | \uc2e0\ub8b0\ub3c4: {conf:.0f}%",
        f"\ubc29\ud5a5 \uc77c\uce58: {match_mark}",
        "",
    ]

    # Stage 1: 아시안 리스크
    audjpy = s1.get("audjpy_ret_pct", "?")
    cnh = s1.get("usdcnh_ret_pct", "?")
    s1_dir = s1.get("direction", "?")
    s1_sig = s1.get("audjpy_signal", "?")
    lines.append(f"[1\ub2e8] \uc544\uc2dc\uc548 \ub9ac\uc2a4\ud06c ({s1_dir})")
    lines.append(f"  AUD/JPY: {audjpy}% [{s1_sig}]")
    lines.append(f"  USD/CNH: {cnh}% [{s1.get('cnh_signal', '?')}]")
    lines.append(f"  ES\uc120\ubb3c: {s1.get('es_futures_ret_pct', '?')}%")
    lines.append("")

    # Stage 2: 유럽 오픈
    dax = s2.get("dax_30m_ret_pct", "?")
    eurusd = s2.get("eurusd_30m_ret_pct", "?")
    s2_dir = s2.get("direction", "?")
    dax_sig = s2.get("dax_signal", "?")
    lines.append(f"[2\ub2e8] \uc720\ub7fd \uc624\ud508 ({s2_dir})")
    lines.append(f"  DAX 30\ubd84: {dax}% [{dax_sig}]")
    lines.append(f"  EUR/USD: {eurusd}% [{s2.get('eurusd_signal', '?')}]")
    lines.append(f"  ES\uc120\ubb3c 30\ubd84: {s2.get('es_futures_30m_pct', '?')}%")
    lines.append("")

    # Stage 3: 괴리 감지
    alerts = s3.get("alerts", [])
    if alerts:
        lines.append(f"[3\ub2e8] \uad34\ub9ac \uac10\uc9c0 ({len(alerts)}\uac74)")
        for a in alerts:
            icon = "\u26a0\ufe0f" if "STRESS" in a["type"] else "\U0001f4a1"
            lines.append(f"  {icon} {a['type']}: {a['desc']}")
            lines.append(f"     \u2192 {a['action']}")
    else:
        lines.append("[3\ub2e8] \uad34\ub9ac \uac10\uc9c0: \uc5c6\uc74c")

    # 경고
    if result.get("divergence_alert"):
        lines.append("")
        lines.append("\u26a0\ufe0f \uad34\ub9ac \ubc1c\uacac \u2192 \uc2e0\ud638 \uc8fc\uc758!")

    return "\n".join(lines)


def save_blind_log(result: dict):
    """블라인드 테스트 로그 저장."""
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    log_entry = {
        "date": today,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_signal": result.get("final_signal"),
        "final_score": result.get("final_score"),
        "confidence": result.get("confidence"),
        "direction_match": result.get("direction_match"),
        "s1_direction": result.get("stage1_asian_risk", {}).get("direction"),
        "s1_score": result.get("stage1_asian_risk", {}).get("score"),
        "s2_direction": result.get("stage2_europe_open", {}).get("direction"),
        "s2_score": result.get("stage2_europe_open", {}).get("score"),
        "s2_dax_30m": result.get("stage2_europe_open", {}).get("dax_30m_ret_pct"),
        "divergence_alerts": result.get("stage3_divergence", {}).get("alert_count", 0),
        "divergence_alert": result.get("divergence_alert", False),
    }

    log_path = BLIND_DIR / f"{today}.json"
    log_path.write_text(
        json.dumps(log_entry, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # _index.json
    index_path = BLIND_DIR / "_index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        idx = {"start_date": today, "logs": []}
    if today not in idx["logs"]:
        idx["logs"].append(today)
    idx["total_days"] = len(idx["logs"])
    idx["last_updated"] = today
    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"블라인드 로그: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="3단 예측 체인 실행")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--blind", action="store_true", help="블라인드 테스트 로그")
    args = parser.parse_args()

    settings = _load_settings()

    # 1. 데이터 수집
    logger.info("5분봉 인트라데이 데이터 수집 중...")
    from src.nightwatch.predict_chain import PredictChainEngine, fetch_intraday

    intraday = fetch_intraday()
    logger.info(f"수집 완료: {len(intraday)}개 티커")

    if len(intraday) < 4:
        logger.error("최소 4개 티커 필요. 중단.")
        sys.exit(1)

    # 2. 엔진 실행
    engine = PredictChainEngine(settings)
    result = engine.compute(intraday)

    # 3. JSON 저장
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"저장: {SIGNAL_PATH}")

    # 4. 텔레그램
    msg = format_telegram_message(result)

    if args.send:
        try:
            from src.telegram_sender import send_message
            send_message(msg)
            logger.info("텔레그램 발송 완료")
        except Exception as e:
            logger.warning(f"텔레그램 실패: {e}")
            print(msg)
    else:
        print(msg)

    # 5. 블라인드 테스트
    if args.blind or settings.get("predict_chain", {}).get("blind_test", {}).get("enabled", False):
        save_blind_log(result)

    # 요약
    sig = result["final_signal"]
    sc = result["final_score"]
    conf = result["confidence"]
    match = result["direction_match"]
    logger.info(f"결과: {sig} (score:{sc:+.2f}% conf:{conf:.0f}% match:{match})")


if __name__ == "__main__":
    main()
