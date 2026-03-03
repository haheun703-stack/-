"""장중 AI 매도 모니터 — sell_brain + 자동 매도 실행

BAT-E 이후 09:30~14:50까지 주기적으로 실행:
  - 10:00, 12:00, 14:00: sell_brain 장중 체크
  - 14:30: preclose 체크
  - SELL_NOW → 즉시 시장가 매도
  - PARTIAL_SELL → 50% 시장가 매도
  - HOLD/WATCH → 유지

안전장치:
  - data/KILL_SWITCH 존재 시 중단
  - AI 장애 시 전종목 HOLD (매도 안 함)
  - 텔레그램 알림 (매도 실행 / 판단 변경)

사용법:
  python scripts/sell_monitor.py              # 풀 세션 (09:30~14:50)
  python scripts/sell_monitor.py --once       # 1회 체크 후 종료
  python scripts/sell_monitor.py --dry-run    # 매도 없이 판단만
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sell_monitor")

ROOT = Path(__file__).resolve().parent.parent
KILL_SWITCH = ROOT / "data" / "KILL_SWITCH"

# 체크 스케줄 (장중)
CHECK_TIMES = ["10:00", "12:00", "14:00"]
PRECLOSE_TIME = "14:30"


def load_json(name: str, default=None):
    p = ROOT / "data" / name
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default or {}


def save_json(name: str, data: dict):
    p = ROOT / "data" / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manual_entries() -> dict:
    """수동매수 종목 로드 → {ticker: {name, auto_sell, ...}}"""
    data = load_json("manual_entries.json")
    return data.get("entries", {})


def load_target_prices() -> dict:
    """tomorrow_picks.json에서 종목별 목표가/손절가 로드.

    Returns: {ticker: {target_price, stop_loss, entry_price, grade}}
    """
    tp_path = ROOT / "data" / "tomorrow_picks.json"
    if not tp_path.exists():
        return {}
    try:
        with open(tp_path, encoding="utf-8") as f:
            data = json.load(f)
        result = {}
        for p in data.get("picks", []):
            ticker = p.get("ticker", "")
            if ticker and p.get("target_price"):
                result[ticker] = {
                    "target_price": int(p.get("target_price", 0)),
                    "stop_loss": int(p.get("stop_loss", 0)),
                    "entry_price": int(p.get("entry_price", 0) or p.get("close", 0)),
                    "grade": p.get("grade", ""),
                }
        return result
    except Exception as e:
        logger.warning("목표가 로드 실패: %s", e)
        return {}


def fetch_supply_demand(ticker: str) -> dict:
    """KIS API로 당일 수급(외인/기관/개인) 조회"""
    try:
        from src.adapters.kis_intraday_adapter import KisIntradayAdapter
        adapter = KisIntradayAdapter()
        flow = adapter.fetch_investor_flow(ticker)
        return {
            "foreign_net": flow.get("foreign_net_buy", 0),
            "inst_net": flow.get("inst_net_buy", 0),
            "individual_net": flow.get("individual_net_buy", 0),
        }
    except Exception as e:
        logger.warning("수급 조회 실패 %s: %s", ticker, e)
        return {"foreign_net": 0, "inst_net": 0, "individual_net": 0}


def fetch_holdings() -> list[dict]:
    """KIS API에서 현재 보유 종목 조회 → sell_brain 입력 형식으로 변환

    보강 데이터:
      - 목표가/손절가 (tomorrow_picks.json)
      - 당일 수급 (KIS 투자자별 매매동향)
    """
    import mojito

    broker = mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=os.getenv("MODEL") != "REAL",
    )

    resp = broker.fetch_balance()
    holdings = resp.get("output1", [])

    # 목표가 맵 + 수동매수 맵 로드
    target_map = load_target_prices()
    manual_map = load_manual_entries()

    # v3 picks 목표가도 로드
    v3 = load_json("ai_v3_picks.json")
    for b in v3.get("buys", []):
        ticker = b.get("ticker", "")
        if ticker and ticker not in target_map:
            entry = int(b.get("entry_price", 0))
            if entry > 0:
                target_map[ticker] = {
                    "target_price": int(entry * (1 + b.get("target_pct", 15) / 100)),
                    "stop_loss": int(entry * (1 + b.get("stop_loss_pct", -8) / 100)),
                    "entry_price": entry,
                    "grade": "v3",
                }

    positions = []
    for h in holdings:
        qty = int(h.get("hldg_qty", 0))
        if qty == 0:
            continue

        ticker = h.get("pdno", "")
        name = h.get("prdt_name", "?")
        avg_price = float(h.get("pchs_avg_pric", 0))
        curr_price = int(h.get("prpr", 0))
        pnl_pct = float(h.get("evlu_pfls_rt", 0))

        # 목표가/손절가 (시그널 기반 or 기본 -8%/+10%)
        targets = target_map.get(ticker, {})
        target_price = targets.get("target_price", int(avg_price * 1.10))
        stop_loss = targets.get("stop_loss", int(avg_price * 0.92))
        grade = targets.get("grade", "수동매수")

        # 목표가 대비 현재 위치
        if target_price > stop_loss:
            progress = (curr_price - avg_price) / (target_price - avg_price) * 100 if target_price != avg_price else 0
        else:
            progress = 0

        # 당일 수급 조회 (장중에만)
        now_h = datetime.now().hour
        supply_demand = {}
        if 9 <= now_h <= 15:
            supply_demand = fetch_supply_demand(ticker)
            time.sleep(0.3)  # API 쓰로틀링

        # 수동매수 플래그
        is_manual = ticker in manual_map

        pos = {
            "ticker": ticker,
            "name": name,
            "entry_price": int(avg_price),
            "current_price": curr_price,
            "quantity": qty,
            "pnl_pct": pnl_pct,
            "hold_days": "?",
            "grade": grade,
            "trigger_type": "manual",
            "stop_loss": stop_loss,
            # 보강 데이터
            "target_price": target_price,
            "target_progress_pct": round(progress, 1),
            "foreign_net": supply_demand.get("foreign_net", 0),
            "inst_net": supply_demand.get("inst_net", 0),
            "individual_net": supply_demand.get("individual_net", 0),
            # 수동매수 보호
            "manual_entry": is_manual,
        }
        positions.append(pos)

        sd_str = ""
        if supply_demand:
            f_net = supply_demand.get("foreign_net", 0)
            i_net = supply_demand.get("inst_net", 0)
            sd_str = f" | 외인{f_net:+,} 기관{i_net:+,}"
        logger.info(
            "  %s: %+.1f%% (목표 %d→진행 %.0f%%){sd}".format(sd=sd_str),
            name, pnl_pct, target_price, progress,
        )

    return positions, broker


def execute_sell(broker, ticker: str, name: str, qty: int, sell_type: str, dry_run: bool) -> dict:
    """시장가 매도 실행"""
    if sell_type == "PARTIAL_SELL":
        qty = max(1, qty // 2)  # 50% 매도

    if dry_run:
        logger.info("[DRY-RUN] 매도 스킵: %s %d주", name, qty)
        return {"status": "dry_run", "ticker": ticker, "qty": qty}

    try:
        resp = broker.create_market_sell_order(ticker, qty)
        rt_cd = resp.get("rt_cd", "?")
        msg = resp.get("msg1", "")
        odno = resp.get("output", {}).get("ODNO", "")

        if rt_cd == "0":
            logger.info("[매도 성공] %s %d주, 주문번호=%s", name, qty, odno)
            return {"status": "ok", "ticker": ticker, "qty": qty, "order_no": odno}
        else:
            logger.error("[매도 실패] %s: %s", name, msg)
            return {"status": "failed", "ticker": ticker, "msg": msg}
    except Exception as e:
        logger.error("[매도 에러] %s: %s", name, e)
        return {"status": "error", "ticker": ticker, "msg": str(e)}


def send_telegram(text: str):
    """텔레그램 알림"""
    try:
        from src.telegram_sender import send_message
        send_message(text)
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


def apply_consensus(
    claude_result: dict,
    gpt_result: dict,
    positions: list[dict],
) -> dict:
    """GPT 촉매 + Claude 기술 → 합의 규칙 적용.

    합의 매트릭스:
      ALIVE + SELL → HOLD (촉매 우선)
      ALIVE + PARTIAL → HOLD (촉매 우선)
      FADING + SELL → PARTIAL (절충)
      FADING + PARTIAL → PARTIAL (일치)
      DEAD + SELL → SELL (일치)
      DEAD + PARTIAL → SELL (촉매 없으니 빠른 청산)
      manual_entry → 알림만 (자동매도 금지)
    """
    # GPT 촉매 맵 생성
    gpt_map = {}
    for gp in gpt_result.get("positions", []):
        gpt_map[gp.get("ticker", "")] = gp

    # Claude 판단 맵
    claude_map = {}
    for cp in claude_result.get("positions", []):
        claude_map[cp.get("ticker", "")] = cp

    # 포지션 manual 맵
    manual_map = {p["ticker"]: p.get("manual_entry", False) for p in positions}

    final_decisions = []
    catalyst_overrides = 0
    manual_blocks = 0

    for pos in positions:
        ticker = pos["ticker"]
        name = pos.get("name", "?")
        is_manual = manual_map.get(ticker, False)

        claude_pos = claude_map.get(ticker, {})
        claude_action = claude_pos.get("action", "HOLD")
        claude_reasoning = claude_pos.get("reasoning", "")

        gpt_pos = gpt_map.get(ticker, {})
        catalyst_status = gpt_pos.get("catalyst_status", "CATALYST_DEAD")
        catalyst_strength = gpt_pos.get("catalyst_strength", 0)
        tomorrow_dir = gpt_pos.get("tomorrow_direction", "UNCERTAIN")
        tomorrow_conf = gpt_pos.get("tomorrow_confidence", 0)
        catalyst_note = gpt_pos.get("primary_catalyst", "")

        # confidence < 0.5인 ALIVE는 무시 (이상 감지)
        if catalyst_status == "CATALYST_ALIVE" and catalyst_strength < 0.5:
            catalyst_status = "CATALYST_FADING"

        # ── 수동매수 보호 ──
        if is_manual:
            manual_blocks += 1
            final_decisions.append({
                "ticker": ticker,
                "name": name,
                "claude_action": claude_action,
                "gpt_catalyst": catalyst_status,
                "final_action": "HOLD",
                "override_reason": f"수동매수 보호 (Claude: {claude_action})",
                "manual_entry": True,
                "manual_blocked": True,
                "manual_alert": claude_action in ("SELL_NOW", "PARTIAL_SELL"),
                "reasoning": claude_reasoning,
                "catalyst_note": catalyst_note,
                "tomorrow_direction": tomorrow_dir,
            })
            continue

        # ── 합의 매트릭스 적용 ──
        final_action = claude_action  # 기본값: Claude 따름
        override_reason = ""

        if catalyst_status == "CATALYST_ALIVE":
            if claude_action in ("SELL_NOW", "PARTIAL_SELL"):
                final_action = "HOLD"
                override_reason = f"촉매 생존(강도 {catalyst_strength:.0%}) → 매도 보류"
                catalyst_overrides += 1
            # HOLD/WATCH → 그대로 HOLD

        elif catalyst_status == "CATALYST_FADING":
            if claude_action == "SELL_NOW":
                final_action = "PARTIAL_SELL"
                override_reason = "촉매 약화 중 → 절충 PARTIAL"
                catalyst_overrides += 1
            # PARTIAL/HOLD/WATCH → 그대로

        elif catalyst_status == "CATALYST_DEAD":
            if claude_action == "PARTIAL_SELL":
                final_action = "SELL_NOW"
                override_reason = "촉매 소멸 → 빠른 청산"
            # SELL_NOW → 그대로, HOLD → WATCH로 올릴 수도 있지만 Claude 판단 존중

        # CATALYST_NEW는 Claude 판단 그대로 (새 촉매 방향에 따라 다름)

        final_decisions.append({
            "ticker": ticker,
            "name": name,
            "claude_action": claude_action,
            "gpt_catalyst": catalyst_status,
            "final_action": final_action,
            "override_reason": override_reason,
            "manual_entry": False,
            "manual_blocked": False,
            "manual_alert": False,
            "reasoning": claude_reasoning,
            "catalyst_note": catalyst_note,
            "tomorrow_direction": tomorrow_dir,
        })

    return {
        "final_decisions": final_decisions,
        "consensus_stats": {
            "catalyst_overrides": catalyst_overrides,
            "manual_blocks": manual_blocks,
            "total": len(positions),
        },
    }


async def run_check(positions: list[dict], broker, check_type: str, dry_run: bool) -> dict:
    """1회 매도 체크 + 실행 — 듀얼 AI (GPT 촉매 + Claude 기술)"""
    import yaml
    from src.agents.sell_brain import SellBrainAgent

    strategic = load_json("ai_strategic_analysis.json")
    sector_focus = load_json("ai_sector_focus.json")

    # ── 듀얼 AI 활성화 여부 확인 ──
    dual_enabled = False
    try:
        with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dual_enabled = cfg.get("dual_sell_system", {}).get("enabled", False)
    except Exception:
        pass

    # ── Step 1: GPT 뉴스 촉매 분석 (듀얼 AI 활성 시) ──
    gpt_result = None
    if dual_enabled:
        try:
            from src.agents.gpt_catalyst import GPTCatalystAgent
            gpt_agent = GPTCatalystAgent()
            gpt_result = await gpt_agent.analyze_catalysts(positions)
            save_json("gpt_catalyst_analysis.json", gpt_result)
            logger.info("GPT 촉매 분석 완료: %d종목", len(gpt_result.get("positions", [])))
        except Exception as e:
            logger.error("GPT 촉매 분석 실패 (Claude 단독 진행): %s", e)
            gpt_result = None

    # ── Step 2: Claude Sell Brain (GPT 결과 포함) ──
    agent = SellBrainAgent()

    if check_type == "preclose":
        overnight = load_json(str(Path("us_market") / "overnight_signal.json"))
        claude_result = await agent.preclose_check(
            positions, strategic, overnight, gpt_catalyst=gpt_result,
        )
    else:
        claude_result = await agent.check_sell(
            positions, strategic, sector_focus, gpt_catalyst=gpt_result,
        )

    # 결과 저장
    save_json("ai_sell_cache.json", claude_result)

    # ── Step 3: 합의 규칙 적용 ──
    if dual_enabled and gpt_result and not gpt_result.get("error"):
        consensus = apply_consensus(claude_result, gpt_result, positions)
        save_json("ai_sell_consensus.json", consensus)
        stats = consensus.get("consensus_stats", {})
        logger.info(
            "합의 규칙: 촉매오버라이드=%d, 수동차단=%d (총 %d종목)",
            stats.get("catalyst_overrides", 0),
            stats.get("manual_blocks", 0),
            stats.get("total", 0),
        )
    else:
        # 듀얼 비활성 or GPT 장애 → Claude 단독 결정
        consensus = {
            "final_decisions": [
                {
                    "ticker": p.get("ticker", ""),
                    "name": p.get("name", ""),
                    "claude_action": p.get("action", "HOLD"),
                    "gpt_catalyst": "N/A",
                    "final_action": p.get("action", "HOLD"),
                    "override_reason": "",
                    "manual_entry": False,
                    "manual_blocked": any(
                        pos.get("manual_entry") for pos in positions
                        if pos.get("ticker") == p.get("ticker")
                    ),
                    "manual_alert": False,
                    "reasoning": p.get("reasoning", ""),
                    "catalyst_note": "",
                    "tomorrow_direction": "",
                }
                for p in claude_result.get("positions", [])
            ],
            "consensus_stats": {"catalyst_overrides": 0, "manual_blocks": 0, "total": len(positions)},
        }

    # ── Step 4: 최종 결정에 따라 매도 실행 ──
    sells_executed = []
    manual_alerts = []

    for decision in consensus.get("final_decisions", []):
        ticker = decision["ticker"]
        name = decision["name"]
        final_action = decision["final_action"]
        reasoning = decision.get("reasoning", "")
        override = decision.get("override_reason", "")

        # 수동매수 보호 — 알림만
        if decision.get("manual_blocked"):
            if decision.get("manual_alert"):
                manual_alerts.append(decision)
            continue

        if final_action in ("SELL_NOW", "PARTIAL_SELL"):
            pos = next((x for x in positions if x["ticker"] == ticker), None)
            if pos is None:
                continue

            qty = pos["quantity"]
            pnl = pos.get("pnl_pct", 0)

            if override:
                logger.info("[합의 매도] %s → %s (오버라이드: %s)", name, final_action, override)
            else:
                logger.info("[매도 결정] %s → %s (사유: %s)", name, final_action, reasoning[:50])

            sell_result = execute_sell(broker, ticker, name, qty, final_action, dry_run)
            sells_executed.append({
                "ticker": ticker,
                "name": name,
                "action": final_action,
                "claude_action": decision.get("claude_action", ""),
                "gpt_catalyst": decision.get("gpt_catalyst", ""),
                "override_reason": override,
                "qty": qty if final_action == "SELL_NOW" else max(1, qty // 2),
                "pnl_pct": pnl,
                "reasoning": reasoning,
                "result": sell_result,
            })

    # ── Step 5: 텔레그램 알림 ──
    _send_dual_telegram(check_type, consensus, sells_executed, manual_alerts, dry_run)

    return {
        "check_type": check_type,
        "total": len(positions),
        "sells": len(sells_executed),
        "catalyst_overrides": consensus.get("consensus_stats", {}).get("catalyst_overrides", 0),
        "manual_blocks": len(manual_alerts),
        "details": sells_executed,
    }


def _send_dual_telegram(
    check_type: str,
    consensus: dict,
    sells_executed: list,
    manual_alerts: list,
    dry_run: bool,
):
    """듀얼 AI 매도 결과 텔레그램 알림."""
    lines = []
    mode = "DRY" if dry_run else "LIVE"
    stats = consensus.get("consensus_stats", {})

    # 매도 실행 건
    if sells_executed:
        lines.append(f"{'🔴' if not dry_run else '🟠'} AI 매도 [{mode}] ({check_type})")
        for s in sells_executed:
            status = s["result"].get("status", "?")
            lines.append(f"  {s['name']}: {s['action']} {s['qty']}주 [{status}]")
            lines.append(f"    손익 {s['pnl_pct']:+.1f}% | {s['reasoning'][:40]}")
            if s.get("override_reason"):
                lines.append(f"    → {s['override_reason']}")
            gpt = s.get("gpt_catalyst", "")
            if gpt and gpt != "N/A":
                lines.append(f"    GPT: {gpt} | Claude: {s.get('claude_action', '')}")

    # 촉매 오버라이드 (매도 보류)
    overrides = [
        d for d in consensus.get("final_decisions", [])
        if d.get("override_reason") and d["final_action"] == "HOLD"
    ]
    if overrides:
        lines.append("")
        lines.append("🟢 촉매 보호 (매도 보류)")
        for o in overrides:
            lines.append(f"  {o['name']}: Claude {o['claude_action']} → HOLD")
            lines.append(f"    GPT: {o['gpt_catalyst']} | {o.get('catalyst_note', '')[:40]}")
            if o.get("tomorrow_direction"):
                lines.append(f"    내일: {o['tomorrow_direction']}")

    # 수동매수 보호
    if manual_alerts:
        lines.append("")
        lines.append("🔒 수동매수 보호")
        for m in manual_alerts:
            lines.append(f"  {m['name']}: Claude {m['claude_action']} → 자동매도 차단")

    # 아무것도 없으면 요약
    if not sells_executed and not overrides and not manual_alerts:
        hold_count = sum(
            1 for d in consensus.get("final_decisions", [])
            if d["final_action"] == "HOLD"
        )
        if check_type == "preclose":
            lines.append(f"🟡 프리클로즈: HOLD {hold_count}종목")
            if stats.get("catalyst_overrides"):
                lines.append(f"  촉매보호 {stats['catalyst_overrides']}건")

    if lines:
        send_telegram("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="장중 AI 매도 모니터")
    parser.add_argument("--once", action="store_true", help="1회 체크 후 종료")
    parser.add_argument("--dry-run", action="store_true", help="매도 없이 판단만")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  Sell Monitor 시작 (%s)", "DRY-RUN" if args.dry_run else "LIVE")
    logger.info("=" * 50)

    # KILL_SWITCH 체크
    if KILL_SWITCH.exists():
        logger.warning("KILL_SWITCH 존재 → 중단")
        return

    # 보유 종목 조회
    positions, broker = fetch_holdings()
    if not positions:
        logger.info("보유 종목 없음 → 종료")
        return

    logger.info("보유 종목: %d개", len(positions))
    for p in positions:
        logger.info("  %s(%s): %+.1f%%", p["name"], p["ticker"], p["pnl_pct"])

    if args.once:
        # 1회 체크
        now = datetime.now().strftime("%H:%M")
        check_type = "preclose" if now >= "14:20" else "intraday"
        result = asyncio.run(run_check(positions, broker, check_type, args.dry_run))
        logger.info("결과: %d종목 중 %d매도", result["total"], result["sells"])
        return

    # 풀 세션: 스케줄대로 체크
    checked = set()
    logger.info("풀 세션 모드: %s + %s 대기 중...", CHECK_TIMES, PRECLOSE_TIME)

    while True:
        if KILL_SWITCH.exists():
            logger.warning("KILL_SWITCH 감지 → 중단")
            send_telegram("🛑 Sell Monitor: KILL_SWITCH로 중단됨")
            break

        now = datetime.now()
        now_str = now.strftime("%H:%M")

        # 14:50 이후 종료
        if now_str >= "14:50":
            logger.info("14:50 → 세션 종료")
            break

        # 체크 시간 확인
        for ct in CHECK_TIMES:
            if ct not in checked and now_str >= ct:
                logger.info("=== %s 장중 체크 ===", ct)
                # 최신 잔고 다시 조회
                positions, broker = fetch_holdings()
                if positions:
                    result = asyncio.run(run_check(positions, broker, "intraday", args.dry_run))
                    logger.info("결과: %d종목 중 %d매도", result["total"], result["sells"])
                checked.add(ct)

        # 프리클로즈
        if PRECLOSE_TIME not in checked and now_str >= PRECLOSE_TIME:
            logger.info("=== %s 프리클로즈 체크 ===", PRECLOSE_TIME)
            positions, broker = fetch_holdings()
            if positions:
                result = asyncio.run(run_check(positions, broker, "preclose", args.dry_run))
                logger.info("프리클로즈: %d종목 중 %d매도", result["total"], result["sells"])
            checked.add(PRECLOSE_TIME)

        # 30초 대기
        time.sleep(30)


if __name__ == "__main__":
    main()
