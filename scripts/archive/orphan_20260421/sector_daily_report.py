"""섹터 순환매 엔진 — Phase 1-5: 통합 대시보드 리포트.

Phase 1-1~1-4의 결과를 종합하여 장시작전 텍스트 리포트를 생성한다.

출력:
  1. 섹터 모멘텀 Top/Bottom
  2. 강세 섹터 내 래깅(catch-up) 종목
  3. 수급 신호 (스마트머니 / 스텔스 매집)
  4. 종합 추천

사용법:
  python scripts/sector_daily_report.py              # 텍스트 리포트
  python scripts/sector_daily_report.py --telegram   # 텔레그램 전송
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_json(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("%s 없음", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 리포트 생성
# ─────────────────────────────────────────────

def generate_report() -> str:
    """통합 대시보드 텍스트 리포트 생성."""
    momentum = load_json("sector_momentum.json")
    zscore = load_json("sector_zscore.json")
    flow = load_json("investor_flow.json")

    lines = []
    date_str = ""

    # ── 1. 섹터 모멘텀 ──
    if momentum:
        date_str = momentum.get("date", "")
        sectors = momentum.get("sectors", [])

        lines.append(f"{'━' * 50}")
        lines.append(f"  섹터 순환매 일일 리포트 — {date_str}")
        lines.append(f"{'━' * 50}")

        lines.append("")
        lines.append("▣ 섹터 모멘텀 순위")
        lines.append(f"  {'순위':>3} {'섹터':<8} {'점수':>5} {'20일%':>7} {'상대강도':>7} {'RSI':>5}")
        lines.append(f"  {'─' * 44}")

        for s in sectors:
            rank = s["rank"]
            tag = "★" if rank <= 3 else "▽" if rank > len(sectors) - 3 else " "
            lines.append(
                f"  {rank:>3} {s['sector']:<8} {s['momentum_score']:>5.1f} "
                f"{s['ret_20']:>+7.2f} {s['rel_strength']:>+7.2f} {s['rsi_14']:>5.1f} {tag}"
            )

        # Top 3 / Bottom 3 요약
        top3 = [s["sector"] for s in sectors[:3]]
        bottom3 = [s["sector"] for s in sectors[-3:]]
        lines.append(f"\n  ★ 강세: {', '.join(top3)}")
        lines.append(f"  ▽ 약세: {', '.join(bottom3)}")

    # ── 수급 데이터 전처리 (dict→list 변환) ──
    flow_list = []
    if flow:
        flow_sectors = flow.get("sectors", {})
        if isinstance(flow_sectors, dict):
            for name, vals in flow_sectors.items():
                fc = vals.get("foreign_cum", vals.get("foreign_cum_bil", 0))
                ic = vals.get("inst_cum", vals.get("inst_cum_bil", 0))
                flow_list.append({"sector": name, "foreign_cum_bil": fc, "inst_cum_bil": ic})
        else:
            flow_list = flow_sectors

    # ── 2. 수급 신호 ──
    if flow:
        cum_days = flow.get("cum_days", 5)
        lines.append(f"\n▣ 수급 신호 ({cum_days}일 누적)")

        smart_money = [s for s in flow_list if s.get("foreign_cum_bil", 0) > 0 and s.get("inst_cum_bil", 0) > 0]
        foreign_sell = [s for s in flow_list if s.get("foreign_cum_bil", 0) < -1000]

        if smart_money:
            lines.append(f"  ◆ 스마트머니 유입 ({len(smart_money)}개):")
            for s in smart_money[:5]:
                lines.append(
                    f"    {s['sector']}: 외인 {s['foreign_cum_bil']:+,.0f}억 + 기관 {s['inst_cum_bil']:+,.0f}억"
                )

        if foreign_sell:
            lines.append(f"  ⚠ 외인 대량매도:")
            for s in foreign_sell:
                lines.append(
                    f"    {s['sector']}: 외인 {s['foreign_cum_bil']:+,.0f}억 (기관 {s['inst_cum_bil']:+,.0f}억)")

    # ── 3. 래깅 종목 (z-score) ──
    if zscore:
        z_threshold = zscore.get("z_threshold", -0.8)
        total_candidates = zscore.get("total_candidates", 0)
        sector_results = zscore.get("sectors", {})

        lines.append(f"\n▣ 섹터 내 래깅 종목 (z < {z_threshold})")

        # 모멘텀 Top 5 섹터 + z-score 결합
        if momentum:
            top5_names = [s["sector"] for s in momentum["sectors"][:5]]
        else:
            top5_names = list(sector_results.keys())

        any_candidate = False
        for sector_name in top5_names:
            if sector_name not in sector_results:
                continue

            stocks = sector_results[sector_name]
            candidates = [s for s in stocks if s.get("z_20", 0) <= z_threshold]
            if not candidates:
                continue

            # 모멘텀 순위 찾기
            rank_str = ""
            if momentum:
                for m in momentum["sectors"]:
                    if m["sector"] == sector_name:
                        rank_str = f" [#{m['rank']}]"
                        break

            lines.append(f"  [{sector_name}]{rank_str}:")
            for c in candidates[:5]:
                z = c.get("z_20", 0)
                tag = "◆강" if z < -1.5 else "●중" if z < -1.0 else "○약"
                lines.append(
                    f"    {c['name']:<10} z={z:+.2f} 종목20일 {c.get('stock_ret_20', 0):+.1f}% {tag}"
                )
            any_candidate = True

        if not any_candidate:
            lines.append("  강세 섹터 내 래깅 종목 없음")

    # ── 4. 슈퍼섹터 릴레이 감지 ──
    relay = load_json("relay_signal.json")
    if relay and relay.get("relays"):
        lines.append(f"\n▣ 슈퍼섹터 릴레이 감지")
        for r in relay["relays"]:
            active = any(c["override"] for c in r.get("relay_candidates", []))
            status = "ACTIVE" if active else "WATCH"
            emoji = "🔥" if active else "👀"

            lines.append(
                f"  {emoji} [{r['supersector']}] {status}"
            )
            lines.append(
                f"    선행: {r['leader_sector']} "
                f"(#{r['leader_rank']}, RSI {r['leader_rsi']}, "
                f"20일 {r['leader_ret_20']:+.1f}%)"
            )
            if r["leader_rsi"] >= 75:
                lines.append("    → 과열 주의! 추격 금지")

            for c in r.get("relay_candidates", []):
                relay_mark = " ← RELAY!" if c["override"] else ""
                lines.append(
                    f"    릴레이: {c['sector']} "
                    f"거래대금 {c['volume_change_pct']:+.1f}%, "
                    f"RSI {c['rsi']:.0f}{relay_mark}"
                )

        override_count = relay.get("summary", {}).get("total_override_stocks", 0)
        if override_count:
            lines.append(f"  → Zone B→A 오버라이드: {override_count}종목 (HALF, -3% 손절)")

    # ── 4-2. 모멘텀 가속도 ──
    if momentum:
        accel_sectors = [
            s for s in momentum.get("sectors", [])
            if s.get("acceleration")
        ]
        if accel_sectors:
            lines.append(f"\n▣ 모멘텀 가속 감지 ⚡")
            for s in accel_sectors:
                lines.append(
                    f"  {s['sector']}: 순위 {s.get('rank_prev','?')}→{s['rank']} "
                    f"({s.get('rank_change',0):+d}), "
                    f"거래량 {s.get('vol_change_pct',0):+.0f}%"
                )

    # ── 5. 종합 추천 ──
    lines.append(f"\n{'━' * 50}")
    lines.append("▣ 종합 추천")
    lines.append(f"{'━' * 50}")

    # 모멘텀 Top + 스마트머니 교집합
    if momentum and flow_list:
        top5_set = set(s["sector"] for s in momentum["sectors"][:5])
        smart_set = set(
            s["sector"] for s in flow_list
            if s["foreign_cum_bil"] > 0 and s["inst_cum_bil"] > 0
        )
        overlap = top5_set & smart_set
        if overlap:
            lines.append(f"  ★ 모멘텀 + 스마트머니: {', '.join(overlap)}")
        else:
            lines.append("  모멘텀 Top5 중 스마트머니 겹침 없음")

        # 모멘텀 Top + 외인매도 → 주의
        foreign_sell_set = set(s["sector"] for s in flow_list if s["foreign_cum_bil"] < -1000)
        warn = top5_set & foreign_sell_set
        if warn:
            lines.append(f"  ⚠ 모멘텀 강세 + 외인매도 (주의): {', '.join(warn)}")

    # z-score 최적 후보: 모멘텀 Top + z-score 래깅
    if zscore and momentum:
        top3_names = [s["sector"] for s in momentum["sectors"][:3]]
        best_candidates = []
        for sn in top3_names:
            if sn in zscore.get("sectors", {}):
                for c in zscore["sectors"][sn]:
                    if c.get("z_20", 0) <= -0.8:
                        best_candidates.append(c)

        if best_candidates:
            best_candidates.sort(key=lambda x: x.get("z_20", 0))
            lines.append(f"\n  ◆ 최적 catch-up 후보:")
            for c in best_candidates[:5]:
                lines.append(
                    f"    {c.get('sector','')}/{c['name']} z={c.get('z_20',0):+.2f}"
                )

    report = "\n".join(lines)
    return report


# ─────────────────────────────────────────────
# 텔레그램 전송
# ─────────────────────────────────────────────

def send_telegram(text: str):
    """텔레그램으로 리포트 전송."""
    try:
        from src.telegram_sender import send_message
        # 텔레그램은 4096자 제한
        if len(text) > 4000:
            # 분할 전송
            chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                send_message(chunk)
        else:
            send_message(text)
        logger.info("텔레그램 전송 완료")
    except Exception as e:
        logger.error("텔레그램 전송 실패: %s", e)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 순환매 통합 리포트")
    parser.add_argument("--telegram", action="store_true",
                        help="텔레그램으로 전송")
    args = parser.parse_args()

    report = generate_report()
    print(report)

    # 파일 저장
    out_path = DATA_DIR / "daily_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("리포트 → %s", out_path)

    if args.telegram:
        send_telegram(report)


if __name__ == "__main__":
    main()
