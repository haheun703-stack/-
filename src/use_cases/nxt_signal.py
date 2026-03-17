"""
NXT(넥스트레이드) 수급 시그널 분석기

프리마켓 / 애프터마켓 데이터를 분석하여:
  1. 애프터마켓 수급 → 다음날 프리마켓 매도/보유 판단
  2. 프리마켓 수급 → 본장 방향 예측 + SmartEntry 가격 반영

출력: data/nxt/nxt_signal.json
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NXT_DATA_DIR = PROJECT_ROOT / "data" / "nxt"
SIGNAL_PATH = NXT_DATA_DIR / "nxt_signal.json"


class NxtSignalAnalyzer:
    """NXT 프리/애프터마켓 수급 시그널 분석"""

    def __init__(self, config: dict | None = None):
        nxt_cfg = (config or {}).get("nxt", {})
        self.net_buy_strong = nxt_cfg.get("net_buy_strong_ratio", 0.60)
        self.premium_threshold = nxt_cfg.get("premium_threshold_pct", 0.3)
        self.volume_min = nxt_cfg.get("min_volume", 500)
        self.gap_up_threshold = nxt_cfg.get("pre_sell", {}).get("gap_up_threshold", 1.5)

    def analyze_aftermarket(self, target_date: str | None = None) -> list[dict]:
        """
        애프터마켓 수급 분석 → 다음날 프리마켓 매도/매수 후보

        판정 기준:
          1. 순매수비율 > 60% → 강한 매집 시그널
          2. 애프터 종가 > KRX 종가 → 프리미엄 존재
          3. 거래량 충분 (500주 이상)

        시그널:
          STRONG_BUY: 순매수 강 + 프리미엄 + 거래량 충분
          BUY: 순매수 강 OR 프리미엄
          NEUTRAL: 특이사항 없음
          SELL: 순매도 우위 + 디스카운트
        """
        dt = target_date or date.today().isoformat()
        after_path = NXT_DATA_DIR / f"nxt_after_{dt}.json"

        if not after_path.exists():
            logger.info("[NXT시그널] 애프터마켓 데이터 없음: %s", dt)
            return []

        data = json.loads(after_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        if not summary:
            logger.info("[NXT시그널] 애프터마켓 요약 비어있음: %s", dt)
            return []

        picks = []
        for ticker, info in summary.items():
            volume = info.get("volume", 0)
            if volume < self.volume_min:
                continue

            prev_close = info.get("prev_close", 0)
            last_price = info.get("last_price", 0)
            if prev_close <= 0 or last_price <= 0:
                continue

            # 프리미엄 계산
            premium_pct = (last_price / prev_close - 1) * 100
            net_buy_ratio = info.get("net_buy_ratio", 0.5)
            session_change = info.get("session_change_pct", 0)

            # 시그널 판정
            is_strong_buy = (
                net_buy_ratio >= self.net_buy_strong
                and premium_pct >= self.premium_threshold
                and volume >= self.volume_min * 2
            )
            is_buy = (
                net_buy_ratio >= self.net_buy_strong
                or premium_pct >= self.premium_threshold
            )
            is_sell = (
                net_buy_ratio < 0.40
                and premium_pct < -self.premium_threshold
            )

            if is_strong_buy:
                signal = "STRONG_BUY"
                confidence = min(0.95, 0.6 + net_buy_ratio * 0.3 + premium_pct * 0.05)
            elif is_buy:
                signal = "BUY"
                confidence = min(0.85, 0.4 + net_buy_ratio * 0.3 + premium_pct * 0.1)
            elif is_sell:
                signal = "SELL"
                confidence = min(0.80, 0.4 + (1 - net_buy_ratio) * 0.3)
            else:
                signal = "NEUTRAL"
                confidence = 0.50

            picks.append({
                "ticker": ticker,
                "after_premium_pct": round(premium_pct, 2),
                "net_buy_ratio": round(net_buy_ratio, 3),
                "volume": volume,
                "session_change_pct": round(session_change, 2),
                "signal": signal,
                "confidence": round(confidence, 2),
                "last_price": last_price,
                "prev_close": prev_close,
            })

        # 시그널 강도 순 정렬
        signal_order = {"STRONG_BUY": 0, "BUY": 1, "NEUTRAL": 2, "SELL": 3}
        picks.sort(key=lambda x: (signal_order.get(x["signal"], 9), -x["confidence"]))

        return picks

    def analyze_premarket(self, target_date: str | None = None) -> list[dict]:
        """
        프리마켓 수급 분석 → 본장 방향 예측

        판정 기준:
          1. 프리마켓 거래량 상위 = 당일 주도주 후보
          2. 갭업/갭다운 → SmartEntry 가격 조정에 반영
          3. 순매수/순매도 → 본장 수급 방향 예측
        """
        dt = target_date or date.today().isoformat()
        pre_path = NXT_DATA_DIR / f"nxt_pre_{dt}.json"

        if not pre_path.exists():
            logger.info("[NXT시그널] 프리마켓 데이터 없음: %s", dt)
            return []

        data = json.loads(pre_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        if not summary:
            # 스냅샷에서 마지막 데이터 활용
            snapshots = data.get("snapshots", [])
            if snapshots:
                summary = snapshots[-1].get("tickers", {})

        if not summary:
            return []

        directions = []
        for ticker, info in summary.items():
            price = info.get("last_price", info.get("price", 0))
            prev_close = info.get("prev_close", 0)
            volume = info.get("volume", 0)

            if price <= 0 or prev_close <= 0:
                continue

            gap_pct = (price / prev_close - 1) * 100
            net_buy_ratio = info.get("net_buy_ratio", 0.5)

            # 본장 방향 예측
            if gap_pct >= 2.0 and net_buy_ratio >= 0.55:
                expected = "GAP_UP_STRONG"
            elif gap_pct >= 0.5:
                expected = "GAP_UP"
            elif gap_pct <= -1.0 and net_buy_ratio < 0.45:
                expected = "GAP_DOWN"
            elif gap_pct <= -0.3:
                expected = "GAP_DOWN_MILD"
            else:
                expected = "FLAT"

            # SmartEntry 가격 조정 제안
            price_adjustment = 0
            if gap_pct >= self.gap_up_threshold:
                price_adjustment = round(gap_pct * 0.5, 1)  # 갭업의 50% 반영
            elif gap_pct <= -1.0:
                price_adjustment = round(gap_pct * 0.3, 1)  # 갭다운의 30% 반영 (더 싸게)

            directions.append({
                "ticker": ticker,
                "pre_price": price,
                "prev_close": prev_close,
                "pre_gap_pct": round(gap_pct, 2),
                "pre_volume": volume,
                "net_buy_ratio": round(net_buy_ratio, 3) if isinstance(net_buy_ratio, float) else net_buy_ratio,
                "expected_open": expected,
                "price_adjustment_pct": price_adjustment,
                "confidence": round(min(0.85, 0.5 + abs(gap_pct) * 0.1), 2),
            })

        # 거래량 기준 정렬 → 주도주 후보 순위
        directions.sort(key=lambda x: x.get("pre_volume", 0), reverse=True)
        for i, d in enumerate(directions):
            d["volume_rank"] = i + 1

        return directions

    def generate_signal(self, target_date: str | None = None) -> dict:
        """
        전체 NXT 시그널 생성 + 저장

        애프터마켓 분석 + 프리마켓 분석 통합.
        """
        dt = target_date or date.today().isoformat()

        # 애프터마켓: 전일 데이터로 분석
        yesterday = (date.fromisoformat(dt) - timedelta(days=1)).isoformat()
        # 주말 건너뛰기
        yd = date.fromisoformat(yesterday)
        while yd.weekday() >= 5:  # 토(5), 일(6)
            yd -= timedelta(days=1)
        yesterday = yd.isoformat()

        aftermarket_picks = self.analyze_aftermarket(yesterday)
        premarket_direction = self.analyze_premarket(dt)

        signal = {
            "date": dt,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "aftermarket_date": yesterday,
            "aftermarket_picks": aftermarket_picks,
            "premarket_direction": premarket_direction,
            "summary": {
                "after_strong_buy": sum(1 for p in aftermarket_picks if p["signal"] == "STRONG_BUY"),
                "after_buy": sum(1 for p in aftermarket_picks if p["signal"] == "BUY"),
                "after_sell": sum(1 for p in aftermarket_picks if p["signal"] == "SELL"),
                "pre_gap_up": sum(1 for d in premarket_direction if "GAP_UP" in d.get("expected_open", "")),
                "pre_gap_down": sum(1 for d in premarket_direction if "GAP_DOWN" in d.get("expected_open", "")),
                "pre_flat": sum(1 for d in premarket_direction if d.get("expected_open") == "FLAT"),
            },
        }

        # 저장
        SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNAL_PATH.write_text(
            json.dumps(signal, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[NXT시그널] 저장 완료: %s", SIGNAL_PATH)

        return signal

    def get_premarket_price_adjustments(self, target_date: str | None = None) -> dict[str, float]:
        """
        SmartEntry용: 종목별 NXT 프리마켓 가격 조정 비율 반환

        Returns:
            {ticker: adjustment_pct} 예: {"000660": 1.2} → 지정가 1.2% 상향
        """
        directions = self.analyze_premarket(target_date)
        adjustments = {}
        for d in directions:
            adj = d.get("price_adjustment_pct", 0)
            if abs(adj) >= 0.1:  # 0.1% 이상만
                adjustments[d["ticker"]] = adj
        return adjustments


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    analyzer = NxtSignalAnalyzer()
    dt = sys.argv[1] if len(sys.argv) > 1 else None
    result = analyzer.generate_signal(dt)

    print(f"\n=== NXT 시그널 ({result['date']}) ===")
    summary = result["summary"]
    print(f"  애프터({result['aftermarket_date']}): "
          f"STRONG_BUY {summary['after_strong_buy']}건, "
          f"BUY {summary['after_buy']}건, "
          f"SELL {summary['after_sell']}건")
    print(f"  프리마켓: "
          f"갭업 {summary['pre_gap_up']}건, "
          f"갭다운 {summary['pre_gap_down']}건, "
          f"보합 {summary['pre_flat']}건")

    if result["aftermarket_picks"]:
        print(f"\n  애프터 TOP 5:")
        for p in result["aftermarket_picks"][:5]:
            print(f"    {p['ticker']} {p['signal']} "
                  f"(프리미엄 {p['after_premium_pct']:+.1f}%, "
                  f"순매수 {p['net_buy_ratio']:.1%}, "
                  f"신뢰도 {p['confidence']:.0%})")

    if result["premarket_direction"]:
        print(f"\n  프리마켓 거래량 TOP 5:")
        for d in result["premarket_direction"][:5]:
            print(f"    #{d['volume_rank']} {d['ticker']} "
                  f"{d['expected_open']} "
                  f"(갭 {d['pre_gap_pct']:+.1f}%, "
                  f"거래량 {d['pre_volume']:,})")
