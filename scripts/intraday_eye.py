"""
INTRADAY EYE — 장중 실시간 위험/기회 감시 시스템

5분 간격으로 보유 종목 + KOSPI를 모니터링하여
위험(급락, 시장급변) 또는 기회(목표 접근, 신규 기회) 발생 시
텔레그램 이벤트 알림 (0~3건/일).

8개 감지기:
  EYE-01: 수급 반전 (외국인+기관 순매도 전환)
  EYE-02: 급락 감지 (전일종가 대비 -3% 이하)
  EYE-03: 거래량 폭발 (5분 거래량 > 20일 평균 × 3)
  EYE-04: 이평선 이탈 (20일선 하향 돌파)
  EYE-05: 시장 급변 (KOSPI 등락 ±1.5%)
  EYE-06: 목표가 접근 (수익률 +8% 이상)
  EYE-07: 신규 기회 (워치리스트 종목 시그널)
  EYE-08: 긴급 뉴스 (JGIS 정보봇 breaking_alerts 감지)

독립 실행: brain.py/signal_engine.py 수정 없음.
API 부하: 5분당 6~8콜 (KIS rate limit 이내)

사용법:
    python scripts/intraday_eye.py              # 실행 (08:55~15:20 루프)
    python scripts/intraday_eye.py --once       # 1회만 실행 후 종료
    python scripts/intraday_eye.py --dry-run    # 텔레그램 미발송 (로그만)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.telegram_sender import send_message

logger = logging.getLogger("intraday_eye")

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
EYE_LOG_DIR = DATA_DIR / "eye_events"

# ─────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────

def _load_settings() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("intraday_eye", {})


def _load_json(path: Path) -> dict | list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────
# KIS API 래퍼 (경량)
# ─────────────────────────────────────────

class _KisLite:
    """IntradayEye 전용 경량 KIS API 래퍼.
    kis_order_adapter 전체를 로드하지 않고, 필요한 것만 가져옴.
    """

    def __init__(self):
        import mojito
        is_mock = os.getenv("MODEL") != "REAL"
        self.broker = mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY", ""),
            api_secret=os.getenv("KIS_APP_SECRET", ""),
            acc_no=os.getenv("KIS_ACC_NO", ""),
            mock=is_mock,
        )

    def fetch_price(self, ticker: str) -> dict:
        """현재가 조회 → {current_price, change_pct, volume, open_price, prev_close}"""
        try:
            data = self.broker.fetch_price(ticker)
            o = data.get("output", {})
            return {
                "current_price": int(o.get("stck_prpr", 0)),
                "change_pct": float(o.get("prdy_ctrt", 0)),
                "volume": int(o.get("acml_vol", 0)),
                "open_price": int(o.get("stck_oprc", 0)),
                "prev_close": int(o.get("stck_sdpr", 0)),
                "high_price": int(o.get("stck_hgpr", 0)),
                "low_price": int(o.get("stck_lwpr", 0)),
                "name": o.get("hts_kor_isnm", ""),
            }
        except Exception as e:
            logger.error("[KIS] %s 현재가 조회 실패: %s", ticker, e)
            return {}

    def fetch_balance(self) -> list[dict]:
        """보유종목 목록 조회"""
        try:
            data = self.broker.fetch_balance()
            holdings = data.get("output1", [])
            return [
                {
                    "ticker": h.get("pdno", ""),
                    "name": h.get("prdt_name", ""),
                    "quantity": int(h.get("hldg_qty", 0)),
                    "avg_price": float(h.get("pchs_avg_pric", 0)),
                    "current_price": int(h.get("prpr", 0)),
                    "pnl_pct": float(h.get("evlu_pfls_rt", 0)),
                }
                for h in holdings if int(h.get("hldg_qty", 0)) > 0
            ]
        except Exception as e:
            logger.error("[KIS] 잔고 조회 실패: %s", e)
            return []

    def fetch_investor(self, ticker: str) -> dict:
        """투자자별 매매동향 (외국인/기관 당일 순매수)"""
        try:
            from src.adapters.kis_intraday_adapter import KisIntradayAdapter
            adapter = KisIntradayAdapter(broker=self.broker)
            data = adapter._api_get(
                "uapi/domestic-stock/v1/quotations/inquire-investor",
                "FHKST01010900",
                {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            )
            items = data.get("output", [])
            if not items:
                return {"foreign_net": 0, "inst_net": 0}
            today = items[0] if items else {}
            return {
                "foreign_net": int(today.get("frgn_ntby_qty", 0)),
                "inst_net": int(today.get("orgn_ntby_qty", 0)),
            }
        except Exception as e:
            logger.error("[KIS] %s 투자자 조회 실패: %s", ticker, e)
            return {"foreign_net": 0, "inst_net": 0}


# ─────────────────────────────────────────
# 쿨다운 관리
# ─────────────────────────────────────────

class _Cooldown:
    """종목+감지기 키 기반 쿨다운 (기본 30분)"""

    def __init__(self, cooldown_minutes: int = 30):
        self._cooldown_sec = cooldown_minutes * 60
        self._last_fired: dict[str, float] = {}

    def can_fire(self, key: str) -> bool:
        last = self._last_fired.get(key, 0)
        return (time.time() - last) >= self._cooldown_sec

    def mark_fired(self, key: str):
        self._last_fired[key] = time.time()


# ─────────────────────────────────────────
# EYE 이벤트 데이터
# ─────────────────────────────────────────

_EYE_EMOJI = {
    "EYE-01": "🔄",  # 수급 반전
    "EYE-02": "🔻",  # 급락
    "EYE-03": "💥",  # 거래량 폭발
    "EYE-04": "📉",  # 이평선 이탈
    "EYE-05": "🌊",  # 시장 급변
    "EYE-06": "🎯",  # 목표가 접근
    "EYE-07": "✨",  # 신규 기회
}


def _format_alert(eye_id: str, name: str, ticker: str,
                  price: int, change_pct: float, reason: str) -> str:
    """텔레그램 알림 포맷"""
    emoji = _EYE_EMOJI.get(eye_id, "👁")
    sign = "+" if change_pct > 0 else ""
    return (
        f"{emoji} [{eye_id}] {name}({ticker})\n"
        f"   현재가 {price:,}원 ({sign}{change_pct:.1f}%)\n"
        f"   사유: {reason}"
    )


# ─────────────────────────────────────────
# IntradayEye 메인 클래스
# ─────────────────────────────────────────

class IntradayEye:
    """장중 실시간 위험/기회 감시"""

    # KOSPI 지수 코드 (KIS API)
    KOSPI_TICKER = "0001"

    def __init__(self, dry_run: bool = False, killer_picks: bool = False):
        self.dry_run = dry_run
        self.settings = _load_settings()
        self.cooldown = _Cooldown(self.settings.get("cooldown_minutes", 30))
        self.kis = _KisLite()

        # 보유종목 (세션 시작 시 1회 로드, 5분봉마다 갱신하지 않음)
        self.holdings: list[dict] = []
        # 워치리스트 종목
        self.watchlist: list[str] = list(self.settings.get("watchlist", []))

        # 킬러픽 종목 워치리스트 자동 추가
        if killer_picks:
            self._load_killer_watchlist()

        # 전일종가 캐시 (세션 중 고정)
        self._prev_close: dict[str, int] = {}
        # 전일 거래량 캐시 (20일 평균 대용)
        self._avg_volume: dict[str, int] = {}

        # 당일 이벤트 로그
        self._events: list[dict] = []

        # 이전 사이클 데이터 (KOSPI 전 체크용)
        self._prev_kospi_price: int = 0

    def _load_killer_watchlist(self):
        """킬러픽 종목을 워치리스트에 추가."""
        p = DATA_DIR / "killer_picks.json"
        if not p.exists():
            logger.warning("[EYE] killer_picks.json 없음 — 킬러픽 워치리스트 스킵")
            return
        with open(p, encoding="utf-8") as f:
            kp = json.load(f)
        extra = set()
        for s in kp.get("cross_validated_top5", []):
            extra.add(s["ticker"])
        for e in kp.get("etf_top5", []):
            if e.get("ticker") and e["ticker"] != "cash":
                extra.add(e["ticker"])
        new = extra - set(self.watchlist)
        if new:
            self.watchlist.extend(sorted(new))
            logger.info("[EYE] 킬러픽에서 %d종목 워치리스트 추가 (총 %d)", len(new), len(self.watchlist))

    def _load_holdings(self):
        """KIS 잔고에서 보유종목 로드"""
        self.holdings = self.kis.fetch_balance()
        logger.info("[EYE] 보유종목 %d개 로드", len(self.holdings))
        for h in self.holdings:
            ticker = h["ticker"]
            if ticker not in self._prev_close:
                p = self.kis.fetch_price(ticker)
                if p:
                    self._prev_close[ticker] = p.get("prev_close", 0)
                    # 20일 평균 거래량 대용: 전일 거래량 (완벽하진 않지만 API 절약)
                    self._avg_volume[ticker] = max(p.get("volume", 0), 1)
                time.sleep(0.1)

    def _fire_alert(self, eye_id: str, ticker: str, name: str,
                    price: int, change_pct: float, reason: str) -> bool:
        """쿨다운 체크 후 텔레그램 발송"""
        key = f"{eye_id}:{ticker}"
        if not self.cooldown.can_fire(key):
            logger.debug("[EYE] 쿨다운 중 — %s %s", eye_id, ticker)
            return False

        msg = _format_alert(eye_id, name, ticker, price, change_pct, reason)
        event = {
            "time": datetime.now().strftime("%H:%M"),
            "eye_id": eye_id,
            "ticker": ticker,
            "name": name,
            "price": price,
            "change_pct": change_pct,
            "reason": reason,
        }
        self._events.append(event)

        if self.dry_run:
            logger.info("[DRY-RUN] %s", msg.replace("\n", " | "))
        else:
            send_message(msg)
            logger.info("[SENT] %s", msg.replace("\n", " | "))

        self.cooldown.mark_fired(key)
        return True

    # ─── 감지기 ───

    def _eye02_sharp_drop(self, ticker: str, name: str, price_data: dict):
        """EYE-02: 급락 감지 — 전일종가 대비 -3% 이하"""
        threshold = self.settings.get("eye02_drop_pct", -3.0)
        change = price_data.get("change_pct", 0)
        if change <= threshold:
            self._fire_alert(
                "EYE-02", ticker, name,
                price_data["current_price"], change,
                f"전일 대비 {change:.1f}% 급락 (임계: {threshold}%)",
            )

    def _eye05_market_shock(self, kospi_data: dict):
        """EYE-05: 시장 급변 — KOSPI 등락 ±1.5%"""
        threshold = self.settings.get("eye05_kospi_pct", 1.5)
        change = kospi_data.get("change_pct", 0)
        if abs(change) >= threshold:
            direction = "급등" if change > 0 else "급락"
            self._fire_alert(
                "EYE-05", self.KOSPI_TICKER, "KOSPI",
                kospi_data.get("current_price", 0), change,
                f"KOSPI {direction} {change:+.1f}% (임계: ±{threshold}%)",
            )

    def _eye01_flow_reversal(self, ticker: str, name: str, price_data: dict):
        """EYE-01: 수급 반전 — 외국인+기관 순매도 전환"""
        investor = self.kis.fetch_investor(ticker)
        frgn = investor.get("foreign_net", 0)
        inst = investor.get("inst_net", 0)
        # 둘 다 순매도이면 알림
        if frgn < 0 and inst < 0:
            total_sell = abs(frgn) + abs(inst)
            min_qty = self.settings.get("eye01_min_sell_qty", 1000)
            if total_sell >= min_qty:
                self._fire_alert(
                    "EYE-01", ticker, name,
                    price_data.get("current_price", 0),
                    price_data.get("change_pct", 0),
                    f"외국인 {frgn:+,}주 + 기관 {inst:+,}주 (쌍매도 {total_sell:,}주)",
                )

    def _eye03_volume_explosion(self, ticker: str, name: str, price_data: dict):
        """EYE-03: 거래량 폭발 — 당일 거래량 > 평균 × 배수"""
        vol_mult = self.settings.get("eye03_vol_mult", 3.0)
        avg_vol = self._avg_volume.get(ticker, 0)
        cur_vol = price_data.get("volume", 0)
        if avg_vol > 0 and cur_vol > avg_vol * vol_mult:
            ratio = cur_vol / avg_vol
            self._fire_alert(
                "EYE-03", ticker, name,
                price_data.get("current_price", 0),
                price_data.get("change_pct", 0),
                f"거래량 {cur_vol:,} (평균의 {ratio:.1f}배, 임계: {vol_mult}배)",
            )

    def _eye04_ma_break(self, ticker: str, name: str, price_data: dict):
        """EYE-04: 이평선 이탈 — 현재가 < 20일선 (전일종가 >= 20일선이었던 경우)

        NOTE: 장중에 20일선 정확한 값은 없음. 전일종가의 -2%를 간이 판정으로 사용.
        parquet 데이터의 sma_20 값이 있으면 정확하지만, API 부하 절감을 위해 생략.
        """
        prev = self._prev_close.get(ticker, 0)
        if prev <= 0:
            return
        # 간이 20일선 판정: 전일종가 대비 -2% 이하 → MA 이탈 가능성
        ma_proxy_pct = self.settings.get("eye04_ma_proxy_pct", -2.0)
        change = price_data.get("change_pct", 0)
        if change <= ma_proxy_pct:
            self._fire_alert(
                "EYE-04", ticker, name,
                price_data.get("current_price", 0), change,
                f"이평선 이탈 가능 ({change:.1f}%, 간이 판정)",
            )

    def _eye06_target_approach(self, ticker: str, name: str, holding: dict, price_data: dict):
        """EYE-06: 목표가 접근 — 수익률 +8% 이상"""
        target_pct = self.settings.get("eye06_target_pct", 8.0)
        avg_price = holding.get("avg_price", 0)
        if avg_price <= 0:
            return
        cur_price = price_data.get("current_price", 0)
        pnl_pct = (cur_price / avg_price - 1) * 100
        if pnl_pct >= target_pct:
            self._fire_alert(
                "EYE-06", ticker, name,
                cur_price, price_data.get("change_pct", 0),
                f"수익률 +{pnl_pct:.1f}% (목표 +{target_pct}% 도달)",
            )

    def _eye07_new_opportunity(self, price_data_map: dict):
        """EYE-07: 신규 기회 — 워치리스트 종목 급등/급락

        워치리스트 종목 중 전일 대비 +3% 이상 → 매수 기회 시그널
        """
        threshold = self.settings.get("eye07_opp_pct", 3.0)
        held_tickers = {h["ticker"] for h in self.holdings}
        for ticker in self.watchlist:
            if ticker in held_tickers:
                continue  # 이미 보유 중이면 스킵

            try:
                p = self.kis.fetch_price(ticker)
                if not p or not p.get("current_price"):
                    continue
                change = p.get("change_pct", 0)
                if change >= threshold:
                    name = p.get("name") or ticker
                    self._fire_alert(
                        "EYE-07", ticker, name,
                        p["current_price"], change,
                        f"워치리스트 급등 +{change:.1f}% (임계: +{threshold}%)",
                    )
            except Exception as e:
                logger.warning("[EYE-07] %s 조회 실패: %s", ticker, e)
            finally:
                time.sleep(0.1)

    def _eye08_breaking_news(self):
        """EYE-08: JGIS 긴급 뉴스 — breaking_alerts.json 체크"""
        try:
            from src.alpha.scenario_detector import ScenarioDetector
        except ImportError:
            return

        detector = ScenarioDetector()
        alerts = detector.check_breaking_alerts()
        if not alerts:
            return

        for alert in alerts:
            severity = alert.get("severity", "MEDIUM")
            title = alert.get("title", "알 수 없는 뉴스")
            source = alert.get("source", "")
            scenarios = alert.get("scenarios_triggered", [])
            sectors = alert.get("sectors_impact", {})

            # 쿨다운 체크 (같은 제목은 하루 1회)
            key = f"EYE-08:{title[:20]}"
            if not self.cooldown.can_fire(key):
                continue

            # 메시지 포맷
            lines = [
                f"🚨 [긴급 뉴스] {title}",
                f"출처: {source} | 심각도: {severity}",
            ]
            if scenarios:
                lines.append(f"시나리오 활성: {', '.join(scenarios)}")
            if sectors:
                for sec, impact in sectors.items():
                    lines.append(f"  {sec}: {impact}")

            # 보유종목 영향 체크
            if self.holdings and sectors:
                # 간단한 매칭: 보유종목 이름에 섹터명 포함 여부
                for h in self.holdings:
                    name = h.get("name", "")
                    for sec in sectors:
                        if sec in name:
                            lines.append(f"  → 보유: {name} ({sec} 관련)")

            msg = "\n".join(lines)

            event = {
                "time": datetime.now().strftime("%H:%M"),
                "eye_id": "EYE-08",
                "ticker": "",
                "name": title[:30],
                "price": 0,
                "change_pct": 0,
                "reason": f"JGIS 긴급뉴스 [{severity}]",
            }
            self._events.append(event)

            if self.dry_run:
                logger.info("[DRY-RUN] %s", msg.replace("\n", " | "))
            else:
                send_message(msg)
                logger.info("[SENT] EYE-08 긴급뉴스: %s", title[:30])

            self.cooldown.mark_fired(key)

    # ─── 메인 사이클 ───

    def run_cycle(self):
        """1회 감시 사이클 실행"""
        now = datetime.now()
        logger.info("[EYE] === 사이클 시작 %s ===", now.strftime("%H:%M:%S"))

        if not self.holdings:
            self._load_holdings()

        price_data_map: dict[str, dict] = {}

        # 1) KOSPI 시장 전체 체크 (EYE-05)
        kospi = self.kis.fetch_price(self.KOSPI_TICKER)
        if kospi and kospi.get("current_price"):
            self._eye05_market_shock(kospi)
            self._prev_kospi_price = kospi["current_price"]

        # 2) 보유종목별 감지기 (EYE-01~04, 06)
        for h in self.holdings:
            ticker = h["ticker"]
            name = h.get("name", ticker)
            time.sleep(0.15)  # rate limit

            p = self.kis.fetch_price(ticker)
            if not p or not p.get("current_price"):
                continue

            price_data_map[ticker] = p

            # 방어 감지기 (우선)
            self._eye02_sharp_drop(ticker, name, p)
            self._eye04_ma_break(ticker, name, p)

            # 수급 감지기 (API 1콜 추가)
            # 보유종목이 5개 이하일 때만 수급 체크 (API 부하 절감)
            if len(self.holdings) <= 5:
                self._eye01_flow_reversal(ticker, name, p)

            # 거래량 감지기
            self._eye03_volume_explosion(ticker, name, p)

            # 기회 감지기
            self._eye06_target_approach(ticker, name, h, p)

        # 3) 워치리스트 신규 기회 (EYE-07)
        if self.watchlist:
            self._eye07_new_opportunity(price_data_map)

        # 4) JGIS 긴급 뉴스 (EYE-08)
        try:
            self._eye08_breaking_news()
        except Exception as e:
            logger.warning("[EYE-08] 긴급뉴스 체크 실패: %s", e)

        logger.info("[EYE] === 사이클 완료 (이벤트 누적 %d건) ===", len(self._events))

    def save_daily_log(self):
        """당일 이벤트 로그 저장"""
        EYE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        today_str = date.today().isoformat()
        log_path = EYE_LOG_DIR / f"{today_str}.json"
        log_data = {
            "date": today_str,
            "total_events": len(self._events),
            "events": self._events,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        logger.info("[EYE] 일일 로그 저장: %s (%d건)", log_path.name, len(self._events))

    def run_loop(self, once: bool = False):
        """메인 루프: 09:05~15:20, 5분 간격"""
        interval = self.settings.get("interval_seconds", 300)
        start_time = self.settings.get("start_time", "09:05")
        end_time = self.settings.get("end_time", "15:20")

        logger.info(
            "[EYE] INTRADAY EYE 시작 — %s~%s, %d초 간격, dry_run=%s",
            start_time, end_time, interval, self.dry_run,
        )

        # 초기 보유종목 로드
        self._load_holdings()

        if once:
            self.run_cycle()
            self.save_daily_log()
            return

        while True:
            now = datetime.now()
            now_str = now.strftime("%H:%M")

            if now_str < start_time:
                wait = 30
                logger.info("[EYE] 장 시작 전 대기 (%s < %s)", now_str, start_time)
                time.sleep(wait)
                continue

            if now_str > end_time:
                logger.info("[EYE] 장 마감 → 종료 (%s > %s)", now_str, end_time)
                break

            try:
                self.run_cycle()
            except Exception as e:
                logger.error("[EYE] 사이클 오류: %s", e, exc_info=True)

            time.sleep(interval)

        # 장 마감 → 일일 로그 저장
        self.save_daily_log()
        logger.info("[EYE] INTRADAY EYE 종료")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="INTRADAY EYE — 장중 실시간 감시")
    parser.add_argument("--once", action="store_true", help="1회만 실행 후 종료")
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 미발송 (로그만)")
    parser.add_argument("--killer-picks", action="store_true",
                        help="킬러픽 종목을 워치리스트에 추가")
    args = parser.parse_args()

    # 로깅 설정
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                LOG_DIR / "intraday_eye.log", encoding="utf-8",
            ),
        ],
    )

    eye = IntradayEye(dry_run=args.dry_run, killer_picks=args.killer_picks)
    eye.run_loop(once=args.once)


if __name__ == "__main__":
    main()
