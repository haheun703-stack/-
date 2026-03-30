"""FLOWX Supabase 업로드 어댑터.

퀀트봇 데이터를 FLOWX PRO 대시보드용 Supabase DB에 업로드.
담당 테이블: etf_signals, foreign_flow, paper_trades, short_signals

Usage:
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()
    uploader.upload_etf_signals(rows)
    uploader.upload_foreign_flow(rows)
    uploader.upload_ai_picks(rows)
    uploader.upload_paper_trade(trade)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class FlowxUploader:
    """FLOWX Supabase 업로드 클라이언트."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

        if not url or not key:
            logger.warning("[FLOWX] SUPABASE_URL/KEY 미설정 — 업로드 비활성")
            self.client = None
            return

        try:
            from supabase import create_client
            self.client = create_client(url, key)
            logger.info("[FLOWX] Supabase 연결 완료")
        except Exception as e:
            logger.error("[FLOWX] Supabase 연결 실패: %s", e)
            self.client = None

    @property
    def is_active(self) -> bool:
        return self.client is not None

    # ── ETF 시그널 ─────────────────────────────────

    def upload_etf_signals(self, rows: list[dict]) -> bool:
        """ETF 시그널 업로드 (UPSERT on date+code)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("etf_signals").upsert(
                rows, on_conflict="date,code"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] ETF 시그널 업로드 응답 비어있음")
                return False
            logger.info("[FLOWX] ETF 시그널 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] ETF 시그널 업로드 실패: %s", e)
            return False

    # ── 외국인 자금 흐름 ─────────────────────────────

    def upload_foreign_flow(self, rows: list[dict]) -> bool:
        """외국인 자금 흐름 업로드 (UPSERT on date+code).

        Supabase 테이블명: china_flow (기존 테이블 재활용, 스키마 동일).
        """
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("china_flow").upsert(
                rows, on_conflict="date,code"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 외국인 자금 업로드 응답 비어있음")
                return False
            logger.info("[FLOWX] 외국인 자금 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 외국인 자금 업로드 실패: %s", e)
            return False

    # ── AI 추천 (short_signals) ─────────────────────

    def upload_ai_picks(self, rows: list[dict]) -> bool:
        """AI 추천 종목 업로드 (UPSERT on date+code)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("short_signals").upsert(
                rows, on_conflict="date,code"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] AI 추천 업로드 응답 비어있음")
                return False
            logger.info("[FLOWX] AI 추천 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] AI 추천 업로드 실패: %s", e)
            return False

    # ── 모닝 브리핑 ──────────────────────────────────

    def upload_morning_briefing(self, briefing: dict) -> bool:
        """모닝 브리핑 업로드 (UPSERT on date)."""
        if not self.is_active or not briefing:
            return False
        try:
            import json as _json
            row = {
                "date": briefing["date"],
                "market_status": briefing.get("market_status", "NEUTRAL"),
                "us_summary": briefing.get("us_summary", ""),
                "kr_summary": briefing.get("kr_summary", ""),
                "news_picks": _json.loads(_json.dumps(briefing.get("news_picks", []))),
                "sector_focus": _json.loads(_json.dumps(briefing.get("sector_focus", []))),
            }
            result = self.client.table("morning_briefings").upsert(
                row, on_conflict="date"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 모닝 브리핑 업로드 응답 비어있음: %s", briefing["date"])
                return False
            logger.info("[FLOWX] 모닝 브리핑 업로드: %s", briefing["date"])
            return True
        except Exception as e:
            logger.error("[FLOWX] 모닝 브리핑 업로드 실패: %s", e)
            return False

    # ── 시그널 로깅 ──────────────────────────────────

    def insert_signal(self, signal: dict) -> bool:
        """시그널 INSERT (STEP 2/3용)."""
        if not self.is_active or not signal:
            return False
        try:
            result = self.client.table("signals").insert(signal).execute()
            if not result.data:
                logger.warning("[FLOWX] 시그널 기록 응답 비어있음")
                return False
            logger.info("[FLOWX] 시그널 기록: %s %s %s",
                        signal.get("bot_type", "?"),
                        signal.get("signal_type", "?"),
                        signal.get("ticker_name", "?"))
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 기록 실패: %s", e)
            return False

    def update_signal_performance(self, signal_id: str, updates: dict) -> bool:
        """시그널 수익률/상태 업데이트 (STEP 2용)."""
        if not self.is_active or not signal_id:
            return False
        try:
            result = self.client.table("signals").update(updates).eq(
                "id", signal_id
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 시그널 업데이트 응답 비어있음: %s", signal_id[:8])
                return False
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 업데이트 실패: %s", e)
            return False

    def upsert_scoreboard(self, rows: list[dict]) -> bool:
        """성적표 집계 UPSERT (STEP 2용)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("scoreboard").upsert(
                rows, on_conflict="bot_type,period"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 성적표 업로드 응답 비어있음")
                return False
            logger.info("[FLOWX] 성적표 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 성적표 업로드 실패: %s", e)
            return False

    # ── 시그널 조회 ──────────────────────────────────

    def fetch_open_signals(self, bot_type: str = "") -> list[dict]:
        """OPEN 상태 시그널 조회."""
        if not self.is_active:
            return []
        try:
            q = self.client.table("signals").select("*").eq("status", "OPEN")
            if bot_type:
                q = q.eq("bot_type", bot_type)
            result = q.execute()
            return result.data or []
        except Exception as e:
            logger.error("[FLOWX] 시그널 조회 실패: %s", e)
            return []

    def fetch_signals_by_period(
        self, bot_type: str, status_list: list[str], from_date: str
    ) -> list[dict]:
        """기간별 시그널 조회 (성적표 집계용)."""
        if not self.is_active:
            return []
        try:
            q = (
                self.client.table("signals")
                .select("*")
                .in_("status", status_list)
                .gte("signal_date", from_date)
            )
            if bot_type and bot_type != "ALL":
                q = q.eq("bot_type", bot_type)
            result = q.execute()
            return result.data or []
        except Exception as e:
            logger.error("[FLOWX] 기간별 시그널 조회 실패: %s", e)
            return []

    def close_signal(self, signal_id: str, close_data: dict) -> bool:
        """시그널 종료 (CLOSED/STOPPED)."""
        if not self.is_active or not signal_id:
            return False
        try:
            result = self.client.table("signals").update(close_data).eq(
                "id", signal_id
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 시그널 종료 응답 비어있음: %s", signal_id[:8])
                return False
            logger.info("[FLOWX] 시그널 종료: %s → %s", signal_id[:8], close_data.get("status", "?"))
            return True
        except Exception as e:
            logger.error("[FLOWX] 시그널 종료 실패: %s", e)
            return False

    # ── 시나리오 대시보드 (FLOWX /quant) ─────────────

    def upload_quant_scenarios(self, scenario_data: dict, date_str: str) -> bool:
        """시나리오 대시보드 데이터 업로드 (UPSERT on date)."""
        if not self.is_active or not scenario_data:
            return False
        try:
            row = {
                "date": date_str,
                "data": scenario_data,
            }
            result = self.client.table("quant_scenario_dashboard").upsert(
                row, on_conflict="date"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 시나리오 대시보드 업로드 응답 비어있음: %s", date_str)
                return False
            sc_count = len(scenario_data.get("active_scenarios", []))
            logger.info("[FLOWX] 시나리오 대시보드 업로드: %s (%d개 시나리오)", date_str, sc_count)
            return True
        except Exception as e:
            logger.error("[FLOWX] 시나리오 대시보드 업로드 실패: %s", e)
            return False

    # ── 자비스 컨트롤타워 ─────────────────────────────

    def upload_jarvis_data(self, jarvis_data: dict, date_str: str) -> bool:
        """자비스 컨트롤타워 데이터 업로드 (UPSERT on date).

        jarvis_data 구조:
          picks: {target_date_label, mode_label, total_candidates, stats, picks[]}
          accuracy: {pullback_scan: {hit_rate, total}, ...}
          brain: {regime, vix, cash_ratio}
          shield: {status, max_drawdown}
        """
        if not self.is_active or not jarvis_data:
            return False
        try:
            row = {
                "date": date_str,
                "data": jarvis_data,
            }
            result = self.client.table("quant_jarvis").upsert(
                row, on_conflict="date"
            ).execute()
            if not result.data:
                logger.warning("[FLOWX] 자비스 업로드 응답 비어있음: %s", date_str)
                return False
            n_picks = len(jarvis_data.get("picks", {}).get("picks", []))
            logger.info("[FLOWX] 자비스 컨트롤타워 업로드: %s (%d종목)", date_str, n_picks)
            return True
        except Exception as e:
            logger.error("[FLOWX] 자비스 업로드 실패: %s", e)
            return False

    # ── 퀀트 6개 테이블 (웹봇 독립 API용) ─────────────

    def _upsert_date_data(self, table: str, data: dict, date_str: str, label: str) -> bool:
        """공통 date+data UPSERT 패턴."""
        if not self.is_active or not data:
            return False
        try:
            row = {"date": date_str, "data": data}
            result = self.client.table(table).upsert(row, on_conflict="date").execute()
            if not result.data:
                logger.warning("[FLOWX] %s 업로드 응답 비어있음: %s", label, date_str)
                return False
            logger.info("[FLOWX] %s 업로드: %s (%d chars)", label, date_str, len(json.dumps(data, ensure_ascii=False)))
            return True
        except Exception as e:
            logger.error("[FLOWX] %s 업로드 실패: %s", label, e)
            return False

    def upload_market_brain(self, date_str: str) -> bool:
        """brain_decision.json → quant_market_brain."""
        p = DATA_DIR / "brain_decision.json"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            return self._upsert_date_data("quant_market_brain", json.load(f), date_str, "시장브레인")

    def upload_sector_flow(self, date_str: str) -> bool:
        """sector_institutional_flow.json → quant_sector_flow."""
        p = DATA_DIR / "institutional_flow" / "sector_institutional_flow.json"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            return self._upsert_date_data("quant_sector_flow", json.load(f), date_str, "업종수급")

    def upload_sector_momentum(self, date_str: str) -> bool:
        """sector_composite.json → quant_sector_momentum."""
        p = DATA_DIR / "sector_rotation" / "sector_composite.json"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            return self._upsert_date_data("quant_sector_momentum", json.load(f), date_str, "업종모멘텀")

    def upload_sector_rotation(self, date_str: str) -> bool:
        """etf_trading_signal.json → sector_rotation (다중행)."""
        p = DATA_DIR / "sector_rotation" / "etf_trading_signal.json"
        if not self.is_active or not p.exists():
            return False
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            signals = raw.get("smart_money_etf", [])
            if not signals:
                return False
            rows = []
            for i, s in enumerate(signals):
                rows.append({
                    "date": date_str,
                    "rank": i + 1,
                    "sector": s.get("sector", s.get("etf_sector", "")),
                    "etf_code": s.get("etf_code", ""),
                    "signal": s.get("signal", ""),
                    "data": s,
                })
            # 기존 날짜 데이터 삭제 후 삽입
            self.client.table("sector_rotation").delete().eq("date", date_str).execute()
            result = self.client.table("sector_rotation").insert(rows).execute()
            if not result.data:
                logger.warning("[FLOWX] 섹터로테이션 업로드 응답 비어있음: %s", date_str)
                return False
            logger.info("[FLOWX] 섹터로테이션 업로드: %s (%d행)", date_str, len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 섹터로테이션 업로드 실패: %s", e)
            return False

    def upload_etf_fund_flow(self, date_str: str) -> bool:
        """sector_momentum.json → quant_etf_fund_flow."""
        p = DATA_DIR / "sector_rotation" / "sector_momentum.json"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            return self._upsert_date_data("quant_etf_fund_flow", json.load(f), date_str, "ETF자금흐름")

    def upload_etf_recommendation(self, date_str: str) -> bool:
        """etf_recommendations.json → quant_etf_recommendation."""
        p = DATA_DIR / "etf_recommendations.json"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            return self._upsert_date_data("quant_etf_recommendation", json.load(f), date_str, "ETF추천")

    def upload_all_quant_tables(self, date_str: str) -> dict[str, bool]:
        """6개 퀀트 테이블 일괄 업로드."""
        results = {
            "market_brain": self.upload_market_brain(date_str),
            "sector_flow": self.upload_sector_flow(date_str),
            "sector_momentum": self.upload_sector_momentum(date_str),
            "sector_rotation": self.upload_sector_rotation(date_str),
            "etf_fund_flow": self.upload_etf_fund_flow(date_str),
            "etf_recommendation": self.upload_etf_recommendation(date_str),
        }
        ok = sum(v for v in results.values())
        logger.info("[FLOWX] 퀀트 6테이블 업로드: %d/6 성공 %s", ok, results)
        return results

    # ── 페이퍼 트레이딩 ──────────────────────────────

    def upload_paper_trade(self, trade: dict) -> bool:
        """페이퍼 매매 기록 업로드 (INSERT, UPSERT 아님)."""
        if not self.is_active or not trade:
            return False
        try:
            result = self.client.table("paper_trades").insert(trade).execute()
            if not result.data:
                logger.warning("[FLOWX] 매매기록 업로드 응답 비어있음")
                return False
            side = trade.get("side", "?")
            name = trade.get("name", "?")
            price = trade.get("price", 0)
            logger.info("[FLOWX] 매매기록: %s %s @ %s", side, name, f"{price:,}")
            return True
        except Exception as e:
            logger.error("[FLOWX] 매매기록 업로드 실패: %s", e)
            return False


# ── 데이터 변환 함수 ─────────────────────────────

def build_etf_signal_rows(date_str: str = "") -> list[dict]:
    """기존 JSON → etf_signals 테이블 포맷 변환.

    소스:
      - data/sector_rotation/sector_momentum.json (섹터 순위 + RSI)
      - data/etf_rotation_result.json (order_queue → BUY/SELL)
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    rows = []

    # 1) 섹터 모멘텀 → 20개 섹터 ETF
    sm_path = DATA_DIR / "sector_rotation" / "sector_momentum.json"
    if sm_path.exists():
        with open(sm_path, encoding="utf-8") as f:
            sm = json.load(f)

        for sec in sm.get("sectors", []):
            rows.append({
                "date": date_str,
                "code": sec.get("etf_code", ""),
                "name": sec.get("sector", ""),
                "signal": "HOLD",  # 기본값, 아래에서 order_queue로 오버라이드
                "score": round(sec.get("momentum_score", 0), 1),
                "change_1d": round(sec.get("ret_1", sec.get("ret_5", 0) / 5 if sec.get("ret_5") else 0), 2),
                "change_5d": round(sec.get("ret_5", 0), 2),
                "rsi": round(sec.get("rsi_14", 0), 1),
                "sector_rotation_rank": sec.get("rank", 0),
                "group_tag": None,
            })

    # 2) order_queue에서 BUY/SELL 오버라이드
    etf_path = DATA_DIR / "etf_rotation_result.json"
    if etf_path.exists():
        with open(etf_path, encoding="utf-8") as f:
            etf = json.load(f)

        code_signal_map = {}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            action = order.get("action", "").upper()
            if code and action in ("BUY", "SELL"):
                code_signal_map[code] = action

        for row in rows:
            if row["code"] in code_signal_map:
                row["signal"] = code_signal_map[row["code"]]

        # order_queue에 있지만 sector_momentum에 없는 ETF (인버스, 레버리지 등)
        existing_codes = {r["code"] for r in rows}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            if code and code not in existing_codes:
                action = order.get("action", "").upper()
                if action in ("BUY", "SELL"):
                    rows.append({
                        "date": date_str,
                        "code": code,
                        "name": order.get("name", ""),
                        "signal": action,
                        "score": 0,
                        "change_1d": 0,
                        "change_5d": 0,
                        "rsi": 0,
                        "sector_rotation_rank": 0,
                        "group_tag": None,
                    })

    return rows


def build_foreign_flow_rows(date_str: str = "") -> list[dict]:
    """국적별 외국인 수급 → foreign_flow 테이블 포맷 변환.

    소스: data/krx_nationality/nationality_signal.json
    시그널 매핑:
      - foreign_direction=="BUY" → INFLOW
      - signal=="SELL" or foreign_direction=="SELL" → OUTFLOW
      - 나머지 → NEUTRAL
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    sig_path = DATA_DIR / "krx_nationality" / "nationality_signal.json"
    if not sig_path.exists():
        return []

    with open(sig_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for sig in data.get("signals", []):
        # 시그널 매핑: 외국인 순매수/순매도 방향 기준
        foreign_dir = sig.get("foreign_direction", "NEUTRAL")
        raw_signal = sig.get("signal", "NEUTRAL")

        if foreign_dir == "BUY":
            mapped = "INFLOW"
        elif foreign_dir == "SELL" or raw_signal == "SELL":
            mapped = "OUTFLOW"
        else:
            mapped = "NEUTRAL"

        rows.append({
            "date": date_str,
            "code": sig.get("ticker", ""),
            "name": sig.get("name", ""),
            "signal": mapped,
            "score": sig.get("score", 0),
            "z_score": round(sig.get("inst_zscore", 0), 2),
            "change_5d_pct": round(sig.get("price_change_pct", 0), 2),
        })

    return rows


def build_ai_pick_rows(date_str: str = "") -> list[dict]:
    """AI 추천 종목 → short_signals 테이블 포맷 변환.

    소스: data/tomorrow_picks.json
      - ai_largecap: AI 두뇌 대형주 추천 (confidence 기반)
      - picks: 전략 종합 추천 (적극매수/매수 등급만)

    short_signals 스키마:
        date, code, name, grade, total_score, foreign_detail,
        inst_support, entry_price, stop_loss, target_price,
        holding_days, signal_type, volume_ratio, momentum_regime
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # FLOWX 모드 JSON 우선 로드 (없으면 기본 tomorrow_picks.json 폴백)
    flowx_path = DATA_DIR / "tomorrow_picks_flowx.json"
    picks_path = flowx_path if flowx_path.exists() else DATA_DIR / "tomorrow_picks.json"
    if not picks_path.exists():
        return []
    is_flowx = picks_path == flowx_path

    with open(picks_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    seen_codes: set[str] = set()

    # v6: AI 대형주 독립 섹션 제거 — 시그널 기반 picks만 사용
    # (ai_largecap은 이제 항상 빈 리스트)

    # picks → 적극매수/매수 (FLOWX: 관심매수도 포함)
    allowed_grades = {"적극매수", "매수", "관심매수"} if is_flowx else {"적극매수", "매수"}
    for pick in data.get("picks", []):
        ticker = pick.get("ticker", "")
        grade_kr = pick.get("grade", "")
        if not ticker or ticker in seen_codes:
            continue
        if grade_kr not in allowed_grades:
            continue
        seen_codes.add(ticker)

        grade_map = {"적극매수": "AA", "매수": "A", "관심매수": "B"}
        grade = grade_map.get(grade_kr, "B")
        close = pick.get("close", 0) or _get_close(ticker)
        if close <= 0:
            continue  # 종가 없으면 스킵 (stop_loss=0 방지)

        # 시나리오 필드 (scan_tomorrow_picks.py에서 생성)
        sc_tag = pick.get("scenario_tag", "")
        sc_narrative = pick.get("scenario_narrative", "")
        sc_rr = pick.get("scenario_risk_reward", {})

        row = {
            "date": date_str,
            "code": ticker,
            "name": pick.get("name", ""),
            "grade": grade,
            "total_score": round(pick.get("total_score", 0), 1),
            "foreign_detail": None,
            "inst_support": bool(pick.get("inst_5d", 0) > 0),
            "entry_price": pick.get("entry_price", close),
            "stop_loss": pick.get("stop_loss") or int(close * 0.92),
            "target_price": pick.get("target_price", int(close * 1.1)),
            "holding_days": 5,
            "signal_type": "BUY",
            "volume_ratio": round(pick.get("volume_ratio", 1.0), 1),
            "momentum_regime": "QUANT",
        }
        # 시나리오 데이터가 있으면 추가 (Supabase 컬럼 있을 때만 유효)
        if sc_tag:
            row["scenario_tag"] = sc_tag
            row["narrative"] = sc_narrative[:200] if sc_narrative else ""
            rr_parts = []
            if sc_rr.get("commodity"):
                rr_parts.append(f"{sc_rr['commodity']}")
            if sc_rr.get("cost_gap_pct") is not None:
                rr_parts.append(f"갭{sc_rr['cost_gap_pct']:.0f}%")
            if sc_rr.get("zone"):
                rr_parts.append(sc_rr["zone"])
            row["risk_reward"] = " ".join(rr_parts) if rr_parts else ""
        rows.append(row)

    return rows


def _get_vix_grade(vix: float) -> str:
    """VIX 값 → 등급 (brain.py VIX_BUCKETS 기준)."""
    if vix < 15:
        return "LOW"
    if vix < 20:
        return "NORMAL"
    if vix < 25:
        return "ELEVATED"
    if vix < 30:
        return "HIGH"
    if vix < 40:
        return "EXTREME"
    return "PANIC"


def _get_danger_mode(regime: str, vix: float, shield_status: str) -> str:
    """레짐 + VIX + SHIELD 종합 → 위험 모드 결정.

    Returns: NORMAL, WARNING, DANGER, PANIC
    """
    vix_grade = _get_vix_grade(vix)

    # PANIC: VIX >= 40 OR CRISIS 레짐
    if vix_grade == "PANIC" or regime == "CRISIS":
        return "PANIC"

    # DANGER: VIX EXTREME(30~40) + BEAR/CRISIS, 또는 SHIELD RED + BEAR
    if vix_grade == "EXTREME" and regime in ("BEAR", "CRISIS"):
        return "DANGER"
    if shield_status == "RED" and regime == "BEAR":
        return "DANGER"

    # WARNING: BEAR 레짐 or VIX HIGH/EXTREME or SHIELD RED
    if regime == "BEAR" or vix_grade in ("HIGH", "EXTREME") or shield_status == "RED":
        return "WARNING"

    return "NORMAL"


def _check_data_stale(brain: dict) -> dict | None:
    """brain_decision.json 타임스탬프 → stale 여부 (3일 초과 시 경고)."""
    from datetime import datetime, timedelta
    ts = brain.get("timestamp", "")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        age = datetime.now() - dt
        if age > timedelta(days=3):
            return {
                "stale": True,
                "age_days": age.days,
                "last_update": ts[:10],
                "message": f"데이터가 {age.days}일 전 기준입니다. 최신 장 데이터가 반영되지 않았을 수 있습니다.",
            }
    except (ValueError, TypeError):
        pass
    return None


def _build_market_guide(brain: dict, shield: dict, sector_momentum: dict) -> dict:
    """시장 맥락 가이드 자동 생성 — 구독자에게 '지금 왜 이걸 사야 하는지' 안내."""
    regime = brain.get("regime") or brain.get("effective_regime") or brain.get("direction") or "NEUTRAL"
    shield_status = shield.get("status", "YELLOW")
    vix = brain.get("vix") or brain.get("vix_level") or 0
    cash_ratio = brain.get("cash_ratio") or brain.get("cash_pct") or 0

    REGIME_KR = {
        "BULL": "상승장", "CAUTION": "주의", "BEAR": "하락장",
        "CRISIS": "위기", "NEUTRAL": "보합",
        "PRE_BULL": "상승 전환 중", "PRE_BEAR": "하락 전환 중",
    }
    SHIELD_KR = {"GREEN": "안전", "YELLOW": "주의", "RED": "위험"}

    STRATEGY_MAP = {
        "BULL": "적극 매수 구간 — 상승 흐름을 타는 종목 위주로 진입",
        "CAUTION": "선별 매수 — 여러 신호가 겹치는 종목만 골라서 진입",
        "BEAR": "신중한 매수 — 2개 이상 신호 + 거래량 급증 종목만 진입",
        "CRISIS": "매수 자제 — 현금 비중 높이고 방어 자산(금/채권) 위주",
    }

    # 위험 모드 결정
    danger_mode = _get_danger_mode(regime, vix, shield_status)
    vix_grade = _get_vix_grade(vix)

    regime_kr = REGIME_KR.get(regime, regime)
    shield_kr = SHIELD_KR.get(shield_status, shield_status)

    sectors = sector_momentum.get("sectors", [])
    hot = [{"sector": s["sector"], "ret_5": s.get("ret_5", 0)} for s in sectors[:3]]
    cold = [{"sector": s["sector"], "ret_5": s.get("ret_5", 0)} for s in sectors[-3:]] if len(sectors) >= 6 else []

    # 한글 요약 (위험 모드 우선)
    if danger_mode == "PANIC":
        summary = "패닉 — 신규 매수 자제! 금/채권 방어 + 현금 55% 이상 유지"
        strategy = "전면 방어 — 개별주 진입 금지, ETF(금/채권/인버스)로 자산 보호"
    elif danger_mode == "DANGER":
        summary = f"{regime_kr} + 공포지수 경고 — 방어 자산 위주, 개별주 최소화"
        strategy = "방어 우선 — ETF 배분 유지, 개별주는 수급 통과 종목만 소량"
    elif regime == "CRISIS" or shield_status == "RED":
        summary = f"{regime_kr} + 위험 방어 {shield_kr} — 매수 자제, 현금 위주 운영"
        strategy = STRATEGY_MAP.get("CRISIS", "")
    elif regime == "BEAR":
        summary = f"{regime_kr} — 확실한 종목만 소량 진입"
        strategy = STRATEGY_MAP.get("BEAR", "")
    elif regime == "CAUTION":
        summary = f"{regime_kr} — 선별 매수, 손절 라인 꼭 지키기"
        strategy = STRATEGY_MAP.get("CAUTION", "")
    else:
        summary = f"{regime_kr} — 적극 매수 구간"
        strategy = STRATEGY_MAP.get("BULL", "")

    # stale 데이터 경고
    stale_info = _check_data_stale(brain)

    return {
        "summary": summary,
        "strategy": strategy,
        "hot_sectors": hot,
        "cold_sectors": cold,
        "vix": vix,
        "vix_grade": vix_grade,
        "cash_ratio": cash_ratio,
        "danger_mode": danger_mode,
        "stale": stale_info,
    }


def _build_sectors_data(sector_momentum: dict) -> dict:
    """섹터 모멘텀 TOP 10 추출."""
    sectors = sector_momentum.get("sectors", [])
    top = []
    for s in sectors[:10]:
        top.append({
            "rank": s.get("rank"),
            "sector": s.get("sector"),
            "etf_code": s.get("etf_code"),
            "score": s.get("momentum_score", 0),
            "ret_5": s.get("ret_5", 0),
            "ret_20": s.get("ret_20", 0),
            "rsi": s.get("rsi_14", 0),
            "acceleration": s.get("acceleration", False),
            "rank_change": s.get("rank_change", 0),
        })
    return {
        "date": sector_momentum.get("date", ""),
        "top": top,
    }


def _build_etf_picks(etf_result: dict) -> dict:
    """ETF 배분 + 가속 섹터 추출."""
    allocation = etf_result.get("allocation", {})
    predator = etf_result.get("predator_result", {})
    accelerations = predator.get("accelerations", [])[:5]

    return {
        "regime": etf_result.get("regime", "NEUTRAL"),
        "allocation": allocation,
        "accelerations": [
            {
                "sector": a.get("sector"),
                "rank_change": a.get("rank_change", 0),
                "score": a.get("acceleration_score", 0),
                "ret_5d": a.get("ret_5d", 0),
            }
            for a in accelerations
        ],
    }


def _build_performance_data() -> dict:
    """최근 5일 시그널 적중률 + 시장 요약."""
    import glob
    learn_dir = DATA_DIR / "market_learning"
    files = sorted(glob.glob(str(learn_dir / "????-??-??.json")))[-5:]

    daily = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            acc = d.get("signal_accuracy", {})
            summary = d.get("summary", {})
            # 적중률 평균 (total > 0인 소스만)
            rates = [v["hit_rate"] for v in acc.values() if isinstance(v, dict) and v.get("total", 0) > 0]
            avg_hit = round(sum(rates) / len(rates), 1) if rates else 0
            daily.append({
                "date": d.get("date", ""),
                "avg_hit_rate": avg_hit,
                "market_avg_ret": summary.get("avg_ret", 0),
                "up_ratio": round(summary.get("up_count", 0) / max(summary.get("total", 1), 1) * 100, 1),
                "sources": {
                    k: {"hit_rate": v.get("hit_rate", 0), "total": v.get("total", 0), "avg_ret": v.get("avg_ret", 0)}
                    for k, v in acc.items()
                    if isinstance(v, dict) and v.get("total", 0) > 0
                },
            })
        except Exception:
            continue

    # 최신일 상세
    latest = daily[-1] if daily else {}

    return {
        "daily_trend": daily,
        "latest": latest,
    }


def _build_signals_data(all_picks: list, accuracy: dict) -> dict:
    """매매 신호 분석 데이터 구축."""
    from collections import Counter

    # 소스별 감지 수
    source_counts: dict[str, int] = Counter()
    for p in all_picks:
        for src in p.get("sources", []):
            source_counts[src] += 1

    # 교차검증 분포
    ns_dist = Counter(p.get("n_sources", 0) for p in all_picks)
    single = sum(v for k, v in ns_dist.items() if k == 1)
    double = sum(v for k, v in ns_dist.items() if k == 2)
    triple_plus = sum(v for k, v in ns_dist.items() if k >= 3)

    # 소스 조합 TOP 5 (2소스 이상)
    combo_counts: dict[str, int] = Counter()
    for p in all_picks:
        srcs = sorted(p.get("sources", []))
        if len(srcs) >= 2:
            combo_counts["+".join(srcs)] += 1
    top_combos = [
        {"combo": k, "count": v}
        for k, v in combo_counts.most_common(5)
    ]

    # 소스별 활성도 + 정확도 합산
    sources_detail = []
    for src, cnt in source_counts.most_common(12):
        acc = accuracy.get(src, {})
        hit_rate = acc.get("hit_rate", 0) if isinstance(acc, dict) else 0
        total = acc.get("total", 0) if isinstance(acc, dict) else 0
        sources_detail.append({
            "source": src,
            "count": cnt,
            "hit_rate": hit_rate,
            "total_tested": total,
        })

    return {
        "total": len(all_picks),
        "sources": sources_detail,
        "cross_validation": {
            "single": single,
            "double": double,
            "triple_plus": triple_plus,
        },
        "top_combos": top_combos,
    }


def build_jarvis_payload() -> dict:
    """자비스 컨트롤타워 데이터 통합 빌드.

    data/tomorrow_picks.json + brain_decision.json + shield_report.json +
    market_learning/signal_accuracy.json + sector_momentum.json + etf_rotation_result.json 통합.
    """
    def _load(name: str) -> dict:
        p = DATA_DIR / name
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    picks_raw = _load("tomorrow_picks.json")
    brain = _load("brain_decision.json")
    shield = _load("shield_report.json")
    sector_momentum = _load("sector_rotation/sector_momentum.json")
    etf_result = _load("etf_rotation_result.json")

    # signal_accuracy: 독립 파일 우선, 없으면 _index 경유
    acc_file = _load("market_learning/signal_accuracy.json")
    acc_raw = acc_file.get("signals", {})
    if not acc_raw:
        learn_idx = _load("market_learning/_index.json")
        latest_learn = learn_idx.get("latest", "")
        if latest_learn:
            learn = _load(f"market_learning/{latest_learn}.json")
            acc_raw = learn.get("signal_accuracy", {})

    # 위험 모드 결정
    regime = brain.get("regime") or brain.get("effective_regime") or brain.get("direction") or "NEUTRAL"
    vix = brain.get("vix") or brain.get("vix_level") or 0
    shield_status = shield.get("status", "YELLOW")
    danger_mode = _get_danger_mode(regime, vix, shield_status)

    # picks 요약 (전체 picks는 너무 크므로 관찰 이상만)
    all_picks = picks_raw.get("picks", [])
    buyable_grades = {"적극매수", "매수", "관심매수", "관찰"}
    buyable = [p for p in all_picks if p.get("grade") in buyable_grades]
    buyable.sort(key=lambda x: -x.get("total_score", 0))

    # PANIC/DANGER → 개별주 최대 3개 + "관찰" 이상만
    if danger_mode == "PANIC":
        buyable = [p for p in buyable if p.get("n_sources", 0) >= 3][:3]
    elif danger_mode == "DANGER":
        buyable = [p for p in buyable if p.get("n_sources", 0) >= 2][:5]

    # 상위 20개만 전송 (Supabase JSONB 크기 제한)
    top_picks = buyable[:20]

    # ── killer_picks 병합: 기관수급 조기감지 종목을 메인 추천에 통합 ──
    killer_raw = _load("killer_picks.json")
    if killer_raw:
        existing_tickers = {p.get("ticker") for p in top_picks}
        killer_items = _convert_killer_to_picks(killer_raw)
        for ki in killer_items:
            if ki["ticker"] not in existing_tickers:
                top_picks.append(ki)
                existing_tickers.add(ki["ticker"])
        # 점수순 재정렬
        top_picks.sort(key=lambda x: -x.get("total_score", 0))
        top_picks = top_picks[:25]  # 병합 후 최대 25개

    # why-now-engine: AI 판단을 pick에 병합
    ai_judgments = _load("ai_brain_judgment.json")
    ai_map: dict[str, dict] = {}
    for j in ai_judgments.get("stock_judgments", []):
        t = j.get("ticker", "")
        if t:
            ai_map[t] = j

    for pick in top_picks:
        if "why_now" not in pick:
            pick["why_now"] = _build_why_now(pick, ai_map.get(pick.get("ticker", ""), {}))

    return {
        "picks": {
            "target_date_label": picks_raw.get("target_date_label", ""),
            "mode_label": picks_raw.get("mode_label", ""),
            "total_candidates": picks_raw.get("total_candidates", 0),
            "stats": picks_raw.get("stats", {}),
            "picks": top_picks,
        },
        "accuracy": acc_raw,
        "brain": {
            "regime": regime,
            "vix": vix,
            "vix_grade": _get_vix_grade(vix),
            "cash_ratio": brain.get("cash_ratio") or brain.get("cash_pct") or 0,
            "recommendation": brain.get("recommendation", ""),
            "danger_mode": danger_mode,
        },
        "shield": {
            "status": shield_status,
            "sector_concentration": shield.get("sector_concentration", 0),
            "max_drawdown": shield.get("max_drawdown", 0),
        },
        # Phase 1 신규: 시장 맥락 + 섹터 + ETF
        "market_guide": _build_market_guide(brain, shield, sector_momentum),
        "sectors": _build_sectors_data(sector_momentum),
        "etf_picks": _build_etf_picks(etf_result),
        # Phase 3: 매매 신호 분석
        "signals": _build_signals_data(all_picks, acc_raw),
        # Phase 4: 성과
        "performance": _build_performance_data(),
        # Phase 5: CFO/CTO + 펀더멘탈
        "cfo": _build_cfo_data(),
        "cto": _build_cto_data(),
        "fundamentals": _build_fundamentals_data(),
        # Phase 6: 매크로 레짐
        "macro": _build_macro_data(),
        # Phase 7: 킬러픽 자비스 보고서
        "killer_picks": _build_killer_picks_data(),
    }


def _build_why_now(pick: dict, ai_judgment: dict) -> dict:
    """왜 지금 이 종목인가 — 5가지 소스를 카테고리별 통합."""
    why: dict = {"technical": [], "macro": None, "entry": None, "safety": None, "warnings": []}

    # 1) 기술적 근거 (reasons 중 ⚠ 제외)
    for r in pick.get("reasons", []):
        if isinstance(r, str):
            if r.startswith("\u26a0"):
                why["warnings"].append(r.lstrip("\u26a0 "))
            else:
                why["technical"].append(r)

    # 2) 거시/AI 분석 (ai_brain_judgment)
    reasoning = ai_judgment.get("reasoning", "")
    if reasoning:
        catalysts = ai_judgment.get("catalysts", [])
        confidence = ai_judgment.get("confidence", 0)
        why["macro"] = {
            "reasoning": reasoning,
            "catalysts": catalysts[:3] if isinstance(catalysts, list) else [],
            "confidence": round(confidence, 2) if isinstance(confidence, (int, float)) else 0,
        }
    # ai_tag 보조 (AI 판단은 없지만 태그만 있는 경우)
    elif pick.get("ai_tag"):
        why["macro"] = {
            "reasoning": pick["ai_tag"],
            "catalysts": [],
            "confidence": 0,
        }

    # 3) 진입 조건
    entry_cond = pick.get("entry_condition", "")
    if entry_cond:
        why["entry"] = entry_cond

    # 4) 안전성 평가
    safety = pick.get("safety_reason", "")
    if safety:
        why["safety"] = safety

    # 5) 과열 경고 보강
    for flag in pick.get("overheat_flags", []):
        if isinstance(flag, str) and flag not in why["warnings"]:
            why["warnings"].append(flag)

    # 보너스 태그 (시나리오/컨센서스/수급)
    tags = []
    for key, label in [("scenario_tag", "시나리오"), ("consensus_tag", "컨센서스"), ("nat_tag", "수급")]:
        tag = pick.get(key, "")
        bonus = pick.get(key.replace("_tag", "_bonus"), 0)
        if tag:
            tags.append({"label": label, "tag": tag, "bonus": round(bonus, 1) if bonus else 0})
    if tags:
        why["bonus_tags"] = tags

    return why


def _get_close(ticker: str) -> int:
    """parquet에서 최신 종가 조회."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            if len(df) > 0:
                return int(df.iloc[-1]["close"])
        except Exception:
            pass
    return 0


def _build_cfo_data() -> dict:
    """CFO 포트폴리오 건강 리포트 → FLOWX payload."""
    p = DATA_DIR / "cfo_report.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            cfo = json.load(f)
    except Exception:
        return {}

    health = cfo.get("health", {})
    dd = cfo.get("drawdown", {})
    budget = cfo.get("allocation_budget", {})

    return {
        "generated_at": cfo.get("generated_at", ""),
        "health_score": health.get("overall_score", 0),
        "risk_level": health.get("risk_level", ""),
        "positions_count": health.get("positions_count", 0),
        "cash_ratio": round(health.get("cash_ratio", 0) * 100, 1),
        "max_sector_name": health.get("max_sector_name", ""),
        "max_sector_pct": health.get("max_sector_pct", 0),
        "var_95": round(health.get("estimated_var_95", 0) * 100, 1),
        "warnings": health.get("warnings", []),
        "recommendations": health.get("recommendations", []),
        "drawdown_action": dd.get("action_label", ""),
        "drawdown_pct": dd.get("current_drawdown_pct", 0),
        "investable": budget.get("investable", 0),
        "max_new_invest": budget.get("max_new_invest", 0),
        "regime": budget.get("regime", ""),
    }


def _build_cto_data() -> dict:
    """CTO 시스템 성과 리포트 → FLOWX payload."""
    p = DATA_DIR / "cto_report.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            cto = json.load(f)
    except Exception:
        return {}

    perf = cto.get("signal_performance", {})
    data_health = cto.get("data_health", {})
    suggestions = cto.get("weight_suggestions", [])

    # 소스별 승률 요약 (상위 10)
    sources_raw = perf.get("sources", {})
    source_perf = sorted(
        sources_raw.items(),
        key=lambda x: x[1].get("total_picks", 0),
        reverse=True,
    )[:10]

    return {
        "generated_at": cto.get("generated_at", ""),
        "total_records": perf.get("total_records", 0),
        "source_performance": [
            {
                "source": src,
                "win_rate": round(d.get("win_rate", 0) * 100, 1),
                "avg_return": round(d.get("avg_return", 0), 2),
                "total": d.get("total_picks", 0),
                "decay": d.get("signal_decay", False),
            }
            for src, d in source_perf
        ],
        "decay_alerts": perf.get("decay_alerts", []),
        "data_health_score": data_health.get("health_score", 0),
        "stale_count": data_health.get("stale", 0),
        "missing_count": data_health.get("missing", 0),
        "suggestions": [
            {"action": s.get("action", ""), "detail": s.get("detail", ""), "priority": s.get("priority", "")}
            for s in suggestions[:8]
        ],
    }


def _build_fundamentals_data() -> dict:
    """실적 가속도 + 턴어라운드 → FLOWX payload."""
    result = {}

    # 실적 가속도
    ea_path = DATA_DIR / "earnings_acceleration.json"
    if ea_path.exists():
        try:
            with open(ea_path, encoding="utf-8") as f:
                ea = json.load(f)
            result["earnings"] = {
                "date": ea.get("date", ""),
                "total_analyzed": ea.get("total_analyzed", 0),
                "status_counts": ea.get("status_counts", {}),
                "turnaround_strong": ea.get("turnaround_strong", [])[:10],
                "turnaround_early": ea.get("turnaround_early", [])[:10],
                "accelerating": ea.get("accelerating", [])[:10],
            }
        except Exception:
            pass

    # 턴어라운드
    ta_path = DATA_DIR / "turnaround_candidates.json"
    if ta_path.exists():
        try:
            with open(ta_path, encoding="utf-8") as f:
                ta = json.load(f)
            result["turnaround"] = {
                "date": ta.get("date", ""),
                "total_screened": ta.get("total_screened", 0),
                "candidates_found": ta.get("candidates_found", 0),
                "strong": ta.get("strong", [])[:15],
                "early": ta.get("early", [])[:10],
            }
        except Exception:
            pass

    return result


def _convert_killer_to_picks(killer_raw: dict) -> list[dict]:
    """killer_picks.json → PickItem 형식으로 변환하여 메인 추천에 병합."""
    GRADE_MAP = {
        "STRONG": ("적극매수", 70),
        "MODERATE": ("매수", 60),
        "NOTABLE": ("관심매수", 50),
        "EARLY_DUAL": ("관심매수", 55),
        "EARLY_ACCEL": ("관심매수", 52),
        "EARLY_SURGE": ("관심매수", 50),
        "WATCH": ("관찰", 40),
    }
    items: list[dict] = []
    seen: set[str] = set()

    def _add(ticker: str, name: str, grade_key: str, sources: list[str],
             reasons: list[str], extra: dict | None = None):
        if not ticker or ticker in seen:
            return
        seen.add(ticker)
        grade_label, base_score = GRADE_MAP.get(grade_key, ("관찰", 40))
        item = {
            "ticker": ticker,
            "name": name,
            "grade": grade_label,
            "total_score": base_score,
            "sources": sources,
            "n_sources": len(sources),
            "close": 0,
            "rsi": 0,
            "stoch_k": 0,
            "foreign_5d": 0,
            "inst_5d": 0,
            "reasons": reasons,
        }
        if extra:
            item.update(extra)
        items.append(item)

    # 1) cross_validated_top5: 교차검증 통과 (최고 신뢰)
    for cv in killer_raw.get("cross_validated_top5", []):
        conv = cv.get("conviction", "MEDIUM")
        grade = "STRONG" if conv == "HIGH" else "MODERATE"
        score = 75 if conv == "HIGH" else 65
        _add(
            cv.get("ticker", ""), cv.get("name", ""), grade,
            sources=cv.get("matched_from", []),
            reasons=cv.get("matched_from", []),
            extra={
                "total_score": score,
                "entry_info": {
                    "entry": cv.get("entry_price", 0),
                    "stop": cv.get("stop_loss", 0),
                    "target": cv.get("target_price", 0),
                },
                "close": cv.get("entry_price", 0),
            },
        )

    # 2) early_detection.strong: 기관+외인 강매집
    ed = killer_raw.get("early_detection", {})
    for s in ed.get("strong", []):
        inst_5d = s.get("inst_5d", 0)
        frgn_5d = s.get("frgn_5d", 0)
        reasons = []
        if s.get("dual_today"):
            reasons.append("기관+외인 동시매수")
        reasons.append("기관 %d일 연속" % s.get("inst_consec", 0))
        if s.get("chg_pct", 0) != 0:
            reasons.append("등락 %+.1f%%" % s["chg_pct"])
        _add(
            s.get("ticker", ""), s.get("name", ""), "STRONG",
            sources=["기관수급", "조기감지"],
            reasons=reasons,
            extra={"inst_5d": inst_5d, "foreign_5d": frgn_5d, "close": s.get("price", 0)},
        )

    # 3) early_detection.early: 조기 감지 (1~2일차)
    for e in ed.get("early", []):
        inst_5d = e.get("inst_5d", 0)
        frgn_5d = e.get("frgn_5d", 0)
        grade = e.get("grade", "EARLY_DUAL")
        reasons = ["조기감지: %s" % grade]
        if e.get("inst_consec", 0):
            reasons.append("기관 %d일" % e["inst_consec"])
        _add(
            e.get("ticker", ""), e.get("name", ""), grade,
            sources=["기관수급", "조기감지"],
            reasons=reasons,
            extra={"inst_5d": inst_5d, "foreign_5d": frgn_5d, "close": e.get("price", 0)},
        )

    # 4) institutional_picks: 기관매집 상위
    for ip in killer_raw.get("institutional_picks", []):
        _add(
            ip.get("ticker", ""), ip.get("name", ""), ip.get("grade", "NOTABLE"),
            sources=["기관수급"],
            reasons=[ip.get("verdict", "")],
            extra={
                "total_score": 45 + min(ip.get("inst_consecutive", 0) * 3, 20),
                "inst_5d": int(ip.get("inst_5d_bil", 0) * 1e8),
                "foreign_5d": int(ip.get("foreign_5d_bil", 0) * 1e8),
            },
        )

    return items


def _build_killer_picks_data() -> dict:
    """킬러픽 자비스 보고서 → FLOWX payload."""
    p = DATA_DIR / "killer_picks.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_macro_data() -> dict:
    """매크로 시장 흐름 → FLOWX payload.

    data/macro/macro_regime.json 을 읽어
    FLOWX 매크로 탭에 표시할 데이터를 구성한다.
    details 배열 포함 (이전 vs 현재 비교 문장).
    """
    regime_path = DATA_DIR / "macro" / "macro_regime.json"
    result = {}

    if regime_path.exists():
        try:
            with open(regime_path, encoding="utf-8") as f:
                data = json.load(f)

            # 각 섹션에서 필요한 필드만 추출 (series 제외, details 포함)
            rate = data.get("rate", {})
            inf = data.get("inflation", {})
            fx = data.get("fx", {})
            erp = data.get("erp", {})

            result = {
                "date": data.get("date", ""),
                "overall": data.get("overall", {}),
                "rate": {
                    "base_rate": rate.get("base_rate"),
                    "bond_3y": rate.get("bond_3y"),
                    "bond_10y": rate.get("bond_10y"),
                    "spread_10y_3y": rate.get("spread_10y_3y"),
                    "direction": rate.get("direction"),
                    "signal": rate.get("signal"),
                    "details": rate.get("details", []),
                },
                "inflation": {
                    "cpi": inf.get("cpi"),
                    "direction": inf.get("direction"),
                    "signal": inf.get("signal"),
                    "details": inf.get("details", []),
                },
                "fx": {
                    "usd_krw": fx.get("usd_krw"),
                    "direction": fx.get("direction"),
                    "signal": fx.get("signal"),
                    "details": fx.get("details", []),
                },
                "erp": {
                    "erp": erp.get("erp"),
                    "kospi_per": erp.get("kospi_per"),
                    "verdict": erp.get("verdict"),
                    "signal": erp.get("signal"),
                    "details": erp.get("details", []),
                },
                "sectors": data.get("sectors", {}),
                "strategy": data.get("strategy", {}),
                "market_phase": data.get("market_phase", ""),
            }
        except Exception:
            pass

    return result


def build_paper_trade(
    code: str,
    name: str,
    side: str,
    price: float,
    quantity: int,
    strategy: str,
    pnl_pct: float | None = None,
    memo: str = "",
    stats: dict | None = None,
) -> dict:
    """페이퍼 매매 기록 dict 생성.

    Args:
        stats: paper_portfolio.json의 stats dict (PF/MDD/승률).
    """
    s = stats or {}
    total = s.get("total_trades", 0)
    wins = s.get("wins", 0)

    return {
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "code": code,
        "name": name,
        "side": side.upper(),
        "price": price,
        "quantity": quantity,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "strategy": strategy,
        "cumulative_pf": round(s.get("pf", 0), 2) if s.get("pf") else None,
        "cumulative_mdd": round(s.get("mdd", 0), 2) if s.get("mdd") else None,
        "win_rate": round(wins / total * 100, 1) if total > 0 else None,
        "memo": memo,
    }
