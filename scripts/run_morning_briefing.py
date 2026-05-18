"""장전 마켓 브리핑 — BAT-B에서 호출

1) RSS 테마 스캔 + Grok 확장 (텔레그램 OFF → JSON만)
2) 뉴스 스캔 (텔레그램 OFF → JSON만)
3) 통합 아침 브리핑 1건 텔레그램 발송
   (KOSPI예측 + US + 증권사 + 테마 + ETF → 1건)
4) Supabase quant_bot_advisory INSERT (msg_type='MORNING_BRIEFING')
   동생 단타봇이 09:00 진입 전 SELECT (사장님 5/18 11:30 지시)
"""
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trading_calendar import should_run_bat


def parse_briefing_meta(msg: str) -> dict:
    """텔레그램 메시지 본문에서 핵심 수치 추출."""
    meta = {}
    m = re.search(r"상승\s*(\d+)%", msg)
    if m:
        meta["kospi_up_pct"] = int(m.group(1))
    m = re.search(r"레인지[:\s]*([+\-]?\d+\.?\d*%\s*~\s*[+\-]?\d+\.?\d*%)", msg)
    if m:
        meta["kospi_range"] = m.group(1).strip()
    for sym in ("SPY", "QQQ", "SOXX", "DIA"):
        m = re.search(rf"{sym}\s*([+\-]\d+\.?\d*)%", msg)
        if m:
            meta[f"us_{sym.lower()}_chg"] = float(m.group(1))
    m = re.search(r"VIX\s*(\d+\.?\d*)", msg)
    if m:
        meta["vix"] = float(m.group(1))
    m = re.search(r"EWY\s*([+\-]\d+\.?\d*)%", msg)
    if m:
        meta["ewy_chg"] = float(m.group(1))
    m = re.search(r"Signal\s*:?\s*(\w+)", msg)
    if m:
        meta["signal"] = m.group(1)
    m = re.search(r"➜\s*([^\n]+)", msg)
    if m:
        meta["conclusion"] = m.group(1).strip()
    return meta


def insert_morning_briefing_advisory(msg: str) -> int | None:
    """장전 브리핑을 quant_bot_advisory에 INSERT — 동생 단타봇용."""
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        print("[WARN] psycopg2 미설치 — advisory INSERT 스킵")
        return None

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("[WARN] DATABASE_URL 미설정 — advisory INSERT 스킵")
        return None

    meta = parse_briefing_meta(msg)
    today = datetime.now().strftime("%Y-%m-%d")

    kospi_up = meta.get("kospi_up_pct", 50)
    vix = meta.get("vix", 20)
    if kospi_up >= 70 and vix < 20:
        regime, risk = "MILD_BULL", "LOW"
    elif kospi_up >= 60 and vix < 22:
        regime, risk = "NEUTRAL", "LOW"
    elif kospi_up < 40 or vix > 25:
        regime, risk = "CAUTION", "MED"
    else:
        regime, risk = "NEUTRAL", "MED"

    title = (
        f"[장전 브리핑 07:00] KOSPI 상승 {kospi_up}% / VIX {vix} / "
        f"Signal {meta.get('signal', 'NEUTRAL')} → {meta.get('conclusion', '관망')}"
    )

    try:
        con = psycopg2.connect(url, connect_timeout=10)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO quant_bot_advisory
              (advisory_date, advisory_time, msg_type, severity, target_bot,
               market_regime, risk_level,
               title, body, alert_codes, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                today,
                "07:00",
                "MORNING_BRIEFING",
                "INFO",
                "scalper",
                regime,
                risk,
                title,
                msg,
                ["MORNING-BRIEFING", "BAT-B-AUTO"],
                Json(meta),
            ),
        )
        new_id = cur.fetchone()[0]
        con.commit()
        con.close()
        return new_id
    except Exception as e:
        print(f"[WARN] morning_briefing advisory INSERT 실패: {e}")
        return None


def main():
    # ── 비거래일 스킵 ──
    if not should_run_bat("kr"):
        from datetime import date
        print(f"[SKIP] 한국 비거래일 ({date.today()}) — 브리핑 스킵")
        return
    # 1) 테마 스캔 — 폐지 (2026-05-17): theme_scan_runner는 archive/deprecated.
    #    CLAUDE.md LOCK 규칙에 따라 호출 자체를 제거. 필요 시 모듈 복구 후 재활성화.

    # 2) 뉴스 스캔
    try:
        from main import step_news_scan
        step_news_scan(send_telegram=False)
        print("[OK] 뉴스 스캔 완료")
    except Exception as e:
        print(f"[WARN] 뉴스 스캔 실패: {e}")
        traceback.print_exc()

    # 3) 통합 아침 브리핑 — 1건 텔레그램 + Supabase 적재
    try:
        from src.use_cases.morning_briefing import build_unified_morning
        from src.telegram_sender import send_message
        msg = build_unified_morning()
        ok = send_message(msg)
        print(f"[OK] 통합 브리핑 발송 {'성공' if ok else '실패'} ({len(msg)}자)")

        # 4) 동생 단타봇용 Supabase advisory INSERT (5/18 11:30 사장님 지시)
        advisory_id = insert_morning_briefing_advisory(msg)
        if advisory_id:
            print(f"[OK] morning_briefing advisory id={advisory_id} INSERT 성공")
        else:
            print("[WARN] morning_briefing advisory INSERT 실패 또는 스킵")
    except Exception as e:
        print(f"[WARN] 통합 브리핑 실패: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
