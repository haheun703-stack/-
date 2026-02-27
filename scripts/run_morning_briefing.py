"""장전 마켓 브리핑 — BAT-B에서 호출

1) RSS 테마 스캔 + Grok 확장 (텔레그램 OFF → JSON만)
2) 뉴스 스캔 (텔레그램 OFF → JSON만)
3) 통합 아침 브리핑 1건 텔레그램 발송
   (KOSPI예측 + US + 증권사 + 테마 + ETF → 1건)
"""
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    # 1) 테마 스캔 — 텔레그램 OFF (통합 메시지에서 읽어감)
    try:
        from scripts.theme_scan_runner import run_theme_scan
        run_theme_scan(use_grok=True, send_telegram=False)
        print("[OK] 테마 스캔 완료 (JSON 저장)")
    except Exception as e:
        print(f"[WARN] 테마 스캔 실패: {e}")
        traceback.print_exc()

    # 2) 뉴스 스캔
    try:
        from main import step_news_scan
        step_news_scan(send_telegram=False)
        print("[OK] 뉴스 스캔 완료")
    except Exception as e:
        print(f"[WARN] 뉴스 스캔 실패: {e}")
        traceback.print_exc()

    # 3) 통합 아침 브리핑 — 1건 텔레그램
    try:
        from scripts.send_market_briefing import build_unified_morning
        from src.telegram_sender import send_message
        msg = build_unified_morning()
        ok = send_message(msg)
        print(f"[OK] 통합 브리핑 발송 {'성공' if ok else '실패'} ({len(msg)}자)")
    except Exception as e:
        print(f"[WARN] 통합 브리핑 실패: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
