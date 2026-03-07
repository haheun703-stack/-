"""ETF 매매 시그널 텔레그램 발송 — BAT-C에서 호출

전일 생성된 etf_trading_signal.json을 읽어 텔레그램 발송
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = Path(__file__).resolve().parent.parent / "data" / "sector_rotation" / "etf_trading_signal.json"
    if not p.exists():
        print("[SKIP] etf_trading_signal.json 없음")
        return

    signals = json.loads(p.read_text(encoding="utf-8"))

    from scripts.etf_trading_signal import build_telegram_message
    from src.telegram_sender import send_message

    msg = build_telegram_message(signals)
    ok = send_message(msg)
    print(f"[OK] ETF 시그널 텔레그램 발송 {'성공' if ok else '실패'}")


if __name__ == "__main__":
    main()
