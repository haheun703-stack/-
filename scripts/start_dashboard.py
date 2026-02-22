"""
자비스 컨트롤 타워 — 웹 대시보드 서버 시작

Usage:
    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8080

접속: http://localhost:8000
"""

import argparse
import os
import sys
from pathlib import Path

# Windows cp949 인코딩 문제 방지
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Jarvis Control Tower 웹서버")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="포트 (기본: 8000)")
    args = parser.parse_args()

    import uvicorn

    print(f"[Jarvis] Control Tower -- http://localhost:{args.port}")
    uvicorn.run(
        "src.adapters.web_dashboard_adapter:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
