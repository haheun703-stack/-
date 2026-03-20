"""파이프라인 실행 결과 빠른 확인

사용: python scripts/check_pipeline_status.py
"""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== 파이프라인 상태 확인 ({today}) ===\n")

    # 1. parallel_pipeline.log 마지막 실행
    plog = ROOT / "logs" / "parallel_pipeline.log"
    if plog.exists():
        lines = plog.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
        # 마지막 "파이프라인 완료" 찾기
        for line in reversed(lines):
            if "파이프라인 완료" in line:
                print(f"[파이프라인] {line.strip()}")
                break
            if "FAIL" in line or "실패" in line:
                print(f"[경고] {line.strip()}")
        # 성공/실패 카운트
        recent = [l for l in lines[-50:] if today in l]
        fails = [l for l in recent if "FAIL" in l or "TIMEOUT" in l]
        if fails:
            print(f"  ❌ 오늘 실패 {len(fails)}건:")
            for f in fails[-5:]:
                print(f"    {f.strip()}")
        elif recent:
            print(f"  ✅ 오늘 로그 {len(recent)}줄, 실패 없음")
        else:
            print(f"  ⚠️ 오늘 실행 기록 없음")
    else:
        print("[파이프라인] 로그 없음 (아직 미실행)")

    print()

    # 2. health check
    hlog = ROOT / "logs" / "health_check.log"
    if hlog.exists():
        content = hlog.read_text(encoding="utf-8", errors="ignore")
        # 마지막 등급 찾기
        blocks = content.split("=" * 50)
        if blocks:
            last = blocks[-1].strip()
            if last:
                for line in last.split("\n")[:3]:
                    print(f"[건강검진] {line.strip()}")
    else:
        print("[건강검진] 로그 없음 (아직 미실행)")

    print()

    # 3. 핵심 데이터 파일 수정 시각
    checks = {
        "brain_decision": ROOT / "data" / "brain_decision.json",
        "regime_macro": ROOT / "data" / "regime_macro_signal.json",
        "dart_disclosures": ROOT / "data" / "dart_disclosures.json",
        "tomorrow_picks": ROOT / "data" / "tomorrow_picks.json",
        "shield_report": ROOT / "data" / "shield_report.json",
    }

    print("[핵심 데이터 최신 여부]")
    for name, path in checks.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            fresh = "✅" if mtime.strftime("%Y-%m-%d") == today else "⚠️"
            print(f"  {fresh} {name}: {mtime.strftime('%m/%d %H:%M')}")
        else:
            print(f"  ❌ {name}: 파일 없음")


if __name__ == "__main__":
    main()
