"""Read-only preflight for Quantum Master trading safety.

Usage:
  python tools/quant_preflight.py --expect blocked

The check intentionally avoids broker calls and does not print secrets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_batch(path: Path) -> str:
    for encoding in ("utf-8", "cp949", "mbcs"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    return path.read_text(errors="replace")


def _status(label: str, ok: bool, detail: str) -> tuple[bool, str]:
    mark = "PASS" if ok else "FAIL"
    return ok, f"[{mark}] {label}: {detail}"


def _simulate_paper_checks() -> list[tuple[bool, str]]:
    """S1~S6 paper-intent 시뮬레이션 가드 (§9 dry-run 단계 3).

    부작용 0 보장:
      - S3: tempfile로 ORDER_INTENTS_DIR 치환 후 원복 → 운영 jsonl 오염 0
      - S4: KisOrderAdapter.__new__ 우회 → __init__ mojito 토큰 발급 네트워크 회피
      - S5: PaperOrderAdapter는 __init__ 부작용 없음 + buy_limit mode 차단이 부작용 전
    각 가드 독립 try/except — 한 가드 실패가 나머지를 가리지 않음.
    예외 타입을 좁게 (ValueError/PermissionError) 잡아 "진짜 차단"만 PASS.
    """
    import tempfile
    from datetime import datetime, timedelta, timezone

    results: list[tuple[bool, str]] = []
    kst = timezone(timedelta(hours=9))
    now = datetime.now(tz=kst)

    # S1: paper intent dict (필수 9필드, expires_at 상대 미래 — 만료 시한폭탄 회피)
    intent = {
        "intent_id": "sim_240810_preflight",
        "bot": "quant",
        "engine": "preflight_simulate",
        "ticker": "240810",
        "side": "BUY",
        "mode": "paper",
        "score": 0.0,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=4)).isoformat(),
    }
    results.append(_status("S1 paper intent dict", True, "생성 OK (expires 상대 미래)"))

    # S2: HMAC 서명 생성 + 검증
    try:
        from src.use_cases.order_intents_gate import _compute_signature, _verify_signature
        intent["hmac_signature"] = _compute_signature(intent)
        ok_s2 = _verify_signature(intent)
        results.append(
            _status("S2 HMAC 서명+검증", ok_s2, "verify OK" if ok_s2 else "verify FAIL")
        )
    except Exception as e:
        results.append(_status("S2 HMAC 서명+검증", False, f"{type(e).__name__}: {e}"))

    # S3: order_intents_gate paper+quant 통과 (tempfile 격리 — 운영 jsonl 오염 0)
    try:
        import src.use_cases.order_intents_gate as gate
        saved_dir = gate.ORDER_INTENTS_DIR
        with tempfile.TemporaryDirectory() as td:
            gate.ORDER_INTENTS_DIR = Path(td)
            try:
                gate.register_intent(intent, bot="quant")
                gate.assert_order_intent_exists(
                    ticker="240810", side="BUY", mode="paper", executor_bot="quant",
                )
                results.append(
                    _status("S3 intent gate paper+quant", True, "register+assert 통과")
                )
            finally:
                gate.ORDER_INTENTS_DIR = saved_dir
    except Exception as e:
        results.append(
            _status("S3 intent gate paper+quant", False, f"{type(e).__name__}: {e}")
        )

    # S4: KisOrderAdapter mode=paper 차단 (live 전용). __new__ 우회 = 네트워크 0
    try:
        from src.adapters.kis_order_adapter import KisOrderAdapter
        obj = KisOrderAdapter.__new__(KisOrderAdapter)
        try:
            obj._guard("240810", 1, side="BUY", mode="paper", executor_bot="quant")
            results.append(
                _status("S4 KisAdapter mode=paper 차단", False, "raise 안 됨 (위험)")
            )
        except ValueError:
            results.append(
                _status("S4 KisAdapter mode=paper 차단", True, "ValueError 차단 OK")
            )
    except Exception as e:
        results.append(
            _status("S4 KisAdapter mode=paper 차단", False, f"예상밖 {type(e).__name__}: {e}")
        )

    # S5: PaperOrderAdapter mode=live 차단 (paper 전용)
    try:
        from src.adapters.paper_order_adapter import PaperOrderAdapter
        padapter = PaperOrderAdapter()
        try:
            padapter.buy_limit("240810", 1000, 1, mode="live", executor_bot="quant")
            results.append(
                _status("S5 PaperAdapter mode=live 차단", False, "raise 안 됨 (위험)")
            )
        except ValueError:
            results.append(
                _status("S5 PaperAdapter mode=live 차단", True, "ValueError 차단 OK")
            )
    except Exception as e:
        results.append(
            _status("S5 PaperAdapter mode=live 차단", False, f"예상밖 {type(e).__name__}: {e}")
        )

    # S6: runtime guard — 차단(PermissionError)=PASS, 비차단=FAIL
    #     (KILL_SWITCH/PAPER_ONLY 부재 = 안전장치 OFF → FAIL, fail-closed)
    try:
        from src.utils.trade_runtime_safety import assert_runtime_orders_allowed
        try:
            assert_runtime_orders_allowed()
            results.append(
                _status("S6 runtime guard 차단", False, "미차단 — KILL_SWITCH/PAPER_ONLY 부재")
            )
        except PermissionError as e:
            results.append(_status("S6 runtime guard 차단", True, f"차단 OK: {e}"))
    except Exception as e:
        results.append(
            _status("S6 runtime guard 차단", False, f"예상밖 {type(e).__name__}: {e}")
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        choices=("blocked", "live"),
        default="blocked",
        help="Expected runtime state.",
    )
    parser.add_argument(
        "--simulate-paper",
        action="store_true",
        help="Run S1~S6 paper-intent simulation guards (§9 dry-run step 3).",
    )
    args = parser.parse_args()

    _load_env(PROJECT_ROOT / ".env")

    from src.utils.trade_runtime_safety import runtime_order_block_reasons

    data_dir = PROJECT_ROOT / "data"
    upper_kill = data_dir / "KILL_SWITCH"
    lower_kill = data_dir / "kill_switch.flag"
    schedule_e = PROJECT_ROOT / "scripts" / "schedule_E_smart_entry.bat"
    schedule_p = PROJECT_ROOT / "scripts" / "schedule_P_daytrading.bat"

    runtime_reasons = runtime_order_block_reasons()
    schedule_e_text = _read_batch(schedule_e) if schedule_e.exists() else ""
    schedule_p_text = _read_batch(schedule_p) if schedule_p.exists() else ""
    smart_entry_command_live = "scripts/smart_entry_runner.py --live" in schedule_e_text

    # P0 (5/28 추가): cron/shell 진입점 live/real 인자 검사 (5/27 사고 후속)
    # scripts/cron/run_bat.sh 또는 .bat 파일에서 위험 인자 등장 시 BLOCK
    run_bat_sh = PROJECT_ROOT / "scripts" / "cron" / "run_bat.sh"
    run_bat_sh_text = _read_batch(run_bat_sh) if run_bat_sh.exists() else ""
    danger_args = ("--live", "--real", "--force", "--no-dry-run")
    bat_danger_hits: list[str] = []
    for ln_no, line in enumerate(run_bat_sh_text.splitlines(), 1):
        # # 이전 코드 부분만 검사 (라인 중간 코멘트 포함 제외)
        code_part = line.split("#", 1)[0]
        for arg in danger_args:
            if arg in code_part:
                bat_danger_hits.append(f"run_bat.sh:{ln_no} {arg}")

    # raw mojito broker 호출 정적 검사 (KisOrderAdapter 외부에서)
    # src/ 와 scripts/ 디렉토리만 검사 (tests/ 제외)
    raw_broker_hits: list[str] = []
    for sub in ("src", "scripts"):
        sub_dir = PROJECT_ROOT / sub
        if not sub_dir.exists():
            continue
        for py_file in sub_dir.rglob("*.py"):
            # KisOrderAdapter 자체는 raw 호출 허용 (어댑터 내부)
            if py_file.name == "kis_order_adapter.py":
                continue
            try:
                txt = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for ln_no, line in enumerate(txt.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # broker.create_market_*_order( or broker.create_limit_*_order( 실제 호출 패턴
                # ( 까지 매치하여 backtick docstring 안의 텍스트는 제외
                if ("broker.create_market_" in line and "_order(" in line) or \
                   ("broker.create_limit_" in line and "_order(" in line):
                    raw_broker_hits.append(f"{py_file.relative_to(PROJECT_ROOT).as_posix()}:{ln_no}")

    checks: list[tuple[bool, str]] = []
    checks.append(
        _status(
            "runtime order gate",
            bool(runtime_reasons) if args.expect == "blocked" else not runtime_reasons,
            ", ".join(runtime_reasons) if runtime_reasons else "no runtime blockers",
        )
    )
    checks.append(
        _status(
            "data/KILL_SWITCH",
            upper_kill.exists() if args.expect == "blocked" else not upper_kill.exists(),
            "exists" if upper_kill.exists() else "absent",
        )
    )
    checks.append(
        _status(
            "legacy kill_switch.flag",
            True,
            "exists" if lower_kill.exists() else "absent",
        )
    )
    checks.append(
        _status(
            "AUTO_TRADING_ENABLED",
            (os.getenv("AUTO_TRADING_ENABLED", "0") != "1")
            if args.expect == "blocked"
            else os.getenv("AUTO_TRADING_ENABLED", "0") == "1",
            os.getenv("AUTO_TRADING_ENABLED", "0"),
        )
    )
    checks.append(
        _status(
            "MODEL",
            os.getenv("MODEL", "") in {"REAL", "MOCK", "PAPER", ""},
            os.getenv("MODEL", ""),
        )
    )
    checks.append(
        _status(
            "QM-E schedule live command",
            not smart_entry_command_live if args.expect == "blocked" else smart_entry_command_live,
            "live" if smart_entry_command_live else "dry-run/no-live",
        )
    )
    checks.append(
        _status(
            "QM-P missing script guard",
            "signal_logger_daytrading.py missing" in schedule_p_text
            and "daily_close_daytrading.py missing" in schedule_p_text,
            "guarded" if "missing" in schedule_p_text else "unguarded",
        )
    )
    # P0 신규 (5/28): cron/shell --live/--real/--force 인자 검사
    checks.append(
        _status(
            "cron/shell danger args (run_bat.sh)",
            len(bat_danger_hits) == 0,
            "no danger args" if not bat_danger_hits else f"{len(bat_danger_hits)} hits: {bat_danger_hits[:3]}",
        )
    )
    # P0 신규 (5/28): raw mojito broker 호출 정적 검사
    checks.append(
        _status(
            "raw mojito broker calls (use_cases/scripts)",
            len(raw_broker_hits) == 0,
            "no raw calls" if not raw_broker_hits else f"{len(raw_broker_hits)} hits: {raw_broker_hits[:5]}",
        )
    )
    # C3 fix (5/28 코덱스 검수): ORDER_INTENTS_HMAC_KEY 존재 + 길이 fail-fast
    # 키 부재 시 register_intent 매번 실패 → D+1 매매 0건 → 운영 안정성 위협
    # 키 값은 로그 출력 금지 (길이/존재만 보고)
    hmac_key = os.getenv("ORDER_INTENTS_HMAC_KEY", "")
    hmac_ok = bool(hmac_key) and len(hmac_key) >= 32
    checks.append(
        _status(
            "ORDER_INTENTS_HMAC_KEY (Phase 1 L10 가드)",
            hmac_ok,
            f"present (len={len(hmac_key)})" if hmac_key else "MISSING — set 32+ chars in .env",
        )
    )

    if args.simulate_paper:
        checks.extend(_simulate_paper_checks())

    ok_all = all(ok for ok, _ in checks)
    print(f"Quantum preflight expect={args.expect}")
    for _, line in checks:
        print(line)
    # kodex shadow ledger 미확정(provisional) 자가 감지 — 매매 안전 게이트(checks)와 분리된
    # read-only 경고. shadow는 매매 무관이라 RESULT/카운트에 넣지 않되, 저녁 --write 재실행
    # 누락 시 어제 레코드가 미확정으로 남은 걸 시스템이 스스로 노출(사람 기억 의존 제거).
    try:
        from src.etf.kodex_hedge_regime_shadow import stale_provisional_warning
        _kodex_warn = stale_provisional_warning()
        if _kodex_warn:
            print(f"[WARN] kodex shadow ledger: {_kodex_warn}")
    except Exception:
        pass
    # 카운트는 simulate 모드에서만 덧붙임 (비-simulate 출력 불변 = 회귀 격리)
    total = len(checks)
    passed = sum(1 for ok, _ in checks if ok)
    tail = f" ({passed}/{total})" if args.simulate_paper else ""
    print("RESULT:", ("PASS" if ok_all else "FAIL") + tail)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
