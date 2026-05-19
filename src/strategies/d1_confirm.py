"""D+1 양봉 확인 — 차트영웅 매매법의 진입 트리거.

차트영웅 룰 (메모리 backtest_results.md L46~47):
  "저변동+눌림+D+1양봉 = D+3 +5.56%, WR 91.7%, PF 31.65" (n=24)
  "눌림+수급+D+1양봉 = D+3 +5.75%, WR 90.9%, PF 25.10" (n=33)

→ D+1 양봉 확인이 알파의 핵심 분기점.

진입 조건 (2개 모두 만족):
  1) D+1 종가 > D+1 시가          (당일 양봉)
  2) D+1 종가 > D0 종가 × 0.95    (-5% 이상 안 빠짐)

확인 시점:
  - 14:30 KST: 1차 체크 (마감 30분 전)
  - 14:50 KST: 2차 체크 (마감 10분 전)
  - 14:55 KST: 최종 확정 (예약매수 발주)
  - 15:00 KST: 매수 체결 (KIS 예약매수)

5/22 paper mirror 운영:
  매일 14:55 호출 → 5-Gate picks 중 D+1 양봉 확인된 종목만 다음날 종가 매수.
"""

import datetime as dt
from src.adapters.kis_nxt_kit import _BASE, _headers, get_nx_price
import requests


def get_current_intraday(code: str) -> dict | None:
    """현재가 + 당일 시가/고가/저가 (KIS NX 통합 시세).

    Returns:
        { code, current, open, high, low, prev_close, chg_pct, volume }
    """
    r = requests.get(
        f"{_BASE}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=_headers("FHKST01010100"),
        params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
        timeout=5,
    )
    if r.status_code != 200 or r.json().get("rt_cd") != "0":
        return None
    o = r.json().get("output", {})
    try:
        return {
            "code": code,
            "current": int(o.get("stck_prpr", 0) or 0),
            "open":    int(o.get("stck_oprc", 0) or 0),
            "high":    int(o.get("stck_hgpr", 0) or 0),
            "low":     int(o.get("stck_lwpr", 0) or 0),
            "prev_close": int(o.get("stck_sdpr", 0) or 0),  # 전일 종가
            "chg_pct":  float(o.get("prdy_ctrt", 0) or 0),
            "volume":   int(o.get("acml_vol", 0) or 0),
        }
    except Exception:
        return None


def check_d1_candle(code: str, d0_close: int | None = None,
                    min_recovery_pct: float = -5.0) -> dict:
    """D+1 양봉 진입 조건 체크.

    Args:
        code: 종목코드
        d0_close: D0(전날, 상한가일) 종가. None이면 KIS prev_close 사용
        min_recovery_pct: D+1 종가가 D0 종가 대비 최저 허용 비율 (기본 -5%)

    Returns:
        {
          code, current, open, prev_close, d0_close,
          is_bullish_candle: bool,    # D+1 종가 > D+1 시가
          recovery_pct: float,         # D+1 vs D0
          is_recovery_ok: bool,        # recovery_pct >= -5%
          entry_ok: bool,              # 두 조건 모두 만족
          reason: str,
        }
    """
    p = get_current_intraday(code)
    if not p:
        return {"code": code, "entry_ok": False, "reason": "KIS API 응답 없음"}

    d0 = d0_close or p["prev_close"]
    cur = p["current"]
    op  = p["open"]

    # 조건 1: 당일 양봉 (현재가 > 시가)
    is_bullish = cur > op

    # 조건 2: D0 대비 회복률
    rec_pct = round((cur - d0) / d0 * 100, 2) if d0 > 0 else 0
    is_recovery_ok = rec_pct >= min_recovery_pct

    entry_ok = is_bullish and is_recovery_ok

    reasons = []
    reasons.append(f"양봉={'✓' if is_bullish else '✗'} ({cur} vs 시가 {op})")
    reasons.append(f"회복={'✓' if is_recovery_ok else '✗'} ({rec_pct}% vs {min_recovery_pct}%)")

    return {
        "code": code,
        "current": cur,
        "open": op,
        "prev_close": p["prev_close"],
        "d0_close": d0,
        "high": p["high"],
        "low": p["low"],
        "chg_pct": p["chg_pct"],
        "is_bullish_candle": is_bullish,
        "recovery_pct": rec_pct,
        "is_recovery_ok": is_recovery_ok,
        "entry_ok": entry_ok,
        "reason": " | ".join(reasons),
    }


def monitor_picks(picks: list[dict], check_time: str | None = None) -> list[dict]:
    """5-Gate 후보 종목들 D+1 양봉 확인.

    Args:
        picks: surge_d1_picker.run_picker() 출력의 picks
        check_time: 'HH:MM' (None이면 현재 시각)

    Returns:
        [{...pick, d1_check: {...}, will_enter: bool}, ...]
    """
    now = check_time or dt.datetime.now().strftime("%H:%M")
    out = []
    for p in picks:
        check = check_d1_candle(p["ticker"], d0_close=p.get("current_price"))
        out.append({
            **p,
            "check_time": now,
            "d1_check": check,
            "will_enter": check.get("entry_ok", False),
            "entry_price_estimate": check.get("current", 0),
        })
    return out


def is_check_window(now: dt.time | None = None) -> bool:
    """현재 시각이 D+1 양봉 확인 윈도우(14:30~15:00 KST)인지."""
    now = now or dt.datetime.now().time()
    start = dt.time(14, 30)
    end = dt.time(15, 0)
    return start <= now <= end


if __name__ == "__main__":
    # 검증: 사장님 시드 32종목 중 5종목 현재 시점 D+1 양봉 확인 시뮬레이션
    test_targets = [
        ("005930", "삼성전자",       275500),
        ("000660", "SK하이닉스",     1745000),
        ("454910", "두산로보틱스",   100100),
        ("032500", "케이엠더블유",   31950),
        ("012450", "한화에어로",     1286000),
    ]
    print(f"=== D+1 양봉 확인 시뮬레이션 (현재 시각 {dt.datetime.now().strftime('%H:%M:%S')}) ===")
    print(f"체크 윈도우(14:30~15:00 KST): {'✓ 안에 있음' if is_check_window() else '✗ 밖 (장외)'}\n")

    for code, name, d0 in test_targets:
        r = check_d1_candle(code, d0_close=d0)
        if "current" not in r:
            print(f"  ❌ {name}: {r.get('reason')}")
            continue
        flag = "⭐ ENTER" if r["entry_ok"] else "🛑 SKIP"
        print(f"{flag} {name:12} cur={r['current']:>10} d0={d0:>10} "
              f"양봉={r['is_bullish_candle']} 회복={r['recovery_pct']:>5.1f}%")
        print(f"   {r['reason']}")
