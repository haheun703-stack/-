"""EWY(KIS ETF 구성종목) 빈 응답률 실측 — 재시도 횟수 산정 근거.

7/23 최초 측정: 12회 연속 호출 중 7회가 `rt_cd=0`·`msg1="정상처리"`인데 output2만 0건
= **빈응답률 58%**. 장중 14시대 측정이었고 3초 간격에도 EMPTY↔OK가 혼재해,
기존 주석의 "제공 시간창 의존" 추정은 틀렸고 **시각 무관 서버측 상시 랜덤**으로 판명됐다.

이 값으로 collect_ewy_holdings.py의 재시도를 3→8회로 정했다:
  0.58^3 = 19.9% ≈ 5일에 1일 실패 (7/20 3회차·7/21 2회차 성공·7/22 실패와 일치)
  0.58^8 =  1.3% ≈ 75일에 1일

**재측정이 필요할 때**(EWY 실패가 다시 잦아지면) 이 스크립트로 빈응답률을 다시 재고
`KIS_RETRY_MAX`를 재산정한다. 감으로 횟수를 올리지 말 것.

실행: VPS에서만 (KIS API가 고정 IP 화이트리스트 기반)
  ./venv/bin/python3.11 scripts/probe_ewy_empty_rate.py [--n 12] [--interval 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="연속 호출 횟수")
    ap.add_argument("--interval", type=float, default=3.0, help="호출 간격(초)")
    args = ap.parse_args()

    from scripts.collect_ewy_holdings import KIS_ETF_TICKER
    from src.adapters.kis_investor_adapter import (
        _issue_token, KIS_BASE_URL, KIS_APP_KEY, KIS_APP_SECRET,
    )

    token = _issue_token()
    url = f"{KIS_BASE_URL}/uapi/etfetn/v1/quotations/inquire-component-stock-price"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST121600C0",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": KIS_ETF_TICKER,
        "FID_COND_SCR_DIV_CODE": "11216",
    }

    print(f"[probe] {args.n}회 연속 호출 ({args.interval}초 간격), 종목 {KIS_ETF_TICKER}")
    ok = 0
    for i in range(args.n):
        try:
            d = requests.get(url, headers=headers, params=params, timeout=15).json()
            n2 = len(d.get("output2") or [])
            rt = d.get("rt_cd")
            good = rt == "0" and n2 > 0
            ok += good
            print(f"  {i + 1:2d}. {'OK   ' if good else 'EMPTY'} rt_cd={rt} output2={n2:2d}건 "
                  f"msg={(d.get('msg1') or '').strip()[:36]}")
        except Exception as e:  # noqa: BLE001 — 진단 도구
            print(f"  {i + 1:2d}. ERROR {str(e)[:60]}")
        if i < args.n - 1:
            time.sleep(args.interval)

    if not args.n:
        return
    p_fail = (args.n - ok) / args.n
    print(f"\n[결과] 성공 {ok}/{args.n} = {100 * ok / args.n:.0f}%  "
          f"빈응답률 {100 * p_fail:.0f}%")
    if 0 < p_fail < 1:
        print("  재시도 k회 시 전부 실패 확률(독립 가정):")
        for k in (3, 5, 6, 8, 10):
            print(f"    k={k:2d} → {p_fail ** k * 100:6.2f}%  "
                  f"(≈ {1 / (p_fail ** k):.0f}일에 1일)")
        print("  ※ 독립 가정 — 연속 실패에 자기상관이 있으면 실제 실패율은 이보다 높다.")
        print("    현행 collect_ewy_holdings.KIS_RETRY_MAX 와 대조해 재산정할 것.")


if __name__ == "__main__":
    main()
