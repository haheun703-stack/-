"""매크로 4-시그널 게이트 — 차트영웅 매매법 Gate 1 (시장 진입 판정).

차트영웅 영상 4/24 "1년에 단 한 번 나오는 승률 100% 타점":
시장 저점은 4가지 시그널이 동시에 충족될 때 잡힌다.
1) 전쟁/이슈 충격 (일시적, 1개월 이내)         — 우리는 환율로 대체
2) PER 저평가 (S&P 500 < 20배)                — 우리는 미국 10년 국채로 대체
3) 기술적 분석 (주봉 스토캐스틱/RSI 과매도)      — KOSPI 주봉 %K
4) 공포탐욕지수 (< 10~15)                       — CNN F&G

퀀트봇 4-시그널 (한국 시장 맞춤):
1) KOSPI 주봉 %K < 30
2) 미국 10년 국채 < 4.5%
3) USD/KRW < 1,450원
4) CNN 공포탐욕지수 < 25

3개 이상 만족 = 신규 진입 GO. 2개 이하 = 보유만 (신규 매수 X).

검증 (2026-05-19):
  KOSPI %K 74.20  /  US10Y 4.62%  /  KRW 1,508원  /  F&G 62.17 (greed)
  → 0/4 GO. 전면 진입 금지.
"""

import csv
import datetime as dt
import os
import requests
from pathlib import Path

# ※ kis_weekly_kit는 compute_four_signal_gate 안에서 lazy import — import 부작용 격리(아래 주석).

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36"}


def get_us_10y_yield() -> float | None:
    """미국 10년 국채 수익률 (%, ^TNX) via Yahoo v8 chart API."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=5d&interval=1d"
    try:
        r = requests.get(url, headers=_UA, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()["chart"]["result"][0]
        closes = data["indicators"]["quote"][0]["close"]
        return round(float(closes[-1]), 4)
    except Exception:
        return None


def get_usd_krw() -> float | None:
    """USD/KRW 환율 via Yahoo v8."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/KRW=X?range=5d&interval=1d"
    try:
        r = requests.get(url, headers=_UA, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()["chart"]["result"][0]
        closes = data["indicators"]["quote"][0]["close"]
        return round(float(closes[-1]), 2)
    except Exception:
        return None


def get_cnn_fear_greed() -> tuple[float, str] | None:
    """CNN 공포탐욕지수 (0~100, rating) via CNN graphdata API.

    Returns:
        (score, rating) — rating ∈ {'extreme fear', 'fear', 'neutral', 'greed', 'extreme greed'}
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        r = requests.get(url, headers=_UA, timeout=10)
        if r.status_code != 200:
            return None
        fg = r.json().get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "")
        if score is None:
            return None
        return round(float(score), 2), str(rating)
    except Exception:
        return None


def compute_four_signal_gate(today: str | None = None) -> dict:
    """매크로 4-시그널 합성 → 신규 진입 GO/NO-GO 판정.

    Args:
        today: 'YYYY-MM-DD' (None이면 오늘)

    Returns:
        {
          date, kospi_weekly_k, us10y, krw, fg_score, fg_rating,
          s1_kospi_oversold:bool, s2_us10y_safe:bool,
          s3_krw_safe:bool, s4_fg_fearful:bool,
          gate_score: int (0~4),
          gate_pass: bool (>=3),
          reason: str (사람 읽는 설명)
        }
    """
    today = today or dt.date.today().isoformat()

    # ★lazy import (테스트 격리·import 부작용 차단): kis_weekly_kit는 하위 체인에서 .env 전체를
    #   os.environ에 로드하는 부작용이 있어, four_signal_gate를 단순 import하는 것만으로 다른
    #   모듈의 모듈 레벨 os.getenv 캐시(예: adaptive_buy_queue.SPLIT_MAX_QTY)를 오염시킨다.
    #   호출 시점으로 가둬 stale_warning 등 순수 헬퍼의 import는 .env를 건드리지 않게 한다.
    from src.adapters.kis_weekly_kit import get_index_weekly, compute_weekly_stoch_k

    # Signal 1: KOSPI 주봉 %K (최근 6개월 데이터로 계산)
    end_dt = dt.date.today()
    start_dt = end_dt - dt.timedelta(days=200)
    kospi_w = get_index_weekly("0001", start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    kospi_k = compute_weekly_stoch_k(kospi_w, period=14) if kospi_w else None
    s1 = kospi_k is not None and kospi_k < 30

    # Signal 2: 미국 10년 국채
    us10y = get_us_10y_yield()
    s2 = us10y is not None and us10y < 4.5

    # Signal 3: USD/KRW
    krw = get_usd_krw()
    s3 = krw is not None and krw < 1450

    # Signal 4: CNN 공포탐욕지수
    fg = get_cnn_fear_greed()
    fg_score, fg_rating = (fg if fg else (None, None))
    s4 = fg_score is not None and fg_score < 25

    score = int(s1) + int(s2) + int(s3) + int(s4)
    gate_pass = score >= 3

    reasons = []
    reasons.append(f"KOSPI%K={kospi_k} {'✓' if s1 else '✗'}")
    reasons.append(f"US10Y={us10y}% {'✓' if s2 else '✗'}")
    reasons.append(f"KRW={krw} {'✓' if s3 else '✗'}")
    reasons.append(f"F&G={fg_score}({fg_rating}) {'✓' if s4 else '✗'}")

    return {
        "date": today,
        "kospi_weekly_k": kospi_k,
        "us10y": us10y,
        "krw": krw,
        "fg_score": fg_score,
        "fg_rating": fg_rating,
        "s1_kospi_oversold": s1,
        "s2_us10y_safe": s2,
        "s3_krw_safe": s3,
        "s4_fg_fearful": s4,
        "gate_score": score,
        "gate_pass": gate_pass,
        "reason": " | ".join(reasons),
    }


def save_daily_record(result: dict, csv_path: str | None = None) -> str:
    """일별 합성 결과 CSV append."""
    csv_path = csv_path or "data/macro_four_signal_daily.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "kospi_k", "us10y", "krw", "fg_score", "fg_rating",
                        "s1", "s2", "s3", "s4", "gate_score", "gate_pass"])
        w.writerow([
            result["date"], result["kospi_weekly_k"], result["us10y"], result["krw"],
            result["fg_score"], result["fg_rating"],
            result["s1_kospi_oversold"], result["s2_us10y_safe"],
            result["s3_krw_safe"], result["s4_fg_fearful"],
            result["gate_score"], result["gate_pass"],
        ])
    return csv_path


def _last_record_date(csv_path: str) -> dt.date | None:
    """CSV 마지막 데이터 행의 date 컬럼(YYYY-MM-DD) → date. 실패/빈 파일 → None."""
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r and r[0].strip()]
    except OSError:
        return None
    # 헤더(첫 행 date 컬럼) 제외 후 마지막 데이터 행
    if rows and rows[0] and rows[0][0].strip().lower() == "date":
        rows = rows[1:]
    if not rows:
        return None
    try:
        return dt.date.fromisoformat(rows[-1][0].strip())
    except (ValueError, IndexError):
        return None


def stale_warning(
    csv_path: str | None = None,
    max_age_days: int = 5,
    today: dt.date | None = None,
) -> str | None:
    """four_signal 일별 CSV가 max_age_days(달력일) 이상 묵었으면 경고 문자열, 아니면 None.

    ★재발방지 (fx-liquidity P0-1, 6/13): 생산자 save_daily_record는 이 파일 __main__에서만
      호출 — BAT/cron 어디에도 미배선(고아)이라 5/19 수동 실행 후 조용히 멈췄다. chart_hero
      파이프라인 휴면(unfreeze-checklist D)이라 의도적 미가동이지만, '조용히 죽은 것'과 '의도적
      휴면'을 사람이 구분 못 하면 6/8 설계서처럼 stale을 인지하고도 3주 방치된다. preflight가
      이 경고로 stale을 자가 노출 — 매매 안전 게이트(preflight checks)와 분리된 read-only 경고라
      RESULT/카운트 불변(회귀 격리, 6/11 finality '게이트 분리 경고' 원칙).
      ★부활(자동 배선)·backfill·MacroRiskScore 연결은 fx-liquidity P1 별건 승인 — 그 전까진
      이 경고가 휴면 데이터의 '조용한 죽음'을 막는 안전판이다.

    파일 자체가 없으면 None(생성된 적 없는 상태와 구분 불가 → 휴면 정상으로 간주, 경고 안 함).
    """
    csv_path = csv_path or "data/macro_four_signal_daily.csv"
    today = today or dt.date.today()
    if not Path(csv_path).exists():
        return None
    last = _last_record_date(csv_path)
    if last is None:
        return f"{csv_path}: 데이터 행 없음/파싱 불가 (파일은 존재 — 손상 의심)"
    age = (today - last).days
    if age >= max_age_days:
        return (
            f"{csv_path} {age}일 stale (마지막={last.isoformat()}). 생산자 save_daily_record는 "
            f"BAT/cron 미배선(수동 전용)이라 조용히 멈춤 — chart_hero 휴면이면 정상 / "
            f"부활 의도면 일일 배선 필요(fx-liquidity P1)."
        )
    return None


if __name__ == "__main__":
    print("=== 매크로 4-시그널 게이트 검증 (차트영웅 Gate 1) ===")
    r = compute_four_signal_gate()
    print(f"\n[{r['date']}] 합성 결과")
    print(f"  Signal 1 KOSPI 주봉 %K   : {r['kospi_weekly_k']:>8} (< 30)   {'GO' if r['s1_kospi_oversold'] else 'NO-GO'}")
    print(f"  Signal 2 미국 10년 국채   : {r['us10y']:>8}% (< 4.5%) {'GO' if r['s2_us10y_safe'] else 'NO-GO'}")
    print(f"  Signal 3 USD/KRW         : {r['krw']:>8}원 (< 1450)  {'GO' if r['s3_krw_safe'] else 'NO-GO'}")
    print(f"  Signal 4 CNN 공포탐욕     : {r['fg_score']:>8} ({r['fg_rating']}) (< 25)  {'GO' if r['s4_fg_fearful'] else 'NO-GO'}")
    print(f"\n  GATE 점수 : {r['gate_score']}/4")
    print(f"  진입 판정 : {'⭐ GO (3/4 이상)' if r['gate_pass'] else '🛑 NO-GO (신규 진입 금지)'}")
    print(f"\n  요약: {r['reason']}")

    path = save_daily_record(r)
    print(f"\n  저장: {path}")
