"""차트영웅 매매법 advisory — 역발상 (Mean Reversion).

단타봇 advisory(추세 추종)와 정반대 철학:
- 단타봇:    PANIC = "매수 자제!"        (위험 회피)
- 차트영웅:  PANIC = "100% 타점, 분할매수 GO!"  (기회 활용)

같은 시장 상황 → 두 봇이 다른 액션:
- 단타봇 → scalper_advisory (현 flowx_uploader 그대로)
- 퀀트봇 → quant_chart_hero_advisory (본 모듈)

5/19 결단 (사장님 옵션 B): 각자 스타일로 운영 → 수익률 비교 후 통합 검토.

검증 (2026-05-19, 4-시그널 0/4):
  "🚫 시장 고점 영역. 차트영웅 룰: 절대 진입 X (탐욕 시기)"
"""

import datetime as dt
from src.macro.four_signal_gate import compute_four_signal_gate


# 차트영웅 식 시장 모드 (역발상)
CHART_HERO_MODE_MAP = {
    4: ("PERFECT_PANIC",  "🎯 1년에 한 번 100% 타점!",
        "적극 분할매수 — 1차 1.5% + 추매 -10/-20/-30%, 다종목 분산"),
    3: ("BUYABLE_FEAR",   "⭐ 우수한 진입 기회 (3/4 시그널 일치)",
        "분할매수 GO — 1차 1.5%, 추매 -10% 단위, 차트영웅 룰 그대로"),
    2: ("WATCH_ZONE",     "⚠️ 아직 공포 부족 (2/4만 충족)",
        "종목 발굴 + 대기 — 신규 매수 보류, 보유 종목 차트영웅 룰 유지"),
    1: ("GREED_WARN",     "🛑 시장 탐욕 또는 추세 강함 (1/4만 충족)",
        "매수 기다림 — 4-시그널 3/4 이상 충족까지 신규 진입 X"),
    0: ("EXTREME_GREED",  "🚫 시장 고점 영역 (0/4 시그널)",
        "절대 진입 금지 — 차트영웅 룰: 탐욕 시기 = 매수 후회 시기"),
}


def build_chart_hero_advisory(today: str | None = None) -> dict:
    """차트영웅 매매법 advisory 생성.

    Args:
        today: 'YYYY-MM-DD' (None = 오늘)
    Returns:
        {
          date, mode, summary, strategy,
          gate_score, gate_pass, entry_signal: bool,
          signals: { kospi_k, us10y, krw, fg_score, fg_rating },
          per_signal: { s1, s2, s3, s4 },
          reasoning: str,        # 사람 읽는 추론 과정
        }
    """
    g = compute_four_signal_gate(today)
    score = g["gate_score"]
    mode_key, summary, strategy = CHART_HERO_MODE_MAP[score]

    # 사람 읽는 추론
    if score == 4:
        reason = ("4개 시그널 모두 충족: 시장 과매도+공포+안전. "
                  "차트영웅 4/24 영상 '1년에 한 번 100% 타점' 시점. "
                  "비중 1~2% 분할 진입 즉시 가능.")
    elif score == 3:
        reason = ("3개 시그널 충족. 차트영웅 룰 GO 조건 만족. "
                  "분할매수 시작 가능 — 단, 미충족 시그널 1개는 모니터링.")
    elif score == 2:
        reason = ("절반만 충족. 차트영웅 영상에서 '겁먹지 마세요'라 한 시점은 "
                  "3개 이상 동시 충족 때. 지금은 발굴+대기.")
    elif score == 1:
        reason = ("1개만 충족. 시장이 아직 탐욕 또는 고점 영역. "
                  "차트영웅 5/19 영상 '고점이다 댓글 다는 사람들 보면 아직 더 남았다' 단계.")
    else:  # 0
        reason = ("4-시그널 전부 미충족. 시장 탐욕 + 과매수 상태. "
                  "차트영웅: '몰빵 본능 = 누군가 보내는 매도 신호' — 절대 진입 금지.")

    return {
        "date": g["date"],
        "mode": mode_key,
        "summary": summary,
        "strategy": strategy,
        "gate_score": score,
        "gate_pass": g["gate_pass"],
        "entry_signal": g["gate_pass"],   # 3/4 이상 = 진입 시그널
        "signals": {
            "kospi_weekly_k": g["kospi_weekly_k"],
            "us10y":          g["us10y"],
            "usd_krw":        g["krw"],
            "fg_score":       g["fg_score"],
            "fg_rating":      g["fg_rating"],
        },
        "per_signal": {
            "s1_kospi_oversold": g["s1_kospi_oversold"],
            "s2_us10y_safe":     g["s2_us10y_safe"],
            "s3_krw_safe":       g["s3_krw_safe"],
            "s4_fg_fearful":     g["s4_fg_fearful"],
        },
        "reasoning": reason,
        "philosophy": "MEAN_REVERSION (역발상)",     # 단타봇 = "TREND_FOLLOWING"
    }


def to_telegram_text(adv: dict) -> str:
    """텔레그램/카톡 전송용 한글 메시지."""
    s = adv["signals"]
    p = adv["per_signal"]
    return (
        f"📊 차트영웅 매매 advisory ({adv['date']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{adv['summary']}\n"
        f"{adv['strategy']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"GATE 점수: {adv['gate_score']}/4  "
        f"{'⭐ 진입 GO' if adv['entry_signal'] else '🛑 진입 X'}\n\n"
        f"  KOSPI 주봉 %K = {s['kospi_weekly_k']:>6}  (<30) {'✓' if p['s1_kospi_oversold'] else '✗'}\n"
        f"  미국 10년물   = {s['us10y']:>6}%  (<4.5%) {'✓' if p['s2_us10y_safe'] else '✗'}\n"
        f"  USD/KRW      = {s['usd_krw']:>6}원 (<1450) {'✓' if p['s3_krw_safe'] else '✗'}\n"
        f"  공포탐욕지수  = {s['fg_score']:>6}  ({s['fg_rating']}) (<25) {'✓' if p['s4_fg_fearful'] else '✗'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 {adv['reasoning']}\n"
        f"\n철학: {adv['philosophy']} (단타봇 advisory와 반대 방향)"
    )


if __name__ == "__main__":
    adv = build_chart_hero_advisory()
    print(to_telegram_text(adv))
    print()
    print("=" * 60)
    print("단타봇 vs 퀀트봇(차트영웅) 결과 비교")
    print("=" * 60)
    print(f"  현재 시장: {adv['signals']['fg_rating']} (F&G {adv['signals']['fg_score']})")
    print(f"  단타봇 advisory (예상): '매수 자제!' (PANIC 차단)")
    print(f"  퀀트봇 차트영웅 advisory: '{adv['summary']}'")
    print(f"  → 같은 시장, 다른 결단. 5/26 이후 수익률 비교 → 통합 검토.")
