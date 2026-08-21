"""
Smart Money Z-score 계산

참조 코드 기반:
  SmartRaw = 0.6 * Foreign3 + 0.4 * Inst3
  SmartZ = (SmartRaw - 60MA) / 60STD

외국인/기관 수급의 3일 누적 → 60일 기준 정규화
"""

import numpy as np
import pandas as pd


def calc_smart_money_z(
    df: pd.DataFrame,
    foreign_col: str = "foreign_net",
    inst_col: str = "inst_net",
    foreign_weight: float = 0.6,
    inst_weight: float = 0.4,
    rolling_sum: int = 3,
    ma_period: int = 60,
) -> pd.Series:
    """
    Smart Money Z-score 계산.

    Args:
        df: 수급 컬럼(foreign_net, inst_net)이 포함된 DataFrame
        foreign_weight: 외국인 가중치 (기본 0.6)
        inst_weight: 기관 가중치 (기본 0.4)

    Returns:
        SmartZ Series
    """
    # 수급 컬럼 존재 확인
    f_col = None
    i_col = None

    # ★★8/21 교정(B-92): 후보에 **한글 컬럼명**을 추가한다.
    #   구 코드는 영문 이름만 찾았는데 이 저장소의 parquet 수급 컬럼은
    #   `외국인합계`·`기관합계`(한글)다(`extend_parquet_data.py`의 supply_cols가
    #   정본). 그래서 f_col/i_col이 **항상 None**이 되어 아래에서
    #   `Series(0.0)`을 반환했고, `smart_z`가 **전 종목·전 기간 0**이었다.
    #   B-67(소비자가 `("late","over")` 영문 리터럴을 하드코딩했는데 실제 어휘는
    #   한국어라 배포 이래 0건 동작)과 **정확히 같은 구조**다.
    #   영문을 먼저 보고 없을 때 한글로 떨어지므로 기존 호출부는 영향 없다.
    for candidate in [foreign_col, "ForeignNet", "foreign_buy_sell", "외국인합계"]:
        if candidate in df.columns:
            f_col = candidate
            break

    for candidate in [inst_col, "InstNet", "inst_buy_sell", "기관합계"]:
        if candidate in df.columns:
            i_col = candidate
            break

    if f_col is None or i_col is None:
        return pd.Series(0.0, index=df.index, name="smart_z")

    # 3일 누적
    foreign_3 = df[f_col].rolling(rolling_sum).sum()
    inst_3 = df[i_col].rolling(rolling_sum).sum()

    # Smart Money Raw
    smart_raw = foreign_weight * foreign_3 + inst_weight * inst_3

    # 60일 기준 Z-score
    smart_ma = smart_raw.rolling(ma_period).mean()
    smart_std = smart_raw.rolling(ma_period).std()

    smart_z = (smart_raw - smart_ma) / smart_std.replace(0, np.nan)
    smart_z = smart_z.fillna(0)
    smart_z.name = "smart_z"

    return smart_z


def check_smart_money_gate(
    row: pd.Series,
    min_smart_z: float = 0.0,
) -> tuple[bool, str]:
    """
    Smart Money 게이트 체크.

    Returns:
        (passed, block_reason)
    """
    # ★8/21(B-92): `min_smart_z`가 None이면 게이트 **명시적 비활성**.
    #   지금까지 이 게이트는 smart_z가 전량 0이라 `0 < 0.0` = False로 아무도
    #   차단하지 않았다 — 즉 **사실상 꺼져 있었는데 켜져 있는 것처럼 보였다**.
    #   컬럼명 버그를 고쳐 smart_z를 살리면 `min_smart_z: 0.0`이 그대로 발동해
    #   **820종목(69.3%)이 즉시 차단**된다(VPS 1,183종목 실측).
    #   그 임계는 smart_z가 죽어 있는 동안 정해진 값이라 의도된 컷이라는 근거가
    #   없고, 백테스트 없이 69%를 자르는 것은 8/11 B-63에서 세운 원칙
    #   ("임의 매핑 금지 · 배수는 백테스트 선행")에 어긋난다.
    #   → 값 복구와 게이트 활성화를 **분리**한다. 임계는 백테스트로 정한 뒤 켠다.
    if min_smart_z is None:
        return True, ""

    sz = row.get("smart_z", np.nan)
    if pd.isna(sz):
        return True, ""  # 데이터 없으면 통과

    if sz < min_smart_z:
        return False, "low_smart_z"

    return True, ""


# ══════════════════════════════════════════════
# v3.1 Smart Money v2 — 강화 함수
# ══════════════════════════════════════════════

def calc_enhanced_smart_z(
    df: pd.DataFrame,
    accum_bonus: int = 0,
    div_bonus: int = 0,
) -> pd.Series:
    """
    v3.1 강화 SmartZ = 기존 SmartZ + 매집 보너스 + 다이버전스 보너스.

    Args:
        df: smart_z 컬럼이 있는 DataFrame
        accum_bonus: 매집 단계 보너스 점수 (0/5/10/15/-20)
        div_bonus: 다이버전스 보너스 점수

    Returns:
        enhanced_smart_z Series
    """
    base_z = df.get("smart_z", pd.Series(0.0, index=df.index))
    # 보너스를 z-score 스케일에 맞게 변환 (10점 = +0.5 z)
    bonus_z = (accum_bonus + div_bonus) / 20.0
    enhanced = base_z + bonus_z
    enhanced.name = "enhanced_smart_z"
    return enhanced


def calc_institutional_streak(series: pd.Series) -> pd.Series:
    """
    v3.1 연속 순매수/순매도 일수 계산.

    양수 = 연속 순매수 일수, 음수 = 연속 순매도 일수.
    """
    streak = pd.Series(0, index=series.index, dtype=int)
    for i in range(1, len(series)):
        val = series.iloc[i]
        prev = streak.iloc[i - 1]
        if val > 0:
            streak.iloc[i] = max(prev, 0) + 1
        elif val < 0:
            streak.iloc[i] = min(prev, 0) - 1
        else:
            streak.iloc[i] = 0
    streak.name = f"{series.name}_streak" if series.name else "streak"
    return streak
