"""
Phase Transition Detector -- 상전이 5대 전조 현상 감지

수학적 근거: 복잡계 이론의 임계점(critical point) 전조 지표
  -> 포물선적 상승이 시작되기 직전, 시스템은 "임계 상태"를 보임
  -> 5가지 독립적 전조를 동시에 감지하면 신뢰도 극대화

5대 전조 현상:
  1. Critical Slowing (임계 감속) -- 자기상관 급증
  2. Vol of Vol (변동성의 변동성) -- 내부 균형 다툼
  3. Hurst Exponent (허스트 지수) -- 추세 형성 전조
  4. Flickering (깜빡임) -- 저항선 반복 터치 + 되돌림 축소
  5. Asymmetric Fluctuation (비대칭 요동) -- 상승/하락 비대칭

의존성: numpy, pandas (scipy 금지)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PhaseTransitionDetector:
    """상전이 5대 전조 현상 감지기"""

    # composite_score 가중치
    WEIGHTS = {
        "vol_of_vol": 0.25,
        "hurst": 0.25,
        "asymmetric": 0.20,
        "flickering": 0.15,
        "critical_slowing": 0.15,
    }

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("phase_transition", {})
        self.min_data_days = cfg.get("min_data_days", 60)
        self.critical_slowing_window = cfg.get("critical_slowing_window", 20)
        self.atr_window = cfg.get("atr_window", 5)
        self.vov_lookback = cfg.get("vov_lookback", 60)
        self.hurst_min_window = cfg.get("hurst_min_window", 10)
        self.hurst_max_window = cfg.get("hurst_max_window", 100)
        self.flickering_lookback = cfg.get("flickering_lookback", 20)
        self.asymmetric_window = cfg.get("asymmetric_window", 10)

    # --- 1. Critical Slowing (임계 감속) -------------------

    def critical_slowing(self, returns: np.ndarray, window: int = 20) -> dict:
        """
        임계 감속: 수익률의 lag-1 자기상관이 1에 근접하면 임계 상태.

        Parameters:
            returns: 일별 수익률 배열
            window: rolling autocorrelation 윈도우 크기

        Returns:
            {"autocorr": float, "score": 0.0~1.0, "detected": bool}
        """
        try:
            returns = np.asarray(returns, dtype=float)
            if len(returns) < window + 1:
                return {"autocorr": 0.0, "score": 0.0, "detected": False}

            # rolling lag-1 autocorrelation: 마지막 윈도우 사용
            r = returns[-(window + 1):]
            x = r[:-1]
            y = r[1:]

            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_dev = x - x_mean
            y_dev = y - y_mean

            denom = np.sqrt(np.sum(x_dev ** 2) * np.sum(y_dev ** 2))
            if denom < 1e-15:
                autocorr = 0.0
            else:
                autocorr = float(np.sum(x_dev * y_dev) / denom)

            score = min(1.0, abs(autocorr) / 0.5)
            detected = score >= 0.7

            return {
                "autocorr": round(autocorr, 4),
                "score": round(score, 4),
                "detected": detected,
            }
        except Exception as e:
            logger.warning("critical_slowing 계산 실패: %s", e)
            return {"autocorr": 0.0, "score": 0.0, "detected": False}

    # --- 2. Vol of Vol (변동성의 변동성) -------------------

    def vol_of_vol(self, prices: np.ndarray, atr_window: int = 5, lookback: int = 60) -> dict:
        """
        변동성의 변동성: ATR(5)의 표준편차/평균으로 내부 균형 다툼 감지.

        Parameters:
            prices: 종가 배열
            atr_window: ATR 계산 윈도우
            lookback: 비교 기간

        Returns:
            {"vov": float, "vov_ratio": float, "score": 0.0~1.0, "detected": bool}
        """
        try:
            prices = np.asarray(prices, dtype=float)
            if len(prices) < lookback + atr_window:
                return {"vov": 0.0, "vov_ratio": 0.0, "score": 0.0, "detected": False}

            # ATR 근사: |close[t] - close[t-1]| 의 rolling mean
            daily_range = np.abs(np.diff(prices))
            if len(daily_range) < atr_window:
                return {"vov": 0.0, "vov_ratio": 0.0, "score": 0.0, "detected": False}

            # rolling ATR(atr_window)
            atr_series = np.array([
                np.mean(daily_range[max(0, i - atr_window + 1):i + 1])
                for i in range(len(daily_range))
            ])

            if len(atr_series) < lookback:
                return {"vov": 0.0, "vov_ratio": 0.0, "score": 0.0, "detected": False}

            # 현재 VoV: 최근 atr_window 기간의 ATR 변동계수
            recent_atr = atr_series[-atr_window:]
            recent_mean = np.mean(recent_atr)
            if recent_mean < 1e-15:
                return {"vov": 0.0, "vov_ratio": 0.0, "score": 0.0, "detected": False}
            current_vov = float(np.std(recent_atr) / recent_mean)

            # 과거 평균 VoV: lookback 기간을 atr_window 단위로 분할하여 평균
            past_atr = atr_series[-lookback:-atr_window]
            if len(past_atr) < atr_window:
                return {"vov": current_vov, "vov_ratio": 1.0, "score": 0.0, "detected": False}

            past_vovs = []
            for i in range(0, len(past_atr) - atr_window + 1, atr_window):
                chunk = past_atr[i:i + atr_window]
                chunk_mean = np.mean(chunk)
                if chunk_mean > 1e-15:
                    past_vovs.append(np.std(chunk) / chunk_mean)

            if not past_vovs:
                return {"vov": current_vov, "vov_ratio": 1.0, "score": 0.0, "detected": False}

            avg_past_vov = float(np.mean(past_vovs))
            if avg_past_vov < 1e-15:
                vov_ratio = 1.0
            else:
                vov_ratio = current_vov / avg_past_vov

            score = min(1.0, max(0.0, (vov_ratio - 1.0) / 1.5))
            detected = score >= 0.6

            return {
                "vov": round(current_vov, 4),
                "vov_ratio": round(vov_ratio, 4),
                "score": round(score, 4),
                "detected": detected,
            }
        except Exception as e:
            logger.warning("vol_of_vol 계산 실패: %s", e)
            return {"vov": 0.0, "vov_ratio": 0.0, "score": 0.0, "detected": False}

    # --- 3. Hurst Exponent (허스트 지수) -------------------

    def hurst_exponent(self, prices: np.ndarray, min_window: int = 10, max_window: int = 100) -> dict:
        """
        DFA (Detrended Fluctuation Analysis) 방식의 Hurst 지수 계산.

        Parameters:
            prices: 종가 배열
            min_window: 최소 세그먼트 윈도우
            max_window: 최대 세그먼트 윈도우

        Returns:
            {"hurst": float, "score": 0.0~1.0, "detected": bool, "interpretation": str}
        """
        try:
            prices = np.asarray(prices, dtype=float)
            if len(prices) < max_window:
                # max_window를 데이터에 맞게 축소
                max_window = len(prices) // 2
                if max_window < min_window + 5:
                    return {
                        "hurst": 0.5,
                        "score": 0.0,
                        "detected": False,
                        "interpretation": "데이터 부족",
                    }

            # 1. 로그 수익률
            log_returns = np.diff(np.log(prices + 1e-10))
            n = len(log_returns)

            if n < min_window * 2:
                return {
                    "hurst": 0.5,
                    "score": 0.0,
                    "detected": False,
                    "interpretation": "데이터 부족",
                }

            # 2. 누적합 (profile)
            mean_ret = np.mean(log_returns)
            profile = np.cumsum(log_returns - mean_ret)

            # 3. 여러 윈도우 크기에서 fluctuation 계산
            window_sizes = np.unique(
                np.logspace(
                    np.log10(min_window),
                    np.log10(min(max_window, n // 2)),
                    num=15,
                ).astype(int)
            )
            window_sizes = window_sizes[window_sizes >= min_window]

            if len(window_sizes) < 3:
                return {
                    "hurst": 0.5,
                    "score": 0.0,
                    "detected": False,
                    "interpretation": "윈도우 부족",
                }

            log_windows = []
            log_flucts = []

            for w in window_sizes:
                num_segments = n // w
                if num_segments < 1:
                    continue

                rms_values = []
                for seg_idx in range(num_segments):
                    start = seg_idx * w
                    end = start + w
                    segment = profile[start:end]

                    # 선형 추세 제거
                    x = np.arange(w)
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    residual = segment - trend

                    rms = np.sqrt(np.mean(residual ** 2))
                    if rms > 1e-15:
                        rms_values.append(rms)

                if rms_values:
                    avg_rms = np.mean(rms_values)
                    if avg_rms > 1e-15:
                        log_windows.append(np.log(w))
                        log_flucts.append(np.log(avg_rms))

            if len(log_windows) < 3:
                return {
                    "hurst": 0.5,
                    "score": 0.0,
                    "detected": False,
                    "interpretation": "회귀 불충분",
                }

            # 4. log-log 선형 회귀: 기울기 = Hurst 지수
            log_w = np.array(log_windows)
            log_f = np.array(log_flucts)
            coeffs = np.polyfit(log_w, log_f, 1)
            hurst = float(coeffs[0])

            # 해석
            if hurst < 0.45:
                interpretation = "평균회귀"
            elif hurst <= 0.55:
                interpretation = "랜덤워크"
            elif hurst <= 0.7:
                interpretation = "추세 형성"
            else:
                interpretation = "강한 추세"

            score = max(0.0, min(1.0, (hurst - 0.5) / 0.2))
            detected = hurst > 0.6

            return {
                "hurst": round(hurst, 4),
                "score": round(score, 4),
                "detected": detected,
                "interpretation": interpretation,
            }
        except Exception as e:
            logger.warning("hurst_exponent 계산 실패: %s", e)
            return {
                "hurst": 0.5,
                "score": 0.0,
                "detected": False,
                "interpretation": "계산 오류",
            }

    # --- 4. Flickering (깜빡임) ----------------------------

    def flickering(self, prices: np.ndarray, resistance: float | None = None, lookback: int = 20) -> dict:
        """
        깜빡임: 저항선 반복 터치 + 되돌림 폭 감소 감지.

        Parameters:
            prices: 종가 배열
            resistance: 저항선 가격 (None이면 자동 탐지)
            lookback: 분석 기간

        Returns:
            {"touches": int, "avg_retracement": float, "retracement_trend": str,
             "score": 0.0~1.0, "detected": bool}
        """
        try:
            prices = np.asarray(prices, dtype=float)
            if len(prices) < lookback:
                return {
                    "touches": 0,
                    "avg_retracement": 0.0,
                    "retracement_trend": "데이터 부족",
                    "score": 0.0,
                    "detected": False,
                }

            recent = prices[-lookback:]

            # 저항선 자동 탐지
            if resistance is None:
                resistance = float(np.max(recent))

            if resistance <= 0:
                return {
                    "touches": 0,
                    "avg_retracement": 0.0,
                    "retracement_trend": "저항선 오류",
                    "score": 0.0,
                    "detected": False,
                }

            # 터치 감지: 종가가 resistance의 99% 이상
            threshold = resistance * 0.99
            touch_indices = []
            for i in range(len(recent)):
                if recent[i] >= threshold:
                    # 연속 터치는 하나로 취급 (직전 인덱스와 연속이 아닌 경우만)
                    if not touch_indices or i > touch_indices[-1] + 1:
                        touch_indices.append(i)
                    else:
                        # 연속 터치면 마지막 인덱스 갱신
                        touch_indices[-1] = i

            touches = len(touch_indices)

            # 되돌림 폭 계산: 각 터치 후 다음 터치(또는 끝)까지의 최저점
            retracements = []
            for idx, touch_i in enumerate(touch_indices):
                # 다음 터치까지 또는 데이터 끝까지의 구간
                end_i = touch_indices[idx + 1] if idx + 1 < len(touch_indices) else len(recent)
                if touch_i + 1 < end_i:
                    segment = recent[touch_i + 1:end_i]
                    low = float(np.min(segment))
                    touch_price = recent[touch_i]
                    if touch_price > 0:
                        retracement = (touch_price - low) / touch_price
                        retracements.append(retracement)

            avg_retracement = float(np.mean(retracements)) if retracements else 0.0

            # 되돌림 추세 판단
            if len(retracements) >= 2:
                first_half = np.mean(retracements[:len(retracements) // 2])
                second_half = np.mean(retracements[len(retracements) // 2:])
                if second_half < first_half * 0.8:
                    retracement_trend = "감소"
                elif second_half > first_half * 1.2:
                    retracement_trend = "증가"
                else:
                    retracement_trend = "유지"
            else:
                retracement_trend = "판단 불가"

            # 스코어 계산
            touch_score = min(1.0, touches / 5) * 0.5
            retr_score = max(0.0, (1 - avg_retracement / 0.03)) * 0.5 if avg_retracement > 0 else 0.25
            score = min(1.0, touch_score + retr_score)

            detected = touches >= 3 and retracement_trend == "감소"

            return {
                "touches": touches,
                "avg_retracement": round(avg_retracement, 4),
                "retracement_trend": retracement_trend,
                "score": round(score, 4),
                "detected": detected,
            }
        except Exception as e:
            logger.warning("flickering 계산 실패: %s", e)
            return {
                "touches": 0,
                "avg_retracement": 0.0,
                "retracement_trend": "계산 오류",
                "score": 0.0,
                "detected": False,
            }

    # --- 5. Asymmetric Fluctuation (비대칭 요동) -----------

    def asymmetric_fluctuation(self, returns: np.ndarray, window: int = 10) -> dict:
        """
        비대칭 요동: 상승일 평균 수익률 / 하락일 평균 수익률 비율.

        Parameters:
            returns: 일별 수익률 배열
            window: 분석 기간

        Returns:
            {"up_mean": float, "down_mean": float, "asymmetry_ratio": float,
             "direction": str, "score": 0.0~1.0, "detected": bool}
        """
        try:
            returns = np.asarray(returns, dtype=float)
            if len(returns) < window:
                return {
                    "up_mean": 0.0,
                    "down_mean": 0.0,
                    "asymmetry_ratio": 1.0,
                    "direction": "데이터 부족",
                    "score": 0.0,
                    "detected": False,
                }

            recent = returns[-window:]
            up_days = recent[recent > 0]
            down_days = recent[recent < 0]

            up_mean = float(np.mean(up_days)) if len(up_days) > 0 else 0.0
            down_mean = float(np.mean(np.abs(down_days))) if len(down_days) > 0 else 0.0

            # 비대칭 비율
            if down_mean > 1e-15:
                asymmetry_ratio = up_mean / down_mean
            elif up_mean > 1e-15:
                asymmetry_ratio = 2.0  # 하락일이 없으면 강한 상방
            else:
                asymmetry_ratio = 1.0  # 변동 없음

            # 방향 판단
            if asymmetry_ratio > 1.3:
                direction = "상방"
            elif asymmetry_ratio < 0.7:
                direction = "하방"
            else:
                direction = "대칭"

            # 스코어 (상방 기준)
            score = min(1.0, max(0.0, (asymmetry_ratio - 1.0) / 0.8))
            detected = asymmetry_ratio > 1.5 or asymmetry_ratio < 0.5

            return {
                "up_mean": round(up_mean, 6),
                "down_mean": round(down_mean, 6),
                "asymmetry_ratio": round(asymmetry_ratio, 4),
                "direction": direction,
                "score": round(score, 4),
                "detected": detected,
            }
        except Exception as e:
            logger.warning("asymmetric_fluctuation 계산 실패: %s", e)
            return {
                "up_mean": 0.0,
                "down_mean": 0.0,
                "asymmetry_ratio": 1.0,
                "direction": "계산 오류",
                "score": 0.0,
                "detected": False,
            }

    # --- 통합 분석 ------------------------------------------

    def analyze(self, prices: np.ndarray, returns: np.ndarray | None = None) -> dict:
        """
        5가지 전조 현상을 모두 실행하여 종합 결과 반환.

        Parameters:
            prices: 종가 배열 (최소 60일)
            returns: 일별 수익률 (None이면 prices에서 계산)

        Returns:
            통합 분석 결과 dict
        """
        prices = np.asarray(prices, dtype=float)

        if len(prices) < self.min_data_days:
            return self._empty_result(f"데이터 부족: {len(prices)}일 < {self.min_data_days}일")

        # returns 계산
        if returns is None:
            returns = np.diff(prices) / prices[:-1]
            # 0으로 나누기 방지
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        returns = np.asarray(returns, dtype=float)

        # 5가지 전조 개별 실행 (하나가 실패해도 나머지 계속)
        cs_result = self.critical_slowing(returns, window=self.critical_slowing_window)
        vov_result = self.vol_of_vol(prices, atr_window=self.atr_window, lookback=self.vov_lookback)
        hurst_result = self.hurst_exponent(
            prices, min_window=self.hurst_min_window, max_window=self.hurst_max_window,
        )
        flicker_result = self.flickering(prices, lookback=self.flickering_lookback)
        asym_result = self.asymmetric_fluctuation(returns, window=self.asymmetric_window)

        # detected 카운트
        results_map = {
            "critical_slowing": cs_result,
            "vol_of_vol": vov_result,
            "hurst": hurst_result,
            "flickering": flicker_result,
            "asymmetric": asym_result,
        }
        precursor_count = sum(1 for r in results_map.values() if r.get("detected", False))

        # composite score (가중 평균)
        composite_score = sum(
            self.WEIGHTS[key] * results_map[key].get("score", 0.0)
            for key in self.WEIGHTS
        )

        return {
            "critical_slowing": cs_result,
            "vol_of_vol": vov_result,
            "hurst": hurst_result,
            "flickering": flicker_result,
            "asymmetric": asym_result,
            "precursor_count": precursor_count,
            "composite_score": round(composite_score, 4),
            "phase_transition_imminent": precursor_count >= 3,
        }

    # --- 프롬프트 텍스트 ------------------------------------

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API 입력용 텍스트 변환"""
        lines = ["[상전이 분석]"]

        # 데이터 부족 등 빈 결과 처리
        if "reason" in result:
            lines.append(f"  {result['reason']}")
            return "\n".join(lines)

        count = result.get("precursor_count", 0)
        lines.append(f"  전조 감지: {count}/5개")

        # Vol of Vol
        vov = result.get("vol_of_vol", {})
        vov_ratio = vov.get("vov_ratio", 0)
        vov_detected = "!" if vov.get("detected") else ""
        lines.append(f"  Vol of Vol: {vov_ratio:.1f}배 (평소 대비) -> 내부 균형 다툼{vov_detected}")

        # Hurst
        hurst = result.get("hurst", {})
        h_val = hurst.get("hurst", 0.5)
        h_interp = hurst.get("interpretation", "?")
        hurst_detected = "!" if hurst.get("detected") else ""
        lines.append(f"  허스트 지수: {h_val:.2f} -> {h_interp}{hurst_detected}")

        # Asymmetric
        asym = result.get("asymmetric", {})
        a_ratio = asym.get("asymmetry_ratio", 1.0)
        a_dir = asym.get("direction", "?")
        asym_detected = "!" if asym.get("detected") else ""
        lines.append(f"  비대칭 요동: {a_ratio:.2f} ({a_dir}){asym_detected}")

        # Flickering
        flicker = result.get("flickering", {})
        f_touches = flicker.get("touches", 0)
        f_retr = flicker.get("avg_retracement", 0)
        f_trend = flicker.get("retracement_trend", "?")
        flicker_detected = "!" if flicker.get("detected") else ""
        lines.append(
            f"  깜빡임: {f_touches}회 터치, "
            f"되돌림 {f_retr:.1%} ({f_trend}){flicker_detected}"
        )

        # Critical Slowing
        cs = result.get("critical_slowing", {})
        cs_autocorr = cs.get("autocorr", 0)
        cs_detected = "!" if cs.get("detected") else ""
        lines.append(f"  임계 감속: autocorr={cs_autocorr:.3f}{cs_detected}")

        # 종합
        composite = result.get("composite_score", 0)
        imminent = result.get("phase_transition_imminent", False)
        status = "상전이 임박" if imminent else "정상 범위"
        lines.append(f"  종합: {status} (composite: {composite:.2f})")

        return "\n".join(lines)

    # --- 빈 결과 --------------------------------------------

    @staticmethod
    def _empty_result(reason: str) -> dict:
        """데이터 부족 등 빈 결과 반환"""
        empty_sub = {"score": 0.0, "detected": False}
        return {
            "critical_slowing": {"autocorr": 0.0, **empty_sub},
            "vol_of_vol": {"vov": 0.0, "vov_ratio": 0.0, **empty_sub},
            "hurst": {"hurst": 0.5, "interpretation": reason, **empty_sub},
            "flickering": {
                "touches": 0,
                "avg_retracement": 0.0,
                "retracement_trend": reason,
                **empty_sub,
            },
            "asymmetric": {
                "up_mean": 0.0,
                "down_mean": 0.0,
                "asymmetry_ratio": 1.0,
                "direction": reason,
                **empty_sub,
            },
            "precursor_count": 0,
            "composite_score": 0.0,
            "phase_transition_imminent": False,
            "reason": reason,
        }
