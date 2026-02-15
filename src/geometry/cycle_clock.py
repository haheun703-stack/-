"""
⑧ Cycle Clock — 3D 구조: 3주파수 사이클 위치 추적

수학적 근거: sin(θ), sin(2θ), sin(θ/2)
  → 같은 시점에서 다른 주파수의 위상이 다름
  → 장기/중기/단기 사이클의 현재 위치를 "시계"로 표현

핵심 기능:
  1. 밴드패스 필터로 3개 주파수대 분리
  2. 힐베르트 변환으로 즉시 위상 추출
  3. 시계 위치 (0~12시) + 위상 정렬도 계산

의존성: parquet 일봉 close 데이터 (최소 120일)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CycleClock:
    """3주파수 사이클 위치 추적기"""

    # 주파수 대역 정의 (일 단위)
    DEFAULT_BANDS = {
        "long": (40, 120),   # 장기: 40~120일 주기
        "mid": (10, 40),     # 중기: 10~40일 주기
        "short": (3, 10),    # 단기: 3~10일 주기
    }

    # 시계 위치별 해석
    CLOCK_INTERPRET = {
        (5, 7): "바닥~반등 (매수 구간)",
        (7, 9): "상승 초중반 (보유)",
        (9, 11): "상승 후반 (이익실현 준비)",
        (11, 13): "고점 (매도 구간)",      # 13 = 1시 (wrap)
        (1, 3): "하락 초기 (관망)",
        (3, 5): "하락 후반 (바닥 탐색)",
    }

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("cycle", {})
        self.bands = {
            "long": tuple(cfg.get("long_band", self.DEFAULT_BANDS["long"])),
            "mid": tuple(cfg.get("mid_band", self.DEFAULT_BANDS["mid"])),
            "short": tuple(cfg.get("short_band", self.DEFAULT_BANDS["short"])),
        }
        self.min_data_days = cfg.get("min_data_days", 120)

    # ─── 밴드패스 필터 ──────────────────────────

    @staticmethod
    def bandpass_filter(prices: np.ndarray, low_period: int, high_period: int) -> np.ndarray:
        """
        간단한 밴드패스 필터: 두 이동평균의 차이로 특정 주파수대 추출.

        low_period~high_period 사이의 주기 성분을 추출.
        (scipy 없이도 동작하는 근사 구현)
        """
        if len(prices) < high_period:
            return np.zeros_like(prices)

        # 느린 MA (장주기 트렌드 제거) - 고주파 통과
        slow_ma = np.convolve(prices, np.ones(high_period) / high_period, mode="same")
        detrended = prices - slow_ma

        # 빠른 MA (노이즈 제거) - 저주파 통과
        if low_period > 1:
            result = np.convolve(detrended, np.ones(low_period) / low_period, mode="same")
        else:
            result = detrended

        return result

    # ─── 힐베르트 변환 (순수 numpy) ──────────────

    @staticmethod
    def hilbert_phase(signal: np.ndarray) -> np.ndarray:
        """
        힐베르트 변환을 FFT로 구현하여 순시 위상(instantaneous phase) 추출.

        scipy.signal.hilbert와 동일한 결과.
        """
        n = len(signal)
        if n == 0:
            return np.array([])

        # FFT
        freq = np.fft.fft(signal)

        # 힐베르트 변환 필터: 양의 주파수만 2배, 음의 주파수 제거
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = 1
            h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2

        analytic = np.fft.ifft(freq * h)
        phase = np.angle(analytic)  # -π ~ +π
        return phase.real

    # ─── 위상 → 시계 변환 ────────────────────────

    @staticmethod
    def phase_to_clock(phase_rad: float) -> float:
        """
        라디안 위상 (-π ~ +π) → 시계 위치 (0~12).
        6시 = 바닥 (위상 -π/0), 12시 = 고점 (위상 +π/-π)
        """
        # -π~+π → 0~2π
        norm = (phase_rad + np.pi) / (2 * np.pi)  # 0~1
        # 0~12시 (6시가 바닥이 되도록 오프셋)
        clock = (norm * 12 + 6) % 12
        return clock

    @staticmethod
    def interpret_clock(clock_hour: float) -> str:
        """시계 위치 → 한글 해석"""
        h = clock_hour % 12
        if 5 <= h < 7:
            return "바닥~반등 (매수 구간)"
        elif 7 <= h < 9:
            return "상승 초중반 (보유)"
        elif 9 <= h < 11:
            return "상승 후반 (이익실현 준비)"
        elif h >= 11 or h < 1:
            return "고점 (매도 구간)"
        elif 1 <= h < 3:
            return "하락 초기 (관망)"
        else:  # 3 <= h < 5
            return "하락 후반 (바닥 탐색)"

    # ─── 위상 정렬도 ─────────────────────────────

    @staticmethod
    def phase_alignment(phase_a: float, phase_b: float) -> float:
        """
        두 위상의 정렬도 = cos(위상차).
        +1 = 완벽 정렬 (같은 방향), -1 = 완벽 역방향, 0 = 직교
        """
        return float(np.cos(phase_a - phase_b))

    # ─── 메인 분석 ───────────────────────────────

    def get_clock_position(self, prices: np.ndarray | list) -> dict:
        """
        현재 3주파수 시계 위치 + 정렬도 계산.

        Parameters:
            prices: 종가 배열 (최소 120일, 최신이 마지막)

        Returns:
            {
                "long": {"clock": 8.2, "phase_rad": -0.5, "interpretation": "..."},
                "mid": {"clock": 5.1, ...},
                "short": {"clock": 7.0, ...},
                "alignment_long_mid": 0.85,
                "alignment_long_short": 0.72,
                "overall_alignment": 0.78,
                "summary": "장기 상승 + 중기 바닥 = 매수 최적 접근"
            }
        """
        prices = np.asarray(prices, dtype=float)

        if len(prices) < self.min_data_days:
            return self._empty_result(f"데이터 부족: {len(prices)}일 < {self.min_data_days}일")

        # 가격 정규화 (로그)
        log_prices = np.log(prices + 1e-10)

        positions = {}
        for name, (low_p, high_p) in self.bands.items():
            # 밴드패스 필터
            filtered = self.bandpass_filter(log_prices, low_p, high_p)

            # 힐베르트 변환 → 위상
            phases = self.hilbert_phase(filtered)
            current_phase = float(phases[-1])

            # 시계 변환
            clock = self.phase_to_clock(current_phase)
            interp = self.interpret_clock(clock)

            positions[name] = {
                "clock": round(clock, 1),
                "phase_rad": round(current_phase, 3),
                "interpretation": interp,
            }

        # 위상 정렬도
        align_lm = self.phase_alignment(
            positions["long"]["phase_rad"],
            positions["mid"]["phase_rad"],
        )
        align_ls = self.phase_alignment(
            positions["long"]["phase_rad"],
            positions["short"]["phase_rad"],
        )
        overall = (align_lm + align_ls) / 2

        # 종합 해석
        summary = self._build_summary(positions, align_lm)

        return {
            "long": positions["long"],
            "mid": positions["mid"],
            "short": positions["short"],
            "alignment_long_mid": round(align_lm, 2),
            "alignment_long_short": round(align_ls, 2),
            "overall_alignment": round(overall, 2),
            "summary": summary,
        }

    # ─── 내부 헬퍼 ───────────────────────────────

    def _build_summary(self, positions: dict, alignment: float) -> str:
        """종합 해석 문자열 생성"""
        long_clock = positions["long"]["clock"]
        mid_clock = positions["mid"]["clock"]

        # 장기 방향
        if 5 <= long_clock < 9:
            long_dir = "장기 상승"
        elif 9 <= long_clock < 11 or long_clock >= 11:
            long_dir = "장기 고점"
        else:
            long_dir = "장기 하락"

        # 중기 방향
        if 4 <= mid_clock < 7:
            mid_dir = "중기 바닥"
        elif 7 <= mid_clock < 10:
            mid_dir = "중기 상승"
        else:
            mid_dir = "중기 하락"

        # 정렬도
        if alignment > 0.5:
            align_text = "같은 방향 (강한 추세)"
        elif alignment > -0.3:
            align_text = "중립 (과도기)"
        else:
            align_text = "반대 방향 (조정/전환)"

        return f"{long_dir} + {mid_dir} = {align_text}"

    @staticmethod
    def _empty_result(reason: str) -> dict:
        empty = {"clock": 0, "phase_rad": 0, "interpretation": reason}
        return {
            "long": empty,
            "mid": empty,
            "short": empty,
            "alignment_long_mid": 0,
            "alignment_long_short": 0,
            "overall_alignment": 0,
            "summary": reason,
        }

    # ─── 프롬프트 텍스트 ─────────────────────────

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API 입력용 텍스트"""
        lines = ["[사이클 시계]"]
        for name, label in [("long", "장기(40~120일)"), ("mid", "중기(10~40일)"), ("short", "단기(3~10일)")]:
            pos = result[name]
            lines.append(f"  {label}: {pos['clock']:.0f}시 ({pos['interpretation']})")

        lines.append(f"  위상 정렬: {result['overall_alignment']:.2f}")
        lines.append(f"  해석: {result['summary']}")
        return "\n".join(lines)
