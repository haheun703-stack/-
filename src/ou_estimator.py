"""
OU (Ornstein-Uhlenbeck) 프로세스 파라미터 롤링 추정

논문 기반:
  - OU+BB 논문: theta>0.03, half-life 5~200일, 2*sigma 스톱로스
  - OUbv 논문: 모멘텀/평균회귀 phi(t) 전환점
  - 최적 차익거래 논문: 스프레드 거리 기반 포지션 사이징

AR(1) 근사: x_t = a + b*x_{t-1} + eps
  → kappa = -ln(b), mu = a/(1-b), sigma, half_life = ln(2)/kappa
"""

import numpy as np
import pandas as pd


class OUEstimator:
    """OU 프로세스 파라미터 롤링 추정기"""

    def __init__(self, window: int = 60):
        self.window = window

    def estimate_rolling(self, close: pd.Series) -> pd.DataFrame:
        """
        종가 시리즈에서 OU 파라미터를 롤링 추정.

        Returns:
            DataFrame with columns: kappa, mu, sigma, half_life, ou_z, snr
        """
        log_price = np.log(close).dropna()
        n = len(log_price)
        w = self.window

        out = pd.DataFrame(
            index=log_price.index,
            columns=["kappa", "mu", "sigma", "half_life", "ou_z", "snr"],
            dtype=float,
        )

        for i in range(w, n):
            xs = log_price.iloc[i - w : i]
            x_lag = xs.iloc[:-1].values
            x_cur = xs.iloc[1:].values

            # OLS: x_cur = a + b * x_lag
            var_lag = np.var(x_lag)
            if var_lag == 0:
                continue
            b = np.cov(x_lag, x_cur)[0, 1] / var_lag
            a = np.mean(x_cur) - b * np.mean(x_lag)

            # OU 변환: b = exp(-kappa*dt), dt=1
            if b <= 0 or b >= 0.9999:
                continue

            kappa = -np.log(b)
            mu = a / (1 - b)

            resid = x_cur - (a + b * x_lag)
            resid_std = np.std(resid)
            if resid_std == 0:
                continue

            sigma = resid_std * np.sqrt(2 * kappa / (1 - b**2))
            half_life = np.log(2) / kappa

            # 현재 log price의 OU z-score
            current_x = log_price.iloc[i]
            ou_z = (current_x - mu) / (sigma / np.sqrt(2 * kappa)) if sigma > 0 else 0

            # SNR (Signal-to-Noise Ratio)
            snr = abs(kappa * (current_x - mu)) / sigma if sigma > 0 else 0

            out.iloc[i] = [kappa, mu, sigma, half_life, ou_z, snr]

        return out.astype(float)

    @staticmethod
    def check_ou_gate(
        row: pd.Series,
        z_entry: float = -1.2,
        hl_min: float = 2,
        hl_max: float = 25,
        snr_min: float = 0.15,
    ) -> tuple[bool, str]:
        """
        OU 게이트 체크.

        Returns:
            (passed, block_reason)
        """
        hl = row.get("half_life", np.nan)
        z = row.get("ou_z", np.nan)
        snr = row.get("snr", np.nan)

        if pd.isna(hl) or pd.isna(z) or pd.isna(snr):
            return False, "ou_nan"

        if not (hl_min <= hl <= hl_max):
            return False, "half_life"

        if z > z_entry:
            return False, "z_score"

        if snr < snr_min:
            return False, "snr"

        return True, ""
