"""
v6.2 Config 범위 검증 유틸리티

설정 파일의 파라미터 범위를 검증하여 운영 안정성을 보장.
잘못된 파라미터 설정 시 경고 메시지를 반환.
"""

import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """v6.0 config 파라미터 범위 검증기"""

    RULES = {
        "martin_momentum": {
            "n_fast": (1, 50, int),
            "n_slow": (10, 200, int),
            "epsilon": (0.0, 5.0, float),
            "sigmoid_k": (0.1, 20.0, float),
            "min_confidence": (0.0, 1.0, float),
        },
        "extreme_volatility": {
            "atr_ratio_threshold": (0.5, 10.0, float),
            "vol_ratio_threshold": (1.0, 20.0, float),
            "daily_range_threshold": (1.0, 50.0, float),
        },
        "wavelsformer.risk_normalization": {
            "target_daily_vol": (0.001, 0.1, float),
            "max_scale": (1.0, 5.0, float),
            "min_scale": (0.1, 1.0, float),
        },
    }

    @classmethod
    def validate(cls, config: dict) -> list[str]:
        """
        검증 실패 시 경고 메시지 리스트 반환.

        빈 리스트 = OK (모든 파라미터 정상)
        """
        warnings = []

        for section, rules in cls.RULES.items():
            # 점(.)으로 중첩 접근
            cfg = config
            for key in section.split("."):
                cfg = cfg.get(key, {}) if isinstance(cfg, dict) else {}

            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                continue

            for param, (min_v, max_v, _expected_type) in rules.items():
                val = cfg.get(param)
                if val is None:
                    continue
                if not isinstance(val, (int, float)):
                    warnings.append(
                        f"{section}.{param}: 타입 오류 ({type(val).__name__})"
                    )
                elif not (min_v <= val <= max_v):
                    warnings.append(
                        f"{section}.{param}={val}: 범위 [{min_v}, {max_v}] 초과"
                    )

        # max_scale > min_scale 교차 검증
        risk = config.get("wavelsformer", {}).get("risk_normalization", {})
        if isinstance(risk, dict) and risk.get("enabled"):
            max_s = risk.get("max_scale", 2.0)
            min_s = risk.get("min_scale", 0.3)
            if isinstance(max_s, (int, float)) and isinstance(min_s, (int, float)):
                if max_s <= min_s:
                    warnings.append("wavelsformer: max_scale <= min_scale")

        return warnings
