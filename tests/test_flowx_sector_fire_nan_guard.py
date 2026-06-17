"""sector_fire 업로드 NaN/Inf 가드 — 6/16 'Out of range float values are not JSON
compliant' 회귀 방지.

서버 cron_20260616.log 실측: scan/JSON 정상, upload_sector_fire만 FAIL =
build_sector_fire_rows가 NaN을 통과시켜 Supabase JSON upsert 거부.
"""
import json
import math

import pytest

from src.adapters import flowx_uploader as fu
from src.adapters.flowx_uploader import _drop_nonfinite_floats, build_sector_fire_rows


def test_drop_nonfinite_replaces_nan_inf_with_zero():
    row = {
        "sector": "반도체",            # str 불변
        "fire_score": float("nan"),    # NaN → 0.0
        "rsi_avg": float("inf"),       # Inf → 0.0
        "ma20_avg_dev": float("-inf"), # -Inf → 0.0
        "flow_score": 12.3,            # 정상 float 불변
        "s1_score": 5,                 # int 불변
        "etf_code": None,              # None 불변
    }
    out = _drop_nonfinite_floats(row)
    assert out["fire_score"] == 0.0
    assert out["rsi_avg"] == 0.0
    assert out["ma20_avg_dev"] == 0.0
    assert out["flow_score"] == 12.3
    assert out["s1_score"] == 5
    assert out["sector"] == "반도체"
    assert out["etf_code"] is None
    # 결과는 JSON 엄격 직렬화(allow_nan=False) 통과 = Supabase upsert 가능
    json.dumps(out, allow_nan=False)


def test_normal_row_unchanged():
    row = {"a": 1.0, "b": "x", "c": None, "d": 0.0, "e": -3.5}
    assert _drop_nonfinite_floats(row) == row


def test_build_sector_fire_rows_sanitizes_nan(tmp_path, monkeypatch):
    """NaN 섞인 sector_fire JSON → 산출 행 전부 유한(JSON 엄격 직렬화 통과)."""
    date_str = "2026-06-16"
    monkeypatch.setattr(fu, "DATA_DIR", tmp_path)
    payload = {
        "sectors": [
            {  # 정상 섹터
                "sector": "반도체", "fire_score": 88.0, "fire_grade": "A",
                "rsi_avg": 61.2, "vol_ratio_avg": 1.4,
            },
            {  # 빈 섹터 — 평균/비율이 NaN으로 산출된 케이스(6/16 재현)
                "sector": "빈섹터", "fire_score": float("nan"),
                "rsi_avg": float("nan"), "ma20_avg_dev": float("nan"),
                "vol_ratio_avg": float("nan"),
            },
        ]
    }
    # allow_nan=True로 NaN 포함 JSON을 디스크에 기록(원본 버그 재현)
    (tmp_path / "sector_fire_20260616.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    rows = build_sector_fire_rows(date_str)
    assert len(rows) == 2
    for r in rows:
        for v in r.values():
            assert not (isinstance(v, float) and not math.isfinite(v)), f"비유한 잔존: {r}"
    # 전체 배치 JSON 엄격 직렬화 통과 = upsert FAIL 재발 없음
    json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
