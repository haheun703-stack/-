# -*- coding: utf-8 -*-
"""risk/ + kill_switch/ 안전선 정적 검사 — 실주문 경로·운영 인프라 접촉 0 보증.

소스 텍스트만 읽는 read-only 검사. 네트워크 0. 운영 파일 무접촉.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re

import pytest

ROOT = Path(__file__).resolve().parent.parent

# 주문/배포 인프라 접촉을 드러내는 금지 토큰 — 하나라도 있으면 게이트가 '판정만 한다' 계약 위반
FORBIDDEN_TOKENS = [
    "mojito",
    "kis_order",
    "buy_limit",
    "sell_limit",
    "create_market_",
    "create_limit_",
    "systemctl",
    "scheduler.service",
    "src.adapters",
]


def _source_files(subdir: str) -> list[Path]:
    files = sorted((ROOT / subdir).glob("*.py"))
    assert files, f"{subdir}/*.py가 없으면 검사 자체가 무의미 — 경로 확인 필요"
    return files


RISK_FILES = _source_files("risk")
KILL_SWITCH_FILES = _source_files("kill_switch")


@pytest.mark.parametrize(
    "src_file",
    RISK_FILES + KILL_SWITCH_FILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_no_forbidden_tokens(src_file):
    # 사고 방지: 리스크/킬스위치 모듈이 실주문·배포 인프라를 직접 건드려 '판정 전용' 계약 파괴
    text = src_file.read_text(encoding="utf-8")
    found = [tok for tok in FORBIDDEN_TOKENS if tok in text]
    assert found == [], f"{src_file.name}에 금지 토큰 발견: {found}"


def test_kill_switch_does_not_import_risk():
    # 사고 방지: 킬스위치가 risk를 import하면 §5 의도적 격리 붕괴 — 리스크 엔진 버그가 최후 방어선까지 전염
    text = (ROOT / "kill_switch" / "monitor.py").read_text(encoding="utf-8")
    bad = re.findall(r"^\s*(?:import\s+risk\b|from\s+risk\b)", text, flags=re.MULTILINE)
    assert bad == [], f"kill_switch/monitor.py가 risk를 import: {bad}"


@pytest.mark.parametrize("src_file", RISK_FILES, ids=lambda p: p.name)
def test_risk_does_not_import_kill_switch(src_file):
    # 사고 방지: risk가 킬스위치를 import하면 역방향 결합 — 게이트 장애가 킬스위치 동작에 간섭
    text = src_file.read_text(encoding="utf-8")
    bad = re.findall(
        r"^\s*(?:import\s+kill_switch\b|from\s+kill_switch\b)", text, flags=re.MULTILINE
    )
    assert bad == [], f"{src_file.name}이 kill_switch를 import: {bad}"
