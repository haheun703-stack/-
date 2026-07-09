# -*- coding: utf-8 -*-
"""환경/설정 로더 — .env(수동 파싱, python-dotenv 불필요) + YAML config."""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("insight_signals.env")

DEFAULT_CONFIG = {
    "keywords": ["사상 최대", "역대 최대", "사상 최고", "최대 실적", "게임 체인저", "게임체인저"],
    "negative_keywords": ["적자", "급락", "하한가", "리콜", "소송"],
    "rss_feeds": [
        ["한경증권", "https://www.hankyung.com/feed/finance"],
        ["한경경제", "https://www.hankyung.com/feed/economy"],
        ["한경IT", "https://www.hankyung.com/feed/it"],
    ],
    "use_naver_mainnews": True,
    "naver_pages": 2,
    "name_blacklist": ["한국", "서울", "대상", "동양", "한일", "경남", "무학", "금강", "전방", "선진", "화성", "국보", "신원"],
    "min_name_len": 2,
    "dart": {"days_back": 3},
    "flow": {
        "days": 3,
        "kis_base_url": "https://openapi.koreainvestment.com:9443",
        # 기존 봇의 KIS 토큰 캐시 파일 경로 (비우면 자체 캐시 사용).
        # 발급 횟수 제한 충돌 방지를 위해 기존 봇과 공유 권장.
        "kis_token_cache": "",
    },
    "env_names": {
        "dart_key": "DART_API_KEY",
        "kis_app_key": "KIS_APP_KEY",
        "kis_app_secret": "KIS_APP_SECRET",
    },
    "picks": {
        "top_n": 5,
        "min_score": 0.25,
        "weights": {"news_keyword": 0.3, "dart_insider": 0.45, "flow_contrarian": 0.25},
    },
    "paths": {
        "data_dir": "data/insight_signals",
        "log_dir": "logs",
    },
}


def project_root() -> str:
    """이 패키지가 프로젝트 루트/insight_signals/ 에 있다고 가정."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_dotenv_manual(root: str) -> None:
    """루트의 .env를 읽어 os.environ에 주입 (이미 있는 키는 유지)."""
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError as e:
        log.warning(".env 로드 실패: %s", e)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(root: str) -> dict:
    """config/insight_signals.yaml 이 있으면 기본값 위에 덮어씀."""
    cfg_path = os.path.join(root, "config", "insight_signals.yaml")
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(cfg_path):
        try:
            import yaml  # 프로젝트가 settings.yaml을 쓰므로 PyYAML 존재 가정

            with open(cfg_path, encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, user_cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("config 로드 실패(기본값 사용): %s", e)
    return cfg


def setup_logging(root: str, name: str) -> None:
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
