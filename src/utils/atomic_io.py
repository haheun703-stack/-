"""원자적 파일 쓰기 유틸리티.

스캐너가 JSON을 쓰는 도중 다른 프로세스(scan_tomorrow_picks 등)가
부분 기록 상태를 읽어 파싱 에러나 잘못된 시그널이 발생할 위험을 차단.

전략: tempfile에 먼저 쓴 후 os.replace로 원자적 rename.
- POSIX: rename(2) 자체가 atomic
- Windows: os.replace는 MoveFileExW(MOVEFILE_REPLACE_EXISTING)로 atomic

같은 디렉토리에 tempfile을 만들어야 cross-device link 에러를 피할 수 있음.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """문자열을 원자적으로 파일에 쓰기.

    Args:
        path: 대상 파일 경로 (절대/상대)
        text: 기록할 문자열
        encoding: 인코딩 (기본 utf-8)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 같은 디렉토리에 임시 파일 생성 (rename atomic 보장)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())  # 디스크 플러시 (크래시 대비)
            except OSError:
                pass  # 일부 FS는 fsync 미지원 (메모리 FS 등)
        os.replace(tmp_path, path)
    except Exception:
        # 실패 시 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path | str,
    data: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """dict/list를 원자적으로 JSON 파일에 쓰기.

    Args:
        path: 대상 파일 경로
        data: JSON 직렬화 가능한 객체
        ensure_ascii: 비-ASCII 문자 이스케이프 여부 (기본 False, 한글 그대로)
        indent: JSON 들여쓰기 (기본 2)
    """
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_text(path, text)
