"""Upload the paper-only strategy scoreboard batch to the FLOWX API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ★8/29: `load_dotenv()` 누락으로 첫 업로드가 `FLOWX_SCOREBOARD_TOKEN is required`로
#   죽었다. `.env`에는 값이 있었는데 스크립트가 읽지 않았다.
#   CLAUDE.md가 명시한 함정이다 — "새 스크립트 load_dotenv 필수(누락=수급 0행)".
#   cron은 셸 환경변수를 물려주지 않으므로 `.env`를 직접 읽어야 한다.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_BATCH = ROOT / "data" / "flowx_public" / "strategy_validation_latest.json"
# ★8/29: apex(flowx.kr)는 307로 www에 리다이렉트되고 그 과정에서 Authorization이
#   소실된다(정보봇 실측). Vercel Domains 실물도 www.flowx.kr만 Production이다.
DEFAULT_ENDPOINT = "https://www.flowx.kr/api/strategy-scoreboard"
MAX_BODY_BYTES = 256 * 1024


def upload_batch(
    batch_path: Path,
    *,
    token: str,
    endpoint: str = DEFAULT_ENDPOINT,
    post: Callable[..., object] = requests.post,
) -> dict:
    if not token.strip():
        raise ValueError("FLOWX_SCOREBOARD_TOKEN is required")
    if not endpoint.startswith("https://"):
        raise ValueError("FLOWX scoreboard endpoint must use HTTPS")

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("schema_version") != "1.0" or batch.get("producer") != "quant-bot":
        raise ValueError("unexpected FLOWX strategy scoreboard batch identity")
    results = batch.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("FLOWX strategy scoreboard batch has no results")

    body = json.dumps(batch, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("FLOWX strategy scoreboard batch exceeds 256KiB")

    # ★★8/29(정보봇 제보): apex(`flowx.kr`)로 보내면 **307로 www에 리다이렉트**되고
    #   `requests`가 그것을 따라갈 때 **Authorization 헤더가 떨어진다**(정보봇 실측 재현).
    #   토큰도 서버도 맞는데 영원히 401이고, 로그에는 원인이 안 보인다.
    #   정보봇도 `upload_flowx_stock_evidence.py`에서 같은 자리에 걸려 있었다.
    #   → ⑴엔드포인트를 www로 고정 ⑵리다이렉트를 **따라가지 않고 실패로 끊는다**.
    #     따라가면 헤더를 잃고 401로 위장되는데, 그건 이 스크립트의 fail-closed 원칙에 어긋난다.
    response = post(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
        allow_redirects=False,
    )
    if 300 <= getattr(response, "status_code", 0) < 400:
        loc = ""
        try:
            loc = response.headers.get("Location", "")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"FLOWX scoreboard endpoint redirected ({response.status_code} → {loc}). "
            "리다이렉트를 따라가면 Authorization 헤더가 소실돼 401로 위장된다. "
            f"엔드포인트를 리다이렉트 없는 주소로 지정할 것 (현재: {endpoint})"
        )
    response.raise_for_status()
    payload = response.json()
    accepted = payload.get("data", {}).get("accepted")
    run_id = payload.get("data", {}).get("runId")
    if accepted != len(results) or run_id != batch.get("run_id"):
        raise RuntimeError("FLOWX scoreboard receipt does not match the submitted batch")
    return {"accepted": accepted, "run_id": run_id, "producer": "quant-bot"}


def verify_live(endpoint: str = DEFAULT_ENDPOINT,
                get: Callable[..., object] = requests.get) -> dict:
    """업로드 후 GET으로 **실제 화면 값**을 확인한다.

    ★★8/29: 이 검사가 없어서 하루를 썼다. GET 응답에는 처음부터
      `meta.fallbackReason: "table_unavailable"` · `liveRowCount: 0`이 들어 있었는데,
      우리는 `rows`만 파싱하고 `meta`를 읽지 않았다. 그래서
      "테이블이 없어 스냅샷으로 폴백 중"이라는 답을 눈앞에 두고
      "전달 경로가 없다"로 진단했다(웹봇이 서버 로그 없이 잡아 알려줬다).

      → POST가 200이어도 **화면이 스냅샷이면 성공이 아니다**. 그것까지 확인한다.
        `fallbackReason`이 있거나 `source`가 live가 아니면 실패로 끊는다.
    """
    r = get(endpoint, timeout=30, allow_redirects=False)
    r.raise_for_status()
    payload = r.json()
    meta = payload.get("meta") or {}
    reason = meta.get("fallbackReason")
    source = meta.get("source")
    live = meta.get("liveRowCount")
    if reason:
        raise RuntimeError(
            f"업로드는 200이지만 화면은 폴백 상태다 — fallbackReason={reason!r} "
            f"source={source!r} liveRowCount={live!r}. 서버가 실제 테이블을 못 읽고 있다."
        )
    if source != "supabase_live":
        raise RuntimeError(f"화면 소스가 live가 아니다 — source={source!r} liveRowCount={live!r}")
    return {"source": source, "liveRowCount": live,
            "latestDataAsOf": meta.get("latestDataAsOf")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload FLOWX paper strategy scoreboard batch")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--no-verify", action="store_true",
                        help="업로드 후 GET 확인 생략(디버깅용)")
    args = parser.parse_args()
    receipt = upload_batch(
        args.batch,
        token=os.getenv("FLOWX_SCOREBOARD_TOKEN", ""),
    )
    if not args.no_verify:
        receipt["live"] = verify_live()
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
