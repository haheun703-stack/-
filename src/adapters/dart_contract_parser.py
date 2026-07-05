"""DART '단일판매ㆍ공급계약체결' 공시 본문 파서 — 계약금액/매출대비/기간 추출.

미래가치 엔진 v1 O축(수주 모멘텀)의 핵심 입력. 기존 dart_event_signal은 공시
"제목"만 분류했고, 본 모듈이 document.xml 원문에서 정량 필드를 뽑는다.

서식 실측(7/4, 파라텍 20260703900836·현대로템 20260703800009):
  - 본문은 (라벨 셀, 값 셀) 1:1 교대 시퀀스.
    "계약금액 총액(원)"→22,400,000,000 / "최근 매출액(원)"→... / "매출액 대비(%)"→12.88
    "3. 계약상대방"→텍스트 / "시작일"→날짜 / "종료일"→날짜 / "8. 계약(수주)일자"→날짜
  - ★[기재정정]은 앞부분=정정 요약(정정전/정정후), **뒷부분=정정 반영 전체 본문**.
    → 라벨의 "마지막" 발생 위치를 쓰면 정정후 본문을 자연히 취득(정정전 오염 없음).
  - 계약상대방 블록의 "- 최근 매출액(원)"(상대방 매출)은 접두 "-" 라벨이라
    정확매칭(^...$)으로 배제.
실패 필드는 None(graceful) — 개별 공시 파싱 실패가 파이프라인을 죽이지 않는다.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

# 셀 텍스트 추출: <span ...>텍스트</span> 단위 (DART 서식 XML은 HTML-ish 테이블)
_SPAN_RE = re.compile(r"<span[^>]*>(.*?)</span>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_AMOUNT_RE = re.compile(r"^-?[\d,]{4,}$")           # 콤마 금액(4자리+) — "2." 라벨 오인 차단
_RATIO_RE = re.compile(r"^-?\d{1,5}(?:\.\d+)?$")    # 소수 비율
_DATE_RE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")

# 라벨 → (정확매칭 패턴, 값 종류)
_FIELD_PATTERNS = {
    "contract_amount": (r"^계약금액\s*총액\s*\(원\)$|^계약금액\s*\(원\)$", "amount"),
    "contract_amount_fallback": (r"^확정\s*계약금액$", "amount"),
    "recent_revenue": (r"^최근\s*매출액\s*\(원\)$|^최근매출액\s*\(원\)$", "amount"),
    "revenue_ratio_pct": (r"^매출액\s*대비\s*\(%\)$|^매출액대비\s*\(%\)$", "ratio"),
    "counterparty": (r"^3\.\s*계약상대(방)?$", "text"),
    "period_start": (r"^시작일$", "date"),
    "period_end": (r"^종료일$", "date"),
    "contract_date": (r"^8\.\s*계약\(수주\)일자$", "date"),
}


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text.replace("&nbsp;", " ")).strip()


def _cells(xml_text: str) -> list[str]:
    """문서 내 셀 텍스트 시퀀스 (서식 순서 보존, 빈 셀 제외)."""
    return [c for c in (_clean(m.group(1)) for m in _SPAN_RE.finditer(xml_text)) if c]


def _parse_value(cell: str, kind: str):
    cell = cell.strip()
    if cell in ("-", "", "미해당", "해당없음"):
        return None
    if kind == "amount":
        if _AMOUNT_RE.match(cell.replace(" ", "")):
            try:
                return float(cell.replace(",", "").replace(" ", ""))
            except ValueError:
                return None
        return None
    if kind == "ratio":
        if _RATIO_RE.match(cell.replace(",", "").replace(" ", "")):
            try:
                v = float(cell.replace(",", "").replace(" ", ""))
                return v if 0 <= v <= 10000 else None
            except ValueError:
                return None
        return None
    if kind == "date":
        m = _DATE_RE.search(cell)
        if not m:
            return None
        try:
            return datetime(int(m.group(1)), int(m.group(2)),
                            int(m.group(3))).strftime("%Y-%m-%d")
        except ValueError:
            return None
    # text: 라벨성 셀(항목 번호/불릿 접두) 배제
    if re.match(r"^\d+\.\s|^-\s", cell):
        return None
    return cell[:60]


def _last_label_value(cells: list[str], label_pat: str, kind: str):
    """라벨 정확매칭의 '마지막' 발생 위치 → 바로 다음 셀 값.

    정정공시 = 앞(요약)+뒤(정정반영 본문) 2부 구성이라 마지막 발생이 최종값.
    ★마지막 발생의 값이 '-'(해지 등 공란화)면 None을 그대로 반환 — 앞쪽(정정전)
    발생으로 후퇴하면 취소된 값이 부활한다(7/4 적대검수 확정 발견).
    """
    pat = re.compile(label_pat)
    idxs = [i for i, c in enumerate(cells) if pat.match(c)]
    if not idxs:
        return None
    i = idxs[-1]
    if i + 1 >= len(cells):
        return None
    return _parse_value(cells[i + 1], kind)


def parse_contract_xml(xml_text: str) -> dict:
    """공급계약 공시 원문 XML → 정량 필드 dict (실패 필드 None)."""
    cells = _cells(xml_text)
    out = {}
    for field in ("contract_amount", "recent_revenue", "revenue_ratio_pct",
                  "counterparty", "period_start", "period_end", "contract_date"):
        pat, kind = _FIELD_PATTERNS[field]
        out[field] = _last_label_value(cells, pat, kind)
    if out["contract_amount"] is None:  # 총액 라벨 없는 조건부 서식 폴백
        pat, kind = _FIELD_PATTERNS["contract_amount_fallback"]
        out["contract_amount"] = _last_label_value(cells, pat, kind)
    # 비율 결측 시 계산 폴백
    if (out["revenue_ratio_pct"] is None and out["contract_amount"]
            and out["recent_revenue"] and out["recent_revenue"] > 0):
        out["revenue_ratio_pct"] = round(
            out["contract_amount"] / out["recent_revenue"] * 100, 1)
    return out


def fetch_contract_detail(rcept_no: str, api_key: str, *, timeout: int = 30,
                          session: requests.Session | None = None) -> dict | None:
    """rcept_no → document.xml 다운로드 → 파싱. 실패 시 None (graceful)."""
    try:
        sess = session or requests
        r = sess.get(DOCUMENT_URL, params={"crtfc_key": api_key, "rcept_no": rcept_no},
                     timeout=timeout)
        if r.status_code != 200 or len(r.content) < 100:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = z.namelist()
        if not names:
            return None
        # 다중 파일 시 최대 크기 XML 선택
        name = max(names, key=lambda n: z.getinfo(n).file_size)
        xml_text = z.read(name).decode("utf-8", errors="replace")
        out = parse_contract_xml(xml_text)
        out["rcept_no"] = rcept_no
        return out
    except Exception as e:  # noqa: BLE001 — 개별 공시 실패는 스킵
        logger.debug("[contract] %s 파싱 실패: %s", rcept_no, e)
        return None
