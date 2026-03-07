"""KRX 국가별 외국인 보유량 스크래핑 — 브라우저 UI 직접 조작

1. Playwright 브라우저 열기
2. 사용자가 KRX 로그인 (수동)
3. 시그널 파일 감지 후 → 페이지 UI 직접 조작으로 8종목 수집
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright, Page

TARGET_STOCKS = {
    "012450": "한화에어로스페이스",
    "064350": "현대로템",
    "079550": "LIG넥스원",
    "011210": "현대위아",
    "047810": "한국항공우주",
    "010950": "S-Oil",
    "096770": "SK이노베이션",
    "004090": "한국석유",
}

MENU_URL = "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020203"
OUTPUT_PATH = PROJECT_ROOT / "data" / "krx_nationality_foreign.json"
SIGNAL_FILE = PROJECT_ROOT / "data" / "KRX_LOGIN_READY"


def wait_for_signal(timeout_sec: int = 300) -> bool:
    """시그널 파일 대기."""
    if SIGNAL_FILE.exists():
        SIGNAL_FILE.unlink()

    print(f"\n{'='*60}")
    print(f"  Playwright 브라우저에서 KRX 로그인하세요")
    print(f"  로그인 완료 후 여기서 알려주세요")
    print(f"  → data/KRX_LOGIN_READY 파일 생성 시 자동 진행")
    print(f"{'='*60}\n")

    for i in range(timeout_sec // 2):
        time.sleep(2)
        if SIGNAL_FILE.exists():
            SIGNAL_FILE.unlink()
            print("✅ 시그널 감지! 10초 후 수집 시작...")
            time.sleep(10)  # 로그인 완전 완료 대기
            return True
        if i % 15 == 0 and i > 0:
            print(f"  ... 대기 중 ({i*2}초)")
    return False


def extract_table_data(page: Page) -> list[dict]:
    """현재 페이지의 데이터 테이블을 추출."""
    time.sleep(2)

    # 방법 1: CI-GRID (KRX 커스텀 그리드)
    try:
        grid = page.query_selector(".CI-GRID-BODY-TABLE, .jqGrid, #grid1")
        if grid:
            rows = grid.query_selector_all("tr")
            if rows:
                data = []
                for row in rows:
                    cells = row.query_selector_all("td")
                    if cells:
                        data.append([c.inner_text().strip() for c in cells])
                if data:
                    return [{"cells": r} for r in data]
    except Exception:
        pass

    # 방법 2: 일반 table
    try:
        tables = page.query_selector_all("table")
        for table in tables:
            rows = table.query_selector_all("tbody tr")
            if len(rows) > 2:  # 데이터가 있는 테이블
                headers_el = table.query_selector_all("thead th, thead td")
                headers = [h.inner_text().strip() for h in headers_el] if headers_el else []
                data = []
                for row in rows:
                    cells = row.query_selector_all("td")
                    vals = [c.inner_text().strip() for c in cells]
                    if vals and any(v for v in vals):
                        if headers and len(headers) == len(vals):
                            data.append(dict(zip(headers, vals)))
                        else:
                            data.append({"cells": vals})
                if data:
                    return data
    except Exception:
        pass

    # 방법 3: 페이지 전체 텍스트에서 숫자 패턴 추출
    try:
        text = page.inner_text("body")
        # 국가명 + 숫자 패턴
        lines = text.split("\n")
        data_lines = []
        for line in lines:
            line = line.strip()
            # 국가명이 포함된 줄 (미국, 영국, 중국 등)
            countries = ["미국", "영국", "중국", "일본", "싱가포르", "홍콩",
                        "케이만", "캐나다", "노르웨이", "룩셈부르크", "사우디",
                        "아랍에미", "쿠웨이트", "스위스", "프랑스", "독일",
                        "네덜란드", "호주", "대만", "아일랜드"]
            for country in countries:
                if country in line:
                    data_lines.append(line)
                    break
        if data_lines:
            return [{"raw": l} for l in data_lines]
    except Exception:
        pass

    return []


def scrape_stock(page: Page, ticker: str, name: str) -> list[dict]:
    """한 종목의 국가별 외국인 보유량을 페이지 UI로 조회."""
    print(f"\n📊 [{ticker}] {name} 조회...")

    # 페이지 이동
    page.goto(MENU_URL)
    time.sleep(3)
    page.wait_for_load_state("networkidle", timeout=20000)
    time.sleep(2)

    # === 종목 검색 ===
    # KRX는 종목 찾기 버튼 → 팝업 → 검색 패턴을 사용
    try:
        # 종목 찾기 버튼 (돋보기 아이콘)
        finder_btn = page.query_selector(
            "button.CI-ICON-SEARCH, "
            "img[src*='search'], "
            "a.CI-ICON-SEARCH, "
            ".CI-FIND-BTN, "
            "button:has(img[src*='ico_search'])"
        )
        if finder_btn:
            finder_btn.click()
            time.sleep(2)

            # 팝업 내 검색 입력란
            popup_input = page.query_selector(
                ".CI-FIND-POPUP input, "
                "#popup_isuCd input, "
                ".layer_pop input[type='text'], "
                "input.CI-FIND-INPUT"
            )
            if popup_input:
                popup_input.fill(name[:4])
                time.sleep(1)
                popup_input.press("Enter")
                time.sleep(2)

                # 검색 결과에서 선택
                result_items = page.query_selector_all(
                    ".CI-FIND-RESULT tr td, "
                    ".layer_pop table tr, "
                    "li:has-text('" + name[:4] + "')"
                )
                for item in result_items:
                    text = item.inner_text()
                    if name[:4] in text or ticker in text:
                        item.click()
                        time.sleep(1)
                        break
        else:
            # 직접 입력 방식
            inputs = page.query_selector_all("input[type='text']")
            for inp in inputs:
                placeholder = inp.get_attribute("placeholder") or ""
                name_attr = inp.get_attribute("name") or ""
                if "종목" in placeholder or "isuCd" in name_attr or "finder" in name_attr:
                    inp.click()
                    inp.fill(name)
                    time.sleep(2)
                    # 자동완성 선택
                    auto = page.query_selector(f"li:has-text('{name[:4]}')")
                    if auto:
                        auto.click()
                        time.sleep(1)
                    break

        # 날짜 설정 (최근 10일)
        date_inputs = page.query_selector_all("input[name*='strtDd'], input[name*='endDd']")
        if len(date_inputs) >= 2:
            date_inputs[0].fill("20260224")
            date_inputs[1].fill("20260305")

        # 조회 버튼 클릭
        search_clicked = False
        btns = page.query_selector_all("button, a, input[type='button'], input[type='submit']")
        for btn in btns:
            try:
                text = btn.inner_text().strip()
            except Exception:
                text = btn.get_attribute("value") or ""
            if text == "조회":
                btn.click()
                search_clicked = True
                break

        if not search_clicked:
            # CSS 클래스로 시도
            search_btn = page.query_selector(".CI-BTN-PRIMARY, button.btn_search")
            if search_btn:
                search_btn.click()
                search_clicked = True

        if search_clicked:
            print(f"  조회 버튼 클릭됨")
            time.sleep(5)
            page.wait_for_load_state("networkidle", timeout=20000)
            time.sleep(3)
        else:
            print(f"  ⚠️ 조회 버튼 못 찾음")

    except Exception as e:
        print(f"  검색 오류: {e}")

    # === 데이터 추출 ===
    data = extract_table_data(page)
    print(f"  결과: {len(data)}행")
    for d in data[:3]:
        print(f"    {json.dumps(d, ensure_ascii=False)[:150]}")

    return data


def main():
    print("=" * 60)
    print("KRX 국가별 외국인 보유량 — UI 직접 조작")
    print("=" * 60)

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False, args=["--start-maximized"])
    context = browser.new_context(no_viewport=True)
    page = context.new_page()

    # KRX 메인 페이지
    page.goto("http://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd")
    page.wait_for_load_state("networkidle", timeout=30000)

    # 로그인 대기
    if not wait_for_signal():
        print("⚠️ 타임아웃 — 비회원으로 시도")

    # 8종목 수집
    all_data = {}
    for ticker, name in TARGET_STOCKS.items():
        rows = scrape_stock(page, ticker, name)
        all_data[ticker] = {
            "name": name,
            "data": rows,
            "count": len(rows),
        }
        time.sleep(2)

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = sum(d["count"] for d in all_data.values())
    print(f"\n💾 저장: {OUTPUT_PATH}")
    print(f"총 {total}행 수집")

    # 스크린샷 저장
    page.screenshot(path=str(PROJECT_ROOT / "data" / "krx_screenshot.png"))
    print("스크린샷 저장: data/krx_screenshot.png")

    print("\n완료! 브라우저를 닫아주세요...")
    try:
        page.wait_for_event("close", timeout=300000)
    except Exception:
        pass

    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
