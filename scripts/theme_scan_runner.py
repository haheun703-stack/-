"""
RSS + Grok 하이브리드 테마 스캐너 러너

1. RSS 피드에서 테마 키워드 매칭 (1차 감지)
2. Grok API로 관련주 확장 (2차 확장)
3. 각 종목의 기술 상태 조회 (parquet → RSI, MA20)
4. 텔레그램 알림 발송

Usage:
    python scripts/theme_scan_runner.py              # 기본 실행
    python scripts/theme_scan_runner.py --no-grok     # Grok 확장 없이 RSS만
    python scripts/theme_scan_runner.py --no-send     # 텔레그램 미발송
    python scripts/theme_scan_runner.py --dry-run     # 전체 시뮬레이션
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.adapters.rss_theme_scanner import RssThemeScanner
from src.entities.news_models import ThemeAlert, ThemeStock

logger = logging.getLogger(__name__)

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
THEME_ALERTS_PATH = PROJECT_ROOT / "data" / "theme_alerts.json"


def run_theme_scan(
    use_grok: bool = True,
    send_telegram: bool = True,
    dry_run: bool = False,
) -> list[ThemeAlert]:
    """테마 스캔 메인 플로우.

    Args:
        use_grok: Grok API로 관련주 확장 여부
        send_telegram: 텔레그램 발송 여부
        dry_run: True면 실제 API 호출/발송 없이 시뮬레이션

    Returns:
        감지된 ThemeAlert 리스트
    """
    # 설정 로드
    cfg = _load_settings()
    scanner_cfg = cfg.get("theme_scanner", {})

    if not scanner_cfg.get("enabled", True):
        logger.info("테마 스캐너 비활성화 상태")
        return []

    rss_feeds = scanner_cfg.get("rss_feeds", [])
    dedup_hours = scanner_cfg.get("dedup_hours", 24)
    max_articles = scanner_cfg.get("max_articles", 50)

    # 1. RSS 스캔
    logger.info("=" * 50)
    logger.info("[Phase 1] RSS 테마 스캔 시작")
    scanner = RssThemeScanner(
        rss_feeds=rss_feeds,
        dedup_hours=dedup_hours,
        max_articles=max_articles,
    )
    alerts = scanner.scan()
    logger.info("[Phase 1] RSS 감지 완료: %d건 테마", len(alerts))

    if not alerts:
        logger.info("감지된 테마 없음 — 종료")
        return []

    # 2. Grok 확장 (선택)
    if use_grok and not dry_run:
        logger.info("[Phase 2] Grok 관련주 확장 시작")
        alerts = _expand_with_grok(alerts)
    else:
        logger.info("[Phase 2] Grok 확장 건너뜀 (use_grok=%s, dry_run=%s)",
                     use_grok, dry_run)

    # 3. 기술 상태 조회
    logger.info("[Phase 3] 종목 기술 상태 조회")
    alerts = _enrich_technical_data(alerts)

    # 4. 결과 저장
    _save_alerts(alerts)

    # 5. 텔레그램 발송
    if send_telegram and not dry_run:
        logger.info("[Phase 4] 텔레그램 발송")
        _send_alerts(alerts)
    else:
        logger.info("[Phase 4] 텔레그램 발송 건너뜀")

    # 결과 출력
    _print_summary(alerts)

    return alerts


def _load_settings() -> dict:
    """settings.yaml 로드"""
    if not SETTINGS_PATH.exists():
        return {}
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_with_grok(alerts: list[ThemeAlert]) -> list[ThemeAlert]:
    """Grok API로 각 테마의 관련주 확장"""
    try:
        from src.adapters.grok_news_adapter import GrokNewsAdapter
    except ImportError:
        logger.warning("GrokNewsAdapter 임포트 실패 — Grok 확장 건너뜀")
        return alerts

    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        logger.warning("XAI_API_KEY 미설정 — Grok 확장 건너뜀")
        return alerts

    adapter = GrokNewsAdapter(api_key=api_key)

    for alert in alerts:
        known_names = [s.name for s in alert.related_stocks]
        try:
            expanded = asyncio.run(
                adapter.expand_theme_stocks(alert.theme_name, known_names)
            )
            if expanded:
                known_tickers = {s.ticker for s in alert.related_stocks}
                for item in expanded:
                    ticker = str(item.get("ticker", "")).zfill(6)
                    if ticker not in known_tickers:
                        alert.related_stocks.append(ThemeStock(
                            ticker=ticker,
                            name=item.get("name", ""),
                            order=item.get("order", 2),
                            source="grok_expanded",
                        ))
                        known_tickers.add(ticker)
                alert.grok_expanded = True
                logger.info("  '%s' Grok 확장: +%d종목",
                            alert.theme_name, len(expanded))
        except Exception as e:
            logger.warning("  '%s' Grok 확장 실패: %s", alert.theme_name, e)

    return alerts


def _enrich_technical_data(alerts: list[ThemeAlert]) -> list[ThemeAlert]:
    """각 종목의 기술 지표(RSI, MA20 괴리율) 조회"""
    import pandas as pd

    parquet_dir = PROJECT_ROOT / "data" / "processed"

    for alert in alerts:
        for stock in alert.related_stocks:
            pq_path = parquet_dir / f"{stock.ticker}.parquet"
            if not pq_path.exists():
                stock.pipeline_status = "NOT_IN_UNIVERSE"
                continue

            try:
                df = pd.read_parquet(pq_path)
                if df.empty:
                    continue

                last = df.iloc[-1]
                stock.current_rsi = float(last.get("rsi", 0) or 0)

                close = float(last.get("close", 0) or 0)
                ma20 = float(last.get("ma20", 0) or 0)
                if ma20 > 0 and close > 0:
                    stock.ma20_dist_pct = round((close / ma20 - 1) * 100, 1)

                # 간단한 파이프라인 상태 태깅
                adx = float(last.get("adx", 0) or 0)
                if stock.current_rsi > 70 or stock.ma20_dist_pct > 20:
                    stock.pipeline_status = "OVERHEATED"
                elif adx < 18:
                    stock.pipeline_status = "FAIL_G1"
                else:
                    stock.pipeline_status = "WATCHABLE"

            except Exception as e:
                logger.debug("종목 %s 기술 데이터 로드 실패: %s", stock.ticker, e)

    return alerts


def _save_alerts(alerts: list[ThemeAlert]) -> None:
    """알림 결과를 JSON으로 저장"""
    data = {
        "scan_time": datetime.now().isoformat(),
        "theme_count": len(alerts),
        "alerts": [],
    }
    for alert in alerts:
        stocks_data = []
        for s in alert.related_stocks:
            stocks_data.append({
                "ticker": s.ticker,
                "name": s.name,
                "order": s.order,
                "source": s.source,
                "pipeline_status": s.pipeline_status,
                "current_rsi": s.current_rsi,
                "ma20_dist_pct": s.ma20_dist_pct,
            })
        data["alerts"].append({
            "theme_name": alert.theme_name,
            "matched_keyword": alert.matched_keyword,
            "news_title": alert.news_title,
            "news_url": alert.news_url,
            "news_source": alert.news_source,
            "published": alert.published,
            "grok_expanded": alert.grok_expanded,
            "timestamp": alert.timestamp,
            "related_stocks": stocks_data,
        })

    THEME_ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THEME_ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장: %s", THEME_ALERTS_PATH)


def _send_alerts(alerts: list[ThemeAlert]) -> None:
    """텔레그램 알림 발송"""
    try:
        from src.telegram_sender import send_theme_alert
    except ImportError:
        logger.warning("send_theme_alert 임포트 실패")
        return

    for alert in alerts:
        try:
            ok = send_theme_alert(alert)
            if ok:
                logger.info("텔레그램 발송 성공: '%s'", alert.theme_name)
            else:
                logger.warning("텔레그램 발송 실패: '%s'", alert.theme_name)
        except Exception as e:
            logger.error("텔레그램 발송 오류: %s", e)


def _print_summary(alerts: list[ThemeAlert]) -> None:
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print(f"테마 스캔 결과: {len(alerts)}건 감지")
    print("=" * 60)

    for alert in alerts:
        grok_tag = " [+Grok]" if alert.grok_expanded else ""
        print(f"\n  테마: {alert.theme_name}{grok_tag}")
        print(f"  뉴스: {alert.news_title}")
        print(f"  출처: {alert.news_source}")
        print(f"  관련주 {len(alert.related_stocks)}개:")

        for s in alert.related_stocks:
            src_tag = " [G]" if s.source == "grok_expanded" else ""
            if s.current_rsi > 0:
                status = "과열" if (s.current_rsi > 70 or s.ma20_dist_pct > 20) else "관심"
                print(f"    {s.order}차 {s.name}({s.ticker}) "
                      f"RSI={s.current_rsi:.1f} MA20={s.ma20_dist_pct:+.1f}% "
                      f"[{status}]{src_tag}")
            else:
                print(f"    {s.order}차 {s.name}({s.ticker}) "
                      f"[{s.pipeline_status}]{src_tag}")

    print()


def main():
    parser = argparse.ArgumentParser(description="RSS + Grok 테마 스캐너")
    parser.add_argument("--no-grok", action="store_true",
                        help="Grok API 확장 없이 RSS만 사용")
    parser.add_argument("--no-send", action="store_true",
                        help="텔레그램 발송 안 함")
    parser.add_argument("--dry-run", action="store_true",
                        help="전체 시뮬레이션 (API 호출/발송 없음)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 로그 출력")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_theme_scan(
        use_grok=not args.no_grok,
        send_telegram=not args.no_send,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
