# -*- coding: utf-8 -*-
"""일일 시그널 수집 (관찰 모드) — 진입점.

실행:
    python -u -X utf8 -m insight_signals.agent.run_daily
옵션:
    --dry-run   외부 API 호출을 최소화한 동작 확인용 (뉴스만, 저장/원장등록 없음)

산출물 (프로젝트 루트 기준):
    data/insight_signals/signals_YYYY-MM-DD.json   전체 시그널
    data/insight_signals/picks_log.csv             누적 픽 로그 (성과 추적용)
    data/insight_signals/perf_ledger.csv           horizon 확정 성과 원장 (v0.2.1)
    data/insight_signals/seen_ledger.json          처리 완료 기사/공시 원장 (v0.2.1)
    data/insight_signals/report_YYYY-MM-DD.md      일일 리포트

v0.2.1 (7/10 전체검수 픽스):
    - seen 원장: 같은 기사(link)·같은 공시(rcept_no)가 다음날 또 시그널을 만들어
      picks_log 표본 n을 부풀리던 문제 차단 (DART days_back=3 창 중복 포함)
    - 성과 평가를 perf_ledger(HORIZON 도달일 종가 고정) 방식으로 교체
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys

# BAT 안전장치와 동일한 규칙: 프로젝트 루트를 sys.path에 보장
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from insight_signals.adapters import history_client, news_client, price_client  # noqa: E402
from insight_signals.adapters.dart_client import DartClient               # noqa: E402
from insight_signals.adapters.kis_flow_client import KisFlowClient        # noqa: E402
from insight_signals.agent import _env, report                            # noqa: E402
from insight_signals.use_cases import collect, evaluate                   # noqa: E402

log = logging.getLogger("insight_signals.run_daily")

SEEN_KEEP_DAYS = 30


def _load_seen(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {"news": dict(d.get("news", {})), "dart": dict(d.get("dart", {}))}
    except (OSError, ValueError):
        return {"news": {}, "dart": {}}


def _save_seen(path: str, seen: dict, today: dt.date) -> None:
    cutoff = (today - dt.timedelta(days=SEEN_KEEP_DAYS)).isoformat()
    pruned = {
        k: {kk: vv for kk, vv in seen[k].items() if vv >= cutoff}
        for k in ("news", "dart")
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pruned, f, ensure_ascii=False)
    os.replace(tmp, path)


def _news_key(a) -> str:
    return hashlib.sha1((a.link or a.title).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="뉴스 수집만 (키 불필요, 저장 없음)")
    args = parser.parse_args()

    root = _env.project_root()
    _env.setup_logging(root, "insight_signals_daily")
    _env.load_dotenv_manual(root)
    cfg = _env.load_config(root)

    today = dt.date.today()
    today_s = today.isoformat()
    data_dir = os.path.join(root, cfg["paths"]["data_dir"])
    os.makedirs(data_dir, exist_ok=True)
    seen_path = os.path.join(data_dir, "seen_ledger.json")
    seen = _load_seen(seen_path)

    env_names = cfg["env_names"]
    dart_key = os.environ.get(env_names["dart_key"], "")
    kis = KisFlowClient(
        app_key=os.environ.get(env_names["kis_app_key"], ""),
        app_secret=os.environ.get(env_names["kis_app_secret"], ""),
        cache_dir=data_dir,
        base_url=cfg["flow"]["kis_base_url"],
        token_cache_path=cfg["flow"].get("kis_token_cache", ""),
    )

    signals = []

    # ------------------------------------------------------------------
    # 1) 뉴스 키워드 시그널 (seen 원장으로 기수집 기사 제외 — 7/10 픽스)
    # ------------------------------------------------------------------
    articles = news_client.fetch_rss([tuple(x) for x in cfg["rss_feeds"]])
    if cfg.get("use_naver_mainnews", True):
        articles += news_client.fetch_naver_mainnews(pages=int(cfg.get("naver_pages", 2)))
    matched = news_client.match_keywords(
        articles, cfg["keywords"], cfg.get("negative_keywords")
    )
    fresh = [a for a in matched if _news_key(a) not in seen["news"]]
    log.info("키워드 매칭 %d건 (신규 %d / 기수집 %d) / 전체 %d건",
             len(matched), len(fresh), len(matched) - len(fresh), len(articles))
    matched = fresh

    name_map = {}
    dart = None
    if dart_key:
        try:
            dart = DartClient(dart_key, cache_dir=data_dir)
            name_map = dart.listed_name_map()
        except Exception as e:  # noqa: BLE001
            log.error("DART 초기화 실패 (뉴스 종목 매핑/임원 시그널 생략): %s", e)
    else:
        log.warning("%s 미설정 — DART 시그널과 뉴스 종목 매핑을 건너뜁니다", env_names["dart_key"])

    if name_map:
        matched = news_client.map_stocks(
            matched, name_map,
            blacklist=cfg.get("name_blacklist"),
            min_name_len=int(cfg.get("min_name_len", 2)),
        )
        signals += collect.news_signals(today_s, matched)

    # ------------------------------------------------------------------
    # 2) DART 임원 자사주 매수 시그널 (rcept_no 원장으로 재처리 차단 — 7/10 픽스)
    # ------------------------------------------------------------------
    filings = []
    if dart is not None and not args.dry_run:
        days_back = int(cfg["dart"]["days_back"])
        bgn = (today - dt.timedelta(days=days_back)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        try:
            corps = dart.insider_filing_corps(bgn, end)
            for corp_code in sorted({c["corp_code"] for c in corps if c.get("stock_code")}):
                filings += dart.insider_details(corp_code, since_yyyymmdd=bgn)
            n_all = len(filings)
            filings = [f for f in filings if f.rcept_no not in seen["dart"]]
            log.info("DART 상세 %d건 (신규 %d / 기처리 %d)", n_all, len(filings), n_all - len(filings))
            signals += collect.dart_signals(today_s, filings)
        except Exception as e:  # noqa: BLE001
            log.error("DART 임원 시그널 수집 실패: %s", e)

    # ------------------------------------------------------------------
    # 3) 역발상 수급 필터 (후보 종목에만)
    # ------------------------------------------------------------------
    if kis.available and not args.dry_run:
        for code, name in collect.candidate_stocks(signals):
            snap = kis.investor_flow(code, days=int(cfg["flow"]["days"]))
            sig = collect.flow_signal(today_s, code, name, snap)
            if sig:
                signals.append(sig)
    elif not kis.available:
        log.warning("KIS 키 미설정 — 수급 필터를 건너뜁니다 (뉴스+DART만으로 진행)")

    # ------------------------------------------------------------------
    # 합산 픽
    # ------------------------------------------------------------------
    pcfg = cfg["picks"]
    picks = collect.combine(
        today_s, signals,
        weights=pcfg["weights"],
        top_n=int(pcfg["top_n"]),
        min_score=float(pcfg["min_score"]),
    )

    # dry-run은 저장하지 않는다 (7/9 검수 픽스: picks_log는 승격 판단 데이터셋 —
    # 수동 dry-run 픽이 섞이면 forward 성과 통계가 오염됨. seen 원장도 미등록.)
    if args.dry_run:
        log.info("dry-run 완료 — 시그널 %d건, 픽 %d건 (저장 생략)", len(signals), len(picks))
        return 0

    for p in picks:
        p.price_at_pick = price_client.get_price(p.stock_code, kis_client=kis)

    # ------------------------------------------------------------------
    # 저장 + seen 원장 등록 (저장 성공 경로에서만)
    # ------------------------------------------------------------------
    sig_path = os.path.join(data_dir, f"signals_{today_s}.json")
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": today_s,
                "signals": [s.to_dict() for s in signals],
                "picks": [p.to_dict() for p in picks],
            },
            f, ensure_ascii=False, indent=2,
        )

    picks_log = os.path.join(data_dir, "picks_log.csv")
    evaluate.append_picks(picks_log, picks)

    for a in matched:
        seen["news"][_news_key(a)] = today_s
    for fl in filings:
        seen["dart"][fl.rcept_no] = today_s
    _save_seen(seen_path, seen, today)

    perf_ledger = os.path.join(data_dir, "perf_ledger.csv")
    perf = evaluate.evaluate(picks_log, perf_ledger, closes_fn=history_client.daily_closes)
    md = report.render_daily(today_s, signals, picks, perf_summary=perf.get("summary"))
    md_path = os.path.join(data_dir, f"report_{today_s}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    log.info("완료 — 시그널 %d건, 픽 %d건, 성과확정 %d행", len(signals), len(picks), len(perf["rows"]))
    log.info("리포트: %s", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
