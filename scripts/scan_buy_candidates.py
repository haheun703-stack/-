"""
전체 종목 매수 후보 스캔 -> Kill→Rank→Tag -> 텔레그램 발송

v10.0: 4축 100점 제거, Kill 중복 제거, Trap 제거
  - Kill: K3(트리거) + K4(유동성) — K1/K2/K5는 v8 Gate G1/G2/G3과 중복이므로 제거
  - Rank: R:R × Zone × Catalyst (선행 100%)
  - Tag: 수급 streak 기반 (Part 2에서 5D 교차필터로 확장 예정)

사용법:
    python scripts/scan_buy_candidates.py --grade AB
    python scripts/scan_buy_candidates.py --grade AB --no-news --no-send
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.market_signal_scanner import MarketSignalScanner

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# =========================================================
# Kill→Rank→Tag 파이프라인
# =========================================================

def detect_regime() -> dict:
    """공매도 상태 판정 (로깅 + 향후 G4용)."""
    import yaml
    from datetime import date

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    today = date.today()
    calendar = cfg.get("short_selling_calendar", [])
    status = "active"

    for period in calendar:
        start = date.fromisoformat(str(period["start"]))
        end = date.fromisoformat(str(period["end"]))
        if start <= today <= end:
            status = period["status"]
            break

    return {"status": status}


def kill_filters(sig: dict) -> tuple[bool, list[str]]:
    """Kill Filters — K3(트리거) + K4(유동성).

    K1(Zone), K2(R:R), K5(고점근접)은 v8 Gate G1/G2/G3과 중복 → 제거.
    """
    kills = []

    # K3: Trigger 미발동
    trigger = sig.get("trigger_type", "none")
    if trigger in ("none", "waiting", "setup"):
        kills.append(f"K3:Trigger({trigger})")

    # K4: 20일 평균 거래대금 < 10억
    avg_tv = sig.get("avg_trading_value_20d", 0)
    if avg_tv < 1_000_000_000:
        kills.append(f"K4:유동성({avg_tv / 1e8:.0f}억<10억)")

    return len(kills) == 0, kills


def generate_tags(sig: dict) -> list[str]:
    """정보 태그 (참고용).

    수급 streak + SD 교차 + 관찰자 밀도 태그.
    """
    tags = []

    # 외국인/기관 연속 매수
    f_streak = sig.get("foreign_streak", 0)
    i_streak = sig.get("inst_streak", 0)
    if f_streak >= 5:
        tags.append(f"외{f_streak}D연속")
    elif f_streak >= 3:
        tags.append(f"외{f_streak}D")
    if i_streak >= 5:
        tags.append(f"기{i_streak}D연속")
    elif i_streak >= 3:
        tags.append(f"기{i_streak}D")

    # SD 교차 태그
    sd = sig.get("sd_cross", "")
    if sd == "양호":
        tags.append("수급양호")
    elif sd == "경고":
        tags.append("수급경고")

    # 관찰자 밀도 태그
    density = sig.get("density", "")
    if density == "저밀도":
        tags.append("숨은종목")
    elif density == "고밀도":
        tags.append("과밀")

    return tags


def run_pipeline(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Kill→Rank→Tag 파이프라인.

    반환: (survivors, killed_list)
    """
    # 레짐 감지 (로깅용)
    regime = detect_regime()
    print(f"  Regime: {regime['status']}")

    killed_list = []
    survivors = []

    for sig in candidates:
        # Kill Filters (K3 + K4)
        passed, kill_reasons = kill_filters(sig)
        if not passed:
            sig["v9_kill_reasons"] = kill_reasons
            killed_list.append(sig)
            continue

        survivors.append(sig)

    # Rank: R:R × zone_score × catalyst × sd_cross × density
    #
    # SD 교차필터 (Part 2):
    #   외국인 매수 + 공매도 커버링 → ×1.05 (양호)
    #   외국인 매도 + 공매도 빌딩  → ×0.90 (경고)
    #   혼재/데이터 없음            → ×1.00 (중립)
    #
    # 관찰자 밀도 (Part 2):
    #   avg_trading_value_20d 기준 분위로 역가중
    #   상위 고밀도(>500억) → ×0.95, 하위 저밀도(<50억) → ×1.05

    # 밀도 분위 계산 (전체 survivor 기준)
    tv_values = [s.get("avg_trading_value_20d", 0) for s in survivors]
    tv_p75 = sorted(tv_values)[int(len(tv_values) * 0.75)] if len(tv_values) >= 4 else 50e9
    tv_p25 = sorted(tv_values)[int(len(tv_values) * 0.25)] if len(tv_values) >= 4 else 5e9

    for sig in survivors:
        zone = sig.get("zone_score", 0)
        rr = sig.get("risk_reward", 0)

        catalyst_boost = 1.0

        # Grok 뉴스 실적 서프라이즈
        news_data = sig.get("news_data")
        if news_data:
            earnings = news_data.get("earnings_estimate", {})
            if earnings.get("surprise_direction") == "beat":
                catalyst_boost = 1.10

        # DART 공시 촉매 부스트
        dart = sig.get("dart_analysis", {})
        if dart.get("catalyst_type") == "catalyst" and dart.get("confidence", 0) >= 0.7:
            catalyst_boost *= 1.10

        # SD 교차필터: 외국인 × 공매도
        f_net = sig.get("foreign_net_5d", 0)
        s_chg = sig.get("short_balance_chg_5d", 0)
        sd_mult = 1.0
        if f_net > 0 and s_chg < 0:       # 외국인 매수 + 공매도 커버링
            sd_mult = 1.05
            sig["sd_cross"] = "양호"
        elif f_net < 0 and s_chg > 0:      # 외국인 매도 + 공매도 빌딩
            sd_mult = 0.90
            sig["sd_cross"] = "경고"
        else:
            sig["sd_cross"] = "중립"

        # 관찰자 밀도 (역가중)
        tv = sig.get("avg_trading_value_20d", 0)
        density_mult = 1.0
        if tv > tv_p75:                    # 고밀도 (과밀)
            density_mult = 0.95
            sig["density"] = "고밀도"
        elif tv < tv_p25:                  # 저밀도 (숨은 보석)
            density_mult = 1.05
            sig["density"] = "저밀도"
        else:
            sig["density"] = "보통"

        sig["v9_rank_score"] = round(rr * zone * catalyst_boost * sd_mult * density_mult, 4)
        sig["v9_catalyst_boost"] = catalyst_boost
        sig["v9_sd_mult"] = sd_mult
        sig["v9_density_mult"] = density_mult

    # Tags (sd_cross, density 태그 포함)
    for sig in survivors:
        sig["v9_tags"] = generate_tags(sig)

    # 순위 정렬
    survivors.sort(key=lambda s: s["v9_rank_score"], reverse=True)

    return survivors, killed_list


# =========================================================
# Grok 뉴스 검색
# =========================================================

def fetch_grok_news(name: str, ticker: str) -> dict | None:
    """Grok API로 종목 심층 분석 (동기 직접 호출)."""
    try:
        import requests as _req
        from dotenv import load_dotenv as _ld
        _ld(Path(__file__).resolve().parent.parent / ".env")

        from src.adapters.grok_news_adapter import GrokNewsAdapter
        adapter = GrokNewsAdapter()
        if not adapter.api_key:
            return None

        prompt = adapter._deep_analysis_prompt(name, ticker)
        payload = {
            "model": "grok-4-1-fast",
            "input": [
                {"role": "system", "content": (
                    "너는 한국 주식시장 전문 리서치 애널리스트다. "
                    "웹과 X(트위터)를 검색해서 종목의 최신 뉴스뿐 아니라, "
                    "아직 해소되지 않은 과거 이슈, 실적 전망, 수급 동향을 "
                    "종합적으로 분석한다. 반드시 요청된 JSON 형식으로만 응답한다."
                )},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
        }
        resp = _req.post(
            "https://api.x.ai/v1/responses",
            headers=adapter.headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        return adapter._parse_response(resp.json())
    except Exception as e:
        print(f"  ! Grok fail ({name}): {e}")
        return None


# =========================================================
# 데이터 로드 + SignalEngine Pipeline
# =========================================================

def load_all_parquets() -> dict[str, pd.DataFrame]:
    """data/processed/*.parquet 로드 → {ticker: DataFrame}"""
    data = {}
    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq)
        if len(df) >= 200:
            data[pq.stem] = df
    return data


def load_name_map() -> dict:
    """stock_data_daily CSV 파일명에서 종목명 추출."""
    name_map = {}
    csv_dir = Path(__file__).resolve().parent.parent / "stock_data_daily"
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            match = re.search(r"_(\d{6})$", f.stem)
            if match:
                ticker = match.group(1)
                name = f.stem[: f.stem.rfind("_")]
                name_map[ticker] = name
    return name_map


def _calc_di(df: pd.DataFrame, idx: int, period: int = 14) -> tuple[float, float]:
    """ADX의 +DI, -DI를 직접 계산 (parquet에 미포함이므로)."""
    if len(df) < period + 2 or idx < period + 1:
        return 0.0, 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # 상대적으로 큰 쪽만 살림
    mask_plus = plus_dm < minus_dm
    mask_minus = minus_dm < plus_dm
    plus_dm = plus_dm.where(~mask_plus, 0)
    minus_dm = minus_dm.where(~mask_minus, 0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period).mean()
    p_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
    m_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

    return float(p_di.iloc[idx] or 0), float(m_di.iloc[idx] or 0)


def _calc_streak(series: pd.Series) -> int:
    """최근 연속 순매수 일수 계산 (음수면 순매도 연속)."""
    if series.empty:
        return 0
    vals = series.values
    if pd.isna(vals[-1]) or vals[-1] == 0:
        return 0
    direction = 1 if vals[-1] > 0 else -1
    count = 0
    for v in reversed(vals):
        if pd.isna(v) or v == 0:
            break
        if (v > 0 and direction > 0) or (v < 0 and direction < 0):
            count += 1
        else:
            break
    return count * direction


# =========================================================
# 메인 스캔
# =========================================================

def scan_all(
    grade_filter: str = "A",
    use_news: bool = True,
    use_dart: bool = False,
) -> tuple[list[dict], dict]:
    """전 종목 스캔 -> Grade 필터 -> Kill→Rank→Tag 반환."""
    from src.signal_engine import SignalEngine

    # 데이터 로드 (parquet)
    data_dict = load_all_parquets()
    name_map = load_name_map()
    print(f"scan: {len(data_dict)} stocks (parquet) | grade={grade_filter} | news={'ON' if use_news else 'OFF'}")

    # SignalEngine 초기화
    engine = SignalEngine("config/settings.yaml")
    scanner = MarketSignalScanner()

    candidates = []
    stats = {
        "total": len(data_dict),
        "loaded": 0,
        "passed_pipeline": 0,
        "trigger_impulse": 0,
        "trigger_confirm": 0,
        "grade_A": 0,
        "grade_B": 0,
        "grade_C": 0,
        "after_grade_filter": 0,
    }

    t0 = time.time()

    for i, (ticker, df) in enumerate(data_dict.items()):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(data_dict)} ({elapsed:.1f}s)")

        stats["loaded"] += 1
        idx = len(df) - 1

        try:
            result = engine.calculate_signal(ticker, df, idx)
        except Exception:
            continue

        if not result["signal"]:
            continue

        grade = result["grade"]
        stats["passed_pipeline"] += 1

        trigger_type = result.get("trigger_type", "none")
        if trigger_type == "impulse":
            stats["trigger_impulse"] += 1
        elif trigger_type == "confirm":
            stats["trigger_confirm"] += 1

        grade_key = f"grade_{grade}"
        stats[grade_key] = stats.get(grade_key, 0) + 1

        if grade not in grade_filter:
            continue

        stats["after_grade_filter"] += 1

        # DataFrame에서 필드 추출
        row = df.iloc[idx]
        name = name_map.get(ticker, ticker)

        # 수급 streak 계산
        foreign_streak = 0
        inst_streak = 0
        foreign_amount_5d = 0
        inst_amount_5d = 0

        if "foreign_net" in df.columns:
            f_series = df["foreign_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            foreign_streak = _calc_streak(f_series)
            foreign_amount_5d = int(df["foreign_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        if "inst_net" in df.columns:
            i_series = df["inst_net"].iloc[max(0, idx - 19) : idx + 1].fillna(0)
            inst_streak = _calc_streak(i_series)
            inst_amount_5d = int(df["inst_net"].iloc[max(0, idx - 4) : idx + 1].fillna(0).sum())

        sig = {
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "zone_score": result["zone_score"],
            "trigger_type": trigger_type,
            "confidence": result.get("trigger_confidence", 0),
            "entry_price": result["entry_price"],
            "stop_loss": result["stop_loss"],
            "target_price": result["target_price"],
            "risk_reward": result.get("risk_reward_ratio", 0),
            # DataFrame에서 직접 추출
            "rsi": float(row.get("rsi_14", 50) or 50),
            "adx": float(row.get("adx_14", 0) or 0),
            "plus_di": 0.0,
            "minus_di": 0.0,
            "vol_surge": float(row.get("volume_surge_ratio", 1.0) or 1.0),
            "obv_trend": (
                "up" if (row.get("obv", 0) or 0) > (df.iloc[max(0, idx - 20)].get("obv", 0) or 0)
                else "down"
            ),
            # 수급
            "foreign_streak": foreign_streak,
            "inst_streak": inst_streak,
            "foreign_amount_5d": foreign_amount_5d,
            "inst_amount_5d": inst_amount_5d,
            # SD 교차필터 (Part 2: 외국인 × 공매도)
            "foreign_net_5d": float(row.get("foreign_net_5d", 0) or 0),
            "short_balance_chg_5d": float(row.get("short_balance_chg_5d", 0) or 0),
            # SignalEngine 고급 필드
            "consensus": result.get("consensus"),
            # 유동성 + 고점 필드
            "avg_trading_value_20d": float(
                (df["close"] * df["volume"]).iloc[max(0, idx - 19) : idx + 1].mean()
            ),
            "pct_of_52w_high": float(row.get("pct_of_52w_high", 0) or 0),
        }

        # +DI/-DI 직접 계산 (parquet에 미포함)
        try:
            p_di, m_di = _calc_di(df, idx)
            sig["plus_di"] = p_di
            sig["minus_di"] = m_di
        except Exception:
            pass

        # Market Signal Scanner
        try:
            market_signals = scanner.scan_all(df, idx)
            sig["market_signals"] = [
                {"title": s.title, "importance": s.importance, "confidence": s.confidence}
                for s in market_signals
            ] if market_signals else []
        except Exception:
            sig["market_signals"] = []

        candidates.append(sig)

    scan_elapsed = time.time() - t0
    stats["scan_sec"] = round(scan_elapsed, 1)

    # -- Grok 뉴스 적용 --
    if use_news and candidates:
        print(f"\nGrok news ({len(candidates)} stocks)...")
        news_t0 = time.time()
        for sig in candidates:
            print(f"  {sig['name']}({sig['ticker']})...", end=" ", flush=True)
            news_data = fetch_grok_news(sig["name"], sig["ticker"])
            sig["news_data"] = news_data
            if news_data:
                sentiment = news_data.get("overall_sentiment", "?")
                takeaway = news_data.get("key_takeaway", "")[:30]
                print(f"OK [{sentiment}] {takeaway}")
            else:
                print("- (empty)")
        stats["news_sec"] = round(time.time() - news_t0, 1)
    else:
        for sig in candidates:
            sig["news_data"] = None
        stats["news_sec"] = 0

    # -- DART 공시 분류 --
    if use_dart and candidates:
        try:
            from src.adapters.dart_adapter import DartAdapter
            from src.adapters.openai_classifier import classify_batch

            print(f"\nDART 공시 분류 ({len(candidates)} stocks)...")
            dart = DartAdapter()
            dart_results = classify_batch(candidates, dart)

            dart_catalyst_count = 0
            for sig in candidates:
                dr = dart_results.get(sig["ticker"], {})
                sig["dart_analysis"] = dr
                if dr.get("catalyst_type") == "catalyst" and dr.get("confidence", 0) >= 0.7:
                    dart_catalyst_count += 1
                    print(f"  촉매 발견: {sig['name']}({sig['ticker']}) — "
                          f"{dr.get('catalyst_category','')} ({dr.get('reason','')[:40]})")

            stats["dart_catalyst_count"] = dart_catalyst_count
            stats["dart_total"] = len(dart_results)
            print(f"  DART 분류 완료: {dart_catalyst_count}/{len(dart_results)} 촉매 발견")
        except Exception as e:
            print(f"  DART 공시 분류 실패 (fail-safe 계속): {e}")
            stats["dart_catalyst_count"] = 0
    else:
        stats["dart_catalyst_count"] = 0

    # -- Kill→Rank→Tag --
    survivors, killed = run_pipeline(candidates)
    stats["v9_killed"] = len(killed)
    stats["v9_survivors"] = len(survivors)
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["v9_killed_list"] = killed
    return survivors, stats


# =========================================================
# 텔레그램 메시지 포맷
# =========================================================

def format_telegram_message(candidates: list[dict], stats: dict) -> str:
    """Kill→Rank→Tag 텔레그램 메시지 포맷."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # -- Header --
    lines.append(f"[Quant v10.0] {now} Kill\u2192Rank\u2192Tag")
    lines.append("")

    # -- 파이프라인 설명 --
    lines.append("[ 파이프라인 ]")
    lines.append("Kill(K3+K4) \u2192 Rank(R:R\u00d7Zone\u00d7SD\u00d7Den) \u2192 Tag")
    lines.append("  \u00b7 선행: Zone + R:R + Trigger")
    lines.append("  \u00b7 SD교차: 외국인\u00d7공매도 (+5%/-10%)")
    lines.append("  \u00b7 밀도: 고밀도(-5%), 저밀도(+5%)")
    lines.append("")

    # -- 스캔 통계 --
    killed = stats.get("v9_killed", 0)
    survivors = stats.get("v9_survivors", 0)
    lines.append("[ 스캔 통계 ]")
    lines.append(
        f"전체: {stats['total']:,}종목 > Pipeline: {stats['passed_pipeline']}종목"
    )
    lines.append(
        f"등급: A:{stats.get('grade_A',0)} B:{stats.get('grade_B',0)} "
        f"C:{stats.get('grade_C',0)} | 필터 후: {stats.get('after_grade_filter',0)}종목"
    )
    lines.append(f"Kill: {killed}종목 | 생존: {survivors}종목")
    lines.append(f"소요: 스캔 {stats.get('scan_sec',0)}초 + 뉴스 {stats.get('news_sec',0)}초")
    lines.append("")

    if not candidates:
        lines.append("Kill 필터 통과 종목 없음")
        killed_list = stats.get("v9_killed_list", [])
        if killed_list:
            lines.append("")
            lines.append(f"[ Kill ({len(killed_list)}종목) ]")
            for sig in killed_list:
                reasons = ", ".join(sig.get("v9_kill_reasons", []))
                lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        return "\n".join(lines)

    # -- 1순위 추천 매수 --
    top = candidates[0]
    top_trigger = "확인매수" if top["trigger_type"] == "confirm" else "IMP"
    top_tags = ", ".join(top.get("v9_tags", []))
    boost = top.get("v9_catalyst_boost", 1.0)
    sd_m = top.get("v9_sd_mult", 1.0)
    den_m = top.get("v9_density_mult", 1.0)
    mods = []
    if boost > 1.0:
        mods.append("촉매")
    if sd_m != 1.0:
        mods.append(f"SD{sd_m:.2f}")
    if den_m != 1.0:
        mods.append(f"밀도{den_m:.2f}")
    mod_str = f" [{','.join(mods)}]" if mods else ""

    lines.append("[ 1순위 추천 매수 ]")
    lines.append(f"{top['name']} ({top['ticker']}) [{top_trigger}]")
    lines.append(
        f"Rank {top['v9_rank_score']:.3f} = "
        f"R:R({top['risk_reward']:.1f}) x Zone({top['zone_score']:.2f}){mod_str}"
    )
    lines.append(
        f"현재 {top['entry_price']:,}원 | "
        f"목표 {top['target_price']:,} (+{((top['target_price']/top['entry_price'])-1)*100:.1f}%) | "
        f"손절 {top['stop_loss']:,} ({((top['stop_loss']/top['entry_price'])-1)*100:.1f}%)"
    )
    if top_tags:
        lines.append(f"태그: {top_tags}")

    # -- 나머지 후보 --
    if len(candidates) > 1:
        lines.append("")
        lines.append(f"[ 매수 후보 ({len(candidates)-1}개) ]")
        for i, sig in enumerate(candidates[1:], start=2):
            tags = ", ".join(sig.get("v9_tags", []))
            b = sig.get("v9_catalyst_boost", 1.0)
            b_str = " x1.10" if b > 1.0 else ""
            lines.append(
                f"{i}. {sig['name']}({sig['ticker']}) "
                f"Rank {sig['v9_rank_score']:.3f} "
                f"RR:{sig['risk_reward']:.1f} Zone:{sig['zone_score']:.2f}{b_str}"
            )
            if tags:
                lines.append(f"   [{tags}]")

    # -- Kill 요약 --
    killed_list = stats.get("v9_killed_list", [])
    if killed_list:
        lines.append("")
        lines.append(f"[ Kill ({len(killed_list)}종목) ]")
        for sig in killed_list[:5]:
            reasons = ", ".join(sig.get("v9_kill_reasons", []))
            lines.append(f"  {sig['name']}({sig['ticker']}): {reasons}")
        if len(killed_list) > 5:
            lines.append(f"  ... +{len(killed_list)-5}종목")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Quant Buy Scan (Kill→Rank→Tag)")
    parser.add_argument("--no-send", action="store_true", help="No telegram send")
    parser.add_argument("--grade", type=str, default="A", help="Grade filter (A, AB, ABC)")
    parser.add_argument("--no-news", action="store_true", help="Skip Grok news")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--dart", action="store_true", help="DART 공시 + OpenAI 촉매 분류")
    args = parser.parse_args()

    dart_label = " + DART" if args.dart else ""
    print("=" * 50)
    print(f"  [Quant v10.0] Kill\u2192Rank\u2192Tag{dart_label}")
    print(f"  Kill(K3+K4) \u2192 Rank(R:R\u00d7Zone) \u2192 Tag")
    print("=" * 50)

    candidates, stats = scan_all(
        grade_filter=args.grade.upper(),
        use_news=not args.no_news,
        use_dart=args.dart,
    )

    msg = format_telegram_message(candidates, stats)
    print("\n" + msg)

    # HTML 보고서 생성 + PNG 변환
    png_path = None
    if not args.no_html and candidates:
        try:
            from src.html_report import generate_premarket_report
            print("\nHTML 보고서 생성 중...")
            html_path, png_path = generate_premarket_report(candidates, stats)
            print(f"HTML: {html_path}")
            if png_path:
                print(f"PNG:  {png_path}")
        except Exception as e:
            print(f"HTML 보고서 생성 실패: {e}")

    if not args.no_send:
        from src.telegram_sender import send_message

        # 1) PNG 이미지 전송 (보고서)
        if png_path and png_path.exists():
            from src.html_report import send_report_to_telegram
            print("\nSending report image to Telegram...")
            caption = f"[Quant v10.0] 장시작전 분석 | {len(candidates)}종목 | Grade {args.grade.upper()}"
            img_ok = send_report_to_telegram(png_path, caption)
            print("OK - Report image sent" if img_ok else "FAIL - Image send")

        # 2) 텍스트 메시지 전송
        print("Sending text to Telegram...")
        success = send_message(msg)
        print("OK - Text sent" if success else "FAIL - Check .env")
    else:
        print("\n(--no-send: skipped)")

    output_path = Path(__file__).parent.parent / "data" / "scan_result.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(msg, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
