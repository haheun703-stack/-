"""보유종목 ICT 분석 + 5분봉 지표 + 액션 신호 → 텔레그램 전송"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
INTRADAY_5MIN = DATA_DIR / "intraday" / "5min"
OR_IR_DIR = DATA_DIR / "daily" / "or_ir"


def _load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _latest_date_dir(base_dir: pathlib.Path) -> str | None:
    """디렉토리 내 최신 날짜 폴더/파일명 반환"""
    if not base_dir.exists():
        return None
    candidates = sorted(
        [d.stem if d.is_file() else d.name for d in base_dir.iterdir()
         if d.name[0].isdigit()],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _calc_5min_indicators(ticker: str, date_str: str, current_price: int) -> dict:
    """5분봉 parquet → 패턴 + VWAP 계산"""
    result = {"pattern": None, "pattern_kr": None, "vwap": None, "vwap_pos": None}

    pf = INTRADAY_5MIN / date_str / f"{ticker}.parquet"
    if not pf.exists():
        return result

    df = pd.read_parquet(pf)
    if df.empty or len(df) < 2:
        return result

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── VWAP 계산 (전체 봉) ──
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).sum()
    cum_vol = df["volume"].sum()
    if cum_vol > 0:
        vwap = round(cum_tpv / cum_vol)
        result["vwap"] = vwap
        result["vwap_pos"] = "above" if current_price >= vwap else "below"

    # ── 5분봉 패턴 (마지막 6봉 기준) ──
    recent = df.tail(6)
    if len(recent) < 2:
        return result

    first = recent.iloc[0]
    last = recent.iloc[-1]
    first_body = first["close"] - first["open"]
    last_body = last["close"] - last["open"]
    volumes = recent["volume"].tolist()
    vol_up = volumes[-1] > volumes[0]

    # 패턴 1: 눌림 반등
    if first_body < 0 and last_body > 0:
        lows = recent["low"].tolist()
        if lows[-1] >= lows[0]:
            result["pattern"] = "pullback_bounce"
            result["pattern_kr"] = "눌림반등"
    # 패턴 2: 추세 지속
    elif first_body > 0 and last_body > 0 and vol_up:
        result["pattern"] = "trend_continue"
        result["pattern_kr"] = "양봉지속"
    # 패턴 3: 갭실패 (연속 음봉)
    elif all(recent.iloc[i]["close"] < recent.iloc[i]["open"] for i in range(-2, 0)):
        result["pattern"] = "gap_fail"
        result["pattern_kr"] = "연속음봉"
    # 패턴 4: VWAP 위치 기반
    elif result["vwap"]:
        if current_price < result["vwap"]:
            result["pattern"] = "below_vwap"
            result["pattern_kr"] = "VWAP하회"
        else:
            result["pattern"] = "above_vwap"
            result["pattern_kr"] = "VWAP상회"
    else:
        result["pattern"] = "mixed"
        result["pattern_kr"] = "혼조"

    return result


def _fetch_fresh_balance() -> dict:
    """KIS API에서 최신 잔고 조회 → kis_balance.json 갱신"""
    from datetime import datetime
    try:
        from src.adapters.kis_order_adapter import KisOrderAdapter
        adapter = KisOrderAdapter()
        balance = adapter.fetch_balance()
        balance["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 저장
        path = DATA_DIR / "kis_balance.json"
        path.write_text(json.dumps(balance, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[KIS] 잔고 갱신 완료: {len(balance.get('holdings', []))}종목")
        return balance
    except Exception as e:
        print(f"[KIS] 잔고 조회 실패: {e} → 캐시 사용")
        return _load_json(DATA_DIR / "kis_balance.json")


def main():
    from src.telegram_sender import send_message

    # KIS API에서 최신 보유종목 가져오기
    balance = _fetch_fresh_balance()
    holdings = balance.get("holdings", [])
    if not holdings:
        print("보유종목 없음")
        return

    held_tickers = {h["ticker"] for h in holdings}
    held_map = {h["ticker"]: h for h in holdings}

    # ICT 데이터 로드 (최신)
    premium_map, equal_map = {}, {}
    for subdir, target in [("premium_levels", premium_map), ("equal_levels", equal_map)]:
        ict_dir = DATA_DIR / subdir
        if ict_dir.exists():
            files = sorted(ict_dir.glob("*.json"), reverse=True)
            if files:
                ict_data = _load_json(files[0])
                for lv in ict_data.get("levels", []):
                    sym = lv.get("symbol", "")
                    if sym in held_tickers:
                        target[sym] = lv

    # OR/IR 데이터 로드 (최신)
    orir_map = {}
    orir_date = _latest_date_dir(OR_IR_DIR)
    if orir_date:
        orir_data = _load_json(OR_IR_DIR / f"{orir_date}.json")
        for rec in orir_data.get("records", []):
            sym = rec.get("symbol", "")
            if sym in held_tickers:
                orir_map[sym] = rec

    # 5분봉 최신 날짜
    candle_date = _latest_date_dir(INTRADAY_5MIN)

    # 레벨 이름 매핑
    level_names = {
        "prev_day_high": "전일고", "prev_day_low": "전일저",
        "prev_week_high": "주간고", "prev_week_low": "주간저",
        "prev_month_high": "월간고", "prev_month_low": "월간저",
    }

    lines = []
    lines.append("📊 보유종목 ICT 전술 분석")
    lines.append(f"📅 기준: {balance.get('fetched_at', '?')[:10]}")
    lines.append(f"💼 {len(holdings)}종목 │ 평가 {balance.get('total_eval', 0):,}원")
    total_pnl = balance.get("total_pnl", 0)
    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    lines.append(f"{pnl_icon} 총손익: {total_pnl:+,}원")

    for h in holdings:
        ticker = h["ticker"]
        name = h["name"]
        current = h["current_price"]
        avg = h["avg_price"]
        pnl_pct = h["pnl_pct"]

        # 손익 이모지
        if pnl_pct >= 10:
            pnl_icon = "🟢"
        elif pnl_pct >= 0:
            pnl_icon = "🔵"
        elif pnl_pct >= -5:
            pnl_icon = "🟡"
        elif pnl_pct >= -10:
            pnl_icon = "🟠"
        else:
            pnl_icon = "🔴"

        lines.append("")
        lines.append(f"━━━━━━━━━━━━━━━━")
        lines.append(f"{pnl_icon} {name} ({ticker})")
        lines.append(f"  💲 현재 {current:,} │ 평단 {avg:,.0f} │ {pnl_pct:+.1f}%")

        # ICT 프리미엄 레벨
        plv = premium_map.get(ticker)
        if plv:
            distances = plv.get("distances", {})
            res = plv.get("nearest_resistance")
            sup = plv.get("nearest_support")

            if res:
                res_name = level_names.get(res["level"], res["level"])
                lines.append(
                    f"  📐 저항: {res_name} {res['price']:,} ({res['distance_pct']:+.1f}%)"
                )
            if sup:
                sup_name = level_names.get(sup["level"], sup["level"])
                lines.append(
                    f"  📐 지지: {sup_name} {sup['price']:,} ({sup['distance_pct']:+.1f}%)"
                )

            # 추가 레벨 (주간, 월간)
            levels = plv.get("levels", {})
            extra = []
            for lkey in ["prev_week_high", "prev_week_low", "prev_month_high", "prev_month_low"]:
                price = levels.get(lkey)
                dist = distances.get(f"to_{lkey}", 0)
                if price and lkey not in [
                    res.get("level", "") if res else "",
                    sup.get("level", "") if sup else "",
                ]:
                    lname = level_names.get(lkey, lkey)
                    extra.append(f"{lname} {price:,}({dist:+.1f}%)")
            if extra:
                lines.append(f"  📏 {' │ '.join(extra[:3])}")

        # Equal Levels
        elv = equal_map.get(ticker)
        if elv:
            for eq in (elv.get("equal_lows") or [])[:2]:
                star = " ★" if eq.get("strength") == "strong" else ""
                dist = eq["distance_pct"]
                lines.append(
                    f"  ⚖️ EqLow {eq['price_center']:,} "
                    f"x{eq['touches']}({dist:+.1f}%){star}"
                )
            for eq in (elv.get("equal_highs") or [])[:2]:
                star = " ★" if eq.get("strength") == "strong" else ""
                dist = eq["distance_pct"]
                lines.append(
                    f"  ⚖️ EqHigh {eq['price_center']:,} "
                    f"x{eq['touches']}({dist:+.1f}%){star}"
                )

        # ── 액션 신호 판정 ──
        actions = []

        if plv:
            res = plv.get("nearest_resistance")
            sup = plv.get("nearest_support")

            # 저항 근접 (2% 이내) → 부분익절 고려
            if res and 0 < res["distance_pct"] <= 2.0:
                actions.append(f"⚡ 저항 근접 ({res_name} {res['distance_pct']:+.1f}%) → 부분익절 고려")

            # 저항 돌파 → 추세 전환 가능
            if res and res["distance_pct"] < 0:
                actions.append(f"🚀 저항 돌파 ({res_name}) → 추세 강화")

            # 지지 근접 (2% 이내) → 추가매수 관심
            if sup and -2.0 <= sup["distance_pct"] < 0:
                actions.append(f"🛒 지지 근접 ({sup_name} {sup['distance_pct']:+.1f}%) → 반등 관찰")

            # 지지 이탈 → 손절 경고
            if sup and sup["distance_pct"] > 0:
                # 현재가가 지지 아래
                pass
            # 주간저 근접/이탈
            wk_low = levels.get("prev_week_low", 0)
            wk_low_dist = distances.get("to_prev_week_low", 0)
            if wk_low and -3.0 <= wk_low_dist <= 0:
                actions.append(f"⚠️ 주간저 근접 ({wk_low:,}, {wk_low_dist:+.1f}%)")
            if wk_low and wk_low_dist > 0:
                actions.append(f"🚨 주간저 이탈! ({wk_low:,})")

        # Equal Level 근접 경고
        if elv:
            for eq in (elv.get("equal_lows") or [])[:2]:
                dist = eq["distance_pct"]
                if -3.0 <= dist <= 0:
                    touches = eq["touches"]
                    actions.append(
                        f"🎯 EqLow 유동성 x{touches} 근접 ({dist:+.1f}%) → 스윕 주의"
                    )
            for eq in (elv.get("equal_highs") or [])[:2]:
                dist = eq["distance_pct"]
                if -2.0 <= dist <= 0:
                    actions.append(
                        f"🎯 EqHigh 돌파 임박 ({dist:+.1f}%) → 상방 트리거"
                    )

        # 손실 경고
        if pnl_pct <= -15:
            actions.append(f"🔴 손실 -15% 초과 → 손절 검토 필수")
        elif pnl_pct <= -10:
            actions.append(f"🟠 손실 -10% 초과 → 경계 수준")

        # 이익 구간
        if pnl_pct >= 15:
            actions.append(f"🟢 +15% 이상 → 트레일링 스탑 권장")
        elif pnl_pct >= 10:
            actions.append(f"🟢 +10% 이상 → 부분익절 고려")

        if actions:
            lines.append(f"  ─── 액션 ───")
            for a in actions:
                lines.append(f"  {a}")
        else:
            lines.append(f"  ─── 액션 ───")
            lines.append(f"  ✅ HOLD — 특이사항 없음")

    # ── 5분봉 상태판 (하단 별도) ──
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    date_label = candle_date or "?"
    lines.append(f"🕯 5분봉 상태판 ({date_label})")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

    for h in holdings:
        ticker = h["ticker"]
        name = h["name"]
        current = h["current_price"]
        pnl_pct = h["pnl_pct"]

        # 손익 이모지
        if pnl_pct >= 10:
            dot = "🟢"
        elif pnl_pct >= 0:
            dot = "🔵"
        elif pnl_pct >= -5:
            dot = "🟡"
        elif pnl_pct >= -10:
            dot = "🟠"
        else:
            dot = "🔴"

        # 5분봉 지표
        intra = _calc_5min_indicators(ticker, candle_date, current) if candle_date else {}
        pat = intra.get("pattern", "")
        vwap = intra.get("vwap")
        vwap_pos = intra.get("vwap_pos")

        # VWAP
        if vwap:
            vp = "▲" if vwap_pos == "above" else "▼"
            vd = round((current - vwap) / vwap * 100, 1)
            vwap_str = f"V{vp}{abs(vd):.1f}%"
        else:
            vwap_str = "V─"

        # OR/IR
        orir = orir_map.get(ticker)
        if orir:
            bias = orir.get("daily_bias", "?")
            b = {"bullish": "↑", "bearish": "↓", "neutral": "─"}.get(bias, "?")
            close_vs = orir.get("close_vs_or", "?")
            cv = {"above": "▲", "below": "▼", "inside": "─"}.get(close_vs, "?")
            ir_pct = orir.get("ir_range_pct", 0)
            or_str = f"OR{b}{cv}"
            ir_str = f"IR{ir_pct:.1f}%"
        else:
            or_str = "OR─"
            ir_str = "IR─"

        # 패턴 태그
        tag = ""
        if pat == "pullback_bounce":
            tag = "  반등↑"
        elif pat == "trend_continue":
            tag = "  양봉↑"
        elif pat == "gap_fail":
            tag = "  음봉↓"

        # ⭐ 별표: 조건 카운트
        stars = 0
        if vwap_pos == "above" and vwap and abs(round((current - vwap) / vwap * 100, 1)) >= 2.0:
            stars += 1  # VWAP 위 + 괴리 2%+
        if orir and orir.get("daily_bias") == "bullish":
            stars += 1  # OR 상방 돌파
        if orir and orir.get("close_vs_or") == "above":
            stars += 1  # 종가 OR 위
        if pat in ("pullback_bounce", "trend_continue"):
            stars += 1  # 긍정 패턴

        star_str = f"  ⭐{stars}" if stars >= 2 else ""

        # 손절?/익절? 태그
        if pnl_pct <= -15:
            tag = "  손절?"
        elif pnl_pct >= 15:
            tag = "  익절?"

        short_name = name[:5]
        lines.append(
            f"{dot} {short_name:<5} {pnl_pct:+.1f}%"
            f"  {vwap_str}  {or_str}  {ir_str}{star_str}{tag}"
        )

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🔄 매일 장마감 후 자동 발송")

    text = "\n".join(lines)
    print(text)
    print(f"\n--- {len(text)} chars ---")

    send_message(text)
    print("텔레그램 전송 완료!")


if __name__ == "__main__":
    main()
