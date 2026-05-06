#!/usr/bin/env python3
"""5/4 종가 기준 NXT 매수후보 분석"""
import json, csv, sys

# 1. supply_surge
with open("data/supply_surge_20260504.json") as f:
    surge = json.load(f)
surge_map = {}
for b in surge["buy_candidates"]:
    surge_map[b["ticker"]] = b

# 2. picks_v2
picks_map = {}
with open("data/picks_v2_20260504.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        picks_map[r["ticker"]] = r

# 3. NXT 후보 스코어링
candidates = []
for tk, s in surge_map.items():
    score = 0
    reasons = []

    # 수급 점수
    tp = s["type"]
    if tp == "A_쌍끌이":
        score += 25
        reasons.append("쌍끌이(외{:.0f}+기{:.0f})".format(s["fgn"], s["inst"]))
    elif tp == "C_3주체합류":
        score += 20
        reasons.append("3주체합류")
    elif tp == "D_외인폭발":
        score += 15
        reasons.append("외인폭발({:.0f})".format(s["fgn"]))
    elif tp == "E_연기금매집":
        score += 15
        reasons.append("연기금매집({:.0f})".format(s["pension"]))

    if s["pension"] > 20:
        score += 8
        reasons.append("연기금+{:.0f}".format(s["pension"]))

    # 기술적 위치
    rsi = s["rsi"]
    ma = s["ma20_dev"]

    if 40 <= rsi <= 65:
        score += 15
        reasons.append("RSI적정({:.0f})".format(rsi))
    elif 30 <= rsi < 40:
        score += 12
        reasons.append("RSI과매도반등({:.0f})".format(rsi))
    elif 65 < rsi <= 75:
        score += 5
        reasons.append("RSI높음({:.0f})".format(rsi))
    else:
        score -= 10
        reasons.append("RSI과열({:.0f})".format(rsi))

    if -5 <= ma <= 5:
        score += 15
        reasons.append("MA20눌림({:+.1f}%)".format(ma))
    elif 5 < ma <= 10:
        score += 10
        reasons.append("MA20초기이격({:+.1f}%)".format(ma))
    elif 10 < ma <= 15:
        score += 3
        reasons.append("MA20이격({:+.1f}%)".format(ma))
    else:
        score -= 5
        reasons.append("MA20과이격({:+.1f}%)".format(ma))

    # 등락률
    ret = s["ret0"]
    if 3 <= ret <= 8:
        score += 10
        reasons.append("적정상승({:+.1f}%)".format(ret))
    elif 8 < ret <= 15:
        score += 5
        reasons.append("강세({:+.1f}%)".format(ret))
    elif ret > 15:
        score += 0
        reasons.append("급등주의({:+.1f}%)".format(ret))
    else:
        score += 3
        reasons.append("소폭({:+.1f}%)".format(ret))

    # picks_v2 보너스
    p = picks_map.get(tk)
    if p:
        ps = float(p.get("score", 0) or 0)
        if ps >= 40:
            score += 10
            reasons.append("picks{:.0f}".format(ps))
        elif ps >= 30:
            score += 5
            reasons.append("picks{:.0f}".format(ps))

    candidates.append({
        "name": s["name"], "ticker": tk,
        "close": s["close"], "ret0": ret,
        "type": tp, "surge_score": s["final_score"],
        "fgn": s["fgn"], "inst": s["inst"], "pension": s["pension"],
        "rsi": rsi, "ma20_dev": ma,
        "nxt_score": score, "reasons": " / ".join(reasons)
    })

candidates.sort(key=lambda x: -x["nxt_score"])

print("=" * 100)
print("  NXT 매수후보 분석 (5/4 종가 기준)")
print("  조건: 수급급변 + 기술적위치(RSI<70,MA20<10%) + 적정등락률")
print("=" * 100)

# A등급
print()
print("### A등급 (NXT 55점+) --- D+1 양봉 확인 시 매수 ###")
for c in candidates:
    if c["nxt_score"] < 55:
        break
    print("  {:>3}pt | {:12s} {} | {:>8,} {:>+5.1f}% | {:12s} | 외{:>7.0f} 기{:>7.0f} 연{:>6.0f} | RSI{:>3.0f} MA{:>+5.1f}%".format(
        c["nxt_score"], c["name"], c["ticker"], c["close"], c["ret0"],
        c["type"], c["fgn"], c["inst"], c["pension"], c["rsi"], c["ma20_dev"]))
    print("        근거: {}".format(c["reasons"]))
    print()

# B등급
print("### B등급 (NXT 40~54점) --- 관찰 후보 ###")
for c in candidates:
    if c["nxt_score"] >= 55 or c["nxt_score"] < 40:
        continue
    print("  {:>3}pt | {:12s} {} | {:>8,} {:>+5.1f}% | {:12s} | 외{:>7.0f} 기{:>7.0f} 연{:>6.0f} | RSI{:>3.0f} MA{:>+5.1f}%".format(
        c["nxt_score"], c["name"], c["ticker"], c["close"], c["ret0"],
        c["type"], c["fgn"], c["inst"], c["pension"], c["rsi"], c["ma20_dev"]))

# C등급
print()
print("### C등급/과열 (40점 미만) --- 추격매수 주의 ###")
cnt = 0
for c in candidates:
    if c["nxt_score"] >= 40:
        continue
    cnt += 1
    if cnt <= 10:
        print("  {:>3}pt | {:12s} {} | {:>8,} {:>+5.1f}% | RSI{:>3.0f} MA{:>+5.1f}% | {}".format(
            c["nxt_score"], c["name"], c["ticker"], c["close"], c["ret0"],
            c["rsi"], c["ma20_dev"], c["type"]))
if cnt > 10:
    print("  ... 외 {}건".format(cnt - 10))
