# 🌙 NIGHTWATCH — JARVIS 야간 수호자 모듈

> **"미국장이 열리기 전, 밤을 지킨다"**  
> 매일 17:00 자동 실행 → NXT 매수/관망/금지 신호 → 텔레그램 + Control Tower 발송

---

## 📐 설계 철학

```
기존 트레이더:  미국 선물만 보고 NXT 진입
NIGHTWATCH:    7개 레이어를 통과한 신호만 진입 허용

같은 🟢라도 진짜 🟢 vs 가짜 🟢를 구별한다
```

---

## 🏗️ 전체 구조도

```
17:00 자동 실행
        │
        ▼
┌───────────────────────────────┐
│  LAYER 0  💎 선행지표          │  ← 시장이 움직이기 전에 먼저 안다
│  PCR / HYG / CNH / USD/JPY   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 1  ⭐⭐⭐ 채권 자경단    │  ← 트럼프의 진짜 상사
│  10년물 금리 / 30년물 금리     │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 2  ⭐⭐⭐ 공포 & 달러   │
│  VIX / DXY                   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 3  ⭐⭐ 선물 방향       │
│  S&P500 선물 / 나스닥 선물     │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 4  💎 외국인 국적별 수급 │  ← 차이나머니 추적 핵심
│  🇺🇸미국 / 🇨🇳중국 / 🇪🇺유럽 / 🇯🇵일본 │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 5  ⭐⭐ 환율 삼각형     │
│  원/달러 / USD/JPY / CNH       │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 6  ⭐ 이벤트 리스크     │
│  경제지표 일정 / 트럼프 뉴스   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  LAYER 7  🎯 최종 판단         │
│  🟢🟢 / 🟢 / 🟡 / 🔴 / 💀    │
│  NXT 액션 + 주목 섹터 + 종목   │
└───────────────────────────────┘
        │
        ▼
   텔레그램 + Control Tower
```

---

## 📊 레이어별 상세 기준

### LAYER 0 — 💎 선행지표 (가장 먼저 움직임)

| 지표 | 설명 | 🟢 | 🟡 | 🔴 |
|------|------|----|----|-----|
| **PCR** | Put/Call Ratio | 0.7 이하 | 0.7~1.0 | 1.0 이상 |
| **HYG** | 하이일드 채권 ETF | +0.3% 이상 | ±0.3% | -0.3% 이하 |
| **CNH** | 역외 위안화 | 7.3 이하 | 7.3~7.5 | 7.5 이상 |
| **USD/JPY** | 엔캐리 트레이드 | 145 이상 안정 | 142~145 | 142 이하 |

> 💡 **핵심 원리**
> - PCR: 스마트머니는 주식 전에 옵션에서 먼저 움직인다
> - HYG: 크레딧이 배신하면 주식도 곧 배신한다
> - CNH: 위안화 강세 = 차이나머니 해외 투자 여력 증가
> - USD/JPY: 엔캐리 청산 = 전 세계 동시 자금 회수 (2024년 8월 코스피 -8.77%)

---

### LAYER 1 — ⭐⭐⭐ 채권 자경단

| 지표 | 🟢 | 🟡 | 🔴 |
|------|----|----|-----|
| **^TNX** 10년물 | -0.05% 이하 | ±0.05% | +0.10% 이상 |
| **^TYX** 30년물 | 10년물과 동반 하락 | 한쪽만 변화 | 10년물과 동반 상승 |

> 💡 **판단 공식**
> - 주식↓ + 금리↑ = 🔴 최악 (자경단 발동, NXT 절대 금지)
> - 주식↓ + 금리↓ = 🟡 일반 리스크오프
> - 주식↑ + 금리 안정 = 🟢 정상 랠리

---

### LAYER 2 — ⭐⭐⭐ 공포 & 달러

| 지표 | 🟢 | 🟡 | 🔴 | 💀 |
|------|----|----|-----|-----|
| **VIX** | 20 이하 | 20~25 | 25 이상 | 30 이상 |
| **DXY** | 약세 (-0.3% 이하) | ±0.3% | 강세 (+0.5% 이상) | 급등 |

---

### LAYER 3 — ⭐⭐ 선물 방향

| 지표 | 🟢 | 🟡 | 🔴 |
|------|----|----|-----|
| **ES** S&P500 선물 | +0.5% 이상 | ±0.5% | -0.5% 이하 |
| **NQ** 나스닥 선물 | +0.5% 이상 | ±0.5% | -0.5% 이하 |

---

### LAYER 4 — 💎 외국인 국적별 수급

| 국적 | 의미 | 해석 |
|------|------|------|
| 🇺🇸 **미국계** | 장기 펀드, 추세 신뢰도 높음 | 미국장 방향과 동조 확인 |
| 🇨🇳 **중국계** | 단기성, 정책 민감, 홍콩 경유 | 중국 내부 신호 선반영 |
| 🇪🇺 **유럽계** | 매크로 민감 | 글로벌 흐름 확인 |
| 🇯🇵 **일본계** | 엔캐리와 연동 | USD/JPY와 함께 봐야 |

| 조합 | 신호 |
|------|------|
| 세 개 동시 매수 | 🟢🟢 |
| 미국+유럽 매수, 중국 매도 | 🟡 중국 뭔가 있음 |
| 중국만 단독 매수 | 🟢 차이나 내부 호재 |
| 세 개 동시 매도 | 🔴🔴 탈출 |

---

### LAYER 5 — ⭐⭐ 환율 삼각형

| 지표 | 🟢 | 🟡 | 🔴 |
|------|----|----|-----|
| **원/달러** | 1,400 이하 + 하락 | 1,400 근방 | 1,400 이상 + 상승 |
| **USD/JPY** | 145 이상 안정 | 142~145 | 142 이하 하락 |
| **CNH** | 7.3 이하 | 7.3~7.5 | 7.5 이상 |

---

### LAYER 6 — ⭐ 이벤트 리스크

| 이벤트 | 신호 |
|--------|------|
| 오늘밤 CPI/PPI/FOMC/고용지표 | 🔴 변동성 주의 |
| 연준 위원 발언 예정 | 🟡 주의 |
| 트럼프 관세 위협 뉴스 | 🔴 |
| 트럼프 딜 타결 뉴스 | 🟢 |
| 이벤트 없음 | 🟢 |

---

### LAYER 7 — 최종 판단 매트릭스

| 조합 | 신호 | NXT 액션 |
|------|------|----------|
| L0🟢 + L1🟢 + L2🟢 + L3🟢 | 🟢🟢 | **강한 매수** |
| 대부분 🟢, 일부 🟡 | 🟢 | **매수 고려** |
| 혼조 (🟢+🔴 섞임) | 🟡 | **관망** |
| 이벤트 예정 있음 | 🟡 | **관망** |
| L0🔴 or L1🔴 발생 | 🔴 | **진입 금지** |
| L1🔴🔴 + L2💀 | 💀 | **전체 포지션 점검** |

---

### 🎯 신호별 주목 종목

| 신호 | 섹터 | 종목 |
|------|------|------|
| 🟢🟢 강한 상승 | 반도체 | SK하이닉스, 삼성전자 |
| 🟢🟢 강한 상승 | 방산 | 한화에어로스페이스, 현대로템, LIG넥스원 |
| 🟢 중간 상승 | 조선 | HD현대중공업 |
| 🟢 + CNH강세 | 2차전지 | POSCO홀딩스, 에코프로비엠 |
| 🟢 + 금 상승 | 금 관련 | 고려아연, KODEX 골드선물 |
| 🔴 | 인버스 | KODEX 200선물인버스2X |

---

## 💻 구현 코드

```python
"""
🌙 NIGHTWATCH - JARVIS 야간 수호자 모듈
매일 17:00 자동 실행
"""

import yfinance as yf
import requests
import json
from datetime import datetime, date
import os

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


# ──────────────────────────────────────────
# LAYER 0: 선행지표 수집
# ──────────────────────────────────────────
def get_leading_indicators():
    result = {}

    tickers = {
        "PCR": None,          # 별도 계산 필요
        "HYG": "HYG",         # 하이일드 채권 ETF
        "CNH": "CNHUSD=X",    # 역외 위안화
        "USDJPY": "JPY=X",    # 엔캐리
    }

    for key, ticker in tickers.items():
        if ticker is None:
            continue
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                prev = data["Close"].iloc[-2]
                curr = data["Close"].iloc[-1]
                pct = ((curr - prev) / prev) * 100
                result[key] = {"value": round(curr, 4), "change_pct": round(pct, 3)}
        except Exception as e:
            result[key] = {"value": None, "change_pct": None, "error": str(e)}

    return result


# ──────────────────────────────────────────
# LAYER 1: 채권 자경단
# ──────────────────────────────────────────
def get_bond_vigilante():
    result = {}

    bonds = {
        "TNX": "^TNX",   # 10년물
        "TYX": "^TYX",   # 30년물
    }

    for key, ticker in bonds.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                prev = data["Close"].iloc[-2]
                curr = data["Close"].iloc[-1]
                change = round(curr - prev, 3)
                result[key] = {"value": round(curr, 3), "change": change}
        except Exception as e:
            result[key] = {"value": None, "change": None, "error": str(e)}

    return result


# ──────────────────────────────────────────
# LAYER 2: 공포 & 달러
# ──────────────────────────────────────────
def get_fear_dollar():
    result = {}

    tickers = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
    }

    for key, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                prev = data["Close"].iloc[-2]
                curr = data["Close"].iloc[-1]
                pct = ((curr - prev) / prev) * 100
                result[key] = {"value": round(curr, 2), "change_pct": round(pct, 2)}
        except Exception as e:
            result[key] = {"value": None, "change_pct": None, "error": str(e)}

    return result


# ──────────────────────────────────────────
# LAYER 3: 선물 방향
# ──────────────────────────────────────────
def get_futures():
    result = {}

    futures = {
        "ES": "ES=F",    # S&P500 선물
        "NQ": "NQ=F",    # 나스닥 선물
    }

    for key, ticker in futures.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                prev = data["Close"].iloc[-2]
                curr = data["Close"].iloc[-1]
                pct = ((curr - prev) / prev) * 100
                result[key] = {"value": round(curr, 2), "change_pct": round(pct, 2)}
        except Exception as e:
            result[key] = {"value": None, "change_pct": None, "error": str(e)}

    return result


# ──────────────────────────────────────────
# LAYER 5: 환율
# ──────────────────────────────────────────
def get_fx():
    result = {}

    fx = {
        "USDKRW": "KRW=X",
        "USDJPY": "JPY=X",
        "USDCNH": "CNHUSD=X",
    }

    for key, ticker in fx.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                prev = data["Close"].iloc[-2]
                curr = data["Close"].iloc[-1]
                pct = ((curr - prev) / prev) * 100
                result[key] = {"value": round(curr, 2), "change_pct": round(pct, 3)}
        except Exception as e:
            result[key] = {"value": None, "change_pct": None, "error": str(e)}

    return result


# ──────────────────────────────────────────
# LAYER 6: 트럼프/이벤트 뉴스
# ──────────────────────────────────────────
def get_trump_news():
    if not NEWS_API_KEY:
        return {"risk": "unknown", "headlines": []}

    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q=Trump+tariff+OR+Trump+Fed+OR+Trump+China"
            f"&language=en&sortBy=publishedAt&pageSize=5"
            f"&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        articles = resp.json().get("articles", [])

        risk_keywords = ["tariff", "sanction", "ban", "attack", "threat", "impose"]
        positive_keywords = ["deal", "agreement", "truce", "pause", "relief"]

        risk_score = 0
        headlines = []
        for a in articles:
            title = a.get("title", "").lower()
            headlines.append(a.get("title", ""))
            for kw in risk_keywords:
                if kw in title:
                    risk_score += 1
            for kw in positive_keywords:
                if kw in title:
                    risk_score -= 1

        if risk_score >= 2:
            risk = "HIGH"
        elif risk_score <= -1:
            risk = "LOW"
        else:
            risk = "MEDIUM"

        return {"risk": risk, "score": risk_score, "headlines": headlines[:3]}

    except Exception as e:
        return {"risk": "unknown", "error": str(e), "headlines": []}


# ──────────────────────────────────────────
# 신호 판단 함수
# ──────────────────────────────────────────
def get_signal(value, green_cond, red_cond):
    """범용 신호 판단"""
    if green_cond(value):
        return "🟢"
    elif red_cond(value):
        return "🔴"
    else:
        return "🟡"


def calculate_score(data):
    """
    전체 데이터를 받아 -10 ~ +10 점수 산출
    """
    score = 0

    # LAYER 0: 선행지표
    leading = data.get("leading", {})

    hyg = leading.get("HYG", {}).get("change_pct")
    if hyg is not None:
        if hyg > 0.3:
            score += 2
        elif hyg < -0.3:
            score -= 2

    cnh = leading.get("CNH", {}).get("value")
    if cnh is not None:
        if cnh < 7.3:
            score += 2
        elif cnh > 7.5:
            score -= 2

    usdjpy = leading.get("USDJPY", {}).get("value")
    if usdjpy is not None:
        if usdjpy > 145:
            score += 1
        elif usdjpy < 142:
            score -= 3  # 엔캐리 청산은 강한 마이너스

    # LAYER 1: 채권 자경단
    bonds = data.get("bonds", {})

    tnx_chg = bonds.get("TNX", {}).get("change")
    if tnx_chg is not None:
        if tnx_chg <= -0.05:
            score += 2
        elif tnx_chg >= 0.10:
            score -= 3  # 채권 자경단은 강한 마이너스
        elif tnx_chg >= 0.05:
            score -= 1

    # LAYER 2: VIX
    fear = data.get("fear", {})

    vix = fear.get("VIX", {}).get("value")
    if vix is not None:
        if vix < 18:
            score += 2
        elif vix < 20:
            score += 1
        elif vix < 25:
            score -= 1
        elif vix < 30:
            score -= 2
        else:
            score -= 4  # 패닉

    dxy_pct = fear.get("DXY", {}).get("change_pct")
    if dxy_pct is not None:
        if dxy_pct < -0.3:
            score += 1
        elif dxy_pct > 0.5:
            score -= 2

    # LAYER 3: 선물
    futures = data.get("futures", {})

    es_pct = futures.get("ES", {}).get("change_pct")
    nq_pct = futures.get("NQ", {}).get("change_pct")

    for pct in [es_pct, nq_pct]:
        if pct is not None:
            if pct > 0.5:
                score += 1
            elif pct < -0.5:
                score -= 1

    # LAYER 6: 트럼프 뉴스
    trump = data.get("trump", {})
    trump_risk = trump.get("risk", "MEDIUM")
    if trump_risk == "HIGH":
        score -= 2
    elif trump_risk == "LOW":
        score += 1

    return max(-10, min(10, score))


def score_to_signal(score):
    if score >= 5:
        return "🟢🟢", "강한 매수"
    elif score >= 2:
        return "🟢", "매수 고려"
    elif score >= -1:
        return "🟡", "관망"
    elif score >= -4:
        return "🔴", "진입 금지"
    else:
        return "💀", "전체 포지션 점검"


def get_sector_recommendation(score, data):
    if score < 2:
        return "없음 (관망/금지 구간)"

    leading = data.get("leading", {})
    cnh = leading.get("CNH", {}).get("value", 9999)
    usdjpy = leading.get("USDJPY", {}).get("value", 0)

    recs = []

    if score >= 5:
        recs.append("🔵 반도체: SK하이닉스, 삼성전자")
        recs.append("🟠 방산: 한화에어로스페이스, 현대로템")

    if cnh < 7.3:
        recs.append("🟡 2차전지: POSCO홀딩스, 에코프로비엠")

    if score >= 2:
        recs.append("🔷 조선: HD현대중공업")

    return "\n".join(recs) if recs else "코스피 대형주 전반"


# ──────────────────────────────────────────
# 텔레그램 메시지 생성
# ──────────────────────────────────────────
def build_telegram_message(data, score):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    signal_emoji, signal_text = score_to_signal(score)
    sector = get_sector_recommendation(score, data)

    leading = data.get("leading", {})
    bonds = data.get("bonds", {})
    fear = data.get("fear", {})
    futures = data.get("futures", {})
    fx = data.get("fx", {})
    trump = data.get("trump", {})

    def fmt(d, key, unit="", is_pct=False):
        v = d.get(key, {})
        val = v.get("value") if not is_pct else v.get("change_pct")
        if val is None:
            return "N/A"
        sign = "+" if val > 0 else ""
        return f"{sign}{val}{unit}"

    def sig(value, green_fn, red_fn):
        if value is None:
            return "⬜"
        if green_fn(value):
            return "🟢"
        if red_fn(value):
            return "🔴"
        return "🟡"

    # 각 지표별 신호
    hyg_pct = leading.get("HYG", {}).get("change_pct")
    cnh_val = leading.get("CNH", {}).get("value")
    usdjpy_val = leading.get("USDJPY", {}).get("value")

    hyg_sig = sig(hyg_pct, lambda x: x > 0.3, lambda x: x < -0.3)
    cnh_sig = sig(cnh_val, lambda x: x < 7.3, lambda x: x > 7.5)
    jpy_sig = sig(usdjpy_val, lambda x: x > 145, lambda x: x < 142)

    tnx_chg = bonds.get("TNX", {}).get("change")
    tyx_chg = bonds.get("TYX", {}).get("change")
    tnx_val = bonds.get("TNX", {}).get("value")
    tyx_val = bonds.get("TYX", {}).get("value")
    tnx_sig = sig(tnx_chg, lambda x: x <= -0.05, lambda x: x >= 0.10)
    tyx_sig = sig(tyx_chg, lambda x: x <= -0.05, lambda x: x >= 0.10)

    vix_val = fear.get("VIX", {}).get("value")
    dxy_pct = fear.get("DXY", {}).get("change_pct")
    vix_sig = sig(vix_val, lambda x: x < 20, lambda x: x >= 25)
    dxy_sig = sig(dxy_pct, lambda x: x < -0.3, lambda x: x > 0.5)

    es_pct = futures.get("ES", {}).get("change_pct")
    nq_pct = futures.get("NQ", {}).get("change_pct")
    es_sig = sig(es_pct, lambda x: x > 0.5, lambda x: x < -0.5)
    nq_sig = sig(nq_pct, lambda x: x > 0.5, lambda x: x < -0.5)

    krw_val = fx.get("USDKRW", {}).get("value")
    krw_pct = fx.get("USDKRW", {}).get("change_pct")
    krw_sig = sig(krw_val, lambda x: x < 1400, lambda x: x > 1430)

    trump_risk = trump.get("risk", "MEDIUM")
    trump_sig = "🟢" if trump_risk == "LOW" else ("🔴" if trump_risk == "HIGH" else "🟡")
    headlines = trump.get("headlines", [])
    headline_str = "\n".join([f"  • {h[:50]}..." for h in headlines[:2]]) if headlines else "  • 특이사항 없음"

    msg = f"""🌙 NIGHTWATCH | {now}
━━━━━━━━━━━━━━━━━━━━━━

💎 선행지표
{hyg_sig} HYG: {hyg_pct:+.2f}% (크레딧)
{cnh_sig} CNH: {cnh_val} (위안화)
{jpy_sig} USD/JPY: {usdjpy_val} (엔캐리)

⭐ 채권 자경단
{tnx_sig} 10년물: {tnx_val}% ({tnx_chg:+.3f})
{tyx_sig} 30년물: {tyx_val}% ({tyx_chg:+.3f})

⭐ 공포 & 달러
{vix_sig} VIX: {vix_val}
{dxy_sig} DXY: {dxy_pct:+.2f}%

📈 선물
{es_sig} S&P500: {es_pct:+.2f}%
{nq_sig} 나스닥: {nq_pct:+.2f}%

💱 환율
{krw_sig} 원/달러: {krw_val} ({krw_pct:+.2f}%)

🇨🇳 차이나머니 신호
{cnh_sig} CNH {cnh_val} → {"유입 우호" if cnh_val and cnh_val < 7.3 else "주의 구간"}

⚡ 트럼프/이벤트 리스크
{trump_sig} 리스크: {trump_risk}
{headline_str}

━━━━━━━━━━━━━━━━━━━━━━
종합 점수: {score:+d} / 10
{signal_emoji} {signal_text}

🎯 주목 섹터:
{sector}
━━━━━━━━━━━━━━━━━━━━━━"""

    return msg


# ──────────────────────────────────────────
# 텔레그램 발송
# ──────────────────────────────────────────
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[NIGHTWATCH] 텔레그램 토큰 없음, 콘솔 출력")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("[NIGHTWATCH] 텔레그램 발송 성공")
        else:
            print(f"[NIGHTWATCH] 발송 실패: {resp.text}")
    except Exception as e:
        print(f"[NIGHTWATCH] 텔레그램 오류: {e}")


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────
def run_nightwatch():
    print(f"[NIGHTWATCH] 실행 시작: {datetime.now()}")

    # 데이터 수집
    data = {
        "leading": get_leading_indicators(),
        "bonds": get_bond_vigilante(),
        "fear": get_fear_dollar(),
        "futures": get_futures(),
        "fx": get_fx(),
        "trump": get_trump_news(),
    }

    # 점수 계산
    score = calculate_score(data)

    # 메시지 생성
    message = build_telegram_message(data, score)

    # 발송
    send_telegram(message)

    print(f"[NIGHTWATCH] 완료. 점수: {score}")
    return data, score


if __name__ == "__main__":
    run_nightwatch()
```

---

## ⏰ 자동 실행 설정 (cron)

```bash
# 매일 17:00 실행
0 17 * * 1-5 cd /path/to/jarvis && python nightwatch.py >> logs/nightwatch.log 2>&1
```

---

## 📦 의존성 설치

```bash
pip install yfinance requests python-telegram-bot --break-system-packages
```

---

## 🗂️ JARVIS 생태계 위치

```
JARVIS/
├── v10.3/          스윙 + ETF 로테이션
├── body_hunter/    섹터 과열 감지
├── prophet_agent/  3개월 선행 예측
├── control_tower/  ppwangga.com 대시보드
└── nightwatch/     ← 야간 수호자 (NEW)
    ├── nightwatch.py
    ├── NIGHTWATCH.md
    └── logs/
```

---

## 📝 버전 히스토리

| 버전 | 내용 |
|------|------|
| v1.0 | 최초 설계 - 미국 선물 + VIX + 금리 |
| v1.1 | 채권 자경단 레이어 추가 |
| v1.2 | 외국인 국적별 수급 (차이나머니) 추가 |
| v1.3 | 선행지표 레이어 (PCR/HYG/CNH/JPY) 추가 — **현재** |

---

*🌙 NIGHTWATCH — JARVIS Global Intelligence System*  
*Built by ppwangga | Powered by JARVIS*
