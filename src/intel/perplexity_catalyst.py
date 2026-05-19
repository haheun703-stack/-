"""Perplexity 종목별 재료(catalyst) 분석기 — 차트영웅 매매법 핵심.

차트영웅 영상 인용:
  "기업과 사료(재료) 둘 다 빡세게 공부 안 되어 있으면 매매하기 어렵다"
  "ICTK 양자내성암호" vs "철강주 호르무즈" — 연속성 차이가 매매 가능성 결정

이 모듈의 역할:
  종목 → Perplexity 호출 → catalyst_summary + 카테고리 + 일회성 판정

정보봇과의 관계:
  정보봇 quant_surge_catalyst (1차 작성 중) → 동일 스키마 호환
  본 모듈 = 우리 자체 백업 + 즉시 사용 가능 + 정보봇 데이터 보강용

비용: $0.0005/종목, 5/22 가동 시 일 5~10종목 → $0.005/일
"""

import json
import os
import re
import requests
import time
from pathlib import Path


# .env 로드
def _load_env():
    if os.getenv("PERPLEXITY_API_KEY"):
        return
    p = Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ[k.strip()] = v.strip()
    except Exception:
        pass

_load_env()


_PPLX_URL = "https://api.perplexity.ai/chat/completions"

# 차트영웅 카테고리 분류 (재료 카테고리)
CATEGORIES = [
    "AI반도체", "HBM", "양자컴퓨팅", "양자내성암호",
    "방산", "원전", "조선", "해운",
    "바이오", "신약", "비만치료제",
    "2차전지", "리튬", "전기차",
    "로봇", "휴머노이드", "자율주행",
    "우주항공", "위성", "드론",
    "엔터", "K-POP", "콘텐츠",
    "건설", "리모델링", "수해복구",
    "금융", "증권", "은행", "보험",
    "유틸리티", "전력", "에너지", "수소",
    "철강", "비철금속", "농업",
    "정책수혜", "내수회복", "환율수혜",
    "단발성호재", "일회성이슈",
]


def analyze_catalyst(ticker: str, name: str, surge_date: str | None = None,
                     model: str = "sonar-pro") -> dict:
    """종목 1개 catalyst 분석.

    Args:
        ticker: '005930'
        name: '삼성전자'
        surge_date: 상한가/급등일 'YYYY-MM-DD' (선택)
        model: 'sonar' (빠름) | 'sonar-pro' (정확)

    Returns:
        {
          ticker, name, date,
          catalyst_summary: str (1줄, 50자),
          catalyst_category: str (CATEGORIES 중 1개),
          is_one_off_event: bool,  # 호르무즈/일회성 이슈?
          continuity_factors: list[str],  # 연속성 근거
          confidence: float,  # 0~1
          source: 'perplexity_sonar_pro'
        }
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {"error": "PERPLEXITY_API_KEY 없음"}

    date_hint = f" ({surge_date} 기준)" if surge_date else ""
    sys_prompt = "JSON 객체만 출력. 마크다운/설명 없이."
    user_prompt = (
        f"{name}({ticker}) 종목의 현재 핵심 재료/투자 포인트{date_hint}를 분석해.\n"
        f"카테고리: {', '.join(CATEGORIES[:20])} 등 중 가장 적합한 것 1개.\n"
        f'JSON: {{"catalyst_summary":"한 줄 50자","catalyst_category":"카테고리",'
        f'"is_one_off_event":boolean,"continuity_factors":["근거1","근거2"],"confidence":소수}}'
    )

    try:
        r = requests.post(
            _PPLX_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.0,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        content = r.json()["choices"][0]["message"]["content"]
        usage = r.json().get("usage", {})

        # JSON 추출
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            return {"error": "JSON 추출 실패", "raw": content[:300]}
        data = json.loads(m.group(0))

        return {
            "ticker": ticker,
            "name": name,
            "date": surge_date or "",
            "catalyst_summary":   data.get("catalyst_summary", "")[:100],
            "catalyst_category":  data.get("catalyst_category", "기타"),
            "is_one_off_event":   bool(data.get("is_one_off_event", False)),
            "continuity_factors": data.get("continuity_factors", []),
            "confidence":         float(data.get("confidence", 0.5)),
            "source": f"perplexity_{model}",
            "tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def compute_continuity_score(catalyst: dict) -> float:
    """재료 연속성 점수 (0~100). 차트영웅 5-Gate Gate 5 가산점.

    공식:
      base = 50 if not one_off else 0
      + continuity_factors 수 × 10
      + confidence × 20
      + 미래산업 카테고리(AI/방산/원전/바이오 등) 가산점 10
      - 일회성 이슈 카테고리 감점 20
    """
    if "error" in catalyst:
        return 0.0
    base = 0 if catalyst.get("is_one_off_event") else 50
    factors_score = min(len(catalyst.get("continuity_factors", [])) * 10, 20)
    conf_score = catalyst.get("confidence", 0.5) * 20
    cat = catalyst.get("catalyst_category", "")
    future_cats = ["AI반도체", "HBM", "양자컴퓨팅", "양자내성암호", "방산",
                   "원전", "조선", "바이오", "신약", "2차전지", "로봇", "휴머노이드"]
    one_off_cats = ["단발성호재", "일회성이슈"]
    cat_score = 10 if cat in future_cats else (-20 if cat in one_off_cats else 0)
    return round(min(max(base + factors_score + conf_score + cat_score, 0), 100), 1)


def analyze_batch(tickers: list[tuple[str, str]], delay: float = 1.0) -> list[dict]:
    """여러 종목 일괄 분석 (rate limit 고려).

    Args:
        tickers: [(ticker, name), ...]
        delay: 호출 사이 대기 (초)
    """
    out = []
    for tk, nm in tickers:
        c = analyze_catalyst(tk, nm)
        c["continuity_score"] = compute_continuity_score(c)
        out.append(c)
        if delay > 0:
            time.sleep(delay)
    return out


if __name__ == "__main__":
    # 사장님 시드 32종목 중 대표 5종목 테스트
    samples = [
        ("005930", "삼성전자"),
        ("000660", "SK하이닉스"),
        ("454910", "두산로보틱스"),
        ("032500", "케이엠더블유"),
        ("012450", "한화에어로스페이스"),
    ]
    print("=== Perplexity catalyst 분석 (사장님 5/19 시드 5종목) ===\n")
    results = analyze_batch(samples, delay=0.5)
    total_tokens = 0
    for r in results:
        if "error" in r:
            print(f"  ❌ {r}")
            continue
        cs = r.get("continuity_score", 0)
        flag = "⭐" if cs >= 60 else ("✓" if cs >= 40 else "✗")
        print(f"{flag} {r['name']:14} [{r['catalyst_category']:10}] 연속성={cs:>5.1f}")
        print(f"   재료: {r['catalyst_summary']}")
        print(f"   일회성={r['is_one_off_event']}, 신뢰도={r['confidence']:.2f}, "
              f"근거={len(r['continuity_factors'])}개")
        total_tokens += r.get("tokens", 0)
        print()
    print(f"총 토큰: {total_tokens}, 예상 비용: ${total_tokens * 0.000001:.4f}")
