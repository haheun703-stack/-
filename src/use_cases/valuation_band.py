"""src/use_cases/valuation_band.py — 밸류 밴드 관측 패널 (영상 손주부 + 우리 보정).

영상(전업투자자)의 "시총 1~30위 우량주를 쌀 때 사라" + EPS/ROE/PER/PBR 필터를 살리되,
영상이 못 보는 세 가지를 더한다:
  1. 이익 방향(사이클 프록시): PER vs 컨센서스(추정) PER. 추정<현재 = 이익 성장(사이클 저점),
     추정>현재 = 이익 감소(사이클 정점 = "PER 거꾸로 함정"). 반도체에 필수.
  2. ★FCF(잉여현금흐름): ROE가 진짜인지 부채로 부풀렸는지의 결정타. ROE 높아도 FCF 음수면
     (오라클형 — CAPEX로 현금 태우며 자사주매입+레버리지로 ROE 부풀림) = 가치함정. ROE는
     분식 가능하지만 FCF는 못 속인다. 영상의 최대 맹점.
  3. 부채(순부채/시총·D/E): FCF와 함께 가치함정을 확정.

★관측 전용 — hard gate 아님, 실주문 0, classify_tier 무접촉. 6/10 sector_momentum_label과
  동일 계층(관측 라벨). 매일 기록은 별도(--write, 6/11 finality 교훈).

데이터: 미국=yfinance(전부). 한국=네이버 모바일 API(PER/추정PER/PBR) + yfinance .KS(ROE/FCF/
  순부채/시총/52주 — 한국은 trailingPE/PBR을 None으로 주므로 네이버 보완). 시총 top 30 정렬.
  SSL은 인증서 한글경로(…퀀트봇…)를 curl이 못 읽어 ASCII 임시경로로 복사해 우회(error 77).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, replace

TOP_N = 30

# 데이터계약 §1 표준 verdict 5종(정보봇 검증 대상). 내부 상세 라벨은 verdict()가 보유하되
# 대시보드 적재는 이 5종으로 정규화한다(웹 렌더 분기 단순화).
STANDARD_VERDICTS = ("저점후보", "가치함정", "이미오름", "관망", "데이터부족")

# 시총 상위 후보 풀(marketCap으로 정렬해 top 30 추림 — 순위 변동 흡수)
US_POOL = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
    "JPM", "V", "WMT", "XOM", "UNH", "ORCL", "MA", "HD", "PG", "COST",
    "JNJ", "NFLX", "ABBV", "BAC", "KO", "CVX", "CRM", "AMD", "TMUS", "WFC",
    "PEP", "LIN", "MRK", "ACN", "MCD",
]
# 미래가치 US 유니버스 리더(US_POOL 미포함 반도체·제약·에너지·산업재 — 애널리스트 커버 우량
#   ADR 포함). scan_consensus_us·valuation_band_history_us 공용(클린아키텍처: use_cases 소유,
#   scripts 역참조 금지). basis 가드가 해외통화 ADR(TSM=TWD·NVO=DKK·TM=JPY) PER밴드는 제외.
US_FV_LEADERS = ["TSM", "ASML", "MU", "QCOM", "PFE", "LMT", "RTX", "GM", "FCX", "SHEL", "TM", "NVO"]


def us_fv_universe() -> list[str]:
    """US 미래가치 유니버스(US_POOL 시총상위 + 리더, 중복제거·순서보존)."""
    seen: set[str] = set()
    out: list[str] = []
    for tk in list(US_POOL) + US_FV_LEADERS:
        if tk not in seen:
            seen.add(tk)
            out.append(tk)
    return out


KR_POOL = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"), ("005380", "현대차"), ("000270", "기아"),
    ("068270", "셀트리온"), ("105560", "KB금융"), ("035420", "NAVER"),
    ("012450", "한화에어로스페이스"), ("329180", "HD현대중공업"), ("005490", "POSCO홀딩스"),
    ("055550", "신한지주"), ("028260", "삼성물산"), ("012330", "현대모비스"),
    ("086790", "하나금융지주"), ("035720", "카카오"), ("000810", "삼성화재"),
    ("015760", "한국전력"), ("042660", "한화오션"), ("009540", "HD한국조선해양"),
    ("051910", "LG화학"), ("006400", "삼성SDI"), ("032830", "삼성생명"),
    ("010130", "고려아연"), ("138040", "메리츠금융지주"), ("096770", "SK이노베이션"),
    ("034020", "두산에너빌리티"), ("003670", "포스코퓨처엠"), ("259960", "크래프톤"),
    ("018260", "삼성에스디에스"), ("010140", "삼성중공업"), ("011200", "HMM"),
    ("316140", "우리금융지주"), ("024110", "기업은행"),
]

# 보험·은행·금융지주 — FCF 개념이 일반기업과 달라(보험영업현금흐름·예대 구조) yfinance
# freeCashflow가 무의미/왜곡됨(예: 삼성생명 fcf_yield −57.8 / 삼성화재 −26.3). 웹 오노출 방지로
# fcf=None 처리(→ fcf_yield None → 웹 graceful "—"). verdict 무영향(전부 ROE<15 → 관망).
# 6/17 정보봇 FCF 주의 인계(§3) 반영.
FCF_NA_TICKERS = {
    "105560", "055550", "086790", "138040", "316140", "024110",  # 은행·금융지주
    "000810", "032830",                                          # 보험(삼성화재·삼성생명)
}


def _setup_ssl() -> None:
    """yfinance/requests curl SSL — 한글 경로 cacert를 ASCII 임시경로로 복사해 우회(error 77)."""
    try:
        import certifi
        ascii_ca = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "cacert_ascii.pem")
        if not os.path.exists(ascii_ca):
            shutil.copy(certifi.where(), ascii_ca)
        for k in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
            os.environ[k] = ascii_ca
    except Exception:  # noqa: BLE001
        pass


@dataclass(frozen=True)
class ValuationSnapshot:
    market: str
    ticker: str
    name: str
    price: float | None
    per: float | None
    fwd_per: float | None
    pbr: float | None
    roe: float | None              # %
    pos_52w: float | None          # 0~100
    debt_to_equity: float | None   # 미국 yfinance / 한국 None
    fcf: float | None              # 자국 통화(미국 $, 한국 원)
    net_debt: float | None         # 순부채 = totalDebt - totalCash (자국 통화)
    market_cap: float | None

    @property
    def earnings_up(self) -> bool | None:
        if self.per and self.fwd_per and self.per > 0 and self.fwd_per > 0:
            return self.fwd_per < self.per
        return None

    @property
    def fcf_yield(self) -> float | None:
        """FCF 수익률 = FCF / 시총 × 100. 통화 무관 비교(음수 = 현금 태우는 중)."""
        if self.fcf is not None and self.market_cap:
            return round(self.fcf / self.market_cap * 100, 1)
        return None

    def verdict(self) -> str:
        """영상 필터 + 우리 보정(FCF·이익방향·부채). 'ROE 높은데 떨어진'을 옥/석으로 가른다."""
        if self.roe is None or self.pos_52w is None:
            return "데이터부족"
        high_roe = self.roe >= 15.0
        fcf_neg = self.fcf is not None and self.fcf < 0
        # ★FCF 음수 + 고ROE = 가치함정(밴드 위치 무관 — 함정은 어디든). 오라클형.
        if high_roe and fcf_neg:
            return "가치함정·FCF음수"
        low_band = self.pos_52w <= 35.0
        if not (high_roe and low_band):
            if high_roe and self.pos_52w >= 80.0:
                return "고ROE·이미오름"
            return "관망"
        # ROE 높고 52주 하단 = 사장님 관심 구간 → 옥/석 분리
        if self.earnings_up is False:
            return "가치함정·이익↓"
        if self.earnings_up is True:
            return "저점후보·이익↑"
        return "저점관찰"


# ═══════════════════════════════════════════════════
# 대시보드 적재 헬퍼 (순수 함수 — I/O 없음, freeze 무손상)
#   dashboard_valuation_band 테이블(정보봇 데이터계약 §1) 행 변환.
# ═══════════════════════════════════════════════════

def verdict_category(detail: str) -> str:
    """내부 상세 verdict → 데이터계약 표준 5종 정규화.

    상세: 데이터부족 / 가치함정·FCF음수 / 가치함정·이익↓ / 고ROE·이미오름 /
          저점후보·이익↑ / 저점관찰 / 관망
    """
    if detail.startswith("데이터부족"):
        return "데이터부족"
    if detail.startswith("가치함정"):
        return "가치함정"
    if "이미오름" in detail:
        return "이미오름"
    if detail.startswith("저점후보") or detail.startswith("저점관찰"):
        return "저점후보"
    return "관망"


def source_of(snap: ValuationSnapshot) -> str:
    """데이터 출처 표기(데이터계약 §1 source). US=yfinance, KR=네이버(PER/PBR 주소스)."""
    return "yfinance" if snap.market == "US" else "naver"


def apply_checkup(
    snaps: list[ValuationSnapshot], checkup_by_code: dict[str, dict]
) -> list[ValuationSnapshot]:
    """checkup(quant_bluechip_checkup LIVE) 재활용 — 데이터계약 §1 하이브리드.

    per/pbr/pos_52w/price가 비어있을 때 checkup 값으로 보완(중복 수집 회피). per/pbr이 채워지면
    roe도 None일 때 pbr/per로 근사(fetch_kr 동일 규칙).

    ★6/16 교정: checkup은 매일 09:3X LIVE 적재(stale 아님, per/pbr 22/30 유효 — 정보봇 검증).
      한국 종목은 yfinance.KS rate-limit으로 per/pbr/pos가 비기 쉬운데 checkup(네이버 기반)이
      이를 보완 → 데이터부족 탈출. 호출처가 stale 가드 후 {code: {per,pbr,position_pct,price}}를 넘긴다.
    """
    if not checkup_by_code:
        return snaps
    out: list[ValuationSnapshot] = []
    for s in snaps:
        ck = checkup_by_code.get(s.ticker)
        if not ck:
            out.append(s)
            continue
        per = s.per or ck.get("per")          # 0/None이면 checkup으로 보완
        pbr = s.pbr or ck.get("pbr")
        price = s.price or ck.get("price")
        pos = s.pos_52w if s.pos_52w is not None else ck.get("position_pct")
        roe = s.roe
        if roe is None and per and pbr and per > 0:
            roe = round(pbr / per * 100, 1)   # fetch_kr 동일 근사
        out.append(replace(
            s, per=per, pbr=pbr, price=price, roe=roe,
            pos_52w=round(float(pos), 1) if pos is not None else None,
        ))
    return out


def to_dashboard_row(snap: ValuationSnapshot, date_str: str, snapshot_iso: str) -> dict:
    """ValuationSnapshot → dashboard_valuation_band 행(데이터계약 §1 컬럼)."""
    return {
        "date": date_str,
        "market": snap.market,
        "ticker": snap.ticker,
        "name": snap.name,
        "price": snap.price,
        "per": snap.per,
        "fwd_per": snap.fwd_per,
        "pbr": snap.pbr,
        "roe": snap.roe,
        "pos_52w": snap.pos_52w,
        "fcf_yield": snap.fcf_yield,
        "debt_to_equity": snap.debt_to_equity,
        "verdict": verdict_category(snap.verdict()),
        "earnings_up": snap.earnings_up,
        "source": source_of(snap),
        "snapshot_time": snapshot_iso,
    }


def fetch_us(symbols: list[str]) -> list[ValuationSnapshot]:
    _setup_ssl()
    import yfinance as yf
    out: list[ValuationSnapshot] = []
    for s in symbols:
        try:
            i = yf.Ticker(s).info
            px = i.get("currentPrice") or i.get("regularMarketPrice")
            hi, lo = i.get("fiftyTwoWeekHigh"), i.get("fiftyTwoWeekLow")
            pos = (px - lo) / (hi - lo) * 100 if (px and hi and lo and hi > lo) else None
            roe = i.get("returnOnEquity")
            debt, cash = i.get("totalDebt"), i.get("totalCash")
            nd = (debt - cash) if (debt is not None and cash is not None) else None
            out.append(ValuationSnapshot(
                "US", s, (i.get("shortName", s) or s)[:14], px,
                i.get("trailingPE"), i.get("forwardPE"), i.get("priceToBook"),
                round(roe * 100, 1) if roe is not None else None,
                round(pos, 1) if pos is not None else None,
                i.get("debtToEquity"), i.get("freeCashflow"), nd, i.get("marketCap")))
        except Exception:  # noqa: BLE001
            out.append(ValuationSnapshot("US", s, s, *([None] * 10)))
    return out


def _kr_num(v) -> float | None:
    if not v:
        return None
    try:
        return float(str(v).replace(",", "").replace("배", "").replace("원", "").replace("%", "").strip())
    except Exception:  # noqa: BLE001
        return None


def fetch_kr(codes: list[tuple[str, str]]) -> list[ValuationSnapshot]:
    """한국 — 네이버(PER/추정PER/PBR) + yfinance .KS(ROE/FCF/순부채/시총/52주)."""
    _setup_ssl()
    import requests
    import yfinance as yf
    H = {"User-Agent": "Mozilla/5.0"}
    out: list[ValuationSnapshot] = []
    for code, name in codes:
        try:
            ig = requests.get(f"https://m.stock.naver.com/api/stock/{code}/integration",
                              headers=H, timeout=15).json()
            ti = {x.get("code"): x.get("value") for x in ig.get("totalInfos", []) if isinstance(x, dict)}
            per, pbr, cns = _kr_num(ti.get("per")), _kr_num(ti.get("pbr")), _kr_num(ti.get("cnsPer"))
            yi = yf.Ticker(f"{code}.KS").info
            px = yi.get("currentPrice") or yi.get("regularMarketPrice")
            roe = yi.get("returnOnEquity")
            roe_pct = round(roe * 100, 1) if roe is not None else (
                round(pbr / per * 100, 1) if (per and pbr and per > 0) else None)
            hi, lo = yi.get("fiftyTwoWeekHigh"), yi.get("fiftyTwoWeekLow")
            pos = (px - lo) / (hi - lo) * 100 if (px and hi and lo and hi > lo) else None
            debt, cash = yi.get("totalDebt"), yi.get("totalCash")
            nd = (debt - cash) if (debt is not None and cash is not None) else None
            fcf_val = None if code in FCF_NA_TICKERS else yi.get("freeCashflow")  # 보험·금융 FCF 무의미
            out.append(ValuationSnapshot(
                "KR", code, name, px, per, cns, pbr,
                roe_pct, round(pos, 1) if pos is not None else None,
                None, fcf_val, nd, yi.get("marketCap")))
        except Exception:  # noqa: BLE001
            out.append(ValuationSnapshot("KR", code, name, *([None] * 10)))
    return out


def _top_by_cap(snaps: list[ValuationSnapshot], n: int) -> list[ValuationSnapshot]:
    have = [s for s in snaps if s.market_cap]
    have.sort(key=lambda s: s.market_cap, reverse=True)
    return have[:n]


def _f(v, suf="", nd=1):
    return "-" if v is None else f"{v:.{nd}f}{suf}"


def print_group(title, rows):
    if not rows:
        return
    print(f"\n── {title} ({len(rows)}) ──")
    for s in rows:
        print(f"  {s.market} {s.name[:14]:<14} ROE {_f(s.roe):>5}% · 52주 {_f(s.pos_52w):>5}% · "
              f"PER {_f(s.per)}→{_f(s.fwd_per)} · FCF수익 {_f(s.fcf_yield):>5}% → {s.verdict()}")


def main():
    print("[밸류밴드] 미국·한국 시총 후보 조회 중...")
    us = _top_by_cap(fetch_us(US_POOL), TOP_N)
    kr = _top_by_cap(fetch_kr(KR_POOL), TOP_N)
    snaps = us + kr
    jade = [s for s in snaps if s.verdict().startswith("저점후보")]
    trap = [s for s in snaps if s.verdict().startswith("가치함정")]
    risen = [s for s in snaps if "이미오름" in s.verdict()]
    watch = [s for s in snaps if "저점관찰" in s.verdict()]
    print(f"\n미국 top{len(us)} · 한국 top{len(kr)} = {len(snaps)}종목")
    print_group("저점후보 (좋은데 싸고 이익↑·FCF+)", jade)
    print_group("가치함정 (FCF음수 / 이익↓)", trap)
    print_group("고ROE·이미오름 (좋지만 밴드 상단 — 삼성·SK 등)", risen)
    print_group("저점관찰 (싸지만 이익방향 불명)", watch)
    return snaps


if __name__ == "__main__":
    main()
