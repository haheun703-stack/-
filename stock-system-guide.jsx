import { useState } from "react";

const tabs = [
  { id: "overview", label: "🎯 전체 설계" },
  { id: "feature1", label: "📡 기능1: 실시간 수급" },
  { id: "feature2", label: "🔮 기능2: 분석+점수" },
  { id: "prompts", label: "💬 Claude Code 프롬프트" },
  { id: "prereq", label: "⚙️ 사전 준비" },
];

export default function App() {
  const [tab, setTab] = useState("overview");
  return (
    <div style={{ fontFamily: "'Pretendard', -apple-system, sans-serif", background: "#0F172A", color: "#E2E8F0", minHeight: "100vh", padding: "16px", maxWidth: "820px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center", fontSize: "22px", marginBottom: "4px", background: "linear-gradient(135deg, #F59E0B, #EF4444)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
        주식 실시간 수급 알림 + AI 분석 시스템
      </h1>
      <p style={{ textAlign: "center", color: "#64748B", fontSize: "12px", marginBottom: "16px" }}>한투 API + 네이버 금융 + Claude AI 기반 · v2</p>
      <div style={{ display: "flex", gap: "4px", marginBottom: "16px", justifyContent: "center", flexWrap: "wrap" }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "7px 12px", borderRadius: "8px", border: "none", fontSize: "12px", cursor: "pointer",
            background: tab === t.id ? "#3B82F6" : "#1E293B", color: tab === t.id ? "white" : "#94A3B8",
            fontWeight: tab === t.id ? "600" : "400",
          }}>{t.label}</button>
        ))}
      </div>
      {tab === "overview" && <OverviewTab />}
      {tab === "feature1" && <Feature1Tab />}
      {tab === "feature2" && <Feature2Tab />}
      {tab === "prompts" && <PromptsTab />}
      {tab === "prereq" && <PrereqTab />}
    </div>
  );
}

function OverviewTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="만들려는 것 2가지" color="#F59E0B">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          <MiniCard icon="📡" title="기능 1: 실시간 수급 모니터링" desc="장중 9~15시, 외국인·기관·개인 매매동향 5분 간격 모니터링 → 큰 변화 시 알림" color="#3B82F6" />
          <MiniCard icon="🔮" title="기능 2: AI 종합분석 + 100점" desc="기술적분석 + 수급 + 매매기법 종합 → 내일 예측을 100점 만점 점수화" color="#EF4444" />
        </div>
      </Card>

      <Card title="데이터 소스 (v2 — pykrx 제거)" color="#10B981">
        <div style={{ display: "grid", gap: "6px" }}>
          <SourceBox name="한국투자증권 오픈API" tag="메인" tagColor="#10B981" items={[
            "현재가 / OHLCV 일봉·주봉·월봉 (REST)",
            "당일 분봉 1분/5분/30분/60분 (REST)",
            "실시간 체결가·거래량 스트리밍 (WebSocket)",
            "투자자별 매매동향 — 장 종료 후 확정 (REST)",
          ]} />
          <SourceBox name="네이버 금융 크롤링" tag="보조" tagColor="#3B82F6" items={[
            "장중 외국인/기관 순매매량 (5분 간격 크롤링)",
            "개인 = -(외국인+기관) 추정",
            "무료·로그인 불필요·pd.read_html() 파싱",
          ]} />
          <SourceBox name="Anthropic Claude API" tag="AI" tagColor="#A78BFA" items={[
            "차트·수급·매매기법·모멘텀 부분 점수 산출",
            "내일 예측 + 유지/대응조건 생성",
          ]} />
          <div style={{ background: "#EF444415", padding: "8px", borderRadius: "6px", display: "flex", gap: "6px", alignItems: "center" }}>
            <span style={{ color: "#EF4444" }}>❌</span>
            <span style={{ fontSize: "11px", color: "#FCA5A5" }}><strong>pykrx 사용 안 함</strong> — KRX 로그인 필수화(2025.12.27~) + IP차단 위험</span>
          </div>
        </div>
      </Card>

      <Card title="전체 시스템 구조 (클린 아키텍처)" color="#A78BFA">
        <Code text={`my-stock-system/
├── CLAUDE.md                    ← Claude Code 규칙
├── .env                         ← API 키 (한투, Claude)
├── src/
│   ├── entities/                ← 🟡 핵심 비즈니스 객체
│   │   ├── stock.py             # Stock, PriceData, ChartData
│   │   ├── investor_flow.py     # InvestorFlow, FlowAlert
│   │   ├── technical.py         # 기술적 지표, 패턴
│   │   └── analysis_report.py   # ScoreCategory, AnalysisReport
│   │
│   ├── use_cases/               ← 🔴 비즈니스 로직
│   │   ├── ports.py             # Port 인터페이스
│   │   ├── realtime_monitor.py  # 기능1: 실시간 수급
│   │   └── stock_analysis.py    # 기능2: AI 분석
│   │
│   ├── adapters/                ← 🟢 외부 연결
│   │   ├── kis_api.py           # 한투 REST/WebSocket
│   │   ├── naver_finance.py     # 네이버 금융 크롤링
│   │   ├── technical_adapter.py # 기술지표 (순수 Python)
│   │   ├── claude_api.py        # Claude AI 분석
│   │   └── console_output.py    # 콘솔 출력 (→나중에 텔레그램)
│   └── agents/                  ← 🔵 AI 서브에이전트 (선택)
│
├── scripts/
│   ├── run_monitor.py           # 기능1 실행
│   └── run_analysis.py          # 기능2 실행
└── requirements.txt`} />
      </Card>

      <Card title="알림 전략 (단계적)" color="#6366F1">
        <div style={{ display: "grid", gap: "6px" }}>
          <PhaseBox phase="지금" label="Phase 1~4" desc="ConsoleOutputAdapter → 콘솔에 이모지+구분선으로 출력" color="#10B981" active />
          <PhaseBox phase="나중에" label="Phase 5" desc="TelegramAdapter 추가 → OutputPort 구현체만 교체" color="#64748B" />
          <PhaseBox phase="옵션" label="Phase 5+" desc="Slack·KakaoTalk 등 추가 가능" color="#334155" />
        </div>
        <p style={{ fontSize: "11px", color: "#94A3B8", marginTop: "8px" }}>클린 아키텍처 덕분에 OutputPort 구현체만 바꾸면 됨. 비즈니스 로직 수정 없음!</p>
      </Card>

      <Card title="개발 순서 5단계" color="#F59E0B">
        <div style={{ display: "grid", gap: "4px" }}>
          {[
            { n: "1", text: "엔티티 + 포트 (핵심 구조)", detail: "Stock, InvestorFlow, AnalysisReport + Port 인터페이스", color: "#10B981" },
            { n: "2", text: "한투 API + 네이버 어댑터", detail: "REST로 OHLCV·분봉·투자자별, 네이버 장중 크롤링", color: "#3B82F6" },
            { n: "3", text: "기술분석 + Claude AI 어댑터", detail: "RSI/MACD/볼린저 계산 + AI 점수 산정", color: "#A78BFA" },
            { n: "4", text: "유스케이스 + 실행 스크립트", detail: "모니터링 루프 + 분석 리포트 → 콘솔 출력", color: "#F59E0B" },
            { n: "5", text: "텔레그램 등 알림 연결 (나중에)", detail: "OutputPort 구현체 추가만 하면 끝", color: "#64748B" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: "10px", padding: "8px 10px", background: "#0F172A", borderRadius: "6px", alignItems: "flex-start" }}>
              <span style={{ background: item.color, borderRadius: "50%", width: "22px", height: "22px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "11px", fontWeight: "800", flexShrink: 0, color: "#0F172A" }}>{item.n}</span>
              <div><strong style={{ color: item.color, fontSize: "12px" }}>{item.text}</strong><p style={{ fontSize: "11px", color: "#64748B", margin: "2px 0 0" }}>{item.detail}</p></div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function Feature1Tab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="기능 1: 실시간 수급 모니터링" color="#3B82F6">
        <p style={{ fontSize: "12px" }}>특정 종목의 외국인·기관·개인 매매동향을 장중 모니터링하고, 의미 있는 변화를 감지해서 알려주는 시스템</p>
      </Card>

      <Card title="데이터 수집 흐름 (v2)" color="#10B981">
        <div style={{ display: "grid", gap: "8px" }}>
          <FlowBox title="장중 (9:00~15:30) — 네이버 금융 크롤링" color="#10B981"
            code={`# 5분 간격 크롤링
url = "https://finance.naver.com/item/frgn.naver?code=005930&page=1"
# pd.read_html() → 오늘 외국인/기관 순매매량
# 개인 = -(외국인 + 기관) 추정`} />
          <FlowBox title="실시간 체결 — 한투 WebSocket (선택)" color="#3B82F6"
            code={`# 실시간 체결가·거래량 (투자자 구분 없음)
# → 거래량 급증 감지에만 활용
# WebSocket: H0STCNT0`} />
          <FlowBox title="장 마감 후 — 한투 REST API" color="#A78BFA"
            code={`# 투자자별 확정 데이터 (FHKST01010900)
# → 외국인/기관/개인 정확한 최종 수치`} />
        </div>
      </Card>

      <Card title="알림 조건" color="#F59E0B">
        <div style={{ display: "grid", gap: "4px" }}>
          {[
            { icon: "🔴", cond: "외국인 순매도 -50억 돌파", desc: "외국인 대량 매도 → 하락 경고" },
            { icon: "🟢", cond: "기관 순매수 +30억 돌파", desc: "기관 매집 신호" },
            { icon: "⚡", cond: "거래량 전일동시간 대비 300%↑", desc: "무언가 발생" },
            { icon: "🔄", cond: "외국인+기관 동시 순매수", desc: "수급 동반 유입 강한 신호" },
            { icon: "📉", cond: "개인↑ + 외국인↓ 동시", desc: "개미 물타기 vs 외국인 탈출" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: "8px", padding: "6px 10px", background: "#0F172A", borderRadius: "6px", alignItems: "center" }}>
              <span>{item.icon}</span>
              <div><strong style={{ fontSize: "12px", color: "#FCD34D" }}>{item.cond}</strong><p style={{ fontSize: "11px", color: "#94A3B8", margin: 0 }}>{item.desc}</p></div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="콘솔 출력 예시" color="#A78BFA">
        <pre style={{ background: "#0F172A", padding: "12px", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px", lineHeight: "1.6", color: "#CBD5E1", overflow: "auto" }}>
{`🚨 [삼성전자 005930] 수급 알림 — 10:35
현재가: 72,400원 (+1.2%)
━━━━━━━━━━━━━━━
`}<span style={{ color: "#EF4444" }}>외국인: -127억 (순매도)</span>{`
`}<span style={{ color: "#10B981" }}>기 관: +89억 (순매수)</span>{`
`}<span style={{ color: "#F59E0B" }}>개 인: +38억 (추정)</span>{`
━━━━━━━━━━━━━━━
`}<span style={{ color: "#EF4444" }}>⚠️ 외국인 -100억 돌파! 매도세 주의</span>
        </pre>
        <p style={{ fontSize: "10px", color: "#64748B", marginTop: "4px" }}>→ Phase 5에서 이 메시지를 그대로 텔레그램으로 발송</p>
      </Card>

      <Card title="실행" color="#6366F1">
        <Code text={`python scripts/run_monitor.py \\
  --stocks "005930,000660,035420" \\
  --interval 300 \\
  --alert-threshold 50`} />
      </Card>
    </div>
  );
}

function Feature2Tab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="기능 2: AI 종합분석 + 100점 스코어링" color="#EF4444">
        <p style={{ fontSize: "12px" }}>기술적 분석, 수급, 매매기법을 종합 → 내일 예측을 100점 만점으로 점수화</p>
      </Card>

      <Card title="점수 배분 (100점 만점)" color="#10B981">
        <div style={{ display: "grid", gap: "6px" }}>
          <ScoreRow cat="📊 기술적 분석" score="35점" items={["이동평균 배열 (5/20/60/120일) — 10점","RSI·MACD·스토캐스틱 — 10점","볼린저밴드 — 5점","캔들 패턴 — 5점","지지/저항선 — 5점"]} color="#3B82F6" />
          <ScoreRow cat="💰 수급 분석" score="30점" items={["외국인 5일 추세 — 12점","기관 5일 추세 — 10점","개인 동향(역지표) — 5점","프로그램 매매 — 3점"]} color="#10B981" />
          <ScoreRow cat="📈 매매기법" score="20점" items={["거래량 분석 — 8점","매물대 분석 — 7점","눌림목/돌파 패턴 — 5점"]} color="#F59E0B" />
          <ScoreRow cat="📰 이슈/모멘텀" score="15점" items={["뉴스 센티먼트 — 8점","섹터/테마 — 4점","공매도·대차잔고 — 3점"]} color="#A78BFA" />
        </div>
      </Card>

      <Card title="점수 해석" color="#F59E0B">
        <div style={{ display: "grid", gap: "4px" }}>
          {[
            { range: "80~100", label: "🟢 강력 매수", color: "#10B981" },
            { range: "65~79", label: "🔵 매수 우위", color: "#3B82F6" },
            { range: "50~64", label: "🟡 중립/관망", color: "#F59E0B" },
            { range: "35~49", label: "🟠 매도 우위", color: "#F97316" },
            { range: "0~34", label: "🔴 강력 매도", color: "#EF4444" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: "10px", padding: "6px 10px", background: "#0F172A", borderRadius: "6px", alignItems: "center" }}>
              <span style={{ color: item.color, fontWeight: "700", fontSize: "12px", minWidth: "55px" }}>{item.range}점</span>
              <strong style={{ color: item.color, fontSize: "12px" }}>{item.label}</strong>
            </div>
          ))}
        </div>
      </Card>

      <Card title="콘솔 리포트 예시" color="#A78BFA">
        <pre style={{ background: "#0F172A", padding: "12px", borderRadius: "8px", fontFamily: "monospace", fontSize: "11px", lineHeight: "1.7", color: "#CBD5E1", overflow: "auto" }}>
{`🔮 [삼성전자] AI 종합분석 리포트
2026.02.11 장 마감 기준
현재가: 72,400원 | +1.2%
━━━━━━━━━━━━━━━━━━━
`}<span style={{color:"#3B82F6"}}>📊 기술적 분석: 26/35점</span>{`
 · MA 정배열 유지 (5>20>60) ✅
 · RSI 58 (중립~매수) ✅
 · MACD 골든크로스 임박 ⚡
`}<span style={{color:"#10B981"}}>💰 수급 분석: 22/30점</span>{`
 · 외국인 3일 연속 순매수 (+420억) ✅
 · 기관 소폭 순매도 (-15억) ⚠️
`}<span style={{color:"#F59E0B"}}>📈 매매기법: 14/20점</span>{`
 · 거래량 20일 평균 180%↑ ✅
 · 71,500~72,000 매물대 돌파 ✅
`}<span style={{color:"#A78BFA"}}>📰 이슈/모멘텀: 10/15점</span>{`
 · HBM3E 양산 뉴스 (긍정) ✅
━━━━━━━━━━━━━━━━━━━
`}<span style={{color:"#FCD34D",fontWeight:"bold",fontSize:"13px"}}>🏆 종합 점수: 72/100 (매수 우위)</span>{`
━━━━━━━━━━━━━━━━━━━
`}<span style={{color:"#10B981"}}>🔮 내일 예측: 상승 (확률 68%)</span>{`
 예상 범위: 72,000 ~ 73,500원
`}<span style={{color:"#10B981"}}>✅ 유지조건: MA20(71,200원) 위 유지</span>{`
`}<span style={{color:"#EF4444"}}>🚨 대응조건: 71,000원 이탈 시 비중 축소</span>
        </pre>
      </Card>

      <Card title="데이터 소스 → 분석 항목 매핑" color="#6366F1">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
          <thead><tr style={{ borderBottom: "1px solid #334155" }}>
            <th style={{ textAlign: "left", padding: "6px", color: "#94A3B8" }}>분석 항목</th>
            <th style={{ textAlign: "left", padding: "6px", color: "#94A3B8" }}>데이터 소스</th>
          </tr></thead>
          <tbody>
            {[
              ["기술적 분석 (35점)", "한투 API — 일봉/분봉 OHLCV"],
              ["수급 분석 (30점)", "한투 API(과거) + 네이버(장중) — 투자자별"],
              ["매매기법 (20점)", "한투 API — 거래량·분봉 데이터"],
              ["이슈/모멘텀 (15점)", "Claude AI — 뉴스 분석"],
            ].map((row, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #1E293B" }}>
                <td style={{ padding: "6px", color: "#E2E8F0" }}>{row[0]}</td>
                <td style={{ padding: "6px", color: "#94A3B8" }}>{row[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function PromptsTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="⭐ Claude Code 프롬프트 v2" color="#F43F5E">
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "6px" }}>
          <Tag text="pykrx ❌" color="#EF4444" /><Tag text="한투 API ✅" color="#10B981" /><Tag text="네이버 금융 ✅" color="#3B82F6" /><Tag text="알림 🔜 후순위" color="#64748B" />
        </div>
        <p style={{ fontSize: "12px", color: "#FCD34D" }}>순서대로 하나씩 Claude Code에 붙여넣기!</p>
      </Card>

      <PromptCard num="0" title="CLAUDE.md 작성" prompt={`CLAUDE.md 파일을 만들어줘:

# 프로젝트: 한국 주식 실시간 수급 모니터링 + AI 분석 시스템

## 아키텍처
- 클린 아키텍처 4계층 (entities → use_cases → adapters → agents)
- 안쪽 계층은 바깥 계층을 절대 import 하지 않는다
- 의존성 역전: use_cases는 Port(인터페이스)만 의존

## 기술 스택
- Python 3.13, asyncio
- 한국투자증권 오픈API (REST + WebSocket): 메인 데이터
- 네이버 금융 크롤링 (requests + pd.read_html): 장중 투자자별
- Anthropic Claude API: AI 분석
- ※ pykrx 사용 안 함 (KRX 로그인 필수화로 불안정)
- ※ 알림은 나중에 추가. 현재는 콘솔 출력

## 기능
1. 실시간 수급 모니터링 (장중 5분 간격)
2. AI 종합분석 + 100점 스코어링

## 코딩 규칙
- 한국어 주석, Type hints 필수
- 외부 호출은 try-except 감싸기
- 환경변수 .env에서 로드
- 한투 API 초당 20회, 네이버 5초+ 간격 준수`} />

      <PromptCard num="1" title="Step 1: 엔티티 (핵심 객체)" prompt={`클린 아키텍처의 entities 계층을 만들어줘.
src/entities/ 폴더에:

1. stock.py — Stock(code, name, market), PriceData(date, open, high, low, close, volume), ChartData(stock, prices) + 기술지표 계산 메서드

2. investor_flow.py — InvestorFlow(date, time, foreign_net, institution_net, individual_net, total_volume), FlowAlert(stock, flow, alert_type, message, severity), AlertType enum

3. technical.py — TechnicalIndicators(rsi, macd, macd_signal, bollinger_upper/mid/lower, stochastic_k/d), TechnicalPattern(name, direction, strength)

4. analysis_report.py — ScoreCategory(name, score, max_score, details), AnalysisScore(technical, supply_demand, trading, momentum) + total_score, AnalysisReport(stock, score, prediction, confidence, price_range, hold_conditions, action_conditions)

모든 엔티티는 순수 Python dataclass. 외부 라이브러리 의존 없이.`} />

      <PromptCard num="2" title="Step 2: 포트 (인터페이스)" prompt={`src/use_cases/ports.py를 만들어줘. ABC로:

1. StockDataPort — get_stock_info(code), get_chart_data(code, days=120), get_current_price(code), get_minute_data(code, minutes=5)

2. InvestorFlowPort — get_today_flow(code), get_flow_history(code, days=5)

3. TechnicalAnalysisPort — analyze(chart_data) -> (TechnicalIndicators, list[TechnicalPattern])

4. OutputPort — send_alert(message) -> bool, send_report(report) -> bool
   ※ 나중에 텔레그램 교체 가능하도록 추상화

5. AIAnalysisPort — analyze_chart(), analyze_supply_demand(), analyze_trading_pattern(), analyze_momentum(), predict_tomorrow()`} />

      <PromptCard num="3" title="Step 3: 유스케이스 (비즈니스 로직)" prompt={`src/use_cases/에:

1. realtime_monitor.py — RealtimeMonitorInteractor
- __init__에서 Port 주입
- monitor_stocks(codes, interval=300): asyncio 루프
  → InvestorFlowPort.get_today_flow() 5분마다 호출
  → 이전 데이터 비교, 변화 감지
  → OutputPort.send_alert()
- 알림 조건: 외국인 ±50억, 기관 ±30억, 거래량 300%↑, 동시순매수

2. stock_analysis.py — StockAnalysisInteractor
- analyze_stock(code) -> AnalysisReport
  → StockDataPort로 차트, InvestorFlowPort로 수급
  → TechnicalAnalysisPort, AIAnalysisPort로 분석
  → OutputPort.send_report()`} />

      <PromptCard num="4" title="Step 4: 한투 API 어댑터 ⭐핵심" prompt={`src/adapters/kis_api.py를 만들어줘.

한국투자증권 오픈API 어댑터. StockDataPort + InvestorFlowPort(과거) 구현.

환경변수: KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_HTS_ID

구현:
1. OAuth 토큰 발급 (/oauth2/tokenP) + 캐싱
2. get_current_price — FHKST01010100 (현재가시세)
3. get_chart_data — FHKST03010100 (기간별시세 일봉)
4. get_minute_data — FHKST03010200 (당일분봉)
5. get_flow_history — FHKST01010900 (투자자별, 장종료후)

기본URL: https://openapi.koreainvestment.com:9443
requests 사용, Rate limit 초당20회, 전부 try-except.`} />

      <PromptCard num="5" title="Step 5: 네이버 크롤링 + 기술분석 + 콘솔" prompt={`나머지 어댑터:

1. src/adapters/naver_finance.py (InvestorFlowPort.get_today_flow)
- https://finance.naver.com/item/frgn.naver?code={code}&page=1
- requests + pd.read_html() → 외국인/기관 순매매 파싱
- 개인 = -(외국인+기관), User-Agent 헤더, 5초+ 간격

2. src/adapters/technical_adapter.py (TechnicalAnalysisPort)
- 순수 Python: RSI(14), MACD(12,26,9), 볼린저(20,2), 스토캐스틱(14,3)
- 캔들 패턴 (망치형, 장악형, 도지 등)

3. src/adapters/console_output.py (OutputPort)
- send_alert(): 이모지+구분선 콘솔 출력
- send_report(): 분석 리포트 포맷팅 출력
- ※ 나중에 TelegramAdapter로 교체`} />

      <PromptCard num="6" title="Step 6: Claude AI + 실행 스크립트" prompt={`마지막:

1. src/adapters/claude_api.py (AIAnalysisPort)
- Anthropic Claude API 호출
- 기술35점, 수급30점, 매매20점, 모멘텀15점 각 프롬프트
- JSON 응답 → ScoreCategory 변환

2. scripts/run_monitor.py
- argparse: --stocks "005930,000660" --interval 300
- 어댑터 → RealtimeMonitorInteractor 주입 → asyncio.run()

3. scripts/run_analysis.py
- argparse: --stock "005930"
- 어댑터 → StockAnalysisInteractor → 콘솔 리포트

4. .env.example + requirements.txt
- KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_HTS_ID, ANTHROPIC_API_KEY
- requests, anthropic, python-dotenv, pandas, numpy`} />
    </div>
  );
}

function PrereqTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="시작 전 준비물 2가지만!" color="#F59E0B">
        <p style={{ fontSize: "12px" }}>알림은 후순위. 지금은 2가지만 준비하면 됩니다.</p>
      </Card>

      <Card title="1️⃣ 한국투자증권 오픈API (필수)" color="#10B981">
        <div style={{ display: "grid", gap: "4px" }}>
          <Step n="1" text="한국투자증권 계좌 개설 (이미 있으면 PASS)" />
          <Step n="2" text="apiportal.koreainvestment.com 접속" />
          <Step n="3" text="KIS Developers 서비스 신청 (종합+모의투자)" />
          <Step n="4" text="App Key + App Secret 발급 (복사)" />
          <Step n="5" text="HTS ID 확인 (로그인 ID)" />
        </div>
        <div style={{ marginTop: "8px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
          <Tag text="무료" color="#10B981" /><Tag text="공식 API" color="#10B981" /><Tag text="REST+WebSocket" color="#3B82F6" />
        </div>
      </Card>

      <Card title="2️⃣ Anthropic API 키 (Claude AI)" color="#A78BFA">
        <div style={{ display: "grid", gap: "4px" }}>
          <Step n="1" text="console.anthropic.com 접속" />
          <Step n="2" text="회원가입 / 로그인" />
          <Step n="3" text="API Keys → Create Key" />
          <Step n="4" text="키 복사 → .env에 저장" />
        </div>
        <div style={{ marginTop: "8px", background: "#A78BFA20", padding: "8px", borderRadius: "6px" }}>
          <span style={{ fontSize: "11px", color: "#C4B5FD" }}>💰 Sonnet 기준 분석 1회 ~$0.01~0.05 (10~50원)</span>
        </div>
      </Card>

      <Card title="🔜 나중에 추가 (Phase 5)" color="#64748B">
        <div style={{ display: "grid", gap: "4px" }}>
          {["텔레그램 봇 → @BotFather + Bot Token + Chat ID","슬랙 → Incoming Webhook URL","카카오톡 → 비즈니스 인증 (복잡)"].map((item, i) => (
            <div key={i} style={{ padding: "4px 8px", background: "#0F172A", borderRadius: "4px", fontSize: "11px", color: "#64748B" }}>🔜 {item}</div>
          ))}
        </div>
      </Card>

      <Card title="라이브러리 설치" color="#3B82F6">
        <Code text="pip install requests anthropic python-dotenv pandas numpy" />
        <p style={{ fontSize: "11px", color: "#64748B", marginTop: "4px" }}>pykrx, python-telegram-bot 등은 지금 불필요</p>
      </Card>

      <Card title="전체 체크리스트" color="#6366F1">
        <div style={{ display: "grid", gap: "3px" }}>
          {[
            { text: "한투 API: App Key + Secret 발급", g: "준비" },
            { text: "한투 API: HTS ID 확인", g: "준비" },
            { text: "Anthropic API 키 발급", g: "준비" },
            { text: ".env 파일에 키 저장", g: "준비" },
            { text: "pip install 라이브러리 설치", g: "준비" },
            null,
            { text: "Step 0: CLAUDE.md 작성", g: "1" },
            { text: "Step 1: entities 생성", g: "1" },
            { text: "Step 2: ports.py 생성", g: "1" },
            { text: "Step 3: 유스케이스 생성", g: "1" },
            { text: "Step 4: 한투 API 어댑터", g: "2" },
            { text: "Step 5: 네이버+기술분석+콘솔", g: "2" },
            { text: "Step 6: Claude AI + 실행 스크립트", g: "3" },
            null,
            { text: "테스트: 한투 API 연결 확인", g: "T" },
            { text: "테스트: 네이버 크롤링 확인", g: "T" },
            { text: "테스트: 분석 리포트 콘솔 출력", g: "T" },
            { text: "실전: 장중 모니터링 실행", g: "!" },
            null,
            { text: "Phase 5: 텔레그램 연결 (나중에)", g: "5" },
          ].map((item, i) => item === null ? (
            <div key={i} style={{ height: "1px", background: "#334155", margin: "4px 0" }} />
          ) : (
            <div key={i} style={{ display: "flex", gap: "8px", padding: "4px 8px", background: "#0F172A", borderRadius: "4px", alignItems: "center" }}>
              <span style={{ fontSize: "10px", color: item.g === "5" ? "#475569" : "#94A3B8" }}>□</span>
              <span style={{ fontSize: "12px", color: item.g === "5" ? "#475569" : "#CBD5E1" }}>{item.text}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ════════ shared components ════════ */
function Card({ title, color, children }) {
  return (<div style={{ background: "#1E293B", borderRadius: "12px", padding: "14px", borderTop: `3px solid ${color}` }}>
    <h3 style={{ color, fontSize: "15px", marginBottom: "8px" }}>{title}</h3>
    <div style={{ color: "#CBD5E1", fontSize: "13px", lineHeight: "1.6" }}>{children}</div>
  </div>);
}
function Code({ text }) {
  return <pre style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", fontSize: "11px", color: "#10B981", overflow: "auto", lineHeight: "1.5", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{text}</pre>;
}
function MiniCard({ icon, title, desc, color }) {
  return (<div style={{ background: "#0F172A", padding: "12px", borderRadius: "8px", borderLeft: `3px solid ${color}` }}>
    <div style={{ fontSize: "16px", marginBottom: "4px" }}>{icon}</div>
    <strong style={{ color, fontSize: "13px" }}>{title}</strong>
    <p style={{ fontSize: "11px", color: "#94A3B8", marginTop: "4px" }}>{desc}</p>
  </div>);
}
function SourceBox({ name, tag, tagColor, items }) {
  return (<div style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", borderLeft: `3px solid ${tagColor}` }}>
    <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "4px" }}>
      <strong style={{ color: tagColor, fontSize: "13px" }}>{name}</strong>
      <Tag text={tag} color={tagColor} />
    </div>
    {items.map((item, i) => <div key={i} style={{ fontSize: "11px", color: "#94A3B8", paddingLeft: "8px" }}>· {item}</div>)}
  </div>);
}
function ScoreRow({ cat, score, items, color }) {
  return (<div style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", borderLeft: `3px solid ${color}` }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
      <strong style={{ color, fontSize: "13px" }}>{cat}</strong>
      <span style={{ color, fontSize: "13px", fontWeight: "700" }}>{score}</span>
    </div>
    {items.map((item, i) => <div key={i} style={{ fontSize: "11px", color: "#94A3B8", paddingLeft: "8px" }}>· {item}</div>)}
  </div>);
}
function PhaseBox({ phase, label, desc, color, active }) {
  return (<div style={{ display: "flex", gap: "10px", padding: "8px 10px", background: active ? `${color}15` : "#0F172A", borderRadius: "6px", alignItems: "center", border: active ? `1px solid ${color}40` : "1px solid transparent" }}>
    <span style={{ color, fontWeight: "700", fontSize: "12px", minWidth: "50px" }}>{phase}</span>
    <span style={{ color: "#94A3B8", fontSize: "11px", minWidth: "65px" }}>{label}</span>
    <span style={{ fontSize: "12px", color: active ? "#E2E8F0" : "#64748B" }}>{desc}</span>
  </div>);
}
function FlowBox({ title, color, code }) {
  return (<div style={{ background: `${color}20`, padding: "10px", borderRadius: "8px" }}>
    <strong style={{ color, fontSize: "13px" }}>{title}</strong>
    <Code text={code} />
  </div>);
}
function PromptCard({ num, title, prompt }) {
  return (<div style={{ background: "#1E293B", borderRadius: "12px", padding: "14px", borderLeft: "4px solid #FCD34D" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
      <span style={{ background: "#FCD34D", color: "#0F172A", borderRadius: "50%", width: "24px", height: "24px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12px", fontWeight: "800", flexShrink: 0 }}>{num}</span>
      <strong style={{ color: "#FCD34D", fontSize: "13px" }}>{title}</strong>
    </div>
    <pre style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", fontSize: "11px", color: "#FCD34D", overflow: "auto", lineHeight: "1.5", whiteSpace: "pre-wrap", wordBreak: "break-word", borderLeft: "3px solid #FCD34D40" }}>{prompt}</pre>
    <p style={{ fontSize: "10px", color: "#64748B", marginTop: "6px", textAlign: "right" }}>↑ Claude Code에 복사 붙여넣기</p>
  </div>);
}
function Tag({ text, color }) {
  return <span style={{ padding: "2px 8px", borderRadius: "4px", background: `${color}20`, color, fontSize: "10px", fontWeight: "600" }}>{text}</span>;
}
function Step({ n, text }) {
  return (<div style={{ display: "flex", gap: "8px", padding: "4px 8px", background: "#0F172A", borderRadius: "4px", alignItems: "center" }}>
    <span style={{ background: "#10B981", borderRadius: "50%", width: "20px", height: "20px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "11px", fontWeight: "700", flexShrink: 0, color: "#0F172A" }}>{n}</span>
    <span style={{ fontSize: "12px" }}>{text}</span>
  </div>);
}
