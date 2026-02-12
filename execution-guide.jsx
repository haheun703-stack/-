import { useState } from "react";

const phases = [
  { id: "roadmap", label: "🗺️ 실행 로드맵" },
  { id: "phase1", label: "🟢 1단계: 데이터 연결" },
  { id: "phase2", label: "🔵 2단계: 모니터링" },
  { id: "phase3", label: "🟣 3단계: AI 분석" },
];

export default function App() {
  const [tab, setTab] = useState("roadmap");
  return (
    <div style={{ fontFamily: "'Pretendard', -apple-system, sans-serif", background: "#0F172A", color: "#E2E8F0", minHeight: "100vh", padding: "16px", maxWidth: "820px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center", fontSize: "20px", marginBottom: "4px", background: "linear-gradient(135deg, #10B981, #3B82F6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
        실전 실행 가이드 — 3단계로 끝내기
      </h1>
      <p style={{ textAlign: "center", color: "#64748B", fontSize: "12px", marginBottom: "16px" }}>한투 API 키 준비 완료 ✅ → 바로 시작</p>
      <div style={{ display: "flex", gap: "4px", marginBottom: "16px", justifyContent: "center", flexWrap: "wrap" }}>
        {phases.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "7px 12px", borderRadius: "8px", border: "none", fontSize: "12px", cursor: "pointer",
            background: tab === t.id ? "#10B981" : "#1E293B", color: tab === t.id ? "white" : "#94A3B8",
            fontWeight: tab === t.id ? "600" : "400",
          }}>{t.label}</button>
        ))}
      </div>
      {tab === "roadmap" && <RoadmapTab />}
      {tab === "phase1" && <Phase1Tab />}
      {tab === "phase2" && <Phase2Tab />}
      {tab === "phase3" && <Phase3Tab />}
    </div>
  );
}

function RoadmapTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="왜 막막했는가" color="#F59E0B">
        <div style={{ background: "#F59E0B15", padding: "10px", borderRadius: "8px", fontSize: "12px" }}>
          이전 가이드: 프롬프트 7개 (Step 0~6) → <strong style={{ color: "#EF4444" }}>"이걸 다 해야 하나?"</strong><br/><br/>
          실제로는 <strong style={{ color: "#10B981" }}>작은 성공을 먼저 보는 게 핵심</strong>입니다.<br/>
          "삼성전자 현재가가 내 터미널에 찍힌다" → 이 한 줄이 나오면 나머지는 쭉 따라갑니다.
        </div>
      </Card>

      <Card title="✅ 새로운 접근: 3단계로 끝내기" color="#10B981">
        <div style={{ display: "grid", gap: "10px" }}>
          <PhaseCard
            num="1" color="#10B981" time="오늘"
            title="데이터가 찍힌다"
            goal="삼성전자 현재가 + 외국인 매매가 콘솔에 출력"
            detail="한투 API 연결 → 네이버 크롤링 → 데이터 확인"
            result="python test.py 하면 삼성전자 72,400원, 외국인 -127억 출력"
          />
          <div style={{ textAlign: "center", color: "#334155", fontSize: "18px" }}>↓</div>
          <PhaseCard
            num="2" color="#3B82F6" time="+1~2일"
            title="알림이 울린다"
            goal="5분마다 수급 체크 → 큰 변화 시 콘솔 알림"
            detail="모니터링 루프 + 알림 조건 로직"
            result="외국인 -50억 돌파하면 🚨 알림 출력"
          />
          <div style={{ textAlign: "center", color: "#334155", fontSize: "18px" }}>↓</div>
          <PhaseCard
            num="3" color="#A78BFA" time="+3~4일"
            title="AI가 점수를 매긴다"
            goal="기술분석 + 수급 → Claude AI 100점 스코어"
            detail="기술지표 계산 + Claude 프롬프트 + 리포트 출력"
            result="삼성전자 종합 72/100 매수우위 리포트"
          />
        </div>
      </Card>

      <Card title="이전 7개 프롬프트 vs 새로운 3단계" color="#6366F1">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
          <thead><tr style={{ borderBottom: "1px solid #334155" }}>
            <th style={{ textAlign: "left", padding: "6px", color: "#EF4444" }}>이전 (추상적)</th>
            <th style={{ textAlign: "left", padding: "6px", color: "#10B981" }}>지금 (구체적)</th>
          </tr></thead>
          <tbody>
            {[
              ["Step 0: CLAUDE.md 만들기", "1단계에 포함"],
              ["Step 1: 엔티티 만들기", "1단계에 포함"],
              ["Step 2: 포트 만들기", "1단계에 포함"],
              ["Step 3: 유스케이스 만들기", "2단계에 포함"],
              ["Step 4: 한투 API 어댑터", "1단계에 포함"],
              ["Step 5: 네이버+기술분석+콘솔", "1단계(네이버) + 3단계(기술분석)"],
              ["Step 6: Claude AI + 실행", "3단계에 포함"],
            ].map((row, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #1E293B" }}>
                <td style={{ padding: "5px 6px", color: "#94A3B8" }}>{row[0]}</td>
                <td style={{ padding: "5px 6px", color: "#10B981" }}>{row[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ fontSize: "11px", color: "#FCD34D", marginTop: "8px" }}>차이점: 각 단계마다 "돌려보면 뭔가 보인다" → 동기부여가 유지됨</p>
      </Card>

      <Card title="💡 핵심 원칙" color="#F59E0B">
        <div style={{ display: "grid", gap: "6px" }}>
          {[
            { icon: "🎯", text: "각 단계 끝나면 python으로 실행해서 결과를 눈으로 확인", color: "#10B981" },
            { icon: "🧱", text: "한 번에 전체를 만들지 않고, 돌아가는 작은 것을 먼저 만든 뒤 확장", color: "#3B82F6" },
            { icon: "🔑", text: "1단계가 제일 중요 — API 연결이 되면 나머지는 로직만 추가하는 것", color: "#F59E0B" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: "8px", padding: "8px 10px", background: "#0F172A", borderRadius: "6px" }}>
              <span style={{ fontSize: "16px" }}>{item.icon}</span>
              <span style={{ fontSize: "12px", color: item.color }}>{item.text}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="시작하기 전 — .env 파일 확인" color="#EF4444">
        <p style={{ fontSize: "12px", marginBottom: "8px" }}>Claude Code 프롬프트를 넣기 전에, 프로젝트 폴더에 .env 파일을 먼저 만들어두세요:</p>
        <Code text={`# 프로젝트 폴더에서
mkdir my-stock-system
cd my-stock-system

# .env 파일 생성 (본인 키 입력)
cat > .env << 'EOF'
KIS_APP_KEY=여기에_앱키_36자리
KIS_APP_SECRET=여기에_시크릿키_180자리
KIS_ACCOUNT_NO=12345678-01
KIS_HTS_ID=여기에_HTS로그인ID
ANTHROPIC_API_KEY=sk-ant-여기에_클로드API키
EOF`} />
        <Warn text="이 파일이 없으면 1단계부터 실행이 안 됩니다. 반드시 먼저!" />
      </Card>
    </div>
  );
}

function Phase1Tab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="🟢 1단계: 데이터가 찍힌다" color="#10B981">
        <div style={{ background: "#10B98120", padding: "10px", borderRadius: "8px" }}>
          <strong style={{ color: "#10B981", fontSize: "14px" }}>목표: python test_data.py 실행하면 삼성전자 데이터가 나온다</strong>
          <p style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}>예상 소요: 30분~1시간 (Claude Code가 코드 작성)</p>
        </div>
      </Card>

      <PromptCard num="1-A" title="프롬프트 A: 프로젝트 세팅 + 한투 API 연결" prompt={`새 프로젝트를 세팅하고, 한국투자증권 오픈API로 삼성전자 데이터를 가져오는 것까지 만들어줘.

## 프로젝트 구조
my-stock-system/ 폴더에 클린 아키텍처로:
- src/entities/ : 비즈니스 객체
- src/use_cases/ : 포트 인터페이스
- src/adapters/ : 외부 연결

## 1단계에서 만들 파일들

### src/entities/stock.py
- Stock: code(str), name(str), market(str "KOSPI"/"KOSDAQ")
- PriceData: date(str), open(int), high(int), low(int), close(int), volume(int), change_rate(float)
- InvestorFlow: date(str), foreign_net(int), institution_net(int), individual_net(int)
모두 Python dataclass로.

### src/use_cases/ports.py
ABC 추상 클래스로:
- StockDataPort: get_current_price(code) -> PriceData, get_daily_prices(code, days) -> list[PriceData]
- InvestorFlowPort: get_today_flow(code) -> InvestorFlow, get_flow_history(code, days) -> list[InvestorFlow]

### src/adapters/kis_api.py ← 핵심!
한국투자증권 오픈API REST 호출. StockDataPort 구현.

환경변수 (.env):
- KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_HTS_ID

구현할 API:
1. 접근토큰 발급: POST /oauth2/tokenP
   - grant_type: "client_credentials"
   - 토큰은 24시간 유효, 발급 후 캐싱
   
2. 주식현재가 시세: GET /uapi/domestic-stock/v1/quotations/inquire-price
   - tr_id: "FHKST01010100"
   - params: FID_COND_MRKT_DIV_CODE="J", FID_INPUT_ISCD=종목코드
   
3. 기간별시세 일봉: GET /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
   - tr_id: "FHKST03010100"

4. 투자자별 매매: GET /uapi/domestic-stock/v1/quotations/inquire-investor
   - tr_id: "FHKST01010900"
   - ⚠️ 당일 데이터는 장 종료 후에만 제공

기본 URL: https://openapi.koreainvestment.com:9443
헤더: authorization, appkey, appsecret, tr_id, Content-Type

### src/adapters/naver_finance.py
네이버 금융 크롤링. InvestorFlowPort.get_today_flow() 구현.
- URL: https://finance.naver.com/item/frgn.naver?code={종목코드}&page=1
- requests.get() + pandas.read_html()로 파싱
- User-Agent 헤더 필수

### test_data.py ← 이걸 실행해서 확인!
삼성전자(005930) 데이터를 가져와서 출력하는 테스트 스크립트:
1. 한투 API로 현재가 조회 → 출력
2. 한투 API로 최근 5일 일봉 → 출력  
3. 한투 API로 투자자별 매매 → 출력
4. 네이버 금융으로 오늘 외국인/기관 → 출력

출력 예시:
━━━ 삼성전자 (005930) ━━━
현재가: 72,400원 (+1.2%)
━━━ 최근 5일 ━━━
02/11: 72,400 (거래량 12,345,678)
02/10: 71,500 ...
━━━ 투자자별 (한투API) ━━━
외국인: -127억 | 기관: +89억 | 개인: +38억
━━━ 투자자별 (네이버) ━━━
외국인: -130억 | 기관: +85억

### requirements.txt
requests, python-dotenv, pandas, numpy

### CLAUDE.md
프로젝트 규칙 파일도 같이 만들어줘:
- 한국어 주석, type hints 필수
- 외부 호출 try-except
- 한투 API 초당 20회, 네이버 5초+ 간격`} />

      <Card title="1-A 실행 후 확인사항" color="#F59E0B">
        <div style={{ display: "grid", gap: "4px" }}>
          <CheckItem text="python test_data.py 실행했을 때 에러 없이 데이터 출력되는가?" />
          <CheckItem text="현재가가 실제 주가와 비슷한가? (장중이면 실시간, 장후면 종가)" />
          <CheckItem text="투자자별 데이터에 외국인/기관 숫자가 나오는가?" />
        </div>
        <Warn text="여기서 에러가 나면? → Claude Code에 에러 메시지를 그대로 붙여넣으세요. 고쳐줍니다." />
      </Card>

      <PromptCard num="1-B" title="프롬프트 B: 여러 종목 + 보기좋은 출력 (선택)" prompt={`test_data.py가 잘 되면, 여러 종목을 한번에 조회하고 보기 좋게 출력하도록 확장해줘.

### scripts/scan_stocks.py
종목 리스트를 받아서 한번에 스캔:
- 기본 종목: 삼성전자(005930), SK하이닉스(000660), NAVER(035420), 카카오(035720), 삼성SDI(006400)
- argparse로 --stocks "005930,000660" 커스텀 가능

각 종목마다:
1. 현재가 + 등락률
2. 오늘 외국인/기관 순매매 (네이버 크롤링)
3. 거래량 (전일 대비 %)

출력 형식:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 수급 스캔 (2026.02.11 14:30)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
삼성전자  72,400 (+1.2%)  외국인:-127억  기관:+89억  거래량:180%
SK하이닉스 198,500 (+2.1%)  외국인:+340억  기관:+120억  거래량:220%⚡
NAVER    215,000 (-0.5%)  외국인:-45억   기관:-12억  거래량:95%
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ = 거래량 150% 이상  🔥 = 외국인+기관 동시 순매수

종목 간 API 호출 사이에 0.1초 sleep (rate limit 준수)`} />

      <Card title="🎉 1단계 완료 기준" color="#10B981">
        <div style={{ background: "#10B98120", padding: "12px", borderRadius: "8px" }}>
          <strong style={{ color: "#10B981", fontSize: "14px" }}>python scripts/scan_stocks.py 하면</strong>
          <p style={{ fontSize: "12px", color: "#CBD5E1", marginTop: "4px" }}>5개 종목의 현재가, 외국인/기관 매매, 거래량이 한눈에 보인다.</p>
          <p style={{ fontSize: "12px", color: "#FCD34D", marginTop: "8px" }}>→ 여기까지 되면 2단계는 "이걸 5분마다 자동 반복"하는 것뿐!</p>
        </div>
      </Card>
    </div>
  );
}

function Phase2Tab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="🔵 2단계: 알림이 울린다" color="#3B82F6">
        <div style={{ background: "#3B82F620", padding: "10px", borderRadius: "8px" }}>
          <strong style={{ color: "#60A5FA", fontSize: "14px" }}>목표: 5분마다 자동 체크 → 큰 변화 시 🚨 알림</strong>
          <p style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}>1단계의 scan을 자동 반복 + 변화 감지 로직 추가</p>
        </div>
      </Card>

      <Card title="2단계 = 1단계 + 이것만 추가" color="#F59E0B">
        <div style={{ display: "grid", gap: "6px" }}>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#0F172A", borderRadius: "6px" }}>
            <span style={{ color: "#10B981", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>1단계에서</span>
            <span style={{ fontSize: "12px" }}>한번 실행하면 데이터가 나온다</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#3B82F620", borderRadius: "6px" }}>
            <span style={{ color: "#60A5FA", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ①</span>
            <span style={{ fontSize: "12px" }}>5분마다 자동 반복 (asyncio 루프)</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#3B82F620", borderRadius: "6px" }}>
            <span style={{ color: "#60A5FA", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ②</span>
            <span style={{ fontSize: "12px" }}>이전 데이터와 비교 → 변화량 계산</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#3B82F620", borderRadius: "6px" }}>
            <span style={{ color: "#60A5FA", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ③</span>
            <span style={{ fontSize: "12px" }}>조건 충족 시 🚨 알림 메시지 출력</span>
          </div>
        </div>
      </Card>

      <PromptCard num="2" title="프롬프트: 실시간 모니터링 시스템" prompt={`1단계에서 만든 코드를 확장해서, 실시간 수급 모니터링 시스템을 만들어줘.

## 추가할 파일

### src/use_cases/realtime_monitor.py
RealtimeMonitorInteractor 클래스:

__init__(self, flow_port: InvestorFlowPort, stock_port: StockDataPort):
  - 이전 데이터 저장용 딕셔너리: self._prev_flows = {}

async monitor(self, codes: list[str], interval: int = 300):
  - while True 루프
  - 각 종목에 대해:
    1. flow_port.get_today_flow(code) 호출
    2. self._prev_flows[code]와 비교
    3. 알림 조건 체크 → 해당하면 메시지 생성
    4. 현재 데이터를 _prev_flows에 저장
  - interval초 대기 후 반복
  - 종목 간 1초 sleep (rate limit)

알림 조건 (하나라도 충족하면 알림):
- 외국인 순매매 절대값 50억 이상
- 기관 순매매 절대값 30억 이상  
- 외국인+기관 동시 순매수 (둘 다 양수)
- 이전 체크 대비 외국인 변화량 30억 이상
- 거래량이 전일 동시간 대비 300% 이상 (stock_port에서 현재가 조회)

### src/adapters/console_output.py
콘솔 출력 어댑터:

def print_alert(self, code, name, price, flow, alert_reasons):
  이모지 + 구분선으로 보기 좋게:
  🚨 [삼성전자 005930] 수급 알림 — 14:35
  현재가: 72,400원 (+1.2%)
  ━━━━━━━━━━━━━━━
  외국인: -127억 (순매도)
  기 관: +89억 (순매수)  
  개 인: +38억 (추정)
  ━━━━━━━━━━━━━━━
  ⚠️ 외국인 -100억 돌파!
  ✅ 기관 +30억 이상 매수!

def print_scan(self, results):
  전체 종목 요약 테이블 (1단계 scan과 동일 형식)

### scripts/run_monitor.py
실행 스크립트:
- argparse:
  --stocks "005930,000660,035420" (기본값: 삼성전자,하이닉스,NAVER)
  --interval 300 (기본 5분)
  --threshold 50 (알림 기준 억원)
- 어댑터 생성 → RealtimeMonitorInteractor 주입
- asyncio.run(interactor.monitor(...))
- Ctrl+C로 깔끔하게 종료 (KeyboardInterrupt 처리)
- 시작할 때 "🟢 모니터링 시작 (5분 간격, 3종목)" 출력
- 매 사이클마다 "⏰ 14:35 체크 완료 (다음: 14:40)" 출력`} />

      <Card title="2단계 완료 기준" color="#3B82F6">
        <div style={{ background: "#3B82F620", padding: "12px", borderRadius: "8px" }}>
          <Code text={`python scripts/run_monitor.py --stocks "005930,000660" --interval 60`} />
          <p style={{ fontSize: "12px", color: "#CBD5E1", marginTop: "8px" }}>→ 1분마다 삼성전자, SK하이닉스의 외국인/기관 매매를 체크</p>
          <p style={{ fontSize: "12px", color: "#CBD5E1" }}>→ 외국인이 -50억 넘으면 🚨 알림 출력</p>
          <p style={{ fontSize: "12px", color: "#FCD34D", marginTop: "8px" }}>테스트 팁: --interval 60 --threshold 10 으로 낮춰서 알림이 잘 뜨는지 확인</p>
        </div>
      </Card>
    </div>
  );
}

function Phase3Tab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <Card title="🟣 3단계: AI가 점수를 매긴다" color="#A78BFA">
        <div style={{ background: "#A78BFA20", padding: "10px", borderRadius: "8px" }}>
          <strong style={{ color: "#C4B5FD", fontSize: "14px" }}>목표: python scripts/run_analysis.py --stock 005930 → 100점 리포트</strong>
          <p style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}>1단계 데이터 + 기술지표 계산 + Claude AI 분석</p>
        </div>
      </Card>

      <Card title="3단계 = 1단계 데이터 + 이것만 추가" color="#F59E0B">
        <div style={{ display: "grid", gap: "6px" }}>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#0F172A", borderRadius: "6px" }}>
            <span style={{ color: "#10B981", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>1단계에서</span>
            <span style={{ fontSize: "12px" }}>일봉 OHLCV + 투자자별 데이터 수집</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#A78BFA20", borderRadius: "6px" }}>
            <span style={{ color: "#C4B5FD", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ①</span>
            <span style={{ fontSize: "12px" }}>RSI, MACD, 볼린저밴드 등 기술지표 계산</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#A78BFA20", borderRadius: "6px" }}>
            <span style={{ color: "#C4B5FD", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ②</span>
            <span style={{ fontSize: "12px" }}>Claude API에 데이터 넘기고 점수 받기</span>
          </div>
          <div style={{ display: "flex", gap: "8px", padding: "8px", background: "#A78BFA20", borderRadius: "6px" }}>
            <span style={{ color: "#C4B5FD", fontWeight: "700", minWidth: "80px", fontSize: "12px" }}>+ 추가 ③</span>
            <span style={{ fontSize: "12px" }}>리포트 포맷팅해서 콘솔 출력</span>
          </div>
        </div>
      </Card>

      <PromptCard num="3-A" title="프롬프트 A: 기술지표 계산" prompt={`src/adapters/technical_adapter.py를 만들어줘.

일봉 데이터(list[PriceData])를 받아서 기술지표를 계산하는 순수 Python 모듈.
외부 라이브러리 없이 numpy만 사용.

### 계산할 지표들:

1. RSI(period=14)
   - 상승폭/하락폭 평균 → RS → RSI
   - 반환: 0~100 float

2. MACD(fast=12, slow=26, signal=9)
   - EMA(12) - EMA(26) = MACD line
   - EMA(9) of MACD = Signal line
   - MACD - Signal = Histogram
   - 반환: (macd, signal, histogram)

3. 볼린저밴드(period=20, std_dev=2)
   - 중심선 = SMA(20)
   - 상단 = 중심 + 2*표준편차
   - 하단 = 중심 - 2*표준편차
   - 반환: (upper, middle, lower)

4. 스토캐스틱(k_period=14, d_period=3)
   - %K = (현재가-최저)/(최고-최저) * 100
   - %D = %K의 3일 SMA
   - 반환: (k, d)

5. 이동평균선 배열 판단
   - MA5, MA20, MA60, MA120 계산
   - 정배열(5>20>60>120) / 역배열 / 혼조 판별

### 인터페이스:
def analyze(prices: list[PriceData]) -> dict:
  return {
    "rsi": 58.3,
    "macd": {"line": 450, "signal": 380, "histogram": 70},
    "bollinger": {"upper": 74000, "middle": 72000, "lower": 70000},
    "stochastic": {"k": 65.2, "d": 62.1},
    "ma": {"ma5": 72400, "ma20": 71200, "ma60": 69800, "ma120": 68500},
    "ma_alignment": "정배열",  # 또는 "역배열", "혼조"
    "summary": "RSI 중립, MACD 골든크로스 임박, 정배열 유지"
  }

### test_technical.py
삼성전자 120일 일봉 가져와서 기술지표 계산 후 출력:
━━━ 삼성전자 기술지표 ━━━
RSI(14): 58.3 (중립)
MACD: 450 / Signal: 380 (매수↑)
볼린저: 하단 70,000 | 중심 72,000 | 상단 74,000
스토캐스틱: %K 65.2 / %D 62.1
이평선: 5일>20일>60일>120일 (정배열 ✅)`} />

      <PromptCard num="3-B" title="프롬프트 B: Claude AI 분석 + 리포트" prompt={`Claude API를 사용해서 종합 분석 리포트를 만드는 시스템을 구현해줘.

### src/adapters/claude_api.py

Anthropic Claude API 호출 어댑터.
환경변수: ANTHROPIC_API_KEY

analyze_stock(code, prices, indicators, flows) -> dict:
  1. 수집된 데이터를 하나의 프롬프트로 구성
  2. Claude API 호출 (model: claude-sonnet-4-20250514)
  3. JSON 응답 파싱

Claude에게 보낼 프롬프트 (시스템 프롬프트):
"""
너는 한국 주식 전문 애널리스트야. 아래 데이터를 분석해서 100점 만점으로 점수를 매겨줘.

점수 기준:
- 기술적 분석 (35점): 이평선 배열, RSI, MACD, 볼린저밴드, 캔들패턴
- 수급 분석 (30점): 외국인/기관 5일 추세, 동시 매수/매도 여부
- 매매기법 (20점): 거래량 변화, 매물대 돌파 여부, 눌림목 패턴
- 모멘텀 (15점): 최근 추세 강도, 섹터 동향

반드시 아래 JSON 형식으로만 응답해:
{
  "technical": {"score": 26, "max": 35, "details": ["MA 정배열 유지 ✅", "RSI 58 중립"]},
  "supply_demand": {"score": 22, "max": 30, "details": ["외국인 3일 연속 순매수"]},
  "trading": {"score": 14, "max": 20, "details": ["거래량 180% 증가"]},
  "momentum": {"score": 10, "max": 15, "details": ["HBM 테마 강세"]},
  "total_score": 72,
  "grade": "매수 우위",
  "prediction": {"direction": "상승", "probability": 68, "target_high": 73500, "target_low": 72000},
  "hold_condition": "MA20(71,200원) 위 유지",
  "exit_condition": "71,000원 이탈 시 비중 축소"
}
"""

유저 메시지에는 실제 데이터를 넣어:
- 최근 20일 OHLCV
- 기술지표 (RSI, MACD, 볼린저, 스토캐스틱, MA)
- 최근 5일 투자자별 매매

### scripts/run_analysis.py
argparse: --stock "005930"

실행 흐름:
1. 한투 API로 120일 일봉 가져오기
2. 기술지표 계산 (technical_adapter)
3. 투자자별 5일 히스토리 (한투 API)
4. Claude AI 분석 요청
5. 콘솔에 리포트 출력

출력 형식:
🔮 [삼성전자] AI 종합분석 리포트
2026.02.11 장 마감 기준 | 현재가: 72,400원
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 기술적 분석: 26/35점
 · MA 정배열 유지 (5>20>60) ✅
 · RSI 58 (중립~매수) ✅
 · MACD 골든크로스 임박 ⚡
💰 수급 분석: 22/30점
 · 외국인 3일 연속 순매수 (+420억) ✅
 · 기관 소폭 순매도 (-15억) ⚠️
📈 매매기법: 14/20점
 · 거래량 20일 평균 180%↑ ✅
📰 모멘텀: 10/15점
 · HBM 테마 강세 ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 종합 점수: 72/100 (매수 우위)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 내일 예측: 상승 (확률 68%)
   예상 범위: 72,000 ~ 73,500원
✅ 유지조건: MA20(71,200원) 위 유지
🚨 대응조건: 71,000원 이탈 시 비중 축소`} />

      <Card title="🎉 3단계 완료 = 시스템 완성!" color="#A78BFA">
        <div style={{ background: "#A78BFA20", padding: "12px", borderRadius: "8px" }}>
          <div style={{ display: "grid", gap: "6px" }}>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span>📡</span>
              <Code text="python scripts/run_monitor.py" />
              <span style={{ fontSize: "11px", color: "#94A3B8" }}>→ 장중 자동 모니터링</span>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span>🔮</span>
              <Code text="python scripts/run_analysis.py --stock 005930" />
              <span style={{ fontSize: "11px", color: "#94A3B8" }}>→ AI 리포트</span>
            </div>
          </div>
          <p style={{ fontSize: "12px", color: "#FCD34D", marginTop: "10px" }}>이 다음은? → 텔레그램 연결 (OutputPort 구현체만 추가)</p>
        </div>
      </Card>
    </div>
  );
}

/* ════════ Components ════════ */
function Card({ title, color, children }) {
  return (<div style={{ background: "#1E293B", borderRadius: "12px", padding: "14px", borderTop: `3px solid ${color}` }}>
    <h3 style={{ color, fontSize: "15px", marginBottom: "8px" }}>{title}</h3>
    <div style={{ color: "#CBD5E1", fontSize: "13px", lineHeight: "1.6" }}>{children}</div>
  </div>);
}
function Code({ text }) {
  return <pre style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", fontSize: "11px", color: "#10B981", overflow: "auto", lineHeight: "1.5", whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0 }}>{text}</pre>;
}
function PromptCard({ num, title, prompt }) {
  return (<div style={{ background: "#1E293B", borderRadius: "12px", padding: "14px", borderLeft: "4px solid #FCD34D" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
      <span style={{ background: "#FCD34D", color: "#0F172A", borderRadius: "8px", padding: "2px 8px", fontSize: "12px", fontWeight: "800", flexShrink: 0 }}>{num}</span>
      <strong style={{ color: "#FCD34D", fontSize: "13px" }}>{title}</strong>
    </div>
    <pre style={{ background: "#0F172A", padding: "10px", borderRadius: "8px", fontSize: "11px", color: "#FCD34D", overflow: "auto", lineHeight: "1.5", whiteSpace: "pre-wrap", wordBreak: "break-word", borderLeft: "3px solid #FCD34D40" }}>{prompt}</pre>
    <p style={{ fontSize: "10px", color: "#64748B", marginTop: "6px", textAlign: "right" }}>↑ Claude Code에 복사 붙여넣기</p>
  </div>);
}
function PhaseCard({ num, color, time, title, goal, detail, result }) {
  return (<div style={{ background: `${color}15`, padding: "14px", borderRadius: "10px", border: `1px solid ${color}30` }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
        <span style={{ background: color, borderRadius: "50%", width: "28px", height: "28px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px", fontWeight: "800", color: "#0F172A" }}>{num}</span>
        <strong style={{ color, fontSize: "15px" }}>{title}</strong>
      </div>
      <span style={{ fontSize: "11px", color: "#94A3B8", background: "#0F172A", padding: "2px 8px", borderRadius: "4px" }}>{time}</span>
    </div>
    <p style={{ fontSize: "13px", color: "#E2E8F0", marginBottom: "4px" }}><strong>목표:</strong> {goal}</p>
    <p style={{ fontSize: "11px", color: "#94A3B8", marginBottom: "6px" }}>{detail}</p>
    <div style={{ background: "#0F172A", padding: "8px 10px", borderRadius: "6px" }}>
      <span style={{ fontSize: "11px", color: "#10B981" }}>✅ 완료 확인: {result}</span>
    </div>
  </div>);
}
function PhaseBox({ phase, label, desc, color, active }) {
  return (<div style={{ display: "flex", gap: "10px", padding: "8px 10px", background: active ? `${color}15` : "#0F172A", borderRadius: "6px", alignItems: "center", border: active ? `1px solid ${color}40` : "1px solid transparent" }}>
    <span style={{ color, fontWeight: "700", fontSize: "12px", minWidth: "50px" }}>{phase}</span>
    <span style={{ color: "#94A3B8", fontSize: "11px", minWidth: "65px" }}>{label}</span>
    <span style={{ fontSize: "12px", color: active ? "#E2E8F0" : "#64748B" }}>{desc}</span>
  </div>);
}
function CheckItem({ text }) {
  return (<div style={{ display: "flex", gap: "8px", padding: "4px 8px", background: "#0F172A", borderRadius: "4px", alignItems: "center" }}>
    <span style={{ fontSize: "10px", color: "#94A3B8" }}>□</span>
    <span style={{ fontSize: "12px" }}>{text}</span>
  </div>);
}
function Warn({ text }) {
  return (<div style={{ display: "flex", gap: "6px", padding: "6px 10px", background: "#F59E0B15", borderRadius: "6px", alignItems: "flex-start" }}>
    <span style={{ color: "#F59E0B", flexShrink: 0, fontSize: "12px" }}>⚠️</span>
    <span style={{ fontSize: "11px", color: "#FCD34D" }}>{text}</span>
  </div>);
}
