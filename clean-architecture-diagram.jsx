import { useState } from "react";

const layers = [
  {
    id: "entities",
    name: "🟡 Entities",
    nameKr: "엔티티 (핵심 비즈니스 객체)",
    color: "#FEF3C7",
    borderColor: "#F59E0B",
    radius: 80,
    items: ["Stock", "ChartData", "TechnicalPattern", "SupplyDemandZone", "Condition", "FlowPrediction"],
    description: "외부 의존 없는 순수 비즈니스 객체. DB나 API가 바뀌어도 이 코드는 절대 변하지 않습니다.",
    files: ["src/entities/models.py"]
  },
  {
    id: "usecases",
    name: "🔴 Use Cases",
    nameKr: "유스케이스 (비즈니스 로직)",
    color: "#FECDD3",
    borderColor: "#F43F5E",
    radius: 160,
    items: ["StockAnalysisInteractor", "Ports (인터페이스)", "차트분석", "흐름예측", "조건판단"],
    description: "서브에이전트들을 조합하여 실제 분석을 수행하는 핵심 로직. Port(인터페이스)로만 외부와 소통합니다.",
    files: ["src/use_cases/stock_analysis.py", "src/use_cases/ports.py"]
  },
  {
    id: "adapters",
    name: "🟢 Interface Adapters",
    nameKr: "어댑터 (서브에이전트 + 변환기)",
    color: "#D1FAE5",
    borderColor: "#10B981",
    radius: 240,
    items: ["ChartAnalysisAgent", "VolumeAnalysisAgent", "FlowPredictionAgent", "ConditionJudgeAgent", "MarkdownPresenter"],
    description: "Port를 구현하는 실제 AI 에이전트들. Claude API를 호출하여 분석을 수행하고 결과를 Entity로 변환합니다.",
    files: ["src/agents/*.py", "src/adapters/*.py"]
  },
  {
    id: "frameworks",
    name: "🔵 Frameworks & Drivers",
    nameKr: "프레임워크 (외부 도구)",
    color: "#DBEAFE",
    borderColor: "#3B82F6",
    radius: 310,
    items: ["Claude API", "증권 API", "웹 스크래핑", "파일 저장소", "Claude Code CLI"],
    description: "가장 바깥 계층. 실제 외부 서비스와의 연결. 교체가 가장 쉬운 계층입니다.",
    files: ["requirements.txt", "외부 라이브러리"]
  }
];

const agents = [
  { name: "차트분석", icon: "📊", desc: "기술적 패턴/지표 분석", color: "#10B981" },
  { name: "거래량분석", icon: "📈", desc: "거래량 & 매물대 분석", color: "#10B981" },
  { name: "흐름예측", icon: "🔮", desc: "내일 흐름 종합 예측", color: "#8B5CF6" },
  { name: "조건판단", icon: "⚖️", desc: "유지/대응 조건 생성", color: "#F43F5E" },
  { name: "오케스트레이터", icon: "🎯", desc: "전체 워크플로우 조율", color: "#3B82F6" },
];

const workflow = [
  { step: 1, name: "데이터 수집", desc: "차트 + 이슈", agent: "오케스트레이터" },
  { step: 2, name: "패턴 분석", desc: "캔들, 이평선, RSI, MACD", agent: "차트분석" },
  { step: 3, name: "거래량 분석", desc: "매물대, 매집/분산", agent: "거래량분석" },
  { step: 4, name: "흐름 예측", desc: "내일 방향 + 가격 범위", agent: "흐름예측" },
  { step: 5, name: "조건 판단", desc: "유지조건 / 대응조건", agent: "조건판단" },
  { step: 6, name: "리포트 출력", desc: "마크다운 리포트", agent: "오케스트레이터" },
];

export default function CleanArchDiagram() {
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [activeTab, setActiveTab] = useState("architecture");

  const selected = layers.find(l => l.id === selectedLayer);

  return (
    <div style={{ fontFamily: "'Pretendard', -apple-system, sans-serif", background: "#0F172A", color: "#E2E8F0", minHeight: "100vh", padding: "24px" }}>
      <h1 style={{ textAlign: "center", fontSize: "28px", marginBottom: "8px", background: "linear-gradient(135deg, #60A5FA, #A78BFA)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
        🏗️ 클린 아키텍처 기반 주식분석 서브에이전트
      </h1>
      <p style={{ textAlign: "center", color: "#94A3B8", marginBottom: "24px" }}>Clean Architecture × AI Sub-Agents for Stock Analysis</p>

      {/* Tab Navigation */}
      <div style={{ display: "flex", justifyContent: "center", gap: "8px", marginBottom: "24px" }}>
        {[
          { id: "architecture", label: "🏗️ 아키텍처" },
          { id: "agents", label: "🤖 서브에이전트" },
          { id: "workflow", label: "⚡ 워크플로우" },
          { id: "output", label: "📋 출력 형식" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "10px 20px",
              borderRadius: "8px",
              border: "none",
              background: activeTab === tab.id ? "#3B82F6" : "#1E293B",
              color: activeTab === tab.id ? "white" : "#94A3B8",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: activeTab === tab.id ? "600" : "400",
              transition: "all 0.2s"
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Architecture Tab */}
      {activeTab === "architecture" && (
        <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", justifyContent: "center" }}>
          <div style={{ position: "relative", width: "400px", height: "400px" }}>
            {[...layers].reverse().map((layer) => (
              <div
                key={layer.id}
                onClick={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
                style={{
                  position: "absolute",
                  left: "50%",
                  top: "50%",
                  transform: "translate(-50%, -50%)",
                  width: layer.radius * 2,
                  height: layer.radius * 2,
                  borderRadius: "50%",
                  background: layer.color + "30",
                  border: `3px solid ${selectedLayer === layer.id ? layer.borderColor : layer.borderColor + "60"}`,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.3s",
                  boxShadow: selectedLayer === layer.id ? `0 0 20px ${layer.borderColor}40` : "none",
                  zIndex: 4 - layers.indexOf(layer),
                }}
              >
                {layer.radius <= 100 && (
                  <span style={{ fontSize: "13px", fontWeight: "600", color: layer.borderColor, textAlign: "center" }}>
                    {layer.name}
                  </span>
                )}
              </div>
            ))}
            {/* Labels for outer rings */}
            <div style={{ position: "absolute", top: "12px", left: "50%", transform: "translateX(-50%)", fontSize: "11px", color: "#3B82F6", fontWeight: "600" }}>Frameworks</div>
            <div style={{ position: "absolute", top: "82px", left: "50%", transform: "translateX(-50%)", fontSize: "11px", color: "#10B981", fontWeight: "600" }}>Adapters</div>
            <div style={{ position: "absolute", top: "132px", left: "50%", transform: "translateX(-50%)", fontSize: "11px", color: "#F43F5E", fontWeight: "600" }}>Use Cases</div>

            {/* Dependency arrow */}
            <div style={{ position: "absolute", left: "8px", top: "50%", transform: "translateY(-50%)", display: "flex", alignItems: "center", gap: "4px" }}>
              <span style={{ color: "#94A3B8", fontSize: "20px" }}>→</span>
              <span style={{ color: "#94A3B8", fontSize: "10px", writingMode: "vertical-rl" }}>의존성 방향</span>
            </div>
          </div>

          <div style={{ flex: 1, minWidth: "300px", maxWidth: "450px" }}>
            {selected ? (
              <div style={{ background: "#1E293B", borderRadius: "12px", padding: "20px", border: `2px solid ${selected.borderColor}40` }}>
                <h3 style={{ color: selected.borderColor, marginBottom: "12px" }}>{selected.name}</h3>
                <p style={{ color: "#94A3B8", marginBottom: "8px", fontSize: "14px" }}>{selected.nameKr}</p>
                <p style={{ color: "#CBD5E1", fontSize: "13px", lineHeight: "1.6", marginBottom: "16px" }}>{selected.description}</p>
                <div style={{ marginBottom: "12px" }}>
                  <strong style={{ color: "#E2E8F0", fontSize: "13px" }}>포함 요소:</strong>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "8px" }}>
                    {selected.items.map(item => (
                      <span key={item} style={{ background: selected.borderColor + "20", color: selected.borderColor, padding: "4px 10px", borderRadius: "6px", fontSize: "12px" }}>
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <strong style={{ color: "#E2E8F0", fontSize: "13px" }}>파일:</strong>
                  <div style={{ marginTop: "6px" }}>
                    {selected.files.map(f => (
                      <code key={f} style={{ display: "block", color: "#60A5FA", fontSize: "12px", marginBottom: "4px" }}>{f}</code>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ background: "#1E293B", borderRadius: "12px", padding: "20px", textAlign: "center" }}>
                <p style={{ color: "#64748B", fontSize: "14px" }}>👈 원형 다이어그램의 레이어를 클릭하세요</p>
                <div style={{ marginTop: "16px", textAlign: "left" }}>
                  <p style={{ color: "#94A3B8", fontSize: "13px", lineHeight: "1.8" }}>
                    <strong style={{ color: "#F59E0B" }}>핵심 원칙: 의존성 규칙</strong><br/>
                    의존성은 항상 <strong style={{ color: "#60A5FA" }}>바깥 → 안쪽</strong>으로만 향합니다.<br/><br/>
                    • 🟡 Entities는 아무것도 모름<br/>
                    • 🔴 Use Cases는 Entities만 앎<br/>
                    • 🟢 Adapters는 Use Cases + Entities를 앎<br/>
                    • 🔵 Frameworks는 모든 것을 앎<br/><br/>
                    이 덕분에 <strong style={{ color: "#10B981" }}>DB나 API를 바꿔도 핵심 로직은 그대로</strong>입니다.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Agents Tab */}
      {activeTab === "agents" && (
        <div style={{ maxWidth: "700px", margin: "0 auto" }}>
          <div style={{ display: "grid", gap: "12px" }}>
            {agents.map(agent => (
              <div key={agent.name} style={{ background: "#1E293B", borderRadius: "12px", padding: "16px", borderLeft: `4px solid ${agent.color}`, display: "flex", alignItems: "center", gap: "16px" }}>
                <span style={{ fontSize: "32px" }}>{agent.icon}</span>
                <div>
                  <h4 style={{ color: agent.color, marginBottom: "4px" }}>{agent.name} Agent</h4>
                  <p style={{ color: "#94A3B8", fontSize: "13px", margin: 0 }}>{agent.desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div style={{ background: "#1E293B", borderRadius: "12px", padding: "16px", marginTop: "16px" }}>
            <h4 style={{ color: "#A78BFA", marginBottom: "8px" }}>🔄 에이전트 간 통신</h4>
            <p style={{ color: "#94A3B8", fontSize: "13px", lineHeight: "1.6" }}>
              모든 에이전트는 <strong style={{ color: "#60A5FA" }}>Port(인터페이스)</strong>를 통해 소통합니다.
              오케스트레이터가 각 에이전트를 순차/병렬로 호출하고, 결과를 <strong style={{ color: "#F59E0B" }}>Entity 객체</strong>로 주고받습니다.
              에이전트를 교체하거나 추가해도 다른 에이전트에 영향이 없습니다.
            </p>
          </div>
        </div>
      )}

      {/* Workflow Tab */}
      {activeTab === "workflow" && (
        <div style={{ maxWidth: "600px", margin: "0 auto" }}>
          {workflow.map((w, i) => (
            <div key={w.step} style={{ display: "flex", gap: "16px", marginBottom: "4px" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: "40px" }}>
                <div style={{ width: "36px", height: "36px", borderRadius: "50%", background: "#3B82F6", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "700", fontSize: "14px" }}>
                  {w.step}
                </div>
                {i < workflow.length - 1 && <div style={{ width: "2px", height: "36px", background: "#334155" }} />}
              </div>
              <div style={{ background: "#1E293B", borderRadius: "10px", padding: "12px 16px", flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <strong style={{ color: "#E2E8F0", fontSize: "14px" }}>{w.name}</strong>
                  <span style={{ color: "#60A5FA", fontSize: "11px", background: "#1E3A5F", padding: "2px 8px", borderRadius: "4px" }}>{w.agent}</span>
                </div>
                <p style={{ color: "#94A3B8", fontSize: "12px", margin: "4px 0 0" }}>{w.desc}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Output Tab */}
      {activeTab === "output" && (
        <div style={{ maxWidth: "700px", margin: "0 auto" }}>
          <div style={{ background: "#1E293B", borderRadius: "12px", padding: "20px" }}>
            <h3 style={{ color: "#60A5FA", marginBottom: "16px" }}>📋 분석 리포트 출력 구조</h3>
            {[
              { num: "1️⃣", title: "업체 정보", desc: "종목코드, 종목명, 시장, 업종" },
              { num: "2️⃣", title: "기술적 형태/지표", desc: "캔들 패턴, 이평선, RSI, MACD, 볼린저, 스토캐스틱" },
              { num: "3️⃣", title: "거래량 분석", desc: "거래량 비율, 추세, 매집/분산 신호" },
              { num: "4️⃣", title: "핵심 분석 포인트", desc: "중요도별 정렬된 분석 요약" },
              { num: "5️⃣", title: "매물대 분석", desc: "지지대/저항대 가격 구간, 강도" },
              { num: "6️⃣", title: "최근 이슈", desc: "종목별 뉴스, 감성 분석" },
              { num: "7️⃣", title: "🔮 내일 흐름 예측", desc: "방향, 확신도, 예상 가격 범위, 핵심 요인" },
              { num: "8️⃣", title: "📋 유지/대응 조건", desc: "", highlight: true },
            ].map(item => (
              <div key={item.num} style={{ 
                display: "flex", gap: "12px", padding: "10px", marginBottom: "6px", borderRadius: "8px",
                background: item.highlight ? "#1E3A5F" : "transparent",
                border: item.highlight ? "1px solid #3B82F640" : "none"
              }}>
                <span style={{ fontSize: "16px" }}>{item.num}</span>
                <div>
                  <strong style={{ color: item.highlight ? "#60A5FA" : "#E2E8F0", fontSize: "14px" }}>{item.title}</strong>
                  {item.desc && <p style={{ color: "#94A3B8", fontSize: "12px", margin: "2px 0 0" }}>{item.desc}</p>}
                  {item.highlight && (
                    <div style={{ marginTop: "8px", display: "flex", gap: "12px" }}>
                      <div style={{ background: "#10B98120", padding: "8px 12px", borderRadius: "8px", flex: 1 }}>
                        <strong style={{ color: "#10B981", fontSize: "12px" }}>✅ 유지 조건</strong>
                        <p style={{ color: "#94A3B8", fontSize: "11px", margin: "4px 0 0" }}>지지대 위 유지, 정배열 유지, RSI 정상범위...</p>
                      </div>
                      <div style={{ background: "#F43F5E20", padding: "8px 12px", borderRadius: "8px", flex: 1 }}>
                        <strong style={{ color: "#F43F5E", fontSize: "12px" }}>🚨 대응 조건</strong>
                        <p style={{ color: "#94A3B8", fontSize: "11px", margin: "4px 0 0" }}>지지대 이탈→손절, 목표가→익절, 급락→긴급매도...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
