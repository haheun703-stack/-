"use client";

import { useState } from "react";

const FAQ = [
  {
    q: "시나리오란 무엇인가요?",
    a: "뉴스/지표에서 감지된 거시경제 이벤트(전쟁, 유가 급등, 금리 변동 등)가 한국 주식시장에 미치는 영향을 단계별로 추적하는 시스템입니다. 각 시나리오는 '이벤트 → 원자재 영향 → 수혜 섹터 → 종목'의 인과관계를 따릅니다.",
  },
  {
    q: "Phase(단계)란 무엇인가요?",
    a: "하나의 시나리오가 시간 경과에 따라 다른 섹터에 영향을 미치는 과정입니다. 예: 중동 전쟁 시나리오 → Phase 1(방산 직접 수혜) → Phase 2(유가 전파, 정유사) → Phase 3(안전자산 이동) → Phase 4(해운/에너지 2차 파급). 현재 어떤 Phase에 있는지 파란색으로 표시됩니다.",
  },
  {
    q: "원가갭이란 무엇인가요?",
    a: "원자재의 현재 시장가격과 생산원가의 차이를 퍼센트로 나타낸 것입니다. 갭이 20% 미만이면 '매수구간'(생산원가에 가까워 반등 가능성), 80% 이상이면 '과열'(가격이 원가 대비 너무 높아 조정 가능성)을 의미합니다.",
  },
  {
    q: "HOT/COLD 섹터란?",
    a: "HOT 섹터(초록)는 현재 Phase에서 수혜가 예상되는 업종, COLD 섹터(빨강)는 피해가 예상되는 업종입니다. 시나리오와 Phase가 바뀌면 HOT/COLD도 달라집니다.",
  },
  {
    q: "충돌 경고란?",
    a: "시나리오 엔진이 'HOT'으로 판단한 섹터에 실제로는 외국인·기관이 순매도(자금 유출)하고 있을 때 나타나는 경고입니다. 이론과 현실의 괴리를 뜻하며, Phase 전환이나 시나리오 종료 신호일 수 있습니다.",
  },
];

export default function ScenarioGuide() {
  const [open, setOpen] = useState<Record<number, boolean>>({});

  return (
    <div className="space-y-2">
      {FAQ.map((item, i) => (
        <div key={i} className="border border-gray-700/50 rounded-lg overflow-hidden">
          <button
            className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-gray-800/30 transition-colors"
            onClick={() => setOpen((p) => ({ ...p, [i]: !p[i] }))}
          >
            <span className="text-sm text-gray-200">{item.q}</span>
            <span className="text-gray-500 text-sm ml-2">{open[i] ? "−" : "+"}</span>
          </button>
          {open[i] && (
            <div className="px-4 pb-3">
              <p className="text-sm text-gray-400 leading-relaxed">{item.a}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
