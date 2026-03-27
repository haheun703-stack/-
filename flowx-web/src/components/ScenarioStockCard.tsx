"use client";

import type { ScenarioStock } from "@/lib/types";

function formatKRW(n: number) {
  if (n >= 1_000_000) return `${(n / 10_000).toFixed(0)}만`;
  return n.toLocaleString();
}

function flowLabel(val: number) {
  if (val === 0) return { text: "-", color: "text-gray-600" };
  const bil = val / 1_000_000_000;
  return {
    text: `${bil >= 0 ? "+" : ""}${bil.toFixed(0)}억`,
    color: bil >= 0 ? "text-red-400" : "text-blue-400",
  };
}

export default function ScenarioStockCard({ stocks }: { stocks: ScenarioStock[] }) {
  if (!stocks.length) {
    return <p className="text-gray-500 text-sm">시나리오 연동 종목이 없습니다.</p>;
  }

  return (
    <div className="space-y-3">
      {stocks.map((st) => (
        <div key={st.ticker} className="rounded-lg border border-gray-700/50 bg-gray-800/30 p-4">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-gray-200 font-bold">{st.name}</span>
              <span className="text-gray-600 text-xs">{st.ticker}</span>
              <span className="text-xs px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-800/50">
                {st.scenario_tag}
              </span>
            </div>
            <span className={`text-sm font-bold ${st.total_score >= 60 ? "text-green-400" : st.total_score >= 40 ? "text-yellow-400" : "text-gray-400"}`}>
              {st.total_score.toFixed(1)}점
            </span>
          </div>

          {/* 시나리오 내러티브 */}
          {st.scenario_narrative && (
            <p className="text-xs text-gray-400 mb-3 leading-relaxed">{st.scenario_narrative}</p>
          )}

          {/* 6칸 그리드 */}
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2 text-xs">
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">현재가</span>
              <span className="text-gray-200 font-mono">{formatKRW(st.close)}</span>
            </div>
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">진입가</span>
              <span className="text-gray-300 font-mono">{formatKRW(st.entry_price)}</span>
            </div>
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">목표가</span>
              <span className="text-green-400 font-mono">{formatKRW(st.target_price)}</span>
            </div>
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">손절가</span>
              <span className="text-red-400 font-mono">{formatKRW(st.stop_loss)}</span>
            </div>
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">RSI</span>
              <span className={`font-mono ${st.rsi >= 70 ? "text-red-400" : st.rsi <= 30 ? "text-green-400" : "text-gray-300"}`}>
                {st.rsi.toFixed(1)}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded p-2">
              <span className="text-gray-600 block">수급 5일</span>
              <div className="flex gap-1">
                <span className={`font-mono ${flowLabel(st.foreign_5d).color}`}>
                  외{flowLabel(st.foreign_5d).text}
                </span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
