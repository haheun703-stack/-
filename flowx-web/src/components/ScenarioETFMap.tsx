"use client";

import type { ScenarioETF } from "@/lib/types";

export default function ScenarioETFMap({ etfMap }: { etfMap: ScenarioETF[] }) {
  if (!etfMap.length) {
    return <p className="text-gray-500 text-sm">ETF 매핑 데이터가 없습니다.</p>;
  }

  // 시나리오별 그룹핑
  const groups: Record<string, ScenarioETF[]> = {};
  for (const item of etfMap) {
    (groups[item.scenario_name] ??= []).push(item);
  }

  return (
    <div className="space-y-4">
      {Object.entries(groups).map(([name, phases]) => (
        <div key={name} className="rounded-lg border border-gray-700/50 bg-gray-800/30 p-4">
          <h4 className="text-sm font-bold text-gray-200 mb-3">{name}</h4>
          <div className="space-y-2">
            {phases.map((ph) => (
              <div key={`${ph.scenario_id}-${ph.phase}`} className="flex items-start gap-3">
                <span className="text-xs text-gray-500 min-w-[80px] pt-0.5">
                  P{ph.phase} {ph.phase_name.split("(")[0].trim()}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {ph.etfs.map((etf) => (
                    <span
                      key={etf}
                      className="text-xs px-2 py-0.5 rounded-full bg-blue-900/20 text-blue-400 border border-blue-800/50"
                    >
                      {etf}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
