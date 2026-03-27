"use client";

import { useState } from "react";
import type { ActiveScenario, SupplyConflict } from "@/lib/types";

function scoreColor(score: number) {
  if (score >= 60) return "text-red-400 bg-red-900/30 border-red-800/50";
  if (score >= 40) return "text-yellow-400 bg-yellow-900/30 border-yellow-800/50";
  return "text-gray-400 bg-gray-800 border-gray-700/50";
}

function PhaseTimeline({ chain, currentPhase }: { chain: ActiveScenario["chain"]; currentPhase: number }) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto py-2">
      {chain.map((ph, i) => {
        const isCurrent = ph.phase === currentPhase;
        const isPast = ph.phase < currentPhase;
        return (
          <div key={ph.phase} className="flex items-center">
            {i > 0 && <div className={`w-4 h-0.5 ${isPast ? "bg-gray-600" : isCurrent ? "bg-blue-500" : "bg-gray-800"}`} />}
            <div className="flex flex-col items-center min-w-[60px]">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                  ${isCurrent ? "bg-blue-600 text-white" : isPast ? "bg-gray-600 text-gray-300" : "bg-gray-800 text-gray-600"}`}
              >
                {ph.phase}
              </div>
              <p className={`text-[10px] mt-1 text-center leading-tight ${isCurrent ? "text-blue-400" : "text-gray-600"}`}>
                {ph.name.split("(")[0].trim()}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SectorPills({ sectors, type }: { sectors: string[]; type: "hot" | "cold" }) {
  if (!sectors.length) return null;
  const style = type === "hot"
    ? "text-green-400 border-green-800/50 bg-green-900/20"
    : "text-red-400 border-red-800/50 bg-red-900/20";
  return (
    <div className="flex flex-wrap gap-1.5">
      {sectors.map((s) => (
        <span key={s} className={`text-xs px-2 py-0.5 rounded-full border ${style}`}>{s}</span>
      ))}
    </div>
  );
}

export default function ScenarioCard({
  scenarios,
  conflicts,
}: {
  scenarios: ActiveScenario[];
  conflicts: SupplyConflict[];
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  if (!scenarios.length) {
    return <p className="text-gray-500 text-sm">활성 시나리오가 없습니다.</p>;
  }

  const conflictMap: Record<string, SupplyConflict[]> = {};
  for (const c of conflicts) {
    (conflictMap[c.scenario] ??= []).push(c);
  }

  return (
    <div className="space-y-4">
      {scenarios.map((sc) => {
        const open = expanded[sc.id] ?? false;
        const scConflicts = conflictMap[sc.id] || [];
        return (
          <div key={sc.id} className={`rounded-lg border ${scoreColor(sc.score)} p-4`}>
            {/* 헤더 */}
            <button
              className="w-full text-left flex items-center justify-between"
              onClick={() => setExpanded((p) => ({ ...p, [sc.id]: !p[sc.id] }))}
            >
              <div className="flex items-center gap-3">
                <span className="text-lg font-bold">{sc.name}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300">
                  P{sc.current_phase}/{sc.total_phases}
                </span>
                <span className="text-xs text-gray-500">D+{sc.days_active}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono font-bold">{sc.score}점</span>
                <span className="text-gray-500 text-sm">{open ? "▲" : "▼"}</span>
              </div>
            </button>

            {/* 충돌 경고 */}
            {scConflicts.map((cf, i) => (
              <div key={i} className="mt-2 text-xs bg-red-900/30 border border-red-800/50 rounded px-3 py-1.5 text-red-300">
                {cf.warning}
              </div>
            ))}

            {/* 요약 (항상 표시) */}
            <div className="mt-3 space-y-2">
              <p className="text-sm text-gray-400">{sc.phase_name}</p>
              <div className="flex gap-4">
                <div>
                  <span className="text-[10px] text-gray-600 block">HOT</span>
                  <SectorPills sectors={sc.hot_sectors} type="hot" />
                </div>
                <div>
                  <span className="text-[10px] text-gray-600 block">COLD</span>
                  <SectorPills sectors={sc.cold_sectors} type="cold" />
                </div>
              </div>
            </div>

            {/* 펼치기: 상세 */}
            {open && (
              <div className="mt-4 space-y-3 border-t border-gray-700/50 pt-3">
                {/* Phase 타임라인 */}
                <PhaseTimeline chain={sc.chain} currentPhase={sc.current_phase} />

                {/* 전략 논리 */}
                {sc.logic && (
                  <p className="text-sm text-gray-300 bg-gray-800/50 rounded px-3 py-2">{sc.logic}</p>
                )}

                {/* 근거 */}
                {sc.reasons.length > 0 && (
                  <div>
                    <span className="text-[10px] text-gray-600 block mb-1">근거</span>
                    <div className="flex flex-wrap gap-1.5">
                      {sc.reasons.map((r, i) => (
                        <span key={i} className="text-xs text-gray-400 bg-gray-800 px-2 py-0.5 rounded">{r}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* HOT 종목 */}
                {sc.hot_tickers.length > 0 && (
                  <div>
                    <span className="text-[10px] text-gray-600 block mb-1">HOT 종목</span>
                    <div className="flex flex-wrap gap-1.5">
                      {sc.hot_tickers.map((t) => (
                        <span key={t.code} className="text-xs text-green-400 bg-green-900/20 border border-green-800/50 px-2 py-0.5 rounded">
                          {t.name}
                          <span className="text-gray-600 ml-1">{t.code}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* ETF */}
                {sc.etf.length > 0 && (
                  <div>
                    <span className="text-[10px] text-gray-600 block mb-1">관련 ETF</span>
                    <div className="flex flex-wrap gap-1.5">
                      {sc.etf.map((e) => (
                        <span key={e} className="text-xs text-blue-400 bg-blue-900/20 border border-blue-800/50 px-2 py-0.5 rounded">{e}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* 다음 Phase 프리뷰 */}
                {sc.next_phase_name && (
                  <div className="border border-dashed border-gray-700 rounded-lg p-3 bg-gray-900/50">
                    <p className="text-xs text-gray-500 mb-1">다음 Phase 프리뷰</p>
                    <p className="text-sm text-gray-300">{sc.next_phase_name}</p>
                    {sc.next_hot.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {sc.next_hot.map((s) => (
                          <span key={s} className="text-xs text-gray-400 bg-gray-800 px-2 py-0.5 rounded">{s}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
