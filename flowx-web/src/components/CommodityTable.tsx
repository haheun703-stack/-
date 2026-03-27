"use client";

import type { CommodityInfo } from "@/lib/types";

const ZONE_STYLE: Record<string, { color: string; label: string }> = {
  buy: { color: "text-green-400", label: "매수구간" },
  watch: { color: "text-yellow-400", label: "관찰" },
  hold: { color: "text-orange-400", label: "보유" },
  overheated: { color: "text-red-400", label: "과열" },
};

export default function CommodityTable({ commodities }: { commodities: CommodityInfo[] }) {
  if (!commodities.length) {
    return <p className="text-gray-500 text-sm">원자재 데이터가 없습니다.</p>;
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs border-b border-gray-800">
              <th className="text-left py-2 px-2">원자재</th>
              <th className="text-right py-2 px-2">현재가</th>
              <th className="text-right py-2 px-2">원가</th>
              <th className="text-right py-2 px-2">갭%</th>
              <th className="text-center py-2 px-2">상태</th>
            </tr>
          </thead>
          <tbody>
            {commodities.map((c) => {
              const zs = ZONE_STYLE[c.zone] ?? { color: "text-gray-400", label: c.zone };
              return (
                <tr key={c.key} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2 px-2">
                    <span className="text-gray-200">{c.name}</span>
                    <span className="text-gray-600 text-xs ml-1">{c.unit}</span>
                  </td>
                  <td className="text-right py-2 px-2 text-gray-300 font-mono">
                    {c.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="text-right py-2 px-2 text-gray-500 font-mono">
                    {c.production_cost.toLocaleString()}
                  </td>
                  <td className={`text-right py-2 px-2 font-mono font-bold ${zs.color}`}>
                    {c.gap_pct.toFixed(1)}%
                  </td>
                  <td className="text-center py-2 px-2">
                    <span className={`text-xs px-2 py-0.5 rounded ${zs.color} bg-gray-800`}>
                      {zs.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 범례 */}
      <div className="flex flex-wrap gap-3 mt-3 text-xs text-gray-600">
        <span><span className="text-green-400">●</span> 매수구간 (&lt;20%)</span>
        <span><span className="text-yellow-400">●</span> 관찰 (20~40%)</span>
        <span><span className="text-orange-400">●</span> 보유 (40~80%)</span>
        <span><span className="text-red-400">●</span> 과열 (80%+)</span>
      </div>
    </div>
  );
}
