"use client";

import type { MarketStatus } from "@/lib/types";

const VERDICT_STYLE: Record<string, { color: string; bg: string }> = {
  "강력 포착": { color: "text-green-400", bg: "bg-green-900/30" },
  "포착": { color: "text-green-400", bg: "bg-green-900/30" },
  "관망": { color: "text-yellow-400", bg: "bg-yellow-900/30" },
  "경계": { color: "text-red-400", bg: "bg-red-900/30" },
  적극매수: { color: "text-green-400", bg: "bg-green-900/30" },
  매수: { color: "text-green-400", bg: "bg-green-900/30" },
  매도: { color: "text-red-400", bg: "bg-red-900/30" },
};

export default function MarketStatusBar({ status }: { status: MarketStatus }) {
  const vs = VERDICT_STYLE[status.verdict] ?? { color: "text-gray-400", bg: "bg-gray-800" };

  const cards = [
    {
      label: "시장 판단",
      value: status.verdict,
      sub: `레짐: ${status.regime}`,
      color: vs.color,
      bg: vs.bg,
    },
    {
      label: "KOSPI",
      value: (Number(status.kospi) || 0).toLocaleString(),
      sub: `${(Number(status.kospi_chg) || 0) >= 0 ? "+" : ""}${(Number(status.kospi_chg) || 0).toFixed(2)}%`,
      color: (Number(status.kospi_chg) || 0) >= 0 ? "text-green-400" : "text-red-400",
      bg: "bg-gray-800/50",
    },
    {
      label: "VIX",
      value: (Number(status.vix) || 0).toFixed(1),
      sub: (Number(status.vix) || 0) >= 25 ? "공포" : (Number(status.vix) || 0) >= 20 ? "경계" : "안정",
      color: (Number(status.vix) || 0) >= 25 ? "text-red-400" : (Number(status.vix) || 0) >= 20 ? "text-yellow-400" : "text-green-400",
      bg: "bg-gray-800/50",
    },
    {
      label: "현금 비중",
      value: `${Number(status.cash_pct) || 0}%`,
      sub: `SHIELD: ${status.shield_status ?? "N/A"}`,
      color: (Number(status.cash_pct) || 0) >= 50 ? "text-yellow-400" : "text-gray-300",
      bg: "bg-gray-800/50",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cards.map((c) => (
        <div key={c.label} className={`${c.bg} rounded-lg p-3 border border-gray-700/50`}>
          <p className="text-gray-500 text-xs mb-1">{c.label}</p>
          <p className={`text-lg font-bold ${c.color}`}>{c.value}</p>
          <p className="text-gray-500 text-xs mt-0.5">{c.sub}</p>
        </div>
      ))}
    </div>
  );
}
