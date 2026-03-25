"use client";

import { useEffect, useState } from "react";
import type { Scoreboard, Signal } from "@/lib/types";
import PaywallBlur from "./PaywallBlur";

const PERIODS = ["30D", "60D", "90D", "ALL"] as const;
const BOT_TYPES = [
  { key: "QUANT", label: "퀀트봇" },
  { key: "DAYTRADING", label: "단타봇" },
  { key: "ALL", label: "전체" },
] as const;

export default function SignalScoreboard({ isPaid = false }: { isPaid?: boolean }) {
  const [botType, setBotType] = useState("QUANT");
  const [period, setPeriod] = useState<string>("30D");
  const [scoreboard, setScoreboard] = useState<Scoreboard | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const [sbRes, sigRes] = await Promise.all([
        fetch(`/api/scoreboard?bot_type=${botType}&period=${period}`),
        fetch(`/api/signals?bot_type=${botType}&limit=10`),
      ]);
      const sb = await sbRes.json();
      const sigs = await sigRes.json();
      // scoreboard 필드명 호환 (loss_count↔lose_count, avg_return↔avg_return_pct)
      if (sb && !sb.error) {
        sb.lose_count = sb.lose_count ?? sb.loss_count ?? 0;
        sb.avg_return_pct = sb.avg_return_pct ?? sb.avg_return ?? 0;
      }
      setScoreboard(sb?.error ? null : sb);
      // signals API가 {signals: [...]} 또는 [...] 형태일 수 있음
      const sigArr = Array.isArray(sigs) ? sigs : (Array.isArray(sigs?.signals) ? sigs.signals : []);
      setSignals(sigArr);
      setLoading(false);
    }
    load();
  }, [botType, period]);

  if (loading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-800 rounded w-48" />
        <div className="h-32 bg-gray-800 rounded" />
      </div>
    );
  }

  const sb = scoreboard;

  return (
    <div className="space-y-6">
      {/* 봇 타입 탭 */}
      <div className="flex gap-2">
        {BOT_TYPES.map((bt) => (
          <button
            key={bt.key}
            onClick={() => setBotType(bt.key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              botType === bt.key
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {bt.label}
          </button>
        ))}
      </div>

      {/* 기간 탭 */}
      <div className="flex gap-2">
        {PERIODS.map((p) => (
          <button
            key={p}
            onClick={() => setPeriod(p)}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              period === p
                ? "bg-gray-700 text-white"
                : "bg-gray-900 text-gray-500 hover:text-gray-300"
            }`}
          >
            {p}
          </button>
        ))}
      </div>

      {/* 성적표 카드 */}
      {sb && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="총 시그널"
            value={`${sb.total_signals}건`}
            color="text-white"
          />
          <StatCard
            label="승률"
            value={`${sb.win_rate}%`}
            color={sb.win_rate >= 50 ? "text-green-400" : "text-red-400"}
          />
          <StatCard
            label="평균 수익률"
            value={`${sb.avg_return_pct > 0 ? "+" : ""}${sb.avg_return_pct}%`}
            color={sb.avg_return_pct >= 0 ? "text-green-400" : "text-red-400"}
          />
          <StatCard
            label="승/패"
            value={`${sb.win_count}W ${sb.lose_count}L`}
            color="text-gray-300"
          />
        </div>
      )}

      {/* 평균 수익/손실 */}
      {sb && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <p className="text-gray-500 text-xs mb-1">평균 수익 (승)</p>
            <p className="text-green-400 text-xl font-bold">
              +{sb.avg_win_pct}%
            </p>
          </div>
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <p className="text-gray-500 text-xs mb-1">평균 손실 (패)</p>
            <p className="text-red-400 text-xl font-bold">
              {sb.avg_lose_pct}%
            </p>
          </div>
        </div>
      )}

      {/* 최근 시그널 리스트 */}
      <div>
        <h3 className="text-gray-400 text-sm font-medium mb-3">
          최근 시그널
        </h3>
        <PaywallBlur locked={!isPaid && signals.length > 2}>
          <div className="space-y-2">
            {(isPaid ? signals : signals.slice(0, 5)).map((sig, i) => (
              <SignalRow key={sig.id} signal={sig} locked={!isPaid && i >= 2} />
            ))}
            {signals.length === 0 && (
              <p className="text-gray-600 text-sm py-4 text-center">
                아직 시그널이 없습니다
              </p>
            )}
          </div>
        </PaywallBlur>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <p className="text-gray-500 text-xs mb-1">{label}</p>
      <p className={`${color} text-xl font-bold`}>{value}</p>
    </div>
  );
}

function SignalRow({ signal, locked }: { signal: Signal; locked: boolean }) {
  const statusColors: Record<string, string> = {
    OPEN: "bg-blue-600",
    CLOSED: "bg-green-600",
    STOPPED: "bg-red-600",
  };

  const gradeColors: Record<string, string> = {
    AA: "text-yellow-400",
    A: "text-green-400",
    B: "text-blue-400",
    C: "text-gray-400",
  };

  return (
    <div
      className={`flex items-center justify-between bg-gray-900 rounded-lg p-3 border border-gray-800 ${
        locked ? "opacity-50" : ""
      }`}
    >
      <div className="flex items-center gap-3">
        <span
          className={`${statusColors[signal.status]} text-white text-xs px-2 py-0.5 rounded`}
        >
          {signal.status}
        </span>
        <div>
          <span className="text-white text-sm font-medium">
            {locked ? "●●●●" : signal.ticker_name}
          </span>
          <span className="text-gray-500 text-xs ml-2">
            {locked ? "●●●●●●" : signal.ticker}
          </span>
        </div>
        <span className={`${gradeColors[signal.grade]} text-xs font-bold`}>
          {signal.grade}
        </span>
      </div>
      <div className="text-right">
        <p
          className={`text-sm font-bold ${
            signal.return_pct >= 0 ? "text-green-400" : "text-red-400"
          }`}
        >
          {signal.return_pct > 0 ? "+" : ""}
          {signal.return_pct}%
        </p>
        <p className="text-gray-600 text-xs">{signal.signal_date}</p>
      </div>
    </div>
  );
}
