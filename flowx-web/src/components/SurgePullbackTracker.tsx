"use client";

import { useEffect, useState } from "react";

/* ─── 타입 ─── */

interface WatchEntry {
  ticker: string;
  name: string;
  layer: number;
  sector: string;
  status: string;
  surge_date: string;
  surge_pct: number;
  surge_close: number;
  peak_price: number;
  latest_close: number;
  pullback_pct: number;
  watch_day: number;
  signal_date: string | null;
  entry_price: number;
  trading_value: number;
  date: string;
}

interface SignalEntry {
  ticker: string;
  name: string;
  layer: number;
  sectors: string[];
  surge_date: string;
  surge_pct: number;
  peak_price: number;
  entry_price: number;
  pullback_pct: number;
  watch_day: number;
  signal_date: string;
}

interface SummaryData {
  date: string;
  updated: string | null;
  active_count: number;
  l1_count: number;
  l2_count: number;
  total_signals: number;
  history_count: number;
  active: {
    ticker: string;
    name: string;
    layer: number;
    sectors: string[];
    surge_pct: number;
    pullback_pct: number;
    watch_day: number;
    peak_price: number;
    latest_close: number;
  }[];
  recent_signals: SignalEntry[];
  performance: Record<string, unknown>;
}

interface ApiResponse {
  summary: SummaryData | null;
  rows: WatchEntry[];
  dates: string[];
  byDate: Record<string, WatchEntry[]>;
}

/* ─── 유틸 ─── */

function fmtDate(d: string) {
  if (!d) return "-";
  const s = d.replace(/-/g, "");
  if (s.length === 8) return `${s.slice(4, 6)}/${s.slice(6, 8)}`;
  return d;
}

function fmtPrice(n: number) {
  if (!n) return "-";
  return n.toLocaleString() + "\uC6D0";
}

function fmtTv(n: number) {
  if (!n) return "-";
  return (n / 1e8).toFixed(0) + "\uC5B5";
}

function pbIcon(pb: number) {
  if (pb <= -8) return "\uD83D\uDD34";
  if (pb <= -5) return "\uD83D\uDFE1";
  return "\uD83D\uDFE2";
}

function statusBadge(status: string) {
  switch (status) {
    case "signal":
      return (
        <span className="px-2 py-0.5 rounded text-xs font-bold bg-red-600 text-white">
          BUY
        </span>
      );
    case "watching":
      return (
        <span className="px-2 py-0.5 rounded text-xs bg-blue-900 text-blue-300">
          {"\uAC10\uC2DC"}
        </span>
      );
    case "expired":
      return (
        <span className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-500">
          {"\uB9CC\uB8CC"}
        </span>
      );
    default:
      return (
        <span className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-500">
          {status}
        </span>
      );
  }
}

function layerBadge(layer: number) {
  return layer === 1 ? (
    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-yellow-700 text-yellow-200">
      L1
    </span>
  ) : (
    <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 text-gray-400">
      L2
    </span>
  );
}

/* ─── 메인 컴포넌트 ─── */

export default function SurgePullbackTracker() {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch("/api/surge-pullback", {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setData(json);
      } catch (err) {
        if (err instanceof Error && err.name !== "AbortError") {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    })();
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="text-gray-500 text-center py-20">
        {"\uB20C\uB9BC\uBAA9 \uC5D4\uC9C4 \uB370\uC774\uD130 \uB85C\uB529..."}
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-400 text-center py-20">
        {"\uC624\uB958: "}{error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-gray-500 text-center py-20">
        {"\uB370\uC774\uD130 \uC5C6\uC74C \u2014 \uC7A5\uB9C8\uAC10 \uD6C4 \uC5C5\uB85C\uB4DC\uB418\uBA74 \uD45C\uC2DC\uB429\uB2C8\uB2E4"}
      </div>
    );
  }

  const summary = data.summary;
  const dates = data.dates;
  const currentDate = selectedDate || (dates.length > 0 ? dates[0] : null);
  const currentRows = currentDate ? data.byDate[currentDate] ?? [] : [];

  const watching = currentRows.filter((r) => r.status === "watching");
  const signaled = currentRows.filter((r) => r.status === "signal");
  const expired = currentRows.filter((r) => r.status === "expired");

  return (
    <div className="space-y-6">
      {/* ── 요약 카드 ── */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard
            label={"\uAC10\uC2DC \uC911"}
            value={summary.active_count}
            sub={`L1: ${summary.l1_count} / L2: ${summary.l2_count}`}
            color="blue"
          />
          <StatCard
            label={"\uB204\uC801 \uC2DC\uADF8\uB110"}
            value={summary.total_signals}
            sub={"\uCD1D \uBC1C\uC0DD \uAC74\uC218"}
            color="red"
          />
          <StatCard
            label={"\uD788\uC2A4\uD1A0\uB9AC"}
            value={summary.history_count}
            sub={"\uB9CC\uB8CC+\uC2DC\uADF8\uB110"}
            color="gray"
          />
          <StatCard
            label={"\uAE30\uB85D \uC77C\uC218"}
            value={dates.length}
            sub={"\uC77C"}
            color="green"
          />
          <StatCard
            label={"\uC5C5\uB370\uC774\uD2B8"}
            value={summary.updated ? fmtDate(summary.updated.split(" ")[0]) : "-"}
            sub={summary.date}
            color="purple"
          />
        </div>
      )}

      {/* ── 최근 시그널 (매수 신호) ── */}
      {summary?.recent_signals && summary.recent_signals.length > 0 && (
        <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-4">
          <h3 className="text-red-400 font-bold mb-3">
            {"\u2605 \uCD5C\uADFC \uB9E4\uC218 \uC2DC\uADF8\uB110"}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="text-left py-2">Layer</th>
                  <th className="text-left py-2">{"\uC885\uBAA9"}</th>
                  <th className="text-right py-2">{"\uAE09\uB4F1"}</th>
                  <th className="text-right py-2">{"\uD53C\uD06C"}</th>
                  <th className="text-right py-2">{"\uC9C4\uC785\uAC00"}</th>
                  <th className="text-right py-2">{"\uB20C\uB9BC"}</th>
                  <th className="text-left py-2">{"\uC2DC\uADF8\uB110\uC77C"}</th>
                </tr>
              </thead>
              <tbody>
                {summary.recent_signals.map((s, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2">{layerBadge(s.layer)}</td>
                    <td className="py-2 text-gray-200 font-medium">
                      {s.name}
                      <span className="text-gray-600 text-xs ml-1">{s.ticker}</span>
                    </td>
                    <td className="py-2 text-right text-green-400">+{s.surge_pct}%</td>
                    <td className="py-2 text-right text-gray-300">{fmtPrice(s.peak_price)}</td>
                    <td className="py-2 text-right text-yellow-400 font-bold">{fmtPrice(s.entry_price)}</td>
                    <td className="py-2 text-right text-red-400">{s.pullback_pct.toFixed(1)}%</td>
                    <td className="py-2 text-gray-400">{fmtDate(s.signal_date)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── 날짜 선택 ── */}
      {dates.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-gray-500 text-sm">{"\uB0A0\uC9DC:"}</span>
          {dates.slice(0, 15).map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDate(d)}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                d === currentDate
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {fmtDate(d)}
            </button>
          ))}
        </div>
      )}

      {/* ── 감시 중 종목 ── */}
      {watching.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-blue-400 font-bold mb-3">
            {"\uD83D\uDD0D \uAC10\uC2DC \uC911"} ({watching.length}{"\uAC74"})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="text-left py-2">{""}</th>
                  <th className="text-left py-2">Layer</th>
                  <th className="text-left py-2">{"\uC885\uBAA9"}</th>
                  <th className="text-left py-2">{"\uC139\uD130"}</th>
                  <th className="text-right py-2">{"\uAE09\uB4F1%"}</th>
                  <th className="text-right py-2">{"\uD53C\uD06C"}</th>
                  <th className="text-right py-2">{"\uD604\uC7AC\uAC00"}</th>
                  <th className="text-right py-2">{"\uB20C\uB9BC%"}</th>
                  <th className="text-center py-2">D</th>
                  <th className="text-right py-2">{"\uAC70\uB798\uB300\uAE08"}</th>
                </tr>
              </thead>
              <tbody>
                {watching
                  .sort((a, b) => a.pullback_pct - b.pullback_pct)
                  .map((r, i) => (
                    <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-1.5">{pbIcon(r.pullback_pct)}</td>
                      <td className="py-1.5">{layerBadge(r.layer)}</td>
                      <td className="py-1.5 text-gray-200">
                        {r.name}
                        <span className="text-gray-600 text-xs ml-1">{r.ticker}</span>
                      </td>
                      <td className="py-1.5 text-gray-500 text-xs">{r.sector}</td>
                      <td className="py-1.5 text-right text-green-400">+{r.surge_pct}%</td>
                      <td className="py-1.5 text-right text-gray-300">{fmtPrice(r.peak_price)}</td>
                      <td className="py-1.5 text-right text-gray-200">{fmtPrice(r.latest_close)}</td>
                      <td className={`py-1.5 text-right font-medium ${
                        r.pullback_pct <= -8
                          ? "text-red-400"
                          : r.pullback_pct <= -5
                            ? "text-yellow-400"
                            : "text-gray-400"
                      }`}>
                        {r.pullback_pct.toFixed(1)}%
                      </td>
                      <td className="py-1.5 text-center text-gray-400">D{r.watch_day}</td>
                      <td className="py-1.5 text-right text-gray-500">{fmtTv(r.trading_value)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── 시그널 발생 종목 ── */}
      {signaled.length > 0 && (
        <div className="bg-red-950/20 border border-red-900/40 rounded-lg p-4">
          <h3 className="text-red-400 font-bold mb-3">
            {"\u2605 \uC2DC\uADF8\uB110 \uBC1C\uC0DD"} ({signaled.length}{"\uAC74"})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="text-left py-2">Layer</th>
                  <th className="text-left py-2">{"\uC885\uBAA9"}</th>
                  <th className="text-right py-2">{"\uAE09\uB4F1%"}</th>
                  <th className="text-right py-2">{"\uD53C\uD06C"}</th>
                  <th className="text-right py-2">{"\uC9C4\uC785\uAC00"}</th>
                  <th className="text-right py-2">{"\uB20C\uB9BC%"}</th>
                </tr>
              </thead>
              <tbody>
                {signaled.map((r, i) => (
                  <tr key={i} className="border-b border-gray-800/50">
                    <td className="py-2">{layerBadge(r.layer)}</td>
                    <td className="py-2 text-gray-200 font-medium">{r.name}</td>
                    <td className="py-2 text-right text-green-400">+{r.surge_pct}%</td>
                    <td className="py-2 text-right text-gray-300">{fmtPrice(r.peak_price)}</td>
                    <td className="py-2 text-right text-yellow-400 font-bold">{fmtPrice(r.entry_price)}</td>
                    <td className="py-2 text-right text-red-400">{r.pullback_pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── 만료 종목 ── */}
      {expired.length > 0 && (
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-lg p-4">
          <h3 className="text-gray-500 font-bold mb-3">
            {"\uB9CC\uB8CC"} ({expired.length}{"\uAC74"})
          </h3>
          <div className="flex flex-wrap gap-2">
            {expired.map((r, i) => (
              <span key={i} className="text-xs text-gray-600 bg-gray-800/50 px-2 py-1 rounded">
                {r.name} +{r.surge_pct}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── 전략 설명 ── */}
      <div className="bg-gray-900/30 border border-gray-800/50 rounded-lg p-4 text-gray-600 text-xs space-y-1">
        <p className="text-gray-500 font-medium">{"\uC804\uB7B5 \uD30C\uB77C\uBBF8\uD130"}</p>
        <p>{"\u2022 \uAE09\uB4F1 \uAE30\uC900: +15% \uC774\uC0C1 | \uB20C\uB9BC \uAE30\uC900: \uD53C\uD06C \uB300\uBE44 -10% | \uAC10\uC2DC: 3\uAC70\uB798\uC77C"}</p>
        <p>{"\u2022 \uD488\uC9C8 \uD544\uD130: \uC8FC\uAC00 \u2265 1\uB9CC\uC6D0, \uAC70\uB798\uB300\uAE08 \u2265 10\uC5B5/\uC77C"}</p>
        <p>{"\u2022 Layer 1: v10c \uD050\uB808\uC774\uC158 93\uC885\uBAA9 (\uBC31\uD14C\uC2A4\uD2B8 91.7% \uC2B9\uB960)"}</p>
        <p>{"\u2022 Layer 2: \uC77C\uC77C \uC804\uC885\uBAA9 \uC2A4\uCE94 \u2192 \uD488\uC9C8 \uD544\uD130 \uD1B5\uACFC \uC790\uB3D9 \uD3B8\uC785"}</p>
        <p className="text-yellow-700 mt-2">{"\u26A0 3\uC8FC \uBAA8\uB2C8\uD130\uB9C1 \uAE30\uAC04 \u2014 \uC2E4\uC804 \uC790\uB3D9\uB9E4\uB9E4 \uC804 \uAC80\uC99D \uC911"}</p>
      </div>
    </div>
  );
}

/* ─── StatCard ─── */

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string | number;
  sub: string;
  color: string;
}) {
  const colors: Record<string, string> = {
    blue: "border-blue-800/50 text-blue-400",
    red: "border-red-800/50 text-red-400",
    green: "border-green-800/50 text-green-400",
    gray: "border-gray-800 text-gray-400",
    purple: "border-purple-800/50 text-purple-400",
  };
  const c = colors[color] || colors.gray;

  return (
    <div className={`bg-gray-900 border ${c.split(" ")[0]} rounded-lg p-3`}>
      <div className="text-gray-500 text-xs">{label}</div>
      <div className={`text-2xl font-bold mt-1 ${c.split(" ")[1]}`}>
        {value}
      </div>
      <div className="text-gray-600 text-xs mt-0.5">{sub}</div>
    </div>
  );
}
