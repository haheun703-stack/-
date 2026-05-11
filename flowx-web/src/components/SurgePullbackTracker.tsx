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

interface PerfEntry {
  ticker: string;
  name: string;
  entry_price: number;
  current_price: number;
  current_pct: number;
  max_gain: number;
  max_loss: number;
  days_held: number;
}

interface PerfSummary {
  total_signals: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_return: number;
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

/** 눌림% → 행 배경 클래스 */
function pbRowBg(pb: number) {
  if (pb <= -10) return "bg-red-500/10";
  if (pb <= -8) return "bg-orange-500/5";
  return "";
}

/** 눌림% → 글자색 클래스 */
function pbTextColor(pb: number) {
  if (pb <= -10) return "text-[#FF4444] font-bold";
  if (pb <= -8) return "text-[#FF8C00]";
  if (pb <= -5) return "text-[#F59E0B]";
  if (pb <= 0) return "text-[#64748B]";
  return "text-[#3fb950]";
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

/** performance 객체에서 _summary와 개별 항목 추출 */
function parsePerformance(perf: Record<string, unknown> | undefined | null) {
  if (!perf) return { summary: null, entries: [] };
  const summary = (perf._summary ?? null) as PerfSummary | null;
  const entries: PerfEntry[] = [];
  for (const [key, val] of Object.entries(perf)) {
    if (key === "_summary" || !val || typeof val !== "object") continue;
    const v = val as Record<string, unknown>;
    if (v.ticker && v.entry_price) {
      entries.push(v as unknown as PerfEntry);
    }
  }
  return { summary, entries };
}

/** 모니터링 주차 계산 (5/9 시작 기준) */
function monitorWeek() {
  const start = new Date("2026-05-09");
  const now = new Date();
  const diff = Math.floor((now.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
  const week = Math.max(1, Math.ceil((diff + 1) / 7));
  return `${week}\uC8FC\uCC28`;
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
      <div className="animate-pulse space-y-4">
        <div className="h-6 bg-gray-800 rounded w-48" />
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-24 bg-gray-800 rounded-lg" />
          ))}
        </div>
        <div className="h-40 bg-gray-800 rounded-lg" />
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
  const expired = currentRows.filter((r) => r.status === "expired");

  const { summary: perfSummary, entries: perfEntries } = parsePerformance(
    summary?.performance,
  );

  // 임박 종목 (pullback_pct <= -5, 정렬)
  const nearSignal = (summary?.active ?? [])
    .filter((a) => a.pullback_pct <= -5)
    .sort((a, b) => a.pullback_pct - b.pullback_pct);

  return (
    <div className="space-y-6">
      {/* ── 헤더 ── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white text-lg font-bold">
            {"\uD83D\uDD25 \uC0C1\uD55C\uAC00 \uB20C\uB9BC\uBAA9 \uC5D4\uC9C4"}
          </h2>
          <p className="text-gray-500 text-xs mt-0.5">
            {"\uAE09\uB4F1 \uD6C4 \uB20C\uB9BC \uAD6C\uAC04\uC5D0\uC11C \uBD84\uD560\uB9E4\uC218 \uC9C4\uC785\uC810 \uD3EC\uCC29"}
          </p>
        </div>
        {summary?.date && (
          <span className="text-gray-500 text-xs">{summary.date}</span>
        )}
      </div>

      {/* ── 섹션 0: 요약 카드 5개 ── */}
      {summary && (
        <SummaryCards
          summary={summary}
          perfSummary={perfSummary}
        />
      )}

      {/* ── 섹션 1: ★ 매수 시그널 ── */}
      <SignalSection
        signals={summary?.recent_signals ?? []}
        nearSignal={nearSignal}
        perfEntries={perfEntries}
        activeCount={summary?.active_count ?? 0}
      />

      {/* ── 날짜 선택 ── */}
      {dates.length > 1 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-gray-500 text-xs">{"\uB0A0\uC9DC:"}</span>
          {dates.slice(0, 10).map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDate(d)}
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
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

      {/* ── 섹션 2: 감시 중 종목 테이블 ── */}
      <WatchTable rows={watching} />

      {/* ── 만료 종목 (접힌 상태) ── */}
      {expired.length > 0 && (
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-lg p-4">
          <h3 className="text-gray-500 text-sm font-medium mb-2">
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

      {/* ── 섹션 3: 수익률 추적 ── */}
      <PerformanceSection perfSummary={perfSummary} perfEntries={perfEntries} />

      {/* ── 섹션 4: 전략 파라미터 ── */}
      <div className="bg-gray-900/30 border border-gray-800/50 rounded-lg p-4 text-gray-600 text-xs space-y-1">
        <p className="text-gray-500 font-medium">{"\uC804\uB7B5 \uD30C\uB77C\uBBF8\uD130"}</p>
        <p>{"\u2022 \uAE09\uB4F1 \uAE30\uC900: +15% \uC774\uC0C1 | \uB20C\uB9BC \uAE30\uC900: \uD53C\uD06C \uB300\uBE44 -10% | \uAC10\uC2DC: 3\uAC70\uB798\uC77C"}</p>
        <p>{"\u2022 \uD488\uC9C8 \uD544\uD130: \uC8FC\uAC00 \u2265 1\uB9CC\uC6D0, \uAC70\uB798\uB300\uAE08 \u2265 10\uC5B5/\uC77C"}</p>
        <p>{"\u2022 Layer 1: v10c \uD050\uB808\uC774\uC158 93\uC885\uBAA9 (\uBC31\uD14C\uC2A4\uD2B8 91.7% \uC2B9\uB960)"}</p>
        <p>{"\u2022 Layer 2: \uC77C\uC77C \uC804\uC885\uBAA9 \uC2A4\uCE94 \u2192 \uD488\uC9C8 \uD544\uD130 \uD1B5\uACFC \uC790\uB3D9 \uD3B8\uC785"}</p>
        <p className="text-yellow-700 mt-2">{"\u26A0 3\uC8FC \uBAA8\uB2C8\uD130\uB9C1 \uAE30\uAC04 (5/9~5/30) \u2014 \uC2E4\uC804 \uC790\uB3D9\uB9E4\uB9E4 \uC804 \uAC80\uC99D \uC911"}</p>
      </div>
    </div>
  );
}

/* ─── 섹션 0: 요약 카드 5개 ─── */

function SummaryCards({
  summary,
  perfSummary,
}: {
  summary: SummaryData;
  perfSummary: PerfSummary | null;
}) {
  const signals = summary.recent_signals ?? [];
  const hasSignals = signals.length > 0;
  const winRate = perfSummary?.win_rate ?? 0;
  const avgReturn = perfSummary?.avg_return ?? 0;
  const wins = perfSummary?.wins ?? 0;
  const losses = perfSummary?.losses ?? 0;

  const winRateColor =
    winRate >= 80 ? "text-green-400" : winRate >= 50 ? "text-yellow-400" : winRate > 0 ? "text-red-400" : "text-gray-400";
  const avgRetColor = avgReturn >= 0 ? "text-green-400" : "text-red-400";

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      {/* 감시 중 */}
      <div className="bg-gray-900 border border-blue-800/50 rounded-lg p-3">
        <div className="text-gray-500 text-xs">{"\uAC10\uC2DC \uC911"}</div>
        <div className="text-blue-400 text-2xl font-bold mt-1">{summary.active_count}</div>
        <div className="text-gray-600 text-xs mt-0.5">L1:{summary.l1_count} / L2:{summary.l2_count}</div>
      </div>

      {/* 매수 시그널 */}
      <div className={`bg-gray-900 border rounded-lg p-3 ${
        hasSignals ? "border-red-600 animate-pulse" : "border-gray-800"
      }`}>
        <div className="text-gray-500 text-xs">{"\uB9E4\uC218 \uC2DC\uADF8\uB110"}</div>
        <div className={`text-2xl font-bold mt-1 ${hasSignals ? "text-red-400" : "text-gray-400"}`}>
          {signals.length}{"\uAC74"}
        </div>
        <div className="text-gray-600 text-xs mt-0.5">
          {"\uB204\uC801 "}{summary.total_signals}{"\uAC74"}
        </div>
      </div>

      {/* 승률 */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
        <div className="text-gray-500 text-xs">{"\uC2B9\uB960"}</div>
        <div className={`text-2xl font-bold mt-1 ${winRateColor}`}>
          {perfSummary ? `${winRate.toFixed(1)}%` : "-"}
        </div>
        <div className="text-gray-600 text-xs mt-0.5">
          {perfSummary ? `${wins}\uC2B9 / ${losses}\uD328` : "\uB370\uC774\uD130 \uC5C6\uC74C"}
        </div>
      </div>

      {/* 평균 수익 */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
        <div className="text-gray-500 text-xs">{"\uD3C9\uADE0 \uC218\uC775"}</div>
        <div className={`text-2xl font-bold mt-1 ${perfSummary ? avgRetColor : "text-gray-400"}`}>
          {perfSummary ? `${avgReturn >= 0 ? "+" : ""}${avgReturn.toFixed(2)}%` : "-"}
        </div>
        <div className="text-gray-600 text-xs mt-0.5">{"\uC2DC\uADF8\uB110 \uD6C4"}</div>
      </div>

      {/* 모니터링 */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
        <div className="text-gray-500 text-xs">{"\uBAA8\uB2C8\uD130\uB9C1"}</div>
        <div className="text-purple-400 text-2xl font-bold mt-1">{monitorWeek()}</div>
        <div className="text-gray-600 text-xs mt-0.5">5/9~5/30</div>
      </div>
    </div>
  );
}

/* ─── 섹션 1: ★ 매수 시그널 ─── */

function SignalSection({
  signals,
  nearSignal,
  perfEntries,
  activeCount,
}: {
  signals: SignalEntry[];
  nearSignal: SummaryData["active"];
  perfEntries: PerfEntry[];
  activeCount: number;
}) {
  const hasSignals = signals.length > 0;

  if (!hasSignals) {
    // 시그널 없을 때: 임박 종목 표시
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-gray-300 font-bold mb-3">
          {"\u2605 \uB9E4\uC218 \uC2DC\uADF8\uB110"}
        </h3>
        <p className="text-gray-500 text-sm mb-3">
          {"\uD604\uC7AC \uBC1C\uC0DD\uD55C \uC2DC\uADF8\uB110\uC774 \uC5C6\uC2B5\uB2C8\uB2E4."}
        </p>
        {nearSignal.length > 0 ? (
          <div>
            <p className="text-gray-400 text-xs mb-2">
              {"\uAC10\uC2DC \uC911 "}{activeCount}{"\uAC74 \uC911 \uB20C\uB9BC \uC784\uBC15 \uC885\uBAA9:"}
            </p>
            <div className="space-y-1.5">
              {nearSignal.slice(0, 5).map((a, i) => {
                const pct = a.pullback_pct;
                const icon = pct <= -8 ? "\uD83D\uDD34" : "\uD83D\uDFE1";
                const nearTarget = pct <= -9;
                return (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span>{icon}</span>
                    {layerBadge(a.layer)}
                    <span className="text-gray-200">{a.name}</span>
                    <span className={`font-medium ${pct <= -8 ? "text-[#FF8C00]" : "text-[#F59E0B]"}`}>
                      {pct.toFixed(1)}%
                    </span>
                    {nearTarget && (
                      <span className="text-red-400 text-xs">
                        {`(\uBAA9\uD45C -10%) \u2190 \uAC70\uC758 \uB3C4\uB2EC!`}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <p className="text-gray-600 text-xs">
            {"\uAC10\uC2DC \uC885\uBAA9 \uC911 -5% \uC774\uC0C1 \uB20C\uB9BC \uC885\uBAA9\uC774 \uC544\uC9C1 \uC5C6\uC2B5\uB2C8\uB2E4"}
          </p>
        )}
      </div>
    );
  }

  // 시그널 있을 때: 빨간 배경 + 상세 카드
  return (
    <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-4">
      <h3 className="text-red-400 font-bold mb-4">
        {"\u2605 \uB9E4\uC218 \uC2DC\uADF8\uB110 \u2014 "}{signals.length}{"\uAC74 \uBC1C\uC0DD!"}
      </h3>
      <div className="space-y-4">
        {signals.map((s, i) => {
          // 매칭 수익률 데이터 찾기
          const perf = perfEntries.find((p) => p.ticker === s.ticker);
          return (
            <div key={i} className="bg-black/30 rounded-lg p-4 border border-red-900/30">
              <div className="flex items-center gap-2 mb-3">
                {layerBadge(s.layer)}
                <span className="text-gray-100 font-bold">{s.name}</span>
                <span className="text-gray-500 text-xs">({s.ticker})</span>
                {s.sectors?.length > 0 && (
                  <span className="text-gray-600 text-xs">[{s.sectors.join(", ")}]</span>
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-3">
                <div>
                  <span className="text-gray-500 text-xs">{"\uAE09\uB4F1\uC77C"}</span>
                  <p className="text-gray-300">{fmtDate(s.surge_date)} <span className="text-green-400">+{s.surge_pct}%</span></p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">{"\uD53C\uD06C\uAC00"}</span>
                  <p className="text-gray-300">{fmtPrice(s.peak_price)}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">{"\uC9C4\uC785\uAC00"}</span>
                  <p className="text-yellow-400 font-bold">
                    {fmtPrice(s.entry_price)}
                    <span className="text-red-400 text-xs ml-1">({s.pullback_pct.toFixed(1)}%)</span>
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">{"\uC2DC\uADF8\uB110\uC77C"}</span>
                  <p className="text-gray-300">{fmtDate(s.signal_date)} D{s.watch_day}</p>
                </div>
              </div>

              {/* 수익률 추적 박스 */}
              {perf && (
                <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                  <p className="text-gray-500 text-xs mb-2">{"\uC2DC\uADF8\uB110 \uD6C4 \uC218\uC775\uB960"}</p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <span className="text-gray-500 text-xs">{"\uD604\uC7AC"}</span>
                      <p className={`text-lg font-bold ${perf.current_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {perf.current_pct >= 0 ? "+" : ""}{perf.current_pct.toFixed(2)}%
                      </p>
                      <p className="text-gray-500 text-xs">{fmtPrice(perf.current_price)}</p>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">{"\uCD5C\uACE0"}</span>
                      <p className="text-green-400 font-medium">+{perf.max_gain.toFixed(1)}%</p>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">{"\uCD5C\uC800"}</span>
                      <p className="text-red-400 font-medium">{perf.max_loss.toFixed(1)}%</p>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">{"\uBCF4\uC720"}</span>
                      <p className="text-gray-300 font-medium">{perf.days_held}{"\uC77C"}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── 섹션 2: 감시 중 종목 테이블 ─── */

function WatchTable({ rows }: { rows: WatchEntry[] }) {
  if (rows.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-500 text-sm">{"\uAC10\uC2DC \uC911\uC778 \uC885\uBAA9\uC774 \uC5C6\uC2B5\uB2C8\uB2E4"}</p>
      </div>
    );
  }

  const sorted = [...rows].sort((a, b) => a.pullback_pct - b.pullback_pct);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h3 className="text-blue-400 font-bold mb-3">
        {"\uD83D\uDD0D \uAC10\uC2DC \uC911"} ({rows.length}{"\uAC74"})
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs border-b border-gray-800">
              <th className="text-left py-2">{""}</th>
              <th className="text-left py-2">Layer</th>
              <th className="text-left py-2">{"\uC885\uBAA9"}</th>
              <th className="text-left py-2 hidden md:table-cell">{"\uC139\uD130"}</th>
              <th className="text-right py-2">{"\uAE09\uB4F1%"}</th>
              <th className="text-right py-2 hidden md:table-cell">{"\uD53C\uD06C"}</th>
              <th className="text-right py-2">{"\uD604\uC7AC\uAC00"}</th>
              <th className="text-right py-2">{"\uB20C\uB9BC%"}</th>
              <th className="text-center py-2">D</th>
              <th className="text-right py-2 hidden md:table-cell">{"\uAC70\uB798\uB300\uAE08"}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={i} className={`border-b border-gray-800/50 ${pbRowBg(r.pullback_pct)}`}>
                <td className="py-1.5">{pbIcon(r.pullback_pct)}</td>
                <td className="py-1.5">{layerBadge(r.layer)}</td>
                <td className="py-1.5 text-gray-200">
                  {r.name}
                  <span className="text-gray-600 text-xs ml-1">{r.ticker}</span>
                </td>
                <td className="py-1.5 text-gray-500 text-xs hidden md:table-cell">{r.sector}</td>
                <td className="py-1.5 text-right text-green-400">+{r.surge_pct}%</td>
                <td className="py-1.5 text-right text-gray-300 hidden md:table-cell">{fmtPrice(r.peak_price)}</td>
                <td className="py-1.5 text-right text-gray-200">{fmtPrice(r.latest_close)}</td>
                <td className={`py-1.5 text-right ${pbTextColor(r.pullback_pct)}`}>
                  {r.pullback_pct.toFixed(1)}%
                </td>
                <td className="py-1.5 text-center text-gray-400">D{r.watch_day}</td>
                <td className="py-1.5 text-right text-gray-500 hidden md:table-cell">{fmtTv(r.trading_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── 섹션 3: 수익률 추적 ─── */

function PerformanceSection({
  perfSummary,
  perfEntries,
}: {
  perfSummary: PerfSummary | null;
  perfEntries: PerfEntry[];
}) {
  if (!perfSummary && perfEntries.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-gray-300 font-bold mb-2">
          {"\uD83D\uDCCA \uC2DC\uADF8\uB110 \uD6C4 \uC218\uC775\uB960 \uCD94\uC801"}
        </h3>
        <p className="text-gray-500 text-sm">{"\uC544\uC9C1 \uBC1C\uC0DD\uD55C \uC2DC\uADF8\uB110\uC774 \uC5C6\uC2B5\uB2C8\uB2E4"}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-300 font-bold">
          {"\uD83D\uDCCA \uC2DC\uADF8\uB110 \uD6C4 \uC218\uC775\uB960 \uCD94\uC801"}
        </h3>
        <span className="text-gray-500 text-xs">{perfEntries.length}{"\uAC74 \uCD94\uC801\uC911"}</span>
      </div>

      {perfEntries.length > 0 && (
        <div className="overflow-x-auto mb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs border-b border-gray-800">
                <th className="text-left py-2">{"\uC885\uBAA9"}</th>
                <th className="text-right py-2">{"\uC9C4\uC785\uAC00"}</th>
                <th className="text-right py-2">{"\uD604\uC7AC\uAC00"}</th>
                <th className="text-right py-2">{"\uC218\uC775\uB960"}</th>
                <th className="text-right py-2">{"\uCD5C\uACE0"}</th>
                <th className="text-right py-2 hidden md:table-cell">{"\uCD5C\uC800"}</th>
                <th className="text-right py-2">{"\uBCF4\uC720\uC77C"}</th>
              </tr>
            </thead>
            <tbody>
              {perfEntries.map((p, i) => (
                <tr key={i} className="border-b border-gray-800/50">
                  <td className="py-2 text-gray-200 font-medium">{p.name}</td>
                  <td className="py-2 text-right text-gray-400">{fmtPrice(p.entry_price)}</td>
                  <td className="py-2 text-right text-gray-200">{fmtPrice(p.current_price)}</td>
                  <td className="py-2 text-right">
                    <span className={`font-bold ${p.current_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {p.current_pct >= 0 ? "+" : ""}{p.current_pct.toFixed(2)}%
                    </span>
                    {/* 프로그레스 바 */}
                    <div className="flex items-center mt-1">
                      <div className="w-full bg-gray-800 rounded-full h-1.5 relative">
                        {p.current_pct >= 0 ? (
                          <div
                            className="bg-green-500 h-1.5 rounded-full"
                            style={{ width: `${Math.min(Math.abs(p.current_pct) * 5, 100)}%` }}
                          />
                        ) : (
                          <div
                            className="bg-red-500 h-1.5 rounded-full ml-auto"
                            style={{ width: `${Math.min(Math.abs(p.current_pct) * 5, 100)}%` }}
                          />
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="py-2 text-right text-green-400/70 text-xs">+{p.max_gain.toFixed(1)}%</td>
                  <td className="py-2 text-right text-red-400/70 text-xs hidden md:table-cell">{p.max_loss.toFixed(1)}%</td>
                  <td className="py-2 text-right text-gray-400">{p.days_held}{"\uC77C"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="text-gray-600 text-xs space-y-0.5">
        <p>{"\u203B 3\uC8FC \uBAA8\uB2C8\uD130\uB9C1 \uAE30\uAC04 (5/9~5/30)"}</p>
        <p>{"\u2192 \uC774\uD6C4 \uC2E4\uC804 \uC790\uB3D9\uB9E4\uB9E4 \uC804\uD658 \uC608\uC815"}</p>
      </div>
    </div>
  );
}
