"use client";

import { useEffect, useState } from "react";

/* ─── 타입 ─── */
interface SectorFireRow {
  sector: string;
  fire_score: number;
  fire_grade: string;
  flow_score: number;
  inflection_score: number;
  rhythm_score: number;
  energy_score: number;
  overheat_penalty: number;
  fgn_5d: number;
  inst_5d: number;
  pension_5d: number;
  ma20_avg_dev: number;
  rsi_avg: number;
  etf_code: string | null;
  etf_name: string | null;
  etf_recommend: string;
  // Structure Score
  s1_score: number;
  s1_ratio: number;
  s2_score: number;
  s2_stoch_k: number | null;
  s3_score: number;
  structure_score: number;
  structure_grade: string;
  composite_score: number;
  composite_grade: string;
  market_kospi_stoch_k: number | null;
  market_vix: number | null;
  market_disparity: number | null;
}

interface SectorFireData {
  sectors: SectorFireRow[];
  date: string | null;
}

/* ─── 등급 색상 ─── */
const GRADE_STYLE: Record<string, { bg: string; text: string }> = {
  "S+": { bg: "bg-purple-500/20 ring-1 ring-purple-400/40", text: "text-purple-300" },
  S: { bg: "bg-red-500/20 ring-1 ring-red-400/40", text: "text-red-300" },
  A: { bg: "bg-orange-500/20", text: "text-orange-300" },
  B: { bg: "bg-yellow-500/20", text: "text-yellow-300" },
  C: { bg: "bg-gray-500/20", text: "text-gray-400" },
  D: { bg: "bg-gray-700/20", text: "text-gray-500" },
};

function gradeStyle(grade: string) {
  return GRADE_STYLE[grade] ?? GRADE_STYLE["D"];
}

/* ─── 수급 포맷 ─── */
function fmtBillion(v: number) {
  const abs = Math.abs(v);
  const sign = v >= 0 ? "+" : "";
  if (abs >= 10000) return `${sign}${(v / 10000).toFixed(1)}조`;
  return `${sign}${v.toLocaleString()}억`;
}

function supplyColor(v: number) {
  return v > 0 ? "text-red-400" : v < 0 ? "text-blue-400" : "text-gray-500";
}

/* ─── 게이지 바 ─── */
function GaugeBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

/* ─── 메인 컴포넌트 ─── */
export default function SectorFireDashboard() {
  const [data, setData] = useState<SectorFireData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch("/api/sector-fire", { signal: controller.signal });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const json = await res.json();
        // API가 sectors 또는 data 키로 반환할 수 있음
        const sectors = json.sectors ?? json.data ?? [];
        setData({ sectors, date: json.date ?? null });
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setData(null);
      }
      setLoading(false);
    }
    load();
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-16 bg-gray-800 rounded-lg" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="h-40 bg-gray-800 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!data || data.sectors.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">섹터 발화 데이터가 아직 없습니다.</p>
        <p className="text-gray-600 text-sm mt-1">매일 장마감 후 업데이트됩니다.</p>
      </div>
    );
  }

  const { sectors, date } = data;
  const first = sectors[0];
  const hasStructure = first?.structure_score != null && first.structure_score > 0;

  return (
    <div className="space-y-6">
      {/* ── 시장 레짐 헤더 ── */}
      {first?.market_vix != null && (
        <div className="bg-gray-900 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-300 text-sm font-semibold">
              {"🌍"} {"시장 레짐"}
            </h3>
            {date && <span className="text-gray-600 text-xs">{date} {"기준"}</span>}
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-gray-500 text-xs mb-1">VIX</p>
              <p className={`text-lg font-bold ${(first.market_vix ?? 0) <= 20 ? "text-green-400" : (first.market_vix ?? 0) <= 25 ? "text-yellow-400" : "text-red-400"}`}>
                {first.market_vix?.toFixed(1) ?? "N/A"}
              </p>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">KOSPI StochRSI</p>
              <p className={`text-lg font-bold ${(first.market_kospi_stoch_k ?? 50) <= 30 ? "text-green-400" : (first.market_kospi_stoch_k ?? 50) >= 70 ? "text-red-400" : "text-gray-200"}`}>
                {first.market_kospi_stoch_k?.toFixed(0) ?? "N/A"}
              </p>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">{"이격도"}</p>
              <p className={`text-lg font-bold ${(first.market_disparity ?? 100) <= 98 ? "text-green-400" : (first.market_disparity ?? 100) >= 105 ? "text-red-400" : "text-gray-200"}`}>
                {first.market_disparity != null ? `${first.market_disparity.toFixed(1)}%` : "N/A"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── 섹터 카드 그리드 ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {sectors.map((s) => {
          const gs = gradeStyle(hasStructure ? s.composite_grade : s.fire_grade);
          const mainScore = hasStructure ? s.composite_score : s.fire_score;
          const mainGrade = hasStructure ? s.composite_grade : s.fire_grade;

          return (
            <div key={s.sector} className={`${gs.bg} rounded-xl p-4 space-y-3`}>
              {/* 상단: 섹터명 + 점수 + 등급 */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-gray-100 font-bold">{s.sector}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded font-bold ${gs.text} bg-black/30`}>
                    {mainGrade}
                  </span>
                </div>
                <span className={`text-2xl font-black ${gs.text}`}>
                  {mainScore}
                </span>
              </div>

              {/* FIRE / Structure 이중 바 */}
              {hasStructure ? (
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="flex justify-between text-gray-500 mb-1">
                      <span>{"FIRE"}</span>
                      <span className="text-gray-300">{s.fire_score} <span className="text-gray-500">/ {s.fire_grade}</span></span>
                    </div>
                    <GaugeBar value={s.fire_score} max={100} color="bg-orange-500" />
                  </div>
                  <div>
                    <div className="flex justify-between text-gray-500 mb-1">
                      <span>{"Structure"}</span>
                      <span className="text-gray-300">{s.structure_score} <span className="text-gray-500">/ {s.structure_grade}</span></span>
                    </div>
                    <GaugeBar value={s.structure_score} max={100} color="bg-cyan-500" />
                  </div>
                </div>
              ) : (
                /* FIRE only — 4요소 바 */
                <div className="grid grid-cols-4 gap-2 text-xs">
                  {[
                    { label: "F", value: s.flow_score, max: 25, color: "bg-blue-500" },
                    { label: "I", value: s.inflection_score, max: 20, color: "bg-green-500" },
                    { label: "R", value: s.rhythm_score, max: 15, color: "bg-yellow-500" },
                    { label: "E", value: s.energy_score, max: 25, color: "bg-red-500" },
                  ].map((g) => (
                    <div key={g.label}>
                      <div className="flex justify-between text-gray-500 mb-1">
                        <span>{g.label}</span>
                        <span className="text-gray-400">{g.value}</span>
                      </div>
                      <GaugeBar value={g.value} max={g.max} color={g.color} />
                    </div>
                  ))}
                </div>
              )}

              {/* Structure 세부 (S1/S2/S3) */}
              {hasStructure && (
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center">
                    <p className="text-gray-600">{"S1 연간돌파"}</p>
                    <p className="text-gray-300 font-bold">{s.s1_score}<span className="text-gray-600">/40</span></p>
                    {s.s1_ratio > 0 && <p className="text-gray-500">{s.s1_ratio.toFixed(2)}x</p>}
                  </div>
                  <div className="text-center">
                    <p className="text-gray-600">{"S2 StochRSI"}</p>
                    <p className="text-gray-300 font-bold">{s.s2_score}<span className="text-gray-600">/30</span></p>
                    {s.s2_stoch_k != null && <p className="text-gray-500">K={s.s2_stoch_k}</p>}
                  </div>
                  <div className="text-center">
                    <p className="text-gray-600">{"S3 시장"}</p>
                    <p className="text-gray-300 font-bold">{s.s3_score}<span className="text-gray-600">/30</span></p>
                  </div>
                </div>
              )}

              {/* 수급 */}
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-gray-600">{"외인5d"}</span>{" "}
                  <span className={supplyColor(s.fgn_5d)}>{fmtBillion(s.fgn_5d)}</span>
                </div>
                <div>
                  <span className="text-gray-600">{"기관5d"}</span>{" "}
                  <span className={supplyColor(s.inst_5d)}>{fmtBillion(s.inst_5d)}</span>
                </div>
                <div>
                  <span className="text-gray-600">{"연기금"}</span>{" "}
                  <span className={supplyColor(s.pension_5d)}>{fmtBillion(s.pension_5d)}</span>
                </div>
              </div>

              {/* 기술지표 + ETF */}
              <div className="flex items-center justify-between text-xs">
                <div className="flex gap-3">
                  <span>
                    <span className="text-gray-600">MA20</span>{" "}
                    <span className={s.ma20_avg_dev >= 0 ? "text-red-400" : "text-blue-400"}>
                      {s.ma20_avg_dev >= 0 ? "+" : ""}{s.ma20_avg_dev.toFixed(1)}%
                    </span>
                  </span>
                  <span>
                    <span className="text-gray-600">RSI</span>{" "}
                    <span className={s.rsi_avg >= 70 ? "text-red-400" : s.rsi_avg <= 30 ? "text-green-400" : "text-gray-300"}>
                      {s.rsi_avg.toFixed(0)}
                    </span>
                  </span>
                </div>
                {s.etf_name && (
                  <span className="text-blue-400/60 truncate max-w-[140px]">
                    {s.etf_name}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── 등급 범례 ── */}
      <div className="bg-gray-900 rounded-xl p-3 text-xs text-gray-500">
        <p className="mb-1 font-semibold text-gray-400">
          {hasStructure ? "Composite = FIRE×0.6 + Structure×0.4" : "FIRE = F(흐름) + I(반전) + R(리듬) + E(에너지)"}
        </p>
        <p>
          <span className="text-purple-300">S+(85+)</span>{" · "}
          <span className="text-red-300">S(70+)</span>{" · "}
          <span className="text-orange-300">A(55+)</span>{" · "}
          <span className="text-yellow-300">B(40+)</span>{" · "}
          <span className="text-gray-400">C(25+)</span>{" · "}
          <span className="text-gray-500">D(&lt;25)</span>
        </p>
      </div>
    </div>
  );
}
