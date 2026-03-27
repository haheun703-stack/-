"use client";

import { useEffect, useState } from "react";

interface PickItem {
  ticker: string;
  name: string;
  grade: string;
  total_score: number;
  sources: string[];
  n_sources: number;
  close: number;
  rsi: number;
  stoch_k: number;
  foreign_5d: number;
  inst_5d: number;
  reasons: string[];
  score_breakdown?: {
    entry_price?: number;
    stop_loss?: number;
    target_price?: number;
  };
  entry_info?: {
    entry: number;
    stop: number;
    target: number;
  };
}

interface JarvisData {
  picks: {
    target_date_label?: string;
    mode_label?: string;
    total_candidates?: number;
    stats?: Record<string, number>;
    picks?: PickItem[];
  } | null;
  accuracy: Record<string, { hit_rate?: number; total?: number }> | null;
  brain: {
    regime?: string;
    direction?: string;
    vix?: number;
    cash_ratio?: number;
    recommendation?: string;
  } | null;
  shield: {
    status?: string;
    sector_concentration?: number;
    max_drawdown?: number;
  } | null;
  updated_at: string | null;
}

const GRADE_COLORS: Record<string, string> = {
  "적극매수": "bg-red-600 text-white",
  "매수": "bg-green-600 text-white",
  "관심매수": "bg-blue-600 text-white",
  "관찰": "bg-yellow-600 text-white",
  "보류": "bg-gray-700 text-gray-400",
};

const REGIME_DISPLAY: Record<string, { icon: string; color: string }> = {
  BULL: { icon: "🟢", color: "text-green-400" },
  BEAR: { icon: "🔴", color: "text-red-400" },
  CAUTION: { icon: "🟡", color: "text-yellow-400" },
  NEUTRAL: { icon: "⚪", color: "text-gray-400" },
};

const SHIELD_DISPLAY: Record<string, { icon: string; color: string }> = {
  GREEN: { icon: "🟢", color: "text-green-400" },
  YELLOW: { icon: "🟡", color: "text-yellow-400" },
  RED: { icon: "🔴", color: "text-red-400" },
};

export default function JarvisControlTower() {
  const [data, setData] = useState<JarvisData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch("/api/quant", { signal: controller.signal });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        setData(await res.json());
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
      <div className="animate-pulse space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 bg-gray-800 rounded-lg" />
          ))}
        </div>
        <div className="h-64 bg-gray-800 rounded-lg" />
      </div>
    );
  }

  if (!data || !data.picks) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">퀀트 데이터가 아직 없습니다.</p>
        <p className="text-gray-600 text-sm mt-1">매일 장마감 후 업데이트됩니다.</p>
      </div>
    );
  }

  const { picks, accuracy, brain, shield } = data;
  const regime = brain?.regime || brain?.direction || "NEUTRAL";
  const regimeInfo = REGIME_DISPLAY[regime] || REGIME_DISPLAY.NEUTRAL;
  const shieldStatus = shield?.status || "YELLOW";
  const shieldInfo = SHIELD_DISPLAY[shieldStatus] || SHIELD_DISPLAY.YELLOW;

  const allPicks = picks?.picks ?? [];
  const buyable = allPicks.filter((p) =>
    ["적극매수", "매수", "관심매수", "관찰"].includes(p.grade),
  );
  buyable.sort((a, b) => b.total_score - a.total_score);

  const stats = picks?.stats ?? {};

  // 시그널 소스별 포착 수
  const sourceCounts: Record<string, number> = {};
  for (const p of allPicks) {
    for (const s of p.sources ?? []) {
      sourceCounts[s] = (sourceCounts[s] || 0) + 1;
    }
  }

  return (
    <div className="space-y-8">
      {/* 1. 상태판: 레짐 + SHIELD + 날짜 + 모드 */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatusCard
          label="레짐"
          value={regime}
          icon={regimeInfo.icon}
          color={regimeInfo.color}
          sub={brain?.vix ? `VIX ${brain.vix}` : undefined}
        />
        <StatusCard
          label="SHIELD"
          value={shieldStatus}
          icon={shieldInfo.icon}
          color={shieldInfo.color}
          sub={
            shield?.max_drawdown
              ? `MDD ${shield.max_drawdown.toFixed(1)}%`
              : undefined
          }
        />
        <StatusCard
          label="현금비중"
          value={brain?.cash_ratio ? `${brain.cash_ratio}%` : "-"}
          icon="💰"
          color="text-blue-400"
        />
        <StatusCard
          label="대상일"
          value={picks?.target_date_label ?? "-"}
          icon="📅"
          color="text-gray-300"
          sub={picks?.mode_label}
        />
      </section>

      {/* 2. 등급 분포 + 시그널 소스 */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 등급 분포 */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-gray-400 text-xs mb-3">등급 분포 ({picks?.total_candidates ?? 0}종목)</h3>
          <div className="space-y-2">
            {Object.entries(stats).map(([grade, count]) => (
              <div key={grade} className="flex items-center justify-between">
                <span className={`text-xs px-2 py-0.5 rounded ${GRADE_COLORS[grade] || "bg-gray-700 text-gray-400"}`}>
                  {grade}
                </span>
                <div className="flex-1 mx-3 bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-blue-600 rounded-full h-2"
                    style={{
                      width: `${Math.min(((count as number) / (picks?.total_candidates ?? 1)) * 100, 100)}%`,
                    }}
                  />
                </div>
                <span className="text-gray-400 text-xs w-12 text-right">{count as number}건</span>
              </div>
            ))}
          </div>
        </div>

        {/* 시그널 소스 활성도 */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-gray-400 text-xs mb-3">시그널 소스 활성도</h3>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(sourceCounts)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 10)
              .map(([source, count]) => (
                <div key={source} className="flex items-center justify-between bg-gray-800 rounded px-2 py-1">
                  <span className="text-gray-300 text-xs">{source}</span>
                  <span className="text-blue-400 text-xs font-bold">{count}</span>
                </div>
              ))}
          </div>
        </div>
      </section>

      {/* 3. 시그널 정확도 */}
      {accuracy && (
        <section>
          <h2 className="text-white text-lg font-bold mb-3">시그널 정확도</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {Object.entries(accuracy)
              .filter(([, v]) => typeof v === "object" && v !== null && ((v as Record<string, number>).total ?? 0) > 0)
              .sort(([, a], [, b]) => ((b as Record<string, number>).hit_rate ?? 0) - ((a as Record<string, number>).hit_rate ?? 0))
              .map(([name, detail]) => {
                const d = detail as Record<string, number>;
                const rate = d.hit_rate ?? 0;
                const color = rate >= 60 ? "text-green-400" : rate >= 45 ? "text-yellow-400" : "text-red-400";
                return (
                  <div key={name} className="bg-gray-900 rounded-lg p-3 border border-gray-800 text-center">
                    <p className="text-gray-500 text-xs">{name}</p>
                    <p className={`${color} text-xl font-bold`}>{rate}%</p>
                    <p className="text-gray-600 text-xs">{d.total ?? 0}건</p>
                  </div>
                );
              })}
          </div>
        </section>
      )}

      {/* 4. TOP 추천 종목 */}
      <section>
        <h2 className="text-white text-lg font-bold mb-3">
          {buyable.length > 0
            ? `추천 종목 (${buyable.length}건)`
            : "전체 관망 — 조건 강화 대기"}
        </h2>
        {buyable.length > 0 ? (
          <div className="space-y-3">
            {buyable.map((p) => (
              <PickRow key={p.ticker} pick={p} />
            ))}
          </div>
        ) : (
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 text-center">
            <p className="text-gray-400 text-sm">
              현재 교차검증 통과 종목이 없습니다
            </p>
            <p className="text-gray-600 text-xs mt-1">
              2개 이상 시그널 + 거래량 2배 이상 조건 충족 시 표시됩니다
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

function StatusCard({
  label,
  value,
  icon,
  color,
  sub,
}: {
  label: string;
  value: string;
  icon: string;
  color: string;
  sub?: string;
}) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <p className="text-gray-500 text-xs">{label}</p>
      <p className={`${color} text-lg font-bold mt-1`}>
        {icon} {value}
      </p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

function PickRow({ pick }: { pick: PickItem }) {
  const f5 = (pick.foreign_5d ?? 0) / 1e8;
  const i5 = (pick.inst_5d ?? 0) / 1e8;
  const entry = pick.entry_info;

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <span
            className={`text-xs px-2 py-0.5 rounded font-bold ${GRADE_COLORS[pick.grade] || "bg-gray-700 text-gray-400"}`}
          >
            {pick.grade}
          </span>
          <span className="text-white font-medium">{pick.name}</span>
          <span className="text-gray-500 text-xs">{pick.ticker}</span>
        </div>
        <span className="text-blue-400 font-bold">{pick.total_score}점</span>
      </div>

      {/* 시그널 소스 태그 */}
      <div className="flex flex-wrap gap-1 mb-2">
        {(pick.sources ?? []).map((s) => (
          <span
            key={s}
            className="bg-blue-900/30 text-blue-400 text-xs px-2 py-0.5 rounded-full border border-blue-800/50"
          >
            {s}
          </span>
        ))}
      </div>

      {/* 수급 + 기술적 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        <div>
          <span className="text-gray-500">외인5일</span>{" "}
          <span className={f5 >= 0 ? "text-red-400" : "text-blue-400"}>
            {f5 >= 0 ? "+" : ""}{f5.toFixed(0)}억
          </span>
        </div>
        <div>
          <span className="text-gray-500">기관5일</span>{" "}
          <span className={i5 >= 0 ? "text-red-400" : "text-blue-400"}>
            {i5 >= 0 ? "+" : ""}{i5.toFixed(0)}억
          </span>
        </div>
        <div>
          <span className="text-gray-500">RSI</span>{" "}
          <span className="text-gray-300">{pick.rsi?.toFixed(0) ?? "-"}</span>
        </div>
        <div>
          <span className="text-gray-500">Stoch</span>{" "}
          <span className="text-gray-300">{pick.stoch_k?.toFixed(0) ?? "-"}</span>
        </div>
      </div>

      {/* 진입가/손절/목표가 */}
      {entry && entry.entry > 0 && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs bg-gray-800 rounded p-2">
          <div>
            <span className="text-gray-500">진입</span>{" "}
            <span className="text-white">{entry.entry.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-gray-500">손절</span>{" "}
            <span className="text-red-400">{entry.stop.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-gray-500">목표</span>{" "}
            <span className="text-green-400">{entry.target.toLocaleString()}</span>
          </div>
        </div>
      )}

      {/* 핵심 근거 */}
      {(pick.reasons?.length ?? 0) > 0 && (
        <p className="text-gray-500 text-xs mt-2">
          {pick.reasons.slice(0, 3).join(" · ")}
        </p>
      )}
    </div>
  );
}
