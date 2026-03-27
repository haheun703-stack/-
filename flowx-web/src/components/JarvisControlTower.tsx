"use client";

import { useEffect, useState } from "react";

/* ─── 타입 ─── */

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
  entry_price?: number;
  stop_loss?: number;
  target_price?: number;
  entry_info?: { entry: number; stop: number; target: number };
}

interface SectorItem {
  rank: number;
  sector: string;
  etf_code: string;
  score: number;
  ret_5: number;
  ret_20: number;
  rsi: number;
  acceleration: boolean;
  rank_change: number;
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
  market_guide?: {
    summary?: string;
    strategy?: string;
    hot_sectors?: { sector: string; ret_5: number }[];
    cold_sectors?: { sector: string; ret_5: number }[];
    vix?: number;
    cash_ratio?: number;
  } | null;
  sectors?: {
    date?: string;
    top?: SectorItem[];
  } | null;
  etf_picks?: {
    regime?: string;
    allocation?: Record<string, number>;
    accelerations?: {
      sector: string;
      rank_change: number;
      score: number;
      ret_5d: number;
    }[];
  } | null;
  updated_at?: string | null;
  date?: string | null;
}

/* ─── 상수 ─── */

const GRADE_COLORS: Record<string, string> = {
  "적극매수": "bg-red-600 text-white",
  "매수": "bg-green-600 text-white",
  "관심매수": "bg-blue-600 text-white",
  "관찰": "bg-yellow-600 text-white",
  "보류": "bg-gray-700 text-gray-400",
};

const REGIME_DISPLAY: Record<string, { icon: string; color: string; bg: string }> = {
  BULL: { icon: "🟢", color: "text-green-400", bg: "border-green-800/50" },
  BEAR: { icon: "🔴", color: "text-red-400", bg: "border-red-800/50" },
  CAUTION: { icon: "🟡", color: "text-yellow-400", bg: "border-yellow-800/50" },
  NEUTRAL: { icon: "⚪", color: "text-gray-400", bg: "border-gray-800" },
};

const SHIELD_DISPLAY: Record<string, { icon: string; color: string }> = {
  GREEN: { icon: "🟢", color: "text-green-400" },
  YELLOW: { icon: "🟡", color: "text-yellow-400" },
  RED: { icon: "🔴", color: "text-red-400" },
};

const TAB_ITEMS = [
  { key: "recommend", label: "오늘의 추천", icon: "🎯" },
  { key: "sectors", label: "섹터 분석", icon: "📊", soon: true },
  { key: "signals", label: "시그널", icon: "📡", soon: true },
  { key: "performance", label: "성과", icon: "📈", soon: true },
];

/* ─── 메인 컴포넌트 ─── */

export default function JarvisControlTower() {
  const [data, setData] = useState<JarvisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("recommend");

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
        <div className="h-20 bg-gray-800 rounded-lg" />
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

  const { picks, accuracy, brain, shield, market_guide, sectors, etf_picks } = data;
  const regime = brain?.regime || brain?.direction || "NEUTRAL";
  const regimeInfo = REGIME_DISPLAY[regime] || REGIME_DISPLAY.NEUTRAL;
  const shieldStatus = shield?.status || "YELLOW";
  const shieldInfo = SHIELD_DISPLAY[shieldStatus] || SHIELD_DISPLAY.YELLOW;

  const allPicks = picks?.picks ?? [];
  const buyable = allPicks
    .filter((p) => ["적극매수", "매수", "관심매수", "관찰"].includes(p.grade))
    .sort((a, b) => b.total_score - a.total_score);

  const stats = picks?.stats ?? {};

  const sourceCounts: Record<string, number> = {};
  for (const p of allPicks) {
    for (const s of p.sources ?? []) {
      sourceCounts[s] = (sourceCounts[s] || 0) + 1;
    }
  }

  return (
    <div className="space-y-6">
      {/* ★ 시장 맥락 배너 */}
      {market_guide && <MarketGuideBanner guide={market_guide} regime={regime} regimeInfo={regimeInfo} />}

      {/* 상태판 */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatusCard label="레짐" value={regime} icon={regimeInfo.icon} color={regimeInfo.color}
          sub={brain?.vix ? `VIX ${brain.vix}` : undefined} />
        <StatusCard label="SHIELD" value={shieldStatus} icon={shieldInfo.icon} color={shieldInfo.color}
          sub={shield?.max_drawdown ? `MDD ${shield.max_drawdown.toFixed(1)}%` : undefined} />
        <StatusCard label="현금비중" value={brain?.cash_ratio ? `${brain.cash_ratio}%` : "-"}
          icon="💰" color="text-blue-400" />
        <StatusCard label="대상일" value={picks?.target_date_label ?? "-"}
          icon="📅" color="text-gray-300" sub={picks?.mode_label} />
      </section>

      {/* 탭 네비게이션 */}
      <nav className="flex gap-1 bg-gray-900 rounded-lg p-1 border border-gray-800">
        {TAB_ITEMS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => !tab.soon && setActiveTab(tab.key)}
            className={`flex-1 py-2 px-3 rounded-md text-xs font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-blue-600 text-white"
                : tab.soon
                  ? "text-gray-600 cursor-not-allowed"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            {tab.icon} {tab.label}
            {tab.soon && <span className="ml-1 text-[10px] opacity-50">soon</span>}
          </button>
        ))}
      </nav>

      {/* 탭 콘텐츠 */}
      {activeTab === "recommend" && (
        <div className="space-y-6">
          {/* ETF 배분 + 가속 섹터 */}
          {etf_picks && <ETFSection etf={etf_picks} />}

          {/* 추천 종목 */}
          <section>
            <h2 className="text-white text-lg font-bold mb-3">
              {buyable.length > 0
                ? `개별종목 추천 (${buyable.length}건)`
                : "전체 관망 — 조건 강화 대기"}
            </h2>
            {buyable.length > 0 ? (
              <div className="space-y-3">
                {buyable.map((p) => <PickRow key={p.ticker} pick={p} />)}
              </div>
            ) : (
              <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 text-center">
                <p className="text-gray-400 text-sm">현재 교차검증 통과 종목이 없습니다</p>
                <p className="text-gray-600 text-xs mt-1">
                  2개 이상 시그널 + 거래량 2배 이상 조건 충족 시 표시됩니다
                </p>
              </div>
            )}
          </section>

          {/* 등급 분포 + 시그널 소스 */}
          <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
              <h3 className="text-gray-400 text-xs mb-3">등급 분포 ({picks?.total_candidates ?? 0}종목)</h3>
              <div className="space-y-2">
                {Object.entries(stats).map(([grade, count]) => (
                  <div key={grade} className="flex items-center justify-between">
                    <span className={`text-xs px-2 py-0.5 rounded ${GRADE_COLORS[grade] || "bg-gray-700 text-gray-400"}`}>
                      {grade}
                    </span>
                    <div className="flex-1 mx-3 bg-gray-800 rounded-full h-2">
                      <div className="bg-blue-600 rounded-full h-2"
                        style={{ width: `${Math.min(((count as number) / (picks?.total_candidates ?? 1)) * 100, 100)}%` }} />
                    </div>
                    <span className="text-gray-400 text-xs w-12 text-right">{count as number}건</span>
                  </div>
                ))}
              </div>
            </div>

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

          {/* 시그널 정확도 */}
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
        </div>
      )}
    </div>
  );
}

/* ─── 서브 컴포넌트 ─── */

function MarketGuideBanner({
  guide,
  regime,
  regimeInfo,
}: {
  guide: NonNullable<JarvisData["market_guide"]>;
  regime: string;
  regimeInfo: { icon: string; color: string; bg: string };
}) {
  const hot = guide.hot_sectors ?? [];
  const cold = guide.cold_sectors ?? [];

  return (
    <div className={`bg-gray-900 rounded-lg p-5 border ${regimeInfo.bg}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className={`${regimeInfo.color} text-sm font-bold`}>
            {regimeInfo.icon} {guide.summary}
          </p>
          {guide.strategy && (
            <p className="text-gray-400 text-xs mt-1">{guide.strategy}</p>
          )}
        </div>
        {guide.vix != null && guide.vix > 0 && (
          <span className={`text-xs px-2 py-1 rounded ${
            guide.vix >= 30 ? "bg-red-900/50 text-red-400" :
            guide.vix >= 20 ? "bg-yellow-900/50 text-yellow-400" :
            "bg-green-900/50 text-green-400"
          }`}>
            VIX {guide.vix}
          </span>
        )}
      </div>

      {(hot.length > 0 || cold.length > 0) && (
        <div className="flex flex-wrap gap-2 text-xs">
          {hot.map((s) => (
            <span key={s.sector} className="bg-red-900/30 text-red-400 px-2 py-0.5 rounded-full border border-red-800/30">
              🔥 {s.sector} {s.ret_5 > 0 ? `+${s.ret_5.toFixed(1)}%` : `${s.ret_5.toFixed(1)}%`}
            </span>
          ))}
          {cold.map((s) => (
            <span key={s.sector} className="bg-blue-900/20 text-blue-300/60 px-2 py-0.5 rounded-full border border-blue-800/20">
              🧊 {s.sector} {s.ret_5.toFixed(1)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ETFSection({ etf }: { etf: NonNullable<JarvisData["etf_picks"]> }) {
  const alloc = etf.allocation ?? {};
  const accel = etf.accelerations ?? [];

  // 0이 아닌 배분만 표시
  const activeAlloc = Object.entries(alloc).filter(([, v]) => v > 0);

  const ALLOC_LABELS: Record<string, string> = {
    sector: "섹터", leverage: "레버리지", index: "지수",
    gold: "금", small_cap: "소형주", bonds: "채권",
    dollar: "달러", cash: "현금",
  };

  const ALLOC_COLORS: Record<string, string> = {
    gold: "bg-yellow-600", bonds: "bg-blue-600", dollar: "bg-green-600",
    cash: "bg-gray-600", sector: "bg-purple-600", leverage: "bg-red-600",
    index: "bg-cyan-600", small_cap: "bg-orange-600",
  };

  return (
    <section className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h3 className="text-white text-sm font-bold mb-3">ETF 배분 전략</h3>

      {/* 배분 바 */}
      {activeAlloc.length > 0 && (
        <div className="mb-3">
          <div className="flex rounded-full h-3 overflow-hidden">
            {activeAlloc.map(([key, val]) => (
              <div
                key={key}
                className={`${ALLOC_COLORS[key] || "bg-gray-500"}`}
                style={{ width: `${val}%` }}
                title={`${ALLOC_LABELS[key] || key}: ${val}%`}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-3 mt-2 text-xs">
            {activeAlloc.map(([key, val]) => (
              <span key={key} className="text-gray-400">
                <span className={`inline-block w-2 h-2 rounded-full mr-1 ${ALLOC_COLORS[key] || "bg-gray-500"}`} />
                {ALLOC_LABELS[key] || key} {val}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 가속 섹터 */}
      {accel.length > 0 && (
        <div>
          <p className="text-gray-500 text-xs mb-2">가속 섹터 (순위 급상승)</p>
          <div className="flex flex-wrap gap-2">
            {accel.map((a) => (
              <span key={a.sector} className="bg-orange-900/30 text-orange-400 text-xs px-2 py-1 rounded border border-orange-800/30">
                🚀 {a.sector}
                <span className="text-orange-300 ml-1">+{a.rank_change}↑</span>
                <span className="text-gray-500 ml-1">({a.ret_5d > 0 ? "+" : ""}{a.ret_5d.toFixed(1)}%)</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function StatusCard({ label, value, icon, color, sub }: {
  label: string; value: string; icon: string; color: string; sub?: string;
}) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <p className="text-gray-500 text-xs">{label}</p>
      <p className={`${color} text-lg font-bold mt-1`}>{icon} {value}</p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

function PickRow({ pick }: { pick: PickItem }) {
  const f5 = (pick.foreign_5d ?? 0) / 1e8;
  const i5 = (pick.inst_5d ?? 0) / 1e8;
  const entry = pick.entry_info ?? (
    pick.entry_price ? { entry: pick.entry_price, stop: pick.stop_loss ?? 0, target: pick.target_price ?? 0 } : null
  );

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-0.5 rounded font-bold ${GRADE_COLORS[pick.grade] || "bg-gray-700 text-gray-400"}`}>
            {pick.grade}
          </span>
          <span className="text-white font-medium">{pick.name}</span>
          <span className="text-gray-500 text-xs">{pick.ticker}</span>
        </div>
        <span className="text-blue-400 font-bold">{pick.total_score}점</span>
      </div>

      <div className="flex flex-wrap gap-1 mb-2">
        {(pick.sources ?? []).map((s) => (
          <span key={s} className="bg-blue-900/30 text-blue-400 text-xs px-2 py-0.5 rounded-full border border-blue-800/50">
            {s}
          </span>
        ))}
      </div>

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

      {(pick.reasons?.length ?? 0) > 0 && (
        <p className="text-gray-500 text-xs mt-2">
          {pick.reasons.slice(0, 3).join(" · ")}
        </p>
      )}
    </div>
  );
}
