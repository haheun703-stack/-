"use client";

import { useEffect, useState } from "react";
import type { MorningBriefing as BriefingType, NewsPick } from "@/lib/types";
import { FREE_PICKS_LIMIT, normalizeNewsPick } from "@/lib/types";
import PaywallBlur from "./PaywallBlur";

const STATUS_ICONS: Record<string, { icon: string; color: string; label: string }> = {
  BULL: { icon: "🟢", color: "text-green-400", label: "BULL" },
  BEAR: { icon: "🔴", color: "text-red-400", label: "BEAR" },
  CAUTION: { icon: "🟡", color: "text-yellow-400", label: "CAUTION" },
  NEUTRAL: { icon: "⚪", color: "text-gray-400", label: "NEUTRAL" },
};

export default function MorningBriefing({ isPaid = false }: { isPaid?: boolean }) {
  const [briefing, setBriefing] = useState<BriefingType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch("/api/briefing", { signal: controller.signal });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const data = await res.json();
        setBriefing(data);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setBriefing(null);
      }
      setLoading(false);
    }
    load();
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="animate-pulse space-y-3">
        <div className="h-6 bg-gray-800 rounded w-64" />
        <div className="h-20 bg-gray-800 rounded" />
        <div className="h-40 bg-gray-800 rounded" />
      </div>
    );
  }

  if (!briefing) {
    return (
      <div className="text-gray-500 text-center py-8">
        오늘의 브리핑이 아직 없습니다
      </div>
    );
  }

  const status = briefing.market_status || briefing.direction || "NEUTRAL";
  const statusInfo = STATUS_ICONS[status] || STATUS_ICONS.NEUTRAL;
  const rawPicks = briefing.news_picks || [];
  const picks = rawPicks.map(normalizeNewsPick);
  const freePicks = picks.slice(0, FREE_PICKS_LIMIT);
  const paidPicks = picks.slice(FREE_PICKS_LIMIT);

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white text-xl font-bold">
            {statusInfo.icon} 모닝 브리핑
          </h2>
          <p className="text-gray-500 text-sm">{briefing.date}</p>
        </div>
        <span
          className={`${statusInfo.color} text-lg font-bold bg-gray-900 px-4 py-2 rounded-lg border border-gray-800`}
        >
          {statusInfo.label}
        </span>
      </div>

      {/* 시장 요약 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {briefing.us_summary && (
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <p className="text-gray-500 text-xs mb-2">🌍 US 야간</p>
            <p className="text-gray-200 text-sm">{briefing.us_summary}</p>
          </div>
        )}
        {briefing.kr_summary && (
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <p className="text-gray-500 text-xs mb-2">🇰🇷 KR 전망</p>
            <p className="text-gray-200 text-sm">{briefing.kr_summary}</p>
          </div>
        )}
      </div>

      {/* 섹터 포커스 */}
      {briefing.sector_focus && briefing.sector_focus.length > 0 && (
        <div>
          <p className="text-gray-500 text-xs mb-2">🎯 섹터 포커스</p>
          <div className="flex flex-wrap gap-2">
            {briefing.sector_focus.map((sector) => (
              <span
                key={sector}
                className="bg-blue-900/30 text-blue-400 text-xs px-3 py-1 rounded-full border border-blue-800/50"
              >
                {sector}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* AI 추천 종목 — FREE */}
      {picks.length > 0 && (
        <div>
          <p className="text-gray-500 text-xs mb-3">
            💰 AI 추천 ({picks.length}종목)
          </p>

          {/* FREE 영역 */}
          <div className="space-y-2 mb-2">
            {freePicks.map((pick) => (
              <PickCard key={pick.ticker} pick={pick} />
            ))}
          </div>

          {/* PAID 영역 */}
          {paidPicks.length > 0 && (
            <PaywallBlur locked={!isPaid}>
              <div className="space-y-2">
                {paidPicks.map((pick) => (
                  <PickCard key={pick.ticker} pick={pick} />
                ))}
              </div>
            </PaywallBlur>
          )}
        </div>
      )}
    </div>
  );
}

function PickCard({ pick }: { pick: NewsPick }) {
  const gradeColors: Record<string, string> = {
    "적극매수": "bg-red-600",
    "매수": "bg-green-600",
    AA: "bg-red-600",
    A: "bg-green-600",
    B: "bg-blue-600",
  };

  return (
    <div className="flex items-center justify-between bg-gray-900 rounded-lg p-3 border border-gray-800">
      <div className="flex items-center gap-3">
        {pick.grade && (
          <span
            className={`${gradeColors[pick.grade] || "bg-gray-600"} text-white text-xs px-2 py-0.5 rounded font-bold`}
          >
            {pick.grade}
          </span>
        )}
        <div>
          <span className="text-white text-sm font-medium">{pick.name}</span>
          {pick.ticker && (
            <span className="text-gray-500 text-xs ml-2">{pick.ticker}</span>
          )}
        </div>
      </div>
      <div className="text-right">
        {pick.score > 0 && (
          <p className="text-blue-400 text-sm font-bold">{pick.score}점</p>
        )}
        {pick.reason && (
          <p className="text-gray-500 text-xs max-w-[200px] truncate">{pick.reason}</p>
        )}
        {pick.signals && (
          <p className="text-gray-600 text-xs">{pick.signals}</p>
        )}
      </div>
    </div>
  );
}
