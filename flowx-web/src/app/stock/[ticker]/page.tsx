"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

interface StockDetail {
  ticker: string;
  name: string;
  // quant picks
  pick?: {
    grade: string;
    score: number;
    signals: string[];
    reasons: string[];
    entry_price: number;
    stop_loss: number;
    target_price: number;
    n_sources: number;
  };
  // recent signals
  recentSignals: {
    signal_type: string;
    grade: string;
    score: number;
    entry_price: number;
    target_price: number | null;
    stop_price: number | null;
    status: string;
    signal_date: string;
    return_pct: number;
  }[];
  // briefing mentions
  briefingMentions: {
    date: string;
    grade: string;
    score: number;
    reason?: string;
  }[];
}

const GRADE_COLOR: Record<string, string> = {
  "\uc801\uadf9\ub9e4\uc218": "text-red-400",
  "\ub9e4\uc218": "text-orange-400",
  "\uad00\uc2ec\ub9e4\uc218": "text-yellow-400",
  "\uad00\ucc30": "text-blue-400",
  AA: "text-red-400",
  A: "text-orange-400",
  B: "text-yellow-400",
  C: "text-gray-400",
};

function GradeBadge({ grade }: { grade: string }) {
  if (!grade) return null;
  return (
    <span
      className={`text-xs font-bold px-2 py-0.5 rounded-lg bg-gray-800 ${GRADE_COLOR[grade] || "text-gray-400"}`}
    >
      {grade}
    </span>
  );
}

function formatPrice(n: number | undefined | null) {
  if (!n) return "-";
  return n.toLocaleString("ko-KR") + "\uc6d0";
}

export default function StockDetailPage() {
  const params = useParams();
  const ticker = params.ticker as string;
  const [data, setData] = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    fetch(`/api/stock?ticker=${encodeURIComponent(ticker)}`)
      .then((res) => res.json())
      .then((d) => {
        if (d.error) setError(d.error);
        else setData(d);
      })
      .catch(() => setError("\ub370\uc774\ud130\ub97c \ubd88\ub7ec\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4"))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-20 text-center">
        <p className="text-gray-400 text-lg mb-4">
          {error || "\uc885\ubaa9 \uc815\ubcf4\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4"}
        </p>
        <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm">
          \ub300\uc2dc\ubcf4\ub4dc\ub85c \ub3cc\uc544\uac00\uae30
        </Link>
      </div>
    );
  }

  const pick = data.pick;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/" className="text-gray-500 hover:text-gray-300">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">{data.name}</h1>
          <span className="text-gray-500 text-sm">{data.ticker}</span>
        </div>
        {pick && <GradeBadge grade={pick.grade} />}
        {pick && pick.score > 0 && (
          <span className="text-blue-400 font-mono text-lg">{pick.score}점</span>
        )}
      </div>

      {/* Quant Pick 상세 */}
      {pick && (
        <section className="bg-gray-900 rounded-xl p-5 space-y-4">
          <h2 className="text-white font-bold">퀀트 분석</h2>

          {/* 가격 정보 */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-gray-500 text-xs">진입가</p>
              <p className="text-gray-200 font-mono">{formatPrice(pick.entry_price)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">목표가</p>
              <p className="text-green-400 font-mono">{formatPrice(pick.target_price)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">손절가</p>
              <p className="text-red-400 font-mono">{formatPrice(pick.stop_loss)}</p>
            </div>
          </div>

          {/* 시그널 소스 */}
          {pick.signals.length > 0 && (
            <div>
              <p className="text-gray-500 text-xs mb-2">
                시그널 소스 ({pick.n_sources}개)
              </p>
              <div className="flex flex-wrap gap-2">
                {pick.signals.map((s, i) => (
                  <span
                    key={i}
                    className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded-lg"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* 분석 이유 */}
          {pick.reasons.length > 0 && (
            <div>
              <p className="text-gray-500 text-xs mb-2">분석 근거</p>
              <ul className="space-y-1">
                {pick.reasons.map((r, i) => (
                  <li key={i} className="text-gray-300 text-sm flex items-start gap-2">
                    <span className="text-blue-400 mt-0.5 shrink-0">{"•"}</span>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {/* 최근 시그널 */}
      {data.recentSignals.length > 0 && (
        <section className="bg-gray-900 rounded-xl p-5 space-y-3">
          <h2 className="text-white font-bold">최근 시그널</h2>
          <div className="space-y-2">
            {data.recentSignals.map((s, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-2"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded ${
                      s.signal_type === "BUY"
                        ? "bg-red-900/30 text-red-400"
                        : "bg-blue-900/30 text-blue-400"
                    }`}
                  >
                    {s.signal_type}
                  </span>
                  <GradeBadge grade={s.grade} />
                  <span className="text-gray-400 text-xs">{s.signal_date}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-gray-300 text-sm font-mono">
                    {formatPrice(s.entry_price)}
                  </span>
                  <span
                    className={`text-sm font-mono ${
                      s.return_pct >= 0 ? "text-red-400" : "text-blue-400"
                    }`}
                  >
                    {s.return_pct >= 0 ? "+" : ""}
                    {s.return_pct.toFixed(1)}%
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      s.status === "OPEN"
                        ? "bg-green-900/30 text-green-400"
                        : s.status === "STOPPED"
                          ? "bg-red-900/30 text-red-400"
                          : "bg-gray-800 text-gray-400"
                    }`}
                  >
                    {s.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 브리핑 언급 이력 */}
      {data.briefingMentions.length > 0 && (
        <section className="bg-gray-900 rounded-xl p-5 space-y-3">
          <h2 className="text-white font-bold">브리핑 언급 이력</h2>
          <div className="space-y-2">
            {data.briefingMentions.map((m, i) => (
              <div key={i} className="flex items-center justify-between py-2">
                <div className="flex items-center gap-3">
                  <span className="text-gray-400 text-xs">{m.date}</span>
                  <GradeBadge grade={m.grade} />
                  {m.score > 0 && (
                    <span className="text-blue-400 text-xs font-mono">
                      {m.score}점
                    </span>
                  )}
                </div>
                {m.reason && (
                  <p className="text-gray-500 text-xs truncate max-w-xs">
                    {m.reason}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 데이터 없음 */}
      {!pick && data.recentSignals.length === 0 && data.briefingMentions.length === 0 && (
        <div className="bg-gray-900 rounded-xl p-8 text-center">
          <p className="text-gray-500">현재 FLOWX에서 추적 중인 데이터가 없습니다</p>
        </div>
      )}
    </div>
  );
}
