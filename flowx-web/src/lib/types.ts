export interface Signal {
  id: string;
  bot_type: "QUANT" | "DAYTRADING";
  ticker: string;
  ticker_name: string;
  signal_type: "BUY" | "SELL";
  grade: "AA" | "A" | "B" | "C";
  score: number;
  entry_price: number;
  target_price: number | null;
  stop_price: number | null;
  current_price: number | null;
  return_pct: number;
  max_return_pct: number;
  status: "OPEN" | "CLOSED" | "STOPPED";
  signal_date: string;
  close_date: string | null;
  close_reason: string | null;
  multiplier: number;
  memo: string | null;
  created_at: string;
  updated_at: string;
}

export interface Scoreboard {
  id: string;
  bot_type: "QUANT" | "DAYTRADING" | "ALL";
  period: "30D" | "60D" | "90D" | "ALL";
  total_signals: number;
  win_count: number;
  lose_count: number;
  win_rate: number;
  avg_return_pct: number;
  avg_win_pct: number;
  avg_lose_pct: number;
  best_signal: SignalSummary | null;
  worst_signal: SignalSummary | null;
  calculated_at: string;
}

export interface SignalSummary {
  ticker: string;
  ticker_name: string;
  return_pct: number;
  signal_date: string;
  close_date?: string;
  close_reason?: string;
}

export interface MorningBriefing {
  id: string;
  date: string;
  market_status?: string;
  direction?: string;
  market_phase?: string;
  us_summary: string | null;
  kr_summary: string | null;
  news_picks: RawNewsPick[];
  sector_focus: string[];
  full_report: string | null;
  created_at: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type RawNewsPick = Record<string, any>;

export interface NewsPick {
  ticker: string;
  name: string;
  grade: string;
  score: number;
  signals: string;   // 백엔드 계약: 문자열 (", " join)
  reason?: string;
}

/**
 * Supabase에서 오는 다양한 news_pick 포맷을 통일.
 * Python 백엔드 flowx_data_contract.py의 NewsPick.from_any()와 동일 로직.
 *
 * 지원 포맷:
 *   1. {name: string, ticker, grade, score, signals}  — 정상
 *   2. {title: string, ticker}                         — 구버전
 *   3. {code: {name, ticker}, name: {name, ticker}}   — 크래시 원인
 *   4. "삼성전자"                                       — 문자열
 */
export function normalizeNewsPick(raw: RawNewsPick): NewsPick {
  if (typeof raw === "string") {
    return { ticker: "", name: raw, grade: "", score: 0, signals: "" };
  }
  if (!raw || typeof raw !== "object") {
    return { ticker: "", name: "N/A", grade: "", score: 0, signals: "" };
  }

  // ── name 추출 ──
  let name = "";
  if (typeof raw.name === "string") {
    name = raw.name;
  } else if (raw.name && typeof raw.name === "object" && raw.name.name) {
    name = raw.name.name;  // 중첩 객체 풀기
  }
  if (!name && typeof raw.title === "string") {
    name = raw.title;
  }

  // ── ticker 추출 ──
  let ticker = "";
  if (typeof raw.ticker === "string" && raw.ticker.length <= 10) {
    ticker = raw.ticker;
  } else if (raw.name && typeof raw.name === "object" && raw.name.ticker) {
    ticker = raw.name.ticker;
  } else if (raw.code && typeof raw.code === "object" && raw.code.ticker) {
    ticker = raw.code.ticker;
  }

  // ── signals: 배열이면 join, 문자열이면 그대로 ──
  let signals = "";
  if (Array.isArray(raw.signals)) {
    signals = raw.signals.filter(Boolean).join(", ");
  } else if (typeof raw.signals === "string") {
    signals = raw.signals;
  }

  return {
    ticker: ticker || "",
    name: name || "N/A",
    grade: typeof raw.grade === "string" ? raw.grade : "",
    score: typeof raw.score === "number" ? raw.score : 0,
    signals,
    reason: typeof raw.reason === "string" ? raw.reason : undefined,
  };
}

// FREE 티어 제한
export const FREE_PICKS_LIMIT = 2;
