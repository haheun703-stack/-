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
  signals: string[];
  reason?: string;
}

/** Supabase에서 오는 다양한 news_pick 포맷을 통일 */
export function normalizeNewsPick(raw: RawNewsPick): NewsPick {
  // 포맷 1: {name: string, ticker: string, grade, score, signals}
  // 포맷 2: {title: string, ticker: string}
  // 포맷 3: {code: {name, ticker}, name: {name, ticker}, reason: string}
  let name = "";
  let ticker = "";

  if (typeof raw.name === "string") {
    name = raw.name;
  } else if (raw.name && typeof raw.name === "object" && raw.name.name) {
    name = raw.name.name;
  } else if (typeof raw.title === "string") {
    name = raw.title;
  }

  if (typeof raw.ticker === "string" && raw.ticker.length <= 10) {
    ticker = raw.ticker;
  } else if (raw.code && typeof raw.code === "object" && raw.code.ticker) {
    ticker = raw.code.ticker;
  }

  return {
    ticker: ticker || "",
    name: name || "N/A",
    grade: typeof raw.grade === "string" ? raw.grade : "",
    score: typeof raw.score === "number" ? raw.score : 0,
    signals: Array.isArray(raw.signals) ? raw.signals : [],
    reason: typeof raw.reason === "string" ? raw.reason : undefined,
  };
}

// FREE 티어 제한
export const FREE_PICKS_LIMIT = 2;
