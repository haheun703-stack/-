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
  market_status: string;
  us_summary: string | null;
  kr_summary: string | null;
  news_picks: NewsPick[];
  sector_focus: string[];
  full_report: string | null;
  created_at: string;
}

export interface NewsPick {
  ticker: string;
  name: string;
  grade: string;
  score: number;
  signals: string[];
}

// FREE 티어 제한
export const FREE_PICKS_LIMIT = 2;
