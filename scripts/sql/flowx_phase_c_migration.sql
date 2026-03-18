-- ============================================================
-- FLOWX Phase C Migration — morning_briefings + signals + scoreboard
-- Supabase Dashboard > SQL Editor 에서 실행
-- v2: 기술 스펙 반영 (CHECK 제약, 트리거, 확장 컬럼)
-- ============================================================

-- ── 1. 모닝 브리핑 테이블 ──────────────────────────
CREATE TABLE IF NOT EXISTS morning_briefings (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    date date NOT NULL UNIQUE,
    market_status text NOT NULL DEFAULT 'NEUTRAL',
    us_summary text,
    kr_summary text,
    news_picks jsonb DEFAULT '[]'::jsonb,
    sector_focus jsonb DEFAULT '[]'::jsonb,
    full_report text,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_briefings_date ON morning_briefings(date DESC);

-- ── 2. 시그널 테이블 (퀀트 + 단타 공용) ──────────────
CREATE TABLE IF NOT EXISTS signals (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_type text NOT NULL CHECK (bot_type IN ('QUANT','DAYTRADING')),
    ticker text NOT NULL,
    ticker_name text NOT NULL,
    signal_type text NOT NULL CHECK (signal_type IN ('BUY','SELL')),
    grade text CHECK (grade IN ('AA','A','B','C')),
    score integer CHECK (score BETWEEN 0 AND 100),
    entry_price integer NOT NULL,
    target_price integer,
    stop_price integer,
    current_price integer,
    return_pct numeric(6,2) DEFAULT 0,
    max_return_pct numeric(6,2) DEFAULT 0,
    status text NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN','CLOSED','STOPPED')),
    signal_date date NOT NULL DEFAULT CURRENT_DATE,
    close_date date,
    close_reason text,
    multiplier numeric(3,1) DEFAULT 1.0,
    memo text,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_bot_type ON signals(bot_type);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);

-- updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER signals_updated_at
    BEFORE UPDATE ON signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── 3. 성적표 집계 테이블 ──────────────────────────
CREATE TABLE IF NOT EXISTS scoreboard (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_type text NOT NULL CHECK (bot_type IN ('QUANT','DAYTRADING','ALL')),
    period text NOT NULL CHECK (period IN ('30D','60D','90D','ALL')),
    total_signals integer DEFAULT 0,
    win_count integer DEFAULT 0,
    lose_count integer DEFAULT 0,
    win_rate numeric(5,2) DEFAULT 0,
    avg_return_pct numeric(6,2) DEFAULT 0,
    avg_win_pct numeric(6,2) DEFAULT 0,
    avg_lose_pct numeric(6,2) DEFAULT 0,
    best_signal jsonb,
    worst_signal jsonb,
    calculated_at timestamptz DEFAULT now(),
    UNIQUE(bot_type, period)
);

-- ── 4. RLS 정책 ──────────────────────────────────
ALTER TABLE morning_briefings ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE scoreboard ENABLE ROW LEVEL SECURITY;

-- morning_briefings
CREATE POLICY "anon_read_briefings" ON morning_briefings FOR SELECT USING (true);
CREATE POLICY "anon_insert_briefings" ON morning_briefings FOR INSERT WITH CHECK (true);
CREATE POLICY "anon_update_briefings" ON morning_briefings FOR UPDATE USING (true);

-- signals
CREATE POLICY "anon_read_signals" ON signals FOR SELECT USING (true);
CREATE POLICY "anon_insert_signals" ON signals FOR INSERT WITH CHECK (true);
CREATE POLICY "anon_update_signals" ON signals FOR UPDATE USING (true);

-- scoreboard
CREATE POLICY "anon_read_scoreboard" ON scoreboard FOR SELECT USING (true);
CREATE POLICY "anon_insert_scoreboard" ON scoreboard FOR INSERT WITH CHECK (true);
CREATE POLICY "anon_update_scoreboard" ON scoreboard FOR UPDATE USING (true);

-- ── 5. 스키마 캐시 리로드 ────────────────────────
NOTIFY pgrst, 'reload schema';
