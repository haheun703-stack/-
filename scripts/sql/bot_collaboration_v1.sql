-- ============================================================
-- Bot Collaboration v1 — 봇 간 양방향 통신 테이블 (2026-05-18)
--
-- 근거: P2 §12-7-3 형(맏이) 리딩 시스템
-- 진행: 5/18 10:04 단타봇 동생과 5게이트 합의 후 즉시 구현
--
-- 테이블 2종:
--   1. quant_bot_advisory   (형 → 동생)
--   2. scalper_bot_feedback (동생 → 형)
--
-- 양방향 흐름 (매일):
--   09:00       동생: RECOMMENDATION INSERT (5게이트 결과 + 추천 등급)
--   09:05~14:00 동생: ENTRY INSERT (매수 실행 결과 + 등급)
--   09:30/11:00/13:30/15:00  형: ADVISORY INSERT (시장 진단 + 매크로)
--   15:25       동생: CLOSE INSERT (청산 PnL)
--   15:40       동생: JOURNAL INSERT (source별 적중률)
--   16:30       형: BAT-D에서 feedback SELECT + 다음날 advisory 정밀화
--
-- 멱등성: IF NOT EXISTS 사용 (재실행 안전)
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- 1. quant_bot_advisory (형 퀀트봇 → 동생 단타봇)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quant_bot_advisory (
    id                    BIGSERIAL PRIMARY KEY,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    advisory_date         DATE NOT NULL,
    advisory_time         TIME,                       -- 09:30 / 11:00 / 13:30 / 15:00

    -- 메시지 분류 (P2 §12-7-3)
    msg_type              VARCHAR(20) NOT NULL,       -- 'PRAISE' | 'ADVICE' | 'CRITICISM' | 'LEADING' | 'SNAPSHOT'
    severity              VARCHAR(10) DEFAULT 'INFO', -- 'INFO' | 'WARN' | 'CRITICAL'
    target_bot            VARCHAR(20) DEFAULT 'scalper',

    -- 시장 컨텍스트 (5/18 09:30 추적 결과 반영)
    market_regime         VARCHAR(20),                -- 'STRONG_BULL' | 'MILD_BULL' | 'NEUTRAL' | 'CAUTION' | 'BEAR' | 'CRISIS'
    market_strength_avg   DECIMAL(6,2),               -- 25 구독 종목 평균 강도 (예: 86.74)
    inverse_etf_strength  DECIMAL(6,2),               -- 252670 KODEX 인버스2X 강도 (예: 159.94 = 약세 베팅)
    inverse_etf_buy_ratio DECIMAL(5,2),               -- 인버스 매수비율 (예: 62.3)
    kospi_chg_pct         DECIMAL(5,2),               -- KOSPI200 ETF 변동률 (예: -2.21)
    risk_level            VARCHAR(10),                -- 'LOW' | 'MED' | 'HIGH' | 'CRISIS'

    -- 메시지 본문
    title                 TEXT NOT NULL,
    body                  TEXT NOT NULL,
    related_tickers       TEXT[],                     -- 관련 종목 (예: ['403870','252670'])
    alert_codes           TEXT[],                     -- 자비스 알림 코드 (예: ['EYE-07','VWAP-DIP'])
    reasoning             JSONB,                      -- 근거 데이터 (가격/강도/프로그램 매수 등)

    -- 동생 응답 (양방향)
    acknowledged_at       TIMESTAMPTZ,
    scalper_response      TEXT,
    scalper_action_taken  VARCHAR(30),                -- 'ENTRY' | 'CLOSE' | 'WATCH' | 'IGNORED'

    -- 사후 결과 (16:30 BAT-D 평가)
    outcome_evaluated_at  TIMESTAMPTZ,
    outcome_label         VARCHAR(20),                -- 'CORRECT' | 'PARTIAL' | 'WRONG' | 'UNTESTED'
    outcome_pnl_pct       DECIMAL(5,2),               -- 동생 매매 결과 (-100 ~ +100%)
    outcome_notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_advisory_date ON quant_bot_advisory(advisory_date DESC);
CREATE INDEX IF NOT EXISTS idx_advisory_type ON quant_bot_advisory(msg_type, target_bot);
CREATE INDEX IF NOT EXISTS idx_advisory_regime ON quant_bot_advisory(market_regime, advisory_date DESC);

COMMENT ON TABLE quant_bot_advisory IS '퀀트봇 형 → 단타봇 동생 advisory (5/18 합의)';

-- ────────────────────────────────────────────────────────────
-- 2. scalper_bot_feedback (동생 단타봇 → 형 퀀트봇)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scalper_bot_feedback (
    id                    BIGSERIAL PRIMARY KEY,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    feedback_date         DATE NOT NULL,
    feedback_time         TIME,

    -- 시점 분류 (동생 약속 4종 + 주간)
    msg_type              VARCHAR(20) NOT NULL,       -- 'RECOMMENDATION' | 'ENTRY' | 'CLOSE' | 'JOURNAL' | 'WEEKLY'

    -- RECOMMENDATION (09:00 morning_recommendation)
    rec_grade             VARCHAR(10),                -- 'STRONG' (5/5) | 'MEDIUM' (4/5) | 'WATCH' (3/5) | 'SKIP' (0~2)
    rec_tickers           TEXT[],                     -- 추천 종목 리스트
    gates_passed_n        INT,                        -- 0~5
    gates_detail          JSONB,                      -- {"vwap_dip":true,"program_price":true,...}

    -- ENTRY (09:05~14:00)
    entry_ticker          VARCHAR(10),
    entry_name            VARCHAR(40),
    entry_price           INT,
    entry_qty             INT,
    entry_amount          INT,
    entry_grade           VARCHAR(10),
    entry_reason          TEXT,
    entry_source          VARCHAR(30),                -- 'jarvis_advisory' | 'self_signal' | 'info_bot' | 'manual'
    entry_advisory_ref    BIGINT REFERENCES quant_bot_advisory(id),

    -- CLOSE (15:25)
    close_ticker          VARCHAR(10),
    close_price           INT,
    close_pnl_pct         DECIMAL(5,2),
    close_pnl_amount      INT,
    close_reason          VARCHAR(20),                -- 'TAKE_PROFIT' | 'STOP_LOSS' | 'TIME_EXIT' | 'MANUAL'
    holding_minutes       INT,

    -- JOURNAL (15:40, source별 적중률)
    journal_total_trades  INT,
    journal_wins          INT,
    journal_losses        INT,
    journal_accuracy_pct  DECIMAL(5,2),
    journal_by_source     JSONB,                      -- {"jarvis_advisory":{"wins":3,"trades":4,"acc":75.0},...}
    journal_avg_pnl_pct   DECIMAL(5,2),

    -- WEEKLY (금요일 저녁)
    weekly_summary        TEXT,
    weekly_top_pattern    TEXT,                       -- "프로그램+가격+VWAP 일치 5게이트 PASS 적중률 72%"
    weekly_worst_pattern  TEXT,                       -- "프로그램 대량매도 -100K+ 시 회피 미작동 케이스 2건"

    notes                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_date ON scalper_bot_feedback(feedback_date DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON scalper_bot_feedback(msg_type);
CREATE INDEX IF NOT EXISTS idx_feedback_advisory_ref ON scalper_bot_feedback(entry_advisory_ref);

COMMENT ON TABLE scalper_bot_feedback IS '단타봇 동생 → 퀀트봇 형 feedback (5/18 합의 호혜 데이터 4종)';

-- ────────────────────────────────────────────────────────────
-- 초기 시드: 5/18 합의 advisory 1건 (메타)
-- ────────────────────────────────────────────────────────────
INSERT INTO quant_bot_advisory (
    advisory_date, advisory_time, msg_type, severity, target_bot,
    market_regime, market_strength_avg, inverse_etf_strength, inverse_etf_buy_ratio, kospi_chg_pct, risk_level,
    title, body, related_tickers, alert_codes, reasoning
) VALUES (
    '2026-05-18', '10:04', 'LEADING', 'INFO', 'scalper',
    'CAUTION', 86.74, 117.87, 62.3, -2.21, 'MED',
    '[형→동생 첫 advisory] 5게이트 합의 + 데이터 채널 가동',
    '5/18 09:00~10:04 자비스 VWAP 눌림 시그널 9분 50% 적중. HPSP 황금 표준 패턴 (프로그램+ + 가격↑ + VWAP 눌림) 검증. 동생 v3 5게이트 100% 채택. 5/19부터 매일 09:30/11:00/13:30/15:00 advisory INSERT 자동 가동. 동생도 호혜 데이터 4종 (RECOMMENDATION/ENTRY/CLOSE/JOURNAL) 5/27 본격 출격 동기화.',
    ARRAY['403870','252670','114800','064400','012030','095340'],
    ARRAY['EYE-07','VWAP-DIP','SNAPSHOT'],
    '{"hpsp_pattern":"program+price+vwap_dip ALL match","db_anti_pattern":"program_buy_but_price_down","lg_cns_avoid":"program_-167K_mass_sell","inverse_signal":"252670 strength 117 → 시장 약세 베팅"}'::jsonb
);

-- ────────────────────────────────────────────────────────────
-- 검증 쿼리 (적용 후 자동 실행)
-- ────────────────────────────────────────────────────────────
-- SELECT table_name FROM information_schema.tables
--  WHERE table_schema='public' AND table_name IN ('quant_bot_advisory','scalper_bot_feedback');
-- SELECT id, advisory_date, msg_type, title FROM quant_bot_advisory ORDER BY id DESC LIMIT 1;
