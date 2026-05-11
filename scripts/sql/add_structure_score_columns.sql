-- ============================================================
-- Structure Score 컬럼 추가 — quant_sector_fire 테이블
-- Supabase Dashboard > SQL Editor 에서 실행
-- ============================================================

ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS s1_score smallint DEFAULT 0;
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS s1_ratio numeric(5,3);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS s2_score smallint DEFAULT 0;
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS s2_stoch_k numeric(5,1);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS s3_score smallint DEFAULT 0;
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS structure_score smallint DEFAULT 0;
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS structure_grade varchar(2);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS composite_score smallint DEFAULT 0;
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS composite_grade varchar(2);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS market_kospi_stoch_k numeric(5,1);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS market_vix numeric(5,1);
ALTER TABLE quant_sector_fire ADD COLUMN IF NOT EXISTS market_disparity numeric(5,1);
