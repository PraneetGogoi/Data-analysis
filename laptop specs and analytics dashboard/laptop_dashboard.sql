-- =============================================================
--  laptop_dashboard.sql
--  PostgreSQL schema, indexes, analytical views & seed data
--  for the Laptop Scrap Analytics Web Dashboard
-- =============================================================

-- ----------------------------------------------------------------
-- 0. DATABASE SETUP
-- ----------------------------------------------------------------
-- Run as superuser:
--   CREATE DATABASE laptops_db;
--   \c laptops_db

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ----------------------------------------------------------------
-- 1. MAIN LAPTOPS TABLE
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS laptops CASCADE;

CREATE TABLE laptops (
    id                 SERIAL PRIMARY KEY,
    company            VARCHAR(50)     NOT NULL,
    type_name          TEXT,
    inches             NUMERIC(4,1),
    screen_resolution  VARCHAR(30),
    cpu                TEXT,
    gpu                TEXT,
    op_sys             VARCHAR(30),
    touch_screen       SMALLINT        CHECK (touch_screen IN (0,1)),
    ips                SMALLINT        CHECK (ips IN (0,1)),
    x_res              INTEGER,
    y_res              INTEGER,
    ppi                NUMERIC(6,2),
    dedicated_gpu      SMALLINT        CHECK (dedicated_gpu IN (0,1)),
    ram_gb             INTEGER,
    weight_kg          NUMERIC(5,2),
    ssd                INTEGER,          -- SSD storage GB
    hhd                INTEGER,          -- HDD storage GB
    storage_type       VARCHAR(30),
    total_storage_gb   INTEGER,
    storage_category   VARCHAR(30),
    price              NUMERIC(10,2)   NOT NULL,
    cpu_brand          VARCHAR(20),
    gpu_brand          VARCHAR(20),
    price_tier         VARCHAR(20),
    screen_size_cat    VARCHAR(25),
    inserted_at        TIMESTAMPTZ     DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- 2. INDEXES
-- ----------------------------------------------------------------
CREATE INDEX idx_laptops_company       ON laptops (company);
CREATE INDEX idx_laptops_price         ON laptops (price);
CREATE INDEX idx_laptops_ram           ON laptops (ram_gb);
CREATE INDEX idx_laptops_cpu_brand     ON laptops (cpu_brand);
CREATE INDEX idx_laptops_gpu_brand     ON laptops (gpu_brand);
CREATE INDEX idx_laptops_price_tier    ON laptops (price_tier);
CREATE INDEX idx_laptops_dedicated_gpu ON laptops (dedicated_gpu);
CREATE INDEX idx_laptops_op_sys        ON laptops (op_sys);
CREATE INDEX idx_laptops_storage_cat   ON laptops (storage_category);

-- ----------------------------------------------------------------
-- 3. ANALYTICAL VIEWS (for dashboard widgets)
-- ----------------------------------------------------------------

-- 3.1  KPI Summary (single-row for header cards)
CREATE OR REPLACE VIEW vw_kpi_summary AS
SELECT
    COUNT(*)                                    AS total_laptops,
    ROUND(AVG(price)::NUMERIC, 2)               AS avg_price,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
          (ORDER BY price)::NUMERIC, 2)         AS median_price,
    ROUND(MIN(price)::NUMERIC, 2)               AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)               AS max_price,
    ROUND(STDDEV(price)::NUMERIC, 2)            AS stddev_price,
    ROUND(AVG(ram_gb)::NUMERIC, 1)              AS avg_ram_gb,
    ROUND(AVG(weight_kg)::NUMERIC, 2)           AS avg_weight_kg,
    ROUND(100.0 * SUM(dedicated_gpu) / COUNT(*), 1) AS pct_dedicated_gpu,
    ROUND(100.0 * SUM(touch_screen) / COUNT(*), 1)  AS pct_touchscreen,
    ROUND(100.0 * SUM(ips)          / COUNT(*), 1)  AS pct_ips,
    COUNT(DISTINCT company)                     AS brand_count
FROM laptops;

-- 3.2  Brand breakdown
CREATE OR REPLACE VIEW vw_brand_stats AS
SELECT
    company,
    COUNT(*)                        AS laptop_count,
    ROUND(AVG(price)::NUMERIC, 2)   AS avg_price,
    ROUND(MIN(price)::NUMERIC, 2)   AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)   AS max_price,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
          (ORDER BY price)::NUMERIC, 2) AS median_price,
    ROUND(AVG(ram_gb)::NUMERIC, 1)  AS avg_ram_gb,
    ROUND(AVG(weight_kg)::NUMERIC, 2) AS avg_weight_kg,
    SUM(dedicated_gpu)              AS count_dedicated_gpu
FROM laptops
GROUP BY company
ORDER BY laptop_count DESC;

-- 3.3  Price tier distribution
CREATE OR REPLACE VIEW vw_price_tiers AS
SELECT
    price_tier,
    COUNT(*)                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price,
    ROUND(MIN(price)::NUMERIC, 2)          AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)          AS max_price
FROM laptops
WHERE price_tier IS NOT NULL
GROUP BY price_tier
ORDER BY avg_price;

-- 3.4  CPU Brand market share
CREATE OR REPLACE VIEW vw_cpu_brand_stats AS
SELECT
    cpu_brand,
    COUNT(*)                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS market_share_pct,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price,
    ROUND(AVG(ram_gb)::NUMERIC, 1)         AS avg_ram_gb
FROM laptops
GROUP BY cpu_brand
ORDER BY count DESC;

-- 3.5  GPU Brand stats
CREATE OR REPLACE VIEW vw_gpu_brand_stats AS
SELECT
    gpu_brand,
    COUNT(*)                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price,
    ROUND(MIN(price)::NUMERIC, 2)          AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)          AS max_price
FROM laptops
GROUP BY gpu_brand
ORDER BY count DESC;

-- 3.6  OS distribution
CREATE OR REPLACE VIEW vw_os_distribution AS
SELECT
    op_sys,
    COUNT(*)                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price
FROM laptops
GROUP BY op_sys
ORDER BY count DESC;

-- 3.7  RAM distribution
CREATE OR REPLACE VIEW vw_ram_distribution AS
SELECT
    ram_gb,
    COUNT(*)                               AS count,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price,
    ROUND(MIN(price)::NUMERIC, 2)          AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)          AS max_price
FROM laptops
GROUP BY ram_gb
ORDER BY ram_gb;

-- 3.8  Storage category breakdown
CREATE OR REPLACE VIEW vw_storage_categories AS
SELECT
    storage_category,
    COUNT(*)                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price
FROM laptops
GROUP BY storage_category
ORDER BY avg_price;

-- 3.9  Screen size category
CREATE OR REPLACE VIEW vw_screen_size_stats AS
SELECT
    screen_size_cat,
    COUNT(*)                               AS count,
    ROUND(AVG(price)::NUMERIC, 2)          AS avg_price,
    ROUND(AVG(ppi)::NUMERIC, 1)            AS avg_ppi,
    ROUND(AVG(weight_kg)::NUMERIC, 2)      AS avg_weight_kg
FROM laptops
WHERE screen_size_cat IS NOT NULL
GROUP BY screen_size_cat
ORDER BY avg_price;

-- 3.10 Price histogram buckets (for bar chart)
CREATE OR REPLACE VIEW vw_price_histogram AS
SELECT
    width_bucket(price, 300, 7500, 50) AS bucket,
    300 + (width_bucket(price, 300, 7500, 50) - 1) * 144 AS bucket_floor,
    COUNT(*) AS count
FROM laptops
GROUP BY bucket
ORDER BY bucket;

-- 3.11 Cross-tab: Brand × Price Tier
CREATE OR REPLACE VIEW vw_brand_tier_crosstab AS
SELECT
    company,
    price_tier,
    COUNT(*) AS count,
    ROUND(AVG(price)::NUMERIC, 2) AS avg_price
FROM laptops
WHERE price_tier IS NOT NULL
GROUP BY company, price_tier
ORDER BY company, avg_price;

-- 3.12 Touch & IPS combinations
CREATE OR REPLACE VIEW vw_touch_ips_combos AS
SELECT
    touch_screen,
    ips,
    COUNT(*)                      AS count,
    ROUND(AVG(price)::NUMERIC, 2) AS avg_price
FROM laptops
GROUP BY touch_screen, ips
ORDER BY touch_screen, ips;

-- 3.13 Top 20 most expensive laptops
CREATE OR REPLACE VIEW vw_top20_expensive AS
SELECT
    id, company, type_name, cpu, gpu,
    ram_gb, total_storage_gb, price
FROM laptops
ORDER BY price DESC
LIMIT 20;

-- 3.14 Top 20 best value (high RAM / low price)
CREATE OR REPLACE VIEW vw_top20_best_value AS
SELECT
    id, company, type_name, cpu, gpu,
    ram_gb, total_storage_gb, price,
    ROUND((ram_gb::NUMERIC / price * 1000), 2) AS value_score
FROM laptops
ORDER BY value_score DESC
LIMIT 20;

-- 3.15 Aggregated price stats per brand × CPU brand
CREATE OR REPLACE VIEW vw_brand_cpu_brand AS
SELECT
    company,
    cpu_brand,
    COUNT(*)                      AS count,
    ROUND(AVG(price)::NUMERIC, 2) AS avg_price,
    ROUND(AVG(ram_gb)::NUMERIC,1) AS avg_ram
FROM laptops
GROUP BY company, cpu_brand
ORDER BY count DESC;

-- ----------------------------------------------------------------
-- 4. STORED PROCEDURE — Refresh / Upsert
-- ----------------------------------------------------------------
CREATE OR REPLACE FUNCTION upsert_laptop(
    p_company       VARCHAR,
    p_type_name     TEXT,
    p_inches        NUMERIC,
    p_cpu           TEXT,
    p_gpu           TEXT,
    p_op_sys        VARCHAR,
    p_touch_screen  SMALLINT,
    p_ips           SMALLINT,
    p_x_res         INT,
    p_y_res         INT,
    p_ppi           NUMERIC,
    p_dedicated_gpu SMALLINT,
    p_ram_gb        INT,
    p_weight_kg     NUMERIC,
    p_ssd           INT,
    p_hhd           INT,
    p_storage_type  VARCHAR,
    p_total_storage INT,
    p_storage_cat   VARCHAR,
    p_price         NUMERIC
)
RETURNS INTEGER LANGUAGE plpgsql AS $$
DECLARE
    v_id INTEGER;
BEGIN
    INSERT INTO laptops (
        company, type_name, inches, cpu, gpu, op_sys,
        touch_screen, ips, x_res, y_res, ppi, dedicated_gpu,
        ram_gb, weight_kg, ssd, hhd, storage_type,
        total_storage_gb, storage_category, price
    ) VALUES (
        p_company, p_type_name, p_inches, p_cpu, p_gpu, p_op_sys,
        p_touch_screen, p_ips, p_x_res, p_y_res, p_ppi, p_dedicated_gpu,
        p_ram_gb, p_weight_kg, p_ssd, p_hhd, p_storage_type,
        p_total_storage, p_storage_cat, p_price
    )
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$;

-- ----------------------------------------------------------------
-- 5. QUICK SANITY CHECKS (run after import)
-- ----------------------------------------------------------------
-- SELECT * FROM vw_kpi_summary;
-- SELECT * FROM vw_brand_stats LIMIT 10;
-- SELECT * FROM vw_price_tiers;
-- SELECT * FROM vw_cpu_brand_stats;
-- SELECT * FROM vw_top20_expensive;

-- End of file
