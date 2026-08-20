-- GraphIQ result-table schema.
--
-- These tables hold precomputed outputs from the modeling/evaluation
-- pipelines (src/models/attrition, src/models/spend). The FastAPI layer
-- (src/api) only ever queries these — it never recomputes a metric at
-- request time. Applied separately from sql/schema.sql (after the core
-- 6-table schema + data are loaded), by src/db/load_to_postgres.py.

BEGIN;

-- =====================================================================
-- Attrition result tables
-- =====================================================================

CREATE TABLE IF NOT EXISTS attrition_risk_scores (
    employee_id             INTEGER PRIMARY KEY REFERENCES employees(employee_id),
    department               TEXT NOT NULL,
    tenure_band              TEXT NOT NULL,
    data_split                TEXT NOT NULL,
    cox_risk_score            DOUBLE PRECISION NOT NULL,
    gbm_risk_score            DOUBLE PRECISION NOT NULL,
    gbm_predicted_survival_12m DOUBLE PRECISION NOT NULL,
    is_top_risk_quartile        BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attrition_risk_scores_department ON attrition_risk_scores(department);
CREATE INDEX IF NOT EXISTS idx_attrition_risk_scores_tenure_band ON attrition_risk_scores(tenure_band);

CREATE TABLE IF NOT EXISTS attrition_model_metrics (
    id             SERIAL PRIMARY KEY,
    model          TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    horizon_months   INTEGER,
    metric_value     DOUBLE PRECISION NOT NULL,
    ci_low          DOUBLE PRECISION,
    ci_high         DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS attrition_ph_assumption (
    id              SERIAL PRIMARY KEY,
    feature         TEXT NOT NULL,
    test_statistic   DOUBLE PRECISION NOT NULL,
    p_value          DOUBLE PRECISION NOT NULL,
    violated        BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS attrition_calibration (
    id                    SERIAL PRIMARY KEY,
    segment_dimension       TEXT NOT NULL,     -- 'department' | 'tenure_band' | 'comp_band'
    segment_value            TEXT NOT NULL,
    horizon_months            INTEGER NOT NULL,
    predicted_survival        DOUBLE PRECISION NOT NULL,
    observed_survival          DOUBLE PRECISION NOT NULL,
    calibration_error          DOUBLE PRECISION NOT NULL,
    logrank_p_value            DOUBLE PRECISION NOT NULL,
    n_at_risk                 INTEGER NOT NULL,   -- employees in this segment (test set)
    event_count                INTEGER NOT NULL    -- departures observed within horizon_months
);

CREATE TABLE IF NOT EXISTS attrition_lead_time (
    id               SERIAL PRIMARY KEY,
    employee_id       INTEGER NOT NULL REFERENCES employees(employee_id),
    flagged_month      INTEGER NOT NULL,
    departure_month     INTEGER NOT NULL,
    lead_time_months    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS attrition_lead_time_summary (
    id                SERIAL PRIMARY KEY,
    statistic          TEXT NOT NULL,   -- 'mean' | 'median'
    point_estimate       DOUBLE PRECISION NOT NULL,
    ci_low             DOUBLE PRECISION NOT NULL,
    ci_high            DOUBLE PRECISION NOT NULL,
    n_bootstrap          INTEGER NOT NULL,
    n_true_positives      INTEGER NOT NULL,
    n_excluded_no_crossing INTEGER NOT NULL,
    risk_threshold_note    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attrition_sensitivity (
    id                SERIAL PRIMARY KEY,
    employee_id        INTEGER NOT NULL REFERENCES employees(employee_id),
    base_risk           DOUBLE PRECISION NOT NULL,
    perturbed_risk       DOUBLE PRECISION NOT NULL,
    risk_change          DOUBLE PRECISION NOT NULL,
    risk_change_pct       DOUBLE PRECISION NOT NULL,
    disclaimer          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attrition_shap_global (
    id            SERIAL PRIMARY KEY,
    feature       TEXT NOT NULL,
    mean_abs_shap  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS attrition_shap_employee (
    id            SERIAL PRIMARY KEY,
    employee_id    INTEGER NOT NULL REFERENCES employees(employee_id),
    feature       TEXT NOT NULL,
    feature_value   DOUBLE PRECISION,
    shap_value     DOUBLE PRECISION NOT NULL,
    base_value     DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attrition_shap_employee_id ON attrition_shap_employee(employee_id);

-- Dashboard interaction heatmap: baseline_tenure_band x review_score_trend
-- bucket, cell = mean baseline GBM risk score. low_confidence = n < 10.
CREATE TABLE IF NOT EXISTS attrition_interaction_heatmap (
    id                  SERIAL PRIMARY KEY,
    tenure_band           TEXT NOT NULL,
    review_trend_bucket    TEXT NOT NULL,
    n                    INTEGER NOT NULL,
    mean_gbm_risk_score     DOUBLE PRECISION,
    low_confidence         BOOLEAN NOT NULL
);

-- ILLUSTRATIVE ONLY (src/analysis/risk_migration.py) -- a rolling re-scoring
-- of the validated GBM model using comp_history/performance_reviews values
-- as of each checkpoint, for the dashboard's risk-migration Sankey. Never
-- used as, or mixed with, any validated evaluation metric.
CREATE TABLE IF NOT EXISTS attrition_risk_migration_checkpoints (
    id                SERIAL PRIMARY KEY,
    employee_id        INTEGER NOT NULL REFERENCES employees(employee_id),
    checkpoint_month     INTEGER NOT NULL,
    risk_score          DOUBLE PRECISION NOT NULL,
    tier               TEXT NOT NULL  -- 'low' | 'medium' | 'high', tercile at that checkpoint
);
CREATE INDEX IF NOT EXISTS idx_attrition_risk_migration_checkpoints_employee_id ON attrition_risk_migration_checkpoints(employee_id);

CREATE TABLE IF NOT EXISTS attrition_risk_migration_sankey (
    id                SERIAL PRIMARY KEY,
    checkpoint_from      INTEGER NOT NULL,
    checkpoint_to        INTEGER NOT NULL,
    tier_from           TEXT NOT NULL,
    tier_to             TEXT NOT NULL,
    employee_count       INTEGER NOT NULL
);

-- =====================================================================
-- Spend result tables
-- =====================================================================

CREATE TABLE IF NOT EXISTS spend_anomaly_scores (
    transaction_id           BIGINT PRIMARY KEY REFERENCES expense_transactions(transaction_id),
    employee_id              INTEGER NOT NULL REFERENCES employees(employee_id),
    department_id            INTEGER NOT NULL REFERENCES departments(department_id),
    isolation_forest_score      DOUBLE PRECISION NOT NULL,
    autoencoder_score           DOUBLE PRECISION NOT NULL,
    cusum_flag                 BOOLEAN NOT NULL,
    ensemble_score              DOUBLE PRECISION NOT NULL,
    predicted_flag              BOOLEAN NOT NULL,
    is_injected_anomaly          BOOLEAN NOT NULL,
    anomaly_type                TEXT
);
CREATE INDEX IF NOT EXISTS idx_spend_anomaly_scores_employee_id ON spend_anomaly_scores(employee_id);
CREATE INDEX IF NOT EXISTS idx_spend_anomaly_scores_predicted_flag ON spend_anomaly_scores(predicted_flag);

CREATE TABLE IF NOT EXISTS spend_eval_metrics (
    id            SERIAL PRIMARY KEY,
    detector       TEXT NOT NULL,   -- 'isolation_forest' | 'autoencoder' | 'cusum' | 'ensemble'
    anomaly_type    TEXT NOT NULL,   -- 'point_spike' | 'slow_drift' | 'coordinated_pattern' | 'overall'
    metric_name     TEXT NOT NULL,   -- 'precision' | 'recall' | 'pr_auc' | 'lift_over_random'
    metric_value     DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS spend_gains_curve (
    id                       SERIAL PRIMARY KEY,
    pct_alerts_raised           DOUBLE PRECISION NOT NULL,
    pct_dollar_volume_captured   DOUBLE PRECISION NOT NULL
);

-- Full monthly CUSUM series per (employee, category) — backs the dashboard's
-- CUSUM chart with visible control limits (h=5sigma, k=0.5sigma; see cusum.py).
CREATE TABLE IF NOT EXISTS spend_cusum_series (
    id               SERIAL PRIMARY KEY,
    employee_id       INTEGER NOT NULL REFERENCES employees(employee_id),
    department_id      INTEGER NOT NULL REFERENCES departments(department_id),
    merchant_category  TEXT NOT NULL,
    month             DATE NOT NULL,
    monthly_total       DOUBLE PRECISION NOT NULL,
    cusum_statistic     DOUBLE PRECISION NOT NULL,
    flagged           BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_cusum_series_employee_id ON spend_cusum_series(employee_id);
CREATE INDEX IF NOT EXISTS idx_spend_cusum_series_department_id ON spend_cusum_series(department_id);

CREATE TABLE IF NOT EXISTS spend_drift_delay (
    id                          SERIAL PRIMARY KEY,
    employee_id                  INTEGER NOT NULL REFERENCES employees(employee_id),
    merchant_category             TEXT NOT NULL,
    onset_month                   DATE NOT NULL,
    end_month                     DATE NOT NULL,   -- last month of the injected drift's active window
    flagged_month                  DATE,
    delay_months                   INTEGER,
    caught_during_active_window      BOOLEAN         -- NULL if never flagged; else flagged_month <= end_month
);

CREATE TABLE IF NOT EXISTS spend_drift_delay_summary (
    id            SERIAL PRIMARY KEY,
    statistic      TEXT NOT NULL,
    point_estimate   DOUBLE PRECISION NOT NULL,
    ci_low         DOUBLE PRECISION NOT NULL,
    ci_high        DOUBLE PRECISION NOT NULL,
    n_bootstrap      INTEGER NOT NULL,
    n_cases         INTEGER NOT NULL
);

-- Complements spend_drift_delay_summary's delay statistic with the honest
-- "caught while still drifting vs. only caught after it already ended" split
-- (see src/models/spend/evaluate.py's drift_detection_timing()).
CREATE TABLE IF NOT EXISTS spend_drift_timing_summary (
    id                        SERIAL PRIMARY KEY,
    n_total_cases                INTEGER NOT NULL,
    n_detected                  INTEGER NOT NULL,
    n_undetected                 INTEGER NOT NULL,
    n_caught_during_active         INTEGER NOT NULL,
    n_caught_after_ended           INTEGER NOT NULL,
    pct_caught_during_active       DOUBLE PRECISION NOT NULL,  -- of DETECTED cases only
    pct_caught_after_ended         DOUBLE PRECISION NOT NULL   -- of DETECTED cases only
);

CREATE TABLE IF NOT EXISTS spend_alert_fatigue (
    id                    SERIAL PRIMARY KEY,
    operating_threshold      DOUBLE PRECISION NOT NULL,
    alerts_raised            INTEGER NOT NULL,
    total_transactions        INTEGER NOT NULL,
    alerts_per_1000_txns       DOUBLE PRECISION NOT NULL,
    precision_at_threshold     DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS spend_robustness (
    id            SERIAL PRIMARY KEY,
    injection_rate  TEXT NOT NULL,  -- '1pct' | '5pct' | '10pct'
    anomaly_type    TEXT NOT NULL,
    precision      DOUBLE PRECISION NOT NULL,
    recall         DOUBLE PRECISION NOT NULL,
    pr_auc         DOUBLE PRECISION NOT NULL
);

-- Dashboard: detector overlap at the existing operating threshold.
CREATE TABLE IF NOT EXISTS spend_detector_overlap (
    id                SERIAL PRIMARY KEY,
    combination         TEXT NOT NULL,    -- 'only_IF' | 'only_AE' | 'only_CUSUM' | 'IF+AE' | 'IF+CUSUM' | 'AE+CUSUM' | 'all_three'
    transaction_count     INTEGER NOT NULL
);

-- Dashboard: department x category x anomaly_type dollar treemap.
CREATE TABLE IF NOT EXISTS spend_dollar_treemap (
    id                SERIAL PRIMARY KEY,
    department_id       INTEGER NOT NULL REFERENCES departments(department_id),
    merchant_category    TEXT NOT NULL,
    anomaly_type        TEXT NOT NULL,
    dollar_volume        DOUBLE PRECISION NOT NULL,
    transaction_count     INTEGER NOT NULL
);

-- Dashboard: curated, annotated CUSUM trajectories for a handful of real
-- detected slow_drift cases (see select_annotated_cusum_cases()).
CREATE TABLE IF NOT EXISTS spend_cusum_annotated_trajectory (
    id                          SERIAL PRIMARY KEY,
    case_label                    TEXT NOT NULL,
    employee_id                   INTEGER NOT NULL REFERENCES employees(employee_id),
    merchant_category               TEXT NOT NULL,
    month                        DATE NOT NULL,
    cusum_statistic                 DOUBLE PRECISION NOT NULL,
    flagged                       BOOLEAN NOT NULL,
    onset_month                    DATE NOT NULL,
    end_month                      DATE NOT NULL,
    flagged_month                   DATE,
    caught_during_active_window       BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_spend_cusum_annotated_trajectory_case_label ON spend_cusum_annotated_trajectory(case_label);

CREATE TABLE IF NOT EXISTS spend_transaction_explain (
    id            SERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES expense_transactions(transaction_id),
    sub_signal     TEXT NOT NULL,  -- 'amount_deviation' | 'frequency' | 'merchant_novelty'
    contribution    DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_transaction_explain_transaction_id ON spend_transaction_explain(transaction_id);

-- =====================================================================
-- Cross-component result tables (src/analysis/cross_component.py)
-- =====================================================================

CREATE TABLE IF NOT EXISTS cross_component_quadrant (
    employee_id            INTEGER PRIMARY KEY REFERENCES employees(employee_id),
    department              TEXT NOT NULL,
    tenure_band              TEXT NOT NULL,
    gbm_risk_score            DOUBLE PRECISION NOT NULL,
    is_top_risk_quartile        BOOLEAN NOT NULL,
    spend_anomaly_score         DOUBLE PRECISION NOT NULL,  -- max ensemble_score across the employee's transactions
    is_top_spend_quartile        BOOLEAN NOT NULL,
    quadrant                   TEXT NOT NULL  -- 'high_risk_high_anomaly' | 'high_risk_low_anomaly' | 'low_risk_high_anomaly' | 'low_risk_low_anomaly'
);
CREATE INDEX IF NOT EXISTS idx_cross_component_quadrant_quadrant ON cross_component_quadrant(quadrant);

CREATE TABLE IF NOT EXISTS cross_component_summary (
    id                     SERIAL PRIMARY KEY,
    spearman_correlation      DOUBLE PRECISION NOT NULL,
    p_value                  DOUBLE PRECISION NOT NULL,
    n_permutations             INTEGER NOT NULL,
    n_employees               INTEGER NOT NULL,
    method_note              TEXT NOT NULL,
    disclaimer               TEXT NOT NULL
);
-- Partial-correlation columns (controlling for department + monthly_income) added
-- after the table already existed in deployed databases -- ADD COLUMN IF NOT EXISTS
-- rather than a new CREATE, so this migration is idempotent against a live table.
ALTER TABLE cross_component_summary ADD COLUMN IF NOT EXISTS partial_spearman_correlation DOUBLE PRECISION;
ALTER TABLE cross_component_summary ADD COLUMN IF NOT EXISTS partial_p_value DOUBLE PRECISION;
ALTER TABLE cross_component_summary ADD COLUMN IF NOT EXISTS partial_confounds TEXT;
ALTER TABLE cross_component_summary ADD COLUMN IF NOT EXISTS partial_method_note TEXT;

CREATE TABLE IF NOT EXISTS cross_component_quadrant_characteristics (
    id                SERIAL PRIMARY KEY,
    quadrant           TEXT NOT NULL,
    dimension          TEXT NOT NULL,  -- 'department' | 'tenure_band'
    dimension_value      TEXT NOT NULL,
    count              INTEGER NOT NULL,
    pct_of_quadrant      DOUBLE PRECISION NOT NULL
);

COMMIT;
