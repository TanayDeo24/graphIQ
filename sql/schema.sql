-- GraphIQ core schema.
--
-- One shared employee/workforce data model that both the attrition-risk
-- (survival analysis) and spend-anomaly-detection components read from.
-- `employee_id` (IBM dataset's EmployeeNumber) is the join key used
-- throughout. All original IBM HR Analytics Attrition columns are kept
-- (renamed to snake_case) on `employees`; nothing from the source dataset
-- is dropped.
--
-- Result tables produced by the modeling pipelines (Sections 5-6 of the
-- build spec) live in sql/results_schema.sql, applied separately by the
-- loaders once model outputs exist — they are not part of this core schema.

BEGIN;

-- ---------------------------------------------------------------------
-- departments: derived from the distinct `department` values already
-- present in the IBM dataset (Sales, Research & Development, Human
-- Resources).
-- ---------------------------------------------------------------------
CREATE TABLE departments (
    department_id   SERIAL PRIMARY KEY,
    department_name TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
-- employees: one row per employee, loaded directly from the IBM HR
-- Analytics Attrition dataset (1,470 rows). Every original column is
-- mapped, none dropped. `duration_months` / `event_observed` are derived
-- convenience columns for survival analysis (tenure in months; whether
-- attrition was actually observed vs. right-censored).
-- ---------------------------------------------------------------------
CREATE TABLE employees (
    employee_id                INTEGER PRIMARY KEY,             -- EmployeeNumber
    age                        INTEGER NOT NULL,                -- Age
    attrition_flag              TEXT NOT NULL CHECK (attrition_flag IN ('Yes', 'No')), -- Attrition
    business_travel             TEXT,                            -- BusinessTravel
    daily_rate                  INTEGER,                         -- DailyRate
    department                  TEXT NOT NULL,                   -- Department (verbatim string, kept for fidelity)
    department_id               INTEGER REFERENCES departments(department_id),
    distance_from_home          INTEGER,                         -- DistanceFromHome
    education                   INTEGER,                         -- Education
    education_field              TEXT,                            -- EducationField
    employee_count               INTEGER,                         -- EmployeeCount
    environment_satisfaction     INTEGER,                         -- EnvironmentSatisfaction
    gender                      TEXT,                            -- Gender
    hourly_rate                  INTEGER,                         -- HourlyRate
    job_involvement              INTEGER,                         -- JobInvolvement
    job_level                    INTEGER,                         -- JobLevel
    job_role                    TEXT,                            -- JobRole
    job_satisfaction             INTEGER,                         -- JobSatisfaction
    marital_status               TEXT,                            -- MaritalStatus
    monthly_income               INTEGER NOT NULL,                -- MonthlyIncome
    monthly_rate                 INTEGER,                         -- MonthlyRate
    num_companies_worked          INTEGER,                         -- NumCompaniesWorked
    over_18                     TEXT,                            -- Over18
    over_time                   TEXT,                            -- OverTime
    percent_salary_hike           INTEGER,                         -- PercentSalaryHike
    performance_rating           INTEGER,                         -- PerformanceRating (1-4 in source)
    relationship_satisfaction     INTEGER,                         -- RelationshipSatisfaction
    standard_hours               INTEGER,                         -- StandardHours
    stock_option_level            INTEGER,                         -- StockOptionLevel
    total_working_years           INTEGER,                         -- TotalWorkingYears
    training_times_last_year      INTEGER,                         -- TrainingTimesLastYear
    work_life_balance             INTEGER,                         -- WorkLifeBalance
    tenure_years                 INTEGER NOT NULL,                -- YearsAtCompany
    years_in_current_role         INTEGER,                         -- YearsInCurrentRole
    years_since_last_promotion     INTEGER,                         -- YearsSinceLastPromotion
    years_with_curr_manager        INTEGER,                         -- YearsWithCurrManager
    duration_months              INTEGER NOT NULL,                -- derived: tenure_years * 12
    event_observed                BOOLEAN NOT NULL                 -- derived: attrition_flag = 'Yes'
);

CREATE INDEX idx_employees_department_id ON employees(department_id);
CREATE INDEX idx_employees_attrition_flag ON employees(attrition_flag);

-- ---------------------------------------------------------------------
-- comp_history: synthetic, time-varying compensation history generated
-- per src/generation/attrition_extension.py. Final month's monthly_income
-- for each employee equals their real, static MonthlyIncome from the IBM
-- dataset (the synthetic path is backward-consistent with the known value).
-- ---------------------------------------------------------------------
CREATE TABLE comp_history (
    id               SERIAL PRIMARY KEY,
    employee_id      INTEGER NOT NULL REFERENCES employees(employee_id),
    effective_month  DATE NOT NULL,
    monthly_income   NUMERIC(12, 2) NOT NULL,
    change_type      TEXT NOT NULL CHECK (change_type IN ('initial', 'raise', 'promotion_adjustment'))
);

CREATE INDEX idx_comp_history_employee_id ON comp_history(employee_id);
CREATE INDEX idx_comp_history_effective_month ON comp_history(effective_month);

-- ---------------------------------------------------------------------
-- performance_reviews: synthetic, one review every 6 months, score
-- centered on the employee's real PerformanceRating (rescaled 1-4 -> 1-5).
-- ---------------------------------------------------------------------
CREATE TABLE performance_reviews (
    id            SERIAL PRIMARY KEY,
    employee_id   INTEGER NOT NULL REFERENCES employees(employee_id),
    review_month  DATE NOT NULL,
    review_score  NUMERIC(3, 2) NOT NULL CHECK (review_score BETWEEN 1 AND 5)
);

CREATE INDEX idx_performance_reviews_employee_id ON performance_reviews(employee_id);
CREATE INDEX idx_performance_reviews_review_month ON performance_reviews(review_month);

-- ---------------------------------------------------------------------
-- benefits_enrollment: synthetic, single row per employee (not
-- time-varying) — plan tier probabilistically linked to job_level /
-- monthly_income. Exists to make the schema genuinely multi-table.
-- ---------------------------------------------------------------------
CREATE TABLE benefits_enrollment (
    id              SERIAL PRIMARY KEY,
    employee_id     INTEGER NOT NULL UNIQUE REFERENCES employees(employee_id),
    plan_tier       TEXT NOT NULL CHECK (plan_tier IN ('basic', 'standard', 'premium')),
    enrolled_month  DATE NOT NULL
);

CREATE INDEX idx_benefits_enrollment_employee_id ON benefits_enrollment(employee_id);
CREATE INDEX idx_benefits_enrollment_enrolled_month ON benefits_enrollment(enrolled_month);

-- ---------------------------------------------------------------------
-- expense_transactions: fully synthetic, generated per
-- src/generation/spend_generator.py, same 1,470 employees + 36-month
-- window as comp_history. is_injected_anomaly / anomaly_type are ground
-- truth kept ONLY for evaluation — never used as a model input feature.
-- ---------------------------------------------------------------------
CREATE TABLE expense_transactions (
    transaction_id       BIGSERIAL PRIMARY KEY,
    employee_id          INTEGER NOT NULL REFERENCES employees(employee_id),
    department_id        INTEGER NOT NULL REFERENCES departments(department_id),
    transaction_date     DATE NOT NULL,
    merchant_category    TEXT NOT NULL CHECK (
        merchant_category IN (
            'travel', 'software_saas', 'meals',
            'office_supplies', 'client_entertainment', 'other'
        )
    ),
    amount_usd            NUMERIC(12, 2) NOT NULL,
    is_injected_anomaly    BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_type          TEXT CHECK (
        anomaly_type IS NULL OR anomaly_type IN ('point_spike', 'slow_drift', 'coordinated_pattern')
    )
);

CREATE INDEX idx_expense_transactions_employee_id ON expense_transactions(employee_id);
CREATE INDEX idx_expense_transactions_department_id ON expense_transactions(department_id);
CREATE INDEX idx_expense_transactions_transaction_date ON expense_transactions(transaction_date);
CREATE INDEX idx_expense_transactions_is_injected_anomaly ON expense_transactions(is_injected_anomaly);

COMMIT;
