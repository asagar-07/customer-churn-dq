CUSTOMER-CHURN

This project simulates a real-world machine learning workflow where raw customer data is often incomplete, inconsistent, messy, or incorrect.

[![CI](https://github.com/asagar-07/customer-churn-dq/actions/workflows/ci.yml/badge.svg)](
https://github.com/asagar-07/customer-churn-dq/actions/workflows/ci.yml
)



Key goals:
	•	Build a dirty data generation engine that systematically injects realistic corruption
	•	Develop a robust ETL pipeline to clean, standardize, and validate the messy dataset
	•	Train and evaluate ML models for customer churn prediction
	•	Compare model performance on:
	•	Clean raw data
	•	Messy data
	•	Cleaned (ETL-processed) messy data

This project showcases data engineering, data quality, ML modeling, and software design skills.


v0.1 – EDA on Raw Data

v0.2 – Messy Data Generation Complete + EDA on Messy Data + Compare Messy vs Raw
    •   Raw telco churn dataset loaded
    •   Custom pipeline to generate:
    •   whitespace noise
    •   numeric outliers and invalid numeric values
    •   logical inconsistencies (InternetService / PhoneService rules)
    •   missing values
    •   UNKNOWN placeholders
    •   inconsistent categorical values (typos/synonyms)
    •   label noise on Churn
    •   duplicate vs raw rows with different corruption intensity
    •   Output: df_messy suitable for ETL & data quality exercises.

v0.3 - ETL Cleaning Pipeline with Audit Logging and a reproducible runner script.
    •   Clean and validates data using an ETL pipeline
    •   tracks what changed and why via structured DQ logs
    •   prepares a clean, model-ready dataset.
    •   whitespace normalization for categorical fields
    •   Unknown value normalization -> to Nan
    •   Reverse map to to correct categorical values (e.g., Y / y / 1 → Yes)
    •   Numeric coercion (string->numeric, infinite->Nan, negative -> Nan)
    •   Logical consistency enforcement ( dependency feature rules)
    •   Duplicate Detection (is_duplicate, id_count, rows preserved)
    •   Missing value Imputation (numeric->median, categorical->mode)
    
    •   ETL Audit Logging (every ETL run produces a JSON log)
    •   row counts before and after ETL
    •   counts of changes per step and column
    •   Type of fixes/actions
    •   Examples of data correction (top 10)
    •   duplicate statistics
    
    •   Runner Script(run_etl.py)
    •   load messy input data, execute ETL pipeline, write clean_dataset and ETL audit log

    Output Artifacts (v0.3)
	•	data/processed/customers_clean_for_model.csv (Clean, ML-ready dataset)
	•	reports/etl_log.json (Detailed ETL audit and data quality log)

v0.4 - DQ reporting workflow to compare before and after ETL.
    •   Updated ETL pipline by introducing additional modules.
    •   a - Enforce logical dependencies betwen parent and dependent features
    •   b - enforced domain specific numeric ranges
    
    DQ checks (dq_checks.py) 
	•	Generates JSON-serializable reports for:
	•	dataset meta + schema overview
	•	missingness totals + per-column
	•	duplicates metrics (counts + top duplicate IDs + flag validation)
	•	numeric anomalies (NaN/inf/negative + min/median/max)
	•	categorical anomalies (NaN, unique count, top values, whitespace count, unknown token count, invalid category count + examples)
	•	logical rule violations (dependency enforcement checks)
	•	label distribution (Churn) counts + percentages + missing
	•	Output Artifacts (v0.4)
	•	reports/dq_before.json (messy input)
	•	reports/dq_after.json (cleaned output)

v0.5 - Baseline Model & Business Thresholding & Artifacts

Problem - Predict customer churn using historical Telco customer data and convert predictions into actionable retention signals.

Data Preparation & Feature Engineering
	•	Performed EDA on raw and cleaned data to understand churn drivers and data quality issues.
	•	Derived additional features (e.g., bundled services count).
	•	Identified:
	•	Target: Churn
	•	Numeric features: tenure, monthly charges, total charges (scaled using MinMaxScaler)
	•	Categorical features: contract type, internet services, billing methods, etc. (one-hot encoded)
	•	Ensured consistent preprocessing across train, validation, and test splits.

Model
	•	Algorithm: Logistic Regression (baseline, interpretable)
	•	Why: Provides stable probabilistic outputs and clear feature influence for business interpretation.
	•	Evaluation metric focus: ROC-AUC, Recall, Precision, F1 (accuracy not sufficient due to class imbalance).

Key Results (Test Set)
	•	ROC-AUC: ~0.79
	•	Accuracy: ~0.77
	•	Recall (Churn = Yes): ~0.47 at default threshold (0.5)

Top churn drivers (by model weights):
	•	Month-to-month contracts
	•	Lack of tech support / online backup
	•	Fiber optic internet
	•	High monthly charges
	•	Electronic check payment method


Threshold Analysis & Business Framing
	•	Evaluated thresholds from 0.50 → 0.25 on the validation set.
	•	Observed strong precision–recall tradeoff:
	•	Higher thresholds miss many churners.
	•	Lower thresholds increase recall but expand campaign size.

Selected threshold: 0.40

At threshold 0.40:
	•	Recall (Yes): ~61%
	•	Precision (Yes): ~61%
	•	Balanced F1 score
	•	Operationally manageable number of customers flagged for retention

Business rationale:
Missing a true churner is costlier than contacting a loyal customer; therefore, recall is prioritized while maintaining reasonable precision.

Artifacts:
- churn_lr_model.pkl
- scaler.pkl
- encoder.pkl
- feature_columns.json
- model_metadata.json


v0.6 - Feature Engineering Comparison and Model Selection

This release extends the baseline churn model with feature engineering experiements, compares their impact systematically and finalizes a business-ready model configuration using vlaidation-driven threshold analysis.

Four controlled experiments were evaluated using same preprocessing, model type and decision thrshold.

Experiment 	Desctiption
E0		   	Baseline 
E1			+ TenureBin (categorical buckets of tenure)
E2			+ AvgMonthlyFromTotal (TotalCharges/tenure)
E3			+ E1 + E2

- All transformations were fit on training data only.
- Applied consistently to validation and test sets
- Evaluated using identical metrics

	•	Model: Logistic Regression
	•	Threshold: 0.40 (chosen via validation threshold sweep)
	•	Primary metrics: ROC-AUC, Precision, Recall, F1 (Churn = “Yes”)

Validation Performance (threshold = 0.40)

Experiment	ROC-AUC	F1(Yes)	Flagged	Customers
E0 			0.813 	0.607 	463
E1 			0.816 	0.611 	444
E2 			0.814 	0.607 	463
E3 			0.817 	0.610 	445

Performance remained stable across experiments, with no overfitting signal.

Final Model Selected: E1 ( TenureBin only)
	•	Best validation F1 score
	•	Comparable ROC-AUC to more complex variants
	•	Clear business interpretability
	•	Minimal feature complexity
	•	Stable customer-flagging volume

"At 0.40 threshold, the model identifies ~60% of churners while flagging ~440 customers for retention outreach — a reasonable trade-off between recall and operational cost."


v0.7 - Persistent Artifacts and Smoke Test (minimal CI)
     - another commit to fix CI.yml


## What This Project Demonstrates

This project demonstrates an end-to-end **Data Quality + Machine Learning pipeline**
built with production-oriented practices.

Key capabilities:

- **Data Quality Engineering**
  - Dataset shape, missingness, duplicates, numeric & categorical anomalies
  - Logical dependency checks (InternetService / PhoneService relationships)
  - Before vs After ETL DQ reporting

- **Robust ETL Pipeline**
  - Whitespace trimming, normalization, type coercion
  - Domain-specific numeric repairs (tenure, charges)
  - Logical consistency enforcement
  - Detailed ETL logging with examples

- **Feature Engineering Experiments**
  - Baseline vs engineered features
  - Tenure bins and derived financial ratios
  - Controlled experiment comparison (E0–E3)

- **Modeling & Evaluation**
  - Logistic Regression baseline
  - Train / validation / test splits
  - ROC-AUC, Precision, Recall, F1 analysis
  - Business-driven threshold tuning

- **Model Artifact Management**
  - Persisted model, scaler, encoder, feature schema
  - Versioned artifacts with metadata and model card

- **Production Readiness**
  - Reproducible pipeline execution
  - Smoke tests
  - GitHub Actions CI for automated validation


  ## Artifacts & Reports

### Model Artifacts
Persisted model and preprocessing artifacts (v0.6 best experiment):

models/
└── v0_6_best/
├── lr_v0_6_best.joblib
├── preprocess_v0_6_best.joblib
└── model_card_v0_6_best.json

These artifacts include:
- Trained Logistic Regression model
- Fitted scaler and encoder
- Feature ordering and threshold
- Model metadata and evaluation summary

---

### Data Quality Reports
Generated during ETL execution:

reports/
├── dq_before.json
├── dq_after.json
└── etl_log.json

These reports capture:
- Dataset health before and after cleaning
- Column-level anomaly counts
- Logical rule violations
- Exact transformations applied during ETL

## Continuous Integration
This repository uses **GitHub Actions** to validate every push and pull request.

CI checks include:
- ETL runner smoke test
- Unit test execution
- Dependency installation validation

Workflow definition: 
.github/workflows/ci.yml