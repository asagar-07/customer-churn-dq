This project simulates a real-world machine learning workflow where raw customer data is often incomplete, inconsistent, messy, or incorrect.

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