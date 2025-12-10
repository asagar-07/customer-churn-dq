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
