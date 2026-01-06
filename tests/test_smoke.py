import unittest
import pandas as pd

from src.data.etl_cleaning_pipeline import run_etl, ETLConfig


class TestProjectSmoke(unittest.TestCase):
    def test_etl_runs_on_tiny_df(self):
        # dataframe with minimal required columns
        df = pd.DataFrame({
            "customerID": ["A", "A", "B"],
            "Churn": ["Yes", "Yes", "No"],
            "tenure": ["1", " 2 ", "1000000"],               # messy numeric
            "MonthlyCharges": ["70.0", "9999", "85.5"],     # messy numeric
            "TotalCharges": ["70.0", "NaN", "1000000"],     # messy numeric
            "InternetService": ["No", "No", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No internet service", "No"],
            "DeviceProtection": ["Yes", "No internet service", "No"],
            "TechSupport": ["Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No internet service", "No"],
            "StreamingMovies": ["Yes", "No internet service", "No"],
            "PhoneService": ["No", "No", "Yes"],
            "MultipleLines": ["Yes", "No phone service", "No"],
            "gender": [" Male", "Female ", "Male"],
            "SeniorCitizen": ["0", "1", "No"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        })

        cfg = ETLConfig()
        df_clean, log = run_etl(df, cfg)

        # Basic assertions
        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertEqual(len(df_clean), len(df))
        self.assertIn("is_duplicate", df_clean.columns)
        self.assertIsInstance(log, dict)


if __name__ == "__main__":
    unittest.main()