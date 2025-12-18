
"""
Runner script:
1.Loads df_messy
2.Runs DQ before
3.Runs ETL (fix + flag)
4.Runs DQ after
5.Saves cleaned output + logs + dq reports

Usage (from repo root):
  python src/data/run_etl.py --in data/processed/telco-Customer-Churn-messy-data.csv --out data/processed/telco-Customer-Churn-cleaned-data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.data.etl_cleaning_pipeline import run_etl, ETLConfig
from src.utils.dq_checks import run_all_checks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True, help="Path to messy CSV input")
    parser.add_argument("--out", dest="output_path", required=True, help="Path to cleaned CSV output")
    parser.add_argument("--reports-dir", dest="reports_dir", default="reports", help="Directory to write reports/logs")
    parser.add_argument("--drop-invalid-rows", action="store_true", help="Drop invalid-category rows instead of setting NaN")
    parser.add_argument("--no-impute", action="store_true", help="Do not impute missing values")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_messy = pd.read_csv(input_path)
    
    cfg = ETLConfig()
    dq_before = run_all_checks(df_messy ,cfg=cfg, dataset_name="before_etl")

    df_clean, etl_log = run_etl( df_messy, cfg, drop_invalid_rows=args.drop_invalid_rows, do_impute=not args.no_impute,)

    dq_after = run_all_checks(df_clean, cfg=cfg, dataset_name="after_etl")

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    # Save ETL log
    (reports_dir / "etl_log.json").write_text(json.dumps(etl_log, indent=2))

    # Save DQ before/after if available
    (reports_dir / "dq_before.json").write_text(json.dumps(dq_before, indent=2))
    (reports_dir / "dq_after.json").write_text(json.dumps(dq_after, indent=2))

    print("✅ ETL done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Reports: {reports_dir}")


if __name__ == "__main__":
    main()