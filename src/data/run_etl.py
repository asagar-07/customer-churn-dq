
"""
Runner script:
- Loads df_messy
- Runs DQ before
- Runs ETL (fix + flag)
- Runs DQ after
- Saves cleaned output + logs + dq reports

Usage (from repo root):
  python scripts/run_etl.py --in data/processed/telco-Customer-Churn-messy-data.csv --out data/processed/telco-Customer-Churn-cleaned-data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.data.etl_cleaning_pipeline import run_etl, ETLConfig

# You’ll implement these in src/utils/dq_checks.py
# They should return JSON-serializable dicts.
try:
    from src.utils.dq_checks import run_all_checks
except Exception:
    run_all_checks = None


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

    dq_before = None
    if run_all_checks is not None:
        dq_before = run_all_checks(df_messy)

    cfg = ETLConfig()
    df_clean, etl_log = run_etl(
        df_messy,
        cfg,
        drop_invalid_rows=args.drop_invalid_rows,
        do_impute=not args.no_impute,
    )

    dq_after = None
    if run_all_checks is not None:
        dq_after = run_all_checks(df_clean)

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    # Save ETL log
    (reports_dir / "etl_log.json").write_text(json.dumps(etl_log, indent=2))

    # Save DQ before/after if available
    if dq_before is not None:
        (reports_dir / "dq_before.json").write_text(json.dumps(dq_before, indent=2))
    if dq_after is not None:
        (reports_dir / "dq_after.json").write_text(json.dumps(dq_after, indent=2))

    print("✅ ETL done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Reports: {reports_dir}")


if __name__ == "__main__":
    main()