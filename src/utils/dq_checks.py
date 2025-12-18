# Dataset Basics, shape(rows, cols), dtypes: counts by dtype (or per-column dtype string)
#Missingness (missing_total, by column(descending), missing_pct_by_column)
#Duplicates (count of rows, duplicate_ids, top 10 duplicate_ids)
# Numeric Health (nan_count, inf_count, negative_count, basic_sats: min/median/max)
# categorical health (unique_count, top_5_values, unknwon_token_count before ETL, whitespace_count)
# valid_values enforcement (ETL_config.valid_values, invalid_category_count, invalid_examples(10))
# Logical consistency checks(logical_violation count, logical_examples)
# Target lable distribution ( churn_dist,counts, %)
# Number Range anamolies



from __future__ import annotations
from datetime import datetime
from src.data.etl_cleaning_pipeline import ETLConfig
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


# ------------------------------------
# DQ Valdiation Functions 
# ------------------------------------

#meta check
def meta_check(df: pd.DataFrame, *, dataset_name: str, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """ Collecting metadata about the DQ run and dataset context. """
    cfg = cfg or ETLConfig()

    rows, cols = df.shape

    meta = {
        "dataset_name": dataset_name,           # e.g., "before_etl" | "after_etl"
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "shape": {
            "rows": int(rows),
            "cols": int(cols),
        },
        "id_column": cfg.id_column if hasattr(cfg, "id_column") else None,
        "target_column": cfg.target_column if hasattr(cfg, "target_column") else None,
        "numeric_columns": list(cfg.numeric_columns) if hasattr(cfg, "numeric_columns") else [],
        "categorical_columns": list(cfg.categorical_columns) if hasattr(cfg, "categorical_columns") else [],
        "has_duplicate_flags": {
            "id_count": "id_count" in df.columns,
            "is_duplicate": "is_duplicate" in df.columns,
        },
    }

    return meta


#Basic Dataset Check
def dataset_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """Check DataFrame shape and dtypes."""
    _ = cfg or ETLConfig()
    
    rows, cols = df.shape

    columns_by_dtype: Dict[str, list]= {}
    for col_name, dtype in df.dtypes.items():
        dtype_name = str(dtype)
        columns_by_dtype.setdefault(dtype_name, []).append(col_name)

    return {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "columns": df.columns.tolist().sort(),
        "columns_by_dtype": columns_by_dtype
    }


#Missingness_total
def missingness_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str,Any]:
    """Aggregrate Missing values"""
    _ = cfg or ETLConfig()

   # Per-column missing counts
    missing_counts = df.isna().sum()
    missing_by_column = {col: int(missing_counts[col]) for col in missing_counts.index}

    total_missing = int(missing_counts.sum())
    rows = int(df.shape[0]) if df.shape[0] else 0

    # Missing percent by column (safe for rows=0)
    if rows == 0:
        missing_pct_by_column = {col: 0.0 for col in df.columns}
    else:
        missing_pct_by_column = {col: float(missing_by_column[col] / rows) for col in df.columns}

    # Sort by missing count descending
    missing_by_column_sorted = dict(sorted(missing_by_column.items(), key=lambda kv: kv[1], reverse=True))
    missing_pct_by_column_sorted = {col: missing_pct_by_column[col] for col in missing_by_column_sorted.keys()}

    return {
        "total_missing": total_missing,
        "missing_by_column": missing_by_column_sorted,
        "missing_pct_by_column": missing_pct_by_column_sorted,
    }


def duplicate_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """Duplicate-related metrics"""
    cfg = cfg or ETLConfig()
    id_column = cfg.id_column  # "customerID"

    actual_duplicates = df.duplicated(subset=[id_column], keep=False)

    # Count ALL rows that are part of duplicated IDs
    duplicate_row_count = int(actual_duplicates.sum())

    unique_customer_id_count = int(df[id_column].nunique())

    duplicate_id_count = int(df.loc[actual_duplicates, id_column].nunique())

    duplicate_flagged_correctly = None
    if "is_duplicate" in df.columns:
        duplicate_flagged_correctly = bool((df["is_duplicate"] == actual_duplicates).all())

    top_duplicate_ids = df[id_column].value_counts().head(10).to_dict()

    return {
        "duplicate_row_count": duplicate_row_count,
        "unique_customer_id_count": unique_customer_id_count,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_flagged_correctly": duplicate_flagged_correctly,
        "top_duplicate_ids": top_duplicate_ids,
    }


#numeric check
def numeric_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """Numeric-related metrics"""
    cfg = cfg or ETLConfig()
    
    per_column:Dict[str, Any] = {}

    for col in cfg.numeric_columns:
        if col not in df.columns:
            continue  # "tenure, monthly charges, total charges"

        numeric_coerced = pd.to_numeric(df[col], errors="coerce")
        #nan_count if present per col
        nan_count = int(numeric_coerced.isna().sum())
        #inf_count if present per col
        inf_count = int(numeric_coerced.isin([np.inf, -np.inf]).sum())
        #negative_count if present per col
        negative_count = int((numeric_coerced < 0).sum())
        #find min, median, max
        non_na = numeric_coerced.replace([np.inf, -np.inf], np.nan).dropna()

        min_value = median_value = max_value = None
        if not non_na.empty:
            min_value = float(non_na.min())
            median_value = float(non_na.median())
            max_value = float(non_na.max())
        
        
        #optional: outliers (IQR, Z-score)

        per_column[col] = {
            "nan_count": nan_count, 
            "inf_count": inf_count,
            "negative_count": negative_count,
            "min_value": min_value,
            "median_value": median_value,
            "max_value": max_value,
        }
    
    return {
        "columns": list(per_column.keys()),
        "per_column": per_column
    }

#categorical check
def categorical_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """
    Categorical DQ metrics for columns listed in cfg.categorical_columns.

    Returns JSON-serializable dict with:
      - columns: list of categorical columns evaluated
      - per_column: per-col metrics
      - flags: config-vs-df categorical column mismatch info

    Rules:
      - unknown tokens come from cfg.unknown_tokens
      - invalid category = not in cfg.valid_values[col] (if present), excluding NaN + unknown tokens
      - whitespace_count = count of non-null values where str(value) != str(value).strip()
    """
    cfg = cfg or ETLConfig()

    # If df empty, report config-vs-df flags, but no per-column metrics
    if df.shape[0] == 0:
        df_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cfg_cat_cols = list(getattr(cfg, "categorical_columns", []) or [])

        return {
            "columns": [],
            "per_column": {},
            "flags": {
                "df_categorical_columns_detected": df_cat_cols,
                "cfg_categorical_columns_expected": cfg_cat_cols,
                "extra_categorical_columns_in_df": sorted(list(set(df_cat_cols) - set(cfg_cat_cols))),
                "missing_categorical_columns_in_df": sorted(list(set(cfg_cat_cols) - set(df_cat_cols))),
            },
        }

    # Config-driven lists/mapping
    cfg_cat_cols = list(getattr(cfg, "categorical_columns", []) or [])
    unknown_tokens = set(getattr(cfg, "unknown_tokens", []) or [])
    valid_values_map: Dict[str, Any] = getattr(cfg, "valid_values", {}) or {}

    # Detect categorical columns in df
    df_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    flags = {
        "df_categorical_columns_detected": df_cat_cols,
        "cfg_categorical_columns_expected": cfg_cat_cols,
        "extra_categorical_columns_in_df": sorted(list(set(df_cat_cols) - set(cfg_cat_cols))),
        "missing_categorical_columns_in_df": sorted(list(set(cfg_cat_cols) - set(df_cat_cols))),
    }

    per_column: Dict[str, Any] = {}

    for col in cfg_cat_cols:
        if col not in df.columns:
            # If cfg expects it but df doesn't have it,  flag it in `flags`
            continue

        s = df[col]  # do not mutate df

        # Convert to string for comparisons without altering df
        s_str = s.astype("string")

        nan_count = int(s_str.isna().sum())
        non_null = s_str.dropna()

        unique_count = int(non_null.nunique())
        top_5_values = non_null.value_counts().head(5).to_dict()

        # Whitespace: leading/trailing whitespace (strip changes)
        whitespace_mask = non_null.ne(non_null.str.strip())
        whitespace_count = int(whitespace_mask.sum())

        # Unknown token: exact raw match
        unknown_mask = non_null.isin(unknown_tokens)
        unknown_values_count = int(unknown_mask.sum())

        # Invalid categories: only if valid set exists for this column
        valid_set = set(valid_values_map.get(col, []) or [])
        invalid_category_count = None
        invalid_examples = []

        if valid_set:
            candidates = non_null[~unknown_mask]  # exclude unknowns from invalid calc
            invalid_mask = ~candidates.isin(valid_set)
            invalid_category_count = int(invalid_mask.sum())

            if invalid_category_count > 0:
                invalid_examples = (
                    candidates[invalid_mask]
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )

        per_column[col] = {
            "nan_count": nan_count,
            "unique_count": unique_count,
            "top_5_values": top_5_values,
            "whitespace_count": whitespace_count,
            "unknown_values_count": unknown_values_count,
            "invalid_category_count": invalid_category_count,
            "invalid_examples": invalid_examples,
        }

    return {
        "columns": list(per_column.keys()),
        "per_column": per_column,
        "flags": flags,
    }


# logical_consistency_check_dependent_columns
def logical_rules_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """
    Rules checked (if relevant columns exist):
      1) Internet dependency:
         - If InternetService == "No" then each dependent col must be "No internet service"
      2) Phone dependency:
         - If PhoneService == "No" then each dependent col must be "No phone service"

    Returns JSON-serializable dict:
      {
        "violations_by_rule": {rule_name: count, ...},
        "examples_by_rule": {rule_name: [ {id:..., parent:..., dependent:..., actual:...}, ... ] }
      }
    """
    cfg = cfg or ETLConfig()

    id_col = getattr(cfg, "id_column", "customerID")

    violations_by_rule: Dict[str, int] = {}
    examples_by_rule: Dict[str, Any] = {}

    # Helper: store up to N examples per rule
    def _add_examples(rule_name: str, df_rows: pd.DataFrame, parent_col: str, dependent_col: str, max_examples: int = 10) -> None:
        if df_rows.empty:
            return
        if rule_name not in examples_by_rule:
            examples_by_rule[rule_name] = []
        current = examples_by_rule[rule_name]
        remaining = max_examples - len(current)
        if remaining <= 0:
            return

        cols_to_take = []
        if id_col in df_rows.columns:
            cols_to_take.append(id_col)
        if parent_col in df_rows.columns:
            cols_to_take.append(parent_col)
        if dependent_col in df_rows.columns:
            cols_to_take.append(dependent_col)   
        if not cols_to_take:
            return     

        sample = df_rows[cols_to_take].head(remaining).copy()

        # Build JSON-safe example objects
        for _, row in sample.iterrows():
            ex = {
                "customerID": str(row[id_col]) if id_col in row and not pd.isna(row[id_col]) else None,
                "parent_column": parent_col,
                "parent_value": None if parent_col not in row or pd.isna(row[parent_col]) else str(row[parent_col]),
                "dependent_column": dependent_col,
                "actual_value": None if dependent_col not in row or pd.isna(row[dependent_col]) else str(row[dependent_col]),
            }
            current.append(ex)

    # -------- Rule 1: Internet dependency --------
    internet_parent = getattr(cfg, "internet_parent", "InternetService")
    internet_dependents = tuple(getattr(cfg, "internet_dependents", ()))

    if internet_parent in df.columns and internet_dependents:
        parent_series = df[internet_parent].astype("string")
        no_internet_mask = parent_series.eq("No")  # raw membership check (case-sensitive)

        if no_internet_mask.any():
            for dep in internet_dependents:
                if dep not in df.columns:
                    continue

                dep_series = df[dep].astype("string")
                # violation: parent == "No" AND dependent != "No internet service" (excluding NaN? count as violation)
                # count NaN as violation because it's not the expected dependent value when parent says No.
                viol_mask = no_internet_mask & (~dep_series.eq("No internet service"))

                rule_name = f"internet_dependency::{dep}"
                viol_count = int(viol_mask.sum())
                if viol_count:
                    violations_by_rule[rule_name] = viol_count
                    cols = [c for c in [id_col, internet_parent, dep] if c in df.columns]
                    _add_examples(rule_name, df.loc[viol_mask, cols], internet_parent, dep)

    # -------- Rule 2: Phone dependency --------
    phone_parent = getattr(cfg, "phone_parent", "PhoneService")
    phone_dependents = tuple(getattr(cfg, "phone_dependents", ()))

    if phone_parent in df.columns and phone_dependents:
        parent_series = df[phone_parent].astype("string")
        no_phone_mask = parent_series.eq("No")

        if no_phone_mask.any():
            for dep in phone_dependents:
                if dep not in df.columns:
                    continue

                dep_series = df[dep].astype("string")
                # violation: parent == "No" AND dependent != "No phone service" (count NaN as violation)
                viol_mask = no_phone_mask & (~dep_series.eq("No phone service"))

                rule_name = f"phone_dependency::{dep}"
                viol_count = int(viol_mask.sum())
                if viol_count:
                    cols = [c for c in [id_col, phone_parent, dep] if c in df.columns]
                    _add_examples(rule_name, df.loc[viol_mask, cols], phone_parent, dep)

    return {
        "violations_by_rule": violations_by_rule,
        "examples_by_rule": examples_by_rule,
    }

# Churn Lable check
def label_check(df: pd.DataFrame, cfg: Optional[ETLConfig] = None) -> Dict[str, Any]:
    """
    Label (target) DQ metrics:
      - counts distribution (Yes/No/etc.)
      - percentage distribution
      - nan_count
    """
    cfg = cfg or ETLConfig()
    target_col = getattr(cfg, "target_column", "Churn")

    if target_col not in df.columns:
        return {
            "target_column": target_col,
            "present": False,
            "nan_count": None,
            "counts": {},
            "pct": {},
        }

    s = df[target_col].astype("string")

    nan_count = int(s.isna().sum())

    # counts excluding NaN
    counts_series = s.dropna().value_counts()
    counts = {str(k): int(v) for k, v in counts_series.to_dict().items()}

    total_non_null = int(counts_series.sum())
    if total_non_null == 0:
        pct = {}
    else:
        pct = {k: float(v / total_non_null) for k, v in counts.items()}

    return {
        "target_column": target_col,
        "present": True,
        "nan_count": nan_count,
        "counts": counts,
        "pct": pct,
    }


def run_all_checks(df: pd.DataFrame, cfg: Optional[ETLConfig]= None, dataset_name: str = "before_etl") -> Dict[str, Any]:
    cfg = cfg or ETLConfig()
    df_dq = df.copy()

    dq_report = {
        "meta": meta_check(df_dq, dataset_name = dataset_name, cfg=cfg),
        "dataset": dataset_check(df_dq, cfg=cfg),
        "missingness": missingness_check(df_dq, cfg=cfg),
        "duplicates": duplicate_check(df_dq, cfg = cfg),
        "numerical_anomalies": numeric_check(df_dq, cfg=cfg),
        "categorical_anomalies": categorical_check(df_dq, cfg=cfg),
        "logical_rules_anomalies": logical_rules_check(df_dq, cfg=cfg),
        "label_distribution": label_check(df_dq, cfg=cfg)
    }

    return dq_report
