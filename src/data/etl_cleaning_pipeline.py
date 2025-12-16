"""
ETL Cleaning Pipeline for Customer Churn Data
This module provides functions to load messy customer churn data,
clean it through various ETL steps, and save the cleansed data.
Design goals:
- Pure functions (no file I/O inside core functions and transformations)
- Modular functions for each cleaning step
- Flag every fix in an ETL log (counts + examples)
- Clear documentation for maintainability
- Handle common data quality issues: missing values, inconsistent formats, logical inconsistencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
#from word2number import w2n as w2n  


# Config columns

@dataclass(frozen=True)
class ETLConfig:
    id_column: str = 'customerID'
    target_column: str = 'Churn'

    numeric_columns: Tuple[str, ...] = ('tenure', 'MonthlyCharges', 'TotalCharges')
    categorical_columns: Tuple[str, ...] = ('gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod')
    
    #normalize unknown tokens to Nan (after whitespace trimming)
    unknown_tokens: Tuple[str, ...] = ('UNKNOWN', 'Unknown', 'unknown', 'N/A', 'na', 'n/a','NOT APPLICABLE', 'Not Applicable','Not provided', 'not provided')

    #consistent values mapping for categorical normalization
    inconsistent_values: Dict[str, List[str]] = field(default_factory=lambda: {
        'Female': ['F', 'f', '0'],
        'Male': ['M', 'm', '1'],
        'Yes': ['YES','Y', 'y', '1'],
        'No': ['NO', 'N', 'n', '0'],
        'No phone service': ['No Phone Service', 'no phone service', 'NPS'],
        'No internet service': ['No Internet Service', 'no internet service', 'NIS'],
        'Month-to-month': ['Monthly', 'month to month','m2m' ,'M2M'], 
        'One year': ['1 year', 'One Yr', '1 Yr'],
        'Two year': ['2 year', 'Two Yr', '2 Yr'],
        'Fiber optic': ['Fiber', 'fiber', 'FO', 'Fiber-Optik'],
        'Electronic check': ['E-check', 'E Check', 'Electronic Payment'],
        'Mailed check': ['Mailed Check', 'Check by Mail','Paper Check'],
        'Bank transfer (automatic)': ['Bank Transfer', 'Direct Debit'],
        'Credit card (automatic)': ['Credit Card', 'CC']
    })  

    # Logical dependencies between columns
    internet_parent: str = "InternetService"
    internet_dependents: Tuple[str, ...] = (
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies")
    phone_parent: str = "PhoneService"
    phone_dependents: Tuple[str, ...] = ("MultipleLines")

    #valid value sets used in DQ checks
    valid_values: Dict[str, List[str]] = field(default_factory=lambda: {
        'gender': ['Female', 'Male'],
        'SeniorCitizen': ['Yes', 'No'], # should be 0, 1 but keeping as Yes/No for consistency
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'Churn': ['Yes', 'No']
    })

    #Numeric sanity rules, convert invalid to Nan and impute later if needed
    enforce_non_negative: Tuple[str, ...] = ('tenure', 'MonthlyCharges', 'TotalCharges')

# ------------------------------------
# ETL Log Helpers (Fix + Flag)
# ------------------------------------


def _log_init() -> Dict[str, Any]:
    """Initialize an ETL log dictionary."""
    return {
        'row_count_in': None,
        'row_count_out': None,
        'changes': {},   # step -> {column -> count}
        'examples': {},  # step -> {column -> [examples]}
        "dropped_rows": 0
    }

def _bump(log: Dict[str, Any], step: str, column: str, count: int, examples: Optional[List[Any]] = None) -> None:
    """Update the ETL log with changes made during a cleaning step."""
    log['changes'].setdefault(step, {})
    log['changes'][step][column] = log['changes'][step].get(column, 0) + int(count)

    if examples:
        log['examples'].setdefault(step, {})
        log['examples'][step].setdefault(column, [])
        # store up to 10 examples per column per step
        for example in examples:
            if len(log['examples'][step][column]) >= 10:
                break
            if example not in log['examples'][step][column]:
                log['examples'][step][column].append(example)


# ------------------------------------
# Core ETL Cleaning Pipeline Functions 
# ------------------------------------

# trim whitespace from all columns
def trim_whitespace(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    """Trim leading and trailing whitespace from columns in the DataFrame."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        before = out[col].copy()
        out[col] = out[col].astype("string").str.strip()
        changed = (before.astype("string") != out[col]).sum()
        if changed:
            _bump(log, 'trim_whitespace', col, changed)
    return out


# Normalize UNKNOWN entries to NaN
def normalize_unknown_tokens(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    unknown_lower = {u.lower() for u in cfg.unknown_tokens}

    for col in out.columns:
        s = out[col]
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            s_str = s.astype("string")
            mask = s_str.str.lower().isin(unknown_lower)
            if mask.any():
                examples = s_str[mask].dropna().unique().tolist()[:10]
                out.loc[mask, col] = np.nan
                _bump(log, "normalize_unknown_tokens", col, int(mask.sum()), examples)
    return out

def _build_variant_to_canonical_map(cfg: ETLConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canonical, variants in cfg.inconsistent_values.items():
        mapping[canonical] = canonical
        for v in variants:
            mapping[v] = canonical
    return mapping

def normalize_categorical_variants(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    mapping = _build_variant_to_canonical_map(cfg)

    cols = [c for c in cfg.categorical_columns if c in out.columns]
    # also normalize target if present
    if cfg.target_column in out.columns:
        cols = cols + [cfg.target_column]

    for col in cols:
        s = out[col]
        if not (s.dtype == "object" or str(s.dtype).startswith("string") or str(s.dtype).startswith("category")):
            continue

        before = s.astype("string")
        after = before.map(lambda x: mapping.get(x, x))
        changed = (before != after).sum()
        if changed:
            examples = before[before != after].dropna().unique().tolist()[:10]
            out[col] = after
            _bump(log, "normalize_categorical_variants", col, int(changed), examples)
    return out


def coerce_numeric(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    for col in cfg.numeric_columns:  
        if col not in out.columns:
            continue

        before = out[col].copy()

        # Numeric interpretation of "before" 
        before_num = pd.to_numeric(before, errors="coerce")

        # Convert to numeric; invalids become NaN
        coerced = pd.to_numeric(out[col], errors="coerce")

        # log infinities then replace 
        inf_mask = coerced.isin([np.inf, -np.inf])
        #print("inf_mask count for col:", col, inf_mask.sum())
        if inf_mask.any():
            examples = before[inf_mask].head(10).tolist()
            _bump(log, "infinite_to_nan", col, int(inf_mask.sum()), examples)
        coerced = coerced.replace([np.inf, -np.inf], np.nan)

        #  Enforce non-negative rule 
        if col in cfg.enforce_non_negative:
            neg_mask = coerced.notna() & (coerced < 0)
            #print("neg_mask count for col: ",col, neg_mask.sum())
            if neg_mask.any():
                examples = coerced[neg_mask].head(10).tolist()
                coerced = coerced.mask(neg_mask, np.nan)
                _bump(log, "numeric_negative_to_nan", col, int(neg_mask.sum()), examples)

        # Value change (numeric meaning changed), treating NaN==NaN as equal
        # True when:
        # - both not NaN and values differ
        # - one side NaN and the other not NaN
        a = before_num.to_numpy()
        b = coerced.to_numpy()

        # both NaN -> equal
        both_nan = np.isnan(a) & np.isnan(b)

        # one NaN and the other not -> changed
        one_nan = np.isnan(a) ^ np.isnan(b)

        # both not NaN and values differ -> changed
        diff_vals = (~np.isnan(a)) & (~np.isnan(b)) & (a != b)

        value_changed_mask = one_nan | diff_vals
        ###value_changed_mask = ~((before_num.eq(coerced)) | (before_num.isna() & coerced.isna()))
        #print("value_changed_mask count for col: ", col, value_changed_mask.sum())
        
        value_changed_count = int(value_changed_mask.sum())
        if value_changed_count:
            examples = before[pd.Series(value_changed_mask, index=before.index)].head(10).tolist()
            _bump(log, "numeric_value_corrected", col, value_changed_count, examples)
        
        # Format-only normalization:
        # numeric meaning same (including both NaN) BUT representation changed
        # Detect representation differences by comparing strings of the original vs the cleaned.
        before_str = before.astype("string")
        after_str = coerced.astype("string")

        repr_changed_mask = (before_str != after_str)

        # same numeric meaning (including both NaN)
        same_numeric_mask = (before_num.eq(coerced)) | (before_num.isna() & coerced.isna())

        format_only_mask = same_numeric_mask & repr_changed_mask

        format_only_count = int(format_only_mask.sum())
        if format_only_count:
            examples = before[format_only_mask].head(10).tolist()
            _bump(log, "numeric_format_normalized", col, format_only_count, examples)

        # count only true value changes (not formatting)
        if value_changed_count:
            _bump(log, "coerce_numeric", col, value_changed_count)

        out[col] = coerced

    return out

def fix_logical_inconsistencies(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    """
    - If InternetService == 'No' => dependents must be 'No internet service'
    - If PhoneService == 'No' => MultipleLines must be 'No phone service'
    """
    out = df.copy()

    # InternetService dependencies
    if cfg.internet_parent in out.columns:
        parent = cfg.internet_parent
        for dep in cfg.internet_dependents:
            if dep not in out.columns:
                continue
            mask = (out[parent] == "No") & (out[dep] == "Yes")
            if mask.any():
                out.loc[mask, dep] = "No internet service"
                _bump(log, "fix_logical_inconsistency_internet", dep, int(mask.sum()), examples=["Yes -> No internet service"])

            # Also if parent == No, any non-null not-equal "No internet service" should be set to "No internet service"
            mask2 = (out[parent] == "No") & out[dep].notna() & (out[dep] != "No internet service")
            # Avoid double counting from mask above
            mask2 = mask2 & ~mask
            if mask2.any():
                before_vals = out.loc[mask2, dep].astype("string").unique().tolist()[:10]
                out.loc[mask2, dep] = "No internet service"
                _bump(log, "force_internet_dependents_when_no", dep, int(mask2.sum()), before_vals)

    # PhoneService dependencies
    if cfg.phone_parent in out.columns:
        parent = cfg.phone_parent
        for dep in cfg.phone_dependents:
            if dep not in out.columns:
                continue
            mask = (out[parent] == "No") & (out[dep] == "Yes")
            if mask.any():
                out.loc[mask, dep] = "No phone service"
                _bump(log, "fix_logical_inconsistency_phone", dep, int(mask.sum()), examples=["Yes -> No phone service"])

            mask2 = (out[parent] == "No") & out[dep].notna() & (out[dep] != "No phone service")
            mask2 = mask2 & ~mask
            if mask2.any():
                before_vals = out.loc[mask2, dep].astype("string").unique().tolist()[:10]
                out.loc[mask2, dep] = "No phone service"
                _bump(log, "force_phone_dependents_when_no", dep, int(mask2.sum()), before_vals)

    return out


def flag_duplicates(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if cfg.id_column not in out.columns:
        return out

    counts = out.groupby(cfg.id_column)[cfg.id_column].transform("count")
    out["id_count"] = counts
    out["is_duplicate"] = counts > 1

    dup_count = int((out["is_duplicate"]).sum())
    if dup_count:
        _bump(log, "flag_duplicates", cfg.id_column, dup_count)

    return out


def enforce_valid_values(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any], drop_invalid_rows: bool = False) -> pd.DataFrame:
    """
    - Always flag invalid values
    - Either:
      * drop_invalid_rows=False: set invalid to NaN (recommended)
      * drop_invalid_rows=True : drop rows with invalid values (flag count + dropped)
    """
    out = df.copy()

    for col, allowed in cfg.valid_values.items():
        if col not in out.columns:
            continue

        s = out[col]
        # treat NaN as allowed (we handle missing later)
        mask_invalid = s.notna() & (~s.isin(allowed))
        if not mask_invalid.any():
            continue

        examples = s[mask_invalid].astype("string").unique().tolist()[:10]
        _bump(log, "invalid_category_values", col, int(mask_invalid.sum()), examples)

        if drop_invalid_rows:
            before_rows = len(out)
            out = out.loc[~mask_invalid].copy()
            dropped = before_rows - len(out)
            log["dropped_rows"] += int(dropped)
        else:
            out.loc[mask_invalid, col] = np.nan

    return out


def impute_missing(df: pd.DataFrame, cfg: ETLConfig, log: Dict[str, Any]) -> pd.DataFrame:
    """
    Simple imputation (baseline):
    - numeric: median
    - categorical: mode
    Note: D this AFTER train/test split to avoid leakage, for better ML practice
    """
    out = df.copy()

    for col in cfg.numeric_columns:
        if col in out.columns:
            med = out[col].median()
            if pd.notna(med):
                missing = out[col].isna().sum()
                if missing:
                    out[col] = out[col].fillna(med)
                    _bump(log, "impute_numeric_median", col, int(missing), examples=[med])

    # include target too only if you truly want a fully-filled dataset artifact
    cat_cols = [c for c in cfg.categorical_columns if c in out.columns]
    if cfg.target_column in out.columns:
        cat_cols.append(cfg.target_column)

    for col in cat_cols:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            fill = mode.iloc[0]
            missing = out[col].isna().sum()
            if missing:
                out[col] = out[col].fillna(fill)
                _bump(log, "impute_categorical_mode", col, int(missing), examples=[fill])

    return out


def run_etl(df: pd.DataFrame, cfg: Optional[ETLConfig]= None, *, drop_invalid_rows: bool = False, do_impute: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = cfg or ETLConfig()
    log = _log_init()
    log["row_count_in"] = int(len(df))

    df_etl = df.copy()


    df_etl = trim_whitespace(df_etl, cfg, log)
    df_etl = normalize_unknown_tokens(df_etl, cfg, log)
    df_etl = normalize_categorical_variants(df_etl, cfg, log)
    df_etl = coerce_numeric(df_etl, cfg, log)
    df_etl = fix_logical_inconsistencies(df_etl, cfg, log)
    df_etl = enforce_valid_values(df_etl, cfg, log, drop_invalid_rows=drop_invalid_rows)
    df_etl = flag_duplicates(df_etl, cfg, log)

    if do_impute:
        df_etl = impute_missing(df_etl, cfg, log)

    log["row_count_out"] = int(len(df_etl))
    return df_etl, log    
