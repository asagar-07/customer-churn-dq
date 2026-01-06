from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    # Tenure bin edges and labels (Telco tenure is 0–72)
    tenure_bins: Tuple[int, ...] = (0, 6, 12, 24, 48, 72)
    tenure_labels: Tuple[str, ...] = ("0-6", "7-12", "13-24", "25-48", "49-72")
    
    tenure_col = "tenure"
    monthly_col = "MonthlyCharges"
    total_col = "TotalCharges"

    # New feature names
    tenure_bin_col = "TenureBin"
    avg_monthly_from_total_col = "AvgMonthlyFromTotal"


def add_tenure_bins(df: pd.DataFrame, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """ Adds a categorical bin feature from tenure. Does NOT mutate input df."""
    cfg = cfg or FeatureConfig()
    out = df.copy()

    if cfg.tenure_col not in out.columns:
        return out

    # pd.cut needs one more edge than labels -> adding +inf as guard
    edges = list(cfg.tenure_bins) + [np.inf]
    labels = list(cfg.tenure_labels) + [f">{cfg.tenure_bins[-1]}"]

    tenure_num = pd.to_numeric(out[cfg.tenure_col], errors="coerce")

    out[cfg.tenure_bin_col] = pd.cut(
        tenure_num,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("string")

    return out


def add_avg_monthly_from_total(df: pd.DataFrame, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Adds AvgMonthlyFromTotal = TotalCharges / max(tenure, 1). Does NOT mutate input df. """
    cfg = cfg or FeatureConfig()
    out = df.copy()

    required = {cfg.tenure_col, cfg.total_col}
    if not required.issubset(out.columns):
        return out

    tenure = pd.to_numeric(out[cfg.tenure_col], errors="coerce")
    total = pd.to_numeric(out[cfg.total_col], errors="coerce")

    denominator = tenure.copy()
    denominator = denominator.mask(denominator < 1, 1)  # avoid divide-by-zero and tiny tenure
    out[cfg.avg_monthly_from_total_col] = (total / denominator).astype("float64")

    return out


def build_features(df: pd.DataFrame, add_tenure_bin: bool = False, add_avg_monthly: bool = False, cfg: Optional[FeatureConfig] = None, ) -> pd.DataFrame:
    """Feature builder for v0.6 experiments."""
    cfg = cfg or FeatureConfig()
    out = df.copy()

    if add_tenure_bin:
        out = add_tenure_bins(out, cfg)

    if add_avg_monthly:
        out = add_avg_monthly_from_total(out, cfg)

    return out