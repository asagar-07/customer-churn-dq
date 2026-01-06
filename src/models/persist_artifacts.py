# src/models/persist_artifacts.py
from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_version(pkg_name: str) -> Optional[str]:
    try:
        import importlib.metadata as md  
        return md.version(pkg_name)
    except Exception:
        return None

def _json_default(o: Any) -> Any:
    """Make metadata JSON-serializable."""
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()  
    return str(o)

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_artifacts(
    *,
    model: Any,
    scaler: Any,
    encoder: Any,
    feature_names: List[str],
    threshold: float,
    out_dir: Union[str, Path] = "models",
    run_name: str = "v0_6",
    label_positive: str = "Yes",
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Persist model + preprocessors + metadata.

    Writes:
      - <out_dir>/lr_<run_name>.joblib
      - <out_dir>/preprocess_<run_name>.joblib
      - <out_dir>/model_card_<run_name>.json

    Returns a dict of paths written.
    """
    out_dir = _ensure_dir(out_dir)

    model_path = out_dir / f"lr_{run_name}.joblib"
    preprocess_path = out_dir / f"preprocess_{run_name}.joblib"
    card_path = out_dir / f"model_card_{run_name}.json"

    # 1. Save model
    joblib.dump(model, model_path)

    # 2. Save preprocess bundle (keep everything needed for inference alignment)
    preprocess_bundle = {
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": list(feature_names),
        "threshold": float(threshold),
        "label_positive": label_positive,
    }
    joblib.dump(preprocess_bundle, preprocess_path)

    # 3. Save a lightweight model card (json)
    card: Dict[str, Any] = {
        "run_name": run_name,
        "created_utc": _utc_now_iso(),
        "paths": {
            "model": str(model_path),
            "preprocess": str(preprocess_path),
        },
        "decision": {
            "threshold": float(threshold),
            "positive_label": label_positive,
        },
        "schema": {
            "n_features": int(len(feature_names)),
            "feature_names": list(feature_names),
        },
        "metrics": metrics or {},
        "config": _json_default(cfg) if cfg is not None else {},
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": _safe_version("scikit-learn"),
            "pandas": _safe_version("pandas"),
            "numpy": _safe_version("numpy"),
            "joblib": _safe_version("joblib"),
        },
    }
    if extra_metadata:
        card["extra"] = extra_metadata

    card_path.write_text(json.dumps(card, indent=2, default=_json_default))

    return {
        "model_path": str(model_path),
        "preprocess_path": str(preprocess_path),
        "card_path": str(card_path),
    }

def load_artifacts(
    *,
    out_dir: Union[str, Path] = "models",
    run_name: str = "v0_6",
) -> Tuple[Any, Any, Any, List[str], float, Dict[str, Any]]:
    """
    Load model + preprocessors + metadata.

    Returns:
      (model, scaler, encoder, feature_names, threshold, model_card_dict)
    """
    out_dir = Path(out_dir)

    model_path = out_dir / f"lr_{run_name}.joblib"
    preprocess_path = out_dir / f"preprocess_{run_name}.joblib"
    card_path = out_dir / f"model_card_{run_name}.json"

    model = joblib.load(model_path)
    bundle = joblib.load(preprocess_path)

    scaler = bundle["scaler"]
    encoder = bundle["encoder"]
    feature_names = list(bundle["feature_names"])
    threshold = float(bundle.get("threshold", 0.5))

    card: Dict[str, Any] = {}
    if card_path.exists():
        card = json.loads(card_path.read_text())

    return model, scaler, encoder, feature_names, threshold, card