"""Outlier profiling using IQR fences and z-scores."""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def detect_outliers_zscore(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    std = s.std()
    if not std or np.isnan(std):
        return pd.Series(False, index=s.index)
    return ((s - s.mean()) / std).abs() > threshold


def outlier_summary(df: pd.DataFrame, method: str = "iqr", k: float = 1.5,
                    z_threshold: float = 3.0) -> pd.DataFrame:
    """Per numeric column outlier counts, bounds and percent."""
    rows = []
    for c in df.select_dtypes("number").columns:
        s = df[c].dropna()
        if s.nunique() < 3 or s.empty:
            continue
        mask = detect_outliers_iqr(s, k) if method == "iqr" else detect_outliers_zscore(s, z_threshold)
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        rows.append({
            "column": c, "outliers": int(mask.sum()),
            "outlier_pct": round(mask.mean() * 100, 2),
            "lower_bound": q1 - k * iqr, "upper_bound": q3 + k * iqr,
            "min": s.min(), "max": s.max(),
        })
    cols = ["column", "outliers", "outlier_pct", "lower_bound", "upper_bound", "min", "max"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("outlier_pct", ascending=False).reset_index(drop=True)
