"""Correlation matrix + strongest pairs."""
from __future__ import annotations
import numpy as np
import pandas as pd


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method)


def top_correlations(corr: pd.DataFrame, n: int = 10, min_abs: float = 0.0) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    iu = np.triu_indices_from(corr, k=1)
    pairs = pd.DataFrame({
        "feature_1": corr.index[iu[0]], "feature_2": corr.columns[iu[1]],
        "correlation": corr.values[iu],
    }).dropna()
    pairs = pairs[pairs.correlation.abs() >= min_abs]
    return (pairs.reindex(pairs.correlation.abs().sort_values(ascending=False).index)
            .head(n).reset_index(drop=True))
