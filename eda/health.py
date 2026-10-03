"""Data health checks: overview, missing values, duplicates, profiling, score."""
from __future__ import annotations
import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        "numeric_cols": int(df.select_dtypes("number").shape[1]),
        "categorical_cols": int(df.select_dtypes(include=["object", "string", "category", "bool"]).shape[1]),
        "datetime_cols": int(df.select_dtypes("datetime").shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
    })


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count/percent, sorted worst first (only columns with gaps)."""
    miss = df.isna().sum()
    out = pd.DataFrame({
        "column": miss.index,
        "missing_count": miss.values,
        "missing_pct": (miss.values / max(len(df), 1) * 100).round(2),
    })
    out = out[out.missing_count > 0].sort_values("missing_pct", ascending=False).copy()
    out["severity"] = pd.cut(
        out.missing_pct, bins=[-0.1, 5, 20, 50, 100],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    return out.reset_index(drop=True)


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    mask = df.duplicated(subset=subset, keep=False)
    sort_cols = list(subset) if subset else list(df.columns[:1])
    return {
        "duplicate_rows": int(df.duplicated(subset=subset).sum()),
        "rows_involved": int(mask.sum()),
        "sample": df[mask].sort_values(by=sort_cols).head(20),
    }


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive profile for every column (numeric stats + categorical top value)."""
    rows = []
    for c in df.columns:
        s = df[c]
        row = {"column": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
               "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row.update(mean=s.mean(), std=s.std(), min=s.min(), q25=s.quantile(.25),
                       median=s.median(), q75=s.quantile(.75), max=s.max(),
                       skew=s.skew())
        elif pd.api.types.is_datetime64_any_dtype(s):
            row.update(min_date=s.min(), max_date=s.max())
        else:
            vc = s.value_counts(dropna=True)
            row.update(top=vc.index[0] if len(vc) else None,
                       top_freq=int(vc.iloc[0]) if len(vc) else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def health_score(df: pd.DataFrame) -> dict:
    """0-100 composite score. Penalises missingness, duplicates, constant columns."""
    n, m = df.shape
    if n == 0 or m == 0:
        return {"score": 0, "grade": "F", "components": {}}
    miss_pct = df.isna().sum().sum() / (n * m) * 100
    dup_pct = df.duplicated().sum() / n * 100
    const_pct = sum(df[c].nunique(dropna=True) <= 1 for c in df.columns) / m * 100

    score = round(max(0, 100 - min(miss_pct * 2, 50) - min(dup_pct * 2, 25) - min(const_pct, 25)), 1)
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    return {
        "score": score, "grade": grade,
        "components": {"missing_pct": round(miss_pct, 2), "duplicate_pct": round(dup_pct, 2),
                       "constant_col_pct": round(const_pct, 2)},
    }
