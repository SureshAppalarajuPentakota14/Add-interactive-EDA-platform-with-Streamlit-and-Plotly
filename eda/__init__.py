"""Reusable EDA building blocks used by the Streamlit app and the notebook."""
from .loader import load_dataset
from .health import (
    dataset_overview, missing_report, duplicate_report,
    column_profile, health_score, type_summary,
)
from .outliers import outlier_summary, detect_outliers_iqr, detect_outliers_zscore
from .correlation import correlation_matrix, top_correlations
from . import plots

__all__ = [
    "load_dataset", "dataset_overview", "missing_report", "duplicate_report",
    "column_profile", "health_score", "type_summary", "outlier_summary",
    "detect_outliers_iqr", "detect_outliers_zscore", "correlation_matrix",
    "top_correlations", "plots",
]
