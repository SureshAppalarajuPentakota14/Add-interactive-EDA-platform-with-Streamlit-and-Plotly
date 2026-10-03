"""File loading with format detection and light type inference."""
from __future__ import annotations
import io
import warnings
import pandas as pd

SUPPORTED = (".csv", ".tsv", ".xlsx", ".xls")


def load_dataset(file, filename: str | None = None, sheet_name=0) -> pd.DataFrame:
    """Load a CSV/TSV/Excel file (path or file-like object) into a DataFrame.

    Tries to parse date-like object columns into datetimes automatically.
    """
    name = (filename or getattr(file, "name", str(file))).lower()
    if not name.endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, sheet_name=sheet_name)
    else:
        sep = "\t" if name.endswith(".tsv") else None
        if hasattr(file, "read"):
            raw = file.read()
            file = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode())
        try:
            df = pd.read_csv(file, sep=sep, engine="python")
        except UnicodeDecodeError:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, sep=sep, engine="python", encoding="latin-1")

    df.columns = [str(c).strip() for c in df.columns]
    return _infer_dates(df)


def _infer_dates(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Convert object columns to datetime when >= `threshold` of values parse."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty or sample.str.len().median() < 6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
