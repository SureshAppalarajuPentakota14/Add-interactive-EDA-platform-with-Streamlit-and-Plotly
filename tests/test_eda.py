import io
import numpy as np
import pandas as pd
import pytest

import eda


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 100, np.nan],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "c": ["x", "y", "x", None, "y", "x"],
    })


def test_overview(df):
    ov = eda.dataset_overview(df)
    assert ov["rows"] == 6 and ov["columns"] == 3 and ov["missing_cells"] == 2


def test_missing_report(df):
    r = eda.missing_report(df)
    assert set(r.column) == {"a", "c"}
    assert r.missing_pct.iloc[0] == pytest.approx(16.67, abs=0.01)


def test_duplicates():
    d = pd.DataFrame({"x": [1, 1, 2]})
    assert eda.duplicate_report(d)["duplicate_rows"] == 1


def test_outliers_iqr(df):
    s = eda.outlier_summary(df)
    assert s.loc[s.column == "a", "outliers"].iloc[0] == 1


def test_correlation(df):
    corr = eda.correlation_matrix(df)
    assert corr.shape == (2, 2)
    assert len(eda.top_correlations(corr)) == 1


def test_health_score_range(df):
    h = eda.health_score(df)
    assert 0 <= h["score"] <= 100


def test_loader_csv_and_dates():
    csv = "d,v\n2024-01-01,1\n2024-02-01,2\n2024-03-01,3\n"
    out = eda.load_dataset(io.BytesIO(csv.encode()), filename="t.csv")
    assert pd.api.types.is_datetime64_any_dtype(out["d"])


def test_loader_rejects_bad_ext():
    with pytest.raises(ValueError):
        eda.load_dataset(io.BytesIO(b""), filename="x.txt")
