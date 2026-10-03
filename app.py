"""Interactive EDA Platform - run with:  streamlit run app.py"""
from __future__ import annotations
import io
from pathlib import Path

import pandas as pd
import streamlit as st

import eda
from eda import plots

st.set_page_config(page_title="Interactive EDA Platform", page_icon="📊", layout="wide")

SAMPLE = Path(__file__).parent / "data" / "sample_retail_orders.csv"


@st.cache_data(show_spinner="Reading file...")
def _load(raw: bytes, name: str, sheet) -> pd.DataFrame:
    return eda.load_dataset(io.BytesIO(raw), filename=name, sheet_name=sheet)


def _excel_sheets(raw: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 EDA Platform")
upload = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "tsv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=upload is None and SAMPLE.exists())

df, source = None, ""
if upload is not None:
    raw = upload.getvalue()
    sheet = 0
    if upload.name.lower().endswith((".xlsx", ".xls")):
        sheet = st.sidebar.selectbox("Sheet", _excel_sheets(raw))
    try:
        df, source = _load(raw, upload.name, sheet), upload.name
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
elif use_sample and SAMPLE.exists():
    df, source = _load(SAMPLE.read_bytes(), SAMPLE.name, 0), f"{SAMPLE.name} (sample)"

if df is None:
    st.title("Interactive Exploratory Data Analysis Platform")
    st.info("Upload a CSV or Excel file in the sidebar to generate an instant data-health report.")
    st.stop()

num_cols = df.select_dtypes("number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]
date_cols = df.select_dtypes("datetime").columns.tolist()

st.title("Interactive EDA Platform")
st.caption(f"Source: **{source}**")

tabs = st.tabs(["🩺 Health Report", "📋 Profile", "🎯 Outliers", "🔗 Correlations", "📈 Visual Explorer", "🧹 Clean & Export"])

# ------------------------------------------------------------------ health
with tabs[0]:
    ov, hs = eda.dataset_overview(df), eda.health_score(df)
    c = st.columns(6)
    c[0].metric("Rows", f"{ov['rows']:,}")
    c[1].metric("Columns", ov["columns"])
    c[2].metric("Missing cells", f"{ov['missing_cells']:,}")
    c[3].metric("Duplicate rows", ov["duplicate_rows"])
    c[4].metric("Memory (MB)", ov["memory_mb"])
    c[5].metric("Health score", f"{hs['score']} ({hs['grade']})")
    st.progress(min(int(hs["score"]), 100) / 100)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Missing values")
        miss = eda.missing_report(df)
        if miss.empty:
            st.success("No missing values 🎉")
        else:
            st.plotly_chart(plots.missing_bar(miss), width="stretch")
            st.dataframe(miss,use_container_width=True, hide_index=True)
    with right:
        st.subheader("Missingness map")
        if miss.empty:
            st.write("Nothing to show.")
        else:
            st.plotly_chart(plots.missing_heatmap(df), width="stretch")

    st.subheader("Duplicates")
    dups = eda.duplicate_report(df)
    st.write(f"**{dups['duplicate_rows']}** duplicate rows ({dups['rows_involved']} rows involved).")
    if dups["duplicate_rows"]:
        st.dataframe(dups["sample"], width=700, hide_index=True)

# ------------------------------------------------------------------ profile
with tabs[1]:
    st.subheader("Column profile")
    st.dataframe(eda.column_profile(df), use_container_width=True, hide_index=True)
    st.subheader("Data preview")
    n = st.slider("Rows to preview", 5, 200, 20)
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ outliers
with tabs[2]:
    if not num_cols:
        st.warning("No numeric columns found.")
    else:
        c1, c2, c3 = st.columns(3)
        method = c1.radio("Method", ["iqr", "zscore"], horizontal=True)
        k = c2.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, disabled=method != "iqr")
        z = c3.slider("Z threshold", 2.0, 5.0, 3.0, 0.1, disabled=method != "zscore")
        summ = eda.outlier_summary(df, method=method, k=k, z_threshold=z)
        st.plotly_chart(plots.outlier_bar(summ), use_container_width=True)
        st.dataframe(summ, use_container_width=True, hide_index=True)

        col = st.selectbox("Inspect column", summ.column.tolist() or num_cols)
        s = df[col].dropna()
        mask = eda.detect_outliers_iqr(s, k) if method == "iqr" else eda.detect_outliers_zscore(s, z)
        st.write(f"**{int(mask.sum())}** outliers in `{col}`")
        st.plotly_chart(plots.box(df, col), use_container_width=True)
        if mask.any():
            st.dataframe(df.loc[s[mask].index], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ correlation
with tabs[3]:
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = eda.correlation_matrix(df, method)
        st.plotly_chart(plots.correlation_heatmap(corr), use_container_width=True)
        st.subheader("Strongest relationships")
        st.dataframe(eda.top_correlations(corr, n=10), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ explorer
with tabs[4]:
    kind = st.selectbox("Chart type", ["Histogram", "Box plot", "Scatter", "Category counts", "Time series"])
    color = st.selectbox("Colour by (optional)", [None] + cat_cols)
    if kind == "Histogram" and num_cols:
        col = st.selectbox("Column", num_cols)
        st.plotly_chart(plots.histogram(df, col, color, bins=st.slider("Bins", 5, 100, 40)),
                        use_container_width=True)
    elif kind == "Box plot" and num_cols:
        y = st.selectbox("Value", num_cols)
        x = st.selectbox("Group by (optional)", [None] + cat_cols)
        st.plotly_chart(plots.box(df, y, x, color), use_container_width=True)
    elif kind == "Scatter" and len(num_cols) >= 2:
        x = st.selectbox("X", num_cols, 0)
        y = st.selectbox("Y", num_cols, 1)
        size = st.selectbox("Size (optional)", [None] + num_cols)
        st.plotly_chart(plots.scatter(df, x, y, color, size, st.checkbox("Trendline", True)),
                        use_container_width=True)
    elif kind == "Category counts" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        st.plotly_chart(plots.category_bar(df, col), use_container_width=True)
    elif kind == "Time series" and date_cols and num_cols:
        d = st.selectbox("Date column", date_cols)
        v = st.selectbox("Value", num_cols)
        agg = st.selectbox("Aggregation", ["sum", "mean", "count"])
        freq = st.selectbox("Frequency", ["D", "W", "ME", "QE"], 2)
        st.plotly_chart(plots.time_series(df, d, v, freq, agg), use_container_width=True)
    else:
        st.info("This chart type needs columns your dataset doesn't have.")

# ------------------------------------------------------------------ clean
with tabs[5]:
    st.subheader("Quick cleaning")
    work = df.copy()
    if st.checkbox("Drop duplicate rows"):
        work = work.drop_duplicates()
    fill = st.selectbox("Missing numeric values", ["Leave as is", "Fill with median", "Fill with mean", "Drop rows"])
    if fill == "Fill with median":
        work[num_cols] = work[num_cols].fillna(work[num_cols].median())
    elif fill == "Fill with mean":
        work[num_cols] = work[num_cols].fillna(work[num_cols].mean())
    elif fill == "Drop rows":
        work = work.dropna(subset=num_cols)
    if st.checkbox("Cap outliers (IQR winsorise)") and num_cols:
        for c in num_cols:
            q1, q3 = work[c].quantile([.25, .75])
            i = q3 - q1
            work[c] = work[c].clip(q1 - 1.5 * i, q3 + 1.5 * i)

    st.write(f"Result: **{len(work):,}** rows x **{work.shape[1]}** columns "
             f"(health score {eda.health_score(work)['score']})")
    st.dataframe(work.head(50), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download cleaned CSV", work.to_csv(index=False).encode(),
                       "cleaned_data.csv", "text/csv")

    report = ["# Data Health Report", f"Source: {source}", "", "## Overview"]
    report += [f"- {k}: {v}" for k, v in eda.dataset_overview(df).items()]
    report += ["", "## Health score", f"{hs['score']} (grade {hs['grade']})", "", "## Missing values",
               eda.missing_report(df).to_markdown(index=False) if not eda.missing_report(df).empty else "None"]
    st.download_button("⬇️ Download health report (Markdown)", "\n".join(report).encode(),
                       "health_report.md", "text/markdown")
