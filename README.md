# 📊 Interactive EDA Platform

A web app that turns any raw **CSV / Excel** file into an instant **data-health report** with interactive Plotly charts — built so business stakeholders can understand a dataset without writing code.

**Stack:** Python · Streamlit · Pandas · NumPy · Plotly · Jupyter Notebook · pytest

## Features
| Tab | What it does |
|---|---|
| 🩺 Health Report | Rows/columns/memory, **0-100 health score + grade**, missing-value report with severity levels, missingness heat-map, duplicate detection |
| 📋 Profile | Per-column stats (mean, std, quartiles, skew, top value, uniqueness) and data preview |
| 🎯 Outliers | **IQR** or **Z-score** detection with adjustable thresholds, per-column outlier %, box plot, drill-down into the offending rows |
| 🔗 Correlations | Pearson / Spearman / Kendall matrix, strongest feature pairs |
| 📈 Visual Explorer | Histograms, box plots, scatter (with trendline), category counts, time series — colour/group/size by any column |
| 🧹 Clean & Export | Drop duplicates, impute (median/mean), winsorise outliers; download the cleaned CSV and a Markdown health report |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/generate_sample_data.py                    # optional: re-create demo data
streamlit run app.py
```
Upload your own file in the sidebar, or tick **Use sample dataset**.

## Project structure
```
p1-eda-platform/
├── app.py                      # Streamlit UI (6 tabs)
├── eda/                        # reusable, UI-independent analysis library
│   ├── loader.py               #   CSV/TSV/Excel loading + automatic date detection
│   ├── health.py               #   overview, missing, duplicates, profile, health score
│   ├── outliers.py             #   IQR & z-score detection
│   ├── correlation.py          #   correlation matrix + top pairs
│   └── plots.py                #   Plotly figure factories
├── notebooks/eda_walkthrough.ipynb   # executed analysis notebook using the same library
├── data/                       # sample dataset + generator (injected missing values, outliers, duplicates)
├── tests/test_eda.py           # 8 unit tests
├── .streamlit/config.toml      # theme + upload limit
└── requirements.txt
```

## Design notes
* **Separation of concerns** – all logic lives in the `eda` package; `app.py` only renders it. The same functions power the notebook and the tests.
* **Health score** – `100 − min(2×missing%, 50) − min(2×duplicate%, 25) − min(constant-column%, 25)`.
* **Robust loading** – encoding fallback, TSV support, Excel sheet picker, silent date-column inference.

## Test
```bash
pytest -q        # 8 passed
```
