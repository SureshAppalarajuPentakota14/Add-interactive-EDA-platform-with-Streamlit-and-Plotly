"""Plotly figure factories (all return plotly Figure objects)."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


def missing_bar(report: pd.DataFrame) -> go.Figure:
    fig = px.bar(report, x="missing_pct", y="column", orientation="h",
                 color="severity", text="missing_pct", template=TEMPLATE,
                 color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B",
                                     "High": "#F97316", "Critical": "#EF4444"},
                 title="Missing values by column (%)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      height=max(300, 28 * len(report) + 120))
    return fig


def missing_heatmap(df: pd.DataFrame, max_rows: int = 300) -> go.Figure:
    sample = df.sample(min(len(df), max_rows), random_state=0).sort_index()
    fig = px.imshow(sample.isna().astype(int).T, aspect="auto", template=TEMPLATE,
                    color_continuous_scale=["#E5E7EB", "#EF4444"],
                    title="Missingness map (red = missing)")
    fig.update_coloraxes(showscale=False)
    return fig


def histogram(df, col, color=None, bins=40, marginal="box") -> go.Figure:
    return px.histogram(df, x=col, color=color, nbins=bins, marginal=marginal,
                        template=TEMPLATE, title=f"Distribution of {col}")


def box(df, y, x=None, color=None) -> go.Figure:
    return px.box(df, y=y, x=x, color=color, points="outliers", template=TEMPLATE,
                  title=f"Box plot of {y}" + (f" by {x}" if x else ""))


def scatter(df, x, y, color=None, size=None, trendline=False) -> go.Figure:
    """Scatter plot. Trendline uses numpy least squares (no statsmodels needed)."""
    fig = px.scatter(df, x=x, y=y, color=color, size=size, opacity=.7,
                     template=TEMPLATE, title=f"{y} vs {x}")
    if trendline:
        d = df[[x, y]].dropna()
        if len(d) > 2 and pd.api.types.is_numeric_dtype(d[x]):
            m = d[x].cov(d[y]) / d[x].var()
            b = d[y].mean() - m * d[x].mean()
            xs = [d[x].min(), d[x].max()]
            fig.add_trace(go.Scatter(x=xs, y=[m * v + b for v in xs], mode="lines",
                                     name="Trend", line=dict(color="black", dash="dash")))
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r", template=TEMPLATE,
                    title="Correlation matrix")
    fig.update_layout(height=max(450, 40 * len(corr)))
    return fig


def category_bar(df, col, top=15) -> go.Figure:
    vc = df[col].value_counts().head(top).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x="count", y=col, orientation="h", template=TEMPLATE,
                 title=f"Top {top} values of {col}")
    return fig.update_layout(yaxis={"categoryorder": "total ascending"})


def time_series(df, date_col, value_col, freq="ME", agg="sum") -> go.Figure:
    ts = df.set_index(date_col)[value_col].resample(freq).agg(agg).reset_index()
    return px.line(ts, x=date_col, y=value_col, markers=True, template=TEMPLATE,
                   title=f"{agg.title()} of {value_col} over time")


def outlier_bar(summary: pd.DataFrame) -> go.Figure:
    return px.bar(summary, x="column", y="outlier_pct", text="outliers", template=TEMPLATE,
                  title="Outlier share by numeric column (%)")
