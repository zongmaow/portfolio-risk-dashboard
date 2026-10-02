"""Plotly chart helpers for the dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .risk_metrics import drawdown_series


def cumulative_returns_chart(cumulative: pd.Series) -> go.Figure:
    fig = px.line(cumulative, title="Cumulative Portfolio Returns")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Growth of $1", hovermode="x unified"
    )
    return fig


def rolling_var_chart(rolling_var: pd.Series, confidence: float) -> go.Figure:
    fig = px.line(
        rolling_var,
        title=f"Rolling {int(confidence * 100)}% VaR (63-day window)",
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="VaR (fraction of portfolio)", hovermode="x unified"
    )
    return fig


def drawdown_chart(cumulative: pd.Series) -> go.Figure:
    dd = drawdown_series(cumulative)
    fig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="firebrick"),
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Asset Return Correlations",
    )
    return fig


def stress_bar_chart(stress_df: pd.DataFrame) -> go.Figure:
    df = stress_df.copy()
    colors = ["red" if v < 0 else "green" for v in df["P&L (%)"]]
    fig = go.Figure(
        go.Bar(
            x=df["Scenario"],
            y=df["P&L (%)"] * 100,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in df["P&L ($)"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Stress Test Scenarios",
        xaxis_title="",
        yaxis_title="P&L (% of portfolio)",
    )
    return fig
