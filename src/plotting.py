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


def backtest_chart(
    returns: pd.Series,
    var_series: pd.Series,
    regimes: pd.Series | None = None,
) -> go.Figure:
    """Violation timeline: daily returns vs the (lagged) VaR band.

    Red dots mark violations (return worse than -VaR). Clustering of dots
    is what the Christoffersen independence test formalizes. When `regimes`
    is given, high-volatility stretches are shaded so the eye can check
    whether violations cluster inside them.
    """
    aligned = pd.DataFrame({"r": returns, "var": var_series}).dropna()
    viol = aligned[aligned["r"] < -aligned["var"]]
    fig = go.Figure()
    if regimes is not None:
        reg = regimes.dropna()
        is_high = (reg == "high").to_numpy()
        idx = reg.index
        blocks, start = [], None
        for t, v in zip(idx, is_high):
            if v and start is None:
                start = t
            elif not v and start is not None:
                blocks.append((start, t))
                start = None
        if start is not None:
            blocks.append((start, idx[-1]))
        for x0, x1 in blocks:
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor="orange", opacity=0.12, line_width=0,
            )
    fig.add_scatter(
        x=aligned.index, y=aligned["r"] * 100, mode="lines",
        name="Daily return", line=dict(color="steelblue", width=1),
    )
    fig.add_scatter(
        x=aligned.index, y=-aligned["var"] * 100, mode="lines",
        name="VaR band", line=dict(color="firebrick", dash="dash", width=1),
    )
    fig.add_scatter(
        x=viol.index, y=viol["r"] * 100, mode="markers",
        name="Violations", marker=dict(color="red", size=6),
    )
    fig.update_layout(
        title="Backtest: daily returns vs lagged VaR",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
    )
    return fig
