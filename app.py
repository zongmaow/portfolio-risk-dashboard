"""Portfolio Risk Dashboard."""

import pandas as pd
import streamlit as st

from src.data import download_prices, get_sectors, sector_exposure
from src.plotting import (
    correlation_heatmap,
    cumulative_returns_chart,
    drawdown_chart,
    rolling_var_chart,
    stress_bar_chart,
)
from src.risk_metrics import (
    annualized_volatility,
    correlation_matrix,
    cvar_historical,
    kupiec_pof_test,
    log_returns,
    max_drawdown,
    portfolio_returns,
    rolling_var,
    sharpe_ratio,
    var_historical,
    var_monte_carlo,
    var_parametric,
)
from src.stress_testing import SHOCK_LEVELS, stress_summary

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Portfolio Setup")
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)", value="AAPL, MSFT, NVDA, JPM, XOM"
)
weights_input = st.sidebar.text_input(
    "Weights (comma-separated, optional)", value="0.3, 0.25, 0.2, 0.15, 0.1"
)
start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2022-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today())
confidence = st.sidebar.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01)
portfolio_value = st.sidebar.number_input(
    "Current portfolio value ($)", min_value=1000.0, value=1_000_000.0, step=10_000.0
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
raw_weights = [float(w) for w in weights_input.split(",") if w.strip()]
weights = pd.Series(
    raw_weights if len(raw_weights) == len(tickers) else [1.0] * len(tickers),
    index=tickers,
)
weights = weights / weights.sum()

run = st.sidebar.button("Run analysis", type="primary")

if run:
    with st.spinner("Downloading market data..."):
        prices = download_prices(tickers, str(start_date), str(end_date))
    asset_rets = log_returns(prices)
    port_rets = portfolio_returns(asset_rets, weights.loc[prices.columns])
    cumulative = (1 + port_rets).cumprod()

    # ---------- KPIs ----------
    var_h = var_historical(port_rets, confidence)
    cvar_h = cvar_historical(port_rets, confidence)
    vol = annualized_volatility(port_rets)
    sharpe = sharpe_ratio(port_rets)
    mdd = max_drawdown(cumulative)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Daily VaR", f"{var_h:.2%}")
    c2.metric("Daily CVaR", f"{cvar_h:.2%}")
    c3.metric("Annualized Vol", f"{vol:.2%}")
    c4.metric("Sharpe", f"{sharpe:.2f}")
    c5.metric("Max Drawdown", f"{mdd:.2%}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Risk Metrics", "Stress Testing", "Backtesting", "Sector Exposure"]
    )

    with tab1:
        st.plotly_chart(cumulative_returns_chart(cumulative), use_container_width=True)
        st.plotly_chart(drawdown_chart(cumulative), use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Parametric VaR",
            f"{var_parametric(port_rets, confidence):.2%}",
        )
        col_b.metric(
            "Monte Carlo VaR (10k sims)",
            f"{var_monte_carlo(port_rets, confidence):.2%}",
        )
        st.plotly_chart(
            rolling_var_chart(rolling_var(port_rets, confidence=confidence), confidence),
            use_container_width=True,
        )
        st.plotly_chart(
            correlation_heatmap(correlation_matrix(asset_rets)),
            use_container_width=True,
        )

    with tab3:
        summary = stress_summary(portfolio_value, port_rets)
        st.plotly_chart(stress_bar_chart(summary), use_container_width=True)
        st.dataframe(
            summary.style.format({"P&L ($)": "${:,.0f}", "P&L (%)": "{:.2%}"}),
            use_container_width=True,
        )
        custom_shock = st.slider(
            "Custom one-day shock (%)", -50, 0, -10, 1
        )
        st.write(
            f"Custom shock P&L: **${portfolio_value * custom_shock / 100:,.0f}**"
        )

    with tab4:
        rvar = rolling_var(port_rets, confidence=confidence)
        test = kupiec_pof_test(port_rets, rvar, confidence)
        st.write(f"**Sample:** {test['n']} trading days")
        st.write(
            f"**Violations:** {test['violations']} "
            f"(expected ≈ {test['expected']:.1f})"
        )
        st.write(f"**LR statistic:** {test['lr_stat']:.3f}")
        st.write(f"**p-value:** {test['p_value']:.4f}")
        if test["reject_h0"]:
            st.error(
                "H₀ rejected at 5%: the VaR model does not match the "
                "nominal violation rate."
            )
        else:
            st.success(
                "H₀ not rejected at 5%: the VaR model's violation rate is "
                "consistent with the nominal level."
            )
        st.caption(
            "Kupiec proportion-of-failures test. A rejected H₀ does not "
            "invalidate the model outright — it flags a mismatch worth "
            "investigating (regime change, fat tails, window choice)."
        )

    with tab5:
        sectors = get_sectors(list(prices.columns))
        exposure = sector_exposure(weights.loc[prices.columns], sectors)
        st.bar_chart(exposure)
        st.dataframe(
            exposure.rename("Weight").to_frame().style.format("{:.2%}"),
            use_container_width=True,
        )
        st.caption("Sector mapping via yfinance with a static fallback table.")
else:
    st.info("Configure your portfolio in the sidebar, then click **Run analysis**.")
