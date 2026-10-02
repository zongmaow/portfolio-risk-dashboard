"""Stress testing scenarios.

Two families:
  1. Hypothetical uniform shocks (-5% / -10% / -20% and custom): a one-day
     market shock applied to current portfolio value.
  2. Historical replay: the worst N-day cumulative loss observed in-sample,
     replayed as the scenario ("what if the worst stretch in our data
     happened again tomorrow").
"""

import numpy as np
import pandas as pd

SHOCK_LEVELS = [-0.05, -0.10, -0.20]


def hypothetical_shock_pnl(portfolio_value: float, shock: float) -> float:
    """One-day P&L impact of a uniform market shock on current portfolio value."""
    return portfolio_value * shock


def worst_n_day_loss(returns: pd.Series, window: int = 21) -> dict:
    """Find the worst `window`-day cumulative return in the sample."""
    rolling_cum = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    worst = float(rolling_cum.min())
    end_date = rolling_cum.idxmin()
    start_date = end_date - pd.Timedelta(days=int(window * 1.6))
    return {
        "worst_return": worst,
        "window_days": window,
        "end_date": end_date,
        "start_date": start_date,
        "scenario_pnl_pct": worst,
    }


def stress_summary(portfolio_value: float, returns: pd.Series) -> pd.DataFrame:
    """Summary table of all stress scenarios: shock -> P&L $ and %."""
    rows = []
    for shock in SHOCK_LEVELS:
        rows.append(
            {
                "Scenario": f"Market shock {shock:.0%}",
                "P&L ($)": hypothetical_shock_pnl(portfolio_value, shock),
                "P&L (%)": shock,
            }
        )
    worst = worst_n_day_loss(returns)
    rows.append(
        {
            "Scenario": f"Worst {worst['window_days']}-day replay "
            f"(ending {worst['end_date'].date()})",
            "P&L ($)": portfolio_value * worst["worst_return"],
            "P&L (%)": worst["worst_return"],
        }
    )
    return pd.DataFrame(rows)
