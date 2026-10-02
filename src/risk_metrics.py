"""Core risk metric calculations.

All functions are pure (no I/O, no Streamlit) so they can be unit tested.
Sign convention: VaR / CVaR are reported as POSITIVE numbers representing
the worst expected loss over the horizon (e.g. VaR 95% = 0.023 means
"we are 95% confident the one-day loss will not exceed 2.3%").
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from an adjusted-close price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted portfolio return series. Weights are normalized to sum to 1."""
    w = weights / weights.sum()
    return (asset_returns * w).sum(axis=1)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std() * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio against a constant risk-free rate."""
    excess = returns - risk_free / periods
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical-simulation VaR (positive loss number)."""
    return float(-np.quantile(returns, 1 - confidence))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR (positive loss number)."""
    mu, sigma = returns.mean(), returns.std()
    return float(-(mu + sigma * stats.norm.ppf(1 - confidence)))


def var_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    horizon_days: int = 1,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR under a fitted normal distribution (positive loss number)."""
    rng = np.random.default_rng(seed)
    mu, sigma = returns.mean(), returns.std()
    sims = rng.normal(
        loc=mu * horizon_days,
        scale=sigma * np.sqrt(horizon_days),
        size=n_sims,
    )
    return float(-np.quantile(sims, 1 - confidence))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall (positive loss number)."""
    cutoff = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return var_historical(returns, confidence)
    return float(-tail.mean())


def rolling_var(
    returns: pd.Series, window: int = 63, confidence: float = 0.95
) -> pd.Series:
    """Rolling historical VaR (positive loss numbers)."""
    return -returns.rolling(window).quantile(1 - confidence)


def drawdown_series(cumulative: pd.Series) -> pd.Series:
    """Drawdown series as fractions (0 at peaks, negative in drawdowns)."""
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def max_drawdown(cumulative: pd.Series) -> float:
    return float(drawdown_series(cumulative).min())


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.corr()


def kupiec_pof_test(
    returns: pd.Series, var_series: pd.Series, confidence: float = 0.95
) -> dict:
    """Kupiec proportion-of-failures test for VaR backtesting.

    Returns violations, expected violations, LR statistic and p-value.
    H0: the observed violation rate equals 1 - confidence.
    """
    aligned = pd.DataFrame({"r": returns, "var": var_series}).dropna()
    n = len(aligned)
    violations = int((aligned["r"] < -aligned["var"]).sum())
    p0 = 1 - confidence
    if n == 0 or violations in (0, n):
        return {
            "n": n,
            "violations": violations,
            "expected": p0 * n,
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "reject_h0": None,
        }
    p_hat = violations / n
    lr = -2 * (
        (n - violations) * np.log((1 - p0) / (1 - p_hat))
        + violations * np.log(p0 / p_hat)
    )
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "n": n,
        "violations": violations,
        "expected": p0 * n,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject_h0": bool(p_value < 0.05),
    }
