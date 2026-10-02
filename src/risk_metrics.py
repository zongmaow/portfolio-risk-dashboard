"""Core risk metric calculations.

All functions are pure (no I/O, no Streamlit) so they can be unit tested.
Sign convention: VaR / CVaR are reported as POSITIVE numbers representing
the worst expected loss over the horizon (e.g. VaR 95% = 0.023 means
"we are 95% confident the one-day loss will not exceed 2.3%").

Returns are simple (arithmetic) throughout: portfolio returns are exact
under weighting, and (1 + r).cumprod() compounds correctly. (Log returns
were used before 2026-10-02; mixing weighted log returns with arithmetic
compounding is inconsistent, so they were removed.)
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns from an adjusted-close price DataFrame."""
    return prices.pct_change().dropna()


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


def cvar_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) CVaR / Expected Shortfall (positive loss number).

    Closed form: ES = sigma * phi(z) / (1 - alpha) - mu,
    where z = Phi^{-1}(1 - alpha).
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(sigma * stats.norm.pdf(z) / (1 - confidence) - mu)


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


def var_monte_carlo_t(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR under a fitted Student-t distribution.

    Unlike the normal MC above, the t distribution has fat tails, so this
    is a genuinely different model rather than a noisier parametric VaR.
    Reported as a positive loss number.
    """
    df, loc, scale = stats.t.fit(returns)
    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims, random_state=seed)
    return float(-np.quantile(sims, 1 - confidence))


def cvar_monte_carlo_t(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Monte Carlo CVaR under a fitted Student-t distribution."""
    df, loc, scale = stats.t.fit(returns)
    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims, random_state=seed)
    cutoff = np.quantile(sims, 1 - confidence)
    tail = sims[sims <= cutoff]
    if len(tail) == 0:
        return var_monte_carlo_t(returns, confidence, n_sims, seed)
    return float(-tail.mean())


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


def beta_to_market(
    asset_returns: pd.DataFrame, market_returns: pd.Series
) -> pd.Series:
    """CAPM-style betas of each asset vs the market: beta = Cov(r_i, r_m) / Var(r_m)."""
    aligned = pd.concat([asset_returns, market_returns.rename("__mkt__")], axis=1, join="inner")
    mkt = aligned["__mkt__"]
    var = mkt.var()
    betas = {}
    for col in asset_returns.columns:
        col_s = aligned[col]
        betas[col] = float(col_s.cov(mkt) / var) if var else 1.0
    return pd.Series(betas)


def detect_vol_regime(
    returns: pd.Series, vol_window: int = 21, min_periods: int = 63
) -> pd.Series:
    """Classify each day into a volatility regime ('high' / 'low').

    The regime is a trailing realized-volatility proxy: the rolling
    `vol_window`-day std of returns versus its expanding median. Both the
    volatility and the threshold are lagged by one day, so the label at t
    uses only information available at t-1 — no look-ahead.
    """
    r = returns.dropna()
    rv = r.rolling(vol_window).std().shift(1)
    thresh = rv.expanding(min_periods=min_periods).median().shift(1)
    regime = pd.Series(np.where(rv > thresh, "high", "low"), index=r.index)
    regime[rv.isna() | thresh.isna()] = np.nan
    return regime


def regime_rolling_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 63,
    vol_window: int = 21,
    min_obs: int = 20,
) -> pd.Series:
    """Regime-aware rolling historical VaR (positive loss numbers).

    For each day t, VaR is the empirical quantile over the most recent
    `window` past returns that share day-t's volatility regime. Only data
    strictly before t is used, so the series is a genuine forecast and is
    directly backtestable WITHOUT an extra shift.
    """
    r = returns.dropna()
    regime = detect_vol_regime(r, vol_window=vol_window)
    out = pd.Series(np.nan, index=r.index)
    vals = r.to_numpy()
    reg = regime.to_numpy()
    for i in range(len(r)):
        if pd.isna(reg[i]):
            continue
        tail = vals[:i][reg[:i] == reg[i]][-window:]
        if len(tail) >= min_obs:
            out.iloc[i] = -np.quantile(tail, 1 - confidence)
    return out


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


def christoffersen_cc_test(
    returns: pd.Series, var_series: pd.Series, confidence: float = 0.95
) -> dict:
    """Christoffersen (1998) conditional-coverage backtest.

    Extends Kupiec's unconditional test with an independence test:
    violations should not cluster in time. LR_cc = LR_pof + LR_ind ~ chi2(2).

    IMPORTANT: pass a *lagged* VaR series (estimated on data up to t-1)
    so day-t returns are tested against information available at t-1.
    """
    aligned = pd.DataFrame({"r": returns, "var": var_series}).dropna()
    viol = (aligned["r"] < -aligned["var"]).astype(int).to_numpy()
    n = len(viol)
    prev, curr = viol[:-1], viol[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n_trans = n00 + n01 + n10 + n11
    pof = kupiec_pof_test(returns, var_series, confidence)

    def _safe(result_dict, lr_ind=None, p_ind=None, lr_cc=None, p_cc=None):
        result_dict.update(
            {
                "lr_ind": lr_ind,
                "p_ind": p_ind,
                "lr_cc": lr_cc,
                "p_cc": p_cc,
                "reject_independence": None
                if p_ind is None
                else bool(p_ind < 0.05),
                "reject_cc": None if p_cc is None else bool(p_cc < 0.05),
                "transitions": {
                    "n00": n00,
                    "n01": n01,
                    "n10": n10,
                    "n11": n11,
                },
            }
        )
        return result_dict

    base = {
        "n": n,
        "violations": int(viol.sum()),
        "expected": (1 - confidence) * n,
    }
    if n_trans == 0 or min(n00 + n01, n10 + n11) == 0:
        return _safe(base)
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / n_trans
    if min(pi0, pi1, pi) in (0, 1):
        return _safe(base)
    lr_ind = -2 * np.log(
        ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11))
        / (
            (1 - pi0) ** n00
            * pi0**n01
            * (1 - pi1) ** n10
            * pi1**n11
        )
    )
    p_ind = float(1 - stats.chi2.cdf(lr_ind, df=1))
    lr_pof = pof["lr_stat"]
    if lr_pof is None or np.isnan(lr_pof):
        return _safe(base, lr_ind=float(lr_ind), p_ind=p_ind)
    lr_cc = float(lr_pof + lr_ind)
    p_cc = float(1 - stats.chi2.cdf(lr_cc, df=2))
    return _safe(base, lr_ind=float(lr_ind), p_ind=p_ind, lr_cc=lr_cc, p_cc=p_cc)
