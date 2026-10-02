# Portfolio Risk Dashboard

Equity portfolio VaR, stress tests, and a simple volatility-regime backtest.

A Streamlit dashboard for market risk on a custom equity portfolio. It computes VaR and Expected Shortfall three ways, runs stress scenarios, and backtests the VaR — including a version conditioned on the volatility regime.

Built with **Python · Streamlit · yfinance · Plotly**.

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| Overview | Cumulative portfolio returns, drawdown chart, KPI cards (VaR, CVaR, vol, Sharpe, max drawdown) |
| Risk Metrics | VaR / Expected Shortfall comparison across three models (historical, normal parametric, Student-t Monte Carlo), rolling 63-day VaR, asset correlation heatmap |
| Stress Testing | -5% / -10% / -20% market shocks, worst 21-day in-sample replay, custom shock slider, single-factor (SPY beta) shock |
| Backtesting | Kupiec POF + Christoffersen independence tests on the lagged rolling VaR model, violation timeline, and a **regime-aware VaR** comparison (high/low volatility regimes with shaded bands) |
| Sector Exposure | Portfolio weights aggregated by GICS sector |

## Screenshots

Default demo portfolio: AAPL / MSFT / NVDA / JPM / XOM on the frozen sample 2022-01-03 – 2026-10-01 (95% VaR).

![Cumulative portfolio returns](screenshots/tab1a_cumulative.png)
![Rolling 63-day VaR](screenshots/tab2a_rolling_var.png)
![Regime-aware backtest: shaded bands mark high-volatility regimes, where the VaR band widens](screenshots/tab4_regime_backtest.svg)
![Stress test summary](screenshots/tab3_stress.png)
![Sector exposure](screenshots/tab5_sector.png)

## Quickstart

```bash
git clone https://github.com/zongmaow/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints, enter tickers and weights in the sidebar, and click **Run analysis**.

Run the test suite (synthetic data, no network needed):

```bash
python -m unittest discover tests
```

## Methodology

- **Returns**: simple (arithmetic) daily returns from adjusted-close prices (yfinance). Arithmetic returns are exact under portfolio weighting and compound correctly with `(1 + r).cumprod()`.
- **VaR** reported as a positive loss number at the chosen confidence level:
  - *Historical simulation* — empirical quantile of the return distribution.
  - *Parametric* — normal assumption, μ + σ·z.
  - *Monte Carlo (Student-t)* — 10,000 simulated one-day returns from a Student-t fitted to the data, capturing fat tails the normal model misses.
- **CVaR / Expected Shortfall** — mean loss conditional on breaching VaR, computed for all three models (historical, normal closed form, Student-t Monte Carlo).
- **Stress tests** — hypothetical uniform shocks, a single-factor shock using each asset's SPY beta, plus a historical replay: the worst 21-day cumulative loss observed in-sample, re-applied as "what if it happened again".
- **Backtesting** — Kupiec POF test plus Christoffersen independence test. The VaR series is **lagged by one day**: day-t returns are tested against VaR estimated on data up to t-1, so there is no look-ahead. H₀ is that the observed violation rate equals 1 − confidence and violations do not cluster. One caveat built into any short-window historical VaR: a 5% quantile read off 63 past returns by linear interpolation is not a true 5% quantile out of sample — for iid data its expected exceedance rate is about 6.4%, so some excess violations are an estimator artifact, not necessarily a model failure.
- **Regime-aware VaR** — each day is classified into a high/low volatility regime from its trailing 21-day realized vol vs. the expanding median (both lagged, so no look-ahead). VaR is then estimated from the most recent 63 past days in the *same* regime (a forecast is made once at least 20 same-regime days exist), so the band widens automatically when volatility clusters. It is a diagnostic check, not a forecasting model.
- **Comparing the two backtests** — the models become valid on different days (the regime forecast needs a longer warm-up), so their full-sample counts are not comparable. The Backtesting tab therefore also scores both models on the **common dates** where both have a forecast, adding mean quantile (pinball) loss, which prices band width as well as violations. On the frozen sample (1,066 common days): single-window 64 violations, Kupiec p = 0.144, quantile loss 0.001534; regime-aware 61 violations, p = 0.290, quantile loss 0.001572, with a wider average band (2.06% vs 1.95%). Split by the day's own regime label, the regime model has fewer violations on high-volatility days (7 vs 12 of 230) and slightly more on low-volatility days (54 vs 52 of 836). Neither model is rejected on the common dates, and the score difference is small — conditioning helps inside volatile stretches but is not a clear overall win.

## Frozen sample

All reported numbers use a fixed sample: adjusted closes for the default tickers plus SPY, **2022-01-03 to 2026-10-01** (1,191 trading days), stored in `data/frozen_prices.csv` and loaded by default (sidebar checkbox). Live downloads remain available by unchecking the box or picking other tickers/dates, but the frozen sample is the reference — results do not drift as new data arrives or as Yahoo revises history.

## Assumptions and Limitations

| Assumption | What it implies |
|---|---|
| Simple returns throughout | Exact under weighting; no log/arithmetic mixing |
| Backtest uses lagged VaR (t−1) | No look-ahead: estimation window excludes the tested day |
| Volatility regime is a simple vol proxy | Trailing 21d vol vs expanding median, both lagged; not a fitted HMM — regimes are descriptive, not structural |
| Simple returns are ~normal (parametric VaR) | Understates tail risk for fat-tailed assets; historical VaR and Student-t MC do not make this assumption |
| 252 trading days per year | Annualized vol/Sharpe scale with √252 |
| Weights normalized to 1, no leverage | Long-only, fully invested portfolio |
| Fixed weights held via daily rebalancing, zero transaction costs | Portfolio returns are a constant-mix series: weights are reapplied to each day's asset returns, which implicitly rebalances back to target every day and ignores trading costs |
| Missing prices forward-filled, then incomplete rows dropped | Applied to the whole price frame before returns are computed; only the correlation matrix uses pairwise-complete observations |
| Uniform shock hits all assets equally | Ignores beta differences across holdings; the SPY-beta shock is the structured alternative |
| Sector via yfinance info, static fallback | Sector for exotic tickers may show as "Unknown" |
| Survivorship-free data not guaranteed | Delisted tickers simply have no price history |

## Project Structure

```
portfolio-risk-dashboard/
├── app.py                  # Streamlit app (sidebar config + 5 tabs)
├── data/
│   └── frozen_prices.csv   # frozen price snapshot (default tickers + SPY, 2022-01-03 – 2026-10-01)
├── src/
│   ├── data.py             # yfinance price download, frozen-snapshot loader, sector lookup
│   ├── risk_metrics.py     # VaR / CVaR / drawdown / Sharpe / Kupiec (pure functions)
│   ├── stress_testing.py   # hypothetical shocks + historical replay
│   └── plotting.py         # Plotly chart helpers
├── tests/
│   ├── test_risk_metrics.py  # unit tests on synthetic data
│   └── test_data.py          # frozen-snapshot integrity + reproducibility tests
└── requirements.txt
```

Core logic is deliberately kept as pure functions (no Streamlit, no I/O) so it can be unit-tested and reused in scripts.

## What I Learned

- Mixing daily returns with annualized volatility in one formula gives numbers that look right and aren't. The unit tests now pin the √252 scaling so it stays fixed.
- Streamlit caching belongs at the app layer (`data.py` stays framework-free), otherwise tests end up importing UI code.

## Disclaimer

Educational project. Not investment advice. Market data via Yahoo Finance may be delayed or inaccurate.

## Author

**Zongmao Wu, CFA** — MS Financial Engineering, USC · [LinkedIn](https://www.linkedin.com/in/zongmaow/)
