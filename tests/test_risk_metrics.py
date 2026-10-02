import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.risk_metrics import (
    beta_to_market,
    christoffersen_cc_test,
    cvar_historical,
    cvar_monte_carlo_t,
    cvar_parametric,
    detect_vol_regime,
    drawdown_series,
    kupiec_pof_test,
    max_drawdown,
    portfolio_returns,
    regime_rolling_var,
    rolling_var,
    sharpe_ratio,
    simple_returns,
    var_historical,
    var_monte_carlo,
    var_monte_carlo_t,
    var_parametric,
)
from src.stress_testing import (
    factor_shock_pnl,
    hypothetical_shock_pnl,
    worst_n_day_loss,
)


def synthetic_prices(n=500, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=(n, 3))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(
        prices, columns=["AAA", "BBB", "CCC"],
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


class TestRiskMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = synthetic_prices()
        cls.rets = simple_returns(cls.prices)
        cls.w = pd.Series([0.5, 0.3, 0.2], index=["AAA", "BBB", "CCC"])
        cls.port = portfolio_returns(cls.rets, cls.w)

    def test_returns_shape(self):
        self.assertEqual(len(self.rets), len(self.prices) - 1)
        self.assertFalse(self.rets.isna().any().any())

    def test_portfolio_weights_normalized(self):
        w2 = pd.Series([5, 3, 2], index=["AAA", "BBB", "CCC"])
        pd.testing.assert_series_equal(
            portfolio_returns(self.rets, w2), self.port, check_names=False
        )

    def test_var_ordering(self):
        # CVaR >= VaR; higher confidence >= lower confidence
        v95 = var_historical(self.port, 0.95)
        v99 = var_historical(self.port, 0.99)
        cv95 = cvar_historical(self.port, 0.95)
        self.assertGreater(v95, 0)
        self.assertGreaterEqual(v99, v95)
        self.assertGreaterEqual(cv95, v95)

    def test_var_methods_agree_on_normal_data(self):
        vh = var_historical(self.port, 0.95)
        vp = var_parametric(self.port, 0.95)
        vm = var_monte_carlo(self.port, 0.95)
        # On ~normal synthetic data the three methods should be within 40% of each other
        self.assertLess(abs(vh - vp) / vh, 0.40)
        self.assertLess(abs(vh - vm) / vh, 0.40)

    def test_gold_standard_normal_var_es(self):
        # Closed-form check against known population parameters.
        rng = np.random.default_rng(123)
        mu, sigma = 0.001, 0.02
        r = pd.Series(rng.normal(mu, sigma, 200_000))
        z = stats.norm.ppf(0.05)
        var_true = -(mu + sigma * z)
        es_true = sigma * stats.norm.pdf(z) / 0.05 - mu
        self.assertAlmostEqual(
            var_parametric(r, 0.95), var_true, delta=abs(var_true) * 0.01
        )
        self.assertAlmostEqual(
            cvar_parametric(r, 0.95), es_true, delta=abs(es_true) * 0.01
        )

    def test_student_t_var_on_heavytail_data(self):
        # On t-distributed data the t-MC VaR should track the theoretical
        # t VaR more closely than the normal parametric VaR does.
        r = pd.Series(
            stats.t.rvs(df=4, loc=0.0005, scale=0.012, size=5000, random_state=7)
        )
        q = stats.t.ppf(0.05, df=4, loc=0.0005, scale=0.012)
        vt = var_monte_carlo_t(r, 0.95, n_sims=20000, seed=1)
        self.assertAlmostEqual(vt, -q, delta=abs(q) * 0.08)
        # On heavy-tailed data the t model should track the true t quantile
        # more closely than the normal parametric VaR does.
        self.assertLess(abs(vt + q), abs(var_parametric(r, 0.95) + q))

    def test_drawdown_nonpositive(self):
        cum = (1 + self.port).cumprod()
        dd = drawdown_series(cum)
        self.assertLessEqual(dd.max(), 0)
        self.assertLessEqual(max_drawdown(cum), 0)

    def test_sharpe_reasonable(self):
        s = sharpe_ratio(self.port)
        self.assertTrue(-5 < s < 5)

    def test_kupiec_output(self):
        # Lagged VaR: estimate on data up to t-1, test day-t returns.
        rvar = rolling_var(self.port, 63, 0.95).shift(1)
        res = kupiec_pof_test(self.port, rvar, 0.95)
        self.assertIn(res["violations"], range(res["n"] + 1))
        self.assertTrue(0 <= res["p_value"] <= 1)

    def test_kupiec_rejects_misspecified_model(self):
        # A deliberately too-tight VaR should be rejected.
        rvar = rolling_var(self.port, 63, 0.95).shift(1) * 0.1
        res = kupiec_pof_test(self.port, rvar, 0.95)
        self.assertTrue(res["reject_h0"])

    def test_christoffersen_output(self):
        rvar = rolling_var(self.port, 63, 0.95).shift(1)
        cc = christoffersen_cc_test(self.port, rvar, 0.95)
        for key in ("lr_ind", "p_ind", "lr_cc", "p_cc"):
            self.assertIn(key, cc)
        if cc["p_cc"] is not None:
            self.assertTrue(0 <= cc["p_cc"] <= 1)
        self.assertEqual(
            cc["transitions"]["n00"]
            + cc["transitions"]["n01"]
            + cc["transitions"]["n10"]
            + cc["transitions"]["n11"],
            cc["n"] - 1,
        )

    def test_beta_to_market(self):
        rng = np.random.default_rng(0)
        mkt = pd.Series(rng.normal(0, 0.01, 500))
        asset = 1.5 * mkt + rng.normal(0, 0.001, 500)
        betas = beta_to_market(pd.DataFrame({"A": asset}), mkt)
        self.assertAlmostEqual(betas["A"], 1.5, delta=0.05)

    def test_factor_shock(self):
        w = pd.Series([0.6, 0.4], index=["A", "B"])
        betas = pd.Series([1.5, 0.5], index=["A", "B"])
        pnl, pb = factor_shock_pnl(w, betas, -0.10, 1_000_000)
        self.assertAlmostEqual(pb, 1.1)
        self.assertAlmostEqual(pnl, -110_000)

    def test_stress(self):
        self.assertAlmostEqual(hypothetical_shock_pnl(1_000_000, -0.10), -100_000)
        worst = worst_n_day_loss(self.port, window=21)
        self.assertLess(worst["worst_return"], 0)
        self.assertEqual(worst["window_days"], 21)
        # start/end cut on actual trading days
        n_days = len(self.port.loc[worst["start_date"]:worst["end_date"]])
        self.assertEqual(n_days, 21)

    def test_regime_detection_no_lookahead(self):
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 300)
        wild = rng.normal(0, 0.03, 300)
        r = pd.Series(np.concatenate([calm, wild]))
        reg = detect_vol_regime(r)
        early = reg.iloc[:200].dropna()
        late = reg.iloc[450:].dropna()
        self.assertTrue(len(early) > 50 and len(late) > 50)
        # 'high' labels must concentrate after the vol jump, not before
        self.assertGreater(
            (late == "high").mean(), (early == "high").mean() + 0.3
        )
        self.assertTrue(set(reg.dropna().unique()) <= {"high", "low"})

    def test_regime_var_no_lookahead(self):
        # An extreme return on the last day must not change any VaR
        # estimate (each day's VaR uses only strictly past data).
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400))
        v1 = regime_rolling_var(r)
        r2 = r.copy()
        r2.iloc[-1] = -0.5
        v2 = regime_rolling_var(r2)
        pd.testing.assert_series_equal(v1, v2)

    def test_regime_var_widens_in_high_vol(self):
        rng = np.random.default_rng(2)
        calm = rng.normal(0, 0.005, 400)
        wild = rng.normal(0, 0.04, 400)
        r = pd.Series(np.concatenate([calm, wild]))
        v = regime_rolling_var(r)
        reg = detect_vol_regime(r)
        high_v = v[reg == "high"].dropna()
        low_v = v[reg == "low"].dropna()
        self.assertTrue(len(high_v) > 20 and len(low_v) > 20)
        self.assertGreater(high_v.median(), low_v.median())


if __name__ == "__main__":
    unittest.main()
