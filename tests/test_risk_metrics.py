import unittest

import numpy as np
import pandas as pd

from src.risk_metrics import (
    cvar_historical,
    drawdown_series,
    kupiec_pof_test,
    log_returns,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    var_historical,
    var_monte_carlo,
    var_parametric,
)
from src.stress_testing import hypothetical_shock_pnl, worst_n_day_loss


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
        cls.rets = log_returns(cls.prices)
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

    def test_drawdown_nonpositive(self):
        cum = (1 + self.port).cumprod()
        dd = drawdown_series(cum)
        self.assertLessEqual(dd.max(), 0)
        self.assertLessEqual(max_drawdown(cum), 0)

    def test_sharpe_reasonable(self):
        s = sharpe_ratio(self.port)
        self.assertTrue(-5 < s < 5)

    def test_kupiec_output(self):
        rvar = -self.port.rolling(63).quantile(0.05)
        res = kupiec_pof_test(self.port, rvar, 0.95)
        self.assertIn(res["violations"], range(res["n"] + 1))
        self.assertTrue(0 <= res["p_value"] <= 1)

    def test_stress(self):
        self.assertAlmostEqual(hypothetical_shock_pnl(1_000_000, -0.10), -100_000)
        worst = worst_n_day_loss(self.port, window=21)
        self.assertLess(worst["worst_return"], 0)
        self.assertEqual(worst["window_days"], 21)


if __name__ == "__main__":
    unittest.main()
