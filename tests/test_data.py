import unittest

import pandas as pd

from src.data import load_frozen_prices
from src.risk_metrics import annualized_volatility, portfolio_returns, simple_returns


class TestFrozenSnapshot(unittest.TestCase):
    def test_snapshot_shape_and_range(self):
        prices = load_frozen_prices()
        self.assertEqual(len(prices), 1191)
        self.assertEqual(prices.index.min(), pd.Timestamp("2022-01-03"))
        self.assertEqual(prices.index.max(), pd.Timestamp("2026-10-01"))
        for col in ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"):
            self.assertIn(col, prices.columns)
        self.assertFalse(prices.isna().any().any())

    def test_snapshot_subset_and_missing_ticker(self):
        sub = load_frozen_prices(["AAPL", "SPY"])
        self.assertEqual(list(sub.columns), ["AAPL", "SPY"])
        with self.assertRaises(ValueError):
            load_frozen_prices(["NOTAREALTICKER"])

    def test_snapshot_reproduces_reported_vol(self):
        # The README/papers report 23.6% annualized vol for the default
        # portfolio on the frozen sample; the snapshot must reproduce it.
        tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=tickers)
        prices = load_frozen_prices(tickers)
        port = portfolio_returns(simple_returns(prices), weights)
        self.assertAlmostEqual(annualized_volatility(port), 0.236, delta=0.002)


if __name__ == "__main__":
    unittest.main()
