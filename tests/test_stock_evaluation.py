import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_stock_retraining import daily_average, forecast_metrics
from verify_stock_retraining import dollar_ledger, lstm_numpy


class RetrainingEvaluationTests(unittest.TestCase):
    def test_daily_mean_does_not_treat_missing_as_zero(self):
        values = np.array([[2., np.nan], [4., 8.]])
        np.testing.assert_allclose(daily_average(values, np.isfinite(values)), [2., 6.])

    def test_common_forecast_mask_and_invalid_counts(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        r = pd.DataFrame(np.random.default_rng(4).normal(0, .01, (100, 2)), index=dates)
        g = pd.DataFrame(.01, index=dates[25:], columns=r.columns)
        h = g.copy()
        h.iloc[0, 0] = np.nan
        h.iloc[1, 0] = -.01
        with patch("evaluate_stock_retraining.mean_difference_interval", return_value={}):
            result = forecast_metrics(r, g, h)
        self.assertEqual(result["common_asset_dates"], 149)
        self.assertEqual(result["positive_subset_asset_dates"], 148)
        self.assertEqual(result["nonpositive_hybrid_asset_dates"], 1)
        self.assertEqual(result["missing_hybrid_asset_dates"], 1)
        self.assertGreater(result["volatility_mse"]["hybrid_raw"]["date_equal_mse"],
                           result["volatility_mse"]["garch"]["date_equal_mse"])

    def test_numpy_lstm_zero_weights_dense_offset(self):
        weights = [np.zeros((1, 8)), np.zeros((2, 8)), np.zeros(8), np.zeros((2, 1)), np.array([3.])]
        np.testing.assert_allclose(lstm_numpy(np.ones((4, 10, 1)), weights), 3.)

    def test_independent_ledger_initial_cost_and_drift(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        r = pd.DataFrame(0., index=dates, columns=["A", "B"])
        r.iloc[2] = [.1, 0.]
        r.iloc[3] = [0., .1]
        target = r*0+.5
        result = dollar_ledger(r, target)
        self.assertAlmostEqual(result.nav.iloc[0], 1.05/1.001)
        self.assertAlmostEqual(result.nav.iloc[1], 1.1/1.001)
        self.assertAlmostEqual(result.traded_notional.iloc[1], 0.)


if __name__ == "__main__":
    unittest.main()
