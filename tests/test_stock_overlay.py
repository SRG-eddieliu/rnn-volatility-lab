import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.stock_overlay import StockConfig, ratio_sequences, training_rows, ticker_seed, rolling_stock_garch


class StockTimingTests(unittest.TestCase):
    def test_sequence_excludes_own_target(self):
        X, y, positions = ratio_sequences(np.arange(20.), 3)
        np.testing.assert_array_equal(X[0], [0, 1, 2])
        self.assertEqual(y[0], 3)
        self.assertEqual(positions[0], 3)

    def test_training_cutoff_and_future_invariance(self):
        values = np.arange(30.)
        first = training_rows(*ratio_sequences(values, 3), 15)
        values[15:] = 100000
        second = training_rows(*ratio_sequences(values, 3), 15)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertLess(first[2].max(), 15)

    def test_missing_ratios_are_not_compressed(self):
        values = np.arange(12.)
        values[5] = np.nan
        _, _, positions = training_rows(*ratio_sequences(values, 3), 12)
        self.assertFalse(set([5, 6, 7, 8]) & set(positions))
        self.assertIn(9, positions)

    def test_seed_is_stable_and_fit_specific(self):
        self.assertEqual(ticker_seed("A", 0, 42), ticker_seed("A", 0, 42))
        self.assertNotEqual(ticker_seed("A", 0, 42), ticker_seed("A", 1, 42))
        self.assertNotEqual(ticker_seed("A", 0, 42), ticker_seed("B", 0, 42))

    def test_garch_uses_only_prior_window_and_flags_failure(self):
        r = np.arange(1., 12.)/100
        seen = []
        def factory(history, **kwargs):
            seen.append(history.copy())
            raise ValueError("forced fit failure")
        with patch("arch.arch_model", side_effect=factory):
            sigma, flags, _ = rolling_stock_garch(r, StockConfig(garch_window=3))
        np.testing.assert_allclose(seen[0], r[:3]*100)
        np.testing.assert_allclose(seen[-1], r[-4:-1]*100)
        self.assertAlmostEqual(sigma[3], r[:3].std(ddof=1))
        self.assertEqual(flags[3], "past_vol_fallback")
        self.assertTrue(np.isnan(sigma[:3]).all())

    def test_garch_does_not_fill_missing_history(self):
        r = np.arange(1., 10.)/100
        r[3] = np.nan
        with patch("arch.arch_model", side_effect=ValueError("forced")):
            sigma, flags, _ = rolling_stock_garch(r, StockConfig(garch_window=3))
        self.assertTrue(np.isnan(sigma[4:7]).all())
        self.assertEqual(flags[6], "insufficient_history")
        self.assertTrue(np.isfinite(sigma[7]))


@unittest.skipUnless(os.environ.get("RUN_TF_TESTS") == "1", "optional TensorFlow integration")
class StockTrainingTests(unittest.TestCase):
    def test_checkpoint_and_future_invariance(self):
        from src.models.stock_overlay import train_stock_ratio
        from tensorflow import keras
        cfg = StockConfig(lookback=3, minimum_sequences=15, retrain_interval=30,
                          realized_window=5, epochs=1)
        rng = np.random.default_rng(3)
        r = rng.normal(0, .01, 90)
        sigma = np.full(90, .01)
        dates = pd.bdate_range("2020-01-01", periods=90)
        mask = np.arange(90) >= 40
        with tempfile.TemporaryDirectory() as root:
            a, _, logs = train_stock_ratio(r, sigma, dates, mask, "TEST", Path(root)/"a", cfg)
            r[65:] = .1
            b, _, _ = train_stock_ratio(r, sigma, dates, mask, "TEST", Path(root)/"b", cfg)
            np.testing.assert_allclose(a[:65], b[:65], equal_nan=True, atol=1e-8, rtol=0)
            for row in logs:
                self.assertLess(row["training_last_target"], row["first_forecast"])
                self.assertGreaterEqual(row["training_sequences"], 15)
            self.assertTrue(np.isfinite(a[40:]).all())
            keras.models.load_model(Path(root)/"a/fit-000.keras")


if __name__ == "__main__":
    unittest.main()
