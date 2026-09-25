import unittest
import numpy as np
import pandas as pd

from src.evaluation.portfolio import (
    overlay_targets, rebalance_after_cost, rebalance_mask, simulate_portfolio, performance_stats,
)


class PortfolioTests(unittest.TestCase):
    def frames(self, rows=8):
        dates = pd.bdate_range("2024-01-02", periods=rows)
        returns = pd.DataFrame(0., index=dates, columns=["A", "B"])
        targets = pd.DataFrame(.5, index=dates, columns=returns.columns)
        return returns, targets

    def test_weighted_simple_returns_and_drift(self):
        returns, targets = self.frames(4)
        returns.iloc[1] = [.1, 0.]
        returns.iloc[2] = [0., .1]
        result = simulate_portfolio(returns, targets, "yearly", 0, execution_lag=0)
        self.assertAlmostEqual(result.ledger.iloc[0].net_return, .05)
        np.testing.assert_allclose(result.end_weights.iloc[0], [11/21, 10/21])
        self.assertAlmostEqual(result.ledger.iloc[1].nav, 1.1)
        self.assertEqual(result.ledger.rebalanced.sum(), 1)

    def test_next_close_execution_excludes_same_day_and_next_day_returns(self):
        returns, targets = self.frames(5)
        returns.iloc[0] = [5., 3.]
        returns.iloc[1] = [4., 2.]
        returns.iloc[2] = [.1, .2]
        result = simulate_portfolio(returns, targets, "daily", 0)
        first = result.ledger.iloc[0]
        self.assertEqual(first.name, returns.index[2])
        self.assertEqual(first.signal_date, returns.index[0])
        self.assertEqual(first.execution_date, returns.index[1])
        self.assertAlmostEqual(first.net_return, .15)

    def test_initial_purchase_cost_is_self_financing(self):
        post, turnover = rebalance_after_cost([0., 0.], [.5, .5], .001)
        self.assertAlmostEqual(post, 1/1.001)
        self.assertAlmostEqual(turnover, post)
        self.assertAlmostEqual(post + .001*turnover, 1.)

    def test_buys_and_sells_cost_actual_dollars(self):
        current, target = np.array([.8, .2]), np.array([.5, .5])
        post, turnover = rebalance_after_cost(current, target, .0025)
        trades = post*target-current
        self.assertAlmostEqual(turnover, abs(trades).sum())
        self.assertAlmostEqual(post + .0025*turnover, 1.)
        self.assertAlmostEqual(post-current.sum()-trades.sum(), 0.)

    def test_drift_requires_turnover_even_if_targets_are_unchanged(self):
        returns, targets = self.frames(4)
        returns.iloc[1] = [.2, 0.]
        result = simulate_portfolio(returns, targets, "daily", 10, execution_lag=0)
        self.assertGreater(result.ledger.iloc[1].traded_notional, 0.)
        self.assertGreater(result.ledger.iloc[1].cost_fraction, 0.)

    def test_missing_held_return_is_not_silently_zero(self):
        returns, targets = self.frames()
        returns.iloc[3, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing held"):
            simulate_portfolio(returns, targets)

    def test_missing_unheld_return_is_allowed(self):
        returns, targets = self.frames()
        targets.loc[:, "B"] = 0.
        targets.loc[:, "A"] = 1.
        returns.loc[:, "B"] = np.nan
        self.assertTrue(np.isfinite(simulate_portfolio(returns, targets).ledger.net_return).all())

    def test_first_loss_counts_as_drawdown(self):
        stats = performance_stats([-.1, 0., 0.])
        self.assertAlmostEqual(stats["max_drawdown"], -.1)
        self.assertAlmostEqual(stats["terminal_wealth"], .9)

    def test_cagr_uses_geometric_growth(self):
        stats = performance_stats([.1, -.1], rf_daily=0)
        self.assertAlmostEqual(stats["cagr"], .99**126-1)
        self.assertAlmostEqual(stats["annualized_arithmetic_return"], 0.)

    def test_week_and_month_schedules_handle_holidays(self):
        dates = pd.DatetimeIndex(["2024-08-30", "2024-09-03", "2024-09-04", "2024-09-09"])
        np.testing.assert_array_equal(rebalance_mask(dates, "weekly"), [True, True, False, True])
        np.testing.assert_array_equal(rebalance_mask(dates, "monthly"), [True, True, False, False])

    def test_future_returns_and_targets_do_not_change_earlier_results(self):
        returns, targets = self.frames(12)
        returns.iloc[:, 0] = np.arange(12)/1000
        first = simulate_portfolio(returns, targets, "daily", 10)
        returns.iloc[8:, 0] = .5
        targets.iloc[8:] = [1., 0.]
        second = simulate_portfolio(returns, targets, "daily", 10)
        pd.testing.assert_frame_equal(first.ledger.iloc[:6], second.ledger.iloc[:6])
        pd.testing.assert_frame_equal(first.end_weights.iloc[:6], second.end_weights.iloc[:6])

    def test_neutral_fallback_and_no_future_eligibility(self):
        returns, base = self.frames(8)
        returns.iloc[:, 0] = np.arange(8)/100
        forecasts = base*.1
        forecasts.iloc[4, 0] = -1.
        forecasts.iloc[4, 1] = np.nan
        neutral, overlay, audit = overlay_targets(returns, forecasts, base, window=3)
        pd.testing.assert_series_equal(neutral.iloc[4], overlay.iloc[4])
        self.assertEqual(audit["nonpositive_forecast_asset_days"], 1)
        self.assertEqual(audit["missing_forecast_asset_days"], 1)
        returns.iloc[6:, 1] = .9
        _, changed, _ = overlay_targets(returns, forecasts, base, window=3)
        pd.testing.assert_frame_equal(overlay.iloc[:6], changed.iloc[:6])

    def test_unavailable_price_history_cannot_enter_weights(self):
        returns, base = self.frames()
        returns.iloc[:4, 1] = np.nan
        forecast = base*.1
        neutral, _, _ = overlay_targets(returns, forecast, base, window=3)
        self.assertEqual(neutral.iloc[5, 1], 0.)
        self.assertGreater(neutral.iloc[6, 1], 0.)

    def test_alignment_and_bad_weight_inputs_are_rejected(self):
        returns, targets = self.frames()
        with self.assertRaises(ValueError):
            simulate_portfolio(returns, targets.iloc[::-1])
        with self.assertRaises(ValueError):
            simulate_portfolio(returns, targets*2)
        with self.assertRaises(ValueError):
            simulate_portfolio(returns, targets, execution_lag=-1)
        with self.assertRaises(ValueError):
            rebalance_after_cost([1., 0.], [np.nan, 0.], .001)


if __name__ == "__main__":
    unittest.main()
