import unittest
from scripts.smoke_demo import run_demo


class SmokeDemoTests(unittest.TestCase):
    def test_offline_demo(self):
        result = run_demo()
        self.assertEqual(result["n_common_forecasts"], 120)
        self.assertEqual(result["sequence_count"], 399)
        self.assertTrue(result["future_return_invariance"])
        self.assertEqual(set(result["metrics"]), {"garch", "ewma"})
