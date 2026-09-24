import unittest
from io import StringIO
import numpy as np
import pandas as pd
from scripts.run_additive_residual import reconstruct_additive
from scripts.evaluate_additive_residual import loss_summary


class AdditiveTests(unittest.TestCase):
    def test_zero_correction_is_baseline(self):
        baseline = np.array([1e-4, 4e-4])
        raw, clipped = reconstruct_additive(baseline, np.zeros(2))
        np.testing.assert_array_equal(raw, baseline)
        np.testing.assert_array_equal(clipped, baseline)

    def test_residual_and_raw_variance_errors_are_identical(self):
        y = np.array([0., 2e-4, 1e-3])
        baseline = np.array([1e-4, 3e-4, 5e-4])
        correction = np.array([-2e-4, 1e-4, 3e-4])
        raw, _ = reconstruct_additive(baseline, correction)
        np.testing.assert_allclose((y-raw)**2, ((y-baseline)-correction)**2, rtol=1e-12)

    def test_negative_raw_is_preserved_and_clipping_is_explicit(self):
        raw, clipped = reconstruct_additive([1e-4, 2e-4], [-2e-4, 3e-4])
        self.assertLess(raw[0], 0)
        self.assertEqual(clipped[0], 1e-12)
        self.assertEqual(raw[1], clipped[1])

    def test_invalid_inputs_are_rejected(self):
        for base, correction in [([0.], [1.]), ([np.nan], [1.]), ([1.], [np.inf]), ([1., 2.], [1.])]:
            with self.assertRaises(ValueError):
                reconstruct_additive(base, correction)
        with self.assertRaises(ValueError):
            reconstruct_additive([1.], [1.], floor=0)

    def test_invalid_raw_qlike_is_not_silently_floored(self):
        summary = loss_summary(np.array([1e-4, 2e-4]), np.array([-1e-4, 2e-4]))
        self.assertIsNone(summary['qlike'])
        self.assertEqual(summary['nonpositive_count'], 1)
        self.assertAlmostEqual(summary['mse'], 2e-8)

    def test_clipped_qlike_matches_explicit_formula(self):
        y = np.array([1e-4, 2e-4])
        _, clipped = reconstruct_additive([1e-4, 2e-4], [-2e-4, 0.])
        summary = loss_summary(y, clipped)
        self.assertAlmostEqual(summary['qlike'], float(np.mean(np.log(clipped)+y/clipped)))
        self.assertEqual(summary['floor_count'], 1)

    def test_float32_csv_roundtrip_preserves_reconstruction(self):
        delta = np.array([-0.000123456789, 0.000234567891], dtype=np.float32)
        baseline = np.array([0.00012, 0.00017], dtype=float)
        raw, _ = reconstruct_additive(baseline, delta)
        csv = pd.DataFrame({'baseline': baseline, 'correction': delta, 'raw': raw}).to_csv(index=False)
        restored = pd.read_csv(StringIO(csv), float_precision='round_trip', dtype={'correction': np.float32})
        actual, _ = reconstruct_additive(restored.baseline, restored.correction)
        np.testing.assert_array_equal(actual, restored.raw)


if __name__ == '__main__':
    unittest.main()
