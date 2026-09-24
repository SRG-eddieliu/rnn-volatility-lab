import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from scripts.evaluate_corrected_experiment import bootstrap_mean_intervals
from scripts.run_all_models import verify_code_signature


class BootstrapTests(unittest.TestCase):
    def test_constant_paired_difference(self):
        data=np.tile([2.,-3.],(100,1))
        bounds=bootstrap_mean_intervals(data,reps=100)
        np.testing.assert_allclose(bounds,[[2.,-3.],[2.,-3.]])

    def test_seed_and_shape(self):
        data=np.random.default_rng(5).normal(size=(200,3))
        a=bootstrap_mean_intervals(data,reps=100,seed=7)
        b=bootstrap_mean_intervals(data,reps=100,seed=7)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(2,3))
        self.assertTrue((a[0]<=a[1]).all())

    def test_bad_data_rejected(self):
        with self.assertRaises(ValueError): bootstrap_mean_intervals(np.array([[np.nan],[1.]]))

    def test_changed_source_rejected_before_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'model.py'
            source.write_text('value = 1\n')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            (root/'input_signature.json').write_text(json.dumps({'code_sha256':{'model.py':digest}}))
            verify_code_signature(root,root)
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError,'Source changed'):
                verify_code_signature(root,root)


if __name__=='__main__': unittest.main()
