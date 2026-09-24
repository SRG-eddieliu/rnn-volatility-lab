"""Optional deterministic CPU integration tests: RUN_TF_TESTS=1 python -m unittest discover -s tests."""
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.pipeline import generate_rolling_splits
from src.models.rnn import RNNTrainingConfig, run_rolling_experiment


@unittest.skipUnless(os.environ.get('RUN_TF_TESTS')=='1','Set RUN_TF_TESTS=1 for TensorFlow integration tests.')
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        tf.get_logger().setLevel('ERROR')
        rng=np.random.default_rng(5)
        returns=rng.normal(0,.01,80)
        cls.df=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=80),'log_return':returns,
                             'sq_return':returns**2,'abs_return':abs(returns),'rv_21d':np.full(80,.16)})
        cls.splits=generate_rolling_splits(cls.df,min_train_size=35,val_size=10,test_size=7,step_size=7).iloc[:2]
        cls.cfg=RNNTrainingConfig(lookback=5,hidden_units=2,epochs=3,patience=2,batch_size=16,seed=7,unroll=True)

    def test_both_architectures_and_checkpoint_losses(self):
        for arch in ['lstm','gru']:
            pred,logs,gates=run_rolling_experiment(self.df,self.splits,arch,'pure',self.cfg,capture_gates=True)
            self.assertEqual(len(pred),14)
            self.assertEqual(logs.n_train.tolist(),[30,37])
            self.assertTrue((pred.y_pred_var>0).all())
            self.assertEqual(len(gates),14*5*(4 if arch=='lstm' else 3))
            np.testing.assert_allclose(logs.checkpoint_val_loss,logs.best_val_loss,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(logs.checkpoint_gap_val_minus_train,
                                       logs.checkpoint_val_loss-logs.checkpoint_train_loss)

    def test_repeatable_initialization(self):
        a,_=run_rolling_experiment(self.df,self.splits.iloc[:1],'lstm','pure',self.cfg)
        b,_=run_rolling_experiment(self.df,self.splits.iloc[:1],'lstm','pure',self.cfg)
        np.testing.assert_array_equal(a.y_pred_var,b.y_pred_var)

    def test_additive_target_stays_signed_through_training(self):
        frame = self.df.copy()
        frame['garch_cond_var'] = 0.01
        frame['residual_var'] = frame.sq_return-frame.garch_cond_var
        config = replace(self.cfg, target_transform='standardize')
        predictions, logs = run_rolling_experiment(
            frame, self.splits.iloc[:1], 'lstm', 'hybrid_residual', config,
            target_col='residual_var', output_activation='linear')
        self.assertTrue((predictions.y_true_var < 0).all())
        self.assertTrue((predictions.y_pred_var < 0).all())
        self.assertTrue(logs.target_transform.eq('standardize').all())
        self.assertTrue(logs.output_activation_used.eq('linear').all())

    def test_incomplete_checkpoint_is_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pred.csv'; log=Path(d)/'log.csv'; gates=Path(d)/'gates.csv'
            kwargs=dict(df=self.df,splits_df=self.splits.iloc[:1],architecture='gru',variant='pure',cfg=self.cfg,
                        capture_gates=True,predictions_path=p,train_logs_path=log,gates_path=gates)
            expected,_,_=run_rolling_experiment(**kwargs)
            log.unlink(); gates.unlink()
            actual,logs,captured=run_rolling_experiment(**kwargs,resume=True)
            np.testing.assert_array_equal(expected.y_pred_var,actual.y_pred_var)
            self.assertEqual(len(pd.read_csv(p)),7)
            self.assertEqual(len(logs),1)
            self.assertEqual(len(captured),7*5*3)


if __name__=='__main__': unittest.main()
