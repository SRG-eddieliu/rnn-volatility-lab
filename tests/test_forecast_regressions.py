"""Regression checks for forecast timing, transforms, and positive reconstruction."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.garch import rolling_garch_forecast, fit_garch_t
from src.models.rnn import (RNNTrainingConfig, _transform_target_by_train_stats,
                            _inverse_target_transform, build_sequence_dataset,
                            reconstruct_log_ratio_variance)


def fake_fit(values):
    return SimpleNamespace(convergence_flag=0,
                           params={"omega":.1,"alpha[1]":.2,"beta[1]":.7,"nu":8.},
                           forecast=lambda **kw: SimpleNamespace(variance=pd.DataFrame([[2.]])))


class GarchTests(unittest.TestCase):
    @patch('src.models.garch.fit_garch_t', side_effect=fake_fit)
    def test_new_return_updates_state_without_refit(self, fit):
        x = pd.Series([.01]*14)
        shocked = x.copy(); shocked.iloc[4] = .10
        a = rolling_garch_forecast(x, 3, 7)
        b = rolling_garch_forecast(shocked, 3, 7)
        self.assertEqual(a.iloc[4],b.iloc[4])
        self.assertGreater(b.iloc[5],a.iloc[5])
        self.assertAlmostEqual(a.iloc[4],(.1+.2*1+.7*2)/10000)
        self.assertAlmostEqual(a.iloc[5],(.1+.2*1+.7*(.1+.2*1+.7*2))/10000)

    @patch('src.models.garch.fit_garch_t', side_effect=fake_fit)
    def test_refit_timing_and_warmup(self, fit):
        x = pd.Series(np.linspace(.01,.02,12),index=pd.date_range('2020-01-01',periods=12))
        diagnostics=[]
        h=rolling_garch_forecast(x,3,4,diagnostics)
        self.assertTrue(h.iloc[:3].isna().all())
        self.assertEqual([len(c.args[0]) for c in fit.call_args_list],[3,7,11])
        self.assertEqual([d['n_fit'] for d in diagnostics],[3,7,11])

    @patch('src.models.garch.fit_garch_t', side_effect=fake_fit)
    def test_nan_index_preserved(self, fit):
        x=pd.Series([.01,.02,np.nan,.01,.03,.02])
        h=rolling_garch_forecast(x,2,9)
        self.assertEqual(list(h.index),list(x.index))
        self.assertTrue(np.isnan(h.iloc[2]))

    def test_input_validation(self):
        for size in [0,-1,2.5,True]:
            with self.assertRaises(ValueError): rolling_garch_forecast(pd.Series([.01]*5),size)
        with self.assertRaises(ValueError): rolling_garch_forecast(pd.Series([.01,.02,.03],index=[2,1,3]),1)
        with self.assertRaises(ValueError): rolling_garch_forecast(pd.Series([.01,np.inf,.02]),1)

    def test_real_arch_fixed_parameter_recursion(self):
        from arch import arch_model
        rng=np.random.default_rng(18)
        values=pd.Series(rng.standard_t(8,size=320)*.01)
        result=fit_garch_t(values.iloc[:300])
        expected=arch_model(values*100,mean='Zero',vol='GARCH',p=1,q=1,dist='t',rescale=False).fix(result.params)
        expected_h=expected.forecast(horizon=1,start=299,reindex=False).variance.iloc[:-1,0].to_numpy()/10000
        with patch('src.models.garch.fit_garch_t',return_value=result):
            actual=rolling_garch_forecast(values,300,100).iloc[300:].to_numpy()
        np.testing.assert_allclose(actual,expected_h,rtol=1e-9,atol=1e-12)


class TargetTests(unittest.TestCase):
    def test_log_roundtrip_above_and_below_floor(self):
        y=np.array([0,1e-10,1e-8,1e-6,1e-4],dtype=np.float32)
        z,_,meta=_transform_target_by_train_stats(y,y,RNNTrainingConfig())
        h=_inverse_target_transform(z,meta)
        np.testing.assert_allclose(h,np.maximum(y,1e-8),rtol=2e-6,atol=1e-14)
        self.assertTrue((h>0).all())

    def test_signed_standardization_roundtrip(self):
        y=np.array([-3,0,2,5],dtype=np.float32)
        cfg=RNNTrainingConfig(target_transform='standardize')
        z,_,meta=_transform_target_by_train_stats(y,y,cfg)
        np.testing.assert_allclose(_inverse_target_transform(z,meta),y,atol=1e-6)

    def test_validation_data_does_not_fit_transform(self):
        y=np.array([1e-6,2e-6,3e-6],dtype=np.float32)
        a,_,ma=_transform_target_by_train_stats(y,y,RNNTrainingConfig())
        b,_,mb=_transform_target_by_train_stats(y,y*100,RNNTrainingConfig())
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ma,mb)

    def test_invalid_targets_fail(self):
        for bad in [np.array([-1.]),np.array([np.inf])]:
            with self.assertRaises(ValueError): _transform_target_by_train_stats(np.array([1.]),bad,RNNTrainingConfig())

    def test_ratio_reconstruction(self):
        base=np.array([1e-4,2e-4,5e-5])
        ratio=np.array([.1,1,10])
        np.testing.assert_allclose(reconstruct_log_ratio_variance(np.log(ratio),base),base*ratio)
        with self.assertRaises(ValueError): reconstruct_log_ratio_variance(np.array([1000.]),np.array([1e-4]))
        with self.assertRaises(ValueError): reconstruct_log_ratio_variance(np.array([0.]),np.array([0.]))


class SequenceTests(unittest.TestCase):
    def test_target_date_not_in_features(self):
        df=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=6),'f':np.arange(6),'y':np.arange(6)*10})
        x,y,dates=build_sequence_dataset(df,['f'],'y',lookback=3)
        np.testing.assert_array_equal(x[0,:,0],[0,1,2])
        self.assertEqual(y[0],30)
        self.assertEqual(pd.Timestamp(dates[0]),df.date.iloc[3])
        df.loc[3:,'f']=999
        changed,_,_=build_sequence_dataset(df,['f'],'y',lookback=3)
        np.testing.assert_array_equal(x[0],changed[0])

    def test_duplicate_dates_rejected(self):
        df=pd.DataFrame({'date':['2020-01-01']*3,'f':[1,2,3],'y':[2,3,4]})
        with self.assertRaises(ValueError): build_sequence_dataset(df,['f'],'y',lookback=1)


if __name__=='__main__':
    unittest.main()
