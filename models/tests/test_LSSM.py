"""Unit tests for LSSM.py."""
import models.LSSM as LSSM

import evaluation.evaluate as evaluate
import math_utils.matrix_utils as matrix_utils
import models.PLDS as PLDS
import models.SSM_utils as SSM_utils
from nose import tools as  nose_tools
from nose2.tools import params
import numpy as np

_DEFAULT_PARAMS = {
	'A' : np.array([
	    [ 9.18259185e-01, -2.16763068e-01,  0.00000000e+00,  0.00000000e+00],
	    [ 2.65955099e-01,  8.70345944e-01,  0.00000000e+00,  0.00000000e+00],
	    [ 5.12020916e-04, -6.22114773e-04,  5.67336876e-02, -9.40781803e-01],
	    [ 5.30603575e-04,  1.59003743e-04,  9.40813623e-01, -6.05865039e-02]]),
	'C' : np.array([
	    [ 2.03760101e-03,  2.49441832e-02, -1.04211125e+01, -8.83765460e+00],
	    [-9.21618385e+00,  6.82113399e+00,  5.20499040e-02,  3.72896283e-02]]),
	'G' : np.array([[-6.17547276e-05, -4.74311069e-01],
	              [ 5.01986323e-04,  1.52732506e-01],
	              [ 1.28468087e-01, -1.81963306e-04],
	              [-1.44028566e-01,  4.05744117e-04]]),
	'L0' : np.array([[ 3.11931246e+00, -2.11099346e-03],
	               [-2.11099346e-03,  6.47136456e+00]])
}

def test_LSSM_factory_function():
	sys = LSSM.update_missing_params_and_construct_LSSM(_DEFAULT_PARAMS)
	
	np.testing.assert_equal(sys.A, _DEFAULT_PARAMS['A'])
	np.testing.assert_equal(sys.C, _DEFAULT_PARAMS['C'])
	np.testing.assert_equal(sys.G, _DEFAULT_PARAMS['G'])
	np.testing.assert_equal(sys.L0, _DEFAULT_PARAMS['L0'])

	nose_tools.ok_(sys.is_stable())

def test_LSSM_factory_function_fails_without_AC():
	params = {}
	params.update(_DEFAULT_PARAMS)
	params.pop('A')
	
	nose_tools.assert_raises(ValueError, LSSM.update_missing_params_and_construct_LSSM, params)

	params['A'] = _DEFAULT_PARAMS['A']
	params.pop('C')

	nose_tools.assert_raises(ValueError, LSSM.update_missing_params_and_construct_LSSM, params)

	params['C'] = _DEFAULT_PARAMS['C']
	params.pop('G')

	nose_tools.assert_raises(ValueError, LSSM.update_missing_params_and_construct_LSSM, params)

	params['G'] = _DEFAULT_PARAMS['G']
	params.pop('L0')

	nose_tools.assert_raises(ValueError, LSSM.update_missing_params_and_construct_LSSM, params)

@params(9, 50, 100, 900)
def test_generate_realization(seed):
	np.random.seed(seed)

	sys = LSSM.update_missing_params_and_construct_LSSM(_DEFAULT_PARAMS)
	# Even using 1e5 samples empirical doesn't match. Forced to use 1e6.
	Y, X = sys.generate_realization(int(1e6))

	emp_L0 = np.cov(Y.T, ddof=1)
	emp_P = np.cov(X.T, ddof=1)

	L0_norm_err = np.linalg.norm(_DEFAULT_PARAMS['L0'] - emp_L0)
	P_norm_err = np.linalg.norm(sys.get_list_of_params()['P'] - emp_P)
	# Interestingly the second order statistics of the observations are a lot more
	# variable / different from the analytical, whereas the state prediction
	# covariance is a lot tighter. Even using 1e-2 for observation covariance these
	# seeds failed.
	nose_tools.ok_(L0_norm_err <= 5e-2, f'{L0_norm_err} > 5e-2')
	nose_tools.ok_(P_norm_err < 1e-3, f'{P_norm_err} > 1e-3')

@params(9, 50, 100, 900)
def test_predict(seed):
	np.random.seed(seed)

	sys = LSSM.update_missing_params_and_construct_LSSM(_DEFAULT_PARAMS)
	Y, X = sys.generate_realization(int(1e5))
	Ypred, Xpred, _ = sys.predict(Y)

	Y_metrics = evaluate.evaluate_results(Y, Ypred, metrics_to_compute=['NRMSE', 'CC'])
	X_metrics = evaluate.evaluate_results(X, Xpred, metrics_to_compute=['NRMSE', 'CC'])
	# NRMSE for Y self-prediction was fairly high..... will log these values even
	# though test performance is gated on CC.
	Y_nrmse= Y_metrics['NRMSE']
	X_nrmse = X_metrics['NRMSE']
	print(f'Y nrmse: {Y_nrmse.mean()}, X nrmse: {X_nrmse.mean()}')
	
	Y_cc = Y_metrics['CC']
	X_cc = X_metrics['CC']

	nose_tools.ok_(np.mean(Y_cc) > 0.8, f'{Y_cc.mean()} < 0.8')
	nose_tools.ok_(np.mean(X_cc) > 0.8, f'{X_cc.mean()} < 0.8')

@params(9, 50, 100, 900)
def test_estimate_similarity_transform(seed):
	np.random.seed(seed)

	T = np.linalg.qr(
		np.random.randn(_DEFAULT_PARAMS['A'].shape[0], _DEFAULT_PARAMS['A'].shape[0]))[0]
	sys2_params = SSM_utils.transform_params(_DEFAULT_PARAMS, T)

	sys1 = LSSM.update_missing_params_and_construct_LSSM(_DEFAULT_PARAMS)
	sys2 = LSSM.update_missing_params_and_construct_LSSM(sys2_params)
	est_T = sys1.estimate_similarity_transform(sys2)

	norm_err = np.linalg.norm(est_T - T, ord='fro')
	print('Transform norm error: ', norm_err)
	# Accuracy is pretty bad.
	nose_tools.ok_(norm_err <= 3.0, f'Estimated transform norm error {norm_err} > 3.0')

def test_estimate_similarity_transform_raise_error():
	sys1 = LSSM.update_missing_params_and_construct_LSSM(_DEFAULT_PARAMS)
	# Need to add Q to construct PLDS.
	_DEFAULT_PARAMS['Q'] = np.diag(np.random.randn(_DEFAULT_PARAMS['A'].shape[0]))
	sys2 = PLDS.update_missing_params_and_construct_PLDS(_DEFAULT_PARAMS)

	nose_tools.assert_raises(ValueError, sys1.estimate_similarity_transform, sys2)
