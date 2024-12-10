"""Unit tests for PLDS.py"""
import models.PLDS as PLDS

import evaluation.evaluate as evaluate
import math_utils.matrix_utils as matrix_utils
import models.LSSM as LSSM
import models.SSM_utils as SSM_utils
from nose import tools as  nose_tools
from nose2.tools import params
import numpy as np
import scipy.linalg

_DEFAULT_SEED = 900
_DEFAULT_OBS_DIM = 2

# Mostly needed for constructing the LSSM in one of the estimate_transform_tests
# below.
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

def generate_random_baseline(obs_dim):
	# Generate a random number between 0 to 1 -> convert to log-scale.
	return np.log(np.random.rand(obs_dim)[:, np.newaxis])

def generate_colored_noise_cov(noise_dim, stddev=0.1):
  noise_cov = np.diag(np.random.rand(noise_dim) * stddev)
  return matrix_utils.make_symmetric(noise_cov @ noise_cov.T)

def generate_observation_matrix(state_dim, obs_dim):
	return np.random.randn(obs_dim, state_dim)

def complete_params_list(params):
	params['Q'] = generate_colored_noise_cov(params['A'].shape[0])
	# Will overwrite the default parameters here.
	params['C'] = generate_observation_matrix(params['A'].shape[0], _DEFAULT_OBS_DIM)
	XCov = scipy.linalg.solve_discrete_lyapunov(params['A'], params['Q'])
	params['XCov'] = XCov
	params['L0'] = params['C'] @ XCov @ params['C'].T
	params['G'] = params['A'] @ XCov @ params['C'].T
	params['d'] = generate_random_baseline(_DEFAULT_OBS_DIM)
	return params

def test_PLDS_factory_function():
	np.random.seed(_DEFAULT_SEED)
	params = complete_params_list(_DEFAULT_PARAMS)

	sys = PLDS.update_missing_params_and_construct_PLDS(_DEFAULT_PARAMS)
	
	np.testing.assert_equal(sys.A, params['A'])
	np.testing.assert_equal(sys.C, params['C'])
	np.testing.assert_equal(sys.Q, params['Q'])
	np.testing.assert_equal(sys.d, params['d'])

	nose_tools.ok_(sys.is_stable())

def test_PLDS_factory_function_fails_without_ACQ_but_d_optional():
	np.random.seed(_DEFAULT_SEED)
	orig_params = complete_params_list(_DEFAULT_PARAMS)

	params = {}
	params.update(orig_params)
	params.pop('A')
	
	nose_tools.assert_raises(ValueError, PLDS.update_missing_params_and_construct_PLDS, params)

	params['A'] = orig_params['A']
	params.pop('C')

	nose_tools.assert_raises(ValueError, PLDS.update_missing_params_and_construct_PLDS, params)

	params['C'] = orig_params['C']
	params.pop('Q')

	nose_tools.assert_raises(ValueError, PLDS.update_missing_params_and_construct_PLDS, params)

	params['Q'] = orig_params['Q']
	params.pop('d')

	sys = PLDS.update_missing_params_and_construct_PLDS(params)
	nose_tools.ok_(np.all(sys.d == 0)) # Should be default if not provided.

@params(9, 50, 100, 900)
def test_generate_realization(seed):
	np.random.seed(seed)
	params = complete_params_list(_DEFAULT_PARAMS)

	sys = PLDS.update_missing_params_and_construct_PLDS(params)
	# Even using 1e5 samples empirical doesn't match. Forced to use 1e6.
	R, X, Y = sys.generate_realization(int(1e6), rates_only=False)

	LR = np.log(R) # Rates -> log rates.
	LRmean = np.mean(LR, axis=0, keepdims=True).T
	Rmean = np.mean(R, axis=0)
	Ymean = np.mean(Y, axis=0)
	np.testing.assert_almost_equal(LRmean, params['d'], decimal=2)
	np.testing.assert_almost_equal(Ymean, Rmean, decimal=2)

	# Interestingly the second order statistics of the observations are a lot more
	# variable / different from the analytical, whereas the state prediction
	# covariance is a lot tighter. Even using 1e-2 for observation covariance these
	# seeds failed.	
	exp_XCov = np.cov(X.T, ddof=1)
	XCov_norm_err = np.linalg.norm(params['XCov'] - exp_XCov)
	nose_tools.ok_(XCov_norm_err < 1e-3, f'{XCov_norm_err} > 1e-3')

	exp_RCov = np.cov(LR.T, ddof=1)
	RCov_norm_err = np.linalg.norm(params['L0'] - exp_RCov)
	nose_tools.ok_(RCov_norm_err < 1e-2, f'{RCov_norm_err} > 1e-2')

@params(9, 50, 100, 900)
def test_predict(seed):
	np.random.seed(seed)
	params = complete_params_list(_DEFAULT_PARAMS)

	sys = PLDS.update_missing_params_and_construct_PLDS(params)
	R, X, Y = sys.generate_realization(int(1e5), rates_only=False)

	# Point process realization has high variance with large number of occurrences
	# per time bin. Using the probability (i.e., rate * time) for the PPF instead.
	# Rpred, Xpred, _ = sys.predict(Y)
	Rpred, Xpred, _ = sys.predict(R)

	R_metrics = evaluate.evaluate_results(R, Rpred, metrics_to_compute=['NRMSE', 'CC'])
	X_metrics = evaluate.evaluate_results(X, Xpred, metrics_to_compute=['NRMSE', 'CC'])

	R_nrmse= R_metrics['NRMSE']
	X_nrmse = X_metrics['NRMSE']
	print(f'R nrmse: {R_nrmse.mean()}, X nrmse: {X_nrmse.mean()}')
	
	R_cc = R_metrics['CC']
	X_cc = X_metrics['CC']
	print(f'R cc: {R_cc.mean()}, X cc: {X_cc.mean()}')

	nose_tools.ok_(np.mean(R_cc) >= 0.5, f'{R_cc.mean()} < 0.5')
	nose_tools.ok_(np.mean(X_cc) >= 0.5, f'{X_cc.mean()} < 0.5')

@params(9, 100) # Default seeds 50 and 900 generate invalid parameter set.
def test_estimate_similarity_transform(seed):
	np.random.seed(seed)

	params = complete_params_list(_DEFAULT_PARAMS)
	T = np.linalg.qr(np.random.randn(params['A'].shape[0], params['A'].shape[0]))[0]
	sys2_params = SSM_utils.transform_params(params, T)

	sys1 = PLDS.update_missing_params_and_construct_PLDS(params)
	sys2 = PLDS.update_missing_params_and_construct_PLDS(sys2_params)
	est_T = sys1.estimate_similarity_transform(sys2)

	norm_err = np.linalg.norm(est_T - T, ord='fro')
	print('Transform norm error: ', norm_err)
	# Accuracy is pretty bad.
	nose_tools.ok_(norm_err <= 3.0, f'Estimated transform norm error {norm_err} > 3.0')

@params(9, 100) # Default seeds 50 and 900 generate invalid parameter set.
def test_estimate_similarity_transform_PLDS_vs_LSSM(seed):
	np.random.seed(seed)

	params = complete_params_list(_DEFAULT_PARAMS)
	T = np.linalg.qr(np.random.randn(params['A'].shape[0], params['A'].shape[0]))[0]
	sys2_params = SSM_utils.transform_params(params, T)

	sys1 = PLDS.update_missing_params_and_construct_PLDS(params)
	sys2_PLDS = PLDS.update_missing_params_and_construct_PLDS(sys2_params)
	sys2_LSSM = LSSM.update_missing_params_and_construct_LSSM(sys2_params)
	
	est_T_PLDS = sys1.estimate_similarity_transform(sys2_PLDS)
	est_T_LSSM = sys1.estimate_similarity_transform(sys2_LSSM)

	norm_err_PLDS = np.linalg.norm(est_T_PLDS - T, ord='fro')
	norm_err_LSSM = np.linalg.norm(est_T_LSSM - T, ord='fro')
	print(f'Transform norm error PLDS={norm_err_PLDS} and LSSM= {norm_err_LSSM}')

	nose_tools.ok_(norm_err_PLDS < norm_err_LSSM, f'PLDS err {norm_err_PLDS} > LSSM err {norm_err_LSSM}')
