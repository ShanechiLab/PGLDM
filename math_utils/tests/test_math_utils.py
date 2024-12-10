"""Unit tests for math_utils.py."""
import math_utils.math_utils as math_utils

import evaluation.evaluate as evaluate
import math_utils.matrix_utils as matrix_utils
import numpy as np
from nose import tools as nose_tools
from nose2.tools import params

_SEED = 900

@params(2, 4, 10, 100)
def test_log_normal_moment_transform(dim):
	np.random.seed(_SEED)

	log_mean = np.random.randn(dim, 1)
	log_covariance = matrix_utils.generate_PD_matrix(dim)
	log_covariance /= np.sqrt(np.linalg.norm(log_covariance))

	mean, covariance = math_utils.log_normal_moment_transform(log_mean, log_covariance)
	
	# Squeezing makes it easier to do element by element checks.
	mean = mean.squeeze()
	log_mean = log_mean.squeeze()
	for i in range(dim):
		nose_tools.ok_(np.exp(0.5 * log_covariance[i, i] + log_mean[i]) == mean[i])

	# Because we've already verified the mean, will use it directly here rather
	# than recomputing.
	for i in range(dim):
		expected_covii = np.exp(2 * log_covariance[i, i] + 2 * log_mean[i])
		expected_covii += mean[i] - np.power(mean[i], 2)
		nose_tools.ok_(abs(expected_covii - covariance[i, i]) < 1e-9,
									f'Expected covii {expected_covii} vs actual {covariance[i, i]}')

		for j in range(i+1, dim):
			expected_covij = log_covariance[i, j]
			expected_covij += 0.5 * log_covariance[i, i] + 0.5 * log_covariance[j, j]
			expected_covij += log_mean[i] + log_mean[j]
			expected_covij = np.exp(expected_covij) - mean[i] * mean[j]
			nose_tools.ok_(abs(expected_covij - covariance[i, j]) < 1e-9,
								f'Expected covij {expected_covij} vs actual {covariance[i, j]}')
			nose_tools.ok_(abs(expected_covij - covariance[j, i]) < 1e-9,
								f'Expected covji {expected_covij} vs actual {covariance[j, i]}')

def test_to_from_log_normal_transform_single_variables():
	np.random.seed(_SEED)
	_NUM_SAMPLES = 100

	log_mean, log_var = np.random.randn(_NUM_SAMPLES, 1), np.random.rand(_NUM_SAMPLES, 1)
	mean, var = math_utils.transform_from_log_normal_single_variable(log_mean, log_var)
	out_log_mean, out_log_var = math_utils.transform_to_log_normal_single_variable(mean, var)
	
	np.testing.assert_almost_equal(log_mean, out_log_mean)
	np.testing.assert_almost_equal(log_var, out_log_var)

	log_mean, log_std = np.random.randn(_NUM_SAMPLES, 1), np.random.rand(_NUM_SAMPLES, 1)
	mean, var = math_utils.transform_from_log_normal_single_variable(log_mean, np.power(log_std, 2))
	out_log_mean, out_log_var = math_utils.transform_to_log_normal_single_variable(mean, var)
	
	np.testing.assert_almost_equal(log_mean, out_log_mean)
	np.testing.assert_almost_equal(log_std, np.sqrt(out_log_var))
