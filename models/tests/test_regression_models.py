"""Tests for regresion_models module."""
import models.regression_models as regression_models

from nose import tools as nose_tools
import numpy as np
import unittest

_SEED = 90
_NUM_SAMPLES = 100
_NUM_INPUT_FEATURES = 2
_NUM_OUTPUT_FEATURES = 5

### NOTE: These tests do not test the actual accuracy of the regression model
### since internally it is using sklearn which should have its own unit tests.
### IF, however, there is strange behavior resulting then those unit tests should
### be added. The current set of tests are predominantly integration tests.

def generate_data():
	np.random.seed(_SEED)
	X = np.random.randn(_NUM_INPUT_FEATURES, _NUM_SAMPLES)
	A = np.random.rand(_NUM_OUTPUT_FEATURES, _NUM_INPUT_FEATURES)
	Y = A @ X
	return X, Y, A

def test_regression_models_OLS():
	X, Y, A = generate_data()
	regression = regression_models.RegressionModel(regression_models.RegressionMethod.OLS_REG)
	nose_tools.ok_(regression.fit(X.T, Y.T))

	test_Y = regression.predict(X.T).T
	np.testing.assert_almost_equal(Y, test_Y)

	test_A, bias = regression.weights()
	np.testing.assert_almost_equal(bias.squeeze(), np.zeros(_NUM_OUTPUT_FEATURES))
	np.testing.assert_almost_equal(A, test_A)

	nose_tools.ok_(regression.alpha() is None)

def test_regression_models_RIDGE():
	_DEFAULT_ALPHA = 1e-3
	reg_kwargs = {'alphas': _DEFAULT_ALPHA, 'fit_intercept': True}

	X, Y, A = generate_data()
	regression = regression_models.RegressionModel(regression_models.RegressionMethod.RIDGE_REG)
	nose_tools.ok_(regression.fit(X.T, Y.T, **reg_kwargs))

	test_Y = regression.predict(X.T).T
	np.testing.assert_almost_equal(Y, test_Y, decimal=_DEFAULT_ALPHA)

	test_A, bias = regression.weights()
	np.testing.assert_almost_equal(bias.squeeze(), np.zeros(_NUM_OUTPUT_FEATURES),
																 decimal=1e-6)
	np.testing.assert_almost_equal(A, test_A, _DEFAULT_ALPHA)

	nose_tools.ok_(regression.alpha() == _DEFAULT_ALPHA)