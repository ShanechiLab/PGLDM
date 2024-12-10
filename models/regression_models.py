"""Regression models (OLS, Ridge, LASSO)."""
from enum import Enum
import models.model_base as model_base
from sklearn.linear_model import LinearRegression, RidgeCV

class RegressionMethod(Enum):
  OLS_REG = 1
  RIDGE_REG = 2

class RegressionModel(model_base.Model):
	def __init__(self, regression_model_type):
		self.regression_model_type = regression_model_type

	def fit(self, regressor, data, **kwargs):
		"""Data should be time-by-features."""
		if self.regression_model_type == RegressionMethod.RIDGE_REG:
			self.model = RidgeCV(**kwargs)
			self.model.fit(regressor, data)
		elif self.regression_model_type == RegressionMethod.OLS_REG:
			self.model = LinearRegression(**kwargs)
			self.model.fit(regressor, data)
		else:
			raise NotImplementError('Only ridge regression supported now for decoding.')
		return True

	def predict(self, data):
		"""Predict using regression model.

		Args:
			data: np.ndarray. Regressor data of dimension (num_samples, features).
		"""
		return self.model.predict(data)

	def get_params(self):
		return self.model.get_params()

	def weights(self):
		return self.model.coef_, self.model.intercept_

	def alpha(self):
		if self.regression_model_type == RegressionMethod.RIDGE_REG:
			return self.model.alpha_
		return None # no other regression model has an alpha value

	def get_list_of_params(self):
		return self.weights()

	def update_params(self, new_params):
		raise NotImplementError('No support for updating params yet.')

	# Abstract method, needs implementation, but similarity transform doesn't
	# exist for regression models.
	def estimate_similarity_transform(self, _):
		pass
