"""Abstract class for models, static or dynamical.

Supports:
	+ Dynamical state-space models.
	+ Direct regression models.
	+ Static models (i.e., PCA).
"""
# Needed to use the class itself as a typing annotation.
# https://peps.python.org/pep-0484/#the-problem-of-forward-declarations
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
# import numpy.typing as npt # Python >= 3.9
from typing import Tuple, Union

class Model(ABC):
	"""Model abstract class."""
	@abstractmethod
	def fit(self,  regressor: np.ndarray, data: np.ndarray, **kwargs) -> bool:
		pass

	# Will need to experiment with the @overload annotation to generalize to
	# variable number of outputs.
	# https://stackoverflow.com/questions/54747253/how-to-annotate-function-that-takes-a-tuple-of-variable-length-variadic-tuple
	@abstractmethod
	def predict(self, Y: np.ndarray, **kwargs):
		pass

	@abstractmethod
	def estimate_similarity_transform(self, other_sys : Model) -> np.ndarray:
		pass

	@abstractmethod
	def get_list_of_params(self) -> dict:
		pass

	@abstractmethod
	def update_params(self, new_params : dict) -> bool:
		pass
