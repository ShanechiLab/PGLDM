"""Tests for model_base module."""
import models.model_base as model_base

from nose import tools as nose_tools
import numpy as np
import unittest

def test_model_base():
	class SampleModel(model_base.Model):
		def fit(self, regressor, data): return False
		def predict(self, Y): return (np.random.empty(10), np.random.empty(10))
		def get_list_of_params(): return {}