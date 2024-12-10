"""Unit tests for SSM_utils.py"""
import models.SSM_utils as SSM_utils

import math_utils.math_utils as math_utils
import math_utils.matrix_utils as matrix_utils
import models.LSSM as LSSM
import models.SSM_utils as SSM_utils
from nose import tools as nose_tools
from nose2.tools import params
import numpy as np
import scipy.linalg
import unittest

_SEED = 900
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

def generate_poles(dim):
  mags = np.random.rand(int(dim/2))
  angles = np.random.rand(int(dim/2)) * np.pi
  # Complex conjugate pairs. We need to have them be next to each other for
  # cdf2rdf to work properly.
  true_eigs = np.empty(dim, dtype=np.complex)
  for i in range(int(dim // 2)):
    true_eigs[2*i] = math_utils.polar_to_rectangular(mags[i], angles[i])
    true_eigs[2*i+1] = math_utils.polar_to_rectangular(mags[i], -1*angles[i])
  
  if dim % 2 != 0: # Handle any remaining poles.
    true_eigs[-1] = np.random.rand()
  return true_eigs

def generate_colored_noise_cov(noise_dim, stddev=0.1):
  noise_cov = np.diag(np.ones(noise_dim) * (np.random.rand(noise_dim) * stddev))
  return matrix_utils.make_symmetric(noise_cov @ noise_cov.T)

def test_transform_params():
  w, v = np.linalg.eig(_DEFAULT_PARAMS['A'])
  new_params = SSM_utils.transform_params(_DEFAULT_PARAMS, matrix_utils.inverse(v))
  
  np.testing.assert_almost_equal(new_params['A'], np.diag(w))
  for k in _DEFAULT_PARAMS.keys():
    if k == 'L0':
      nose_tools.ok_(np.all(new_params[k] == _DEFAULT_PARAMS[k]))
    else:
      nose_tools.ok_(np.all(new_params[k] != _DEFAULT_PARAMS[k]))

def test_compute_canonical_transform():
  T = SSM_utils.compute_canonical_transform(_DEFAULT_PARAMS['A'])
  new_params = SSM_utils.transform_params(_DEFAULT_PARAMS, T)
  
  w, v = np.linalg.eig(_DEFAULT_PARAMS['A'])
  wr, _ = scipy.linalg.cdf2rdf(w, v)
  np.testing.assert_almost_equal(new_params['A'], wr)

def test_eigenvalue_blocks_all_conjugates():
  np.random.seed(_SEED)

  true_eigs = generate_poles(6)
  _EXPECTED_INDS = np.array([[0, 2], [2, 4], [4, 6]])

  blk_inds, eig_cnts = SSM_utils.eigenvalue_blocks(true_eigs)
  nose_tools.ok_(np.all(eig_cnts == 2))
  np.testing.assert_equal(blk_inds, _EXPECTED_INDS)

  block_eigs = scipy.linalg.cdf2rdf(true_eigs, np.identity(np.size(true_eigs)))[0]
  blk_inds, eig_cnts = SSM_utils.eigenvalue_blocks(np.diag(block_eigs))
  nose_tools.ok_(np.all(eig_cnts == 2))
  np.testing.assert_equal(blk_inds, _EXPECTED_INDS)

def test_eigenvalue_blocks_one_real():
  np.random.seed(_SEED)

  true_eigs = generate_poles(3)
  _EXPECTED_INDS = np.array([[0, 2], [2, 3]])

  blk_inds, eig_cnts = SSM_utils.eigenvalue_blocks(true_eigs)
  np.testing.assert_equal(eig_cnts, [2, 1])
  np.testing.assert_equal(blk_inds, _EXPECTED_INDS)

  block_eigs = scipy.linalg.cdf2rdf(true_eigs, np.identity(np.size(true_eigs)))[0]
  blk_inds, eig_cnts = SSM_utils.eigenvalue_blocks(np.diag(block_eigs))
  np.testing.assert_equal(eig_cnts, [2, 1])
  np.testing.assert_equal(blk_inds, _EXPECTED_INDS)

@params((10, 1, 0, True), (100, 1, 1, True), (10, 1, 0, False), (100, 1, 1, False), (900, 1, 1, False),
        (10, 2, 0, True), (100, 2, 1, True), (10, 2, 0, False), (100, 2, 1, False), (900, 2, 1, False),
        (10, 4, 0, True), (100, 4, 1, True), (10, 4, 0, False), (100, 4, 1, False), (900, 4, 1, False))
def test_order_SSM_modes_by_correlation(seed, num_unrelated_pole_pairs, 
                                        num_unrelated_real_poles,
                                        related_is_first_block):
  _NUM_SAMPLES = int(1e5)
  np.random.seed(seed)

  # Generate unrelated poles, and concatenate to the related poles.
  total_unrelated_poles = 2 * num_unrelated_pole_pairs  + num_unrelated_real_poles
  unrelated_eigs = generate_poles(total_unrelated_poles)

  # Generate parameters based on ordering.
  unrelated_A_blk = scipy.linalg.cdf2rdf(unrelated_eigs, np.identity(total_unrelated_poles))[0]
  if related_is_first_block:
    A = np.block([
      [_DEFAULT_PARAMS['A'], np.zeros((_DEFAULT_PARAMS['A'].shape[0], total_unrelated_poles))],
      [np.zeros((total_unrelated_poles, _DEFAULT_PARAMS['A'].shape[1])), unrelated_A_blk]
    ])
  else:
    A = np.block([
      [unrelated_A_blk, np.zeros((total_unrelated_poles, _DEFAULT_PARAMS['A'].shape[1]))],
      [np.zeros((_DEFAULT_PARAMS['A'].shape[0], total_unrelated_poles)), _DEFAULT_PARAMS['A']]
    ])

  related_eig_inds = np.arange(_DEFAULT_PARAMS['A'].shape[0])
  if not related_is_first_block:
    related_eig_inds += np.size(unrelated_eigs)

  obs_dim = _DEFAULT_PARAMS['C'].shape[0]
  C = np.zeros((obs_dim, A.shape[0]))
  C[:, related_eig_inds] = _DEFAULT_PARAMS['C']
  Q = generate_colored_noise_cov(A.shape[0])
  R = generate_colored_noise_cov(obs_dim)
  S = np.zeros((A.shape[0], obs_dim)) # Doesn't matter.

  params = {'A': A, 'C': C, 'Q': Q, 'R': R, 'S': S}
  sys = LSSM.update_missing_params_and_construct_LSSM(params)
  Y, X = sys.generate_realization(_NUM_SAMPLES)
  
  # Now actually order eigs based on correlation to signal.
  _, sorted_modes, _ = SSM_utils.order_SSM_modes_by_correlation(sys, Y, Y)

  expected_eig_inds = np.copy(related_eig_inds)
  if not related_is_first_block:
    # We have to subtract out the real pole because all real poles are rearranged
    # to come after the complex conjugate pairs during the canonical transformation.
    # See the implementation of order_SSM_modes_by_correlation for more details.
    expected_eig_inds -= num_unrelated_real_poles

  nose_tools.ok_(np.all(np.isin(sorted_modes[:np.size(expected_eig_inds)],
                                expected_eig_inds)),
    f'Sorted vs Actual: {sorted_modes} vs {expected_eig_inds}')
