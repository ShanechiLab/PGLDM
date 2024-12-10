"""Unit tests for matrix_utils module."""
import math_utils.matrix_utils as matrix_utils

import evaluation.evaluate as evaluate
import numpy as np
from nose import tools as nose_tools
from nose2.tools import params

_SEED = 900
_TEN_PERCENT = 0.1
_NUM_SAMPLES = int(1e5)
_DEFAULT_DIM = 10

def generate_random_covariance(dim_size, scale=1, diagonal=False, force_psd=False):
  if diagonal:
    covariance = np.diag(np.random.randn(dim_size) * scale)
  else: #  not diagonal
    covariance = np.random.randn(dim_size, dim_size)
    
  covariance = covariance @ covariance.T # make PD
  if not force_psd: return covariance

  num_to_mask = np.random.choice(np.arange(1, int(dim_size * 0.5)))
  mask = np.random.choice(dim_size, size=num_to_mask, replace=False)
  if diagonal:
    covariance[mask, mask] = 0
  else: # not diagonal
    eigs, V = np.linalg.eig(covariance)
    eigs[mask] = 0
    covariance = np.real(matrix_utils.make_PSD(V @ np.diag(eigs) @ np.conjugate(V).T))
  return covariance, mask

def test_inverse():
  A = np.random.rand(3, 5)
  Ainv = matrix_utils.inverse(A, left=True)
  assert np.allclose(A, np.dot(A, np.dot(Ainv, A)))

  Ainv = matrix_utils.inverse(A, left=False)
  assert np.allclose(A, np.dot(np.dot(A, Ainv), A))

def test_extract_block_matching_inds():
  A = np.arange(100).reshape(10, 10)
  inds = [1, 4, 7]
  expected_out = np.array([[11, 14, 17], [41, 44, 47], [71, 74, 77]])
  actual_out = matrix_utils.extract_block(A, inds)
  assert np.array_equal(expected_out, actual_out), 'Out: {0}, Expected: {1}'.format(actual_out, expected_out)

def test_extract_block_different_inds():
  A = np.arange(100).reshape(10, 10)
  inds = [1, 4, 7]
  inds2 = [0, 3, 6]
  expected_out = np.array([[10, 13, 16], [40, 43, 46], [70, 73, 76]])
  actual_out = matrix_utils.extract_block(A, inds, inds2)
  assert np.array_equal(expected_out, actual_out), 'Out: {0}, Expected: {1}'.format(actual_out, expected_out)

def test_extract_diagonal_block_w_inds():
  pass

def test_make_PD_or_PSD_threshold_min_eigs():
  np.random.seed(_SEED)

  _TRUE_RANK = 3
  A_nonsing = np.random.randn(_TRUE_RANK, _TRUE_RANK)
  A_nonsing = matrix_utils.make_symmetric(A_nonsing * A_nonsing.T)
  assert matrix_utils.is_PD(A_nonsing), 'Expected test matrix should be PD.'
  assert np.linalg.matrix_rank(A_nonsing) == _TRUE_RANK, 'Unexpected rank.'

  A = np.zeros((_TRUE_RANK + 1, _TRUE_RANK + 1))
  A[:_TRUE_RANK, :_TRUE_RANK] = A_nonsing
  new_vector = A_nonsing @ np.atleast_2d(np.arange(3)).T
  A[_TRUE_RANK, :np.size(new_vector)] = new_vector.squeeze()
  A[:np.size(new_vector), _TRUE_RANK] = new_vector.squeeze()
  assert not matrix_utils.is_PD(A), 'Input test matrix should not be PD.'
  input_rank = np.linalg.matrix_rank(A)
  assert input_rank == (_TRUE_RANK + 1),\
         'Input rank incorrect: {0}.'.format(input_rank)

  A_pd_enforced = matrix_utils.make_PD(A, threshold_min_eigs=True)
  assert matrix_utils.is_PD(A_pd_enforced), 'Output should be positive definite.'
  output_rank = np.linalg.matrix_rank(A_pd_enforced)
  assert output_rank == _TRUE_RANK,\
         'Unexpected rank {0} != {1}.'.format(output_rank, _TRUE_RANK)

  # make PSD() should also work.
  A_psd_enforced = matrix_utils.make_PSD(A, threshold_min_eigs=True)
  assert matrix_utils.is_PSD(A_psd_enforced), 'Output should be positive semi-definite.'
  output_rank = np.linalg.matrix_rank(A_psd_enforced)
  assert output_rank == _TRUE_RANK,\
         'Unexpected rank {0} != {1}.'.format(output_rank, _TRUE_RANK)

def test_make_PD_or_PSD():
  np.random.seed(_SEED)

  _TRUE_RANK = 3
  A_nonsing = np.random.randn(_TRUE_RANK, _TRUE_RANK)
  A_nonsing = matrix_utils.make_symmetric(A_nonsing * A_nonsing.T)
  assert matrix_utils.is_PD(A_nonsing), 'Expected test matrix should be PD.'
  assert np.linalg.matrix_rank(A_nonsing) == _TRUE_RANK, 'Unexpected rank.'

  A = np.zeros((_TRUE_RANK + 1, _TRUE_RANK + 1))
  A[:_TRUE_RANK, :_TRUE_RANK] = A_nonsing
  new_vector = A_nonsing @ np.atleast_2d(np.arange(3)).T
  A[_TRUE_RANK, :np.size(new_vector)] = new_vector.squeeze()
  A[:np.size(new_vector), _TRUE_RANK] = new_vector.squeeze()
  assert not matrix_utils.is_PD(A), 'Input test matrix should not be PD.'
  input_rank = np.linalg.matrix_rank(A)
  assert input_rank == (_TRUE_RANK + 1),\
   'Input rank incorrect: {0} != {1}.'.format(input_rank, _TRUE_RANK + 1)

  # Because it isn't full rank and we aren't thresholding eigenvalue the best we
  # can do it make it PSD.
  A_psd_enforced = matrix_utils.make_PD(A, threshold_min_eigs=True)
  assert matrix_utils.is_PSD(A_psd_enforced), 'Output should be positive semi-definite.'
  output_rank = np.linalg.matrix_rank(A_psd_enforced)
  assert output_rank == _TRUE_RANK,\
         'Unexpected rank {0} != {1}.'.format(output_rank, _TRUE_RANK)

  # Using make_PSD() should also work.
  A_psd_enforced = matrix_utils.make_PSD(A, threshold_min_eigs=True)
  assert matrix_utils.is_PSD(A_psd_enforced), 'Output should be positive semi-definite.'
  output_rank = np.linalg.matrix_rank(A_psd_enforced)
  assert output_rank == _TRUE_RANK,\
         'Unexpected rank {0} != {1}.'.format(output_rank, _TRUE_RANK)

@params(10, 100, 500)
def test_coloring_whitening_diagonal_covariance_PD(seed):
  np.random.seed(seed)

  covariance = generate_random_covariance(_DEFAULT_DIM, diagonal=True)

  nose_tools.ok_(matrix_utils.is_PD(covariance))

  coloring = matrix_utils.coloring_matrix(covariance)
  resultant_mat = np.random.randn(_NUM_SAMPLES, _DEFAULT_DIM) @ np.conjugate(coloring).T
  resultant_covariance = np.cov(resultant_mat.T, ddof=1)

  err = evaluate.matrix_error_norm(covariance, resultant_covariance)
  nose_tools.ok_(err < _TEN_PERCENT, f'Error {err} > {_TEN_PERCENT}')

  whitening = matrix_utils.whitening_matrix(resultant_covariance)
  whitened_mat = resultant_mat @ np.conjugate(whitening).T
  whitened_covariance = np.cov(whitened_mat.T, ddof=1)

  np.testing.assert_almost_equal(np.identity(_DEFAULT_DIM), whitened_covariance)

@params(10, 100, 500)
def test_coloring_whitening_off_diagonal_covariance_PD(seed):
  np.random.seed(seed)

  covariance = generate_random_covariance(_DEFAULT_DIM, diagonal=False)

  nose_tools.ok_(matrix_utils.is_PD(covariance))

  coloring = matrix_utils.coloring_matrix(covariance)
  resultant_mat = np.random.randn(_NUM_SAMPLES, _DEFAULT_DIM) @ np.conjugate(coloring).T
  resultant_covariance = np.cov(resultant_mat.T, ddof=1)

  err = evaluate.matrix_error_norm(covariance, resultant_covariance)
  nose_tools.ok_(err < _TEN_PERCENT, f'Error {err} > {_TEN_PERCENT}')

  whitening = matrix_utils.whitening_matrix(resultant_covariance)
  whitened_mat = resultant_mat @ np.conjugate(whitening).T
  whitened_covariance = np.cov(whitened_mat.T, ddof=1)

  np.testing.assert_almost_equal(np.identity(_DEFAULT_DIM), whitened_covariance)

@params(10, 100, 500)
def test_coloring_whitening_diagonal_covariance_PSD(seed):
  np.random.seed(seed)

  covariance, mask = generate_random_covariance(_DEFAULT_DIM, diagonal=True, force_psd=True)
  
  nose_tools.ok_(~matrix_utils.is_PD(covariance))

  coloring = matrix_utils.coloring_matrix(covariance)
  resultant_mat = np.random.randn(_NUM_SAMPLES, _DEFAULT_DIM) @ np.conjugate(coloring).T
  resultant_covariance = np.cov(resultant_mat.T, ddof=1)

  err = evaluate.matrix_error_norm(covariance, resultant_covariance)
  nose_tools.ok_(err < _TEN_PERCENT, f'Error {err} > {_TEN_PERCENT}')

  whitening = matrix_utils.whitening_matrix(resultant_covariance)
  whitened_mat = resultant_mat @ np.conjugate(whitening).T
  whitened_covariance = np.cov(whitened_mat.T, ddof=1)

  # Because the test matrix was PSD and we explicitly set some eigenvalues to be
  # equal to 0, we expect some elements to be 0 in the whitened matrix.
  num_zero = np.size(mask)
  expected_covariance = np.block([
    [np.identity(_DEFAULT_DIM - num_zero), np.zeros((_DEFAULT_DIM - num_zero, num_zero))],
    [np.zeros((num_zero, _DEFAULT_DIM - num_zero)), np.zeros((num_zero, num_zero))]
  ])

  np.testing.assert_almost_equal(expected_covariance, whitened_covariance)

@params(10, 200, 500)
def test_coloring_whitening_off_diagonal_covariance_PSD(seed):
  np.random.seed(seed)

  covariance, mask = generate_random_covariance(_DEFAULT_DIM, diagonal=False, force_psd=True)

  nose_tools.ok_(~matrix_utils.is_PD(covariance))

  coloring = matrix_utils.coloring_matrix(covariance)
  resultant_mat = np.random.randn(_NUM_SAMPLES, _DEFAULT_DIM) @ np.conjugate(coloring).T
  resultant_covariance = np.cov(resultant_mat.T, ddof=1)

  err = evaluate.matrix_error_norm(covariance, resultant_covariance)
  nose_tools.ok_(err < _TEN_PERCENT, f'Error {err} > {_TEN_PERCENT}')

  whitening = matrix_utils.whitening_matrix(resultant_covariance)
  whitened_mat = resultant_mat @ np.conjugate(whitening).T
  whitened_covariance = np.cov(whitened_mat.T, ddof=1)

  # Unlike the diagonal PSD test in which some eigenvalues were directly set to
  # 0, here we approximately made a PSD matrix so the whitening can achieve a
  # covariance close to identity.
  diagonal_terms = np.sum(np.abs(np.diag(whitened_covariance) - 1) <= 1e-9)
  nose_tools.ok_(diagonal_terms == _DEFAULT_DIM - np.size(mask),
                 f'Diagonal terms: {diagonal_terms}')
