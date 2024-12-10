"""Unit tests for evaluate module."""
import evaluation.evaluate as evaluate

import math_utils.math_utils as math_utils
import math_utils.matrix_utils as matrix_utils
from nose import tools as nose_tools
from nose2.tools import params
import numpy as np
import unittest

_SEED = 900
_NUM_DEFAULT_SAMPLES = 100
_NUM_DEFAULT_VARIABLES = 5
_DEFAULT_METRICS = ['CC']

def generate_true_and_noisy():
  true = np.random.randn(_NUM_DEFAULT_SAMPLES)
  noisy = true + np.random.randn(_NUM_DEFAULT_SAMPLES) * 1e-2
  return true, noisy

def generate_true_and_noisy_2D():
  true = np.random.randn(_NUM_DEFAULT_VARIABLES, _NUM_DEFAULT_SAMPLES)
  noisy = true + np.random.randn(_NUM_DEFAULT_VARIABLES, _NUM_DEFAULT_SAMPLES) * 1e-2
  return true, noisy

@params(1, 10, 100)
def test_evaluate_results_perfect(num_features):
  np.random.seed(_SEED)
  true_val = np.random.randn(_NUM_DEFAULT_SAMPLES, num_features) # time by features
  eval_res = evaluate.evaluate_results(true_val, true_val, _DEFAULT_METRICS)
  # Need to use assert_almost_equal because of epsilon precision mismatches.
  np.testing.assert_almost_equal(eval_res['CC'], np.ones(num_features),
                              err_msg='Prediction CC should be 1.')

@params(1, 10, 100)
def test_evaluate_results_with_noise(num_features):
  np.random.seed(_SEED)
  true_val = np.random.randn(_NUM_DEFAULT_SAMPLES, num_features) # time by features
  all_noise_std = [0.01, 0.1, 1.0]
  all_metrics = []
  for std in all_noise_std:
    pred_val = true_val + np.random.randn(*true_val.shape) * std
    all_metrics.append(evaluate.evaluate_results(true_val, pred_val,
                                                  _DEFAULT_METRICS))
    nose_tools.ok_(np.all(all_metrics[-1]['CC'] < 1.0))
  nose_tools.ok_(np.all(all_metrics[0]['CC'] > all_metrics[1]['CC']),
                 'std 0.01: {0}, std 0.1: {1}'.format(all_metrics[0]['CC'],
                                                      all_metrics[1]['CC']))
  nose_tools.ok_(np.all(all_metrics[1]['CC'] > all_metrics[2]['CC']),
                 'std 0.1: {0}, std 1.: {1}'.format(all_metrics[1]['CC'],
                                                      all_metrics[2]['CC']))

def test_normalized_frobenius_error():
  true, noisy = generate_true_and_noisy()
  nose_tools.ok_(evaluate.normalized_frobenius_error(true, true) == 0)
  nose_tools.ok_(np.all(evaluate.normalized_frobenius_error(true, noisy) > 0))

def test_compute_eig_id_error_exact_match():
  np.random.seed(_SEED)
  A = np.random.randn(_NUM_DEFAULT_VARIABLES, _NUM_DEFAULT_VARIABLES)
  # Similarity transform.
  T = np.linalg.qr(np.random.randn(_NUM_DEFAULT_VARIABLES, _NUM_DEFAULT_VARIABLES))[0]
  Asim = T.T @ A @ T
  err = evaluate.compute_eig_id_error(np.linalg.eigvals(A), np.linalg.eigvals(Asim))
  nose_tools.ok_(err <= 1e-6)

@params(1, 5, 9, 10, 50, 90, 100, 500, 900, 1000)
def test_compute_eig_id_decreasing_error(seed):
  np.random.seed(seed)

  dim = 6
  mags = np.random.rand(int(dim/2))
  angles = np.random.rand(int(dim/2)) * np.pi

  # Complex conjugate pairs
  mags = np.concatenate((np.atleast_2d(mags).T,
                         np.atleast_2d(mags).T), axis=1).flatten()
  angles = np.concatenate((np.atleast_2d(angles).T,
                           np.atleast_2d(-1*angles).T), axis=1).flatten()
  true_eigs = np.array([math_utils.polar_to_rectangular(m, a) for m, a in zip(mags, angles)])

  sigma = 0.001
  all_errors = []
  for num_to_perturb in range(1, int(dim//2)+1):
    perturbation = sigma * np.random.randn(num_to_perturb) + \
                   sigma * np.random.randn(num_to_perturb)*1j

    perturb_inds = np.random.choice(int(dim//2), replace=False, size=num_to_perturb).astype(int)
    perturb_inds = np.concatenate((perturb_inds, perturb_inds+3)) # conjugate pairs

    # Preturb complex conjugate pairs equally.
    noisy_eigs = true_eigs[perturb_inds] + np.concatenate((perturbation, perturbation))
    all_errors.append(evaluate.compute_eig_id_error(true_eigs, noisy_eigs))
    nose_tools.ok_(all_errors[-1] > 0)

  # As we identify more and more poles the error should decrease, even if identified
  # poles are noisy.
  nose_tools.ok_(np.all(np.diff(all_errors) < 0))

@params((5, 1e-3), (9, 1e-3), (500, 1e-3), (900, 1e-3),
        (5, 1e-2), (9, 1e-2), (500, 1e-2), (900, 1e-2),
        (5, 1e-1), (9, 1e-1), (500, 1e-1), (900, 1e-1))
def test_compute_eig_id_error_perturbed(seed, sigma):
  np.random.seed(seed)
  
  dim = 6
  mags = np.random.rand(int(dim/2))
  angles = np.random.rand(int(dim/2)) * np.pi

  # Complex conjugate pairs
  mags = np.concatenate((np.atleast_2d(mags).T,
                         np.atleast_2d(mags).T), axis=1).flatten()
  angles = np.concatenate((np.atleast_2d(angles).T,
                           np.atleast_2d(-1*angles).T), axis=1).flatten()
  true_eigs = np.array([math_utils.polar_to_rectangular(m, a) for m, a in zip(mags, angles)])

  perturbation = sigma * np.random.randn(int(dim/2)) + sigma * np.random.randn(int(dim/2))*1j

  # Preturb complex conjugate pairs equally.
  noisy_eigs = true_eigs + np.concatenate((perturbation, perturbation))

  expected_err = evaluate.normalized_frobenius_error(true_eigs, noisy_eigs)
  err = evaluate.compute_eig_id_error(true_eigs, noisy_eigs)
  
  nose_tools.ok_(err > 0)
  nose_tools.ok_(np.round(np.abs(err - expected_err), decimals=3) <= 1e-3,
                 f'{np.abs(err - expected_err)} > 1e-3')
