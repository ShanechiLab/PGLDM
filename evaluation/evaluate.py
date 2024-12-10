"""Utility methods for evaluating estimation performance."""
import itertools
import models.SSM_utils as SSM_utils
import numpy as np
import sklearn.metrics
import warnings

def evaluate_results(true_vals, pred_vals, metrics_to_compute=['CC', 'AUC'],
                     predicted_covariances=None):
  """Convenience wrapper around eval_prediction to simulataneously evaluate
  multiple metrics.

  Args:
    true_vals: (num_samples, num_features)
    pred_vals: (num_samples, num_features)
    metrics_to_compute: list. List of metrics to compute.

  Optional:
    predicted_covariances: np.ndarray of shape (num_samples, num_features, num_features).
      Only needed for AUC. Log-rate covariances.

  Returns:
    Dictionary where keys are metrics_to_compute and values are the associated
    metric computed for each feature across time.
  """
  if true_vals.shape != pred_vals.shape:
    raise ValueError('True and predicted values need to be the same dimension.')
  
  eval_res = dict()
  for metric in metrics_to_compute:
    eval_res[metric] = eval_prediction(
      true_vals, pred_vals, metric, predicted_covariances=predicted_covariances)
    if np.size(eval_res[metric]) == 1: eval_res[metric] = eval_res[metric][0]
  return eval_res

def compute_eig_id_error(true_eigs, id_eig_vals, return_best_match=False):
  """Computes the eigenvalue identification error.

  Error is with respect to overall system behavior identification. For example,
  if there are 4 true eigenvalues and only 2 eigenvalues were identified, then the
  overall identified system is effectively 2 non-zero and 2 zero eigenvalues. Error
  is computed by determining best 1-to-1 match of true and identified modes based on
  Euclidean distance.  
  
  Args:
    true_eigs: np.ndarray. True eigenvalues.
    id_eig_vals: np.ndarray. Identified eigenvalues to evaluate against true_eigs.
    return_best_match: bool. Will also return the best matching set of eigenvalues.
      Default False.

  Returns:
    If return_best_match is False: eigenvalue error as a float. Else, a two element
    tuple (error, best eigenvalue match).
  """
  if np.size(id_eig_vals) == 0 or np.size(true_eigs) == 0:
    raise ValueError('Both ID and true eigenvalues must be length greater than 0.')

  if np.size(id_eig_vals) < np.size(true_eigs):
    zero_pad = np.zeros(np.size(true_eigs) - np.size(id_eig_vals))
    id_eig_vals = np.concatenate((id_eig_vals, zero_pad))
  elif np.size(id_eig_vals) > np.size(true_eigs):
    raise NotImplementError('Number of ID eigenvalues > true eigenvalues not supported yet.')

  error_vals = np.array([])
  permut = np.array(list(itertools.permutations(true_eigs)))
  for p in permut:
    error_vals = np.append(error_vals, normalized_frobenius_error(p, id_eig_vals))

  pmi = np.argmin(error_vals)
  if return_best_match:
    error_vals[pmi], permut[pmi]
  return error_vals[pmi]

def eval_prediction(true_value, prediction, measure, predicted_covariances=None):
  """Compute requested metric for given prediction with respect to given ground truth."""
  if prediction.shape[0] == 0: return np.ones(true.shape[1]) * np.nan

  if measure == 'CC':
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      R = np.corrcoef(true_value, prediction, rowvar=False)
    n = true_value.shape[1]
    perf = np.diag(R[n:, :n])
  elif measure == 'AUC':
    perf = AUC(true_value, prediction, predicted_covariances)
  else:
    raise ValueError(f'Unsupported metric {measure}.')
  return perf

def AUC(true, predicted, log_rate_covariances):
  """Computes AUC for provided predicted Poisson rates.

  Args:
    true: np.ndarray of shape (num_samples, num_features). True Poisson observations.
    predicted: np.ndarray of shape (num_samples, num_features). Predicted Poisson rates.
    log_rate_covariances: (num_samples, num_features, num_features). Log-rate 
      cross covariances from which individual feature variances can be extracted.
  """
  bin_observations = np.zeros(true.shape)
  bin_observations[true > 0] = 1 # Binarize the data.
  log_rate_variances = np.zeros((log_rate_covariances.shape[0], true.shape[1]))
  for k in range(log_rate_covariances.shape[-1]):
    log_rate_variances[k, :] = np.diag(log_rate_covariances[k, ...])
  probs = predicted * np.exp(0.5 * log_rate_variances)
  mask = ~np.logical_or(np.isnan(probs), np.isinf(probs))
  output = np.zeros(true.shape[1])
  for dim in range(output.size):
    output[dim] = sklearn.metrics.roc_auc_score(
              bin_observations[mask[:, dim], dim], probs[mask[:, dim], dim])
  return output

########## Error Computation Utils ##########
def matrix_error_norm(true, other):
  """Convenience wrapper for compute matrix Frobenius norm error."""
  return normalized_frobenius_error(true, other, axis=None)

def normalized_frobenius_error(true, other, axis=-1, keepdims=True):
  """Compute normalized Frobenius error.

  Args:
    true: np.ndarray of shape (variable, num_observations) if 2D, otherwise
      np.ndarray of shape (num_observations,) if 1D array.
    other: np.ndarray of shape (variable, num_observations) if 2D, otherwise
      np.ndarray of shape (num_observations,) if 1D array.
    axis: int. The axis along which to compute the error. Default is -1.
    keepdims: bool. Maintain the dimension of inputs after computing norms.

  Returns:
    If 1D array provided, the output is just a single number.
    If 2D array provided, the output is an error for each observation along the
    row axis (i.e., per variable).
  """
  if true.shape != other.shape:
    raise ValueError('True and predicted values must be the same shape.')
  if len(true.shape) > 2:
    raise ValueError('Inputs must be at most 2D.')

  # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html for
  # documentation on norm defaults.
  if axis is not None:
    true_norm = np.linalg.norm(true, axis=axis, keepdims=keepdims)
    true_norm[true_norm == 0] = 1 # Sanity check.
  else:
    norm_type = 'fro' if len(true.shape) == 2 else None
    true_norm = np.linalg.norm(true, ord=norm_type)
    if true_norm == 0: true_norm = 1 # Sanity check.

  if axis is not None:
    return np.linalg.norm(true - other, axis=axis, keepdims=keepdims) / true_norm
  return np.linalg.norm(true - other, ord=norm_type) / true_norm
