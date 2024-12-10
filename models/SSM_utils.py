"""General utility functions for state-space models."""
from collections import defaultdict
import copy
import evaluation.evaluate as evaluate
import math_utils.matrix_utils as matrix_utils
import models.model_base as model_base
import models.regression_models as reg_model
import numpy as np
import scipy.linalg
from typing import Union, Tuple

def is_stable(A : np.ndarray) -> bool:
  """Matrix A is stable if all poles are within the unit circle (discrete case)."""
  return np.all(np.abs(np.linalg.eigvals(A)) < 1)

def has_GL0(params : dict) -> bool:
  return params.get('G', None) is not None and params.get('L0', None) is not None

def has_QRS(params : dict) -> bool:
  return params.get('Q', None) is not None and params.get('R', None) is not None and \
         params.get('S', None) is not None

def compute_observation_covariance(params):
  """Computes the observation process' covariance from the given paramsters."""
  SigX = scipy.linalg.solve_discrete_lyapunov(params['A'], params['Q'])
  return params['C'] @ SigX @ params['C'].T + params['R']

def compute_SNR(Sig : np.ndarray, noise_cov : np.ndarray, C : np.ndarray = None):
  """Computes SNR based on signal covariance and noise covariance."""
  if C is not None: Sig = C @ Sig @ C.T
  noise = np.diag(noise_cov).copy()
  noise[noise == 0] = 1 # Sanity.
  return np.diag(Sig) / noise

def compute_canonical_transform(A : np.ndarray, return_eigs : bool = False) -> Tuple[np.ndarray, np.ndarray]:
  """Compute the unitary transform to change a state-space model to canonical form.

  Canonical form is defined as the set of state-space parameters that corresponds
  to a state process matrix 'A' that is in block diagonal format where the blocks
  correspond to eigenvalues. Note that all complex eigenvalues will be converted
  to their corresponding real formats.

  Args:
    A: np.ndarray. Square matrix corresponding to the process (state) dynamics.

  Returns:
    T: The unitary transform that converts the SSM to its canonical form via the
      computation T @ A @ inv(T).
    wr: Resultant ordering of the system's real eigenvalues.
  """
  w, v = np.linalg.eig(A)
  wr, vr = scipy.linalg.cdf2rdf(w, v)
  if return_eigs:
    return matrix_utils.inverse(vr), wr
  return matrix_utils.inverse(vr)

def transform_params(params : dict, T : np.ndarray) -> dict:
  """Transforms the provided parameters using the given transform matrix.

  NOTE: This transformation is only valid for linear state-space models. The
  user is responsible for ensuring all relevant model parameters are included in
  the input dictionary as needed.

  Args:
    params: dict. A dictionary of state-space model parameters.
    T: np.ndarray. A unitary matrix of dimension params['A'].shape to transform
      the system parameters with.

  Returns:
    Dictionary corresponding to transformed model parameters.
  """
  transformed_params = copy.deepcopy(params)
  Tinv = matrix_utils.inverse(T, left=False) # Store to reuse.

  # For A-like fields that follow: A_new = T @ Asim @ T^{-1}
  for field in {'A', 'Q', 'P', 'Pp', 'P2', 'SigX'}:
    if params.get(field, None) is None: continue
    transformed_params[field] = T @ params[field] @ Tinv
    
  # For B-like fields that follow: B_new = T @ B
  for field in {'B', 'S', 'G', 'K', 'Kf', 'Kv'}:
    if params.get(field, None) is None: continue
    transformed_params[field] = T @ params[field]

  # For C-like fields that follow: C_new = Csim @ T^{-1}
  for field in {'C', 'Cz'}:  
    if params.get(field, None) is None: continue
    transformed_params[field] = params[field] @ Tinv
  return transformed_params

def eigenvalue_blocks(eigs : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Determine complex conjugate eigenvalue block indicess. Real eigenvalues
  correspond to a block of size 1..

  Args:
    eigs: np.ndarray. 1D array of either complex or real eigenvalues. It is
      recommended to use the real version of the eigenvalues (i.e., the output
      of compute_canonical_transform()).

  Returns:
    (blk_inds, val_counts): A two-element tuple where the first element are the
    blk_inds as a np.ndarray of shape ((num_blocks, 2)), where each row
    corresponds to the block start and block end indices, and the second element
    is the number of eigenvalues each block corresponds to.
  """
  if len(eigs.shape) > 1:
    raise ValueError('Provided eigenvalues must be 1D.')

  if not np.all(np.vectorize(np.isrealobj)(eigs)):
    eigs = np.diag(scipy.linalg.cdf2rdf(eigs, np.identity(np.size(eigs)))[0])

  _, val_inds, val_counts = np.unique(eigs, return_index=True, return_counts=True)
  sorted_order_inds = np.argsort(val_inds)
  blk_inds = []
  for ind in sorted_order_inds:
    blk_inds.append([val_inds[ind], val_inds[ind] + val_counts[ind]])
  return np.array(blk_inds), val_counts[sorted_order_inds]

def order_SSM_modes_by_correlation(sys : model_base, Y : np.ndarray, Z : np.ndarray,
  return_metrics : bool = False) -> Union[Tuple[np.ndarray, np.ndarray],
                                          Tuple[np.ndarray, np.ndarray, dict]]:
  """Orders the SSM modes based on linear decoding performance.

  Args:
    sys: model_base. The system whose mode parameters need to be ordered based
      on decoding performance.
    Y: np.ndarray of shape (num_samples, num_features). The observations used to
      estimate latent states.
    Z: np.ndarray of shape (num_samples, num_features). The ground truth
      observations against which to compare the decoding of sets of poles. Z can
      be the same as Y (this would be the case for single observation state-space
      models), but can be a secondary observation that isn't used to predict the
      latent states.
    return_metrics: bool. Optional also return the metrics associated with pole
      decoding. Default is False.

  Returns:
    Four-element tuple:
      (canonical model, sorted modes, sorted mode block indices, optional metrics)
    Will return None if any operation fails.
  """
  # Convert the system to canonical format.
  params = sys.get_list_of_params()
  T, eigs = compute_canonical_transform(params.get('A', None), return_eigs=True)
  transformed_params = transform_params(params, T)
  try: # Construct a model with the transformed parameter set.
    sys = sys.update_params(transformed_params)
  except ValueError as e:
    raise ValueError('Parameter update failed: ', e)

  # Determine the eigenvalue blocks to use for decoding. Note, that sys.A should
  # look like eigs at this point.
  blk_inds, eig_cnts = eigenvalue_blocks(np.diag(eigs))

  # Use the subblocks to perform decoding to determine correlation ordering.
  try:
    _, X_pred, _ = sys.predict(Y) # (num_samples, num_features)
  except ValueError as e:
    print('Warning, prediction step failed: ', e)
    if return_metrics: return None, None, None, None
    return None, None, None

  reg = reg_model.RegressionModel(reg_model.RegressionMethod.OLS_REG)

  metrics = defaultdict(lambda: np.empty(blk_inds.shape[0]))
  for mode_ind, this_blk_inds in enumerate(blk_inds):
    this_X_pred = X_pred[:, this_blk_inds[0]:this_blk_inds[1]]
    reg.fit(this_X_pred, Z)
    Z_pred = reg.predict(this_X_pred)
    this_blk_metrics = evaluate.evaluate_results(Z, Z_pred, ['NRMSE', 'CC'])
    for k, v in this_blk_metrics.items():
      metrics[k][mode_ind] = np.mean(v) # Average across features.

  increasing_err_inds = np.argsort(metrics['NRMSE'])
  sorted_SSM_modes = []
  for ind in increasing_err_inds:
    sorted_SSM_modes.extend(range(blk_inds[ind][0], blk_inds[ind][1]))

  if return_metrics:
    metrics['NRMSE'] = metrics['NRMSE'][increasing_err_inds]
    metrics['CC'] = metrics['CC'][increasing_err_inds]
    return sys, np.array(sorted_SSM_modes), blk_inds[increasing_err_inds, :], metrics
  return sys, np.array(sorted_SSM_modes), blk_inds[increasing_err_inds, :]
