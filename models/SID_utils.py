"""General system identification (SID) utility functions."""
import cvxpy
import math_utils.matrix_utils as matrix_utils
import models.SSM_utils as SSM_utils
import numpy as np
import scipy.linalg
from typing import Tuple

_SYM_DEC = 9

def compute_secondary_params(params : dict, W_P : float=1.0,
                             add_covx_constraint : bool=True) -> dict:
  """Compute secondary parameters, typically needed for Kalman gain."""
  hasQRS = SSM_utils.has_QRS(params)
  hasGL0 = SSM_utils.has_GL0(params)
  
  if not hasQRS and not hasGL0:
    raise ValueError('Invalid: need either GL0 or QRS...')

  force_GL0 = False
  if hasQRS:
    try:
      params = compute_secondary_params_from_QRS(params)
    except ValueError as e:
      if not hasGL0: # Backup is use GL0 if QRS does not work.
        raise e
      print('Could not compute params from QRS, attempting GL0: ', e)
      force_GL0 = True

  if not hasQRS or force_GL0:
    try:
      params = compute_secondary_params_from_GL0(params)
      success = True
    except ValueError as e: # Before giving up, try solving ARE using optimization.
      print(f'Could not compute arams from GL0, will try ARE LMI: {e}')
      success = False

    if not success:
      params = optimize_ARE_LMI(params, solver=cvxpy.SCS, # MOSEK, SCS, CVXOPT
                                W_P=W_P, add_covx_constraint=add_covx_constraint)

  # Only needed for Kalman filtering in LSSM.
  if not hasQRS or force_GL0:
    params['INNOV_COV'], params['K'] = calculate_kalman_params_from_GL0(params['A'],
                         params['C'], params['G'], params['L0'], params['P'])
  else:
    params['INNOV_COV'], params['K'] = calculate_kalman_params_from_QRS(params['A'],
                params['C'], params['Q'], params['R'], params['S'], params['P_err'])
  return params

def calculate_kalman_params_from_GL0(A : np.ndarray, C : np.ndarray,
                                     G : np.ndarray, L0 : np.ndarray,
                                     P : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Compute Kalman parameters from parameter set (A, C, G, L0) [VODM notation]."""
  innov_cov = L0 - C @ P @ C.T
  innov_cov = matrix_utils.make_symmetric(innov_cov)
  innov_cov_inv = matrix_utils.inverse(innov_cov)
  # Calculate the Kalman gain used for prediction + filtering (*not* only filtering).
  K = (G - A @ P @ C.T) @ innov_cov_inv
  return innov_cov, K

def calculate_kalman_params_from_QRS(A : np.ndarray, C : np.ndarray, Q : np.ndarray,
                                     R : np.ndarray, S : np.ndarray,
                                     Perr : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Compute Kalman parameters from the parameter set (A, C, Q, R, S) [VODM text]."""
  innov_cov = C @ Perr @ C.T + R
  innov_cov = matrix_utils.make_symmetric(innov_cov)
  innov_cov_inv = matrix_utils.inverse(innov_cov)
  # Calculate the Kalman gain used for prediction + filtering (*not* only filtering).
  K = (A @ Perr @ C.T + S) @ innov_cov_inv
  return innov_cov, K

def solve_ARE_for_P(A : np.ndarray, C : np.ndarray, G : np.ndarray, L0 : np.ndarray) -> np.ndarray:
  """
  Exists as a separate module for unit testing purposes. Equivalent to VODM
  solvric() function.
  """
  forward_sig = scipy.linalg.solve_discrete_are(A.T, C.T, np.zeros(A.T.shape), -L0, s=-G)
  return matrix_utils.make_symmetric(forward_sig)

def compute_secondary_params_from_GL0(params : dict) -> dict:
  """Compute secondary parameters from parameter set (A, C, G, L0) [VODM notation]."""
  if params.get('P', None) is not None:
    return params # P already satisfied, our work here is done.

  try:
    # solve_discrete_are equivalent to VODM solvric function, VODM p63, p66.
    # Note this is the forward state covariance matrix, which is effectively:
    #   P = SigmaX - Perror, where Perror = E[(x - xhat)(x - xhat).T]
    params['P'] = solve_ARE_for_P(params['A'], params['C'], params['G'], params['L0'])
  except np.linalg.LinAlgError as e:
    raise ValueError(f'Could not solve discrete ARE {e}')
  return params

def compute_secondary_params_from_QRS(params : dict) -> dict:
  """Compute secondary parameters from parameter set (A, C, Q, R, S)."""
  params['XCov'] = scipy.linalg.solve_discrete_lyapunov(params['A'], params['Q'])
  try:
    # Predicted state error covariance.
    params['P_err'] = scipy.linalg.solve_discrete_are(params['A'].T,
                                                      params['C'].T,
                                                      params['Q'],
                                                      params['R'],
                                                      s=params['S'])
  except np.linalg.LinAlgError as e:
    raise ValueError(f'Could not solve discrete ARE {e}.')

  # Note this is the forward state covariance matrix, which is effectively
  # P = SigmaX - Perror, where Perror is E[(x - xhat)(x - xhat).T].
  params['P'] = params['XCov'] - params['P_err']
  params['G'] = params['A'] @ params['XCov'] @ params['C'].T + params['S']
  params['L0'] = params['C'] @ params['XCov'] @ params['C'].T + params['R']
  return params

def optimize_ARE_LMI(est_params_xf : dict,
             # MATLAB implementation usees sdpt3 as the solver, however cvxpy
             # does not support this right now. cvxpy uses three libraries which
             # support SDP: CVXOPT, MOSEK, SCS
             # https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
             # NOTE: this implementation does not work with the python installation
             # of CVXOPT though.
             solver=cvxpy.SCS, # MOSEK, SCS, CVXOPT
             W_P : float=1.0, add_covx_constraint : bool=False, debug_mode : bool=False) -> dict:
  """When scipy.linalg.solve_discrete_are() fails, attempt to solve DARE as a LMI
  convex problem using CVXPY.

  Args:
    W_P: float. Weight applied to the Frobenius norm of P, the state-prediction
      covariance matrix (i.e., the convex problem's variable). Default 1.0.
    add_covx_constraint: bool. Add an additional constraint that P is PSD.
      Default False because the variable definition should handle directly.
  """
  A = est_params_xf['A'].copy()
  C = est_params_xf['C'].copy()
  L0 = est_params_xf['L0'].copy()
  G = est_params_xf['G'].copy()
  nx, ny = A.shape[0], C.shape[0]

  if debug_mode:
    try: # Verify if we have a valid set of parameters by solving ARE.
      solve_ARE_for_P(est_params_xf['A'], est_params_xf['C'],
                      est_params_xf['G'], est_params_xf['L0'])
    except np.linalg.LinAlgError as _:
      print('Optimization input params were not valid to begin with.')

  # PSD true ensures variable is symmetric and positive semi-definite.
  P = cvxpy.Variable((nx, nx), PSD=True)

  # CVXPY does not guarantee that PSD constraint (denoted >>) will guarantee
  # a symmetric matrix so add both constraints to make it equivalent to matlab
  # cvx "semidefinite" keyword. 
  #
  # CVXPY reference: https://www.cvxpy.org/tutorial/advanced/index.html#semidefinite-matrices
  # "The following code shows how to constrain matrix expressions to be positive
  # or negative semidefinite (but not necessarily symmetric)."
  #
  # CVX user guide referenence: https://see.stanford.edu/materials/lsocoee364a/cvx_usrguide.pdf
  # "To require that the matrix expression X be symmetric positive semidefinite,
  # we use the syntax X == semidefinite(n)... which is required to be an n × n
  # symmetric positive semidefinite matrix."
  #
  # Note: The operator >> denotes matrix inequality (i.e., usage below == PSD).
  constraints = []
  if add_covx_constraint: constraints += [ P >> 0 ]

  LMI = cvxpy.bmat([
    [A @ P @ A.T - P, G - A @ P @ C.T],
    [G.T - C @ P @ A.T, C @ P @ C.T - L0],
  ])
  constraints += [ LMI << 0 ]

  # We want to reduce the state prediction covariance 
  cost = W_P * cvxpy.sum_squares(P)
  prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
  prob.solve(solver=solver, verbose=debug_mode)
  if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
    print('Invalid optimization status: ', prob.status)
    raise ValueError('Unable to successfully solve optimization for params.')

  print(f'The optimal value is {prob.value} with status {prob.status}')

  ## NOTE: All symmetric matrices are being truncated to 9 decimal points to
  ## prevent any precision inaccuracies making the matrices "not symmetric".
  est_params_xf['P'] = np.around(P.value, decimals=_SYM_DEC)

  if debug_mode:
    try: # Verify if we have a valid set of parameters by solving ARE.
      solve_ARE_for_P(est_params_xf['A'], est_params_xf['C'],
                      est_params_xf['G'], est_params_xf['L0'])
    except np.linalg.LinAlgError as _:
      print('Not a valid set of parameters post optimization.')
  return est_params_xf
