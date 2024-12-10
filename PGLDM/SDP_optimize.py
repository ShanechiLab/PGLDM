"""Semi-definite programming problem to fit noise statistics for PLDS models."""
import cvxpy
import math_utils.matrix_utils as matrix_utils
import models.SID_utils as SID_utils
import numpy as np
import scipy.linalg

_HANDLE_RS_VALS = ['L0G_omit_RS', 'L0G_use_RS']
_SYM_DEC = 9

def optimize(params, handle_RS='L0G_omit_RS', saveRS=False,
             # MATLAB implementation usees sdpt3 as the solver, however cvxpy
             # does not support this right now. cvxpy uses three libraries which
             # support SDP: CVXOPT, MOSEK, SCS
             # https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
             # NOTE: this implementation does not work with the python installation
             # of CVXOPT though.
             solver=cvxpy.SCS,
             enforce_R_pd=True, enforce_symmetric=True, model_type='DT',
             debug_mode=False) -> dict :
  """
  Implementation of the semi-definite convex optimization solution for fitting
  noise statistics (equations (12) and (13)).

  Args:
    params: dict. Parameters identified by PG-LDS-ID or PLDSID.
    handle_RS: string. One of ['L0G_omit_RS', 'L0G_use_RS'].
      'L0G_omit_RS': Omit R and S when recomputing L0 and G (based on equation (13)).
      'L0G_use_RS': Include R and S when recomputing L0 and G (based on equation (13)).
    saveRS: bool. Irrespective of handle_RS setting, include the resulting R and
      S in the parameters.
    solver: cvxpy.SCS (default) | cvxpy.MOSEK | cvxpy.CVXOPT. See inline comment
      above for more information.
    enforce_R_pd: bool. Default True. Enforce learned R is positive definite.
      Only relevant for model_type='DT' (see below).
    enforce_symmetric: bool. Default True. Add additional constraints for all
      covariance matrices to be symmetric.
    model_type: str one of ['DT', 'CT']. Whether the primary signal time-series
      that is being modeled is discrete or continuous-valued. Default is 'DT'.

  Returns:
    A dictionary of parameters.
  """
  if model_type not in ['CT', 'DT']:
    raise ValueError('model_type must be either DT or CT.')
  if handle_RS not in _HANDLE_RS_VALS:
    raise ValueError(f'handle_RS parameter must be one of: {_HANDLE_RS_VALS}')

  # Define and solve the CVXPY problem.
  A = params['A'].copy()
  C = params['C'].copy()
  L0 = params['L0'].copy()
  G = params['G'].copy()
  W_S, W_R, W_P = 1, 1, 1
  nx, ny = A.shape[0], C.shape[0]

  num_attempts, total_attempts = 0, 1
  while num_attempts < total_attempts:
    # PSD true ensures variable is symmetric and positive semi-definite.
    covX = cvxpy.Variable((nx, nx), PSD=True)

    Q = covX - A @ covX @ A.T
    R = L0 - C @ covX @ C.T
    S = G - A @ covX @ C.T

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
    constraints += [ covX >> 0 ]

    if model_type == 'DT':
      constraints += [ Q >> 0 ] # preconditioned Q PSD
      if enforce_symmetric: constraints += [ Q == Q.T ] # enforce symmetric

      if enforce_R_pd:
        constraints += [ R >> 0 ] # preconditioned R PSD
        if enforce_symmetric: constraints += [ R == R.T ] # enforce symmetric

    else: # model_type == 'CT'
      noise_blk_mat = cvxpy.bmat([[Q, S], [S.T, R]])
      constraints += [ noise_blk_mat >> 0 ] # preconditioned QRS block is PSD
      if enforce_symmetric:
        constraints += [ noise_blk_mat == noise_blk_mat.T ] # enforce symmetric

    cost = 0
    if model_type == 'CT':
      cost += W_P * cvxpy.sum_squares(covX)
    else: # model_type == 'DT'
      # PLDS model has noise statistics R and S that should be 0.
      cost += W_R * cvxpy.sum_squares(R) + W_S * cvxpy.sum_squares(S)

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=solver, verbose=debug_mode)
    num_attempts += 1

    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
      print('Invalid optimization status: ', prob.status)
      raise ValueError('Unable to successfully solve optimization for params.')

    print(f'The optimal value is {prob.value} with status: {prob.status}')

    ## NOTE: All symmetric matrices are being truncated to 9 decimal points to
    ## prevent any precision inaccuracies making the matrices "not symmetric".
    Qval = np.around(Q.value, decimals=_SYM_DEC)
    # Redundant, but to be doubly careful resolving ALE.
    params['SigX'] = np.around(scipy.linalg.solve_discrete_lyapunov(A, Qval),
                               decimals=_SYM_DEC)
    params['Q'] = Qval # Always save Q but don't save RS unless specified.
    # If saving R and S, rounding R for numerical stability and reproducibility.
    if saveRS: Rval, Sval = np.around(R.value, decimals=_SYM_DEC), S.value
    
    ## We recompute L0 and G from the parameters learned by the optimization
    ## algorithm rather than using the L0 and G that were initially learned during
    ## system identification. Although this will introduce bias (as we're modifying
    ## the empirically estimated observation covariance L0), modifying will allow
    ## the parameter set learned be consistent.
    L0 = params['C'] @ params['SigX'] @ params['C'].T
    G = params['A'] @ params['SigX'] @ params['C'].T

    ## Here we include the optimization's output of R and S in the computation of
    ## L0 and G (which is what it should be in 'CT' case).
    if handle_RS == 'L0G_use_RS':
      L0 += Rval
      G += Sval
    elif handle_RS == 'L0G_omit_RS':
      ## Here we explicitly omit R and S when computing L0 and G, rather than using
      ## the output of the optimization (which could be small but not exactly 0).
      if model_type == 'CT': print('[WARNING] Ignoring R and S for CT signal type.')

    L0 = np.around(L0, decimals=_SYM_DEC)
    params['L0'], params['G'] = L0, G
    if saveRS: params['R'], params['S'] = Rval, Sval
    return params
