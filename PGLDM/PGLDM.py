"""PGLDM for spectral learning of shared dynamics between generalized-linear
processes. This implementation models the latent-state dynamics of a primary
generalized-linear time-series while prioritizing identification of dynamics
shared with a secondary Gaussian process."""
import bestlds.moments as bestLDS_moments
import bestlds.ssid as bestLDS_ssid
from enum import Enum
import math_utils.hankel_utils as hankel_utils
import math_utils.matrix_utils as matrix_utils
import numpy as np
import PGLDM.transformation_of_moments as xform

class ObservationType(Enum):
  """Observation types currently explicitly supported by PGLDM."""
  SingleGaussian = 0 # single data-source (stage 2 only = standard SSID)
  SinglePoisson = 1 # single data-source (stage 2 only = standard SSID)
  SingleBernoulli = 2 # single data-source (stage 2 only = standard SSID)
  GausGaus = 3
  PoisGaus = 4
  PoisPois = 5
  BernGaus = 6

  def supports_input_cov_mats(obs_type):
    return (obs_type != ObservationType.SingleBernoulli and 
            obs_type != ObservationType.PoisPois and 
            obs_type != ObservationType.BernGaus)

  def single_timeseries(obs_type):
    return (obs_type == ObservationType.SingleBernoulli or 
            obs_type == ObservationType.SinglePoisson or 
            obs_type == ObservationType.SingleBernoulli)

  def primary_poisson(obs_type):
    return (obs_type == ObservationType.PoisGaus or
            obs_type == ObservationType.PoisPois)

  def secondary_gaussian(obs_type):
    return (obs_type == ObservationType.GausGaus or
            obs_type == ObservationType.PoisGaus or
            obs_type == ObservationType.BernGaus)

def PGLDM(horizon, nx, Y, observation_type, T=None, n1=0, T_horizon=None,
          n3=0, input_cov_mats={}, debug_mode=False, **transform_kwargs) -> dict:
  """PGLDM implementation.

  Without moment conversion, this implementation is equivalent to covariance-based
  SSID with the *additional* capability of prioritized learning of shared dynamics
  between two time-series. (Covariance-based SSID algorithm, reference Katayama
  implementation chapter 7.7 or VODM Ch 3 Algorithm 2.)

  In the implementation we use Y and T as the variables to match the model
  definition in equation (6). However, in the Gaussian-Gaussian case, the variables
  being operated on would be R and Z accordingly.

  Args:
    horizon: int. Horizon.
    nx: int. Number of latent states (shared + disjoint).
    Y: np.ndarray of shape (features, samples). Primary time-series data.
    observation_type: ObservationType enum.
    T: np.ndarray of shape (features, samples). Secondary time-series data.
    n1: int. Number of shared latent states.
    T_horizon: int. Horizon for the secondary time series.
    n3: int. Stage 3 (private secondary) latent dimensionality.
    input_cov_mats: dict. Optional dependency injection used to directly provide
      analytical first and second moments to decouple error for moment conversion
      and system identification. Unit testing purposes only.
    transform_kwargs: keyword arguments for transformation_of_X_moments()

  Returns:
    A dictionary of parameters learned.
  """
  if n1 <= 0 or T is None: # Stage 2 only --> SSID.
    return SSID(horizon, nx, Y, observation_type, input_cov_mats=input_cov_mats,
                debug_mode=debug_mode, **transform_kwargs)

  parameters = {}
  ny, num_measurements = Y.shape
  nz, num_z_measurements = T.shape
  if not T_horizon: T_horizon = horizon
  # Max possible number of measurements.
  j = hankel_utils.compute_hankel_parameters(num_measurements, horizon,
    num_second_observations=num_z_measurements, second_horizon=T_horizon)
      
  if debug_mode:
    parameters['n1'] = n1
    parameters['nx'] = nx
    parameters['ny'] = ny
    parameters['nz'] = nz
    parameters['horizon'] = horizon
    parameters['T_horizon'] = T_horizon
    parameters['j'] = j
    parameters['Ytrain'] = Y
    parameters['Ttrain'] = T

  if observation_type != ObservationType.GausGaus:
    xform_inputs = {}
    meanT, covT, num_min_momentT = None, None, None # Initialize.
    if ObservationType.supports_input_cov_mats(observation_type) and input_cov_mats:
      xform_inputs = { # Unit testing, dependency injection.
       'covS': input_cov_mats['covS'], 'corrS': input_cov_mats['corrS'],
       'meanS': np.tile(input_cov_mats['meanS'], (2*horizon, 1)).squeeze(),
       'meanT': np.tile(input_cov_mats['meanT'], (2*T_horizon, 1)).squeeze(),
       'xcorrTS': input_cov_mats['xcorrTS'], 'xcovTS': input_cov_mats['xcovTS']}

    # Compute future T-past R Hankel matrix equation (8).
    if ObservationType.primary_poisson(observation_type):
      meanR, covR, num_min_moment = xform.transformation_of_poisson_moments(
        Y, horizon, input_stats=xform_inputs, debug_mode=debug_mode, **transform_kwargs)

      if observation_type == ObservationType.PoisGaus:
        Ci_zy, _ = xform.transformation_of_poisgaus_crosscovariate_moments(
          Y, T, horizon, Z_horizon=T_horizon, meanR=meanR, covR=covR,
          input_stats=xform_inputs, debug_mode=debug_mode)

      elif observation_type == ObservationType.PoisPois:
        meanT, covT, num_min_momentT = xform.transformation_of_poisson_moments(
          T, T_horizon, input_stats={}, debug_mode=debug_mode, **transform_kwargs)

        Ci_zy, _ = xform.transformation_of_poispois_crosscovariate_moments(
          Y, T, horizon, T_horizon=T_horizon, debug_mode=debug_mode)

      if debug_mode:
        parameters['num_min_moment'] = num_min_moment
        if num_min_momentT: parameters['num_min_momentT'] = num_min_momentT

    elif observation_type == ObservationType.BernGaus: 
      out = xform.transformation_of_berngaus_crosscovariate_moments(
        Y, T, horizon, Z_horizon=T_horizon)
      meanR, meanT, covR, covT, Ci_zy = out

    parameters['b'] = meanR[:ny, :]
    if meanT is not None: parameters['d'] = meanT[:nz, :] 
    if debug_mode:
      parameters['meanR'] = meanR
      parameters['covR_stacked_hankel'] = covR
      if meanT is not None: parameters['meanT'] = meanT
      if covT is not None: parameters['covT_stacked_hankel'] = covT

  elif observation_type == ObservationType.GausGaus:
    if input_cov_mats: # Unit testing, dependency injection.
      Ci_zy = input_cov_mats['xcovTR']
      meanR = input_cov_mats['meanR']
      covR = input_cov_mats['covR']

    else:
      Yp = hankel_utils.make_hankel(Y, horizon, j)
      Tf = hankel_utils.make_hankel(T, T_horizon, j, start=T_horizon)
      Ci_zy = (Tf @ Yp.T) / (j - 1) # Future T-past R Hankel matrix equation (8).

  # Parameter identification.
  full_U, full_S, full_V = np.linalg.svd(Ci_zy, full_matrices=False)
  S = np.diag(full_S[:n1])
  U = full_U[:,:n1]
  V = full_V[:n1,:]
  if debug_mode:
    parameters['Ci_zy'] = Ci_zy
    parameters['Ci_zy_S'] = full_S
    parameters['Ci_zy_V'] = full_V
    parameters['Ci_zy_U'] = full_U

  Oz = U @ S**(1/2.) # T observability matrix.
  ctrlr_mat1 = S**(1/2.) @ V # Controllability matrix for shared latents.
  A11, Cz1 = extract_AC(Oz, nz)
  G1 = extract_G(ctrlr_mat1, ny)

  if observation_type != ObservationType.GausGaus or input_cov_mats:
    Ci = hankel_utils.extract_correlation(covR, ny, horizon, pair='fp')
  else:
    # Compute Hankel as per equation (2).
    Yf = hankel_utils.make_hankel(Y, horizon, j, start=horizon)
    Ci = (Yf @ Yp.T) / (j - 1)

  # Extract Y observability matrix for shared dynamics.
  Oy1 = Ci @ matrix_utils.inverse(ctrlr_mat1, left=False)
  _, Cy1 = extract_AC(Oy1, ny)

  if debug_mode:
    parameters['Oy1'] = Oy1
    parameters['Oz'] = Oz
    parameters['ctrlr_mat1'] = ctrlr_mat1
    parameters['Ci'] = Ci
    parameters['Ci_S'] = np.linalg.svd(Ci, full_matrices=False)[1]
    parameters['Ci1_S'] = np.linalg.svd(Oy1 @ ctrlr_mat1, full_matrices=False)[1]

  # Estimate A11 by using least-squares and the extended controllability matrices,
  # section 3.2.1 of the manuscript.
  A11 = ctrlr_mat1[:, :-ny] @ matrix_utils.inverse(ctrlr_mat1[:, ny:], left=False)

  n2 = nx - n1
  if n2 > 0: # Optionally model the disjoint dynamics in Y.
    Cz = np.concatenate((Cz1, np.zeros([nz, n2])), axis=1)

    # Subtract out the part of Y that is shared with T.
    Ci2 = Ci - Oy1 @ ctrlr_mat1 # Equation (10) in manuscript.
    full_U, full_S, full_V = np.linalg.svd(Ci2, full_matrices=False)
    S = np.diag(full_S[:n2])
    U = full_U[:,:n2]
    V = full_V[:n2,:]
    if debug_mode:
      parameters['Ci2'] = Ci2
      parameters['Ci2_S'] = full_S
      parameters['Ci2_V'] = full_V
      parameters['Ci2_U'] = full_U

    Oy2 = U @ S**(1/2.)
    # Controllability matrix associated with unshared latent states.
    ctrlr_mat2 = S**(1/2.) @ V # Equivalent to matrix_utils.inverse(Oy2) @ Ci2.
    
    _, Cy2 = extract_AC(Oy2, ny)
    Cy = np.concatenate((Cy1, Cy2), axis=1)
    Oy = np.concatenate((Oy1, Oy2), axis=1)

    G2 = extract_G(ctrlr_mat2, ny)
    G = np.concatenate((G1, G2))
    ctrlr_mat = np.concatenate((ctrlr_mat1, ctrlr_mat2))
    
    if debug_mode:
      parameters['Oy2'] = Oy2
      parameters['Oy'] = Oy
      parameters['ctrlr_mat2'] = ctrlr_mat2
      parameters['ctrlr_mat'] = ctrlr_mat

    # Extract from concatenated controllability matrix, section 3.2.2 of the manuscript.
    A21_22 = ctrlr_mat2[:, :-ny] @ matrix_utils.inverse(ctrlr_mat[:, ny:], left=False)
    A = np.concatenate((np.concatenate((A11, np.zeros((n1, n2))), axis=1), A21_22))

  else: # n2 == 0
    A, Cz, Cy, G = A11, Cz1, Cy1, G1

  parameters['Cz'] = Cz
  parameters['A'] = A
  parameters['Cy'] = parameters['C'] = Cy # 'Cy' is for extra bookkeeping.
  parameters['G'] = G

  # Add covariances to the parameters.
  if observation_type != ObservationType.GausGaus:
    ff_mat = hankel_utils.extract_correlation(covR, ny, horizon, pair='ff')
    pp_mat = hankel_utils.extract_correlation(covR, ny, horizon, pair='pp')
    L0_ff = hankel_utils.compute_average_variance(ff_mat, ny, horizon)
    L0_pp = hankel_utils.compute_average_variance(pp_mat, ny, horizon)
    covariances = { 'L0': matrix_utils.make_symmetric((L0_ff + L0_pp) / 2) }
  else:
    covariances = compute_covariances(Y, horizon, j, Yf=Yf, Yp=Yp, debug_mode=debug_mode)

  for k, v in covariances.items(): parameters[k] = v
  if debug_mode: # Optionally add secondary signal covariances to the parameters.
    for k, v in compute_covariances(T, T_horizon, j, debug_mode=debug_mode).items():
      parameters['T_'+k] = v

  if n3 == 0: return parameters

  # Stage 3 (optional). Appendix A.1.4.
  if not ObservationType.secondary_gaussian(observation_type):
    if observation_type == ObservationType.PoisPois:
      TfTp = hankel_utils.extract_correlation(covT, nz, T_horizon, pair='fp')
    else:
      raise ValueError(f'Unsupported stage 3 for observation: {observation_type}')

  else: # Secondary observation is Gaussian; compute Hankel directly.
    Tf = hankel_utils.make_hankel(T, T_horizon, j, start=T_horizon)
    Tp = hankel_utils.make_hankel(T, T_horizon, j)
    TfTp = Tf @ Tp.T / (j - 1)

  Tctrlr_1 = matrix_utils.inverse(Oz) @ TfTp
  TfTp_res = TfTp - Oz @ Tctrlr_1

  full_U_res, full_S_res, full_V_res = np.linalg.svd(TfTp_res)
  S_res = np.diag(full_S_res[:n3])
  U_res = full_U_res[:, :n3]
  V_res = full_V_res[:n3, :]

  # Parameter identification
  O_res = U_res @ S_res # observability matrix
  Ctrlr_res = V_res # controllability matrix
  A33, C3 = extract_AC(O_res, nz)
  G3_z = extract_G(Ctrlr_res, nz)

  if debug_mode:
    parameters['Tctrlr_mat1'] = Tctrlr_1
    parameters['TOy_3'] = O_res
    parameters['Tctrl_mat3'] = Ctrlr_res

    parameters['TfTp'] = TfTp
    parameters['TfTp_S'] = full_S_res
    parameters['TfTp_U'] = full_U_res
    parameters['TfTp_V'] = full_V_res

  secondary_private_params = {'A': A33, 'C': C3, 'G': G3_z}
  parameters['secondary_private_params'] = secondary_private_params
  return parameters

def SSID(horizon, nx, Y, observation_type, input_cov_mats={}, debug_mode=False,
         **transform_kwargs) -> dict:
  """Single time-series (generalized-)linear dynamical modeling. Encapsulates
    1) Covariance-based SSID (Katayama implementation chapter 7.7 or VODM Ch 3
      Algorithm 2).
    2) PLDSID implementation, Buesing et al 2012.
    3) bestLDS implementation, Stone, Sagiv et al. 2023.

  Args:
    horizon: int. Horizon.
    nx: int. Number of latent states.
    Y: np.ndarray of shape (features, samples). Time-series data.
    observation_type: ObservationType enum. Single time-series only.
    input_cov_mats: dict. Optional dependency injection used to directly provide
      analytical first and second moments to decouple error for moment conversion
      and system identification. Unit testing purposes only. Not all observations
      support input_cov_mats (see ObservationType enum definition).
    transform_kwargs: keyword arguments for transformation_of_poisson_moments()

  Returns:
    A dictionary of parameters learned.
  """
  if not ObservationType.single_timeseries(observation_type):
    raise ValueError('observation_type must be single-timeseries')

  if observation_type == ObservationType.SingleBernoulli:
    return run_bestlds(horizon, nx, Y)

  ny, num_measurements = Y.shape
  j = hankel_utils.compute_hankel_parameters(num_measurements, horizon)
  
  parameters = {}
  if observation_type == ObservationType.SinglePoisson:
    xform_inputs = {}
    if input_cov_mats: # Unit testing, dependency injection.
      xform_inputs = {
        'covS': input_cov_mats['covS'], 'corrS': input_cov_mats['corrS'],
        'meanS': np.tile(input_cov_mats['meanS'], (2*horizon, 1)).squeeze()}      

    # Moment conversion.
    meanR, covR, num_min_moment = xform.transformation_of_poisson_moments(
                    Y, horizon, input_stats=xform_inputs, debug_mode=debug_mode,
                    **transform_kwargs)

    # Hankel matrix equation (2) manuscript.
    Ci = hankel_utils.extract_correlation(covR, ny, horizon, pair='fp')
    parameters['b'] = meanR[:ny, :]
    if debug_mode: parameters['num_min_moment'] = num_min_moment
  
  elif observation_type == ObservationType.SingleGaussian:
    if input_cov_mats: # Unit testing, dependency injection.
      meanR, covR = input_cov_mats['meanR'], input_cov_mats['covR']
      Ci = hankel_utils.extract_correlation(covR, ny, horizon, pair='fp')

    else:
      Yp = hankel_utils.make_hankel(Y, horizon, j)
      Yf = hankel_utils.make_hankel(Y, horizon, j, start=horizon)
      Ci = (Yf @ Yp.T) / (j - 1) # Hankel equation (2).

  # Parameter identification.
  full_U, full_S, full_V = np.linalg.svd(Ci)
  S = np.diag(full_S[:nx])
  U = full_U[:, :nx]
  V = full_V[:nx, :]
  O = U @ S**(1/2.) # Observability matrix
  Ctrlr = S**(1/2.) @ V # Controllability matrix
  A, C = extract_AC(O, ny)
  G = extract_G(Ctrlr, ny)

  parameters['C'] = C
  parameters['Cy'] = parameters['C'] # 'Cy' is for extra bookkeeping.
  parameters['A'] = A
  parameters['G'] = G
  if debug_mode:
    parameters['full_S'] = full_S
    parameters['full_U'] = full_U
    parameters['full_V'] = full_V
    parameters['Ci'] = Ci
    parameters['O'] = O
    parameters['Ctrlr'] = Ctrlr

  # Add covariances to the parameters.
  if observation_type == ObservationType.SinglePoisson:
    ff_mat = hankel_utils.extract_correlation(covR, ny, horizon, pair='ff')
    pp_mat = hankel_utils.extract_correlation(covR, ny, horizon, pair='pp')
    L0_ff = hankel_utils.compute_average_variance(ff_mat, ny, horizon)
    L0_pp = hankel_utils.compute_average_variance(pp_mat, ny, horizon)
    covariances = { 'L0': matrix_utils.make_symmetric((L0_ff + L0_pp) / 2) }

  elif observation_type == ObservationType.SingleGaussian:
    covariances = compute_covariances(Y, horizon, j, Yf=Yf, Yp=Yp,
                                      debug_mode=debug_mode)

  for k, v in covariances.items(): parameters[k] = v
  return parameters

def run_bestlds(horizon, nx, Y):
  """
  Wrapper around bestLDS for the single Bernoulli time-series case. Should not
  be accessed directly, but instead be used through the SSID() method above.
  """
  ny, num_measurements = Y.shape

  # Output is np.array([[Y_f], [Y_p]]).
  # flip will flip Y_p so that the first element is horizon-1 and last 0.
  YfYp = bestLDS_moments.future_past_Hankel_order_stream(Y.T, horizon, ny, flip=True)
  YfYp = YfYp.T # N x (horizon * ny)
  
  # Moment conversion.
  mu_Rs, sigma_R_full = bestLDS_moments.fit_mu_sigma_bernoulli_undriven(
            *bestLDS_moments.get_y_moments(YfYp, only_lower_triag=False))

  # Compute covariance term.
  ff_mat = hankel_utils.extract_correlation(sigma_R_full, ny, horizon, pair='ff')
  ## pp is reversed from VODM notation, but should still be ok here.
  pp_mat = hankel_utils.extract_correlation(sigma_R_full, ny, horizon, pair='pp')
  L0_ff = hankel_utils.compute_average_variance(ff_mat, ny, horizon)
  L0_pp = hankel_utils.compute_average_variance(pp_mat, ny, horizon)
  L0 = matrix_utils.make_symmetric((L0_ff + L0_pp) / 2)
  
  # Rearrange to get estimate of covariance W.
  sigma_what = bestLDS_moments.get_sigmaw_undriven(sigma_R_full)
  R = bestLDS_ssid.get_R(sigma_what) # Cholesky decompose R.
  A, C, Q, R, S, SigVals = bestLDS_ssid.undriven_n4sid(R, horizon, nx, ny)
  return {'A': A, 'C': C, 'L0': L0, 'Q': Q, 'R': R, 'S': S, 'full_S': SigVals}

######## Utility functions. ########
def compute_covariances(Y, i, j, Yf=None, Yp=None, debug_mode=False):
  """Compute all combinations of future-past cross-covariances.

  Args:
    Y: np.ndarray of size (features, samples). Data. Note: Y should be demeaned
      before being passed in as a parameter.
    i: int. Horizon.
    j: int. Number of samples per horizon (i.e., columns of Hankel matrix).
    Yf: np.ndarray of size (features*horizon, j). Optionally provide Yf directly.
    Yp: np.ndarray of size (features*horizon, j). Optionally provide Yp directly.
  """
  covariances = {'L0': np.cov(Y, ddof=1)}
  if not debug_mode: return covariances

  if Yp is None: Yp = hankel_utils.make_hankel(Y, i, j)
  if Yf is None: Yf = hankel_utils.make_hankel(Y, i, j, i)

  num_samples = j
  Sigma_YpYp = np.cov(Yp, ddof=1) # Lambda_i, Lambda_0 along diagonal.
  Sigma_YfYf = np.cov(Yf, ddof=1) # Lambda_i, should be roughly equivalent to above.
  Li = (Sigma_YpYp + Sigma_YfYf) / 2
  Sigma_YfYp = (Yf @ Yp.T) / (num_samples - 1) # C_i
  covariances['Li'], covariances['Ci'] = Li, Sigma_YfYp
  return covariances

# Extended observability matrix (least-squares) approach to extract A and C (see
# section 2.1 of manuscript).
def extract_AC(observability_matrix, ny):
  C = observability_matrix[:ny, :]
  O_minus = observability_matrix[:-ny, :] # omit last ny elements. O_floor
  O_plus = observability_matrix[ny:, :] # omit first ny elements. O_bar
  A = matrix_utils.inverse(O_minus) @ O_plus
  return A, C

# Refer to section 2.1 of the manuscript.
def extract_G(ctrlr_mat, ny):
  return ctrlr_mat[:, -ny:]
