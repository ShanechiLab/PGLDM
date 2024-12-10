"""Transformation of moments. Includes the following moment conversions:
  - Poisson-Poisson and Poisson-Gaussian (Buesing et al 2012)
  - Bernoulli-Bernoulli and Bernoulli-Gaussian (Stone, Sagiv et al. 2023)
"""
import bestlds.moments as bestLDS_moments
import math_utils.hankel_utils as hankel_utils
import math_utils.matrix_utils as matrix_utils
import numpy as np
import sys

# Offset based on the analytical statistics computed from the model directly under
# the assumption of no firing rate noise, i.e., R = 0. For analytical statistics
# analyses, test_transformation_of_moments.py.
_DIAG_OFFSET = 1e-2
_MIN_EIG = 1e-4 # alternative: sys.float_info.epsilon

def transformation_of_poisson_moments(Y, horizon, correct_fano=True,
      correct_PSD=True, use_cov=False, eig_threshold_correction=True,
      input_stats={}, debug_mode=False):
  """Moment conversion implementation: equation (5), derived by Buesing et al. 2012.

  Args:
    Y: np.ndarray of shape (features, samples). Time-series data.
    horizon: int. Horizon value. 
    correct_fano: bool. Default True. Ensure that the diagonal terms of the
      covariance matrix prior to moment conversion have fano factors slight
      greater than 1. Fano factor = variance / mean, which is 1 for Poisson
      distributions.
    correct_PSD: bool Default True. Ensure the resulting covariance after
      moment conversion is positive semi-definite.
    use_cov: bool. Default False. Legacy implementation from original Buesing
      paper used covariance of observations Y in the moment conversion. When
      use_cov = False, we use correlation of observations Y. Both are equivalent.
    eig_threshold_correction: bool. Default True. When true, using the legacy
      implementation of Buesing et al to correct for positive semi-definiteness
      by ensuring all eigenvalues are equal to max(lambda_i, min_threshold>0).
      When false, we ensure PSD by truncating the eigenvector approximation to
      only eigenvalues > threshold. Note: the second approach changes the
      dimensionality of the covariance, which might be undesired.
    input_stats: dict. Dictionary of analytical values used as dependency
      injection for testing purposes.

  Returns:
    Tuple of (meanR, covR, num_min_moments) where meanR is the latent mean,
      covR is the cross-covariance of future-past latent R, and num_min_moments
      is the number of moments adjusted to hit minimum threshold.
  """
  if debug_mode: extra_info = {}

  if not input_stats:
    # This is to threshold the minimum moments in the covariance matrix below.
    MIN_MOMENT = sys.float_info.epsilon * 100

    ny, num_measurements = Y.shape
    j = hankel_utils.compute_hankel_parameters(num_measurements, horizon)
    Yp = hankel_utils.make_hankel(Y, horizon, j)
    Yf = hankel_utils.make_hankel(Y, horizon, j, horizon)
    Ystacked = np.vstack((Yf, Yp))

    # Estimate mean for each observation at each time lag.
    meanY = np.mean(Ystacked, axis=1, keepdims=True)
    fix_hankel_degeneracies = False
    if np.any(meanY == 0):
      print('Poorly sampled moments introducing degeneracies in Hankel.')
      # This means there aren't any observations in the horizon we're considering
      # for some channels, so we instead compute mean over all time (not just horizon)
      # and update the invalid features. This is only valid because of the stationarity
      # assumption and that ergodic mean approximately is equal to the process mean.
      emp_meanY = np.mean(Y, axis=1, keepdims=True)
      if np.any(emp_meanY == 0):
        print('There are flat features...will offset with small epsilon.')
        if debug_mode: extra_info['flat_channels'] = np.where(emp_meanY == 0)[0]
        emp_meanY[emp_meanY == 0] += sys.float_info.epsilon

      print('Fixing Hankel degeneracies.')
      inds_to_fix = np.where(meanY == 0)[0] # Flat channels.
      meanY[inds_to_fix] = emp_meanY[inds_to_fix % ny]
      fix_hankel_degeneracies = True
      if debug_mode: extra_info['degenerate_feature_indices'] = inds_to_fix

    if use_cov: # Demean before computing correlation.
      Ystacked -= meanY

    # Compute the future/past stacked Hankel matrix.
    covS = (Ystacked @ Ystacked.T) / (Ystacked.shape[1] - 1)

    if fix_hankel_degeneracies:
      # Need to also fix the feature variances corresponding to the mean values
      # computed above with the variance computed across all samples.
      var_from_all_Y = np.var(Y, axis=1, ddof=1)
      covS[inds_to_fix, inds_to_fix] = var_from_all_Y[inds_to_fix % ny]
      if not use_cov: # covS is actually corrS, adjust accordingly.
        covS[inds_to_fix, inds_to_fix] += meanY[inds_to_fix].squeeze()**2

  else: # If input_stats: Used for unit test dependency injection.
    MIN_MOMENT = sys.float_info.epsilon * 100
    covS = input_stats['covS'] if use_cov else input_stats['corrS']
    meanY = input_stats['meanY'].reshape((covS.shape[0], 1))

  if debug_mode:
    extra_info['meanY'] = meanY
    extra_info['covS'] = covS

  # Threshold fano factor to be a bit above 1 to ensure the log terms below
  # are well defined. For covS == corrS, thresholding to be a little greater
  # than 1 + meanY_i.
  if correct_fano:
    fano_factors = np.diag(covS) / meanY.squeeze()
    expected_fano = 1 if use_cov else 1 + meanY.squeeze() # else means covS == corrS
    if np.any(fano_factors <= expected_fano):
      print('Correcting fano factor')
      fano_inds = np.where(fano_factors <= expected_fano)[0]
      if debug_mode:
        extra_info['fano_inds'], extra_info['covS_pre_ff'] = fano_inds, covS.copy()

      scale_factor = np.ones(np.size(fano_factors))
      scale_factor[fano_inds] = ((expected_fano + _DIAG_OFFSET) / fano_factors)[fano_inds]
      scale_mat = np.diag(np.sqrt(scale_factor))
      covS = scale_mat @ covS @ scale_mat
      if debug_mode: extra_info['covS'] = covS
  
  if use_cov:
    beta = covS + meanY @ meanY.T
  else:
    beta = covS
  beta -= np.diag(meanY.squeeze())

  min_moment_inds = beta < MIN_MOMENT
  num_min_moment = np.sum(min_moment_inds)
  if np.any(min_moment_inds): # Threshold moments to minimum value.
    print('Adjusting minimum moment')
    beta[min_moment_inds] = MIN_MOMENT
    if debug_mode: extra_info['min_moment_inds'] = min_moment_inds

  meanR = 2 * np.log(meanY) - 0.5 * np.log(np.diag(beta)).reshape(meanY.shape)
  covR = np.log(beta) - np.log(meanY @ meanY.T)
  assert covR.shape == covS.shape, 'covR {0}, covS {1}'.format(covR.shape, covS.shape)

  # Will enforce PSD matrices by thresholding eigenvalues and reconstructing.
  if not input_stats and correct_PSD and not matrix_utils.is_PSD(covR):
    if debug_mode: extra_info['preEig_covR'] = covR.copy()

    E, U = np.linalg.eig(covR)
    E, U = E.real, U.real # np.linalg.eig() complex by default; imag is approx 0.
    if eig_threshold_correction:
      # Using same approach as Buesing implementation: to prevent potentially
      # changing rank of the matrix.
      min_eig_inds = E <= _MIN_EIG
      if debug_mode: extra_info['min_eig_inds'] = min_eig_inds
      E[min_eig_inds] = _MIN_EIG

    else: # not eig_threshold_correction
      # Alternate approach but this changes the dimensionality of the covariance matrix.
      inds_to_keep = np.where(E >= sys.float_info.epsilon)[0]
      U = U[:, inds_to_keep]
      E = E[inds_to_keep]

    covR = U @ np.diag(E) @ U.T # Reconstruct covR.

  if debug_mode: extra_info['presym_covR'] = covR.copy()
  covR = matrix_utils.make_symmetric(covR)

  if debug_mode: return meanR, covR, num_min_moment, extra_info
  return meanR, covR, num_min_moment

def transformation_of_poisgaus_crosscovariate_moments(Y, Z, horizon,
        Z_horizon=None, meanR=None, covR=None, force_mean_compute=False,
        input_stats={}, debug_mode=False):
  """Moment conversion implementation for joint observations: equation (11).

  Args:
    Y: np.ndarray of shape (features, samples). Primary Poisson time-series data.
    Z: np.ndarray of shape (features, samples). Secondary Gaussian time-series data.
    horizon: int. Y horizon value.
    Z_horizon: int. Z horizon value.
    meanR: np.ndarray of shape (y_features*2*horizon, 1). This is the mean
      output of the moment conversion from Buesing et al (transformation_of_moments()
      above). meanR and covR (if provided) can be used to compute meanY for the
      moment conversion.
    covR: np.ndarray of shape (y_features*2*horizon, y_features*2*horizon).
      This is the covariance output of the moment conversion from Buesing et al
      (transformation_of_moments() above). It not provided, meanY will be
      computed empirically. meanR and covR (if provided) can be used to compute
      meanY for the moment conversion.
    force_mean_compute: bool. Default is False. If true, will compute the mean
      of observations Y again rather than using the provided meanR and covR values.
  
  Returns:
    (xcorrZS.T / meanY).T, xcorrZS: Two-element tuple of transformed cross-covariance
      and observed cross-covariance.
  """
  if debug_mode: extra_info = {}

  if not input_stats:
    if Y.shape[1] != Z.shape[1]:
      raise ValueError('Z and Y must have the same number of measurements (i.e., columns).')

    ny, num_measurements = Y.shape
    nz, num_z_measurements = Z.shape
    j = hankel_utils.compute_hankel_parameters(num_measurements, horizon,
          num_second_observations=num_z_measurements, second_horizon=Z_horizon)
    if not Z_horizon: Z_horizon = horizon

    # Demean prior to constructing the Hankel matrix. Optional. See note before
    # the return statement.
    Z = Z - np.mean(Z, axis=1, keepdims=True)
    Yp = hankel_utils.make_hankel(Y, horizon, j)

    Zf = hankel_utils.make_hankel(Z, Z_horizon, j, start=Z_horizon)
    meanZf = np.mean(Zf, axis=1, keepdims=True)
    xcorrZS = (Zf @ Yp.T) / (Zf.shape[1] - 1)

    # The following are needed for computations later.
    if covR is None or meanR is None or force_mean_compute:
      # Compute meanY if prior moment conversion output is not provided.
      meanY = np.mean(Yp, axis=1, keepdims=True)
    else:
      # Otherwise compute meanY from the provided moment converion outputs.
      meanY = np.exp(0.5 * np.diag(covR).reshape(meanR.shape) + meanR)
      meanY = meanY[-horizon*ny:, ...]

  else: # input_states: Used to inject dependencies in unit tests.
    xcorrZS = input_stats['xcorrZS']
    # Only need Yp part of input meanY.
    meanY = input_stats['meanY'].squeeze()[-xcorrZS.shape[1]:, np.newaxis]
    # Take only the future part of meanZ.
    meanZf = input_stats['meanZ'].squeeze()[:xcorrZS.shape[0], np.newaxis]

  if debug_mode: extra_info['xcorrZS'] = xcorrZS

  assert meanY.shape[0] == xcorrZS.shape[1]
  assert meanZf.shape[0] == xcorrZS.shape[0]
  
  # If Z is not demeanded before computation (earlier in the function), use the
  # following commented out line instead (based on transformation of moments).
  # return (xcorrZS.T / meanY).T - meanZf, xcorrZS
  return (xcorrZS.T / meanY).T, xcorrZS

def transformation_of_poispois_crosscovariate_moments(Y, T, horizon,
        T_horizon=None, meanR=None, meanZ=None, covR=None, covZ=None,
        force_mean_compute=False, debug_mode=False):
  """Moment conversion for joint Poisson-Poisson observations: equation (12).

  Args:
    Y: np.ndarray of shape (features, samples). Primary Poisson time-series data.
    T: np.ndarray of shape (features, samples). Secondary Poisson time-series data.
    horizon: int. Y horizon value.
    T_horizon: int. T horizon value.
    meanR: np.ndarray of shape (Y_features*2*horizon, 1). This is the mean
      output of the moment conversion from Buesing et al (transformation_of_moments()
      above). meanR and covR (if provided) can be used to compute meanY for the
      moment conversion.
    meanZ: np.ndarray of shape (T_features*2*T_horizon, 1). This is the mean
      output of the moment conversion from Buesing et al (transformation_of_moments()
      above). meanZ and covZ (if provided) can be used to compute meanT for the
      moment conversion.
    covR: np.ndarray of shape (Y_features*2*horizon, Y_features*2*horizon).
      This is the covariance output of the moment conversion from Buesing et al
      (transformation_of_moments() above). It not provided, meanY will be
      computed empirically. meanR and covR (if provided) can be used to compute
      meanY for the moment conversion.
    covZ: np.ndarray of shape (T_features*2*T_horizon, T_features*2*T_horizon).
      This is the covariance output of the moment conversion from Buesing et al
      (transformation_of_moments() above). It not provided, meanT will be
      computed empirically. meanZ and covZ (if provided) can be used to compute
      meanT for the moment conversion.
    force_mean_compute: bool. Default is False. If true, will compute the mean
      of observations Y again rather than using the provided meanR and covR values.

    Returns:
      (xcorrZR, num_min_moments): Two-element tuple of second-order cross-covariance
        and number adjusted moments.
  """
  if debug_mode: extra_info = {}

  if Y.shape[1] != T.shape[1]:
    raise ValueError('Y and T must have the same number of measurements (i.e., columns).')

  ny, num_measurements = Y.shape
  nt, num_t_measurements = T.shape
  j = hankel_utils.compute_hankel_parameters(num_measurements, horizon,
        num_second_observations=num_t_measurements, second_horizon=T_horizon)
  if not T_horizon: T_horizon = horizon

  Yp = hankel_utils.make_hankel(Y, horizon, j)
  Tf = hankel_utils.make_hankel(T, T_horizon, j, start=T_horizon)

  xcorrTY = (Tf @ Yp.T) / (Tf.shape[1] - 1)

  ### To prevent numerical instabilities when taking log.
  MIN_MOMENT = sys.float_info.epsilon * 100
  min_moment_inds = xcorrTY < MIN_MOMENT
  num_min_moment = np.sum(min_moment_inds)
  if np.any(min_moment_inds): # Threshold moments to minimum value.
    print('Adjusting minimum moment')
    xcorrTY[min_moment_inds] = MIN_MOMENT
    if debug_mode: extra_info['min_moment_inds'] = min_moment_inds

  if debug_mode: extra_info['xcorrTY'] = xcorrTY
  xcorrZR = np.log(xcorrTY)

  # Index i (row) is for T variables and j (column) for Y variables
  if (covZ is not None or covR is not None or meanZ is not None or 
      meanR is not None or force_mean_compute):
    xcorrZR -= 0.5 * np.diag(covZ)[:, np.newaxis]
    xcorrZR -= 0.5 * np.diag(covR)[np.newaxis, :]
    xcorrZR -= meanZ.squeeze()[:, np.newaxis]
    xcorrZR -= meanR.squeeze()[np.newaxis, :]
  
  else: ## Use empirical means.
    Yp_mean = np.mean(Yp, axis=1, keepdims=True)
    Tf_mean = np.mean(Tf, axis=1, keepdims=True)
    xcorrZR -= np.log(Tf_mean)
    xcorrZR -= np.log(Yp_mean.T)

  assert xcorrZR.shape == xcorrTY.shape, \
    'xcorrZR {0}, xcorrTY {1}'.format(xcorrZR.shape, xcorrTY.shape)

  if debug_mode: return xcorrZR, num_min_moment, extra_info
  return xcorrZR, num_min_moment

def transformation_of_berngaus_crosscovariate_moments(Y, Z, horizon, Z_horizon=None):
  """Wrapper around bestLDS Bernoulli-Gaussian moment conversion function.

  Args:
    Y: np.ndarray of shape (features, samples). Primary Bernoulli time-series data.
    Z: np.ndarray of shape (features, samples). Secondary Gaussian time-series data.
    horizon: int. Y horizon value.
    Z_horizon: int. Z horizon value.

  Returns:
    First- and second-order moments (meanR, meanZ, covR, covZ, covZR).
  """
  ny, num_measurements = Y.shape
  nz, num_z_measurements = Z.shape
  if not Z_horizon: Z_horizon = horizon
  # Max possible number of measurements.
  j = hankel_utils.compute_hankel_parameters(num_measurements, horizon,
    num_second_observations=num_z_measurements, second_horizon=Z_horizon)

  Yp = hankel_utils.make_hankel(Y, horizon, j)
  Yf = hankel_utils.make_hankel(Y, horizon, j, horizon)
  Ystacked = np.vstack((Yf, Yp)).T # samples x (2 * horizon * ny)
  
  Zp = hankel_utils.make_hankel(Z, Z_horizon, j)
  Zf = hankel_utils.make_hankel(Z, Z_horizon, j, Z_horizon)
  Zstacked = np.vstack((Zf, Zp)).T # samples x (2 * Z_horizon * nz)

  out = bestLDS_moments.fit_mu_sigma_bernoulli_driven(Ystacked, Zstacked)
  meanR, meanZ, covR, covZ, xcovZR = out

  # Only the future-past relationship is needed for PGLDM.
  xcovZR = hankel_utils.extract_crosscorrelation(xcovZR, nz,
              Z_horizon, pair='fp', second_var_dim=ny, second_horizon=horizon)
  covR = bestLDS_moments.tril_to_full(covR, 2 * horizon * ny)
  return meanR[:, np.newaxis], meanZ[:, np.newaxis], covR, covZ, xcovZR
