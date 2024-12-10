""""Hankel matrix related utility functions."""

import math_utils.matrix_utils as matrix_utils
import numpy as np

def compute_hankel_parameters(num_observations, horizon,
                              num_second_observations=None, second_horizon=None):
  """Computes the number of observations in a Hankel matrix.

  This method by default assumes there will be separate future and past Hankel
  matrices and therefore adjusts the number of observations based on the horizon
  length.

  Args:
    num_observations: int. Number of observations all together.
    horizon: int. The horizon to use in the Hankel matrix.
    num_second_observations: int, optional. If there will be a cross Hankel
      product computed between two different observations, provide number of
      secondary observations.
    second_horizon: int, optional. Horizon for the second signal if using
      different horizons.
  """
  if not num_second_observations:
    return num_observations - 2 * horizon + 1

  if not second_horizon:
    second_horizon = horizon
  second_j = compute_hankel_parameters(num_second_observations, second_horizon)
  first_j = compute_hankel_parameters(num_observations, horizon)
  return min(first_j, second_j)

def make_hankel(Y, horizon, j, start=0):
  """Construct a Hankel matrix using the columns of Y.
  Arguments:
    Y: Y matrix of dimension (ny, num_samples).
    horizon: The number of Y samples that make up the horizon. Output matrix is
       (ny*horizon, j).
    j: The number of samples/columns. Output matrix is (ny*i, j).
    start: Optional integer, corresponding to which column of Y we should start
      building the matrix from, i.e. H[0, 0] = Y[:, i].
  """
  ny, num_samples = Y.shape
  Y_hank = np.empty((ny*horizon, j))
  for col in range(j):
    begin = start + col
    Y_hank[:, col:col+1] = np.reshape(Y[:, begin:begin+horizon], (ny*horizon, 1), order='F')
  return Y_hank

#### The following are helper methods intended to work with stacked
#### future-past Hankel matrices of the form:
####   E[ZZ.T] = [[YfYf.T, YfYp.T], [YpYf.T, YpYp.T]]
#### where Z = [[Yf], [Yp]] (future & past observations stacked on top of
#### each other).

def compute_average_variance(cov_mat, var_dim, horizon):
  if cov_mat.shape != (var_dim * horizon, var_dim * horizon):
    raise ValueError('cov_mat must be of shape (var_dim * horizon, var_dim * horizon)')
  return compute_average_lag_mat(cov_mat, 0, var_dim, horizon)

def compute_average_lag_mat(cov_mat, lag, var_dim, horizon):
  """Will compute the average cross-correlation matrix for a desired lag.
  Implementation support instantaneous correlations and correlations between
  past and future, (i.e., negative lags). However, due to the symmetric structure
  of the correlation matrices as defined in the VODM text, simply pass in the
  transpose of the covariance matrix for a future lag (i.e., future correlated
  with the past).

  Args:
    cov_mat: np.ndarray. Either the instantaneous correlation or
      lagged-correlations Hankel matrix from which to extract a particular lag.
    lag: int. Desired lag relative to zero.
    var_dim: int. Variable feature dimensions.
    horizon: int. Horizon corresponding to input Hankel matrix.
  """
  if lag < 0:
    lag = np.abs(lag)

  num_to_average = horizon - lag
  ave = np.zeros((var_dim, var_dim))
  enum_inds = np.arange(var_dim)
  row = 0
  for i in range(lag, horizon):
    ave += matrix_utils.extract_block(cov_mat, enum_inds + (row*var_dim), enum_inds + (i*var_dim))
    row += 1
  ave = np.divide(ave, num_to_average)
  return ave

def extract_correlation(cov_mat, var_dim, horizon, pair='fp'):
  """
  Extract the desired correlation quadrant from a stacked future-past Hankel
  matrix. This method works with Hankel matrices of a single variable. For Hankel
  matrices between two different variables see extract_crosscorrelation() below.
  """
  block_one = horizon * var_dim
  if pair == 'ff':
    return matrix_utils.extract_block(cov_mat, np.arange(block_one),
                         np.arange(block_one))
  elif pair == 'pf':
    return matrix_utils.extract_block(cov_mat, np.arange(block_one) + block_one,
                         np.arange(block_one))
  elif pair == 'fp':
    return matrix_utils.extract_block(cov_mat, np.arange(block_one),
                         np.arange(block_one) + block_one)
  elif pair == 'pp':
    return matrix_utils.extract_block(cov_mat, np.arange(block_one) + block_one,
                         np.arange(block_one) + block_one)
  raise ValueError('pair must be one of: "ff", "pp", "fp", "pf"')

def extract_crosscorrelation(cov_mat, var_dim, horizon, second_var_dim,
                             second_horizon=None, pair='fp'):
  """
  Extract the desired correlation quadrant from a stacked future-past Hankel
  matrix between two different variables of potentially different dimensions.
  Also supports Hankel matrices with different horizon values for each variable.
  """
  first_var_block = horizon * var_dim
  if second_var_dim and second_horizon:
    second_var_block = second_horizon * second_var_dim
  elif second_var_dim:
    second_var_block = horizon * second_var_dim
  elif second_horizon:
    second_var_block = second_horizon * var_dim
  else:
    second_var_block = first_var_block

  if pair == 'ff':
    return matrix_utils.extract_block(cov_mat, np.arange(first_var_block),
                                      np.arange(second_var_block))
  elif pair == 'pf':
    return matrix_utils.extract_block(cov_mat, np.arange(first_var_block) + first_var_block,
                                      np.arange(second_var_block))
  elif pair == 'fp':
    return matrix_utils.extract_block(cov_mat, np.arange(first_var_block),
                                      np.arange(second_var_block) + second_var_block)
  elif pair == 'pp':
    return matrix_utils.extract_block(cov_mat, np.arange(first_var_block) + first_var_block,
                                      np.arange(second_var_block) + second_var_block)
  raise ValueError('pair must be one of: "ff", "pp", "fp", "pf"')
    
def extract_lag_mat(cov_mat, lag, var_dim, horizon, corr_pair):
  if corr_pair not in ['ff', 'fp', 'pf', 'pp']:
    raise ValueError('corr_pair must be one of: "ff", "pp", "fp", "pf"')

  # Order matters here because we're trying to convert the lags and the overall
  # matrix to a consistent format in order to use the same subroutine. Notation
  # follows VODM text.
  if corr_pair == 'pf':
      lag = -1 * lag
  if corr_pair == 'fp' or corr_pair == 'pf':
      lag -= horizon
  if lag > 0: # Symmetric property of the lag matrix.
      cov_mat = cov_mat.T
  return compute_average_lag_mat(cov_mat, np.abs(lag), var_dim, horizon)

def construct_future_past_stacked_hankel(observation_mat, horizon,
                    observation_mat2=None, secondary_horizon=None):
  """Constructs the future-past *stacked* Hankel matrix described above.

  Args:
    observation_mat: Observations as a np.array of shape
      (observation_dims, num_measurements).
    horizon: int. Horizon.
    observation_mat2: Optional second matrix of observations. If provided the
      resultant matrix will be:
        E[ZZ.T] = [[YfYf.T, YfYp.T], [YpYf.T, YpYp.T]]
      where Z = [[Obs_1_f], [Obs_2_p]]. Specifically, future observations from
      matrix 1 & past observations from matrix 2 stacked on top of each other.
    secondary_horizon: int. Optional second horizon to be used with past
      observations vector. If not provided will use same horizon for future and
      past observations.

  Returns:
    Square Hankel matrix of dimension:
      ((past_horizon*num_past_obs_dims + future_horizon*num_future_obs_dims), j)
    where j is computed as:
      min(num_past_obs - 2*past_horizon, num_future_obs - 2*future_horizon)
  """
  past_obs_mat = observation_mat
  past_horiz = horizon
  if observation_mat2 is not None:
    past_obs_mat = observation_mat2
  if secondary_horizon:
    past_horiz = secondary_horizon

  num_pastobs_measurements = past_obs_mat.shape[1]
  num_futureobs_measurements = observation_mat.shape[1]
  j = 1 + min(num_futureobs_measurements - 2*horizon,
              num_pastobs_measurements - 2*past_horiz)

  Of = make_hankel(observation_mat, horizon, j, horizon)
  Op = make_hankel(past_obs_mat, past_horiz, j)
  return np.vstack((Of, Op))

def construct_future_past_stacked_correlation(observation_mat, horizon,
          observation_mat2=None, secondary_horizon=None, demean=False):
  """Constructs the future-past *stacked* correlation matrix described above.

  These are matrices of the form:
      E[ZZ.T] = [[YfYf.T, YfYp.T], [YpYf.T, YpYp.T]]
  where Z = [[Yf], [Yp]] (future & past observations stacked on top of each
  other).

  Args:
    observation_mat: Observations as a np.array of shape
      (observation_dims, num_measurements).
    horizon: int. Horizon.
    observation_mat2: Optional second matrix of observations. If provided the
      resultant matrix will be:
        E[ZZ.T] = [[YfYf.T, YfYp.T], [YpYf.T, YpYp.T]]
      where Z = [[Obs_1_f], [Obs_2_p]]. Specifically, future observations from
      matrix 1 & past observations from matrix 2 stacked on top of each other.
    secondary_horizon: Optional second horizon to be used with past observations
      vector. If not provided will use same horizon for future and past
      observations.
    demean: Optional boolean argument. If true, will demean observations before
      computing the Hankel (i.e., covariance vs. correlation).

  Returns:
    Square Hankel matrix of dimension:
      i) (horizon * observation_dims) x (horizon * observation_dims) if only
        one observation matrix is provided.
      ii) (horizon * observation1_dims) x (horizon * observation2_dims) if two
        observation matrices are provided.
      iii) (horizon * observation1_dims) x (horizon2 * observation_dims) if one
        observation matrix and two horizons are provided.
      iv) (horizon * observation1_dims) x (horizon2 * observation2_dims) if one
        observation matrix and two horizons are provided.
  """
  Ostacked = construct_future_past_stacked_hankel(observation_mat, horizon,
    observation_mat2=observation_mat2, secondary_horizon=second_horizon)
  if demean:
    Ostacked -= np.mean(Ostacked, axis=1).reshape((Ostacked.shape[0], 1))
  return (Ostacked @ Ostacked.T) / j
