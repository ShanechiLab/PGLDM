"""Poisson Linear Dynamical System (PLDS) object."""
import math_utils.matrix_utils as matrix_utils
import models.LSSM as LSSM
import models.model_base as model_base
import models.noise_generator as noise_gen
import models.SSM_utils as SSM_utils
import numpy as np

class PLDS(model_base.Model):
  """Poisson Linear Dynamical System SSM.

    x[k+1] = A @ x[k] + w[k]
    r[k] = C @ x[k] + d
    y[k] | r[k] ~ Poisson(exp(r[k]))
    E[w.T @ w] = Q

  where the observations have a Poisson distribution conditioned on the linear
  log-rate term. (Generalized linear model with a log link function.)

  Where the model is defined as per VODM text p58 and p63.
  Required parameters are:
    - A : as defined above
    - C : as defined above
    - Q: as defined above

  Recommended usage (see factory function at the end of the module):
    sys = update_missing_params_and_construct_PLDS(params)
  """
  def __init__(self, params):
    self.params = self.__format_params(params)
    self.state_dim = self.params['A'].shape[0]
    self.output_dim = self.params['C'].shape[0]
    self.lssm = LSSM.LSSM(params)
    self.__set_attributes(self.params)

  def __format_params(self, params):
    for k, v in params.items():
      if type(v) is np.ndarray: params[k] = np.atleast_2d(v)
    return params

  def __set_attributes(self, params):
    # Set the params as attributes.
    for f, v in params.items():
      if not hasattr(self, f):
        setattr(self, f, v)

  def __kalman(self, Y, x0=None):
    """Perform Kalman filtering using the provided log-rates.

    NOTE: Providing spikes will results in erroneous outputs. Kalman filtering
    will ONLY work with log firing rates. This function is intended to be used
    for ground truth error validation purposes only.
    """
    return self.lssm.kalman(Y, x0=x0)

  def __ppf(self, Y, x0=None, P0=None, delta_t=1, return_all=False, mod_log=1e6):
    """Implementation of a stochastic point process filter for estimating latent
    states from point-process observations (see Eden et al 2004)."""
    if Y.shape[1] != self.output_dim:
      raise ValueError('Input Y does not match PLDS output_dim.')

    _ILL_TOL = 1e6 # Threshold values for ill-conditioning check.

    num_samples = Y.shape[0]

    # Initialize output data containers.
    all_Xp = np.empty((num_samples, self.state_dim))  # x(i|i-1)
    all_Pp = np.empty((num_samples, self.state_dim, self.state_dim)) # P(i|i-1)
    if return_all:
      all_Rp = np.empty((num_samples, self.output_dim))  # r(i|i-1)
      all_Xu = np.empty((num_samples, self.state_dim))  # x(i|i)
      all_Pu = np.empty((num_samples, self.state_dim, self.state_dim)) # P(i|i)

    # First state prediction is initial state and first error covariance
    # prediction is the "starting" covariance guess, typically identity matrix.
    if x0 is None: x0 = np.zeros((self.state_dim, 1))
    if P0 is None: P0 = np.eye(self.state_dim)

    xp, Pp = x0, P0
    log_throttle = -1 # Initialize logging throttle. 

    def record_current_states(i, xp, Pp, xu, Pu, rp):
      """Helper to log current states during each iteration."""
      all_Xp[i, :] = xp.T # x(i|i-1) -- state prediction
      all_Pp[i, :, :] = Pp # P(i|i-1) -- error covariance prediction
      if return_all:
        all_Xu[i, :] = xu.T # x(i|i) -- state update
        all_Pu[i, :, :] = Pu # P(i|i) -- error covariance update
        all_Rp[i, :] = rp.T # r(i|i-1) -- rate prediction

    def propagate(x, P):
      """Helper to propagate state & error covariance predictions."""
      xp = self.A @ x # Eden et al Eq. 2.7. x(i+1|i)
      Pp = self.A @ P @ self.A.T + self.Q # Eden et al Eq. 2.8. P(i+1|i)
      return xp, Pp

    for i in range(num_samples):
      log_throttle += 1

      # Predict r(i|i-1) from x(i|i-1)
      rp = np.exp(self.C @ xp + self.b) * delta_t # Precompute before try/catches.

      try:
        Pinv = matrix_utils.inverse(Pp)
      except np.linalg.LinAlgError as e:
        if log_throttle % mod_log == 0:
          print(f'Error inverting prediction matrix: {e}. Propagating without update.')

        record_current_states(i, xp, Pp, xp, Pp, rp) # 4th argument is what we propagate.
        xp, Pp = propagate(xp, Pp)
        continue # Cannot update, continue to next timestep.

      # This is a workaround to support single dimensional observations.
      if np.size(rp) == 1:
        rp_diag = np.diag(rp[0])
      else:
        rp_diag = np.diag(rp.squeeze())

      try:
        # Perform the update state with the latest measurement Y(i).
        Pinv += self.C.T @ rp_diag @ self.C # Eden et al 2004, Eq. 2.9. P(i|i)^(-1)
        cov_cond = np.linalg.cond(Pinv)
        if cov_cond >= _ILL_TOL or np.isnan(cov_cond) or np.isinf(cov_cond):
          if log_throttle % mod_log == 0:
            print('Warning: ill-conditioned Pinv in PPF! Propagating without update.')

          record_current_states(i, xp, Pp, xp, Pp, rp) # 4/5th argument is what we propagate.
          xp, Pp = propagate(xp, Pp)
          continue # Cannot update, continue to next timestep.
        
        # Not ill conditioned, continue with the update.
        Pu = matrix_utils.inverse(Pinv)
        y = Y[i:i+1, :].T # Observation y(i)
        innov = y - rp
        xu = xp + Pu @ self.C.T @ innov # Eden et al 2004, Eq. 2.10. x(i|i)

        record_current_states(i, xp, Pp, xu, Pu, rp) # 4th argument is what we propagate.
        xp, Pp = propagate(xu, Pu)

      except np.linalg.LinAlgError as e:
        if log_throttle % mod_log == 0:
          print(f'Ill conditioning check did not work: {e}. Condition was: {cov_cond}.')
          print('Propagating without update.')

        record_current_states(i, xp, Pp, xp, Pp, rp) # 4th argument is what we propagate.
        xp, Pp = propagate(xp, Pp)

    if return_all:
      return all_Xp, all_Pp, all_Xu, all_Pu
    return all_Xp, all_Pp

  def is_stable(self):
    return SSM_utils.is_stable(self.A)

  def get_list_of_params(self):
    params = {}
    for field in dir(self): 
      val = self.__getattribute__(field)
      if not field.startswith('__') and isinstance(val,
                                        (np.ndarray, list, tuple, type(self))):
        params[field] = val
    return params

  def update_params(self, new_params):
    """Returns a PLDS model with the new params. Does *not* modify in place."""
    return update_missing_params_and_construct_PLDS(new_params)

  # PLDS module does not support a default fit function. User is recommended to
  # fit model parameters via some approach such as ML and provide learned
  # parameters to this class.
  def fit(self, _):
    pass

  def estimate_similarity_transform(self, other_sys : model_base.Model):
    """Compute a similarity transform that converts other_sys to the same bases
    as this system.

    Estimates a similarity transform between system 1 and system 2, such that:
      sys1_x = T @ sys2_x  <------> T^{-1} @ sys1_x = sys2_x
      sys1_A = T @ sys2_A @ T^{-1} <------> T^{-1} @ sys1_A @ T = sys2_A
      sys1_C = sys2_C @ T^{-1} <------> T @ sys1_C = sys2_C
    In this notation, sys1 can be thought of as the "target" system we want to
    align the second system with.

    NOTE: This method will only work if full set of parameters are provided (i.e.,
    G and L0 are required if QRS isn't provided). Normally PLDS just requires Q
    and assumes R = S = 0, but these need to be explicitly specified.

    Args:
      other_sys: Model. The "source" model that needs to be transformed to this
        ("target") model bases.

    Returns:
      Similarity transform T as a np.ndarray of shape (state_dim, state_dim).
    """
    if not isinstance(other_sys, type(self)) and not isinstance(other_sys, LSSM.LSSM):
      raise ValueError('other_sys must be either LSSM or PLDS models.')
    
    R, _, Y = self.generate_realization(int(1e6), rates_only=False)
    Y = Y.astype('float64') # Default type is int64.
    
    # Need to convert to demeaned log-rates in order to use kalman filtering.
    LR = np.log(R) - self.b.T

    # Need to reconstruct LSSM and not use internal LSSM to ensure secondary
    # parameters (i.e., the Kalman gain) are computed.
    this_sys = LSSM.update_missing_params_and_construct_LSSM(self.get_list_of_params())
    sys1_x_pred = this_sys.kalman(LR)

    # Ditto.
    other_sys = LSSM.update_missing_params_and_construct_LSSM(other_sys.get_list_of_params())
    if isinstance(other_sys, type(self)):
      sys2_x_pred = other_sys.kalman(LR)
    else: # isinstance(other_sys, LSSM):
      Y -= np.mean(Y, axis=0, keepdims=True)
      sys2_x_pred = other_sys.kalman(Y) # Use demeaned spikes for linear model.
    return np.transpose(matrix_utils.inverse(sys2_x_pred) @ sys1_x_pred)

  def predict(self, Y, other_outputs={}, delta_t=1, debug_mode=False):
    """Predicts states, Poisson rates, and other user-requested outputs.

    Args:
      Y: np.ndarray of shape (num_samples, output_dim). Point-process observations.
      other_outputs: dict. Optional dictionary of other outputs to predict. Keys
        must be variable name and the value is corresponding linear observation
        model as a np.ndarray. Eg: {'Z': Cz}

    Returns:
      (R, X, other_predictions) where R and X are each of shape
      (num_samples, num_features) with R corresponding to the the rates and X to
      the states. other_predictions is a dictionary where the key values
      correspond to the keys in other_ouputs and the values are the associated
      realization prediction.
    """
    all_Xp, all_Pp = self.__ppf(Y, delta_t=delta_t)
    all_LRp, other_predictions = self.lssm._readout_states(all_Xp, other_outputs)
    all_LRp += self.b.T # Add baseline firing rate.
    all_Rp = np.exp(all_LRp) # Log rates -> rates.
    other_predictions['Pp'] = all_Pp # State prediction error covariance matrices.
    return all_Rp, all_Xp, other_predictions

  def generate_realization(self, num_samples, x0=None, w0=None, rates_only=True,
                           delta_t=1):
    """Generate realization using PLDS model.

    Args:
      num_samples: Number of samples to generate in realization.
      x0: np.ndarray of shape (state_dim, 1). Initial state at t=0.
      w0: np.ndarray of shape (state_dim, 1). Initial state at t=0.
      rates_only: bool. Returns only rate realization. If False, will also return
        Poisson observations. Default is True.
      delta_t: float. The duration of the time bin. Rates will be multiplied by
        delta_t before sampling the Poisson. (e.g. Poisson(delta_t * rates)).

    Returns:
      If rates_only is True:
        (R, X) each of shape (num_samples, num_features) where R are the rates &
        X are the states.
      If rates_only is False:
        (R, X, Y) each of shape (num_samples, num_features) where Y is the
        point-process realization.
    """
    X = np.zeros((num_samples, self.state_dim))
    LR = np.zeros((num_samples, self.output_dim))
    if x0 is None:
      x0 = np.zeros((self.state_dim, 1))
    if w0 is None:
      w0 = np.zeros((self.state_dim, 1))

    w = noise_gen.generate_random_gaussian_noise(num_samples, self.Q.shape[0], self.Q)
    for sample in range(num_samples):
      if sample == 0:
        wt_1, xt_1 = w0, x0
      else:
        xt_1 = X[sample-1:sample, :].T
        wt_1 = w[sample-1:sample, :].T

      X[sample, :] = np.transpose(self.A @ xt_1 + wt_1)
      LR[sample, :] = np.transpose(self.C @ X[sample:sample+1, :].T + self.b)

    R = np.exp(LR) # Log rates -> rates.
    if rates_only:
      return R, X
    Y = np.random.poisson(R * delta_t)
    return R, X, Y

def update_missing_params_and_construct_PLDS(params : dict) -> PLDS:
  """Factory function for constructing the corresponding PLDS model. Will return
  a ValueError() if provided params fail any of the validity checks."""
  missing_params = []
  for required_param in ['A', 'C', 'Q']:
    if params.get(required_param, None) is None:
        missing_params.append(required_param)
  if missing_params:
    raise ValueError('Missing required params: {0}'.format(', '.join(missing_params)))

  if params.get('b', None) is None:
    params['b'] = np.zeros((params['C'].shape[0], 1))
  elif params['b'].shape != (params['C'].shape[0], 1):
    raise ValueError('b vector must be shape ny-by-1.')

  if params.get('Cz', None) is not None: # Only for joint models.
    if params.get('d', None) is None:
      print('Warning: no baseline d is provided...will set d=0.')
      params['d'] = np.zeros((params['Cz'].shape[0], 1))
    elif params['d'].shape != (params['Cz'].shape[0], 1):
      raise ValueError('d vector must be shape nz-by-1.')

  return PLDS(params)
