"""Linear state-space model class."""
import math_utils.matrix_utils as matrix_utils
import models.model_base as model_base
import models.noise_generator as noise_gen
import models.SID_utils as SID_utils
import models.SSM_utils as SSM_utils
import numpy as np

class LSSM(model_base.Model):
  """Linear SSM, defined by the following Gaus-Markov relationships:
  1) Supports forward stochastic model:
      x[k+1] = A @ x[k] + w[k]
      y[k] = C @ x[k] + v[k]
      E[[w v].T @ [w v]] = [[Q S], [S, R]]
  2) Forward stochastic innovation model:
      x[k+1] = A @ x[k] + K @ fe[k]
      y[k] = C @ x[k] + e[k]
      e[k] = y[k] - C @ x[k](Yp) <- innovation term, where x[t](Yp) is est. of x

  Where the model is defined as per VODM text p58 and p63.
  Required parameters are:
    - A : as defined above
    - C : as defined above

  and one of the following sets:
    1) G & L0, where:
      - G : E[x[k+1] @ y[k].T] correlation betweek x[k+1] and y[k]
      - L0 : E[y[k] @ y[k].T] covariance of (zero-mean) y[k]
    2) Q, R, S, the noise statistics as defined above.

  Recommended usage (see factory function at the end of the module):
    sys = update_missing_params_and_construct_LSSM(params)
  """
  def __init__(self, params):
    self.params = self.__format_params(params)
    self.state_dim = params['A'].shape[0]
    self.output_dim = params['C'].shape[0]
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

  def _readout_states(self, X, other_outputs={}):
    """Generate outputs for provided states.

    Arguments:
      X: np.ndarray of shape (num_samples, state_dim)

    Returns:
      all_Yp: Predicted output (Y) as a np.ndarry of shape
        (num_samples, output_dim).
      other_predictions: Dictionary of other outputs using the provided readout
        matrices.
    """
    all_Yp = np.transpose(self.C @ X.T)
    # Perform any other predictions if requested (e.g., Z via provided Cz).
    other_predictions = dict()
    for output, output_C in other_outputs.items():
      if output_C.shape[1] != self.state_dim:
        print("Ignoring output {0}, wrong state dim".format(output))
        continue
      other_predictions[output] = np.transpose(output_C @ X.T)
    return all_Yp, other_predictions

  def _generate_realization_with_QRS(self, num_samples, x0=None, w0=None, return_noise=False):
    QRS = np.block([[self.Q, self.S], [self.S.T, self.R]])
    # wv dimensions are samples x (state_dim + output_dim)
    wv = noise_gen.generate_random_gaussian_noise(num_samples, QRS.shape[0], QRS)
    # Dimensions are samples x state_dim
    w = wv[:, :self.state_dim]
    # Dimensions are samples x output_dim
    v = wv[:, self.state_dim:]

    if x0 is None:
      x0 = np.zeros((self.state_dim, 1))
    if w0 is None:
      w0 = np.zeros((self.state_dim, 1))
    X = np.empty((num_samples, self.state_dim))
    Y = np.empty((num_samples, self.output_dim))
    for i in range(num_samples):
      if i == 0:
        xt_1, wt_1 = x0, w0
      else:
        xt_1 = X[i-1, :].T
        wt_1 = w[i-1, :].T
      X[i, :] = (self.A @ xt_1 + wt_1).T
      Y[i, :] = (self.C @ X[i, :].T + v[i, :].T).T
    if return_noise: return Y, X, v, w
    return Y, X

  def _generate_realization_with_kalman(self, num_samples, x0=None, **kwargs):
    innovations = noise_gen.generate_random_gaussian_noise(num_samples,
                                        self.INNOV_COV.shape[0], self.INNOV_COV)
    if x0 is None:
      x0 = np.zeros((self.state_dim, 1))
    X = np.empty((num_samples, self.state_dim))
    Y = np.empty((num_samples, self.output_dim))
    Xp = x0
    for i in range(num_samples):
      innov = innovations[i, :][:, np.newaxis]
      Yk = self.C @ Xp + innov
      X[i, :] = np.squeeze(Xp)
      Y[i, :] = np.squeeze(Yk)
      Xp = self.A @ Xp + self.K @ innov
    return Y, X

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
    """Returns a LSSM with the new params. Does *not* modify in place."""
    return update_missing_params_and_construct_LSSM(new_params)

  # LSSM module does not support a default fit function. User is recommended to
  # fit model parameters via some approach such as ML and provide learned
  # parameters to this class.
  def fit(self, _):
    pass

  def kalman(self, Y, x0=None):
    """Runs a Kalman filter on the provided observations.

    NOTE: This is one-step ahead prediction, *not* estimation of current state.

    Args:
      Y: np.ndarray, (num_samples, output_dim). Observations.
      x0: np.ndarray, (state_dim, 1). Initial state. Default is 0.

    Returns:
      Xf: np.ndarray, (num_samples, state_dim). Predicted states.
    """
    if Y.shape[1] != self.output_dim:
      raise ValueError("Input Y does not match LSSM output_dim.")

    # Following VODM definition p63
    #   x_f[k+1] = A @ x_f[k] + Kf @ innov[k]
    #   y[k] = C @ x_f[k] + innov[k]
    # where:
    #   innov[k] = Y[k] - C @ x_f[k]
    num_samples = Y.shape[0]
    all_Xf = np.empty((num_samples, self.state_dim))  # x_f[i] a.k.a. x(i|i-1)
    if x0 == None:
      x0 = np.zeros((self.state_dim, 1))
    xf = x0 # First prediction is initial state.
    for i in range(num_samples):
      all_Xf[i, :] = xf.T # x_f[i] a.k.a. x(i|i-1)
      y = np.reshape(Y[i, :], (self.output_dim, 1))
      zi =  y - self.C @ xf # innovation
      xf = self.A @ xf + self.K @ zi
      # xf = (self.A - self.K @ self.C) @ xf + self.K @ y # Alternate equation.
      if np.any(np.isnan(xf)) or np.any(np.isinf(xf)):
        raise ValueError('Kalman filter blew up.')
    return all_Xf

  def estimate_similarity_transform(self, other_sys : model_base.Model):
    """Compute a similarity transform that converts other_sys to the same bases
    as this system.

    Estimates a similarity transform between system 1 and system 2, such that:
      sys1_x = T @ sys2_x  <------> T^{-1} @ sys1_x = sys2_x
      sys1_A = T @ sys2_A @ T^{-1} <------> T^{-1} @ sys1_A @ T = sys2_A
      sys1_C = sys2_C @ T^{-1} <------> T @ sys1_C = sys2_C
    In this notation, sys1 can be thought of as the "target" system we want to
    align the second system with.

    Args:
      other_sys: Model. The "source" model that needs to be transformed to this
        ("target") model bases. Currently other_sys *must* be LSSM object.

    Returns:
      Similarity transform T as a np.ndarray of shape (state_dim, state_dim).
    """
    if not isinstance(other_sys, type(self)):
      raise ValueError('Can only compute similarity transform for a LSSM.')

    # Need a large amount of samples to ensure we actually converge.
    Y, _ = self.generate_realization(int(1e6))

    sys1_x_pred = self.kalman(Y)
    sys2_x_pred = other_sys.kalman(Y)
    return np.transpose(matrix_utils.inverse(sys2_x_pred) @ sys1_x_pred)

  def predict(self, Y, other_outputs={}):
    """Predicts states, observations, and other user-requested outputs.

    Args:
      Y: np.ndarray, (num_samples, output_dim). Observations.
      other_outputs: dict. Optional dictionary of other outputs to predict. Keys
        must be variable name and the value is corresponding linear observation
        model as a np.ndarray. Eg: {'Z': Cz}

    Returns:
      (Yp, Xp, other_predictions).
        Yp: np.ndarray (num_samples, output_dim). One-step ahead observation
          predictions.
        Xp: np.ndarray (num_samples, state_dim). Predicted states.
        other_predictions: dict. Key values correspond to the keys in other_ouputs
          and the values are the associated realization prediction.
    """
    all_Xp = self.kalman(Y)
    all_Yp, other_predictions = self._readout_states(all_Xp, other_outputs)
    return all_Yp, all_Xp, other_predictions

  def generate_realization(self, num_samples, **kwargs):
    """Generate realization using linear state-space model.

    Args:
      num_samples: Number of samples to generate in realization.

    Optional kwargs:
      x0: A (state_dim,1) sized np.array. Initial state at t=0.
      w0: A (state_dim,1) sized np.array. Initial state at t=0.

    Returns:
      Y, X each of shape (num_samples, num_features) where Y are the observations
      and X are the states.
    """
    # If system Q an R are defined and are positive definite matrices, use noise
    # to generate the realization.
    if SSM_utils.has_QRS(self.params) and matrix_utils.is_PD(self.R) and matrix_utils.is_PD(self.Q):
      return self._generate_realization_with_QRS(num_samples, **kwargs)

    # NOTE: This method is guaranteed to work even when we solve the algebraic
    # Ricatti equation because the innovation covariance is forced to be symmetric,
    # therefore PD and therefore can be used to generate "noise" to generate a
    # realization.
    return self._generate_realization_with_kalman(num_samples, **kwargs)

def update_missing_params_and_construct_LSSM(params : dict) -> LSSM:
  """Factory function for finishing SID if any computable params are missing and
  constructing the corresponding LSSM. Will return a ValueError() if provided
  params fail any of the validity checks."""
  missing_params = []
  for required_param in ['A', 'C']:
    if params.get(required_param, None) is None:
        missing_params.append(required_param)
  if missing_params:
    raise ValueError('Missing required params: {0}'.format(', '.join(missing_params)))

  if not SSM_utils.has_GL0(params) and not SSM_utils.has_QRS(params):
    raise ValueError('Params need to include either QRS or G & L0.')

  params = SID_utils.compute_secondary_params(params)
  return LSSM(params)
