"""Math utility functions."""
import numpy as np

def polar_to_rectangular(magnitude, theta):
  """Convert polar coordinates to rectangular coordinates."""
  return magnitude * np.exp(1j*theta)

def log_normal_moment_transform(log_mean, log_covariance):
  """Computes first and second moments of a log normal variable."""
  mean = np.exp(0.5 * np.diag(log_covariance)[:, np.newaxis] + log_mean)
  
  exp_log_covariance = np.exp(log_covariance)
  diag_exp_log_covariance = np.power(np.diag(exp_log_covariance)[:, np.newaxis], 0.5)
  covariance = exp_log_covariance * (diag_exp_log_covariance @ diag_exp_log_covariance.T)
  exp_log_mean = np.exp(log_mean)
  covariance *= exp_log_mean @ exp_log_mean.T
  covariance += np.diag(np.atleast_1d(mean.squeeze())) - mean @ mean.T

  return mean, covariance

def transform_to_log_normal_single_variable(mean, var):
  """Compute moment conversion from scalar Poisson mean/variance to log-rate mean/variance."""
  beta = var + mean * mean - mean
  log_mean = 2 * np.log(mean) - 0.5 * np.log(beta)
  log_cov = np.log(beta) - np.log(mean * mean)
  return log_mean, log_cov

def transform_from_log_normal_single_variable(log_mean, log_var):
  """Compute moment conversion from scalar log-rate mean/variance to Poisson mean/variance."""
  mean = np.exp(log_mean + 0.5 * log_var)
  var = mean + np.exp(log_var) * mean * mean - mean * mean
  return mean, var