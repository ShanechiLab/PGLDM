"""Module used for generating noise."""
import numpy as np

def generate_random_gaussian_noise(num_samples, noise_dim, covariance=None):
  """Generates random gaussian noise vectors. Can be colored noise.

  Args:
  	num_samples: int. Number of time samples to generate.
  	noise_dim: int. Noise dimension.
  	covariance: np.ndarray of shape (noise_dim, noise_dim). Covariance for the
  		colored Gaussian noise.

  Returns:
  	Noise as a np.ndarray of shape (num_samples, noise_dims).
  """
  if covariance is None:
    covariance = np.identity(noise_dim)
  elif covariance.shape[0] != noise_dim:
    print('Mismatch between covariance dimensions and input noise_dim param.')
    print('Will use covariance dimensions.')
    noise_dim = covariance.shape[0]

  eigvals, V = np.linalg.eig(covariance)
  if np.any(eigvals < 0):
    print('Warning: Noise covariance matrix is not PSD: ', eigvals)
    coloring = np.real(np.matmul(V, np.sqrt(np.diag(eigvals))))
  else:
    coloring = np.real(np.linalg.cholesky(covariance)) # is real needed?
  w = coloring @ np.random.randn(noise_dim, num_samples)
  return w.T