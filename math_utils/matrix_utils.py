"""Matrix utility module."""
import numpy as np
import scipy.linalg
import sys

def extract_block(Y, inds, inds2=None):
  """Extracts the desired block from the matrix.
  Note: Block here means corresponding input and output column/row pairs. So
  for example we may have a matrix A:
  A = [[1, 2, 3, 4],
       [2, 3, 4, 5],
       [3, 4, 5, 6],
       [4, 5, 6, 7]]
  And we may want the following row/column indices [1, 3]. The corresponding
  output will be:
  B = [[3, 5],
       [5, 7]]

  Args:
    Y: Matrix to extract a block from.
    inds: Indices along the row-axis to select out.
    inds2: Indices along the column-axis to select out. If None provided, will
      use inds along the rows as well.
  """
  if inds2 is None:
    inds2 = inds
  return Y[np.ix_(inds, inds2)]

def extract_diagonal_blocks_w_inds(T, empty_side='lower', thresh=np.spacing(1)):
  """Extracts the block diagonal elements of the given matrix. Also returns the
  indices of the original matrix that correspond to each of the extracted blocks."""
  # T_threshed = T.copy()
  if empty_side == 'lower':
    inds = np.tril(np.abs(T) <= thresh)
  elif empty_side == 'upper':
    inds = np.triu(np.abs(T) <= thresh)
  elif empty_side == 'both':
    raise NotImplementedError('both side empty not supported yet.')
  else:
    raise ValueError('empty_side must be lower or upper or both.')

  blks_locations = np.nonzero(~np.logical_or(inds, inds.T))[0]
  unique_elems, unique_counts = np.unique(blks_locations, return_counts=True)
  blks, blks_inds, curr_ind = [], [], 0
  while curr_ind < len(unique_counts):
    blks.append(unique_counts[curr_ind])
    blks_inds.append([unique_elems[curr_ind],
                      unique_elems[curr_ind] + unique_counts[curr_ind]])
    curr_ind += unique_counts[curr_ind]
  return blks, np.array(blks_inds)

def make_symmetric(A):
  """Compute the symmetric part of A."""
  return (A + A.T) / 2

def inverse(A, left=True):
  """Compute the pseudoinverse of A (left or right)."""
  if left:
    return np.linalg.pinv(A)
  return A.T @ np.linalg.pinv(A @ A.T)

def is_PSD(A, tol=1e-8):
  """True if positive semi definite."""
  # https://stackoverflow.com/questions/5563743/check-for-positive-definiteness-or-positive-semidefiniteness
  if is_PD(A): return True
  eigs = np.linalg.eigvalsh(A)
  return np.all(eigs > -tol)

def is_PD(A):
  """True if positive definite."""
  # https://stackoverflow.com/questions/5563743/check-for-positive-definiteness-or-positive-semidefiniteness
  try:
    np.linalg.cholesky(A)
    return True
  except np.linalg.LinAlgError as e:
    return False

def make_PD(A, threshold_min_eigs=False, tol=sys.float_info.epsilon):
  """Make a given matrix positive definite by thresholding the eigenvalues."""
  e, u = np.linalg.eig(A)
  if threshold_min_eigs:
    e[e <= tol] = tol
    return u @ np.diag(e) @ np.conjugate(u).T

  inds = e > 0
  if np.size(inds) != np.size(e):
    print('Warning: best we can do is PSD not PD without thresholding.')
  return u[:, inds] @ np.diag(e[inds]) @ np.conjugate(u[:, inds]).T

def make_PSD(A, threshold_min_eigs=False, tol=sys.float_info.epsilon):
  """Make a given matrix positive semi-definite by thresholding the eigenvalues."""
  e, u = np.linalg.eig(A)
  if threshold_min_eigs:
    e[e < tol] = tol
    return u @ np.diag(e) @ np.conjugate(u).T

  inds = e >= 0
  return u[:, inds] @ np.diag(e[inds]) @ np.conjugate(u[:, inds]).T

def generate_PD_matrix(dim):
  """Generate a random positive definite matrix."""
  while True: # Ensure PD.
    tmp = np.random.randn(dim, dim)
    tmp = tmp @ tmp.T
    if is_PD(tmp): return tmp

def coloring_matrix(target_covariance):
  """Compute the coloring matrix for the given target covariance matrix."""
  if not is_PD(target_covariance):
    eigvals, V = np.linalg.eig(target_covariance)
    if np.any(np.iscomplex(eigvals)) and ~np.all(np.imag(eigvals) < 1e-9):
      raise ValueError('Provided target covariance has complex eigenvalues.')

    if not is_PSD(target_covariance) and np.any(np.abs(eigvals[eigvals < 0]) > 1e-9):
      raise ValueError('Warning: Target noise covariance matrix is not PSD: ', eigvals)
    else:
      eigvals[eigvals < 0] = np.finfo(np.float32).eps # Set to a small value close to zero.
      print('Warning: Target noise covariance matrix is not PD: ', eigvals)

    # Due to floating point precision even PSD covariance matrices can have
    # complex eigenvalues with the complex part very small in magnitude. Just in
    # case this is the situation, we take np.real().
    coloring = np.real(np.matmul(V, np.sqrt(np.diag(eigvals))))
  else: # is positive definite
    coloring = np.linalg.cholesky(target_covariance)
  return coloring

def whitening_matrix(covariance):
  """Compute the whitening matrix for the given covariance matrix."""
  return inverse(coloring_matrix(covariance), left=True)
