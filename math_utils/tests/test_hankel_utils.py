"""Unit tests for hankel_utils module."""

import math_utils.hankel_utils as hankel_utils
import numpy as np
from nose.tools import with_setup

_SEED = 900

def test_make_hankel():
  Y = np.array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
                [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]])
  Y_hank = hankel_utils.make_hankel(Y, 5, 5)
  expected_hankel = np.array([[ 0,  2,  4,  6,  8 ],
                              [ 1,  3,  5,  7,  9 ],
                              [ 2,  4,  6,  8,  10 ],
                              [ 3,  5,  7,  9,  11 ],
                              [ 4,  6,  8,  10, 12 ],
                              [ 5,  7,  9,  11, 13 ],
                              [ 6,  8,  10, 12, 14 ],
                              [ 7,  9,  11, 13, 15 ],
                              [ 8,  10, 12, 14, 16 ],
                              [ 9,  11, 13, 15, 17 ]])
  assert np.array_equal(Y_hank, expected_hankel)

def test_make_hankel_with_start():
  Y = np.array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
                [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]])
  Y_hank = hankel_utils.make_hankel(Y, 2, 7, start=2)
  expected_hankel = np.array([[ 4,  6,  8,  10, 12, 14, 16 ],
                              [ 5,  7,  9,  11, 13, 15, 17 ],
                              [ 6,  8,  10, 12, 14, 16, 18 ],
                              [ 7,  9,  11, 13, 15, 17, 19 ]])
  assert np.array_equal(Y_hank, expected_hankel)

def test_construct_future_past_stacked_hankel():
  Y = np.array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
                [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]])
  Z = np.array([[ 10,  12,  14,  16,  18, 20, 22, 24, 26, 28],
                [ 11,  13,  15,  17,  19, 21, 23, 25, 27, 29]])
  YZ_hank = hankel_utils.construct_future_past_stacked_hankel(Y, 2, observation_mat2=Z)
  expected_hankel = np.array([[ 4,  6,  8,  10, 12, 14, 16 ],
                              [ 5,  7,  9,  11, 13, 15, 17 ],
                              [ 6,  8,  10, 12, 14, 16, 18 ],
                              [ 7,  9,  11, 13, 15, 17, 19 ],
                              [ 10,  12,  14, 16, 18, 20, 22 ],
                              [ 11,  13,  15, 17, 19, 21, 23 ],
                              [ 12,  14,  16, 18, 20, 22, 24 ],
                              [ 13,  15,  17, 19, 21, 23, 25 ]])
  assert np.array_equal(YZ_hank, expected_hankel), 'expected\n{0}\nactual\n{1}'.format(YZ_hank, expected_hankel)

def test_construct_future_past_stacked_hankel_different_horizon():
  Y = np.array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
                [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]])
  Z = np.array([[ 10,  12,  14,  16,  18, 20, 22, 24, 26, 28],
                [ 11,  13,  15,  17,  19, 21, 23, 25, 27, 29]])
  YZ_hank = hankel_utils.construct_future_past_stacked_hankel(Y, 2, observation_mat2=Z, i2=3)
  expected_hankel = np.array([[ 4,  6,  8,  10, 12 ],
                              [ 5,  7,  9,  11, 13 ],
                              [ 6,  8,  10, 12, 14 ],
                              [ 7,  9,  11, 13, 15 ],
                              [ 10,  12,  14, 16, 18 ],
                              [ 11,  13,  15, 17, 19 ],
                              [ 12,  14,  16, 18, 20 ],
                              [ 13,  15,  17, 19, 21 ],
                              [ 14,  16,  18, 20, 22 ],
                              [ 15,  17,  19, 21, 23 ]])
  assert np.array_equal(YZ_hank, expected_hankel), 'expected\n{0}\nactual\n{1}'.format(YZ_hank, expected_hankel)

def test_compute_hankel_parameters():
  observations, horizon = 100, 5
  second_observations, second_horizon = 80, 10
  j = hankel_utils.compute_hankel_parameters(observations, horizon)
  j2 = hankel_utils.compute_hankel_parameters(observations, horizon,
                                               num_second_observations=second_observations)
  assert j2 < j
  assert j == 91
  assert j2 == 71
  j3 = hankel_utils.compute_hankel_parameters(observations, horizon,
                                               num_second_observations=second_observations,
                                               second_horizon=second_horizon)
  assert j3 < j2
  assert j3 == 61