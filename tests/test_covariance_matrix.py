""" Various of tests for the pint covariance."""

import pytest
import os

import numpy as np
import astropy.units as u
from pint.pint_matrix import CovarianceMatrix, combine_covariance_matrix

from pinttestdata import datadir

os.chdir(datadir)


class TestCovarianceMatrix:
    """Test for covariance matrix"""

    def setup_method(self):
        self.matrix1 = np.arange(16).reshape((4, 4))
        self.label1 = [{"c": (0, 4, u.s)}] * 2
        self.matrix2 = np.arange(9).reshape((3, 3))
        self.label2 = [{"b": (0, 3, u.m)}] * 2
        self.matrix3 = np.arange(25).reshape((5, 5))
        self.label3 = [{"a": (0, 5, u.kg)}] * 2

    def test_covariance_matrix_init(self):
        cm = CovarianceMatrix(self.matrix1, self.label1)
        assert cm.shape == (4, 4)
        with pytest.raises(ValueError):
            cm = CovarianceMatrix(self.matrix1.reshape((2, 8)), self.label1)
        with pytest.raises(ValueError):
            test_wrong_label = self.label1
            test_wrong_label[1] = {"c2": (0, 4, u.s), "c3": (0, 5, u.m)}
            cm = CovarianceMatrix(self.matrix1, test_wrong_label)

    def test_matrix_combine_two(self):
        cm1 = CovarianceMatrix(self.matrix1, self.label1)
        cm2 = CovarianceMatrix(self.matrix2, self.label2)

        combine_cm = combine_covariance_matrix([cm1, cm2])
        assert combine_cm.shape == ((7, 7))
        assert combine_cm.labels[0] == combine_cm.labels[1]
        assert combine_cm.labels[0] == [("c", (0, 4, u.s)), ("b", (4, 7, u.m))]
        assert np.all(combine_cm.matrix[0:4, 0:4] == self.matrix1)
        assert np.all(combine_cm.matrix[4:7, 4:7] == self.matrix2)
        assert np.all(combine_cm.matrix[4:7, 0:4] == np.zeros((3, 4)))
        assert np.all(combine_cm.matrix[0:4, 4:7] == np.zeros((4, 3)))

    def test_matrix_combine_two_other_padding(self):
        cm1 = CovarianceMatrix(self.matrix1, self.label1)
        cm2 = CovarianceMatrix(self.matrix2, self.label2)

        combine_cm = combine_covariance_matrix([cm1, cm2], crossterm_padding=1.5)
        assert np.all(combine_cm.matrix[0:4, 0:4] == self.matrix1)
        assert np.all(combine_cm.matrix[4:7, 4:7] == self.matrix2)
        t1 = np.zeros((3, 4))
        t2 = np.zeros((4, 3))
        t1.fill(1.5)
        t2.fill(1.5)
        assert np.all(combine_cm.matrix[4:7, 0:4] == t1)
        assert np.all(combine_cm.matrix[0:4, 4:7] == t2)

    def test_combine_three(self):
        cm1 = CovarianceMatrix(self.matrix1, self.label1)
        cm2 = CovarianceMatrix(self.matrix2, self.label2)
        cm3 = CovarianceMatrix(self.matrix3, self.label3)
        combine_cm = combine_covariance_matrix([cm1, cm2, cm3])
        assert combine_cm.shape == ((12, 12))
        assert combine_cm.labels[0] == combine_cm.labels[1]
        assert combine_cm.labels[0] == [
            ("c", (0, 4, u.s)),
            ("b", (4, 7, u.m)),
            ("a", (7, 12, u.kg)),
        ]
        assert np.all(combine_cm.matrix[0:4, 0:4] == self.matrix1)
        assert np.all(combine_cm.matrix[4:7, 4:7] == self.matrix2)
        assert np.all(combine_cm.matrix[7:12, 7:12] == self.matrix3)
        assert np.all(combine_cm.matrix[4:7, 0:4] == np.zeros((3, 4)))
        assert np.all(combine_cm.matrix[0:4, 4:7] == np.zeros((4, 3)))
        assert np.all(combine_cm.matrix[7:12, 4:7] == np.zeros((5, 3)))
        assert np.all(combine_cm.matrix[7:12, 0:4] == np.zeros((5, 4)))
