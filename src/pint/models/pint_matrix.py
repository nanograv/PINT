""" pint_matrix module defines the pint matrix base class, the design matrix .
and the covariance matrix
"""

import numpy as np

__all__ = ["PintMatrix", "DesignMatrix", "CovarianceMatrix"]


class PintMatrix(np.ndarray):
    """ PINT matrix is a base class for PINT fitters matrix.
    """
    def __init__(self, shape, dtype):
        super(PintMartix, self).__init__(shape, dtype)

    def _hstack(self):
        pass


class DesignMatrix(PintMatrix):
    def __init__(self, toas, model):
        super(DesignMatrix, self).__init__(toa, model)


class CovarianceMatrix(PintMatrix):
    def __init__(self, toas, model):
        super(CovarianceMatrix, self).__init__(toa, model)
