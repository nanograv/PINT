""" pint_matrix module defines the pint matrix base class, the design matrix .
and the covariance matrix
"""

import numpy as np

__all__ = ["PintMatrix", "DesignMatrix", "CovarianceMatrix"]


class PintMatrixBase:
    """ PINT matrix is a base class for PINT fitters matrix.

    Parameters
    ----------
    data: `numpy.ndarray`
        Matrix data.
    axis_labels: list of dictionary
        The labels of the axises. Each list element contains the names and
        indices of the labels for the dimension.
        [{dim0_label0: (start, end), dim0_label1:(start, end)}, {dim1_lable0:.}]
    """
    def __init__(self, data, axis_labels):
        self.data = data
        self.axis_labels = axis_labels
        # Check dimensions
        if len(axis_label) != self.data.ndim:
            raise ValueError("Axis label dimension does not match the data "
                             "dimension.")

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def labels(self):
        labels = []
        for dim in self.axis_labels:
            labels.append(dim.keys())
        return labels

    def get_label_slice(self, label_names):
        """ Return the given lable slices.
        """
        pass

    def labeled_data(self, label_names):
        """ Return the data by given labels.
        """
        pass



    def match_labels(self, pint_matrix):
        """ Match the label index between the current matrix and input pint
        matrix.

        Parametere
        ----------
        pint_matrix: `PintMatrixBase` object or its sub-classes.
            The input pint matrix for label matching.
        """


    def append_x(self, pint_matrix):
        """ Append one pint matrix on x axis.
        """
        # Check shape and resize data
        new_shape = (max(self.shape[0], pint_matrix.shape[0]),
                     max(self.shape[1], pint_matrix.shape[1]))
        # Match the input matrix y axis to the current y axis
        for y in self.y_axis:
            if y in pint_matrix.y_axis:
                input_idx = pint_matrix.y_axis.index(y)


    def append_y(self, pint_matrix):
        """ Append one pint matrix on the y axis.
        """


class DesignMatrix(PintMatrixBase):
    """ Design matrix for least square fitting.

    Parameters
    ----------
    data: `pint.toa.TOAs` object or other data object
        The data point where the derivatives are evaluated.
    model: `pint.models.TimingModel` object
        The model that provides the derivatives.
    derivative_quantity: str
        The differentiated quantity name. It will be used to search for the
        derivative functions. For instance, if derivative_quantity = 'phase',
        it will search for the 'd_phase_d_param' function in the model.
    derivative_param: list
        The parameter list for the derivatives 'd_quantity_d_param'
    scale_by_F0: bool, optional
        If it is a phase derivative, if it is scaled by spin period. Default is
        False.
    incoffset: bool, optional
        If add the absolute phase offset. Default is False

    Note:
    -----
        The incoffset maybe replaced by the absolute phase fitting in the future.
    """
    def __init__(self, data, model, derivative_quantity, derivative_param,
                 scaled_by_F0=False, incoffset=False):
        # Check if the derivate quantity a phase derivative


        super(DesignMatrix, self).__init__(toa, model, derivative_quantity)


class CovarianceMatrix(PintMatrix):
    def __init__(self, toas, model):
        super(CovarianceMatrix, self).__init__(toa, model)
