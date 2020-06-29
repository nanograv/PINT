""" pint_matrix module defines the pint matrix base class, the design matrix .
and the covariance matrix
"""

import numpy as np
from itertools import combinations


__all__ = ["PintMatrix", "DesignMatrix", "CovarianceMatrix"]


class PintMatrix:
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

        # Check label index overlap TODO: should we allow overlap?
        self._check_index_overlap()

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

    def _check_index_overlap(self):
        for ii, dim in enumerate(self.axis_labels):
            comb = combinations(list(dim.values), 2)
            for cb in comb:
                if cb[0][0] <= cb[1][1] and cb[1][0] <= cb[0][1]:
                    raise ValueError("Label index in dim {} has"
                                     " overlap".format(ii))

    def _get_label(self, lable):
        """ Get the label entry and its dimension. We assume the labels are
        unique in the matrix.
        """
        for ii, dim in enumerate(self.axis_labels):
            if label in dim.keys():
                return(ii, label, dim[label]))
        raise KeyError("Label {} is not in the matrix".format(label))

    def get_label_slice(self, labels):
        """ Return the given lable slices.
        """
        dim_slices = dict([(d, slice(None) for d in range(self.ndim))])
        new_labels =  dict([(d, {} for d in range(self.ndim))])
        for lb in labels:
            lable_info = self._get_label(lb)
            label_size = label_info[2][1] - label_info[2][0] + 1
            # if slice is a list, there is a label already added.
            if isinstance(dim_slices[label_info[0]], list):
                # The start of the new matrix.
                start = len(dim_slices[label_info[0]]) + 1
                dim_slices[label_info[0]] += range(label_info[2][0],
                                                   label_info[2][1] + 1)

            else:
                start = 0
                dim_slices[label_info[0]] = range(label_info[2][0],
                                                  label_info[2][1] + 1)

            new_labels[label_info[0]].update({lb: (start, start + label_size)})
        return list(dim_slices.values()), list(new_labels.values())

    def get_label_matrix(self, labels):
        """ Get a sub-matrix data according to the given labels.
        """
        slice, new_labels = self.get_label_slice(labels)
        return PintMatrix(self.data[slice], new_labels)

    def match_labels(self, pint_matrix):
        """ Match the label index between the current matrix and input pint
        matrix. The labels will be matched along axises, not cross the axises.

        Parametere
        ----------
        pint_matrix: `PintMatrix` object or its sub-classes.
            The input pint matrix for label matching.

        Return
        ------
            Index map between the current labels and input matrix labels.
        """
        pass

    def append_along_axis(self, pint_matrix, axis):
        """ Append one pint matrix on x axis.
        """
        pass

class DesignMatrix(PintMatrix):
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
