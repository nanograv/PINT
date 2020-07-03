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
        The start and end follows the python slice convention (i.e.,
        end = size + 1).
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

    def get_axis_labels(self, axis):
        return self.axis_labels[axis]

    def get_label(self, label):
        """ Get the label entry and its dimension. We assume the labels are
        unique in the matrix.
        """
        for ii, dim in enumerate(self.axis_labels):
            if label in dim.keys():
                return(label, ii, dim[label]))
        raise KeyError("Label {} is not in the matrix".format(label))

    def get_label_slice(self, labels):
        """ Return the given lable slices.
        """
        dim_slices = dict([(d, slice(None) for d in range(self.ndim))])
        new_labels =  dict([(d, {} for d in range(self.ndim))])
        for lb in labels:
            lable_info = self.get_label(lb)
            label_size = label_info[2][1] - label_info[2][0]
            # if slice is a list, there is a label already added.
            if isinstance(dim_slices[label_info[0]], list):
                # The start of the new matrix.
                start = len(dim_slices[label_info[0]]) + 1
                dim_slices[label_info[0]] += range(label_info[2][0],
                                                   label_info[2][1])

            else:
                start = 0
                dim_slices[label_info[0]] = range(label_info[2][0],
                                                  label_info[2][1])

            new_labels[label_info[0]].update({lb: (start, start + label_size)})
        return list(dim_slices.values()), list(new_labels.values())

    def get_label_matrix(self, labels):
        """ Get a sub-matrix data according to the given labels.
        """
        slice, new_labels = self.get_label_slice(labels)
        return PintMatrix(self.data[slice], new_labels)

    def match_labels_along_axis(self, pint_matrix, axis):
        """ Match one axis' labels index between the current matrix and input
        pint matrix. The labels will be matched along axises, not cross the
        axises.

        Parametere
        ----------
        pint_matrix: `PintMatrix` object or its sub-classes.
            The input pint matrix for label matching.

        Return
        ------
            Index map between the current labels and input matrix labels along
            axis.
        """
        current_lalels = self.get_axis_labels(axis)
        input_labels = pint_matrix.get_axis_labels(axis)
        matched = list(set(current_lables).instersection(set(input_labels)))
        match_index = {}
        for lb in matched:
            l1, ax1, idx1 = self.get_label(lb)
            l2, ax2, idx2 = pint_matrix.get_label(lb)
            match_index[lb] = [idx1, idx2]
        return match_index

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
        True.
    incoffset: bool, optional
        If add the absolute phase offset. Default is True.

    Note:
    -----
        The incoffset maybe replaced by the absolute phase fitting in the future.
    """
    def __init__(self, data, model, derivative_quantity, quantity_unit,
                 derivative_param, scaled_by_F0=True, incoffset=True):
        # Check if the derivate quantity a phase derivative
        if derivative_quantity = 'phase':
            self.params = ["Offset"] if incoffset else []
            self.params += derivative_param
        else:
            self.params =  derivative_param

        self.data = data
        self.model = model
        self.derivative_quantity = derivative_quantity
        self.quantity_unit = quantity_unit
        # Searching for the derivative functions. The derivative function should
        # be a wrapper function like d_phase_d_param()
        self.deriv_func = getattr(self.model,
            'd_{}_d_param'.format(derivative_quantity))
        self.units = []

        super(DesignMatrix, self).__init__(toa, model, derivative_quantity)

    def make_design_matrix(self,):
        """ Create the design matrix from the derivatives.
        """
        M = np.zeros((ntoas, nparams))
        for ii, param in enumerate(params):
            if param == "Offset":
                M[:, ii] = 1.0
                self.units.append(u.s / u.s)
            else:
                q = self.d_phase_d_param(toas, delay, param)
                if self.derivative_quantity == 'Phase'
                # NOTE Here we have negative sign here. Since in pulsar timing
                # the residuals are calculated as (Phase - int(Phase)), which is different
                # from the conventional definition of least square definition (Data - model)
                # We decide to add minus sign here in the design matrix, so the fitter
                # keeps the conventional way.

                    M[:, ii] = -q
                else:
                    M[:, ii] = q
                self.units.append(self.quantity_unit / getattr(self, param).units)

        if self.derivative_quantity == 'phase':
            if scale_by_F0:
                mask = []
                for ii, un in enumerate(units):
                    if params[ii] == "Offset":
                        continue
                    units[ii] = un * u.second
                    mask.append(ii)
                M[:, mask] /= F0.value
        return M, params, units, scale_by_F0


class CovarianceMatrix(PintMatrix):
    def __init__(self, toas, model):
        super(CovarianceMatrix, self).__init__(toa, model)
