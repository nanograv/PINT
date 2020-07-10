""" pint_matrix module defines the pint matrix base class, the design matrix .
and the covariance matrix
"""

import numpy as np
from itertools import combinations
import astropy.units as u


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
        [{dim0_label0: (start, end, unit), dim0_label1:(start, end, unit)},
         {dim1_label0:...}]
        The start and end follows the python slice convention (i.e.,
        end = size + 1).

    Note
    ----
    TODO: 1. add index to label mapping
    """
    def __init__(self, matrix, axis_labels):
        self.matrix = matrix
        self.axis_labels = axis_labels
        # Check dimensions
        if len(axis_labels) != self.matrix.ndim:
            raise ValueError("Axis label dimension does not match the matrix "
                             "dimension.")

        # Check label index overlap TODO: should we allow overlap?
        self._check_index_overlap()

    @property
    def ndim(self):
        return self.matrix.ndim

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def labels(self):
        labels = []
        for dim in range(len(self.axis_labels)):
            labels.append(self.get_axis_labels(dim))
        return labels

    @property
    def label_units(self):
        units = []
        for dim in range(len(self.axis_labels)):
            units.append(self.get_axis_labels(dim))
        return units


    def _check_index_overlap(self):
        for ii in range(self.ndim):
            axis_labels = self.get_axis_labels(ii)
            comb = combinations(axis_labels, 2)
            for cb in comb:
                if cb[0][1][0] <= cb[1][1][1] and cb[1][1][0] <= cb[0][1][1] -1:
                    raise ValueError("Label index in dim {} has"
                                     " overlap".format(ii))

    def _get_label_start(self, label_entry):
        return label_entry[1][0]

    def get_axis_labels(self, axis):
        dim_label = list(self.axis_labels[axis].items())
        dim_label.sort(key=self._get_label_start)
        return dim_label

    def get_label(self, label):
        """ Get the label entry and its dimension. We assume the labels are
        unique in the matrix.
        """
        for ii, dim in enumerate(self.axis_labels):
            if label in dim.keys():
                return(label, ii, dim[label])
        raise KeyError("Label {} is not in the matrix".format(label))

    def get_label_slice(self, labels):
        """ Return the given label slices.
        """
        dim_slices = dict([(d, slice(None)) for d in range(self.ndim)])
        new_labels =  dict([(d, {}) for d in range(self.ndim)])
        for lb in labels:
            label_info = self.get_label(lb)
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
        return PintMatrix(self.matrix[slice], new_labels)

    def match_labels_along_axis(self, pint_matrix, axis):
        """ Match one axis' labels index between the current matrix and input
        pint matrix. The labels will be matched along axises, not cross the
        axises.

        Parametere
        ----------
        pint_matrix: `PintMatrix` object or its sub-classes.
            The input pint matrix for label matching.
        axis: int
            The matching axis.

        Return
        ------
            Index map between the current labels and input matrix labels along
            axis.
        """
        current_labels = self.get_axis_labels(axis)
        input_labels = pint_matrix.get_axis_labels(axis)
        curr_label_name = [cl[0] for cl in current_labels]
        input_label_name =  [il[0] for il in input_labels]
        matched = list(set(curr_label_name).intersection(set(input_label_name)))
        print(matched)
        match_index = {}
        for lb in matched:
            l1, ax1, idx1 = self.get_label(lb)
            l2, ax2, idx2 = pint_matrix.get_label(lb)
            match_index[lb] = [idx1, idx2]
        return match_index

    def map_labels(self):
        raise NotImplementedError()

    def append_along_axis(self, pint_matrix, axis):
        """ Append one pint matrix on a given axis.
        """
        raise NotImplementedError()


class DesignMatrix(PintMatrix):
    """ A generic design matrix class for least square fitting.

    Parameters
    ----------
    matrix: `numpy.ndarray`
        Design matrix values.
    axis_labels: list of dictionary
        The labels of the axises. Each list element contains the names and
        indices of the labels for the dimension.
        [{dim0_label0: (start, end, unit), dim0_label1:(start, end, unit)},
         {dim1_label0:...}]
        The start and end follows the python slice convention (i.e.,
        end = size + 1).

    Note
    ----
    TODO: 1. add index to unit mapping.

    """
    def __init__(self, matrix, labels):
        super(DesignMatrix, self).__init__(matrix, labels)
        self.scaled_by_F0 = False

    @property
    def param_units(self):
        param_lb = self.get_axis_labels(1)
        return [lb[1][2] for lb in param_lb]

    @property
    def derivative_quantity(self):
        param_lb = self.get_axis_labels(0)
        return [lb[1][2] for lb in param_lb]

    @property
    def derivative_params(self):
        param_lb = self.get_axis_labels(1)
        return [lb[0] for lb in param_lb]


class DesignMatrixMaker:
    """ Class for pint design matrix maker class.

    Parameters
    ----------
    derivative_quantity: str
        The differentiated quantity name. It will be used to search for the
        derivative functions. For instance, if derivative_quantity = 'phase',
        it will search for the 'd_phase_d_param' function in the model.
    quantity_unit: `astropy.units.unit` object
        The unit of the derivative quantity.
    """

    def __new__(cls, derivative_quantity, quantity_unit):
        target_cls = design_matrix_maker_map.get(derivative_quantity.lower(),
                                                 None)
        # If there is no matching maker, use the current one.
        if target_cls is not None:
            cls = target_cls
        print(cls)
        return super().__new__(cls)

    def __init__(self, derivative_quantity, quantity_unit):
        self.derivative_quantity = derivative_quantity
        self.quantity_unit = quantity_unit
        # The derivative function should be a wrapper function like d_phase_d_param()
        self.deriv_func_name = 'd_{}_d_param'.format(self.derivative_quantity)

    def __call__(self, data, model, derivative_params, offset=False,
                 offset_padding=0.0):
        """ A general method to make design matrix.

        Parameters
        ----------
        data: `pint.toa.TOAs` object or other data object
            The data point where the derivatives are evaluated.
        model: `pint.models.TimingModel` object
            The model that provides the derivatives.
        derivative_params: list
            The parameter list for the derivatives 'd_quantity_d_param'.
        offset: bool, optional
            Add the an offset to the beginning of design matrix. Default is False.
            This is match the current phase offset in the design matrix.
        offset_padding: float, optional
            if including offset, the value for padding.
        """
        # Get derivative functions
        deriv_func = getattr(model, self.deriv_func_name)
        # Check if the derivate quantity a phase derivative
        params = ["Offset"] if offset else []
        params += derivative_params
        labels = []
        M = np.zeros((len(data), len(params)))
        labels.append({self.derivative_quantity: (0, M.shape[0],
                                                  self.quantity_unit)})
        labels_dim2 = {}
        for ii, param in enumerate(params):
            if param == "Offset":
                M[:, ii] = offset_padding
                param_unit = u.Unit("")
            else:
                param_unit = getattr(model,param).units
                q = deriv_func(data, param).to(
                    self.quantity_unit / param_unit
                    )
                # This will strip the units
                M[:, ii] = q
            labels_dim2[param] = (ii, ii + 1, param_unit)

        labels.append(labels_dim2)
        return DesignMatrix(M, labels)


class PhaseDesignMatrixMaker(DesignMatrixMaker):
    """ A specific class for makeing phase design matrix.
    """
    def __call__(self, data, model, derivative_params,
                 scaled_by_F0=True, offset=True,
                 offset_padding=1.0):
        """ Create the phase design matrix.

        Parameters
        ----------
        data: `pint.toa.TOAs` object or other data object
            The data point where the derivatives are evaluated.
        model: `pint.models.TimingModel` object
            The model that provides the derivatives.
        derivative_params: list
            The parameter list for the derivatives 'd_quantity_d_param'.
        scale_by_F0: bool, optional
            Flag for scaling the matrxi by spin rate. Default is True
        offset: bool, optional
            Add the an offset to the beginning of design matrix. Default is True.
        offset_padding: float, optional
            if including offset, the value for padding. Default is 1.0
        """
        deriv_func = getattr(model, self.deriv_func_name)
        # Check if the derivate quantity a phase derivative
        params = ["Offset"] if offset else []
        params += derivative_params
        labels = []
        M = np.zeros((data.ntoas, len(params)))
        labels.append({self.derivative_quantity: (0, M.shape[0],
            self.quantity_unit)})
        labels_dim2 = {}
        delay = model.delay(data)
        for ii, param in enumerate(params):
            if param == "Offset":
                M[:, ii] = offset_padding
                param_unit = u.Unit("")
            else:
                param_unit = getattr(model, param).units
                q = deriv_func(data, delay, param).to(
                    self.quantity_unit / param_unit
                        )


        # NOTE Here we have negative sign here. Since in pulsar timing
        # the residuals are calculated as (Phase - int(Phase)), which is different
        # from the conventional definition of least square definition (Data - model)
        # We decide to add minus sign here in the design matrix, so the fitter
        # keeps the conventional way.
                M[:, ii] = -q
            labels_dim2[param] = (ii, ii + 1, param_unit)

        labels.append(labels_dim2)

        if scaled_by_F0:
            mask = []
            for ii, param in enumerate(params):
                if param == "Offset":
                    continue
                mask.append(ii)
            M[:, mask] /= model.F0.value
            # TODO maybe use defined label is better
            labels[0] = {self.derivative_quantity:
                (0, M.shape[0], self.quantity_unit * u.s)}
        d_matrix = DesignMatrix(M, labels)
        d_matrix.scaled_by_F0 = scaled_by_F0
        return d_matrix


class NoiseDesignMatrixMaker(DesignMatrixMaker):
    """ Specific design matrix for noise model
    """
    def __call__(self, data, model, derivative_params,
                 scaled_by_F0=True, offset=True,
                 offset_padding=0.0):
        pass





design_matrix_maker_map = {'phase': PhaseDesignMatrixMaker}


def combine_design_matrixs(design_matrixs):
    """ A fast method to combine two design matrix along the derivative
    quantity. If requires the parameter list match to each other.

    Parameter
    ---------
    design_matrixs: `pint_matrix.DesignMatrix` object
        The input design matrix.

    """
    axis_labels = [{}, design_matrixs[0].axis_labels[1]]
    all_matrix = []
    for ii, d_matrix in enumerate(design_matrixs):
        if d_matrix.derivative_params != design_matrixs[0].derivative_params:
            raise ValueError("Input design matrix's derivative parameters do "
                             "not match the current derivative parameters.")
        # only update the derivative quantity label.
        if ii == 0:
            axis_labels[0].update(d_matrix.axis_labels[0])
            # Set the start index of next label, which is the current ending index
            offset = d_matrix.get_axis_labels(0)[-1][1][1]
        else:
            new_labels = []
            old_labels = d_matrix.get_axis_labels(0)
            for olb in old_labels:
                # apply offset to the next label.
                new_labels.append((olb[0], (olb[1][0] + offset,
                                            olb[1][1] + offset)))
                off_set = new_labels[-1][1][1]
            axis_labels[0].update(dict(new_labels))
        all_matrix.append(d_matrix.matrix)
    result = DesignMatrix(np.vstack(all_matrix), axis_labels)
    return result


class CovarianceMatrix(PintMatrix):
    def __init__(self, toas, model):
        pass
