# Licensed under the  BSD 3-clause license - see LICENSE

from ..toa import get_TOAs
from ..models import get_model
from .. import fitter as pint_fitter
from .statistics import Ftest
from copy import deepcopy


__all__ = ['ModelTester', 'add_param_via_line', 'del_param']


class ModelTester:
    """This class is designed for running f-test on timing model parameters.

    The f-test quantitatively reports the significance of the parameters to the
    input data by adding or removing them.

    Parameter
    ---------
    toas: `pint.toa.TOAs` object
        Input toas(data).
    input_model: `pint.model.TimingModel` object
        Timing model needs to be tested.
    fitter: str or callable, optional
        The fitting algorithm. Should be one of
            - 'WlsFitter'
            - 'GLSFitter'
    niter: int
        The number of fit iteration.
    """
    def __init__(self, toas, input_model, fitter='WlsFitter', niter=1):
        self.toas = toas
        self.input_model = input_model
        if isinstance(fitter, str):
            self.fitter_cls = getattr(pint_fitter, fitter)
        else:
            self.fitter_cls = fitter
        self.init_fit = self.fitter_cls(self.toas, self.input_model)
        self.niter = niter
        self.init_chi2 = self.init_fit.fit_toas(self.niter).value
        self.init_dof = self.init_fit.resids.dof
        self.model_stack = []
        self.model = deepcopy(self.input_model)

    def save_changes(self):
        # TODO add check if the model is saved
        self.model_stack.append(deepcopy(self.model))

    def undo_changes(self):
        self.model = self.model_stack.pop(-1)

    def add_params(self, params):
        """Add parameters to the current timing model.

        Parameter
        ---------
        params: dict
            The parameters need to be added.
                {parameter_name: parameter default value}
        """
        # form the parameter line
        param_lines = []
        for pname, pvalue in params.items():
            param_lines.append('{}    {}   1'.format(pname, pvalue))

        new_model = add_param_via_line(self.model, param_lines)
        self.save_changes()
        self.model = new_model

    def del_params(self, params):
        """Remove a parameter from the current timing model.

        Parameter
        ---------
        params: list
            List of removing parameters.
        """
        new_model = del_param(self.model, params)
        self.save_changes()
        self.model = new_model

    def compare_model(self):
        """Compare the current model with the input model.
        """
        new_fit = self.fitter_cls(self.toas, self.model)
        new_chi2 = new_fit.fit_toas(self.niter).value
        new_dof = new_fit.resids.dof
        f_result = Ftest(self.init_chi2, self.init_dof, new_chi2, new_dof)
        return (f_result, new_chi2, new_dof)


def add_param_via_line(model, par_lines):
    """Add parameters from parameter lines.

    Parameter
    ---------
    model: `TimingModel` object.
        Input model.
    par_lines: str or list.
        Paremeter lines for adding to the input model.

    Return
    ------
    New `TimingModel` object with added parameters.
    """
    if isinstance(par_lines, str):
        par_lines = [par_lines,]
    par_strings = model.as_parfile().split('\n')
    for par_line in par_lines:
        par_strings.append(par_line)
    new_model = get_model(par_strings)
    return new_model


def del_param(model, params):
    """Delete parameters from the input model.

    Parameter
    ---------
    model: `TimingModel` object.
        Input model.
    params: str or list
        Parameter names.
    """
    new_lines = []
    if isinstance(params, str):
        params = [params,]
    par_strings = model.as_parfile().split('\n')
    for par_line in par_strings:
        line_field = (str(par_line).strip()).split()
        if len(line_field) >= 1 and line_field[0] not in params:
            new_lines.append(par_line)
    return get_model(new_lines)
