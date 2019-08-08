# Licensed under the  BSD 3-clause license - see LICENSE


from ..toa import get_TOAs
from ..models import get_model


__all__ = ['add_param_via_line', 'del_param']


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
