# Licensed under the  BSD 3-clause license - see LICENSE


from ..toa import get_TOAs
from ..models import get_model

__all__ = ['add_param_from_str']


def add_param_via_line(model, par_lines):
    if isinstance(par_lines, str):
        par_lines = [par_lines,]
    par_strings = model.as_parfile().split('\n')
    for par_line in par_lines:
        par_strings.append(par_line)
    new_model = get_model(par_strings)
    return new_model


def del_param(model, params):
    new_lines = []
    if isinstance(params, str):
        params = [params,]
    par_strings = model.as_parfile().split('\n')
    for par_line in par_strings:
        line_field = (str(par_line).strip()).split()
        if len(line_field) >= 1 and line_field[0] not in params:
            new_lines.append(par_line)
    return get_model(new_lines)
