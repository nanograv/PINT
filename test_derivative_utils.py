import pint.models.parameter as p
import astropy.units as u
def num_diff_delay(toas, pint_param, model, h=None):
    par = getattr(model, pint_param)
    if h is not None:
        return model.d_delay_d_param_num(toas, pint_param, h)
    if isinstance(par, p.MJDParameter) or par.units == u.day:
        h = 1e-8
    elif pint_param == 'SINI':
        h = 1e-3
    else:
        h = 1e-2
    return model.d_delay_d_param_num(toas, pint_param, h)

def num_diff_phase(toas, pint_param, model, h=None):
    par = getattr(model, pint_param)
    if h is not None:
        return model.d_phase_d_param_num(toas, pint_param, h)
    if isinstance(par, p.MJDParameter) or par.units == u.day:
        h = 1e-8
    elif pint_param in ['SINI']:
        h = 1e-3
    elif pint_param in ['F1', 'M2', 'PMELONG', 'PMELAT']:
        h = 2
    else:
        h = 1e-2
    return model.d_phase_d_param_num(toas, pint_param, h)

def get_derivative_funcs(model):
    d_der = {}
    p_der = {}
    for p in model.params:
        for dd in model.delay_derivs:
            if dd.__name__.endswith(p):
                d_der[p] = dd
        for pd in model.phase_derivs:
            if pd.__name__.endswith(p):
                p_der[p] = pd

    noder = list((set(model.params).difference(set(d_der.keys()))).difference(set(p_der.keys())))
    return d_der, p_der, noder
