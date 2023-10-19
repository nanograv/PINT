import astropy.units as u

import pint.models.parameter as pa


def num_diff_phase(toas, pint_param, model, h=None):
    par = getattr(model, pint_param)
    if h is not None:
        return model.d_phase_d_param_num(toas, pint_param, h)
    if isinstance(par, pa.MJDParameter) or par.units == u.day:
        h = 1e-8
    elif pint_param in ["SINI", "KIN"]:
        h = 1e-3
    elif pint_param in ["KIN"]:
        h = 1
    elif pint_param in ["F1", "M2", "PMELONG", "PMELAT"]:
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

    noder = list(
        (set(model.params).difference(set(d_der.keys()))).difference(set(p_der.keys()))
    )
    return d_der, p_der, noder


def get_derivative_params(model):
    test_params = {}
    for p in model.params:
        par = getattr(model, p)
        # remove all the JUMP and DMX right now.
        if par.frozen:
            continue
        if p.endswith("EPOCH"):
            continue
        if p.startswith("JUMP") and par.index != 1:
            continue
        if p.startswith("FD") and par.index != 2:
            continue
        if p.startswith("DMX"):
            if not p.startswith("DMX_"):
                continue
            elif par.index != 2:
                continue

        if isinstance(par, pa.MJDParameter) or par.units == u.day:
            h = 1e-8
        elif p in ["SINI"]:
            h = 3e-4
        elif p in ["KIN"]:
            h = 2.55e-2
        elif p in ["KOM"]:
            h = 2e-1
        elif p in ["PX"]:
            h = 2e-1
        elif p in ["FD2"]:
            h = 3e-2
        elif p in ["F1", "M2", "PMELONG", "PMELAT", "PMRA", "PMDEC"]:
            h = 2
        elif p in ["PBDOT"]:
            h = 200
        elif p in ["FB1"]:
            h = 2
        elif p in ["FB0"]:
            h = 1e-7
        elif p in ["FB2", "FB3"]:
            h = 1000
        else:
            h = 1e-2
        test_params[p] = h
    return test_params
