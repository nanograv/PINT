"""Generate random models distributed like the results of a fit"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from astropy import log

import pint.toa as toa
from pint.phase import Phase

__all__ = ["random_models"]
# log.setLevel("INFO")


def random_models(
    fitter, rs_mean, ledge_multiplier=4, redge_multiplier=4, iter=1, npoints=100
):
    """Uses the covariance matrix to produce gaussian weighted random models.

    Returns fake toas for plotting and a list of the random models' phase resid objects.
    rs_mean determines where in residual phase the lines are plotted,
    edge_multipliers determine how far beyond the selected toas the random models are plotted.
    This uses an approximate method based on the cov matrix, it doesn't use MCMC.

    Parameters
    ----------
    fitter
        fitter object with model and toas to vary from
    rs_mean
        average phase residual for toas in fitter object, used to plot random models
    ledge_multiplier
        how far the lines will plot to the left in multiples of the fit toas span, default 4
    redge_multiplier
        how far the lines will plot to the right in multiples of the fit toas span, default 4
    iter
        how many random models will be computed, default 1
    npoints
        how many fake toas will be reated for the random lines, default 100

    Returns
    -------
        TOAs object containing the evenly spaced fake toas to plot the random lines with
        list of residual objects for the random models (one residual object each)
    """
    param_values = fitter.model.get_params_dict("free", "num")
    cov_matrix = fitter.parameter_covariance_matrix
    # this is a list of the parameter names in the order they appear in the coviarance matrix
    param_names = cov_matrix.get_label_names(axis=0)
    mean_vector = np.array([param_values[x] for x in param_names if not x == "Offset"])
    # remove the first column and row (absolute phase)
    if param_names[0] == "Offset":
        cov_matrix = cov_matrix.get_label_matrix(param_names[1:])
        fac = fitter.fac[1:]
        param_names = param_names[1:]
    else:
        fac = fitter.fac
    f_rand = deepcopy(fitter)
    mrand = f_rand.model

    # scale by fac
    mean_vector = mean_vector * fac
    scaled_cov_matrix = ((cov_matrix.matrix * fac).T * fac).T

    toa_mjds = fitter.toas.get_mjds()
    minMJD, maxMJD = toa_mjds.min(), toa_mjds.max()
    spanMJDs = maxMJD - minMJD
    # ledge and redge _multiplier control how far the fake toas extend
    # in either direction of the selected points
    x = toa.make_fake_toas(
        minMJD - spanMJDs * ledge_multiplier,
        maxMJD + spanMJDs * redge_multiplier,
        npoints,
        mrand,
    )
    x2 = toa.make_fake_toas(minMJD, maxMJD, npoints, mrand)

    rss = []
    random_models = []
    for i in range(iter):
        # create a set of randomized parameters based on mean vector and covariance matrix
        rparams_num = np.random.multivariate_normal(mean_vector, scaled_cov_matrix)
        # scale params back to real units
        for j in range(len(mean_vector)):
            rparams_num[j] /= fac[j]
        rparams = OrderedDict(zip(param_names, rparams_num))
        # print("randomized parameters",rparams)
        f_rand.set_params(rparams)
        rs = mrand.phase(x, abs_phase=True) - fitter.model.phase(x, abs_phase=True)
        rs2 = mrand.phase(x2, abs_phase=True) - fitter.model.phase(x2, abs_phase=True)
        # from calc_phase_resids in residuals
        rs -= Phase(0.0, rs2.frac.mean() - rs_mean)
        # TODO: use units here!
        rs = ((rs.int + rs.frac).value / fitter.model.F0.value) * 10 ** 6
        rss.append(rs)
        random_models.append(deepcopy(mrand))

    return x, rss, random_models
