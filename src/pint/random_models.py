"""Generate random models distributed like the results of a fit."""

from collections import OrderedDict
from copy import deepcopy

import numpy as np
from loguru import logger as log

import pint.simulation as simulation
from pint.phase import Phase


__all__ = ["random_models"]


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
        how many fake toas will be related for the random lines, default 100

    Returns
    -------
        TOAs object containing the evenly spaced fake toas to plot the random lines with
        list of residual objects for the random models (one residual object each)
    """
    params = fitter.model.get_params_dict("free", "num")
    mean_vector = params.values()
    # remove the first column and row (absolute phase)
    cov_matrix = (((fitter.covariance_matrix.matrix[1:]).T)[1:]).T
    fac = fitter.fac[1:]
    f_rand = deepcopy(fitter)
    mrand = f_rand.model

    # scale by fac
    log.debug("errors", np.sqrt(np.diag(cov_matrix)))
    log.debug("mean vector", mean_vector)
    mean_vector = np.array(list(mean_vector)) * fac
    cov_matrix = ((cov_matrix * fac).T * fac).T

    toa_mjds = fitter.toas.get_mjds()
    minMJD, maxMJD = toa_mjds.min(), toa_mjds.max()
    spanMJDs = maxMJD - minMJD
    # ledge and redge _multiplier control how far the fake toas extend
    # in either direction of the selected points
    x = simulation.make_fake_toas_uniform(
        minMJD - spanMJDs * ledge_multiplier,
        maxMJD + spanMJDs * redge_multiplier,
        npoints,
        mrand,
    )
    x2 = simulation.make_fake_toas_uniform(minMJD, maxMJD, npoints, mrand)

    rss = []
    random_models = []
    for _ in range(iter):
        # create a set of randomized parameters based on mean vector and covariance matrix
        rparams_num = np.random.multivariate_normal(mean_vector, cov_matrix)
        # scale params back to real units
        for j in range(len(mean_vector)):
            rparams_num[j] /= fac[j]
        rparams = OrderedDict(zip(params.keys(), rparams_num))
        # print("randomized parameters",rparams)
        f_rand.set_params(rparams)
        rs = mrand.phase(x, abs_phase=True) - fitter.model.phase(x, abs_phase=True)
        rs2 = mrand.phase(x2, abs_phase=True) - fitter.model.phase(x2, abs_phase=True)
        # from calc_phase_resids in residuals
        rs -= Phase(0.0, rs2.frac.mean() - rs_mean)
        # TODO: use units here!
        rs = ((rs.int + rs.frac).value / fitter.model.F0.value) * 10**6
        rss.append(rs)
        random_models.append(deepcopy(mrand))

    return x, rss, random_models
