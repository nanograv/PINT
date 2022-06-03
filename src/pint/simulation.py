"""Functions related to simulating TOAs and models
"""
from collections import OrderedDict
from copy import deepcopy

import astropy.units as u
import numpy as np
from loguru import logger as log
from astropy import time

import pint.residuals
import pint.toa
from pint.observatory import Observatory, bipm_default, get_observatory

__all__ = [
    "zero_residuals",
    "make_fake_toas",
    "make_fake_toas_uniform",
    "make_fake_toas_fromMJDs",
    "make_fake_toas_fromtim",
    "calculate_random_models",
]


def _get_freq_array(base_frequencies, ntoas):
    """Make frequency array out of one or more frequencies

    If >1 frequency is specified, will alternate

    Parameters
    ----------
    base_frequencies : astropy.units.Quantity
       array of frequencies
    ntoas : int
       number of TOAs

    Returns
    -------
    astropy.units.Quantity
        array of (potentially alternating) frequencies
    """
    freq = np.zeros(ntoas) * base_frequencies[0].unit
    num_freqs = len(base_frequencies)
    for ii, fv in enumerate(base_frequencies):
        freq[ii::num_freqs] = fv
    return freq


def zero_residuals(ts, model, maxiter=10, tolerance=1 * u.ns):
    """Use a model to adjust a TOAs object, setting residuals to 0 iteratively.

    Parameters
    ----------
    ts : pint.toa.TOAs
        Input TOAs (modified in-place)
    model : pint.models.timing_model.TimingModel
        current model
    maxiter : int, optional
        maximum number of iterations allowed
    tolerance : astropy.units.Quantity
        maximum allowed absolute deviation of residuals from 0
    """
    ts.compute_pulse_numbers(model)
    maxresid = None
    for i in range(maxiter):
        r = pint.residuals.Residuals(ts, model, track_mode="use_pulse_numbers")
        resids = r.calc_time_resids(calctype="taylor")
        if maxresid is not None and (np.abs(resids).max() > maxresid):
            log.warning(
                f"Residual increasing at iteration {i} while attempting to simulate TOAs"
            )
        maxresid = np.abs(resids).max()
        if abs(resids).max() < tolerance:
            break
        ts.adjust_TOAs(-time.TimeDelta(resids))
    else:
        raise ValueError(
            "Unable to make fake residuals - left over errors are {}".format(
                abs(resids).max()
            )
        )


def update_fake_toa_clock(ts, model, include_bipm=False, include_gps=True):
    """Update the clock settings (corrections, etc) for fake TOAs

    Parameters
    ----------
    ts : pint.toa.TOAs
        Input TOAs (modified in-place)
    model : pint.models.timing_model.TimingModel
        current model
    include_bipm : bool, optional
        Whether or not to disable UTC-> TT BIPM clock
        correction (see :class:`pint.observatory.topo_obs.TopoObs`)
    include_gps : bool, optional
        Whether or not to disable UTC(GPS)->UTC clock correction
        (see :class:`pint.observatory.topo_obs.TopoObs`)
    """
    bipm_version = bipm_default
    if model["CLOCK"].value is not None:
        if model["CLOCK"].value == "TT(TAI)":
            include_bipm = False
            log.info("Using CLOCK = TT(TAI), so setting include_bipm = False")
        elif "BIPM" in model["CLOCK"].value:
            clk = model["CLOCK"].value.strip(")").split("(")
            if len(clk) == 2:
                ctype, cvers = clk
                if ctype == "TT" and cvers.startswith("BIPM"):
                    include_bipm = True
                    if bipm_version is None:
                        bipm_version = cvers
                        log.info(f"Using CLOCK = {bipm_version} from the given model")
                else:
                    log.warning(
                        f'CLOCK = {model["CLOCK"].value} is not implemented. '
                        f"Using TT({bipm_default}) instead."
                    )
        else:
            log.warning(
                f'CLOCK = {model["CLOCK"].value} is not implemented. '
                f"Using TT({bipm_default}) instead."
            )

    ts.clock_corr_info.update(
        {
            "include_bipm": include_bipm,
            "bipm_version": bipm_version,
            "include_gps": include_gps,
        }
    )


def make_fake_toas(ts, model, add_noise=False, name="fake"):
    """Make toas from an array of times

    Can include alternating frequencies if fed an array of frequencies,
    only works with one observatory at a time

    Parameters
    ----------
    ts : pint.toa.TOAs
        Input TOAs to match
    model : pint.models.timing_model.TimingModel
        current model
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    name : str, optional
        Name for the TOAs (goes into the flags)

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with toas matching toas but with residuals starting at zero (but then with optional noise)

    Notes
    -----
    `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`
    """
    tsim = deepcopy(ts)
    zero_residuals(tsim, model)
    if add_noise:
        # this function will include EFAC and EQUAD
        err = model.scaled_toa_uncertainty(tsim) * np.random.normal(size=len(tsim))
        # Add the actual TOA noise
        tsim.adjust_TOAs(time.TimeDelta(err))

    for f in tsim.table["flags"]:
        f["name"] = name

    return tsim


def make_fake_toas_uniform(
    startMJD,
    endMJD,
    ntoas,
    model,
    fuzz=0,
    freq=1400 * u.MHz,
    obs="GBT",
    error=1 * u.us,
    add_noise=False,
    dm=None,
    dm_error=1e-4 * pint.dmu,
    name="fake",
    include_bipm=False,
    include_gps=True,
):
    """Make evenly spaced toas

    Can include alternating frequencies if fed an array of frequencies,
    only works with one observatory at a time

    Parameters
    ----------
    startMJD : float
        starting MJD for fake toas
    endMJD : float
        ending MJD for fake toas
    ntoas : int
        number of fake toas to create between startMJD and endMJD
    model : pint.models.timing_model.TimingModel
        current model
    fuzz : astropy.units.Quantity, optional
        Standard deviation of 'fuzz' distribution to be applied to TOAs
    freq : astropy.units.Quantity, optional
        frequency of the fake toas, default 1400 MHz
    obs : str, optional
        observatory for fake toas, default GBT
    error : astropy.units.Quantity
        uncertainty to attach to each TOA
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    dm : astropy.units.Quantity, optional
        DM value to include with each TOA; default is
        not to include any DM information
    dm_error : astropy.units.Quantity
        uncertainty to attach to each DM measurement
    name : str, optional
        Name for the TOAs (goes into the flags)
    include_bipm : bool, optional
        Whether or not to disable UTC-> TT BIPM clock
        correction (see :class:`pint.observatory.topo_obs.TopoObs`)
    include_gps : bool, optional
        Whether or not to disable UTC(GPS)->UTC clock correction
        (see :class:`pint.observatory.topo_obs.TopoObs`)
    Returns
    -------
    TOAs : pint.toa.TOAs
        object with evenly spaced toas spanning given start and end MJD with
        ntoas toas, with optional errors

    Notes
    -----
    `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`

    See Also
    --------
    :func:`make_fake_toas`
    """

    times = np.linspace(startMJD, endMJD, ntoas, dtype=np.longdouble) * u.d
    if fuzz > 0:
        # apply some fuzz to the dates
        fuzz = np.random.normal(scale=fuzz.to_value(u.d), size=len(times)) * u.d
        times += fuzz

    if freq is None or np.isinf(freq).all():
        freq = np.inf * u.MHz
    freq_array = _get_freq_array(np.atleast_1d(freq), len(times))
    t1 = [
        pint.toa.TOA(t.value, obs=obs, freq=f, scale=get_observatory(obs).timescale)
        for t, f in zip(times, freq_array)
    ]
    ts = pint.toa.TOAs(toalist=t1)
    ts.planets = model["PLANET_SHAPIRO"].value
    ts.ephem = model["EPHEM"].value
    update_fake_toa_clock(ts, model, include_bipm=include_bipm, include_gps=include_gps)
    ts.table["error"] = error
    if dm is not None:
        for f in ts.table["flags"]:
            f["pp_dm"] = str(dm.to_value(pint.dmu))
            f["pp_dme"] = str(dm_error.to_value(pint.dmu))
    ts.compute_TDBs()
    ts.compute_posvels()
    return make_fake_toas(ts, model=model, add_noise=add_noise, name=name)


def make_fake_toas_fromMJDs(
    MJDs,
    model,
    freq=1400 * u.MHz,
    obs="GBT",
    error=1 * u.us,
    add_noise=False,
    dm=None,
    dm_error=1e-4 * pint.dmu,
    name="fake",
    include_bipm=False,
    include_gps=True,
):
    """Make evenly spaced toas

    Can include alternating frequencies if fed an array of frequencies,
    only works with one observatory at a time

    Parameters
    ----------
    MJDs : astropy.units.Quantity
        array of MJDs for fake toas
    model : pint.models.timing_model.TimingModel
        current model
    freq : astropy.units.Quantity, optional
        frequency of the fake toas, default 1400 MHz
    obs : str, optional
        observatory for fake toas, default GBT
    error : astropy.units.Quantity
        uncertainty to attach to each TOA
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    dm : astropy.units.Quantity, optional
        DM value to include with each TOA; default is
        not to include any DM information
    dm_error : astropy.units.Quantity
        uncertainty to attach to each DM measurement
    name : str, optional
        Name for the TOAs (goes into the flags)
    include_bipm : bool, optional
        Whether or not to disable UTC-> TT BIPM clock
        correction (see :class:`pint.observatory.topo_obs.TopoObs`)
    include_gps : bool, optional
        Whether or not to disable UTC(GPS)->UTC clock correction
        (see :class:`pint.observatory.topo_obs.TopoObs`)

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with toas matched to input array with optional errors

    Notes
    -----
    `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`

    See Also
    --------
    :func:`make_fake_toas`
    """

    times = MJDs

    if freq is None or np.isinf(freq).all():
        freq = np.inf * u.MHz
    freq_array = _get_freq_array(np.atleast_1d(freq), len(times))
    t1 = [
        pint.toa.TOA(t.value, obs=obs, freq=f, scale=get_observatory(obs).timescale)
        for t, f in zip(times, freq_array)
    ]
    ts = pint.toa.TOAs(toalist=t1)
    ts.planets = model["PLANET_SHAPIRO"].value
    ts.ephem = model["EPHEM"].value
    update_fake_toa_clock(ts, model, include_bipm=include_bipm, include_gps=include_gps)
    ts.table["error"] = error
    if dm is not None:
        for f in ts.table["flags"]:
            f["pp_dm"] = str(dm.to_value(pint.dmu))
            f["pp_dme"] = str(dm_error.to_value(pint.dmu))
    ts.compute_TDBs()
    ts.compute_posvels()
    return make_fake_toas(ts, model=model, add_noise=add_noise, name=name)


def make_fake_toas_fromtim(timfile, model, add_noise=False, name="fake"):
    """Make fake toas with the same times as an input tim file

    Can include alternating frequencies if fed an array of frequencies,
    only works with one observatory at a time

    Parameters
    ----------
    timfile : str or list of strings or file-like
        Filename, list of filenames, or file-like object containing the TOA data.
    model : pint.models.timing_model.TimingModel
        current model
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    name : str, optional
        Name for the TOAs (goes into the flags)

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with evenly spaced toas spanning given start and end MJD with
        ntoas toas, with optional errors

    See Also
    --------
    :func:`make_fake_toas`
    """
    input_ts = pint.toa.get_TOAs(timfile)
    return make_fake_toas(input_ts, model=model, add_noise=add_noise, name=name)


def calculate_random_models(
    fitter, toas, Nmodels=100, keep_models=True, return_time=False, params="all"
):
    """
    Calculates random models based on the covariance matrix of the `fitter` object.

    returns the new phase differences compared to the original model
    optionally returns all of the random models

    Parameters
    ----------
    fitter: pint.fitter.Fitter
        current fitter object containing a model and parameter covariance matrix
    toas: pint.toa.TOAs
        TOAs to calculate models
    Nmodels: int, optional
        number of random models to calculate
    keep_models: bool, optional
        whether to keep and return the individual random models (slower)
    params: list, optional
        if specified, selects only those parameters to vary.  Default ('all') is to use all parameters other than Offset

    Returns
    -------
    dphase : np.ndarray
        phase difference with respect to input model, size is [Nmodels, len(toas)]
    random_models : list, optional
        list of random models (each is a :class:`pint.models.timing_model.TimingModel`)

    Example
    -------
    >>> from pint.models import get_model_and_toas
    >>> from pint import fitter, toa
    >>> import pint.simulation
    >>> import io
    >>>
    >>> # the locations of these may vary
    >>> timfile = "tests/datafile/NGC6440E.tim"
    >>> parfile = "tests/datafile/NGC6440E.par"
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> # fit the model to the data
    >>> f = fitter.WLSFitter(toas=t, model=m)
    >>> f.fit_toas()
    >>>
    >>> # make fake TOAs starting at the end of the
    >>> # current data and going out 100 days
    >>> tnew = simulation.make_fake_toas_uniform(t.get_mjds().max().value,
    >>>                           t.get_mjds().max().value+100, 50, model=f.model)
    >>> # now make random models
    >>> dphase, mrand = pint.simulation.calculate_random_models(f, tnew, Nmodels=100)


    Note
    ----
    To calculate new TOAs, you can use :func:`~pint.simulation.make_fake_toas`

    or similar
    """
    Nmjd = len(toas)
    phases_i = np.zeros((Nmodels, Nmjd))
    phases_f = np.zeros((Nmodels, Nmjd))
    freqs = np.zeros((Nmodels, Nmjd), dtype=np.float128) * u.Hz

    cov_matrix = fitter.parameter_covariance_matrix
    # this is a list of the parameter names in the order they appear in the coviarance matrix
    param_names = cov_matrix.get_label_names(axis=0)
    # this is a dictionary with the parameter values, but it might not be in the same order
    # and it leaves out the Offset parameter
    param_values = fitter.model.get_params_dict("free", "value")
    mean_vector = np.array([param_values[x] for x in param_names if not x == "Offset"])
    if params == "all":
        # remove the first column and row (absolute phase)
        if param_names[0] == "Offset":
            cov_matrix = cov_matrix.get_label_matrix(param_names[1:])
            fac = fitter.fac[1:]
            param_names = param_names[1:]
        else:
            fac = fitter.fac
    else:
        # only select some parameters
        # need to also select from the fac array and the mean_vector array
        idx, labels = cov_matrix.get_label_slice(params)
        cov_matrix = cov_matrix.get_label_matrix(params)
        index = idx[0].flatten()
        fac = fitter.fac[index]
        # except mean_vector does not have the 'Offset' entry
        # so may need to subtract 1
        if param_names[0] == "Offset":
            mean_vector = mean_vector[index - 1]
        else:
            mean_vector = mean_vector[index]
        param_names = cov_matrix.get_label_names(axis=0)

    f_rand = deepcopy(fitter)

    # scale by fac
    mean_vector = mean_vector * fac
    scaled_cov_matrix = ((cov_matrix.matrix * fac).T * fac).T
    random_models = []
    for imodel in range(Nmodels):
        # create a set of randomized parameters based on mean vector and covariance matrix
        rparams_num = np.random.multivariate_normal(mean_vector, scaled_cov_matrix)
        # scale params back to real units
        for j in range(len(mean_vector)):
            rparams_num[j] /= fac[j]
        rparams = OrderedDict(zip(param_names, rparams_num))
        f_rand.set_params(rparams)
        phase = f_rand.model.phase(toas, abs_phase=True)
        phases_i[imodel] = phase.int
        phases_f[imodel] = phase.frac
        r = pint.residuals.Residuals(toas, f_rand.model)
        freqs[imodel] = r.get_PSR_freq(calctype="taylor")
        if keep_models:
            random_models.append(f_rand.model)
            f_rand = deepcopy(fitter)
    phases = phases_i + phases_f
    phases0 = fitter.model.phase(toas, abs_phase=True)
    dphase = phases - (phases0.int + phases0.frac)

    if return_time:
        dphase /= freqs

    if keep_models:
        return dphase, random_models
    else:
        return dphase
