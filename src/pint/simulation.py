"""Functions related to simulating TOAs and models
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Optional, List, Union
import pathlib

import astropy.units as u
import numpy as np
from loguru import logger as log
from astropy import time

import pint.residuals
import pint.toa
import pint.fitter
from pint.observatory import bipm_default, get_observatory

__all__ = [
    "zero_residuals",
    "make_fake_toas",
    "make_fake_toas_uniform",
    "make_fake_toas_fromMJDs",
    "make_fake_toas_fromtim",
    "calculate_random_models",
]


def zero_residuals(
    ts: pint.toa.TOAs,
    model: pint.models.timing_model.TimingModel,
    *,
    subtract_mean: bool = True,
    maxiter: int = 10,
    tolerance: Optional[u.Quantity] = None,
):
    """Use a model to adjust a TOAs object, setting residuals to 0 iteratively.

    Parameters
    ----------
    ts : pint.toa.TOAs
        Input TOAs (modified in-place)
    model : pint.models.timing_model.TimingModel
        current model
    subtract_mean : bool, optional
        Controls whether mean will be subtracted from the residuals when making fake TOAs
    maxiter : int, optional
        maximum number of iterations allowed
    tolerance : astropy.units.Quantity
        maximum allowed absolute deviation of residuals from 0; default is
        1 nanosecond if operating in full precision or 5 us if not.
    """
    ts.compute_pulse_numbers(model)
    maxresid = None
    if tolerance is None:
        tolerance = 1 * u.ns if pint.utils.check_longdouble_precision() else 5 * u.us
    for i in range(maxiter):
        r = pint.residuals.Residuals(
            ts, model, subtract_mean=subtract_mean, track_mode="use_pulse_numbers"
        )
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
            f"Unable to make fake residuals - left over errors are {abs(resids).max()}"
        )


def get_fake_toa_clock_versions(
    model: pint.models.timing_model.TimingModel,
    include_bipm: bool = False,
    include_gps: bool = True,
) -> dict:
    """Get the clock settings (corrections, etc) for fake TOAs

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        current model
    include_bipm : bool, optional
        Whether or not to disable UTC-> TT BIPM clock
        correction (see :class:`pint.observatory.topo_obs.TopoObs`)
    include_gps : bool, optional
        Whether or not to disable UTC(GPS)->UTC clock correction
        (see :class:`pint.observatory.topo_obs.TopoObs`)

    Returns
    -------
    dict
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
                    bipm_version = cvers
                    log.info(f"Using CLOCK = {bipm_version} from the given model")
                else:
                    log.warning(
                        f'CLOCK = {model["CLOCK"].value} is not implemented. '
                        f"Using TT({bipm_default}) instead."
                    )
                include_bipm = True
        else:
            log.warning(
                f'CLOCK = {model["CLOCK"].value} is not implemented. '
                f"Using TT({bipm_default}) instead."
            )
            include_bipm = True

    return {
        "include_bipm": include_bipm,
        "bipm_version": bipm_version,
        "include_gps": include_gps,
    }


def make_fake_toas(
    ts: pint.toa.TOAs,
    model: pint.models.timing_model.TimingModel,
    add_noise: bool = False,
    add_correlated_noise: bool = False,
    name: str = "fake",
    subtract_mean: bool = True,
) -> pint.toa.TOAs:
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
    add_correlated_noise : bool, optional
        Add correlated noise to the TOAs if it's present in the timing mode.
    name : str, optional
        Name for the TOAs (goes into the flags)
    subtract_mean : bool, optional
        Controls whether mean will be subtracted from the residuals when making fake TOAs

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with toas matching toas but with residuals starting at zero (but then with optional noise)

    Notes
    -----
    `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`
    """
    tsim = deepcopy(ts)
    zero_residuals(tsim, model, subtract_mean=subtract_mean)

    if add_correlated_noise:
        U = model.noise_model_designmatrix(tsim)
        b = model.noise_model_basis_weight(tsim)
        corrn = (U @ (b**0.5 * np.random.normal(size=len(b)))) << u.s
        tsim.adjust_TOAs(time.TimeDelta(corrn))

    if add_noise:
        # this function will include EFAC and EQUAD
        err = model.scaled_toa_uncertainty(tsim) * np.random.normal(size=len(tsim))
        # Add the actual TOA noise
        tsim.adjust_TOAs(time.TimeDelta(err))

    for f in tsim.table["flags"]:
        f["name"] = name

    return tsim


def update_fake_dms(
    model: pint.models.timing_model.TimingModel,
    ts: pint.toa.TOAs,
    dm_error: u.Quantity,
    add_noise: bool,
) -> pint.toa.TOAs:
    """Update simulated wideband DM information in TOAs.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
    ts : pint.toa.TOAs
        Input TOAs
    dm_error: u.Quantity
    add_noise : bool, optional
        Add noise to the DMs (otherwise `dm_error` just populates the column)
    """
    toas = deepcopy(ts)

    dm_errors = dm_error * np.ones(len(toas))

    for f, dme in zip(toas.table["flags"], dm_errors):
        f["pp_dme"] = str(dme.to_value(pint.dmu))

    scaled_dm_errors = model.scaled_dm_uncertainty(toas)
    dms = model.total_dm(toas)
    if add_noise:
        dms += scaled_dm_errors.to(pint.dmu) * np.random.randn(len(scaled_dm_errors))

    for f, dm in zip(toas.table["flags"], dms):
        f["pp_dm"] = str(dm.to_value(pint.dmu))

    return toas


def make_fake_toas_uniform(
    startMJD: Union[float, u.Quantity, time.Time],
    endMJD: Union[float, u.Quantity, time.Time],
    ntoas: int,
    model: pint.models.timing_model.TimingModel,
    fuzz: u.Quantity = 0,
    freq: u.Quantity = 1400 * u.MHz,
    obs: str = "GBT",
    error: u.Quantity = 1 * u.us,
    add_noise: bool = False,
    add_correlated_noise: bool = False,
    wideband: bool = False,
    wideband_dm_error: u.Quantity = 1e-4 * pint.dmu,
    name: str = "fake",
    include_bipm: bool = False,
    include_gps: bool = True,
    multi_freqs_in_epoch: bool = False,
    flags: Optional[dict] = None,
    subtract_mean: bool = True,
) -> pint.toa.TOAs:
    """Simulate uniformly spaced TOAs.

    Parameters
    ----------
    startMJD : float or astropy.units.Quantity or astropy.time.Time
        starting MJD for fake toas
    endMJD : float or astropy.units.Quantity or astropy.time.Time
        ending MJD for fake toas
    ntoas : int
        number of fake toas to create between startMJD and endMJD
    model : pint.models.timing_model.TimingModel
        current model
    fuzz : astropy.units.Quantity, optional
        Standard deviation of 'fuzz' distribution to be applied to TOAs
    freq : astropy.units.Quantity, optional
        Frequency (or array of frequencies) for the fake TOAs,
        default is 1400 MHz
    obs : str, optional
        observatory for fake toas, default GBT
    error : astropy.units.Quantity
        uncertainty to attach to each TOA
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    add_correlated_noise : bool, optional
        Add correlated noise to the TOAs if it's present in the timing mode.
    wideband : bool, optional
        Whether to include wideband DM information with each TOA; default is
        not to include any wideband DM information. If True, the DM associated
        with each TOA will be computed using the model, and the `-ppdm` and
        `-ppdme` flags will be set.
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
    multi_freqs_in_epoch : bool, optional
        Whether to generate multiple frequency TOAs for the same epoch.
    flags: None or dict
        Dictionary of flags to be added to all simulated TOAs.
    subtract_mean : bool, optional
        Controls whether mean will be subtracted from the residuals when making fake TOAs

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with evenly spaced toas spanning given start and end MJD with
        ntoas toas, with optional errors

    Notes
    -----
    1. `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`
    2. When `wideband` is set, wideband DM measurement noise will be included
       only if `add_noise` is set. Otherwise, the `-pp_dme` flags will be set
       without adding the measurement noise to the simulated DM values.
    3. The simulated DM measurement noise respects ``DMEFAC`` and ``DMEQUAD``
       values in the `model`.
    4. If `multi_freqs_in_epoch` is True, each epoch will contain TOAs for all
       frequencies given in the `freq` argument. Otherwise, each epoch will have
       only one TOA, and the frequencies are distributed amongst TOAs in an
       alternating manner. In either case, the total number of TOAs will be `ntoas`.
    5. Currently supports simulating only one observatory.

    See Also
    --------
    :func:`make_fake_toas`
    """
    if isinstance(startMJD, time.Time):
        startMJD = startMJD.mjd << u.d
    if isinstance(endMJD, time.Time):
        endMJD = endMJD.mjd << u.d
    if not isinstance(startMJD, u.Quantity):
        startMJD = startMJD << u.d
    if not isinstance(endMJD, u.Quantity):
        endMJD = endMJD << u.d

    if freq is None or np.isinf(freq).all():
        freq = np.inf * u.MHz

    times, freq_array = _get_freqs_and_times(
        startMJD, endMJD, ntoas, freq, multi_freqs_in_epoch=multi_freqs_in_epoch
    )

    if fuzz > 0:
        # apply some fuzz to the dates
        fuzz = np.random.normal(scale=fuzz.to_value(u.d), size=len(times)) * u.d
        times += fuzz

    clk_version = get_fake_toa_clock_versions(
        model, include_bipm=include_bipm, include_gps=include_gps
    )
    ts = pint.toa.get_TOAs_array(
        times,
        obs=obs,
        scale=get_observatory(obs).timescale,
        freqs=freq_array,
        errors=error,
        ephem=model["EPHEM"].value,
        include_bipm=clk_version["include_bipm"],
        bipm_version=clk_version["bipm_version"],
        include_gps=clk_version["include_gps"],
        planets=model["PLANET_SHAPIRO"].value if "PLANET_SHAPIRO" in model else False,
        flags=flags,
    )

    if wideband:
        ts = update_fake_dms(model, ts, wideband_dm_error, add_noise)

    return make_fake_toas(
        ts,
        model=model,
        add_noise=add_noise,
        add_correlated_noise=add_correlated_noise,
        name=name,
        subtract_mean=subtract_mean,
    )


def make_fake_toas_fromMJDs(
    MJDs: Union[u.Quantity, time.Time, np.ndarray],
    model: pint.models.timing_model.TimingModel,
    freq: u.Quantity = 1400 * u.MHz,
    obs: str = "GBT",
    error: u.Quantity = 1 * u.us,
    add_noise: bool = False,
    add_correlated_noise: bool = False,
    wideband: bool = False,
    wideband_dm_error: u.Quantity = 1e-4 * pint.dmu,
    name: str = "fake",
    include_bipm: bool = False,
    include_gps: bool = True,
    multi_freqs_in_epoch: bool = False,
    flags: Optional[dict] = None,
    subtract_mean: bool = True,
) -> pint.toa.TOAs:
    """Simulate TOAs from a list of MJDs

    Parameters
    ----------
    MJDs : astropy.units.Quantity or astropy.time.Time or numpy.ndarray
        array of MJDs for fake toas
    model : pint.models.timing_model.TimingModel
        current model
    freq : astropy.units.Quantity, optional
        Frequency (or array of frequencies) for the fake toas,
        default is 1400 MHz
    obs : str, optional
        observatory for fake toas, default GBT
    error : astropy.units.Quantity
        uncertainty to attach to each TOA
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    add_correlated_noise : bool, optional
        Add correlated noise to the TOAs if it's present in the timing model.
    wideband : astropy.units.Quantity, optional
        Whether to include wideband DM values with each TOA; default is
        not to include any DM information
    wideband_dm_error : astropy.units.Quantity
        uncertainty to attach to each DM measurement
    name : str, optional
        Name for the TOAs (goes into the flags)
    include_bipm : bool, optional
        Whether or not to disable UTC-> TT BIPM clock
        correction (see :class:`pint.observatory.topo_obs.TopoObs`)
    include_gps : bool, optional
        Whether or not to disable UTC(GPS)->UTC clock correction
        (see :class:`pint.observatory.topo_obs.TopoObs`)
    multi_freqs_in_epoch : bool, optional
        Whether to generate multiple frequency TOAs for the same epoch.
    flags: None or dict
        Dictionary of flags to be added to all simulated TOAs.
    subtract_mean : bool, optional
        Controls whether mean will be subtracted from the residuals when making fake TOAs

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with toas matched to input array with optional errors

    Notes
    -----
    1. `add_noise` respects any ``EFAC`` or ``EQUAD`` present in the `model`
    2. When `wideband` is set, wideband DM measurement noise will be included
       only if `add_noise` is set. Otherwise, the `-pp_dme` flags will be set
       without adding the measurement noise to the simulated DM values.
    3. The simulated DM measurement noise respects ``DMEFAC`` and ``DMEQUAD``
       values in the `model`.
    4. If `multi_freqs_in_epoch` is True, each epoch will contain TOAs for all
       frequencies given in the `freq` argument, and the total number of
       TOAs will be `len(MJDs)*len(freq)`. Otherwise, each epoch will have
       only one TOA, and the frequencies are distributed amongst TOAs in an
       alternating manner, and the total number of TOAs will be `len(MJDs)`.
    5. Currently supports simulating only one observatory.

    See Also
    --------
    :func:`make_fake_toas`
    """
    scale = get_observatory(obs).timescale
    if isinstance(MJDs, time.Time):
        times = MJDs.mjd * u.d
        scale = None
    elif not isinstance(MJDs, (u.Quantity, np.ndarray)):
        raise TypeError(
            f"Do not know how to interpret input times of type '{type(MJDs)}'"
        )

    if freq is None or np.isinf(freq).all():
        freq = np.inf * u.MHz
    freqs = np.atleast_1d(freq)

    if not multi_freqs_in_epoch:
        times = MJDs
        freq_array = np.tile(freqs, len(MJDs) // len(freqs) + 1)[: len(times)]
    else:
        times = (
            time.Time(np.repeat(MJDs, len(freqs)))
            if isinstance(MJDs, time.Time)
            else np.repeat(MJDs, len(freqs))
        )
        freq_array = np.tile(freqs, len(MJDs))

    clk_version = get_fake_toa_clock_versions(
        model, include_bipm=include_bipm, include_gps=include_gps
    )

    ts = pint.toa.get_TOAs_array(
        times,
        obs=obs,
        freqs=freq_array,
        errors=error,
        scale=scale,
        ephem=model["EPHEM"].value,
        include_bipm=clk_version["include_bipm"],
        bipm_version=clk_version["bipm_version"],
        include_gps=clk_version["include_gps"],
        planets=model["PLANET_SHAPIRO"].value,
        flags=flags,
    )

    if wideband:
        ts = update_fake_dms(model, ts, wideband_dm_error, add_noise)

    return make_fake_toas(
        ts,
        model=model,
        add_noise=add_noise,
        add_correlated_noise=add_correlated_noise,
        name=name,
        subtract_mean=subtract_mean,
    )


def make_fake_toas_fromtim(
    timfile: Union[str, List[str], pathlib.Path],
    model: pint.models.timing_model.TimingModel,
    add_noise: bool = False,
    add_correlated_noise: bool = False,
    name: str = "fake",
    subtract_mean: bool = True,
) -> pint.toa.TOAs:
    """Simulate fake TOAs with the same times as an input tim file

    Parameters
    ----------
    timfile : str or list of strings or file-like
        Filename, list of filenames, or file-like object containing the TOA data.
    model : pint.models.timing_model.TimingModel
        current model
    add_noise : bool, optional
        Add noise to the TOAs (otherwise `error` just populates the column)
    add_correlated_noise : bool, optional
        Add correlated noise to the TOAs if it's present in the timing mode.
    name : str, optional
        Name for the TOAs (goes into the flags)
    subtract_mean : bool, optional
        Controls whether mean will be subtracted from the residuals when making fake TOAs

    Returns
    -------
    TOAs : pint.toa.TOAs
        object with evenly spaced toas spanning given start and end MJD with
        ntoas toas, with optional errors

    See Also
    --------
    :func:`make_fake_toas`
    """
    ephem = (
        model.EPHEM.value
        if hasattr(model, "EPHEM") and model.EPHEM.value is not None
        else None
    )
    planets = (
        model.PLANET_SHAPIRO.value
        if hasattr(model, "PLANET_SHAPIRO") and model.PLANET_SHAPIRO.value is not None
        else False
    )

    input_ts = pint.toa.get_TOAs(timfile, ephem=ephem, planets=planets)

    if input_ts.is_wideband():
        dm_errors = input_ts.get_dm_errors()
        ts = update_fake_dms(model, input_ts, dm_errors, add_noise)

    return make_fake_toas(
        input_ts,
        model=model,
        add_noise=add_noise,
        add_correlated_noise=add_correlated_noise,
        name=name,
        subtract_mean=subtract_mean,
    )


def calculate_random_models(
    fitter: pint.fitter.Fitter,
    toas: pint.toa.TOAs,
    Nmodels: int = 100,
    keep_models: bool = True,
    return_time: bool = False,
    params: str = "all",
) -> (np.ndarray, Optional[list]):
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
    # this is a list of the parameter names in the order they appear in the covariance matrix
    param_names = cov_matrix.get_label_names(axis=0)
    # this is a dictionary with the parameter values, but it might not be in the same order
    # and it leaves out the Offset parameter
    param_values = fitter.model.get_params_dict("free", "value")
    mean_vector = np.array([param_values[x] for x in param_names if x != "Offset"])
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

    return (dphase, random_models) if keep_models else dphase


def _get_freqs_and_times(
    start: Union[float, u.Quantity, time.Time],
    end: Union[float, u.Quantity, time.Time],
    ntoas: int,
    freqs: u.Quantity,
    multi_freqs_in_epoch: bool = True,
) -> (Union[float, u.Quantity, time.Time], np.ndarray):
    freqs = np.atleast_1d(freqs)
    assert (
        len(freqs.shape) == 1 and len(freqs) <= ntoas
    ), "`freqs` should be a single quantity or a 1D array with length less than `ntoas`."
    nfreqs = len(freqs)

    if multi_freqs_in_epoch:
        nepochs = ntoas // nfreqs + 1

        epochs = np.linspace(start, end, nepochs, dtype=np.longdouble)
        times = np.repeat(epochs, nfreqs)
        tfreqs = np.tile(freqs, nepochs)

        return times[:ntoas], tfreqs[:ntoas]
    else:
        times = np.linspace(start, end, ntoas, dtype=np.longdouble)
        tfreqs = np.tile(freqs, ntoas // nfreqs + 1)[:ntoas]
        return times, tfreqs


# def _get_freq_array(base_frequencies, ntoas):
#     """Make frequency array out of one or more frequencies

#     If >1 frequency is specified, will alternate

#     Parameters
#     ----------
#     base_frequencies : astropy.units.Quantity
#        array of frequencies
#     ntoas : int
#        number of TOAs

#     Returns
#     -------
#     astropy.units.Quantity
#         array of (potentially alternating) frequencies
#     """
#     freq = np.zeros(ntoas) * base_frequencies[0].unit
#     num_freqs = len(base_frequencies)
#     for ii, fv in enumerate(base_frequencies):
#         freq[ii::num_freqs] = fv
#     return freq
