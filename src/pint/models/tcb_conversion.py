"""TCB to TDB conversion of a timing model."""

import numpy as np

from pint.models.parameter import MJDParameter
from loguru import logger as log

__all__ = [
    "IFTE_K",
    "scale_parameter",
    "transform_mjd_parameter",
    "convert_tcb_tdb",
]

# These constants are taken from Irwin & Fukushima 1999.
# These are the same as the constants used in tempo2 as of 10 Feb 2023.
IFTE_MJD0 = np.longdouble("43144.0003725")
IFTE_KM1 = np.longdouble("1.55051979176e-8")
IFTE_K = 1 + IFTE_KM1


def scale_parameter(model, param, n, backwards):
    """Scale a parameter x by a power of IFTE_K
        x_tdb = x_tcb * IFTE_K**n

    The power n depends on the "effective dimensionality" of
    the parameter as it appears in the timing model. Some examples
    are given bellow:

        1. F0 has effective dimensionality of frequency and n = 1
        2. F1 has effective dimensionality of frequency^2 and n = 2
        3. A1 has effective dimensionality of time because it appears as
           A1/c in the timing model. Therefore, its n = -1
        4. DM has effective dimensionality of frequency because it appears
           as DM*DMconst in the timing model. Therefore, its n = 1
        5. PBDOT is dimensionless and has n = 0. i.e., it is not scaled.

    Parameter
    ---------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    n : int
        The power of IFTE_K in the scaling factor
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    assert isinstance(n, int), "The power must be an integer."

    p = -1 if backwards else 1

    factor = IFTE_K ** (p * n)

    if hasattr(model, param) and getattr(model, param).quantity is not None:
        par = getattr(model, param)
        par.value *= factor
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def transform_mjd_parameter(model, param, backwards):
    """Convert an MJD from TCB to TDB or vice versa.
        t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0
        t_tcb = (t_tdb - IFTE_MJD0) * IFTE_K + IFTE_MJD0

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    factor = IFTE_K if backwards else 1 / IFTE_K
    tref = IFTE_MJD0

    if hasattr(model, param) and getattr(model, param).quantity is not None:
        par = getattr(model, param)
        assert isinstance(par, MJDParameter)

        par.value = (par.value - tref) * factor + tref
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def convert_tcb_tdb(model, backwards=False):
    """This function performs a partial conversion of a model
    specified in TCB to TDB. While this should be sufficient as
    a starting point, the resulting parameters are only approximate
    and the model should be re-fit.

    This is based on the `transform` plugin of tempo2.

    The following parameters are converted to TDB:
        1. Spin frequency, its derivatives and spin epoch
        2. Sky coordinates, proper motion and the position epoch
        3. DM, DM derivatives and DM epoch
        4. Keplerian binary parameters and FB1

    The following parameters are NOT converted although they are
    in fact affected by the TCB to TDB conversion:
        1. Parallax
        2. TZRMJD and TZRFRQ
        2. DMX parameters
        3. Solar wind parameters
        4. Binary post-Keplerian parameters including Shapiro delay
           parameters (except FB1)
        5. Jumps and DM Jumps
        6. FD parameters
        7. EQUADs
        8. Red noise parameters including FITWAVES, powerlaw red noise and
           powerlaw DM noise parameters

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
       Timing model to be converted.
    backwards : bool
        Whether to do TDB to TCB conversion. The default is TCB to TDB.
    """

    target_units = "TCB" if backwards else "TDB"

    if model.UNITS.value == target_units or (
        model.UNITS.value is None and not backwards
    ):
        log.warning("The input par file is already in the target units. Doing nothing.")
        return

    log.warning(
        "Converting this timing model from TCB to TDB. "
        "Please note that the TCB to TDB conversion is only approximate and "
        "the resulting timing model should be re-fit to get reliable results."
    )

    if "Spindown" in model.components:
        for n, Fn_par in model.get_prefix_mapping("F").items():
            scale_parameter(model, Fn_par, n + 1, backwards)

        transform_mjd_parameter(model, "PEPOCH", backwards)

    if "AstrometryEquatorial" in model.components:
        scale_parameter(model, "PMRA", 1, backwards)
        scale_parameter(model, "PMDEC", 1, backwards)
    elif "AstrometryEcliptic" in model.components:
        scale_parameter(model, "PMELAT", 1, backwards)
        scale_parameter(model, "PMELONG", 1, backwards)
    transform_mjd_parameter(model, "POSEPOCH", backwards)

    # Although DM has the unit pc/cm^3, the quantity that enters
    # the timing model is DMconst*DM, which has dimensions
    # of frequency. Hence, DM and its derivatives will be
    # scaled by IFTE_K**(i+1).
    if "DispersionDM" in model.components:
        scale_parameter(model, "DM", 1, backwards)
        for n, DMn_par in model.get_prefix_mapping("DM").items():
            scale_parameter(model, DMn_par, n + 1, backwards)
        transform_mjd_parameter(model, "DMEPOCH", backwards)

    if hasattr(model, "BINARY") and getattr(model, "BINARY").value is not None:
        transform_mjd_parameter(model, "T0", backwards)
        transform_mjd_parameter(model, "TASC", backwards)
        scale_parameter(model, "PB", -1, backwards)
        scale_parameter(model, "FB0", 1, backwards)
        scale_parameter(model, "FB1", 2, backwards)
        scale_parameter(model, "A1", -1, backwards)

    model.UNITS.value = target_units

    model.validate(allow_tcb=backwards)
