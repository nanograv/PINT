"""TCB to TDB conversion of a timing model."""

import numpy as np

from pint.models.parameter import MJDParameter
from loguru import logger as log

__all__ = [
    "IFTE_K",
    "scale_parameter",
    "transform_mjd_parameter",
    "convert_tcb_to_tdb",
]

# These constants are taken from Irwin & Fukushima 1999.
# These are the same as the constants used in tempo2 as of 10 Feb 2023.
IFTE_MJD0 = np.longdouble("43144.0003725")
IFTE_KM1 = np.longdouble("1.55051979176e-8")
IFTE_K = 1 + IFTE_KM1


def scale_parameter(model, param, n):
    """Scale a parameter x by a power of IFTE_K
        x_tdb = x_tdb * IFTE_K**n

    Parameter
    ---------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    n : int
        The power of IFTE_K in the scaling factor
    """
    factor = IFTE_K**n

    if hasattr(model, param) and getattr(model, param).quantity is not None:
        par = getattr(model, param)
        par.value *= factor
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def transform_mjd_parameter(model, param):
    """Convert an MJD from TCB to TDB.
        t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    """
    factor = 1 / IFTE_K
    tref = IFTE_MJD0

    if hasattr(model, param) and getattr(model, param).quantity is not None:
        par = getattr(model, param)
        assert isinstance(par, MJDParameter)

        par.value = (par.value - tref) * factor + tref
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def convert_tcb_to_tdb(model):
    """This function performs a partial conversion of a model
    specified in TCB to TDB. While this should be sufficient as
    a starting point, the resulting parameters are only approximate
    and the model should be re-fit.

    This is based on the `transform` plugin of tempo2.

    The following parameters are converted to TDB:
        1. Spin frequency, its derivatives and spin epoch
        2. Sky coordinates, proper motion and the position epoch
        3. DM, DM derivatives and DM epoch
        4. Keplerian binary parameters

    The following parameters are NOT converted although they are
    in fact affected by the TCB to TDB conversion:
        1. Parallax
        2. TZRMJD and TZRFRQ
        2. DMX parameters
        3. Solar wind parameters
        4. Binary post-Keplerian parameters including Shapiro delay
           parameters
        5. Jumps and DM Jumps
        6. FD parameters
        7. EQUADs
        8. Red noise parameters including FITWAVES, powerlaw red noise and
           powerlaw DM noise parameters

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
       Timing model to be converted.
    """

    if model.UNITS in ["TDB", None]:
        log.warning("The input par file is already in TDB units. Doing nothing.")
        return

    log.warning(
        "The TCB to TDB conversion is only approximate. "
        "The resulting timing model should be re-fit to get reliable results."
    )

    if "Spindown" in model.components:
        for Fn_par in model.components["Spindown"].F_terms:
            n = int(Fn_par[1:])
            scale_parameter(model, Fn_par, n + 1)

        transform_mjd_parameter(model, "PEPOCH")

    if "AstrometryEquatorial" in model.components:
        scale_parameter(model, "PMRA", IFTE_K)
        scale_parameter(model, "PMDEC", IFTE_K)
        transform_mjd_parameter(model, "POSEPOCH")
    elif "AstrometryEcliptic" in model.components:
        scale_parameter(model, "PMELAT", IFTE_K)
        scale_parameter(model, "PMELONG", IFTE_K)
        transform_mjd_parameter(model, "POSEPOCH")

    # Although DM has the unit pc/cm^3, the quantity that enters
    # the timing model is DMconst*DM, which has dimensions
    # of frequency. Hence, DM and its derivatives will be
    # scaled by IFTE_K**(i+1).
    if "DispersionDM" in model.components:
        scale_parameter(model, "DM", IFTE_K)
        for i in range(1, len(model.get_DM_terms())):
            scale_parameter(model, f"DM{i}", IFTE_K ** (i + 1))
        transform_mjd_parameter(model, "DMEPOCH")

    if hasattr(model, "BINARY") and getattr(model, "BINARY").value is not None:
        transform_mjd_parameter(model, "T0")
        transform_mjd_parameter(model, "TASC")
        scale_parameter(model, "PB", -1)
        scale_parameter(model, "FB0", 1)
        scale_parameter(model, "A1", -1)

    model.UNITS.value = "TDB"

    model.validate()
