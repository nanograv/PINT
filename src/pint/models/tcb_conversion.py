"""TCB to TDB conversion of a timing model."""

import numpy as np
from loguru import logger as log

from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    floatParameter,
    maskParameter,
    prefixParameter,
)
from pint.models.timing_model import TimingModel

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


def scale_parameter(model: TimingModel, param: str, n: int, backwards: bool) -> None:
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

    if (param in model) and model[param].quantity is not None:
        par = model[param]
        par.value *= factor
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def transform_mjd_parameter(model: TimingModel, param: str, backwards: bool) -> None:
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

    if (param in model) and model[param].quantity is not None:
        par = model[param]
        assert isinstance(par, MJDParameter) or (
            isinstance(par, prefixParameter)
            and isinstance(par.param_comp, MJDParameter)
        )

        par.value = (par.value - tref) * factor + tref
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def convert_tcb_tdb(model: TimingModel, backwards: bool = False) -> None:
    """This function performs a partial conversion of a model
    specified in TCB to TDB. While this should be sufficient as
    a starting point, the resulting parameters are only approximate
    and the model should be re-fit.

    This is roughly based on the `transform` plugin of tempo2, but uses
    a different algorithm and does a more complete conversion.

    The following parameters are NOT converted although they are
    in fact affected by the TCB to TDB conversion:
        1. TZRMJD and TZRFRQ
        2. DMJUMPs (the wideband kind)
        3. FD parameters and FD jumps
        4. EQUADs and ECORRs.
        5. GP Red noise and GP DM noise parameters
        6. Pair parameters such as Wave and IFunc parameters
        7. Variable-index chromatic delay parameters

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
       Timing model to be converted.
    backwards : bool
        Whether to do TDB to TCB conversion. The default is TCB to TDB.
    """

    target_units = "TCB" if backwards else "TDB"

    if model["UNITS"].value == target_units or (
        model["UNITS"].value is None and not backwards
    ):
        log.warning("The input par file is already in the target units. Doing nothing.")
        return

    log.warning(
        "Converting this timing model from TCB to TDB. "
        "Please note that the TCB to TDB conversion is only approximate and "
        "the resulting timing model should be re-fit to get reliable results."
    )

    for par in model.params:
        param = model[par]
        if (
            param.quantity is not None
            and hasattr(param, "convert_tcb2tdb")
            and param.convert_tcb2tdb
        ):
            if isinstance(param, (floatParameter, AngleParameter, maskParameter)) or (
                isinstance(param, prefixParameter)
                and isinstance(param.param_comp, (floatParameter, AngleParameter))
            ):
                scale_parameter(model, par, -param.effective_dimensionality, backwards)
            elif isinstance(param, MJDParameter) or (
                isinstance(param, prefixParameter)
                and isinstance(param.param_comp, MJDParameter)
            ):
                transform_mjd_parameter(model, par, backwards)

    model["UNITS"].value = target_units

    model.validate(allow_tcb=backwards)
