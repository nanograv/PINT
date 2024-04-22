"""TCB to TDB conversion of a timing model."""

import numpy as np

from pint import DMconst
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    floatParameter,
    maskParameter,
    prefixParameter,
)
from pint.models.noise_model import NoiseComponent
from pint.models.solar_wind_dispersion import SolarWindDispersionBase
from pint.models.absolute_phase import AbsPhase
from pint.models.dispersion_model import DispersionJump
from pint.models.frequency_dependent import FD
from pint.models.fdjump import FDJump
from pint.models.binary_bt import BinaryBTPiecewise
from pint.models.wave import Wave
from pint.models.ifunc import IFunc
from pint.models.glitch import Glitch

from loguru import logger as log
from astropy import units as u
from astropy import constants as c

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


def get_scaling_factor(param):
    scale_factors = {
        "DM": DMconst,
        "DMX_": DMconst,
        "DMWXSIN_": DMconst,
        "DMWXCOS_": DMconst,
        # "NE_SW": c.c * DMconst,
        "PX": c.c / u.au,
        "A1": 1 / c.c,
        "A1DOT": 1 / c.c,
        "M2": c.G / c.c**3,
        "MTOT": c.G / c.c**3,
    }

    if param.name in scale_factors:
        return scale_factors[param.name]
    elif hasattr(param, "prefix") and param.prefix in scale_factors:
        return scale_factors[param.prefix]
    else:
        return 1


def compute_effective_dimension(quantity, scaling_factor=1):
    unit = (quantity * scaling_factor).si.unit

    if len(unit.bases) == 0 or unit.bases == [u.rad]:
        return 0
    elif unit.bases == [u.s]:
        return unit.powers[0]
    elif set(unit.bases) == {u.s, u.rad}:
        return unit.powers[unit.bases.index(u.s)]
    else:
        raise ValueError(
            "The scaled quantity has an unsupported unit. Check the scaling_factor.",
        )


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

    # It's unclear how to transform noise parameters, so let them be for the time being.
    # Same thing for DMJUMP. It's weird.
    # I haven't worked out how the rest of these stuff transform. So I am ignoring them for the time being.
    ignore_components = [
        NoiseComponent,
        SolarWindDispersionBase,
        AbsPhase,
        DispersionJump,
        FD,
        FDJump,
        BinaryBTPiecewise,
        Wave,
        IFunc,
        Glitch,
    ]

    for param_name in model.params:
        param = model[param_name]

        if (
            not isinstance(
                param,
                (
                    floatParameter,
                    MJDParameter,
                    AngleParameter,
                    maskParameter,
                    prefixParameter,
                ),
            )
            or param.quantity is None
            or any(
                isinstance(param._parent, component_type)
                for component_type in ignore_components
            )
        ):
            continue

        if isinstance(param, MJDParameter):
            transform_mjd_parameter(model, param_name, backwards)
        else:
            sf = get_scaling_factor(param)
            n = -compute_effective_dimension(param.quantity, sf)
            scale_parameter(model, param_name, n, backwards)

    model.UNITS.value = target_units

    model.validate(allow_tcb=backwards)
