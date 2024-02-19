"""

Potential issues:
* orbital frequency derivatives
* Does EPS1DOT/EPS2DOT imply OMDOT and vice versa?

"""

import numpy as np
from astropy import units as u, constants as c
from astropy.time import Time
import copy
from uncertainties import ufloat, umath
from loguru import logger as log

from pint import Tsun
from pint.models.binary_bt import BinaryBT
from pint.models.binary_dd import BinaryDD, BinaryDDS, BinaryDDGR, BinaryDDH
from pint.models.binary_ddk import BinaryDDK
from pint.models.binary_ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k
from pint.models.parameter import (
    floatParameter,
    MJDParameter,
    intParameter,
    funcParameter,
)

# output types
# DDGR is not included as there is not a well-defined way to get a unique output
binary_types = ["DD", "DDK", "DDS", "DDH", "BT", "ELL1", "ELL1H", "ELL1k"]


__all__ = ["convert_binary"]


def _M2SINI_to_orthometric(model):
    """Convert from standard Shapiro delay (M2, SINI) to orthometric (H3, H4, STIGMA)

    Uses Eqns. 12, 20, 21 from Freire and Wex (2010)
    Also propagates uncertainties if present

    Note that both STIGMA and H4 should not be used

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    stigma : astropy.units.Quantity
    h3 : astropy.units.Quantity
    h4 : astropy.units.Quantity
    stigma_unc : astropy.units.Quantity or None
        Uncertainty on stigma
    h3_unc : astropy.units.Quantity or None
        Uncertainty on H3
    h4_unc : astropy.units.Quantity or None
        Uncertainty on H4

    References
    ----------
    - Freire and Wex (2010), MNRAS, 409, 199 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract

    """
    if not (hasattr(model, "M2") and hasattr(model, "SINI")):
        raise AttributeError(
            "Model must contain M2 and SINI for conversion to orthometric parameters"
        )
    sini = model.SINI.as_ufloat()
    m2 = model.M2.as_ufloat(u.Msun)
    cbar = umath.sqrt(1 - sini**2)
    stigma = sini / (1 + cbar)
    h3 = Tsun.value * m2 * stigma**3
    h4 = h3 * stigma

    stigma_unc = stigma.s if stigma.s > 0 else None
    h3_unc = h3.s * u.s if h3.s > 0 else None
    h4_unc = h4.s * u.s if h4.s > 0 else None

    return stigma.n, h3.n * u.s, h4.n * u.s, stigma_unc, h3_unc, h4_unc


def _orthometric_to_M2SINI(model):
    """Convert from orthometric (H3, H4, STIGMA) to standard Shapiro delay (M2, SINI)

    Inverts Eqns. 12, 20, 21 from Freire and Wex (2010)
    Also propagates uncertainties if present

    If STIGMA is present will use that.  Otherwise will use H4

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    M2 : astropy.units.Quantity.
    SINI : astropy.units.Quantity
    M2_unc : astropy.units.Quantity or None
        Uncertainty on M2
    SINI_unc : astropy.units.Quantity or None
        Uncertainty on SINI

    References
    ----------
    - Freire and Wex (2010), MNRAS, 409, 199 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract

    """
    if not (
        hasattr(model, "H3") and (hasattr(model, "STIGMA") or hasattr(model, "H4"))
    ):
        raise AttributeError(
            "Model must contain H3 and either STIGMA or H4 for conversion to M2/SINI"
        )
    h3 = model.H3.as_ufloat()
    h4 = (
        model.H4.as_ufloat()
        if (hasattr(model, "H4") and model.H4.value is not None)
        else None
    )
    stigma = (
        model.STIGMA.as_ufloat()
        if (hasattr(model, "STIGMA") and model.STIGMA.value is not None)
        else None
    )

    if stigma is not None:
        sini = 2 * stigma / (1 + stigma**2)
        m2 = h3 / stigma**3 / Tsun.value
    else:
        # FW10 Eqn. 25, 26
        sini = 2 * h3 * h4 / (h3**2 + h4**2)
        m2 = h3**4 / h4**3 / Tsun.value

    m2_unc = m2.s * u.Msun if m2.s > 0 else None
    sini_unc = sini.s if sini.s > 0 else None

    return m2.n * u.Msun, sini.n, m2_unc, sini_unc


def _SINI_to_SHAPMAX(model):
    """Convert from standard SINI to alternate SHAPMAX parameterization

    Also propagates uncertainties if present

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    SHAPMAX : astropy.units.Quantity
    SHAPMAX_unc : astropy.units.Quantity or None
        Uncertainty on SHAPMAX
    """
    if not hasattr(model, "SINI"):
        raise AttributeError("Model must contain SINI for conversion to SHAPMAX")
    sini = model.SINI.as_ufloat()
    shapmax = -umath.log(1 - sini)
    return shapmax.n, shapmax.s if shapmax.s > 0 else None


def _SHAPMAX_to_SINI(model):
    """Convert from alternate SHAPMAX to SINI parameterization

    Also propagates uncertainties if present

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    SINI : astropy.units.Quantity
    SINI_unc : astropy.units.Quantity or None
        Uncertainty on SINI
    """
    if not hasattr(model, "SHAPMAX"):
        raise AttributeError("Model must contain SHAPMAX for conversion to SINI")
    shapmax = model.SHAPMAX.as_ufloat()
    sini = 1 - umath.exp(-shapmax)
    return sini.n, sini.s if sini.s > 0 else None


def _from_ELL1(model):
    """Convert from ELL1 parameterization to standard orbital parameterization

    Converts using Eqns. 1, 2, and 3 from Lange et al. (2001)
    Also computes EDOT if present
    Also propagates uncertainties if present

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    ECC : astropy.units.Quantity
    OM : astropy.units.Quantity
    T0 : astropy.units.Quantity
    EDOT : astropy.units.Quantity or None
    OMDOT : astropy.units.Quantity or None
    ECC_unc : astropy.units.Quantity or None
        Uncertainty on ECC
    OM_unc : astropy.units.Quantity or None
        Uncertainty on OM
    T0_unc : astropy.units.Quantity or None
        Uncertainty on T0
    EDOT_unc : astropy.units.Quantity or None
        Uncertainty on EDOT
    OMDOTDOT_unc : astropy.units.Quantity or None
        Uncertainty on OMDOT

    References
    ----------
    - Lange et al. (2001), MNRAS, 326, 274 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract

    """
    if model.BINARY.value not in ["ELL1", "ELL1H", "ELL1k"]:
        raise ValueError(f"Requires model ELL1* rather than {model.BINARY.value}")

    PB, PBerr = model.pb()
    pb = ufloat(PB.to_value(u.d), PBerr.to_value(u.d) if PBerr is not None else 0)
    eps1 = model.EPS1.as_ufloat()
    eps2 = model.EPS2.as_ufloat()
    om = umath.atan2(eps1, eps2)
    if om < 0:
        om += 2 * np.pi
    ecc = umath.sqrt(eps1**2 + eps2**2)

    tasc1, tasc2 = model.TASC.as_ufloats()
    t01 = tasc1
    t02 = tasc2 + (pb / 2 / np.pi) * om
    T0 = Time(
        t01.n,
        val2=t02.n,
        scale=model.TASC.quantity.scale,
        precision=model.TASC.quantity.precision,
        format="jd",
    )
    edot = None
    omdot = None
    if model.BINARY.value == "ELL1k":
        lnedot = model.LNEDOT.as_ufloat(u.Hz)
        edot = lnedot * ecc
        omdot = model.OMDOT.as_ufloat(u.rad / u.s)

    else:
        if model.EPS1DOT.quantity is not None and model.EPS2DOT.quantity is not None:
            eps1dot = model.EPS1DOT.as_ufloat(u.Hz)
            eps2dot = model.EPS2DOT.as_ufloat(u.Hz)
            edot = (eps1dot * eps1 + eps2dot * eps2) / ecc
            omdot = (eps1dot * eps2 - eps2dot * eps1) / ecc**2

    return (
        ecc.n,
        (om.n * u.rad).to(u.deg),
        T0,
        edot.n * u.Hz if edot is not None else None,
        (omdot.n * u.rad / u.s).to(u.deg / u.yr) if omdot is not None else None,
        ecc.s if ecc.s > 0 else None,
        (om.s * u.rad).to(u.deg) if om.s > 0 else None,
        t02.s * u.d if t02.s > 0 else None,
        edot.s * u.Hz if (edot is not None and edot.s > 0) else None,
        (omdot.s * u.rad / u.s).to(u.deg / u.yr)
        if (omdot is not None and omdot.s > 0)
        else None,
    )


def _to_ELL1(model):
    """Convert from standard orbital parameterization to ELL1 parameterization

    Converts using Eqns. 1, 2, and 3 from Lange et al. (2001)
    Also computes EPS?DOT if present
    Also propagates uncertainties if present

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    EPS1 : astropy.units.Quantity
    EPS2 : astropy.units.Quantity
    TASC : astropy.units.Quantity
    EPS1DOT : astropy.units.Quantity or None
    EPS2DOT : astropy.units.Quantity or None
    EPS1_unc : astropy.units.Quantity or None
        Uncertainty on EPS1
    EPS2_unc : astropy.units.Quantity or None
        Uncertainty on EPS2
    TASC_unc : astropy.units.Quantity or None
        Uncertainty on TASC
    EPS1DOT_unc : astropy.units.Quantity or None
        Uncertainty on EPS1DOT
    EPS2DOT_unc : astropy.units.Quantity or None
        Uncertainty on EPS2DOT

    References
    ----------
    - Lange et al. (2001), MNRAS, 326, 274 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract

    """
    if not (hasattr(model, "ECC") and hasattr(model, "T0") and hasattr(model, "OM")):
        raise AttributeError(
            "Model must contain ECC, T0, OM for conversion to EPS1/EPS2"
        )
    ecc = model.ECC.as_ufloat()
    om = model.OM.as_ufloat(u.rad)
    eps1 = ecc * umath.sin(om)
    eps2 = ecc * umath.cos(om)
    PB, PBerr = model.pb()
    pb = ufloat(PB.to_value(u.d), PBerr.to_value(u.d) if PBerr is not None else 0)
    t01, t02 = model.T0.as_ufloats()
    tasc1 = t01
    tasc2 = t02 - (pb * om / 2 / np.pi)
    TASC = Time(
        tasc1.n,
        val2=tasc2.n,
        format="jd",
        scale=model.T0.quantity.scale,
        precision=model.T0.quantity.precision,
    )
    eps1dot = None
    eps2dot = None
    if model.EDOT.quantity is not None or model.OMDOT.quantity is not None:
        if model.EDOT.quantity is not None:
            edot = model.EDOT.as_ufloat(u.Hz)
        else:
            edot = ufloat(0, 0)
        if model.OMDOT.quantity is not None:
            omdot = model.OMDOT.as_ufloat(u.rad * u.Hz)
        else:
            omdot = ufloat(0, 0)
        eps1dot = edot * umath.sin(om) + ecc * umath.cos(om) * omdot
        eps2dot = edot * umath.cos(om) - ecc * umath.sin(om) * omdot
    return (
        eps1.n,
        eps2.n,
        TASC,
        eps1dot.n * u.Hz,
        eps2dot.n * u.Hz,
        eps1.s if eps1.s > 0 else None,
        eps2.s if eps2.s > 0 else None,
        tasc2.s * u.d if tasc2.s > 0 else None,
        eps1dot.s * u.Hz if (eps1dot is not None and eps1dot.s > 0) else None,
        eps2dot.s * u.Hz if (eps2dot is not None and eps2dot.s > 0) else None,
    )


def _ELL1_to_ELL1k(model):
    """Convert from ELL1 EPS1DOT/EPS2DOT to ELL1k LNEDOT/OMDOT

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    LNEDOT: astropy.units.Quantity
    OMDOT: astropy.units.Quantity
    LNEDOT_unc: astropy.units.Quantity or None
        Uncertainty on LNEDOT
    OMDOT_unc: astropy.units.Quantity or None
        Uncertainty on OMDOT

    References
    ----------
    - Susobhanan et al. (2018), MNRAS, 480 (4), 5260-5271 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.5260S/abstract
    """
    if model.BINARY.value not in ["ELL1", "ELL1H"]:
        raise ValueError(f"Requires model ELL1/ELL1H rather than {model.BINARY.value}")
    eps1 = model.EPS1.as_ufloat()
    eps2 = model.EPS2.as_ufloat()
    eps1dot = model.EPS1DOT.as_ufloat(u.Hz)
    eps2dot = model.EPS2DOT.as_ufloat(u.Hz)
    ecc = umath.sqrt(eps1**2 + eps2**2)
    lnedot = (eps1 * eps1dot + eps2 * eps2dot) / ecc
    omdot = (eps2 * eps1dot - eps1 * eps2dot) / ecc

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        lnedot_unc = lnedot.s / u.s if lnedot.s > 0 else None
        omdot_unc = (omdot.s / u.s).to(u.deg / u.yr) if omdot.s > 0 else None
        return lnedot.n / u.s, (omdot.n / u.s).to(u.deg / u.yr), lnedot_unc, omdot_unc


def _ELL1k_to_ELL1(model):
    """Convert from ELL1k LNEDOT/OMDOT to ELL1 EPS1DOT/EPS2DOT

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    EPS1DOT: astropy.units.Quantity
    EPS2DOT: astropy.units.Quantity
    EPS1DOT_unc: astropy.units.Quantity or None
        Uncertainty on EPS1DOT
    EPS2DOT_unc: astropy.units.Quantity or None
        Uncertainty on EPS2DOT

    References
    ----------
    - Susobhanan et al. (2018), MNRAS, 480 (4), 5260-5271 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.5260S/abstract
    """
    if model.BINARY.value != "ELL1k":
        raise ValueError(f"Requires model ELL1k rather than {model.BINARY.value}")
    eps1 = model.EPS1.as_ufloat()
    eps2 = model.EPS2.as_ufloat()
    lnedot = model.LNEDOT.as_ufloat(u.Hz)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        omdot = model.OMDOT.as_ufloat(1 / u.s)
    eps1dot = lnedot * eps1 + omdot * eps2
    eps2dot = lnedot * eps2 - omdot * eps1

    eps1dot_unc = eps1dot.s / u.s if eps1dot.s > 0 else None
    eps2dot_unc = eps2dot.s / u.s if eps2dot.s > 0 else None
    return eps1dot.n / u.s, eps2dot.n / u.s, eps1dot_unc, eps2dot_unc


def _DDGR_to_PK(model):
    """Convert DDGR model to equivalent PK parameters

    Uses ``uncertainties`` module to propagate uncertainties

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    pbdot : uncertainties.core.Variable
    gamma : uncertainties.core.Variable
    omegadot : uncertainties.core.Variable
    s : uncertainties.core.Variable
    r : uncertainties.core.Variable
    Dr : uncertainties.core.Variable
    Dth : uncertainties.core.Variable
    """
    if model.BINARY.value != "DDGR":
        raise ValueError(
            f"Requires DDGR model for conversion, not '{model.BINARY.value}'"
        )
    tsun = Tsun.to_value(u.s)
    mtot = model.MTOT.as_ufloat(u.Msun)
    mc = model.M2.as_ufloat(u.Msun)
    x = model.A1.as_ufloat()
    PB, PBerr = model.pb()
    pb = ufloat(PB.to_value(u.s), PBerr.to_value(u.s) if PBerr is not None else 0)
    n = 2 * np.pi / pb
    mp = mtot - mc
    ecc = model.ECC.as_ufloat()
    # units are seconds
    gamma = (
        tsun ** (2.0 / 3)
        * n ** (-1.0 / 3)
        * ecc
        * (mc * (mp + 2 * mc) / (mp + mc) ** (4.0 / 3))
    )
    # units as seconds
    r = tsun * mc
    # units are radian/s
    omegadot = (
        (3 * tsun ** (2.0 / 3))
        * n ** (5.0 / 3)
        * (1 / (1 - ecc**2))
        * (mp + mc) ** (2.0 / 3)
    )
    if model.XOMDOT.quantity is not None:
        omegadot += model.XOMDOT.as_ufloat(u.rad / u.s)
    fe = (1 + (73.0 / 24) * ecc**2 + (37.0 / 96) * ecc**4) / (1 - ecc**2) ** (
        7.0 / 2
    )
    # units as s/s
    pbdot = (
        (-192 * np.pi / 5)
        * tsun ** (5.0 / 3)
        * n ** (5.0 / 3)
        * fe
        * (mp * mc)
        / (mp + mc) ** (1.0 / 3)
    )
    if model.XPBDOT.quantity is not None:
        pbdot += model.XPBDOT.as_ufloat(u.s / u.s)
    # dimensionless
    s = tsun ** (-1.0 / 3) * n ** (2.0 / 3) * x * (mp + mc) ** (2.0 / 3) / mc
    Dr = (
        tsun ** (2.0 / 3)
        * n ** (2.0 / 3)
        * (3 * mp**2 + 6 * mp * mc + 2 * mc**2)
        / (mp + mc) ** (4.0 / 3)
    )
    Dth = (
        tsun ** (2.0 / 3)
        * n ** (2.0 / 3)
        * (3.5 * mp**2 + 6 * mp * mc + 2 * mc**2)
        / (mp + mc) ** (4.0 / 3)
    )
    return pbdot, gamma, omegadot, s, r, Dr, Dth


def _transfer_params(inmodel, outmodel, badlist=[]):
    """Transfer parameters between an input and output model, excluding certain parameters

    Parameters (input or output) that are :class:`~pint.models.parameter.funcParameter` are not copied

    Parameters
    ----------
    inmodel : pint.models.timing_model.TimingModel
    outmodel : pint.models.timing_model.TimingModel
    badlist : list, optional
        List of parameters to not transfer

    """
    inbinary_component_name = [
        x for x in inmodel.components.keys() if x.startswith("Binary")
    ][0]
    outbinary_component_name = [
        x for x in outmodel.components.keys() if x.startswith("Binary")
    ][0]
    for p in inmodel.components[inbinary_component_name].params:
        if p not in badlist:
            setattr(
                outmodel.components[outbinary_component_name],
                p,
                copy.deepcopy(getattr(inmodel.components[inbinary_component_name], p)),
            )


def convert_binary(model, output, NHARMS=3, useSTIGMA=False, KOM=0 * u.deg):
    """
    Convert between binary models

    Input models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`,
    :class:`~pint.models.binary_dd.BinaryDDGR`, :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`,
    :class:`~pint.models.binary_ell1.BinaryELL1`, :class:`~pint.models.binary_ell1.BinaryELL1H`, :class:`~pint.models.binary_ell1.BinaryELL1k`,
    :class:`~pint.models.binary_dd.BinaryDDH`

    Output models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`,
    :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`, :class:`~pint.models.binary_ell1.BinaryELL1`,
    :class:`~pint.models.binary_ell1.BinaryELL1H`, :class:`~pint.models.binary_ell1.BinaryELL1k`, :class:`~pint.models.binary_dd.BinaryDDH`

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
    output : str
        Output model type
    NHARMS : int, optional
        Number of harmonics (``ELL1H`` only)
    useSTIGMA : bool, optional
        Whether to use STIGMA or H4 (``ELL1H`` only)
    KOM : astropy.units.Quantity
        Longitude of the ascending node (``DDK`` only)

    Returns
    -------
    outmodel : pint.models.timing_model.TimingModel

    Notes
    -----
    Default value in `pint` for `NHARMS` is 7, while in `tempo2` it is 4.
    """
    # Do initial checks
    if output not in binary_types:
        raise ValueError(
            f"Requested output binary '{output}' is not one of the known types ({binary_types})"
        )

    if not model.is_binary:
        raise AttributeError("Input model is not a binary")

    binary_component_name = [
        x for x in model.components.keys() if x.startswith("Binary")
    ][0]
    binary_component = model.components[binary_component_name]
    if binary_component.binary_model_name == output:
        log.debug(
            f"Input model and requested output are both of type '{output}'; returning copy"
        )
        return copy.deepcopy(model)
    log.debug(f"Converting from '{binary_component.binary_model_name}' to '{output}'")

    outmodel = copy.deepcopy(model)
    outmodel.remove_component(binary_component_name)
    outmodel.BINARY.value = output

    if binary_component.binary_model_name in ["ELL1", "ELL1H", "ELL1k"]:
        # from ELL1, ELL1H, ELL1k
        if output == "ELL1H":
            # ELL1,ELL1k -> ELL1H
            stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(model)
            # parameters not to copy
            badlist = ["M2", "SINI", "BINARY", "EDOT", "OMDOT"]
            outmodel.add_component(BinaryELL1H(), validate=False)
            if binary_component.binary_model_name == "ELL1k":
                badlist += ["LNEDOT"]
                EPS1DOT, EPS2DOT, EPS1DOT_unc, EPS2DOT_unc = _ELL1k_to_ELL1(model)
                if EPS1DOT is not None:
                    outmodel.EPS1DOT.quantity = EPS1DOT
                    if EPS1DOT_unc is not None:
                        outmodel.EPS1DOT.uncertainty = EPS1DOT_unc
                if EPS2DOT is not None:
                    outmodel.EPS2DOT.quantity = EPS2DOT
                    if EPS2DOT_unc is not None:
                        outmodel.EPS2DOT.uncertainty = EPS2DOT_unc
                outmodel.EPS1DOT.frozen = model.LNEDOT.frozen or model.OMDOT.frozen
                outmodel.EPS2DOT.frozen = model.LNEDOT.frozen or model.OMDOT.frozen
            _transfer_params(model, outmodel, badlist)
            outmodel.NHARMS.value = NHARMS
            outmodel.H3.quantity = h3
            outmodel.H3.uncertainty = h3_unc
            outmodel.H3.frozen = model.M2.frozen or model.SINI.frozen
            if useSTIGMA:
                # use STIGMA and H3
                outmodel.STIGMA.quantity = stigma
                outmodel.STIGMA.uncertainty = stigma_unc
                outmodel.STIGMA.frozen = outmodel.H3.frozen
            else:
                # use H4 and H3
                outmodel.H4.quantity = h4
                outmodel.H4.uncertainty = h4_unc
                outmodel.H4.frozen = outmodel.H3.frozen
        elif output in ["ELL1"]:
            if model.BINARY.value == "ELL1H":
                # ELL1H -> ELL1
                M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                # parameters not to copy
                badlist = ["H3", "H4", "STIGMA", "BINARY", "EDOT", "OMDOT"]
                if output == "ELL1":
                    outmodel.add_component(BinaryELL1(), validate=False)
                _transfer_params(model, outmodel, badlist)
                outmodel.M2.quantity = M2
                outmodel.SINI.quantity = SINI
                if model.STIGMA.quantity is not None:
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                    outmodel.SINI.frozen = model.STIGMA.frozen
                else:
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                    outmodel.SINI.frozen = model.STIGMA.frozen or model.H3.frozen
                if M2_unc is not None:
                    outmodel.M2.uncertainty = M2_unc
                if SINI_unc is not None:
                    outmodel.SINI.uncertainty = SINI_unc
            elif model.BINARY.value == "ELL1k":
                # ELL1k -> ELL1
                # parameters not to copy
                badlist = ["BINARY", "LNEDOT", "OMDOT", "EDOT"]
                if output == "ELL1":
                    outmodel.add_component(BinaryELL1(), validate=False)
                EPS1DOT, EPS2DOT, EPS1DOT_unc, EPS2DOT_unc = _ELL1k_to_ELL1(model)
                _transfer_params(model, outmodel, badlist)
                if EPS1DOT is not None:
                    outmodel.EPS1DOT.quantity = EPS1DOT
                    if EPS1DOT_unc is not None:
                        outmodel.EPS1DOT.uncertainty = EPS1DOT_unc
                if EPS2DOT is not None:
                    outmodel.EPS2DOT.quantity = EPS2DOT
                    if EPS2DOT_unc is not None:
                        outmodel.EPS2DOT.uncertainty = EPS2DOT_unc
                outmodel.EPS1DOT.frozen = model.LNEDOT.frozen or model.OMDOT.frozen
                outmodel.EPS2DOT.frozen = model.LNEDOT.frozen or model.OMDOT.frozen
        elif output == "ELL1k":
            if model.BINARY.value in ["ELL1"]:
                # ELL1 -> ELL1k
                LNEDOT, OMDOT, LNEDOT_unc, OMDOT_unc = _ELL1_to_ELL1k(model)
                # parameters not to copy
                badlist = ["BINARY", "EPS1DOT", "EPS2DOT", "OMDOT", "EDOT"]
                outmodel.add_component(BinaryELL1k(), validate=False)
                _transfer_params(model, outmodel, badlist)
                outmodel.LNEDOT.quantity = LNEDOT
                outmodel.OMDOT.quantity = OMDOT
                if LNEDOT_unc is not None:
                    outmodel.LNEDOT.uncertainty = LNEDOT_unc
                if OMDOT_unc is not None:
                    outmodel.OMDOT.uncertainty = OMDOT_unc
                outmodel.LNEDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
                outmodel.OMDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
            elif model.BINARY.value == "ELL1H":
                # ELL1H -> ELL1k
                LNEDOT, OMDOT, LNEDOT_unc, OMDOT_unc = _ELL1_to_ELL1k(model)
                M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                # parameters not to copy
                badlist = [
                    "BINARY",
                    "EPS1DOT",
                    "EPS2DOT",
                    "H3",
                    "H4",
                    "STIGMA",
                    "OMDOT",
                    "EDOT",
                ]
                outmodel.add_component(BinaryELL1k(), validate=False)
                _transfer_params(model, outmodel, badlist)
                outmodel.LNEDOT.quantity = LNEDOT
                outmodel.OMDOT.quantity = OMDOT
                if LNEDOT_unc is not None:
                    outmodel.LNEDOT.uncertainty = LNEDOT_unc
                if OMDOT_unc is not None:
                    outmodel.OMDOT.uncertainty = OMDOT_unc
                outmodel.LNEDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
                outmodel.OMDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
                outmodel.M2.quantity = M2
                outmodel.SINI.quantity = SINI
                if model.STIGMA.quantity is not None:
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                    outmodel.SINI.frozen = model.STIGMA.frozen
                else:
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                    outmodel.SINI.frozen = model.STIGMA.frozen or model.H3.frozen
                if M2_unc is not None:
                    outmodel.M2.uncertainty = M2_unc
                if SINI_unc is not None:
                    outmodel.SINI.uncertainty = SINI_unc
        elif output in ["DD", "DDH", "DDS", "DDK", "BT"]:
            # (ELL1, ELL1k, ELL1H) -> (DD, DDH, DDS, DDK, BT)
            # need to convert from EPS1/EPS2/TASC to ECC/OM/TASC
            (
                ECC,
                OM,
                T0,
                EDOT,
                OMDOT,
                ECC_unc,
                OM_unc,
                T0_unc,
                EDOT_unc,
                OMDOT_unc,
            ) = _from_ELL1(model)
            # parameters not to copy
            badlist = [
                "ECC",
                "OM",
                "TASC",
                "EPS1",
                "EPS2",
                "EPS1DOT",
                "EPS2DOT",
                "BINARY",
                "OMDOT",
                "EDOT",
            ]
            if output == "DD":
                outmodel.add_component(BinaryDD(), validate=False)
            elif output == "DDS":
                outmodel.add_component(BinaryDDS(), validate=False)
                badlist.append("SINI")
            elif output == "DDH":
                outmodel.add_component(BinaryDDH(), validate=False)
                badlist.append("M2")
                badlist.append("SINI")
            elif output == "DDK":
                outmodel.add_component(BinaryDDK(), validate=False)
                badlist.append("SINI")
            elif output == "BT":
                outmodel.add_component(BinaryBT(), validate=False)
                badlist += ["M2", "SINI"]
            if binary_component.binary_model_name == "ELL1H":
                badlist += ["H3", "H4", "STIGMA", "VARSIGMA", "STIG"]
            _transfer_params(model, outmodel, badlist)
            outmodel.ECC.quantity = ECC
            outmodel.ECC.uncertainty = ECC_unc
            outmodel.ECC.frozen = model.EPS1.frozen or model.EPS2.frozen
            outmodel.OM.quantity = OM.to(u.deg, equivalencies=u.dimensionless_angles())
            outmodel.OM.uncertainty = OM_unc.to(
                u.deg, equivalencies=u.dimensionless_angles()
            )
            outmodel.OM.frozen = model.EPS1.frozen or model.EPS2.frozen
            outmodel.T0.quantity = T0
            outmodel.T0.uncertainty = T0_unc
            if model.PB.quantity is not None:
                outmodel.T0.frozen = (
                    model.EPS1.frozen
                    or model.EPS2.frozen
                    or model.TASC.frozen
                    or model.PB.frozen
                )
            elif model.FB0.quantity is not None:
                outmodel.T0.frozen = (
                    model.EPS1.frozen
                    or model.EPS2.frozen
                    or model.TASC.frozen
                    or model.FB0.frozen
                )
            if EDOT is not None:
                outmodel.EDOT.quantity = EDOT
            if EDOT_unc is not None:
                outmodel.EDOT.uncertainty = EDOT_unc
            if OMDOT is not None:
                outmodel.OMDOT.quantity = OMDOT
            if OMDOT_unc is not None:
                outmodel.OMDOT.uncertainty = OMDOT_unc
            if binary_component.binary_model_name != "ELL1k":
                outmodel.EDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
                outmodel.OMDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
            else:
                outmodel.EDOT.frozen = model.LNEDOT.frozen
            if binary_component.binary_model_name == "ELL1H":
                if output not in ["DDH", "DDS", "DDK"]:
                    M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                    outmodel.M2.quantity = M2
                    outmodel.SINI.quantity = SINI
                    if M2_unc is not None:
                        outmodel.M2.uncertainty = M2_unc
                    if SINI_unc is not None:
                        outmodel.SINI.uncertainty = SINI_unc
                    if model.STIGMA.quantity is not None:
                        outmodel.SINI.frozen = model.STIGMA.frozen
                        outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                    else:
                        outmodel.SINI.frozen = model.H3.frozen or model.H4.frozen
                        outmodel.M2.frozen = model.H3.frozen or model.H4.frozen
                elif output == "DDH":
                    outmodel.H3.quantity = model.H3.quantity
                    if model.H3.uncertainty is not None:
                        outmodel.H3.uncertainty = model.H3.uncertainty
                        outmodel.H3.frozen = model.H3.frozen
                    if model.STIGMA.quantity is not None:
                        outmodel.STIGMA.quantity = model.STIGMA.quantity
                        if model.STIGMA.uncertainty is None:
                            outmodel.STIGMA.uncertainty = model.STIGMA.uncertainty
                            outmodel.STIGMA.frozen = model.STIGMA.frozen
                    else:
                        outmodel.STIGMA.quantity = model.H3.quantity / model.H4.quantity
                        if (
                            model.H3.uncertainty is not None
                            and model.H4.uncertainty is not None
                        ):
                            outmodel.STIGMA.uncertainty = np.sqrt(
                                (model.H4.uncertainty / model.H3.quantity) ** 2
                                + (
                                    model.H3.uncertainty
                                    * model.H4.quantity
                                    / model.H3.quantity**2
                                )
                                ** 2
                            )
                        outmodel.STIGMA.frozen = model.H3.frozen or model.H4.frozen
                elif output == "DDS":
                    tempmodel = convert_binary(model, "ELL1")
                    outmodel = convert_binary(tempmodel, output)
                elif output == "DDK":
                    tempmodel = convert_binary(model, "ELL1")
                    outmodel = convert_binary(tempmodel, output)
            elif output == "DDH":
                stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(
                    model
                )
                outmodel.STIGMA.quantity = stigma
                outmodel.H3.quantity = h3
                if stigma_unc is not None:
                    outmodel.STIGMA.uncertainty = stigma_unc
                if h3_unc is not None:
                    outmodel.H3.uncertainty = h3_unc
                outmodel.STIGMA.frozen = model.SINI.frozen
                outmodel.H3.frozen = model.SINI.frozen or model.M2.frozen

        else:
            raise ValueError(
                f"Do not know how to convert from {binary_component.binary_model_name} to {output}"
            )
    elif binary_component.binary_model_name in [
        "DD",
        "DDH",
        "DDGR",
        "DDS",
        "DDK",
        "BT",
    ]:
        if output in ["DD", "DDH", "DDS", "DDK", "BT"]:
            # (DD, DDH, DDGR, DDS, DDK, BT) -> (DD, DDH, DDS, DDK, BT)
            # parameters not to copy
            badlist = [
                "BINARY",
            ]
            if binary_component.binary_model_name == "DDS":
                badlist += ["SHAPMAX", "SINI"]
            elif binary_component.binary_model_name == "DDK":
                badlist += ["KIN", "KOM", "SINI"]
            elif binary_component.binary_model_name == "DDH":
                badlist += ["H3", "STIGMA", "M2", "SINI"]
            elif binary_component.binary_model_name == "DDGR":
                badlist += [
                    "PBDOT",
                    "OMDOT",
                    "GAMMA",
                    "DR",
                    "DTH",
                    "SINI",
                    "XOMDOT",
                    "XPBDOT",
                ]
            if output == "DD":
                outmodel.add_component(BinaryDD(), validate=False)
            elif output == "DDS":
                outmodel.add_component(BinaryDDS(), validate=False)
                badlist.append("SINI")
            elif output == "DDH":
                outmodel.add_component(BinaryDDH(), validate=False)
                badlist += ["M2", "SINI"]
            elif output == "DDK":
                outmodel.add_component(BinaryDDK(), validate=False)
                badlist.append("SINI")
            elif output == "BT":
                outmodel.add_component(BinaryBT(), validate=False)
                badlist += ["M2", "SINI"]
            _transfer_params(model, outmodel, badlist)
            if binary_component.binary_model_name == "DDS":
                if output not in ["DDH", "DDK"]:
                    SINI, SINI_unc = _SHAPMAX_to_SINI(model)
                    outmodel.SINI.quantity = SINI
                    if SINI_unc is not None:
                        outmodel.SINI.uncertainty = SINI_unc
                elif output == "DDH":
                    tempmodel = convert_binary(model, "DD")
                    stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(
                        tempmodel
                    )
                    outmodel.STIGMA.quantity = stigma
                    if stigma_unc is not None:
                        outmodel.STIGMA.uncertainty = stigma_unc
                    outmodel.H3.quantity = h3
                    if h3_unc is not None:
                        outmodel.H3.uncertainty = h3_unc
                    outmodel.STIGMA.frozen = model.SHAPMAX.frozen
                    outmodel.H3.frozen = model.SHAPMAX.frozen or model.M2.frozen
                elif output == "DDK":
                    tempmodel = convert_binary(model, "DD")
                    outmodel = convert_binary(tempmodel, output)
            elif binary_component.binary_model_name == "DDH":
                if output not in ["DDS", "DDK"]:
                    M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                    outmodel.M2.quantity = M2
                    outmodel.SINI.quantity = SINI
                    if M2_unc is not None:
                        outmodel.M2.uncertainty = M2_unc
                    if SINI_unc is not None:
                        outmodel.SINI.uncertainty = SINI_unc
                    outmodel.SINI.frozen = model.STIGMA.frozen
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
                else:
                    tempmodel = convert_binary(model, "DD")
                    outmodel = convert_binary(tempmodel, output)
            elif binary_component.binary_model_name == "DDK":
                if output not in ["DDH", "DDS"]:
                    if model.KIN.quantity is not None:
                        outmodel.SINI.quantity = np.sin(model.KIN.quantity)
                        if model.KIN.uncertainty is not None:
                            outmodel.SINI.uncertainty = np.abs(
                                model.KIN.uncertainty * np.cos(model.KIN.quantity)
                            ).to(
                                u.dimensionless_unscaled,
                                equivalencies=u.dimensionless_angles(),
                            )
                        outmodel.SINI.frozen = model.KIN.frozen
                elif output == "DDH":
                    tempmodel = convert_binary(model, "DD")
                    stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(
                        tempmodel
                    )
                    outmodel.STIGMA.quantity = stigma
                    if stigma_unc is not None:
                        outmodel.STIGMA.uncertainty = stigma_unc
                    outmodel.H3.quantity = h3
                    if h3_unc is not None:
                        outmodel.H3.uncertainty = h3_unc
                    outmodel.STIGMA.frozen = model.KIN.frozen
                    outmodel.H3.frozen = model.KIN.frozen or model.M2.frozen
                elif output == "DDS":
                    tempmodel = convert_binary(model, "DD")
                    shapmax, shapmax_unc = _SINI_to_SHAPMAX(tempmodel)
                    outmodel.SHAPMAX.quantity = shapmax
                    if shapmax_unc is not None:
                        outmodel.SHAPMAX.uncertainty = shapmax_unc
                    outmodel.SHAPMAX.frozen = model.KIN.frozen
            elif binary_component.binary_model_name == "DDGR":
                pbdot, gamma, omegadot, s, r, Dr, Dth = _DDGR_to_PK(model)
                outmodel.GAMMA.value = gamma.n
                if gamma.s > 0:
                    outmodel.GAMMA.uncertainty_value = gamma.s
                outmodel.PBDOT.value = pbdot.n
                if pbdot.s > 0:
                    outmodel.PBDOT.uncertainty_value = pbdot.s
                outmodel.OMDOT.value = (omegadot.n * u.rad / u.s).to_value(u.deg / u.yr)
                if omegadot.s > 0:
                    outmodel.OMDOT.uncertainty_value = (
                        omegadot.s * u.rad / u.s
                    ).to_value(u.deg / u.yr)
                outmodel.GAMMA.frozen = model.PB.frozen or model.M2.frozen
                outmodel.OMDOT.frozen = (
                    model.PB.frozen or model.M2.frozen or model.ECC.frozen
                )
                outmodel.PBDOT.frozen = (
                    model.PB.frozen or model.M2.frozen or model.ECC.frozen
                )
                if output != "BT":
                    outmodel.DR.value = Dr.n
                    if Dr.s > 0:
                        outmodel.DR.uncertainty_value = Dr.s
                    outmodel.DTH.value = Dth.n
                    if Dth.s > 0:
                        outmodel.DTH.uncertainty_value = Dth.s
                    outmodel.DR.frozen = model.PB.frozen or model.M2.frozen
                    outmodel.DTH.frozen = model.PB.frozen or model.M2.frozen

                    if output == "DDS":
                        shapmax = -umath.log(1 - s)
                        outmodel.SHAPMAX.value = shapmax.n
                        if shapmax.s > 0:
                            outmodel.SHAPMAX.uncertainty_value = shapmax.s
                        outmodel.SHAPMAX.frozen = (
                            model.PB.frozen
                            or model.M2.frozen
                            or model.ECC.frozen
                            or model.A1.frozen
                        )
                    elif output == "DDH":
                        m2 = model.M2.as_ufloat(u.Msun)
                        cbar = umath.sqrt(1 - s**2)
                        stigma = s / (1 + cbar)
                        h3 = Tsun.value * m2 * stigma**3
                        outmodel.STIGMA.quantity = stigma.n
                        outmodel.H3.value = h3.n
                        if stigma.u > 0:
                            outmodel.STIGMA.uncertainty_value = stigma.u
                        if h3.u > 0:
                            outmodel.H3.uncertainty_value = h3.u
                        outmodel.STIGMA.frozen = (
                            model.PB.frozen
                            or model.M2.frozen
                            or model.ECC.frozen
                            or model.A1.frozen
                        )
                        outmodel.H3.frozen = (
                            model.PB.frozen
                            or model.M2.frozen
                            or model.ECC.frozen
                            or model.A1.frozen
                        )

                    elif output == "DDK":
                        kin = umath.asin(s)
                        outmodel.KIN.value = kin.n
                        if kin.s > 0:
                            outmodel.KIN.uncertainty_value = kin.s
                        outmodel.KIN.frozen = (
                            model.PB.frozen
                            or model.M2.frozen
                            or model.ECC.frozen
                            or model.A1.frozen
                        )
                        log.warning(
                            f"Setting KIN={outmodel.KIN}: check that the sign is correct"
                        )
                    else:
                        outmodel.SINI.value = s.n
                        if s.s > 0:
                            outmodel.SINI.uncertainty_value = s.s
                        outmodel.SINI.frozen = (
                            model.PB.frozen
                            or model.M2.frozen
                            or model.ECC.frozen
                            or model.A1.frozen
                        )

        elif output in ["ELL1", "ELL1H", "ELL1k"]:
            # (DD, DDH, DDGR, DDS, DDK, BT) -> (ELL1, ELL1H, ELL1k)
            # parameters not to copy
            badlist = ["BINARY", "ECC", "OM", "T0", "OMDOT", "EDOT", "GAMMA"]
            if binary_component.binary_model_name == "DDS":
                badlist += ["SHAPMAX", "SINI"]
            elif binary_component.binary_model_name == "DDH":
                badlist += ["M2", "SINI", "STIGMA", "H3"]
            elif binary_component.binary_model_name == "DDK":
                badlist += ["KIN", "KOM", "SINI"]
            if output == "ELL1":
                outmodel.add_component(BinaryELL1(), validate=False)
            elif output == "ELL1H":
                outmodel.add_component(BinaryELL1H(), validate=False)
                badlist += ["M2", "SINI"]
            elif output == "ELL1k":
                outmodel.add_component(BinaryELL1k(), validate=False)
                badlist += ["EPS1DOT", "EPS2DOT"]
                badlist.remove("OMDOT")
            _transfer_params(model, outmodel, badlist)
            (
                EPS1,
                EPS2,
                TASC,
                EPS1DOT,
                EPS2DOT,
                EPS1_unc,
                EPS2_unc,
                TASC_unc,
                EPS1DOT_unc,
                EPS2DOT_unc,
            ) = _to_ELL1(model)
            LNEDOT = None
            LNEDOT_unc = None
            if output == "ELL1k":
                if model.EDOT.quantity is not None and model.ECC.quantity is not None:
                    LNEDOT = model.EDOT.quantity / model.ECC.quantity
                    if (
                        model.EDOT.uncertainty is not None
                        and model.ECC.uncertainty is not None
                    ):
                        LNEDOT_unc = np.sqrt(
                            (model.EDOT.uncertainty / model.ECC.quantity) ** 2
                            + (
                                model.EDOT.quantity
                                * model.ECC.uncertainty
                                / model.ECC.quantity**2
                            )
                            ** 2
                        )
            outmodel.EPS1.quantity = EPS1
            outmodel.EPS2.quantity = EPS2
            outmodel.TASC.quantity = TASC
            outmodel.EPS1.uncertainty = EPS1_unc
            outmodel.EPS2.uncertainty = EPS2_unc
            outmodel.TASC.uncertainty = TASC_unc
            outmodel.EPS1.frozen = model.ECC.frozen or model.OM.frozen
            outmodel.EPS2.frozen = model.ECC.frozen or model.OM.frozen
            outmodel.TASC.frozen = (
                model.ECC.frozen
                or model.OM.frozen
                or model.PB.frozen
                or model.T0.frozen
            )
            if EPS1DOT is not None and output != "ELL1k":
                outmodel.EPS1DOT.quantity = EPS1DOT
                outmodel.EPS2DOT.quantity = EPS2DOT
                outmodel.EPS1DOT.frozen = model.EDOT.frozen or model.OM.frozen
                outmodel.EPS2DOT.frozen = model.EDOT.frozen or model.OM.frozen
                if EPS1DOT_unc is not None:
                    outmodel.EPS1DOT.uncertainty = EPS1DOT_unc
                    outmodel.EPS2DOT.uncertainty = EPS2DOT_unc
            if LNEDOT is not None and output == "ELL1k":
                outmodel.LNEDOT.quantity = LNEDOT
                outmodel.LNEDOT.frozen = model.EDOT.frozen
                if LNEDOT_unc is not None:
                    outmodel.LNEDOT.uncertainty = LNEDOT_unc
            if binary_component.binary_model_name == "DDS":
                if output != "ELL1H":
                    SINI, SINI_unc = _SHAPMAX_to_SINI(model)
                    outmodel.SINI.quantity = SINI
                    if SINI_unc is not None:
                        outmodel.SINI.uncertainty = SINI_unc
                    outmodel.SINI.frozen = model.SHAPMAX.frozen
            elif binary_component.binary_model_name == "DDH":
                if output != "ELL1H":
                    M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                    outmodel.SINI.quantity = SINI
                    outmodel.M2.quantity = M2
                    if SINI_unc is not None:
                        outmodel.SINI.uncertainty = SINI_unc
                    if M2_unc is not None:
                        outmodel.M2.uncertainty = M2_unc
                    outmodel.SINI.frozen = model.STIGMA.frozen
                    outmodel.M2.frozen = model.STIGMA.frozen or model.H3.frozen
            elif binary_component.binary_model_name == "DDK":
                if output != "ELL1H":
                    if model.KIN.quantity is not None:
                        outmodel.SINI.quantity = np.sin(model.KIN.quantity)
                        if model.KIN.uncertainty is not None:
                            outmodel.SINI.uncertainty = np.abs(
                                model.KIN.uncertainty * np.cos(model.KIN.quantity)
                            ).to(
                                u.dimensionless_unscaled,
                                equivalencies=u.dimensionless_angles(),
                            )
                        outmodel.SINI.frozen = model.KIN.frozen
                else:
                    tempmodel = convert_binary(model, "DD")
                    outmodel = convert_binary(tempmodel, output)
            if output == "ELL1H":
                if binary_component.binary_model_name in ["DDGR", "DDH", "DDK"]:
                    model = convert_binary(model, "DD")
                stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(
                    model
                )
                outmodel.NHARMS.value = NHARMS
                outmodel.H3.quantity = h3
                outmodel.H3.uncertainty = h3_unc
                outmodel.H3.frozen = model.M2.frozen or model.SINI.frozen
                if useSTIGMA:
                    # use STIGMA and H3
                    outmodel.STIGMA.quantity = stigma
                    outmodel.STIGMA.uncertainty = stigma_unc
                    outmodel.STIGMA.frozen = outmodel.H3.frozen
                else:
                    # use H4 and H3
                    outmodel.H4.quantity = h4
                    outmodel.H4.uncertainty = h4_unc
                    outmodel.H4.frozen = outmodel.H3.frozen

    if (
        output == "DDS"
        and binary_component.binary_model_name != "DDGR"
        and hasattr(model, "SINI")
    ):
        SHAPMAX, SHAPMAX_unc = _SINI_to_SHAPMAX(model)
        outmodel.SHAPMAX.quantity = SHAPMAX
        if SHAPMAX_unc is not None:
            outmodel.SHAPMAX.uncertainty = SHAPMAX_unc
        outmodel.SHAPMAX.frozen = model.SINI.frozen

    if output == "DDH":
        if binary_component.binary_model_name in ["DDGR", "DDK"]:
            model = convert_binary(model, "DD")
        if binary_component.binary_model_name == "ELL1H":
            model = convert_binary(model, "ELL1")
        stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(model)
        outmodel.H3.quantity = h3
        if h3_unc is not None:
            outmodel.H3.uncertainty = h3_unc
        outmodel.H3.frozen = model.M2.frozen or model.SINI.frozen
        outmodel.STIGMA.quantity = stigma
        if stigma_unc is not None:
            outmodel.STIGMA.uncertainty = stigma_unc
        outmodel.STIGMA.frozen = model.SINI.frozen

    if output == "DDK":
        outmodel.KOM.quantity = KOM
        if binary_component.binary_model_name != "DDGR":
            if hasattr(model, "SINI") and model.SINI.quantity is not None:
                outmodel.KIN.quantity = np.arcsin(model.SINI.quantity).to(
                    u.deg, equivalencies=u.dimensionless_angles()
                )
                if model.SINI.uncertainty is not None:
                    outmodel.KIN.uncertainty = (
                        model.SINI.uncertainty / np.sqrt(1 - model.SINI.quantity**2)
                    ).to(u.deg, equivalencies=u.dimensionless_angles())
                log.warning(
                    f"Setting KIN={outmodel.KIN} from SINI={model.SINI}: check that the sign is correct"
                )
                outmodel.KIN.frozen = model.SINI.frozen
    outmodel.validate()

    return outmodel
