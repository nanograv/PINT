"""

Potential issues:
* orbital frequency derivatives
* Does EPS1DOT/EPS2DOT imply OMDOT and vice versa?

"""

import copy
from typing import Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.time import Time
from loguru import logger as log
from uncertainties import ufloat, umath

import pint.models
from pint import Tsun
from pint.exceptions import TimingModelError
from pint.models.binary_bt import BinaryBT
from pint.models.binary_dd import BinaryDD, BinaryDDH, BinaryDDS
from pint.models.binary_ddk import BinaryDDK
from pint.models.binary_ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k
from pint.models.binary_ddr import (
    BinaryDDR,
    fw10_decode,
    fw10_encode,
    fw10_orbit_decode,
    fw10_orbit_encode,
)
from pint.models.parameter import funcParameter
from pint.models.stand_alone_psr_binaries.ddr_kepler import (
    orbital_phase,
    taylor_shift,
    value as ddr_value,
)
from pint.models.stand_alone_psr_binaries.DDR_model import q_at_tasc

# output types
# DDGR is not included as there is not a well-defined way to get a unique output
binary_types = ["DD", "DDK", "DDS", "DDH", "BT", "ELL1", "ELL1H", "ELL1k", "DDR"]


__all__ = ["convert_binary"]


def _M2SINI_to_orthometric(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _orthometric_to_M2SINI(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
    """Convert from orthometric (H3, H4, STIGMA) to standard Shapiro delay (M2, SINI)

    Inverts Eqns. 12, 20, 21 from Freire and Wex (2010)
    Also propagates uncertainties if present

    If STIGMA is present will use that.  Otherwise will use H4.
    If neither is present, will leave M2, SINI unset.

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
    elif h4 is not None:
        # FW10 Eqn. 25, 26
        sini = 2 * h3 * h4 / (h3**2 + h4**2)
        m2 = h3**4 / h4**3 / Tsun.value
    else:
        return None, None, None, None

    m2_unc = m2.s * u.Msun if m2.s > 0 else None
    sini_unc = sini.s if sini.s > 0 else None

    return m2.n * u.Msun, sini.n, m2_unc, sini_unc


def _SINI_to_SHAPMAX(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _SHAPMAX_to_SINI(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _from_ELL1(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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
        (
            (omdot.s * u.rad / u.s).to(u.deg / u.yr)
            if (omdot is not None and omdot.s > 0)
            else None
        ),
    )


def _to_ELL1(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _ELL1_to_ELL1k(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _ELL1k_to_ELL1(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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


def _DDGR_to_PK(model: pint.models.TimingModel) -> Tuple[u.Quantity]:
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
    fe = (1 + (73.0 / 24) * ecc**2 + (37.0 / 96) * ecc**4) / (1 - ecc**2) ** (7.0 / 2)
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


def _transfer_params(
    inmodel: pint.models.TimingModel,
    outmodel: pint.models.TimingModel,
    badlist: list = [],
) -> None:
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
            getattr(outmodel.components[outbinary_component_name], p)._parent = (
                outmodel.components[outbinary_component_name]
            )
            if p not in outmodel.components[outbinary_component_name].params:
                outmodel.components[outbinary_component_name].params.append(p)


_DDR_UNSUPPORTED_SOURCES = ("ELL1k", "DDGR", "T2")
_H3_ONLY_WD_MASS = np.longdouble("0.2")
_DDR_BINARY_ONLY = (
    "BINARY",
    "SINI",
    "M2",
    "COSI",
    "ECC",
    "OM",
    "T0",
    "EDOT",
    "KIN",
    "H3",
    "H4",
    "STIGMA",
    "NHARMS",
    "EPS1DOT",
    "EPS2DOT",
    "LNEDOT",
    "SHAPMAX",
    "GAMMA",
    "OMDOT",
    "GGAMMA",
    "XPBDOT",
    "PBDOT",
    "KOM",
    "TGEO",
    "DDRPK",
    "DDRPBDOT",
    "DDRGEO",
    "DDRKINE",
    "DR",
    "DTH",
)


def _ddr_report(source):
    return {
        "source": source,
        "chart": "",
        "coordinates": [],
        "shapiro": [],
        "orbital_law": [],
        "pk": [],
        "geometry": [],
        "orientation": [],
        "epochs": "",
        "secular": ["none"],
        "named_conventions": [],
        "pk_replaced": {},
        "free_params": {"source": [], "target": []},
        "fit_space": [],
        "uncertainty_propagation": "copied",
        "constant_offset_s": 0.0,
        "notes": "",
    }


def _sini_to_cosi_prograde(sini):
    s = np.longdouble(sini)
    if s < 0 or s > 1:
        raise TimingModelError("DDR conversion requires 0 <= SINI <= 1")
    return np.sqrt((1 - s) * (1 + s))


def _has_px(model):
    return hasattr(model, "PX") and model.PX.quantity is not None and model.PX.value > 0


def _set_ddr_modes(
    outmodel,
    report,
    *,
    ddrpk,
    ddrpbdot,
    ddrkine,
    ddrgeo,
    has_kom,
    has_px,
    chart,
):
    if ddrpbdot is None:
        ddrpbdot = "absorb_gw"
    ddrpbdot = str(ddrpbdot).strip().lower()
    if ddrpbdot not in ("absorb_gw", "kinematic"):
        raise TimingModelError("DDRPBDOT must be 'absorb_gw' or 'kinematic'")
    if ddrgeo is None:
        ddrgeo = bool(has_kom and has_px)
    if ddrgeo and not has_px:
        raise TimingModelError("DDRGEO Y requires PX>0")
    if ddrgeo and not has_kom:
        raise TimingModelError("DDRGEO Y requires KOM")
    if ddrkine and not has_px:
        raise TimingModelError("DDRKINE Y requires PX>0")
    if ddrgeo and chart == "pb":
        ddrkine = True
    if chart == "fbx" and (ddrkine or ddrpbdot != "absorb_gw"):
        raise TimingModelError("DDR FBX chart requires absorb_gw with DDRKINE N")
    if not has_px:
        ddrgeo = False
        ddrkine = False
        report["geometry"].append("off")
        report["notes"] = (
            report["notes"] + " missing PX disables DDRGEO/DDRKINE"
        ).strip()
    outmodel.DDRPK.value = bool(ddrpk)
    outmodel.DDRPBDOT.value = ddrpbdot
    outmodel.DDRGEO.value = bool(ddrgeo)
    outmodel.DDRKINE.value = bool(ddrkine)
    if ddrpk:
        report["pk"].append("replaced_by_ddrpk")
    else:
        report["pk"].append("source_kept")
    if ddrgeo:
        report["geometry"].append("ddr_projector")
    elif "off" not in report["geometry"]:
        report["geometry"].append("off")


def _ell1h_shapiro_mode(model, override=None):
    if override is not None:
        mode = override
    elif "BinaryELL1H" in model.components:
        mode = getattr(model.components["BinaryELL1H"], "ell1h_shapiro", "full")
    else:
        mode = (getattr(model, "meta", {}) or {}).get("ell1h_shapiro", "full")
    mode = str(mode).strip().lower()
    if mode not in ("full", "absorbed"):
        raise TimingModelError("ell1h_shapiro must be 'full' or 'absorbed'")
    return mode


def _r_s_from_m2(m2):
    return np.longdouble(Tsun.to_value(u.s)) * np.longdouble(m2.to_value(u.Msun))


def _stigma_from_orthometric(model, stigma):
    if model.STIGMA.quantity is not None:
        return np.longdouble(model.STIGMA.quantity.to_value(u.dimensionless_unscaled))
    if (
        hasattr(model, "H4")
        and model.H4.quantity is not None
        and model.H3.quantity is not None
    ):
        return np.longdouble((model.H4.quantity / model.H3.quantity).decompose().value)
    if stigma is not None:
        return np.longdouble(stigma.to_value(u.dimensionless_unscaled))
    raise TimingModelError(f"{model.BINARY.value} H3-only conversion requires stigma=")


_DAY_S = np.longdouble(86400)
_TWO_PI_LD = np.longdouble(2) * np.longdouble(np.pi)


def _quantity_value(par, unit, default=0):
    if par is None or par.quantity is None:
        return np.longdouble(default)
    return np.longdouble(par.quantity.to_value(unit))


def _source_fbx(model):
    comp = model.components[
        next(name for name in model.components if name.startswith("Binary"))
    ]
    mapping = comp._fbx_mapping()
    valued = {
        index: name
        for index, name in mapping.items()
        if getattr(comp, name).quantity is not None
    }
    if not valued:
        return None
    maximum = max(valued)
    return [
        np.longdouble(
            getattr(comp, mapping[index]).quantity.to_value(u.s ** (-(index + 1)))
        )
        for index in range(maximum + 1)
    ]


def _phase_chart(model):
    """Return ``(chart, f_j)`` in SI units and the independent source names."""
    coeffs = _source_fbx(model)
    if coeffs is not None:
        return "fbx", coeffs, [f"FB{j}" for j in range(len(coeffs))]
    pb_s = _quantity_value(model.PB, u.s)
    if not np.isfinite(pb_s) or pb_s <= 0:
        raise TimingModelError("DDR conversion requires positive PB or FB0")
    p = _quantity_value(getattr(model, "PBDOT", None), u.s / u.s)
    return "pb", [1 / pb_s, -p / pb_s**2], ["PB", "PBDOT"]


def _shift_phase_coeffs(coeffs, delta_s):
    """Taylor re-reference all orbital-frequency coefficients (§11.1)."""
    shifted = [
        np.longdouble(np.asarray(c).reshape(-1)[0])
        for c in np.atleast_1d(taylor_shift(coeffs, np.longdouble(delta_s)))
    ]
    if not np.isfinite(shifted[0]) or shifted[0] <= 0:
        raise TimingModelError("DDR phase transfer produced non-positive frequency")
    return shifted


def _phase_cycles(coeffs, delta_s):
    lam, _, _ = orbital_phase(np.longdouble(delta_s), coeffs, check_slope=False)
    cycles = ddr_value(lam) / (np.longdouble(2) * np.pi)
    return np.longdouble(np.asarray(cycles).reshape(-1)[0])


def _phase_frequency(coeffs, delta_s):
    _, lamdot, _ = orbital_phase(np.longdouble(delta_s), coeffs, check_slope=False)
    freq = ddr_value(lamdot) / (np.longdouble(2) * np.pi)
    return np.longdouble(np.asarray(freq).reshape(-1)[0])


def _monotone_phase_root(coeffs, target_cycles, estimate_s):
    """Safeguarded long-double Newton/bisection for a phase-polynomial root."""
    target_cycles = np.longdouble(target_cycles)
    estimate_s = np.longdouble(estimate_s)

    def fun(x):
        return _phase_cycles(coeffs, x) - target_cycles

    scale = abs(1 / np.longdouble(coeffs[0]))
    lo = estimate_s - scale
    hi = estimate_s + scale
    for _ in range(24):
        if fun(lo) <= 0 <= fun(hi):
            break
        scale *= 2
        lo = estimate_s - scale
        hi = estimate_s + scale
    else:
        raise TimingModelError("Could not bracket orbital epoch transfer")

    x = min(max(estimate_s, lo), hi)
    tolerance = np.longdouble("2e-19") * max(abs(lo), abs(hi), np.longdouble(1))
    for _ in range(160):
        fx = fun(x)
        if abs(fx) <= np.longdouble("2e-19") or hi - lo <= tolerance:
            return x
        if fx < 0:
            lo = x
        else:
            hi = x
        slope = _phase_frequency(coeffs, x)
        trial = x - fx / slope if slope > 0 else (lo + hi) / 2
        if not (lo < trial < hi):
            trial = (lo + hi) / 2
        x = trial
    raise TimingModelError("Orbital epoch transfer did not converge")


def _qstar(h, k):
    return np.longdouble(np.asarray(ddr_value(q_at_tasc(h, k))).reshape(()))


def _dd_orientation_at_tasc(ecc, om_dd, kappa):
    """Invert ``OM_DD=(1+κ)ω*-κq*(ω*)`` without reducing the root."""
    ecc = np.longdouble(ecc)
    om_dd = np.longdouble(om_dd) % _TWO_PI_LD
    kappa = np.longdouble(kappa)
    anomaly_max = np.sqrt(1 + ecc) / (1 - ecc) ** np.longdouble("1.5")
    anomaly_min = np.sqrt(1 - ecc) / (1 + ecc) ** np.longdouble("1.5")
    q_derivative_bound = max(anomaly_max - 1, 1 - anomaly_min)
    if abs(kappa) * (1 + q_derivative_bound) >= 1:
        raise TimingModelError("DD → DDR OMDOT is outside the contraction domain")

    def fun(omega):
        return (
            (1 + kappa) * omega
            - kappa * _qstar(ecc * np.sin(omega), ecc * np.cos(omega))
            - om_dd
        )

    seed = om_dd / (1 + kappa)
    width = _TWO_PI_LD * (1 + abs(kappa) + ecc)
    lo, hi = seed - width, seed + width
    while not (fun(lo) <= 0 <= fun(hi)):
        width *= 2
        lo, hi = seed - width, seed + width
        if width > 16 * _TWO_PI_LD:
            raise TimingModelError("Could not bracket DD orientation transfer")
    omega = seed
    for _ in range(120):
        fomega = fun(omega)
        if abs(fomega) <= np.longdouble("2e-19"):
            return omega
        if fomega < 0:
            lo = omega
        else:
            hi = omega
        step = np.longdouble("1e-7")
        derivative = (fun(omega + step) - fun(omega - step)) / (2 * step)
        trial = omega - fomega / derivative
        if not (lo < trial < hi):
            trial = (lo + hi) / 2
        omega = trial
    raise TimingModelError("DD orientation transfer did not converge")


def _set_phase_chart(component, chart, coeffs, source=None):
    if chart == "fbx":
        for j, coefficient in enumerate(coeffs):
            target = component._add_or_get_fbx(j)
            target.quantity = coefficient * u.s ** (-(j + 1))
            if source is not None and hasattr(source, f"FB{j}"):
                original = getattr(source, f"FB{j}")
                target.frozen = original.frozen
                target.uncertainty = copy.deepcopy(original.uncertainty)
        return
    f0, f1 = (np.longdouble(v) for v in coeffs[:2])
    component.PB.quantity = (1 / f0) * u.s
    if not (
        hasattr(component, "DDRPBDOT")
        and str(component.DDRPBDOT.value).strip().lower() == "kinematic"
    ):
        component.PBDOT.quantity = (-f1 / f0**2) * u.dimensionless_unscaled


def _copy_parameter_state(target, source):
    target.frozen = source.frozen
    target.uncertainty = copy.deepcopy(source.uncertainty)


def _set_state_from_dependencies(target, dependencies):
    dependencies = [p for p in dependencies if p is not None]
    target.frozen = any(p.frozen for p in dependencies)


def _nonzero(model, name):
    par = getattr(model, name, None)
    return (
        par is not None and par.quantity is not None and np.longdouble(par.value) != 0
    )


def _active_parameter(model, name):
    par = getattr(model, name, None)
    return (
        par is not None
        and par.quantity is not None
        and (np.longdouble(par.value) != 0 or not par.frozen)
    )


def _check_ddr_import_refusals(model, *, drop_edot=False):
    for name in (
        "EDOT",
        "EPS1DOT",
        "EPS2DOT",
        "LNEDOT",
        "DR",
        "DTH",
        "XOMDOT",
        "A0",
        "B0",
    ):
        if name == "EDOT" and drop_edot:
            continue
        if _active_parameter(model, name):
            raise TimingModelError(
                f"{model.BINARY.value} → DDR cannot transfer {name} (secular: unsupported)"
            )
    component = model.components[
        next(name for name in model.components if name.startswith("Binary"))
    ]
    for prefix in ("ORBWAVEC", "ORBWAVES"):
        for name in component.get_prefix_mapping_component(prefix).values():
            if getattr(component, name).quantity is not None:
                raise TimingModelError(
                    f"{model.BINARY.value} → DDR cannot transfer {name}"
                )
    for name in ("ORBWAVE_OM", "ORBWAVE_EPOCH"):
        if (
            getattr(model, name, None) is not None
            and getattr(model, name).quantity is not None
        ):
            raise TimingModelError(f"{model.BINARY.value} → DDR cannot transfer {name}")


def _positive_h3(model):
    """True when ``H3`` is a usable positive amplitude.

    A missing ``H3`` is absent (ELL1-like). A present but non-positive
    value is not Shapiro-off: it is a conversion error.
    """
    par = getattr(model, "H3", None)
    if par is None or par.quantity is None:
        return False
    value = np.longdouble(par.quantity.to_value(u.s))
    if not np.isfinite(value):
        raise TimingModelError(f"{model.BINARY.value} → DDR requires a finite H3")
    if value < 0:
        raise TimingModelError(f"{model.BINARY.value} → DDR requires H3 > 0")
    return value > 0


def _h3_only_wd_mass(h3, m2_msun=_H3_ONLY_WD_MASS):
    """Companion-mass prior for H3-only ELL1H/DDH: freeze M2, infer ς and c."""
    h3_s = np.longdouble(h3.to_value(u.s))
    tsun = np.longdouble(Tsun.to_value(u.s))
    radius = tsun * np.longdouble(m2_msun)
    if h3_s > radius:
        m2_msun = h3_s / tsun
        sig = np.longdouble(1)
        cosi = np.longdouble(0)
    else:
        sig = (h3_s / radius) ** (np.longdouble(1) / np.longdouble(3))
        cosi = (1 - sig * sig) / (1 + sig * sig)
    return m2_msun * u.Msun, cosi, sig


def _absent_shapiro():
    return 0 * u.Msun, np.longdouble("0.5")


def _m2_sini_shapiro(model, *, cosi=None, report):
    """Return ``(m2, cosi, absent)`` from a source ``(M2, SINI)`` pair.

    An incomplete pair is Shapiro-off: keep ``M2=0`` and the default
    ``COSI``. Restoring a lone nonzero ``M2`` would turn Shapiro on.
    """
    m2_par = getattr(model, "M2", None)
    sini_par = getattr(model, "SINI", None)
    missing_m2 = m2_par is None or m2_par.quantity is None
    missing_sini = sini_par is None or sini_par.quantity is None
    if missing_m2 or missing_sini:
        m2, cosi_value = _absent_shapiro()
        report["orientation"].append("unused_m2_zero")
        report["shapiro"].append("absent_off")
        return m2, cosi_value, True
    m2 = m2_par.quantity
    if cosi is None:
        cosi_value = _sini_to_cosi_prograde(sini_par.value)
        report["orientation"].append("assumed_prograde")
    else:
        cosi_value = np.longdouble(u.Quantity(cosi).to_value(u.dimensionless_unscaled))
        report["orientation"].append("declared_cosi")
    report["shapiro"].append("unchanged")
    return m2, cosi_value, False


def _orthometric_source(model, *, cosi, stigma, report, h3_only="wd_mass"):
    h3 = model.H3.quantity
    if h3 is None or not np.isfinite(np.longdouble(h3.to_value(u.s))):
        raise TimingModelError(f"{model.BINARY.value} → DDR requires a finite H3")
    if np.longdouble(h3.to_value(u.s)) <= 0:
        raise TimingModelError(f"{model.BINARY.value} → DDR requires H3 > 0")
    if cosi is not None:
        c = np.longdouble(u.Quantity(cosi).to_value(u.dimensionless_unscaled))
        if abs(c) >= 1:
            raise TimingModelError("cosi= must satisfy |cosi| < 1")
        s = np.sqrt((1 - c) * (1 + c))
        sig = s / (1 + abs(c))
        report["orientation"].append("declared_cosi")
    elif stigma is not None:
        sig = np.longdouble(u.Quantity(stigma).to_value(u.dimensionless_unscaled))
        c = np.longdouble(fw10_decode(h3, sig * u.dimensionless_unscaled)[1].value)
        report["orientation"].append("assumed_prograde")
    elif model.STIGMA.quantity is not None:
        sig = np.longdouble(model.STIGMA.value)
        c = np.longdouble(fw10_decode(h3, sig * u.dimensionless_unscaled)[1].value)
        report["orientation"].append("assumed_prograde")
    elif hasattr(model, "H4") and model.H4.quantity is not None:
        sig = np.longdouble((model.H4.quantity / h3).decompose().value)
        c = np.longdouble(fw10_decode(h3, sig * u.dimensionless_unscaled)[1].value)
        report["orientation"].append("assumed_prograde")
    elif h3_only == "require":
        raise TimingModelError(
            f"{model.BINARY.value} H3-only conversion requires cosi= or stigma="
        )
    else:
        m2, c, sig = _h3_only_wd_mass(h3)
        report["orientation"].append("h3_only_prior")
        report["notes"] = (
            (report.get("notes") or "")
            + f"H3-only: companion-mass prior M2={float(m2.to_value(u.Msun)):.4g} Msun"
        ).strip()
        return m2, c, sig
    if not (0 < sig <= 1):
        raise TimingModelError("FW10 decode requires 0 < STIGMA <= 1")
    m2, _ = fw10_decode(h3, sig * u.dimensionless_unscaled)
    return m2, c, sig


def _source_secular(model, chart):
    names = [
        name
        for name in ("OMDOT", "PBDOT", "XPBDOT", "A1DOT", "EDOT", "EPS1DOT", "EPS2DOT")
        if _nonzero(model, name)
    ]
    if chart == "fbx":
        names.extend(
            f"FB{j}"
            for j in range(1, len(_source_fbx(model) or []))
            if _nonzero(model, f"FB{j}")
        )
    return names


def _finalize_ddr_report(report, source_model, target_model, *, shrunk=(), released=()):
    report["free_params"] = {
        "source": list(source_model.free_params),
        "target": list(target_model.free_params),
    }
    if released:
        report["fit_space"].append("constraint_released: " + ", ".join(released))
    elif shrunk:
        report["fit_space"].append("shrunk: " + ", ".join(shrunk))
    else:
        report["fit_space"].append("preserved")
    if any(
        getattr(source_model, name).uncertainty is not None
        for name in source_model.params
        if hasattr(source_model, name)
    ):
        report["uncertainty_propagation"] = "diagonal_input"


def _ddk_reference_shift(model, t0_d, kom_quantity=None):
    """Linearized DDK reference shift from DDR ``TGEO`` to emitted ``T0``."""
    if model.TGEO.quantity is None:
        raise TimingModelError("DDR → DDK requires TGEO")
    kom_quantity = model.KOM.quantity if kom_quantity is None else kom_quantity
    if kom_quantity is None:
        raise TimingModelError("DDR → DDK requires KOM")
    dt_s = (np.longdouble(t0_d) - np.longdouble(model.TGEO.value)) * _DAY_S
    kom = np.longdouble(kom_quantity.to_value(u.rad))
    kin = np.arccos(np.longdouble(model.COSI.value))
    component = model.components["BinaryDDR"]
    triad = component._tgeo_triad()
    mu_i = np.longdouble(triad["mu_I"])
    mu_j = np.longdouble(triad["mu_J"])
    delta_i = (-mu_i * np.sin(kom) + mu_j * np.cos(kom)) * dt_s
    delta_omega = (mu_i * np.cos(kom) + mu_j * np.sin(kom)) / np.sin(kin) * dt_s
    delta_a1 = np.longdouble(model.A1.quantity.to_value(u.lsec)) / np.tan(kin) * delta_i
    return delta_i, delta_omega, delta_a1


def _covariance_array(covariance, model):
    if hasattr(covariance, "matrix"):
        labels = [
            name
            for name, (start, stop, _unit) in covariance.get_axis_labels(0)
            if stop - start == 1
        ]
        indices = [covariance.get_label(name, axis=0)[0][2] for name in labels]
        matrix = np.asarray(covariance.matrix, dtype=float)[np.ix_(indices, indices)]
        return labels, matrix
    matrix = np.asarray(covariance, dtype=float)
    labels = list(model.free_params)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            "A bare covariance array must match model.free_params ordering"
        )
    return labels, matrix


def _propagate_conversion_covariance(source, target, covariance, converter):
    """Numerically form the conversion Jacobian in native parameter units."""
    names, cov = _covariance_array(covariance, source)
    usable = [
        (index, name)
        for index, name in enumerate(names)
        if hasattr(source, name)
        and getattr(source, name).quantity is not None
        and not isinstance(getattr(source, name).quantity, (str, bool))
    ]
    component = target.components[
        next(name for name in target.components if name.startswith("Binary"))
    ]
    target_names = [
        name
        for name in component.params
        if hasattr(target, name)
        and getattr(target, name).quantity is not None
        and not isinstance(getattr(target, name), funcParameter)
        and not isinstance(getattr(target, name).quantity, (str, bool))
    ]
    jacobian = np.zeros((len(target_names), len(names)), dtype=float)
    for column, name in usable:
        par = getattr(source, name)
        nominal = np.longdouble(par.value)
        sigma = np.sqrt(max(cov[column, column], 0))
        step = np.longdouble(
            sigma * 1e-3
            if sigma > 0
            else np.sqrt(np.finfo(float).eps) * max(abs(float(nominal)), 1.0)
        )
        plus = copy.deepcopy(source)
        minus = copy.deepcopy(source)
        getattr(plus, name).value = nominal + step
        getattr(minus, name).value = nominal - step
        try:
            yplus = converter(plus)
            yminus = converter(minus)
        except Exception:
            continue
        for row, target_name in enumerate(target_names):
            yp = np.longdouble(getattr(yplus, target_name).value)
            ym = np.longdouble(getattr(yminus, target_name).value)
            jacobian[row, column] = float((yp - ym) / (2 * step))
    target_cov = jacobian @ cov @ jacobian.T
    for row, name in enumerate(target_names):
        variance = target_cov[row, row]
        if np.isfinite(variance) and variance > 0:
            getattr(target, name).uncertainty_value = np.sqrt(variance)
    return target


def _convert_to_ddr(
    model,
    *,
    ddrpk,
    ddrpbdot,
    ddrkine,
    ddrgeo,
    cosi,
    stigma,
    rescale_pb,
    KOM,
    drop_edot=False,
    h3_only="wd_mass",
):
    source = model.BINARY.value
    if h3_only not in ("wd_mass", "require"):
        raise TimingModelError("h3_only must be 'wd_mass' or 'require'")
    if source in _DDR_UNSUPPORTED_SOURCES:
        raise TimingModelError(f"{source} → DDR is not supported")
    _check_ddr_import_refusals(model, drop_edot=drop_edot)
    if source not in ("ELL1", "ELL1H", "DD", "DDS", "DDH", "DDK", "BT"):
        raise TimingModelError(f"Do not know how to convert from {source} to DDR")

    report = _ddr_report(source)
    chart, source_coeffs, _phase_names = _phase_chart(model)
    report["chart"] = chart
    secular_names = _source_secular(model, chart)
    if secular_names:
        report["secular"] = ["transferred"]

    if source == "DDK" and not _has_px(model):
        raise TimingModelError("DDK → DDR geometry requires PX>0")
    if source == "DDK" and ddrgeo is False:
        raise TimingModelError("DDK → DDR requires DDRGEO Y")

    supplied_kom = KOM is not None
    source_kom = source == "DDK" and model.KOM.quantity is not None
    has_kom = source_kom or supplied_kom
    outmodel = copy.deepcopy(model)
    bname = [x for x in outmodel.components if x.startswith("Binary")][0]
    outmodel.remove_component(bname)
    outmodel.BINARY.value = "DDR"

    comp = BinaryDDR()
    _set_ddr_modes(
        comp,
        report,
        ddrpk=ddrpk,
        ddrpbdot=ddrpbdot,
        ddrkine=ddrkine,
        ddrgeo=ddrgeo,
        has_kom=has_kom,
        has_px=_has_px(model),
        chart=chart,
    )
    outmodel.add_component(comp, setup=False, validate=False)

    coeffs = list(source_coeffs)
    delta_s = np.longdouble(0)
    shrink = []
    if source in ("ELL1", "ELL1H"):
        report["coordinates"].append("gauge_transfer")
        report["orbital_law"].append("ell1_series_to_dd")
        if model.TASC.quantity is None:
            raise TimingModelError("ELL1 → DDR requires TASC")
        x = _quantity_value(model.A1, u.lsec)
        h = np.longdouble(model.EPS1.value)
        k = np.longdouble(model.EPS2.value)
        tasc = np.longdouble(model.TASC.value)
        gauge_constant_s = -np.longdouble("1.5") * x * h

        m2 = None
        h3_only_prior = False
        shapiro_absent = False
        treat_as_ell1 = source == "ELL1" or not _positive_h3(model)
        if source == "ELL1H" and treat_as_ell1:
            report["shapiro"].append("ell1h_without_h3")
        if source == "ELL1H" and not treat_as_ell1:
            m2, cosi_value, sig = _orthometric_source(
                model,
                cosi=cosi,
                stigma=stigma,
                report=report,
                h3_only=h3_only,
            )
            h3_only_prior = "h3_only_prior" in report["orientation"]
            mode = _ell1h_shapiro_mode(model)
            if mode == "absorbed":
                if _nonzero(model, "A1DOT"):
                    raise TimingModelError(
                        "ELL1H absorbed → DDR cannot transfer A1DOT (FW map is dot-free)"
                    )
                x, h, k, tasc = fw10_orbit_decode(
                    x,
                    h,
                    k,
                    tasc,
                    1 / source_coeffs[0],
                    _r_s_from_m2(m2),
                    sig,
                )
                gauge_constant_s = -np.longdouble("1.5") * x * h
                report["shapiro"].append("full_from_absorbed")
                report["epochs"] = "TASC from the FW absorbed decode"
            else:
                tasc += np.longdouble("1.5") * x * h / _DAY_S
                report["shapiro"].append("full_from_harmonics")
        else:
            m2, cosi_value, shapiro_absent = _m2_sini_shapiro(
                model, cosi=cosi, report=report
            )
            tasc += np.longdouble("1.5") * x * h / _DAY_S

        outmodel.TASC.value = tasc
        outmodel.EPS1.value = h
        outmodel.EPS2.value = k
        outmodel.A1.quantity = x * u.lsec
        outmodel.M2.quantity = m2
        outmodel.COSI.value = cosi_value
        report["constant_offset_s"] = float(gauge_constant_s)
        report["epochs"] = report["epochs"] or "TASC shifted by the ELL1 Roemer gauge"
        _copy_parameter_state(outmodel.A1, model.A1)
        _copy_parameter_state(outmodel.EPS1, model.EPS1)
        _copy_parameter_state(outmodel.EPS2, model.EPS2)
        _copy_parameter_state(outmodel.TASC, model.TASC)
        if treat_as_ell1:
            if shapiro_absent:
                outmodel.M2.frozen = True
                outmodel.COSI.frozen = True
            else:
                _copy_parameter_state(outmodel.M2, model.M2)
                _set_state_from_dependencies(outmodel.COSI, [model.SINI])
        else:
            deps = [model.H3]
            if cosi is None and not h3_only_prior:
                if model.STIGMA.quantity is not None:
                    deps.append(model.STIGMA)
                elif hasattr(model, "H4") and model.H4.quantity is not None:
                    deps.append(model.H4)
            if h3_only_prior:
                outmodel.M2.frozen = True
                outmodel.COSI.frozen = model.H3.frozen
                shrink.append("M2")
            else:
                _set_state_from_dependencies(outmodel.M2, deps)
                _set_state_from_dependencies(outmodel.COSI, deps)
    elif source in ("DD", "DDS", "DDH", "DDK", "BT"):
        report["coordinates"].append("exact")
        ecc = np.longdouble(model.ECC.value)
        om_dd = np.longdouble(model.OM.quantity.to_value(u.rad))
        omdot = _quantity_value(getattr(model, "OMDOT", None), u.rad / u.s)
        kappa = omdot / (_TWO_PI_LD * source_coeffs[0])
        omega = _dd_orientation_at_tasc(ecc, om_dd, kappa)
        delta_s = _monotone_phase_root(
            source_coeffs,
            -omega / _TWO_PI_LD,
            -omega / (_TWO_PI_LD * source_coeffs[0]),
        )
        coeffs = _shift_phase_coeffs(source_coeffs, delta_s)
        outmodel.TASC.value = np.longdouble(model.T0.value) + delta_s / _DAY_S
        outmodel.EPS1.value = ecc * np.sin(omega)
        outmodel.EPS2.value = ecc * np.cos(omega)
        outmodel.A1.quantity = (
            _quantity_value(model.A1, u.lsec)
            + _quantity_value(model.A1DOT, u.lsec / u.s) * delta_s
        ) * u.lsec

        shapiro_absent = False
        if source in ("DDH",):
            m2, cosi_value, _sig = _orthometric_source(
                model,
                cosi=cosi,
                stigma=stigma,
                report=report,
                h3_only=h3_only,
            )
            report["shapiro"].append("full_from_harmonics")
        elif source == "DDS":
            m2 = model.M2.quantity
            sini = 1 - np.exp(-np.longdouble(model.SHAPMAX.value))
            cosi_value = _sini_to_cosi_prograde(sini)
            report["orientation"].append("assumed_prograde")
            report["shapiro"].append("unchanged")
        elif source == "DDK":
            m2 = model.M2.quantity
            cosi_value = np.cos(model.KIN.quantity.to_value(u.rad))
            report["orientation"].append("from_kin")
            report["shapiro"].append("unchanged")
        elif source == "BT":
            m2, cosi_value = _absent_shapiro()
            shapiro_absent = True
            report["orbital_law"].append("bt_to_dd")
            report["shapiro"].append("absent_off")
            report["orientation"].append("unused_m2_zero")
        else:
            m2, cosi_value, shapiro_absent = _m2_sini_shapiro(model, report=report)
        outmodel.M2.quantity = m2
        outmodel.COSI.value = cosi_value

        _set_state_from_dependencies(outmodel.EPS1, [model.ECC, model.OM])
        _set_state_from_dependencies(outmodel.EPS2, [model.ECC, model.OM])
        outmodel.TASC.frozen = model.T0.frozen
        if model.ECC.uncertainty is not None or model.OM.uncertainty is not None:
            sigma_e = _quantity_value(model.ECC, u.dimensionless_unscaled) * 0
            if model.ECC.uncertainty is not None:
                sigma_e = np.longdouble(
                    model.ECC.uncertainty.to_value(u.dimensionless_unscaled)
                )
            sigma_om = np.longdouble(0)
            if model.OM.uncertainty is not None:
                sigma_om = np.longdouble(model.OM.uncertainty.to_value(u.rad))
            outmodel.EPS1.uncertainty_value = np.hypot(
                np.sin(omega) * sigma_e, ecc * np.cos(omega) * sigma_om
            )
            outmodel.EPS2.uncertainty_value = np.hypot(
                np.cos(omega) * sigma_e, ecc * np.sin(omega) * sigma_om
            )
        outmodel.TASC.uncertainty = copy.deepcopy(model.T0.uncertainty)
        _copy_parameter_state(outmodel.A1, model.A1)
        if source == "DDH":
            _set_state_from_dependencies(outmodel.M2, [model.H3, model.STIGMA])
            _set_state_from_dependencies(outmodel.COSI, [model.H3, model.STIGMA])
        elif source == "DDS":
            _copy_parameter_state(outmodel.M2, model.M2)
            _set_state_from_dependencies(outmodel.COSI, [model.SHAPMAX])
        elif source == "DDK":
            _copy_parameter_state(outmodel.M2, model.M2)
            _set_state_from_dependencies(outmodel.COSI, [model.KIN])
        else:
            if shapiro_absent:
                outmodel.M2.frozen = True
                outmodel.COSI.frozen = True
            else:
                _copy_parameter_state(outmodel.M2, model.M2)
                _set_state_from_dependencies(outmodel.COSI, [model.SINI])
                if model.SINI.uncertainty is not None and cosi_value != 0:
                    outmodel.COSI.uncertainty_value = abs(
                        np.longdouble(model.SINI.value) / cosi_value
                    ) * np.longdouble(
                        model.SINI.uncertainty.to_value(u.dimensionless_unscaled)
                    )
        if model.ECC.frozen != model.OM.frozen:
            shrink = ["EPS1", "EPS2"]

        if not comp.DDRPK.value:
            outmodel.OMDOT.quantity = (
                omdot * (coeffs[0] / source_coeffs[0]) * u.rad / u.s
            )
            if hasattr(model, "OMDOT") and model.OMDOT.quantity is not None:
                _copy_parameter_state(outmodel.OMDOT, model.OMDOT)
            gamma = _quantity_value(getattr(model, "GAMMA", None), u.s)
            if gamma == 0:
                outmodel.GGAMMA.quantity = 0 * u.s
            elif ecc > np.longdouble("1e-6"):
                outmodel.GGAMMA.quantity = (gamma / ecc) * u.s
            else:
                raise TimingModelError(
                    "DD → DDR cannot regularize nonzero GAMMA at ECC <= 1e-6"
                )
            if hasattr(model, "GAMMA") and model.GAMMA.quantity is not None:
                _set_state_from_dependencies(outmodel.GGAMMA, [model.GAMMA, model.ECC])

        report["epochs"] = (
            "TASC solved from the BT phase polynomial"
            if source == "BT"
            else "TASC solved from the DD phase polynomial"
        )
        if _quantity_value(getattr(model, "PBDOT", None), u.s / u.s) != 0:
            report["named_conventions"].append("dd_instantaneous_pb_in_k_and_nhat")
        if source == "DDK":
            outmodel.TGEO.quantity = model.T0.quantity
            report["epochs"] = "TGEO from DDK KIN epoch (T0)"
            report["geometry"] = ["from_ddk"]

    if model.A1DOT.quantity is not None:
        outmodel.A1DOT.quantity = model.A1DOT.quantity
        _copy_parameter_state(outmodel.A1DOT, model.A1DOT)
    _set_phase_chart(outmodel.components["BinaryDDR"], chart, coeffs, source=model)
    if chart == "pb":
        if hasattr(model, "PB") and not isinstance(model.PB, funcParameter):
            _copy_parameter_state(outmodel.PB, model.PB)
        if hasattr(model, "PBDOT") and model.PBDOT.quantity is not None:
            _copy_parameter_state(outmodel.PBDOT, model.PBDOT)

    if source_kom:
        outmodel.KOM.quantity = model.KOM.quantity
        _copy_parameter_state(outmodel.KOM, model.KOM)
    elif supplied_kom:
        outmodel.KOM.quantity = u.Quantity(KOM).to(u.deg)
        outmodel.KOM.frozen = True

    if rescale_pb and ddrpk:
        if source in ("DD", "DDS", "DDH", "DDK"):
            raise TimingModelError("rescale_pb is only defined for ELL1/ELL1H imports")
        report["orbital_law"].append("pb_azimuthal_to_anomalistic")
        from pint.models.stand_alone_psr_binaries.DDR_model import kappa_gr, pulsar_mass

        n = _TWO_PI_LD * coeffs[0]
        mp, s = pulsar_mass(
            n,
            np.longdouble(outmodel.A1.quantity.to_value(u.lsec)),
            np.longdouble(outmodel.M2.quantity.to_value(u.Msun)),
            np.longdouble(outmodel.COSI.value),
        )
        h = np.longdouble(outmodel.EPS1.value)
        k = np.longdouble(outmodel.EPS2.value)
        kap = ddr_value(
            kappa_gr(
                outmodel.A1.quantity.to_value(u.lsec),
                outmodel.M2.quantity.to_value(u.Msun),
                s,
                h * h + k * k,
            )
        )
        coeffs = [f / (1 + kap) for f in coeffs]
        _set_phase_chart(outmodel.components["BinaryDDR"], chart, coeffs)

    transferred_p = -coeffs[1] / coeffs[0] ** 2 if chart == "pb" else None
    if chart == "pb" and outmodel.DDRPBDOT.value == "kinematic":
        outmodel.XPBDOT.value = 0
        if hasattr(model, "PBDOT") and model.PBDOT.quantity is not None:
            outmodel.XPBDOT.frozen = model.PBDOT.frozen
            outmodel.XPBDOT.uncertainty = copy.deepcopy(model.PBDOT.uncertainty)
    outmodel.setup()
    if chart == "pb" and (
        outmodel.DDRPBDOT.value == "kinematic" or outmodel.DDRKINE.value
    ):
        if outmodel.DDRPBDOT.value == "kinematic":
            free = outmodel.XPBDOT
        else:
            free = outmodel.PBDOT
        initial_free = np.longdouble(free.value or 0)
        current = np.longdouble(outmodel.components["BinaryDDR"]._kinematic_p_total())
        free.value = initial_free + transferred_p - current
        report["secular"] = [item for item in report["secular"] if item != "none"] + [
            "pbdot_total_preserved"
        ]
        report["notes"] = (
            f"target total PBDOT={transferred_p:.18g}; "
            "physical contribution at zero free coordinate="
            f"{current - initial_free:.18g}"
        )
        outmodel.setup()
    outmodel.validate()

    if chart == "pb" and _nonzero(model, "PBDOT"):
        report["named_conventions"].append("reference_frequency_inverse_timing")
    if chart == "fbx" and any(_nonzero(model, f"FB{j}") for j in range(1, len(coeffs))):
        report["named_conventions"].append("reference_frequency_inverse_timing")
    if outmodel.DDRPK.value:
        for name in ("OMDOT", "GAMMA"):
            par = getattr(outmodel, name)
            report["pk_replaced"][name] = {
                "value": par.value,
                "uncertainty": par.uncertainty_value,
                "frozen": par.frozen,
            }
    if drop_edot and _active_parameter(model, "EDOT"):
        report["secular"] = [item for item in report["secular"] if item != "none"] + [
            "edot_dropped"
        ]
        if "EDOT" not in shrink:
            shrink.append("EDOT")
    _finalize_ddr_report(report, model, outmodel, shrunk=shrink)
    outmodel.binary_conversion_report = report
    return outmodel


def _convert_from_ddr(model, output, *, NHARMS, useSTIGMA, KOM, ell1h_shapiro="full"):
    if output in _DDR_UNSUPPORTED_SOURCES:
        raise TimingModelError(f"DDR → {output} is not supported")
    if output not in ("DD", "DDS", "DDH", "DDK", "ELL1", "ELL1H"):
        raise TimingModelError(f"DDR → {output} is not supported")
    report = _ddr_report("DDR")
    chart, coeffs, _phase_names = _phase_chart(model)
    report["chart"] = chart
    report["coordinates"].append("exact")
    report["pk"].append("unchanged")
    report["orientation"].append("none")
    if _source_secular(model, chart):
        report["secular"] = ["transferred"]
    if output in ("ELL1", "ELL1H") and (model.DDRPK.value or _nonzero(model, "OMDOT")):
        raise TimingModelError(f"DDR → {output} cannot transfer active OMDOT/DDRPK")
    if output in ("ELL1", "ELL1H") and _nonzero(model, "GGAMMA"):
        raise TimingModelError(f"DDR → {output} cannot transfer active GGAMMA")

    outmodel = copy.deepcopy(model)
    outmodel.remove_component("BinaryDDR")
    outmodel.BINARY.value = output
    if output == "DD":
        outmodel.add_component(BinaryDD(), validate=False)
        report["geometry"].append("dropped" if model.DDRGEO.value else "off")
    elif output == "DDS":
        outmodel.add_component(BinaryDDS(), validate=False)
        report["geometry"].append("dropped" if model.DDRGEO.value else "off")
    elif output == "DDH":
        outmodel.add_component(BinaryDDH(), validate=False)
        report["geometry"].append("dropped" if model.DDRGEO.value else "off")
        report["shapiro"].append("full_from_harmonics")
    elif output == "DDK":
        outmodel.add_component(BinaryDDK(), validate=False)
        report["geometry"].extend(
            ["to_ddk_linearized", "ddk_orbital_parallax"]
            if model.DDRGEO.value
            else ["off"]
        )
    elif output == "ELL1":
        outmodel.add_component(BinaryELL1(), validate=False)
        report["orbital_law"].append("ell1_series_to_dd")
        report["geometry"].append("dropped" if model.DDRGEO.value else "off")
    elif output == "ELL1H":
        outmodel.add_component(BinaryELL1H(), validate=False)
        mode = _ell1h_shapiro_mode(model, override=ell1h_shapiro)
        outmodel.components["BinaryELL1H"].ell1h_shapiro = mode
        report["shapiro"].append(
            "full_from_absorbed" if mode == "absorbed" else "full_from_harmonics"
        )
        report["geometry"].append("dropped" if model.DDRGEO.value else "off")

    target = outmodel.components[f"Binary{output}"]
    if model.A1DOT.quantity is not None:
        target.A1DOT.quantity = model.A1DOT.quantity
        _copy_parameter_state(target.A1DOT, model.A1DOT)

    released = []
    if output in ("DD", "DDS", "DDH", "DDK"):
        h = np.longdouble(model.EPS1.value)
        k = np.longdouble(model.EPS2.value)
        ecc = np.hypot(h, k)
        omega = np.arctan2(h, k) % _TWO_PI_LD
        delta_s = _monotone_phase_root(
            coeffs,
            omega / _TWO_PI_LD,
            omega / (_TWO_PI_LD * coeffs[0]),
        )
        shifted = _shift_phase_coeffs(coeffs, delta_s)
        t0_d = np.longdouble(model.TASC.value) + delta_s / _DAY_S
        omdot_ddr = _quantity_value(model.OMDOT, u.rad / u.s)
        kappa = omdot_ddr / (_TWO_PI_LD * coeffs[0])
        om_dd = ((1 + kappa) * omega - kappa * _qstar(h, k)) % _TWO_PI_LD
        x_t0 = (
            _quantity_value(model.A1, u.lsec)
            + _quantity_value(model.A1DOT, u.lsec / u.s) * delta_s
        )

        outmodel.T0.value = t0_d
        outmodel.OM.quantity = om_dd * u.rad
        outmodel.ECC.value = ecc
        outmodel.A1.quantity = x_t0 * u.lsec
        _set_phase_chart(target, chart, shifted, source=model)
        _set_state_from_dependencies(outmodel.ECC, [model.EPS1, model.EPS2])
        _set_state_from_dependencies(outmodel.OM, [model.EPS1, model.EPS2])
        outmodel.T0.frozen = model.TASC.frozen
        _copy_parameter_state(outmodel.A1, model.A1)
        if chart == "pb":
            _set_state_from_dependencies(outmodel.PB, [model.PB])
            if hasattr(outmodel, "PBDOT") and not isinstance(
                outmodel.PBDOT, funcParameter
            ):
                if isinstance(model.PBDOT, funcParameter):
                    outmodel.PBDOT.frozen = True
                else:
                    _set_state_from_dependencies(outmodel.PBDOT, [model.PBDOT])

        if output not in ("DDH",):
            outmodel.M2.quantity = model.M2.quantity
            _copy_parameter_state(outmodel.M2, model.M2)
        sini = np.longdouble(model.SINI.value)
        if output not in ("DDH", "DDK", "DDS"):
            outmodel.SINI.value = sini
            _set_state_from_dependencies(outmodel.SINI, [model.COSI])
        if output == "DDS":
            if abs(sini) >= 1:
                raise TimingModelError("DDR → DDS requires |SINI| < 1")
            outmodel.SHAPMAX.value = -np.log(1 - sini)
            _set_state_from_dependencies(outmodel.SHAPMAX, [model.COSI])
        if output == "DDH":
            h3, sig, _h4 = fw10_encode(model.M2.quantity, model.COSI.quantity)
            outmodel.H3.quantity = h3
            outmodel.STIGMA.quantity = sig
            _set_state_from_dependencies(outmodel.H3, [model.M2, model.COSI])
            _set_state_from_dependencies(outmodel.STIGMA, [model.COSI])
        if output == "DDK":
            kom = model.KOM.quantity if model.KOM.quantity is not None else KOM
            if kom is None:
                raise TimingModelError("DDR → DDK requires KOM or KOM=")
            outmodel.KOM.quantity = kom
            delta_i, delta_omega, delta_a1 = _ddk_reference_shift(model, t0_d, kom)
            outmodel.KIN.quantity = (
                np.arccos(np.longdouble(model.COSI.value)) + delta_i
            ) * u.rad
            outmodel.OM.quantity = (om_dd + delta_omega) * u.rad
            outmodel.A1.quantity = (x_t0 + delta_a1) * u.lsec
            _set_state_from_dependencies(outmodel.KIN, [model.COSI])
            _copy_parameter_state(outmodel.KOM, model.KOM)
            report["epochs"] = "KIN, OM, and A1 shifted from TGEO to emitted T0"

        if hasattr(outmodel, "OMDOT"):
            outmodel.OMDOT.quantity = kappa * _TWO_PI_LD * shifted[0] * u.rad / u.s
            if model.DDRPK.value:
                outmodel.OMDOT.frozen = True
                released.append("OMDOT")
            else:
                _copy_parameter_state(outmodel.OMDOT, model.OMDOT)
        if hasattr(outmodel, "GAMMA"):
            outmodel.GAMMA.quantity = (
                ecc * _quantity_value(model.GGAMMA, u.s) * u.s
                if not model.DDRPK.value
                else model.GAMMA.quantity
            )
            if model.DDRPK.value:
                outmodel.GAMMA.frozen = True
                released.append("GAMMA")
            else:
                _set_state_from_dependencies(
                    outmodel.GAMMA, [model.GGAMMA, model.EPS1, model.EPS2]
                )
        if chart == "pb" and isinstance(model.PBDOT, funcParameter):
            outmodel.PBDOT.quantity = (
                -shifted[1] / shifted[0] ** 2
            ) * u.dimensionless_unscaled
            outmodel.PBDOT.frozen = True
            released.append("PBDOT")
        if any(_nonzero(model, name) for name in ("OMDOT", "PBDOT", "A1DOT")):
            report["secular"] = ["transferred"]
        report["epochs"] = report["epochs"] or "T0 solved from the DDR phase polynomial"
    else:
        h = np.longdouble(model.EPS1.value)
        x = _quantity_value(model.A1, u.lsec)
        mode = (
            _ell1h_shapiro_mode(model, override=ell1h_shapiro)
            if output == "ELL1H"
            else "full"
        )
        if output == "ELL1H":
            h3, sig, h4 = fw10_encode(model.M2.quantity, model.COSI.quantity)
        if output == "ELL1H" and mode == "absorbed":
            if _nonzero(model, "A1DOT"):
                raise TimingModelError(
                    "DDR → ELL1H absorbed cannot transfer A1DOT (FW map is dot-free)"
                )
            target_x, target_h, target_k, target_tasc = fw10_orbit_encode(
                x,
                h,
                np.longdouble(model.EPS2.value),
                np.longdouble(model.TASC.value),
                1 / coeffs[0],
                _r_s_from_m2(model.M2.quantity),
                np.longdouble(sig.value),
            )
            report["epochs"] = "TASC from the FW absorbed encode"
        else:
            target_x = x
            target_h = h
            target_k = np.longdouble(model.EPS2.value)
            target_tasc = (
                np.longdouble(model.TASC.value) - np.longdouble("1.5") * x * h / _DAY_S
            )
            report["epochs"] = "TASC shifted by the reverse ELL1 Roemer gauge"
        outmodel.TASC.value = target_tasc
        outmodel.EPS1.value = target_h
        outmodel.EPS2.value = target_k
        outmodel.A1.quantity = target_x * u.lsec
        _set_phase_chart(target, chart, coeffs, source=model)
        _copy_parameter_state(outmodel.A1, model.A1)
        _copy_parameter_state(outmodel.EPS1, model.EPS1)
        _copy_parameter_state(outmodel.EPS2, model.EPS2)
        _copy_parameter_state(outmodel.TASC, model.TASC)
        if chart == "pb":
            if hasattr(model, "PB") and not isinstance(model.PB, funcParameter):
                _copy_parameter_state(outmodel.PB, model.PB)
            if hasattr(outmodel, "PBDOT") and not isinstance(
                outmodel.PBDOT, funcParameter
            ):
                if isinstance(model.PBDOT, funcParameter):
                    pass
                elif hasattr(model, "PBDOT") and not isinstance(
                    model.PBDOT, funcParameter
                ):
                    _copy_parameter_state(outmodel.PBDOT, model.PBDOT)
        report["coordinates"] = ["gauge_transfer"]
        report["constant_offset_s"] = float(np.longdouble("1.5") * x * h)
        if output == "ELL1H":
            outmodel.H3.quantity = h3
            outmodel.NHARMS.value = NHARMS
            if useSTIGMA:
                outmodel.STIGMA.quantity = sig
            else:
                outmodel.H4.quantity = h4
            outmodel.components["BinaryELL1H"].ell1h_shapiro = mode
            _set_state_from_dependencies(outmodel.H3, [model.M2, model.COSI])
            if useSTIGMA:
                _set_state_from_dependencies(outmodel.STIGMA, [model.COSI])
            else:
                _set_state_from_dependencies(outmodel.H4, [model.M2, model.COSI])
        else:
            outmodel.M2.quantity = model.M2.quantity
            outmodel.SINI.value = np.longdouble(model.SINI.value)
            _copy_parameter_state(outmodel.M2, model.M2)
            _set_state_from_dependencies(outmodel.SINI, [model.COSI])
        if chart == "pb" and isinstance(model.PBDOT, funcParameter):
            outmodel.PBDOT.quantity = (
                -coeffs[1] / coeffs[0] ** 2
            ) * u.dimensionless_unscaled
            outmodel.PBDOT.frozen = True
            released.append("PBDOT")
        if chart == "fbx" or _nonzero(model, "PBDOT"):
            report["named_conventions"].append("reference_frequency_inverse_timing")
            report["secular"] = ["transferred"]

    outmodel.setup()
    outmodel.validate()
    if not report["shapiro"]:
        report["shapiro"].append("unchanged")
    _finalize_ddr_report(report, model, outmodel, released=released)
    outmodel.binary_conversion_report = report
    return outmodel


def convert_binary(
    model: pint.models.TimingModel,
    output: str,
    NHARMS: int = 7,
    useSTIGMA: bool = False,
    KOM: Optional[u.Quantity] = None,
    *,
    ddrpk: bool = False,
    ddrpbdot: Optional[str] = None,
    ddrkine: bool = False,
    ddrgeo: Optional[bool] = None,
    cosi: Optional[u.Quantity] = None,
    stigma: Optional[u.Quantity] = None,
    rescale_pb: bool = False,
    ell1h_shapiro: str = "full",
    drop_edot: bool = False,
    h3_only: str = "wd_mass",
    covariance=None,
) -> pint.models.TimingModel:
    """
    Convert between binary models

    Input models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`,
    :class:`~pint.models.binary_dd.BinaryDDGR`, :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`,
    :class:`~pint.models.binary_ell1.BinaryELL1`, :class:`~pint.models.binary_ell1.BinaryELL1H`, :class:`~pint.models.binary_ell1.BinaryELL1k`,
    :class:`~pint.models.binary_dd.BinaryDDH`

        Output models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`,
    :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`, :class:`~pint.models.binary_ell1.BinaryELL1`,
    :class:`~pint.models.binary_ell1.BinaryELL1H`, :class:`~pint.models.binary_ell1.BinaryELL1k`, :class:`~pint.models.binary_dd.BinaryDDH`,
    :class:`~pint.models.binary_ddr.BinaryDDR`

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
        Longitude of the ascending node (``DDK`` / ``DDR``). ``None`` means
        absent; zero degrees is a supplied orientation.
    ddrpk : bool, optional
        Opt in to ``DDRPK Y`` after coordinate import (default phenomenological)
    ddrpbdot : str, optional
        ``kinematic`` or ``absorb_gw``. Default follows ``ddrpk``
    ddrkine, ddrgeo : bool, optional
        Opt-in kinematic / geometry flags. Missing ``PX`` turns both off
        unless geometry was explicitly requested (then refuse)
    cosi, stigma : astropy.units.Quantity, optional
        Prior centre for H3-only ELL1H/DDH import
    rescale_pb : bool, optional
        Apply ``P_B = (1+κ) P_φ`` after enabling ``DDRPK Y``
    ell1h_shapiro : {"full", "absorbed"}, optional
        ELL1H Shapiro convention when converting to ``ELL1H``. Absorbed
        applies the Freire-Wex encode of ``(A1, EPS1, EPS2, TASC)``.
    drop_edot : bool, optional
        Permit a source with active ``EDOT`` by dropping it (report
        ``edot_dropped``). Default refuses: DDR has no eccentricity-rate column.
    h3_only : {"wd_mass", "require"}, optional
        H3-only ELL1H/DDH with no ``STIGMA``/``H4``. ``wd_mass`` (default)
        applies a 0.2 solar-mass companion-mass prior; ``require`` keeps
        the old refusal unless ``cosi=`` or ``stigma=`` is supplied.
    covariance : pint.pint_matrix.CovarianceMatrix or ndarray, optional
        Source covariance to propagate through the conversion Jacobian. A
        bare array follows ``model.free_params`` ordering.

    Returns
    -------
    outmodel : pint.models.timing_model.TimingModel
        List-valued conversion report on ``outmodel.binary_conversion_report``
        for DDR conversions.

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

    binary_component_names = [
        x for x in model.components.keys() if x.startswith("Binary")
    ]
    if len(binary_component_names) > 1:
        raise ValueError(
            "convert_binary does not support hierarchical triple systems "
            f"with multiple binary components ({binary_component_names}); "
            "convert each orbit separately or remove BINARY2 first."
        )
    binary_component_name = binary_component_names[0]
    binary_component = model.components[binary_component_name]
    if binary_component.binary_model_name == output:
        log.debug(
            f"Input model and requested output are both of type '{output}'; returning copy"
        )
        return copy.deepcopy(model)
    if output == "DDR":
        result = _convert_to_ddr(
            model,
            ddrpk=ddrpk,
            ddrpbdot=ddrpbdot,
            ddrkine=ddrkine,
            ddrgeo=ddrgeo,
            cosi=cosi,
            stigma=stigma,
            rescale_pb=rescale_pb,
            KOM=KOM,
            drop_edot=drop_edot,
            h3_only=h3_only,
        )
        if covariance is not None:
            result = _propagate_conversion_covariance(
                model,
                result,
                covariance,
                lambda perturbed: _convert_to_ddr(
                    perturbed,
                    ddrpk=ddrpk,
                    ddrpbdot=ddrpbdot,
                    ddrkine=ddrkine,
                    ddrgeo=ddrgeo,
                    cosi=cosi,
                    stigma=stigma,
                    rescale_pb=rescale_pb,
                    KOM=KOM,
                    drop_edot=drop_edot,
                    h3_only=h3_only,
                ),
            )
            result.binary_conversion_report["uncertainty_propagation"] = "covariance"
        return result
    if binary_component.binary_model_name == "DDR":
        result = _convert_from_ddr(
            model,
            output,
            NHARMS=NHARMS,
            useSTIGMA=useSTIGMA,
            KOM=KOM,
            ell1h_shapiro=ell1h_shapiro,
        )
        if covariance is not None:
            result = _propagate_conversion_covariance(
                model,
                result,
                covariance,
                lambda perturbed: _convert_from_ddr(
                    perturbed,
                    output,
                    NHARMS=NHARMS,
                    useSTIGMA=useSTIGMA,
                    KOM=KOM,
                    ell1h_shapiro=ell1h_shapiro,
                ),
            )
            result.binary_conversion_report["uncertainty_propagation"] = "covariance"
        return result

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
                # use H4?
                if NHARMS > 3:
                    outmodel.H4.quantity = h4
                    outmodel.H4.uncertainty = h4_unc
                    outmodel.H4.frozen = outmodel.H3.frozen
                else:
                    outmodel.H4._quantity = None
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
                if output not in ["DDH", "DDS", "DDK", "BT"]:
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
                    outmodel = convert_binary(
                        tempmodel, output, NHARMS=NHARMS, useSTIGMA=useSTIGMA
                    )
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
                    # use H4?
                    if NHARMS > 3:
                        outmodel.H4.quantity = h4
                        outmodel.H4.uncertainty = h4_unc
                        outmodel.H4.frozen = outmodel.H3.frozen
                    else:
                        outmodel.H4._quantity = None
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
        if KOM is not None:
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
    outmodel.setup()

    return outmodel
