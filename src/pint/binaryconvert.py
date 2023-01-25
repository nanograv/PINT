"""

Potential issues:
* Use FB instead of PB
* Does EPS1DOT/EPS2DOT imply OMDOT and vice versa?

"""

import numpy as np
from astropy import units as u, constants as c
import copy
from uncertainties import ufloat, umath
from loguru import logger as log

from pint import Tsun
from pint.models.binary_bt import BinaryBT
from pint.models.binary_dd import BinaryDD, BinaryDDS, BinaryDDGR
from pint.models.binary_ddk import BinaryDDK
from pint.models.binary_ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k
from pint.models.parameter import floatParameter, MJDParameter, intParameter

binary_types = ["DD", "DDK", "DDS", "BT", "ELL1", "ELL1H", "ELL1k"]


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
    cbar = np.sqrt(1 - model.SINI.quantity**2)
    stigma = model.SINI.quantity / (1 + cbar)
    h3 = Tsun * model.M2.quantity.to_value(u.Msun) * stigma**3
    h4 = h3 * stigma
    stigma_unc = None
    h3_unc = None
    h4_unc = None
    if model.SINI.uncertainty is not None:
        stigma_unc = model.SINI.uncertainty / (1 + cbar) / cbar
        if model.M2.uncertainty is not None:
            h3_unc = np.sqrt(
                (Tsun * model.M2.uncertainty.to_value(u.Msun) * stigma**3) ** 2
                + (
                    3
                    * (Tsun * model.M2.quantity.to_value(u.Msun))
                    * stigma**2
                    * stigma_unc
                )
                ** 2
            )
            h4_unc = np.sqrt((h3_unc * stigma) ** 2 + (h3 * stigma_unc) ** 2)
    return stigma, h3, h4, stigma_unc, h3_unc, h4_unc


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

    SINI_unc = None
    M2_unc = None
    if model.STIGMA.quantity is not None:
        stigma = model.STIGMA.quantity
        stigma_unc = model.STIGMA.uncertainty
        SINI = 2 * stigma / (1 + stigma**2)
        M2 = (model.H3.quantity / stigma**3 / Tsun) * u.Msun
        if stigma_unc is not None:
            SINI_unc = np.abs(
                stigma_unc * 2 * (stigma**2 - 1) / (1 + stigma**2) ** 2
            )
            if model.H3.uncertainty is not None:
                M2_unc = (
                    np.sqrt(
                        (model.H3.uncertainty / stigma**3) ** 2
                        + (3 * stigma_unc * model.H3.quantity / stigma**4) ** 2
                    )
                    / Tsun
                ) * u.Msun

    elif model.H4.quantity is not None:
        # FW10 Eqn. 25, 26
        SINI = (2 * model.H3.quantity * model.H4.quantity) / (
            model.H3.quantity**2 + model.H4.quantity**2
        )
        M2 = (model.H3.quantity**4 / model.H4.quantity**3 / Tsun) * u.Msun
        if model.H4.uncertainty is not None and model.H3.uncertainty is not None:
            M2_unc = np.sqrt(
                (
                    4
                    * model.H3.quantity**3
                    * model.H3.uncertainty
                    / model.H4.quantity**3
                )
                ** 2
                + (
                    3
                    * model.H3.quantity**4
                    * model.H4.uncertainty
                    / model.H4.quantity**4
                )
                ** 2
            ) * (u.Msun / Tsun)
            SINI_unc = np.sqrt(
                (
                    (
                        (
                            2 * model.H4.quantity**3
                            - 2 * model.H3.quantity**2 * model.H4.quantity
                        )
                        * model.H3.uncertainty
                    )
                    / (model.H3.quantity**2 + model.H4.quantity**2) ** 2
                )
                ** 2
                + (
                    (
                        (
                            2 * model.H3.quantity**3
                            - 2 * model.H4.quantity**2 * model.H3.quantity
                        )
                        * model.H4.uncertainty
                    )
                    / (model.H3.quantity**2 + model.H4.quantity**2) ** 2
                )
                ** 2
            )
    else:
        raise ValueError("Cannot uniquely convert from ELL1H to ELL1 with only H3")

    return M2, SINI, M2_unc, SINI_unc


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
    SHAPMAX = -np.log(1 - model.SINI.quantity)
    SHAPMAX_unc = (
        model.SINI.uncertainty / (1 - model.SINI.quantity)
        if model.SINI.uncertainty is not None
        else None
    )
    return SHAPMAX, SHAPMAX_unc


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
    SINI = 1 - np.exp(-model.SHAPMAX.quantity)
    SINI_unc = (
        model.SHAPMAX.uncertainty * np.exp(-model.SHAPMAX.quantity)
        if model.SHAPMAX.uncertainty is not None
        else None
    )
    return SINI, SINI_unc


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
    ECC_unc : astropy.units.Quantity or None
        Uncertainty on ECC
    OM_unc : astropy.units.Quantity or None
        Uncertainty on OM
    T0_unc : astropy.units.Quantity or None
        Uncertainty on T0
    EDOT_unc : astropy.units.Quantity or None
        Uncertainty on EDOT

    References
    ----------
    - Lange et al. (2001), MNRAS, 326, 274 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract

    """
    if model.BINARY.value not in ["ELL1", "ELL1H", "ELL1k"]:
        raise ValueError(f"Requires model ELL1* rather than {model.BINARY.value}")
    # do we have to account for FB or PBDOT here?
    ECC = np.sqrt(model.EPS1.quantity**2 + model.EPS2.quantity**2)
    OM = np.arctan2(model.EPS2.quantity, model.EPS1.quantity)
    if OM < 0:
        OM += 360 * u.deg
    T0 = model.TASC.quantity + ((model.PB.quantity / 2 / np.pi) * OM).to(
        u.d, equivalencies=u.dimensionless_angles()
    )
    ECC_unc = None
    OM_unc = None
    T0_unc = None
    if model.EPS1.uncertainty is not None and model.EPS2.uncertainty is not None:
        ECC_unc = np.sqrt(
            (model.EPS1.uncertainty * model.EPS1.quantity / ECC) ** 2
            + (model.EPS2.uncertainty * model.EPS2.quantity / ECC) ** 2
        )
        OM_unc = np.sqrt(
            (model.EPS1.uncertainty * model.EPS2.quantity / ECC**2) ** 2
            + (model.EPS2.uncertainty * model.EPS1.quantity / ECC**2) ** 2
        )
        if model.PB.uncertainty is not None and model.TASC.uncertainty is not None:
            T0_unc = np.sqrt(
                (model.TASC.uncertainty) ** 2
                + (model.PB.quantity / 2 / np.pi * OM_unc).to(
                    u.d, equivalencies=u.dimensionless_angles()
                )
                ** 2
                + (model.PB.uncertainty * OM / 2 / np.pi).to(
                    u.d, equivalencies=u.dimensionless_angles()
                )
                ** 2
            )
    # does there also need to be a computation of OMDOT here?
    EDOT = None
    EDOT_unc = None
    if model.BINARY.value == "ELL1k":
        if model.LNEDOT.quantity is not None and ECC is not None:
            EDOT = model.LNEDOT.quantity * ECC
            if model.LNEDOT.uncertainty is not None or ECC_unc is not None:
                EDOT_unc = 0
                if model.LNEDOT.uncertainty is not None:
                    EDOT_unc += (model.LNEDOT.uncertainty * ECC) ** 2
                if ECC_unc is not None:
                    EDOT_unc += (model.LNEDOT.quantity * ECC_unc) ** 2
                EDOT_unc = np.sqrt(EDOT_unc)
    else:
        if model.EPS1DOT.quantity is not None and model.EPS2DOT.quantity is not None:
            EDOT = (
                model.EPS1DOT.quantity * model.EPS1.quantity
                + model.EPS2DOT.quantity * model.EPS2.quantity
            ) / ECC
            if (
                model.EPS1DOT.uncertainty is not None
                and model.EPS2DOT.uncertainty is not None
            ):
                EDOT_unc = np.sqrt(
                    (
                        model.EPS1.uncertainty
                        * model.EPS2.quantity
                        * (
                            model.EPS1.quantity * model.EPS2DOT.quantity
                            - model.EPS2.quantity * model.EPS1DOT.quantity
                        )
                        / ECC**3
                    )
                    ** 2
                    + (
                        model.EPS2.uncertainty
                        * model.EPS1.quantity
                        * (
                            model.EPS2.quantity * model.EPS1DOT.quantity
                            - model.EPS1.quantity * model.EPS2DOT.quantity
                        )
                        / ECC**3
                    )
                    ** 2
                    + (model.EPS1DOT.uncertainty * model.EPS1.quantity / ECC) ** 2
                    + (model.EPS2DOT.uncertainty * model.EPS2.quantity / ECC) ** 2
                )
    return ECC, OM, T0, EDOT, ECC_unc, OM_unc, T0_unc, EDOT_unc


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
    EPS1_unc = None
    EPS2_unc = None
    TASC_unc = None
    EPS1DOT = None
    EPS2DOT = None
    EPS1DOT_unc = None
    EPS2DOT_unc = None
    EPS1 = model.ECC.quantity * np.cos(model.OM.quantity)
    EPS2 = model.ECC.quantity * np.sin(model.OM.quantity)
    TASC = model.T0.quantity - (model.PB.quantity * model.OM.quantity / 2 / np.pi).to(
        u.d, equivalencies=u.dimensionless_angles()
    )
    if model.ECC.uncertainty is not None and model.OM.uncertainty is not None:
        EPS1_unc = np.sqrt(
            (model.ECC.uncertainty * np.cos(model.OM.quantity)) ** 2
            + (
                model.ECC.quantity * model.OM.uncertainty * np.sin(model.OM.quantity)
            ).to(u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())
            ** 2
        )
        EPS2_unc = np.sqrt(
            (model.ECC.uncertainty * np.sin(model.OM.quantity)) ** 2
            + (
                model.ECC.quantity * model.OM.uncertainty * np.cos(model.OM.quantity)
            ).to(u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())
            ** 2
        )
    if (
        model.OM.uncertainty is not None
        and model.T0.uncertainty is not None
        and model.PB.uncertainty is not None
    ):
        TASC_unc = np.sqrt(
            (model.T0.uncertainty) ** 2
            + (model.PB.uncertainty * model.OM.quantity / 2 / np.pi).to(
                u.d, equivalencies=u.dimensionless_angles()
            )
            ** 2
            + (model.PB.quantity * model.OM.uncertainty / 2 / np.pi).to(
                u.d, equivalencies=u.dimensionless_angles()
            )
            ** 2
        )
    if model.EDOT.quantity is not None:
        EPS1DOT = model.EDOT.quantity * np.cos(model.OM.quantity)
        EPS2DOT = model.EDOT.quantity * np.sin(model.OM.quantity)
        if model.EDOT.uncertainty is not None:
            EPS1DOT_unc = model.EDOT.uncertainty * np.cos(model.OM.quantity)
            EPS2DOT_unc = model.EDOT.uncertainty * np.sin(model.OM.quantity)
    return (
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
    )


def _ELL1_to_ELL1k(model):
    if model.BINARY.value not in ["ELL1", "ELL1H"]:
        raise ValueError(f"Requires model ELL1/ELL1H rather than {model.BINARY.value}")
    LNEDOT = None
    OMDOT = None
    LNEDOT_unc = None
    OMDOT_unc = None
    if (
        model.EPS1.quantity is not None
        and model.EPS2.quantity is not None
        and model.EPS1DOT.quantity is not None
        and model.EPS2DOT.quantity is not None
    ):
        LNEDOT = (
            model.EPS1.quantity * model.EPS1DOT.quantity
            + model.EPS2.quantity * model.EPS2.quantity
        )
        OMDOT = (
            model.EPS2.quantity * model.EPS1DOT.quantity
            - model.EPS1.quantity * model.EPS2DOT.quantity
        )
        if (
            model.EPS1.uncertainty is not None
            and model.EPS2.uncertainty is not None
            and model.EPS1DOT.uncertainty is not None
            and model.EPS2DOT.uncertainty is not None
        ):
            LNEDOT_unc = np.sqrt(
                (model.EPS1.uncertainty * model.EPS1DOT.quantity) ** 2
                + (model.EPS2.uncertainty * model.EPS2DOT.quantity) ** 2
                + (model.EPS1.quantity * model.EPS1DOT.uncertainty) ** 2
                + (model.EPS2.uncertainty * model.EPS2DOT.quantity) ** 2
            )
            OMDOT_unc = np.sqrt(
                (model.EPS2.uncertainty * model.EPS1DOT.quantity) ** 2
                + (model.EPS1.uncertainty * model.EPS2DOT.quantity) ** 2
                + (model.EPS2.quantity * model.EPS1DOT.uncertainty) ** 2
                + (model.EPS1.uncertainty * model.EPS2DOT.quantity) ** 2
            )
    return LNEDOT, OMDOT, LNEDOT_unc, OMDOT_unc


def _ELL1k_to_ELL1(model):
    if model.BINARY.value != "ELL1k":
        raise ValueError(f"Requires model ELL1k rather than {model.BINARY.value}")
    EPS1DOT = None
    EPS2DOT = None
    EPS1DOT_unc = None
    EPS2DOT_unc = None
    if (
        model.LNEDOT.quantity is not None
        and model.OMDOT.quantity is not None
        and model.EPS1.quantity is not None
        and model.EPS2.quantity
    ):
        EPS1DOT = (
            model.LNEDOT.quantity * model.EPS1.quantity
            + model.OMDOT.quantity * model.EPS2.quantity
        )
        EPS2DOT = (
            model.LNEDOT.quantity * model.EPS2.quantity
            - model.OMDOT.quantity * model.EPS1.quantity
        )
        if (
            model.LNEDOT.uncertainty is not None
            and model.OMDOT.uncertainty is not None
            and model.EPS1.uncertainty is not None
            and model.EPS2.uncertainty is not None
        ):
            EPS1DOT_unc = np.sqrt(
                (model.LNEDOT.uncertainty * model.EPS1.quantity) ** 2
                + (model.LNEDOT.quantity * model.EPS1.uncertainty) ** 2
                + (model.OMDOT.uncertainty * model.EPS2.quantity) ** 2
                + (model.OMDOT.quantity * model.EPS2.uncertainty) ** 2
            )
            EPS2DOT_unc = np.sqrt(
                (model.LNEDOT.uncertainty * model.EPS2.quantity) ** 2
                + (model.LNEDOT.quantity * model.EPS2.uncertainty) ** 2
                + (model.OMDOT.uncertainty * model.EPS1.quantity) ** 2
                + (model.OMDOT.quantity * model.EPS1.uncertainty) ** 2
            )
    return EPS1DOT, EPS2DOT, EPS1DOT_unc, EPS2DOT_unc


def _DDK_to_PK(model):
    """Convert DDK model to equivalent PK parameters

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
    tsun = Tsun.to_value(u.s)
    if model.MTOT.uncertainty is not None:
        mtot = ufloat(
            model.MTOT.quantity.to_value(u.Msun),
            model.MTOT.uncertainty.to_value(u.Msun),
        )
    else:
        mtot = ufloat(model.MTOT.quantity.to_value(u.Msun), 0)
    if model.M2.uncertainty is not None:
        mc = ufloat(
            model.M2.quantity.to_value(u.Msun), model.M2.uncertainty.to_value(u.Msun)
        )
    else:
        mc = ufloat(model.M2.quantity.to_value(u.Msun), 0)
    mp = mtot - mc
    if model.A1.uncertainty is not None:
        x = ufloat(model.A1.value, model.A1.uncertainty_value)
    else:
        x = ufloat(model.A1.value, 0)
    if model.PB.uncertainty is not None:
        n = (
            2
            * np.pi
            / ufloat(
                model.PB.quantity.to_value(u.s), model.PB.uncertainty.to_value(u.s)
            )
        )
    else:
        n = 2 * np.pi / model.PB.quantity.to_value(u.s)
    if model.ECC.uncertainty is not None:
        ECC = ufloat(model.ECC.value, model.ECC.uncertainty_value)
    else:
        ECC = ufloat(model.ECC.value, 0)
    # units are seconds
    gamma = (
        tsun ** (2.0 / 3)
        * n ** (-1.0 / 3)
        * (mc * (mp + 2 * mc) / (mp + mc) ** (4.0 / 3))
    )
    # units as seconds
    r = tsun * mc
    # units are radian/s
    omegadot = (
        (3 * tsun ** (2.0 / 3))
        * n ** (5.0 / 3)
        * (1 / (1 - ECC**2))
        * (mp + mc) ** (2.0 / 3)
    )
    fe = (1 + (73.0 / 24) * ECC**2 + (37.0 / 96) * ECC**4) / (1 - ECC**2) ** (
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


def convert_binary(model, output, NHARMS=3, useSTIGMA=False, KOM=0 * u.deg):
    """
    Convert between binary models

    Input models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`, :class:`~pint.models.binary_dd.BinaryDDGR`, :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`, :class:`~pint.models.binary_ell1.BinaryELL1`, :class:`~pint.models.binary_ell1.BinaryELL1H`
    Output models can be from :class:`~pint.models.binary_dd.BinaryDD`, :class:`~pint.models.binary_dd.BinaryDDS`, :class:`~pint.models.binary_bt.BinaryBT`, :class:`~pint.models.binary_ddk.BinaryDDK`, :class:`~pint.models.binary_ell1.BinaryELL1`, :class:`~pint.models.binary_ell1.BinaryELL1H`

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

    if binary_component.binary_model_name in ["ELL1", "ELL1H", "ELL1k"]:
        if output == "ELL1H":
            # ELL1,ELL1k -> ELL1H
            stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(model)
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = ["M2", "SINI", "BINARY"]
            outmodel.add_component(BinaryELL1H(), validate=False)
            if binary_component.binary_model_name == "ELL1k":
                badlist += ["LNEDOT", "OMDOT"]
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
            for p in model.params:
                if p not in badlist:
                    setattr(outmodel, p, getattr(model, p))
            for p in model.components[binary_component_name].params:
                if p not in badlist:
                    setattr(
                        outmodel.components["BinaryELL1H"],
                        p,
                        getattr(model.components[binary_component_name], p),
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
        elif output == "ELL1":
            if model.BINARY.value == "ELL1H":
                # ELL1H -> ELL1
                M2, SINI, M2_unc, SINI_unc = _orthometric_to_M2SINI(model)
                outmodel = copy.deepcopy(model)
                outmodel.remove_component(binary_component_name)
                outmodel.BINARY.value = output
                # parameters not to copy
                badlist = ["H3", "H4", "STIGMA", "BINARY"]
                outmodel.add_component(BinaryELL1(), validate=False)
                for p in model.params:
                    if p not in badlist:
                        setattr(outmodel, p, getattr(model, p))
                for p in model.components[binary_component_name].params:
                    if p not in badlist:
                        setattr(
                            outmodel.components["BinaryELL1"],
                            p,
                            getattr(model.components[binary_component_name], p),
                        )
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
                outmodel = copy.deepcopy(model)
                outmodel.remove_component(binary_component_name)
                outmodel.BINARY.value = output
                # parameters not to copy
                badlist = ["BINARY", "LNEDOT", "OMDOT"]
                outmodel.add_component(BinaryELL1(), validate=False)
                EPS1DOT, EPS2DOT, EPS1DOT_unc, EPS2DOT_unc = _ELL1k_to_ELL1(model)
                for p in model.params:
                    if p not in badlist:
                        setattr(outmodel, p, getattr(model, p))
                for p in model.components[binary_component_name].params:
                    if p not in badlist:
                        setattr(
                            outmodel.components["BinaryELL1"],
                            p,
                            getattr(model.components[binary_component_name], p),
                        )
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
            if model.BINARY.value == "ELL1":
                # ELL1 -> ELL1k
                LNEDOT, OMDOT, LNEDOT_unc, OMDOT_unc = _ELL1_to_ELL1k(model)
                outmodel = copy.deepcopy(model)
                outmodel.remove_component(binary_component_name)
                outmodel.BINARY.value = output
                # parameters not to copy
                badlist = ["BINARY", "EPS1DOT", "EPS2DOT"]
                outmodel.add_component(BinaryELL1k(), validate=False)
                for p in model.params:
                    if p not in badlist:
                        setattr(outmodel, p, getattr(model, p))
                for p in model.components[binary_component_name].params:
                    if p not in badlist:
                        setattr(
                            outmodel.components["BinaryELL1k"],
                            p,
                            getattr(model.components[binary_component_name], p),
                        )
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
                outmodel = copy.deepcopy(model)
                outmodel.remove_component(binary_component_name)
                outmodel.BINARY.value = output
                # parameters not to copy
                badlist = ["BINARY", "EPS1DOT", "EPS2DOT", "H3", "H4", "STIGMA"]
                outmodel.add_component(BinaryELL1k(), validate=False)
                for p in model.params:
                    if p not in badlist:
                        setattr(outmodel, p, getattr(model, p))
                for p in model.components[binary_component_name].params:
                    if p not in badlist:
                        setattr(
                            outmodel.components["BinaryELL1k"],
                            p,
                            getattr(model.components[binary_component_name], p),
                        )
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
        elif output in ["DD", "DDS", "DDK", "BT"]:
            # need to convert
            ECC, OM, T0, EDOT, ECC_unc, OM_unc, T0_unc, EDOT_unc = _from_ELL1(model)
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = [
                "TASC",
                "EPS1",
                "EPS2",
                "EPS1DOT",
                "EPS2DOT",
                "BINARY",
            ]
            if output == "DD":
                outmodel.add_component(BinaryDD(), validate=False)
            elif output == "DDS":
                outmodel.add_component(BinaryDDS(), validate=False)
                badlist.append("SINI")
            elif output == "DDK":
                outmodel.add_component(BinaryDDK(), validate=False)
                badlist.append("SINI")
            elif output == "BT":
                outmodel.add_component(BinaryBT(), validate=False)
                badlist += ["M2", "SINI"]
            if binary_component.binary_model_name == "ELL1H":
                badlist += ["H3", "H4", "STIGMA", "VARSIGMA"]
            for p in model.params:
                if p not in badlist:
                    setattr(outmodel, p, getattr(model, p))
            for p in model.components[binary_component_name].params:
                if p not in badlist:
                    setattr(
                        outmodel.components[f"Binary{output}"],
                        p,
                        getattr(model.components[binary_component_name], p),
                    )

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
            outmodel.T0.frozen = (
                model.EPS1.frozen
                or model.EPS2.frozen
                or model.TASC.frozen
                or model.PB.frozen
            )
            if EDOT is not None:
                outmodel.EDOT.quantity = EDOT
            if EDOT_unc is not None:
                outmodel.EDOT.uncertainty = EDOT_unc
            if binary_component.binary_model_name != "ELL1k":
                outmodel.EDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
            else:
                outmodel.EDOT.frozen = model.LNEDOT.frozen
            if binary_component.binary_model_name == "ELL1H":
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
        else:
            raise ValueError(
                f"Do not know how to convert from {binary_component.binary_model_name} to {output}"
            )
    elif binary_component.binary_model_name in ["DD", "DDGR", "DDS", "DDK", "BT"]:
        if output in ["DD", "DDS", "DDK", "BT"]:
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = [
                "BINARY",
            ]
            if binary_component.binary_model_name == "DDS":
                badlist.append("SHAPMAX")
            elif binary_component.binary_model_name == "DDK":
                badlist += ["KIN", "KOM"]
            if output == "DD":
                outmodel.add_component(BinaryDD(), validate=False)
            elif output == "DDS":
                outmodel.add_component(BinaryDDS(), validate=False)
                badlist.append("SINI")
            elif output == "DDK":
                outmodel.add_component(BinaryDDK(), validate=False)
                badlist.append("SINI")
            elif output == "BT":
                outmodel.add_component(BinaryBT(), validate=False)
                badlist += ["M2", "SINI"]
            for p in model.params:
                if p not in badlist:
                    setattr(outmodel, p, getattr(model, p))
            for p in model.components[binary_component_name].params:
                if p not in badlist:
                    setattr(
                        outmodel.components[f"Binary{output}"],
                        p,
                        getattr(model.components[binary_component_name], p),
                    )
            if binary_component.binary_model_name == "DDS":
                SINI, SINI_unc = _SHAPMAX_to_SINI(model)
                outmodel.SINI.quantity = SINI
                if SINI_unc is not None:
                    outmodel.SINI.uncertainty = SINI_unc
            elif binary_component.binary_model_name == "DDK":
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
            elif binary_component.binary_model_name == "DDGR":
                pbdot, gamma, omegadot, s, r, Dr, Dth = _DDK_to_PK(model)
                outmodel.GAMMA.value = gamma.n
                if gamma.s > 0:
                    outmodel.GAMMA.uncertainty_value = gamma.s
                outmodel.PBDOT.value = pbdot.n
                if pbdot.s > 0:
                    outmodel.PBDOT.uncertainty_value = pbdot.s
                outmodel.OMDOT.value = omegadot.n
                if omegadot.s > 0:
                    outmodel.OMDOT.uncertainty_value = omegadot.s
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
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = ["BINARY", "ECC", "OM", "T0", "OMDOT", "EDOT"]
            if binary_component.binary_model_name == "DDS":
                badlist.append("SHAPMAX")
            elif binary_component.binary_model_name == "DDK":
                badlist += ["KIN", "KOM"]
            if output == "ELL1":
                outmodel.add_component(BinaryELL1(), validate=False)
            elif output == "ELL1H":
                outmodel.add_component(BinaryELL1H(), validate=False)
                badlist += ["M2", "SINI"]
            elif output == "ELL1k":
                outmodel.add_component(BinaryELL1k(), validate=False)
                badlist += ["EPS1DOT", "EPS2DOT"]
                badlist.remove("OMDOT")
                badlist.remove("EDOT")
            for p in model.params:
                if p not in badlist:
                    setattr(outmodel, p, getattr(model, p))
            for p in model.components[binary_component_name].params:
                if p not in badlist:
                    setattr(
                        outmodel.components[f"Binary{output}"],
                        p,
                        getattr(model.components[binary_component_name], p),
                    )
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
                LNEDOT = 0 / u.yr
                if (
                    hasattr(model, "ECCDOT")
                    and model.ECCDOT.quantity is not None
                    and model.ECC.quantity is not None
                ):
                    LNEDOT = model.ECCDOT.quantity / model.ECC.quantity
                    if (
                        model.ECCDOT.uncertainty is not None
                        or model.ECC.uncertainty is not None
                    ):
                        LNEDOT_unc = 0
                        if model.ECCDOT.uncertainty is not None:
                            LNEDOT_unc += (
                                model.ECCDOT.uncertainty / model.ECC.quantity
                            ) ** 2
                        if model.ECC.uncertainty is not None:
                            LNEDOT_unc += (
                                model.ECCDOT.quantity
                                * model.ECC.uncertainty
                                / model.ECC.quantity**2
                            )
                        LNEDOT_unc = np.sqrt(LNEDOT_unc)
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
                SINI, SINI_unc = _SHAPMAX_to_SINI(model)
                outmodel.SINI.quantity = SINI
                if SINI_unc is not None:
                    outmodel.SINI.uncertainty = SINI_unc
            elif binary_component.binary_model_name == "DDK":
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
            if output == "ELL1H":
                if binary_component.binary_model_name == "DDGR":
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

    if output == "DDS" and binary_component.binary_model_name != "DDGR":
        SHAPMAX, SHAPMAX_unc = _SINI_to_SHAPMAX(model)
        outmodel.SHAPMAX.quantity = SHAPMAX
        outmodel.SHAPMAX.uncertainty = SHAPMAX_unc
        outmodel.SHAPMAX.frozen = model.SINI.frozen

    if output == "DDK":
        outmodel.KOM.quantity = KOM
        if binary_component.binary_model_name != "DDGR":
            if model.SINI.quantity is not None:
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
