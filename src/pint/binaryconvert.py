"""

Potential issues:
* Use FB instead of PB
* Does EPS1DOT/EPS2DOT imply OMDOT and vice versa?

"""

import numpy as np
from astropy import units as u, constants as c
import copy

from loguru import logger as log

from pint import Tsun
from pint.models.binary_bt import BinaryBT
from pint.models.binary_dd import BinaryDD, BinaryDDS, BinaryDDGR
from pint.models.binary_ddk import BinaryDDK
from pint.models.binary_ell1 import BinaryELL1, BinaryELL1H
from pint.models.parameter import floatParameter, MJDParameter, intParameter

binary_types = ["DD", "DDK", "DDS", "BT", "ELL1", "ELL1H"]


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
    stigma : u.Quantity
    h3 : u.Quantity
    h4 : u.Quantity
    stigma_unc : u.Quantity or None
        Uncertainty on stigma
    h3_unc : u.Quantity or None
        Uncertainty on H3
    h4_unc : u.Quantity or None
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
    M2 : u.Quantity
    SINI : u.Quantity
    M2_unc : u.Quantity or None
        Uncertainty on M2
    SINI_unc : u.Quantity or None
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
    else:
        stigma = (
            model.STIGMA.quantity
            if model.STIGMA.quantity is not None
            else model.H4.quantity / model.H3.quantity
        )
        stigma_unc = (
            np.sqrt(
                (model.H4.uncertainty / model.H3.quantity) ** 2
                + (model.H3.uncertainty * model.H4.quantity / model.H3.quantity**2)
                ** 2
            )
            if (model.H3.uncertainty is not None and model.H4.uncertainty is not None)
            else None
        )

    SINI = stigma / np.sqrt(stigma**2 + 1)
    M2 = (model.H3.quantity / stigma**3 / Tsun) * u.Msun
    if stigma_unc is not None:
        SINI_unc = stigma_unc / (stigma**2 + 1) ** 1.5
        if model.H3.uncertainty is not None:
            M2_unc = (
                np.sqrt(
                    (model.H3.uncertainty / stigma**3) ** 2
                    + (3 * stigma_unc * model.H3.quantity / stigma**4) ** 2
                )
                / Tsun
            ) * u.Msun
    return M2, SINI, M2_unc, SINI_unc


def _SINI_to_SHAPMAX(model):
    """Convert from standard SINI to alternate SHAPMAX parameterization

    Also propagates uncertainties if present

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel

    Returns
    -------
    SHAPMAX : u.Quantity
    SHAPMAX_unc : u.Quantity or None
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
    SINI : u.Quantity
    SINI_unc : u.Quantity or None
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
    ECC : u.Quantity
    OM : u.Quantity
    T0 : u.Quantity
    EDOT : u.Quantity or None
    ECC_unc : u.Quantity or None
        Uncertainty on ECC
    OM_unc : u.Quantity or None
        Uncertainty on OM
    T0_unc : u.Quantity or None
        Uncertainty on T0
    EDOT_unc : u.Quantity or None
        Uncertainty on EDOT

    References
    ----------
    - Lange et al. (2001), MNRAS, 326, 274 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract

    """
    # do we have to account for FB or PBDOT here?
    ECC = np.sqrt(model.EPS1.quantity**2 + model.EPS2.quantity**2)
    OM = np.arctan(model.EPS2.quantity / model.EPS1.quantity)
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
    EPS1 : u.Quantity
    EPS2 : u.Quantity
    TASC : u.Quantity
    EPS1DOT : u.Quantity or None
    EPS2DOT : u.Quantity or None
    EPS1_unc : u.Quantity or None
        Uncertainty on EPS1
    EPS2_unc : u.Quantity or None
        Uncertainty on EPS2
    TASC_unc : u.Quantity or None
        Uncertainty on TASC
    EPS1DOT_unc : u.Quantity or None
        Uncertainty on EPS1DOT
    EPS2DOT_unc : u.Quantity or None
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
                u.dimensionless_unscaled, equivalencies=u.dimensionless_angles()
            )
            ** 2
            + (model.PB.quantity * model.OM.uncertainty / 2 / np.pi).to(
                u.dimensionless_unscaled, equivalencies=u.dimensionless_angles()
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


def convert_binary(model, output, **kwargs):
    """
    Convert between binary models

    Input models can be from "DD", "DDK", "DDGR", "DDS", "BT", "ELL1", "ELL1H"
    Output models can be from "DD", "DDK", "DDS", "BT", "ELL1", "ELL1H"

    For output "DDK", must also pass value for ``KOM``
    For output "ELL1H", must also pass value for ``NHARMS``

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
    output : str
        Output model type
    kwargs :
        Other parameters (output model dependent)

    Returns
    -------
    outmodel : pint.models.timing_model.TimingModel
    """
    # Do initial checks
    output = output.upper()
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
    log.debug(f"Identified input model '{binary_component_name}'")
    # check for required input depending on output type
    if output == "DDK":
        if "KOM" not in kwargs:
            raise ValueError(
                "For output model 'DDK', must supply value of 'KOM' to this function"
            )
        if not isinstance(kwargs["KOM"], u.Quantity):
            raise ValueError("'KOM' must have units specified")
        if not kwargs["KOM"].unit.is_equivalent("deg"):
            raise ValueError(
                f"'KOM' must have units convertible to 'deg' (not {kwargs['KOM'].unit})"
            )
    elif output == "ELL1H":
        if "NHARMS" not in kwargs:
            raise ValueError(
                "For output model 'ELL1H', must supply value of 'NHARMS' to this function"
            )
        if not isinstance(kwargs["NHARMS"], int) and kwargs["NHARMS"] >= 3:
            raise ValueError(f"'NHARMS' must be an integer >=3 (not {kwargs['NHARMS']}")

    if binary_component.binary_model_name in ["ELL1", "ELL1H"]:
        if output == "ELL1H":
            # this can only be ELL -> ELL1H
            stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(model)
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = ["M2", "SINI", "BINARY"]
            outmodel.add_component(BinaryELL1H(), validate=False)
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
            outmodel.NHARMS.value = kwargs["NHARMS"]
            outmodel.H3.quantity = h3
            outmodel.H3.uncertainty = h3_unc
            outmodel.H3.frozen = model.M2.frozen or model.SINI.frozen
            # use STIGMA and H3
            outmodel.STIGMA.quantity = stigma
            outmodel.STIGMA.uncertainty = stigma_unc
            outmodel.STIGMA.frozen = outmodel.H3.frozen
        elif output == "ELL1":
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
            outmodel.EDOT.quantity = EDOT
            outmodel.EDOT.uncertainty = EDOT_unc
            outmodel.EDOT.frozen = model.EPS1DOT.frozen or model.EPS2DOT.frozen
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
    elif binary_component.binary_model_name in ["DD", "DDGR", "DDS", "DDK", "BT"]:
        if output in ["DD", "DDS", "DDK", "BT"]:
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = [
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
                b = model.components["BinaryDDGR"].binary_instance
                outmodel.GAMMA.value = b.GAMMA.value
                outmodel.PBDOT.value = b.PBDOT.value
                outmodel.OMDOT.value = b._OMDOT.value
                outmodel.DR.value = b.DR.value
                outmodel.DTH.value = b.DTH.value
                if output != "DDS":
                    outmodel.SINI.value = b._SINI.value
                else:
                    outmodel.SHAPMAX.value = -np.log(1 - b._SINI.value)
                log.warning(
                    "For conversion from DDGR model, uncertainties are not propagated on PK parameters"
                )

        elif output in ["ELL1", "ELL1H"]:
            outmodel = copy.deepcopy(model)
            outmodel.remove_component(binary_component_name)
            outmodel.BINARY.value = output
            # parameters not to copy
            badlist = ["BINARY", "ECC", "OM", "T0", "OMDOT", "EDOT"]
            if output == "ELL1":
                outmodel.add_component(BinaryELL1(), validate=False)
            elif output == "ELL1H":
                outmodel.add_component(BinaryELL1H(), validate=False)
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
            if EPS1DOT is not None:
                outmodel.EPS1DOT.quantity = EPS1DOT
                outmodel.EPS2DOT.quantity = EPS2DOT
                outmodel.EPS1DOT.frozen = model.EDOT.frozen or model.OM.frozen
                outmodel.EPS2DOT.frozen = model.EDOT.frozen or model.OM.frozen
                if EPS1DOT_unc is not None:
                    outmodel.EPS1DOT.uncertainty = EPS1DOT_unc
                    outmodel.EPS2DOT.uncertainty = EPS2DOT_unc
            if output == "ELL1H":
                if binary_component.binary_model_name == "DDGR":
                    model = convert_binary(model, "DD")
                stigma, h3, h4, stigma_unc, h3_unc, h4_unc = _M2SINI_to_orthometric(
                    model
                )
                outmodel.NHARMS.value = kwargs["NHARMS"]
                outmodel.H3.quantity = h3
                outmodel.H3.uncertainty = h3_unc
                outmodel.H3.frozen = model.M2.frozen or model.SINI.frozen
                # use STIGMA and H3
                outmodel.STIGMA.quantity = stigma
                outmodel.STIGMA.uncertainty = stigma_unc
                outmodel.STIGMA.frozen = outmodel.H3.frozen

    if output == "DDS" and binary_component.binary_model_name != "DDGR":
        SHAPMAX, SHAPMAX_unc = _SINI_to_SHAPMAX(model)
        outmodel.SHAPMAX.quantity = SHAPMAX
        outmodel.SHAPMAX.uncertainty = SHAPMAX_unc
        outmodel.SHAPMAX.frozen = model.SINI.frozen

    if output == "DDK":
        outmodel.KOM.quantity = kwargs["KOM"]
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
