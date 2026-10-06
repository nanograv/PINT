"""TCB/TDB conversion consistent with PINT's TDB forward model.

Coordinate epochs use Astropy/ERFA's IAU 2006 realization. PINT evaluates
undilated radio frequencies (``DILATEFREQ N``), but the *exponents* of the
DM-family coefficients are set by the ``DILATEFREQ`` on the TCB side of the
conversion, because that is where the dilation factor is O(1e-8) rather than
O(1e-10). See :func:`convert_tcb_tdb`.

Unsupported active deterministic terms are reported but do not stop conversion.
"""

from dataclasses import dataclass

import erfa
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
from pint.pulsar_mjd import time_from_longdouble

__all__ = [
    "TCB_TDB_F",
    "TCB_TDB_K",
    "IFTE_K",
    "TCBTDBConversionReport",
    "scale_parameter",
    "transform_mjd_parameter",
    "convert_tcb_tdb",
]

# PINT evaluates its forward model in IAU 2006 TDB. Obtain the defining rate
# from ERFA rather than maintaining another decimal copy here.
TCB_TDB_F = np.longdouble(1) - np.longdouble(erfa.ELB)
TCB_TDB_K = np.longdouble(1) / TCB_TDB_F
_TCB_TDB_L = np.longdouble(erfa.ELB)

# Backwards-compatible public alias. This is now the IAU/ERFA rate, not the
# historical IFTE common-origin epoch map.
IFTE_K = TCB_TDB_K


def _k_power_minus_one(exponent: int) -> np.longdouble:
    """Return ``K**exponent - 1`` from ``L_B``, not from a factor near 1.

    ``K = 1/(1-L_B)`` differs from 1 by ``~1.55e-8``. Forming ``K**n`` and
    subtracting 1 (or multiplying ``x`` by ``K`` when ``x`` is O(10–100))
    parks that correction in the last bits of a number near ``x``. Then a
    float64 rounding of the factor is a ``~1e-16`` relative error, which is
    ``~10 ns`` after ``F0`` accumulates for 1250 d.

    The increment is O(``n L_B``) and is well resolved even in float64::

        K**n - 1 = (1 - F**n) / F**n    n > 0
        K**n - 1 = F**(-n) - 1          n < 0

    with ``1 - F**n`` built from ``L_B`` so nothing near 1 is subtracted.
    """
    if exponent == 0:
        return np.longdouble(0)
    n = abs(exponent)
    one_minus_fn = _TCB_TDB_L
    fn = TCB_TDB_F
    for _ in range(1, n):
        one_minus_fn = _TCB_TDB_L + TCB_TDB_F * one_minus_fn
        fn *= TCB_TDB_F
    if exponent > 0:
        return one_minus_fn / fn
    return -one_minus_fn


@dataclass(frozen=True)
class TCBTDBConversionReport:
    """Summary of operations performed during TCB/TDB conversion."""

    source_units: str
    target_units: str
    convention: str
    converted: tuple[str, ...]
    invariant: tuple[str, ...]
    unsupported: tuple[str, ...]
    unaudited_components: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """Whether the conversion satisfies the tested no-refit contract.

        The contract is literal. With the TOAs, the clock chain and the
        solar-system ephemeris held fixed, an accepted conversion reproduces
        the source model's residuals to better than 1 ns with no parameter
        refitted. The single exception is the overall phase gauge, the
        constant fixed by ``TZRMJD``, a subtracted mean residual or a
        reference ``JUMP``; each timing package chooses that constant by its
        own convention, so the bound is on the shape of the residuals.
        """
        return not self.unsupported and not self.unaudited_components


def scale_parameter(model: TimingModel, param: str, n: int, backwards: bool) -> None:
    """Scale a parameter x by a power of the IAU TCB/TDB rate K.

        x_tdb = x_tcb * K**n = x_tcb + x_tcb * (K**n - 1)

    The second form is what is applied: ``K**n - 1`` is O(n L_B) and is
    obtained from ``L_B`` rather than by subtracting two values near 1.

    The power n depends on the "effective dimensionality" of
    the parameter as it appears in the timing model. Some examples
    are given bellow:

        1. F0 has effective dimensionality of frequency and n = 1
        2. F1 has effective dimensionality of frequency^2 and n = 2
        3. A1 has effective dimensionality of time because it appears as
           A1/c in the timing model. Therefore, its n = -1
        4. Constant DM has an explicit n = -1 override because PINT keeps
           radio frequency fixed during conversion
        5. PBDOT is dimensionless and has n = 0. i.e., it is not scaled.

    Parameter
    ---------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    n : int
        The power of TCB_TDB_K in the scaling factor
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    assert isinstance(n, int), "The power must be an integer."

    p = -1 if backwards else 1
    delta = _k_power_minus_one(p * n)

    if (param in model) and model[param].quantity is not None:
        par = model[param]
        # x * K**n = x + x*(K**n - 1). The increment is ~1e-8 relative, so
        # this stays accurate if K itself would round to a number near 1.
        x = np.longdouble(par.value)
        par.value = x + x * delta
        if par.uncertainty_value is not None:
            ux = np.longdouble(par.uncertainty_value)
            par.uncertainty_value = ux + ux * delta


def transform_mjd_parameter(model: TimingModel, param: str, backwards: bool) -> None:
    """Convert a coordinate epoch between TCB and IAU 2006 TDB.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    if (param in model) and model[param].quantity is not None:
        par = model[param]
        assert isinstance(par, MJDParameter) or (
            isinstance(par, prefixParameter)
            and isinstance(par.param_comp, MJDParameter)
        )

        source = "tdb" if backwards else "tcb"
        target = "tcb" if backwards else "tdb"
        converted = getattr(time_from_longdouble(par.value, source), target)

        # Re-label before assigning the already-converted two-part Time so the
        # old numeric MJD is not silently reinterpreted in the target scale.
        par.time_scale = target
        par.quantity = converted
        if par.uncertainty_value is not None:
            par.uncertainty_value *= TCB_TDB_K if backwards else TCB_TDB_F


def _scale_exponent(param, dilated: bool = False) -> int:
    """Return the conversion exponent, honoring component-owned overrides.

    ``dilated`` is the ``DILATEFREQ`` of the TCB side of the conversion. The
    declared exponents are the undilated ones; a dilated TCB side adds the
    parameter's ``tcb2tdb_freq_power`` (``alpha`` in
    ``delay ~ x / freqSSB**alpha``), which is 0 for everything but the DM
    family. For constant DM that turns ``K**-1`` into ``K**+1``.
    """
    override = getattr(param, "tcb2tdb_scale_exponent", None)
    if callable(override):
        override = override(param)
    n = int(override) if override is not None else -param.effective_dimensionality
    if dilated:
        n += int(getattr(param, "tcb2tdb_freq_power", 0) or 0)
    return n


def _parameter_conversion_plan(
    model: TimingModel, dilated: bool = False
) -> tuple[set[str], set[str], set[str]]:
    """Classify set parameters without changing the model."""
    convertible: set[str] = set()
    invariant: set[str] = set()
    unsupported: set[str] = set()

    for name in model.params:
        param = model[name]
        if param.quantity is None:
            continue
        if getattr(param, "tcb2tdb_invariant", False):
            invariant.add(name)
            continue
        if not getattr(param, "convert_tcb2tdb", False):
            continue
        if isinstance(param, (floatParameter, AngleParameter, maskParameter)) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, (floatParameter, AngleParameter))
        ):
            if _scale_exponent(param, dilated) == 0:
                invariant.add(name)
            else:
                convertible.add(name)
        elif isinstance(param, MJDParameter) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, MJDParameter)
        ):
            if param.time_scale == "utc":
                invariant.add(name)
            elif param.time_scale in {"tcb", "tdb"}:
                convertible.add(name)
            else:
                unsupported.add(name)
        else:
            unsupported.add(name)

    return convertible, invariant, unsupported


def _active_unsupported_components(
    model: TimingModel,
    converted: set[str],
    invariant: set[str],
    dilated: bool = False,
) -> tuple[set[str], set[str]]:
    """Find unhandled parameters from PINT's actual delay/phase graph.

    A dilated TCB side also un-certifies every component holding a
    frequency-dependent coefficient: the exponents are right, but PINT cannot
    close the loop by evaluating a dilated model, and its undilated evaluation
    of the converted par differs from a dilated one by ~1e-9 of the dispersion
    delay. That is a model difference no parameter value can absorb.
    """
    unsupported: set[str] = set()
    components: set[str] = set()
    forward_components = model.DelayComponent_list + model.PhaseComponent_list

    for component in forward_components:
        component_unsupported = set()
        dilation_sensitive = False
        for name in component.params:
            par = model[name]
            if par.quantity is None:
                continue
            if int(getattr(par, "tcb2tdb_freq_power", 0) or 0) != 0:
                dilation_sensitive = True
            if name in converted or name in invariant:
                continue
            if hasattr(par, "convert_tcb2tdb"):
                component_unsupported.add(name)
        if component_unsupported:
            unsupported.update(component_unsupported)
        if (
            component_unsupported
            or not component.tcb2tdb_certified
            or (dilated and dilation_sensitive)
        ):
            components.add(component.__class__.__name__)

    return unsupported, components


def convert_tcb_tdb(
    model: TimingModel, backwards: bool = False
) -> TCBTDBConversionReport:
    """Convert every supported parameter between TCB and TDB.

    Coordinate epochs follow PINT's IAU 2006 TDB. Unsupported active
    deterministic terms are left unchanged and reported; they never prevent
    supported operations from completing.

    The DM-family exponents depend on ``DILATEFREQ``, and the flag that
    matters is the one on the **TCB** side of the conversion. Requiring the
    physical delay to be the same object in both unit systems gives, for
    ``delay ~ x / freqSSB**alpha``::

        x_tdb = F x_tcb (E_tcb / E_tdb)**alpha

    where ``E`` is the Einstein-rate divisor a dilating engine applies to the
    barycentric frequency. On the TCB side ``E`` carries the full ``K``
    (~1.55e-8); on the TDB side it differs from 1 by only ~5e-10. So a dilated
    TCB par needs ``x_tdb = K x_tcb`` for constant DM, not ``F x_tcb``, and the
    TDB file's own flag is irrelevant at this level. Getting this wrong costs
    ``K**2 - 1 = 3.1e-8`` of the dispersion delay: ~3 ns at 1.4 GHz and ~40 ns
    at 400 MHz for DM = 50, and it is chromatic, so no phase gauge absorbs it.

    TEMPO2 defaults to ``DILATEFREQ Y`` and writes it into the TCB par files it
    produces, so this is the common case rather than an exotic one. PINT
    records the flag in ``TimingModel.meta["tcb_source_dilatefreq"]`` when
    :meth:`~pint.models.timing_model.TimingModel.validate` coerces it to N.
    Going backwards, the TCB side is PINT's own output, which is always
    undilated.

    A dilated source leaves the frequency-dependent components unaudited: the
    exponents are right, but PINT cannot certify them by closure because it
    cannot evaluate a dilated model, and its undilated evaluation of the
    converted par still differs from a dilated one by ``E**2 - 1 ~ 1e-9`` of
    the dispersion delay (~1 ns at 400 MHz for DM = 50), which is an annual
    signal no parameter value can absorb.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
       Timing model to be converted.
    backwards : bool
        Whether to do TDB to TCB conversion. The default is TCB to TDB.

    Returns
    -------
    TCBTDBConversionReport
        Actual converted, invariant, and unsupported model terms. An accepted
        report is covered by PINT's tested no-refit conversion contract.
    """

    target_units = "TCB" if backwards else "TDB"
    source_units = "TDB" if backwards else "TCB"

    # The DILATEFREQ of the TCB side: the input par when converting TCB->TDB,
    # PINT's own (always undilated) output when going backwards.
    dilated = bool(model.meta.get("tcb_source_dilatefreq", False)) and not backwards
    convention = (
        "iau2006-dilated-frequency" if dilated else "iau2006-undilated-frequency"
    )

    convertible, invariant, unsupported = _parameter_conversion_plan(model, dilated)

    if model["UNITS"].value == target_units or (
        model["UNITS"].value is None and not backwards
    ):
        log.warning("The input par file is already in the target units. Doing nothing.")
        graph_unsupported, unaudited_components = _active_unsupported_components(
            model, convertible, invariant, dilated
        )
        unsupported.update(graph_unsupported)
        report = TCBTDBConversionReport(
            source_units=target_units,
            target_units=target_units,
            convention=convention,
            converted=(),
            invariant=tuple(sorted(invariant)),
            unsupported=tuple(sorted(unsupported)),
            unaudited_components=tuple(sorted(unaudited_components)),
        )
        model.tcb_tdb_conversion_report = report
        return report

    for name in convertible:
        param = model[name]
        if isinstance(param, (floatParameter, AngleParameter, maskParameter)) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, (floatParameter, AngleParameter))
        ):
            scale_parameter(model, name, _scale_exponent(param, dilated), backwards)
        else:
            transform_mjd_parameter(model, name, backwards)

    graph_unsupported, unaudited_components = _active_unsupported_components(
        model, convertible, invariant, dilated
    )
    unsupported.update(graph_unsupported)

    model["UNITS"].value = target_units

    model.validate(allow_tcb=backwards)

    report = TCBTDBConversionReport(
        source_units=source_units,
        target_units=target_units,
        convention=convention,
        converted=tuple(sorted(convertible)),
        invariant=tuple(sorted(invariant)),
        unsupported=tuple(sorted(unsupported)),
        unaudited_components=tuple(sorted(unaudited_components)),
    )
    model.tcb_tdb_conversion_report = report

    if dilated:
        log.warning(
            "The input par declares DILATEFREQ Y. DM-family coefficients were "
            "converted with the dilated-frequency exponents. PINT evaluates "
            "them undilated, which differs from a dilated evaluation by ~1e-9 "
            "of the dispersion delay (~1 ns at 400 MHz for DM=50); the "
            "affected components are reported unaudited."
        )

    if not report.accepted:
        log.warning(
            "TCB/TDB conversion completed, but the no-refit accuracy contract "
            "does not cover this model. Unsupported parameters: {}. "
            "Unaudited components: {}.",
            ", ".join(report.unsupported) or "none",
            ", ".join(report.unaudited_components) or "none",
        )

    return report
