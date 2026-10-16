"""Tests for hierarchical triple systems (two binary components).

A hierarchical triple is modelled with a normal inner binary (``BINARY``,
parameters ``PB``, ``A1``, ...) plus an outer-orbit binary (``BINARY2``,
parameters ``PB_2``, ``A1_2``, ...). The outer-orbit component belongs to the
``pulsar_system_outer`` category, which is ordered before ``pulsar_system`` in
:data:`pint.models.timing_model.DEFAULT_ORDER`, so its delay is accumulated
first and propagated into the inner binary's evaluation epoch.
"""

import io
import os

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.models.model_builder as mb
import pint.simulation as sim
from pint.models.binary_bt import BinaryBT2
from pint.models.binary_dd import BinaryDD, BinaryDD2
from pint.models.binary_ell1 import BinaryELL12
from pint.residuals import Residuals

TRIPLE_PAR = os.path.join(datadir, "B1855+09_triple_DD.par")
TRIPLE_PAR_DD = TRIPLE_PAR

TRIPLE_PAR_BT = """\
PSRJ           J1737_triple_BT
RAJ            17:37:47.11235
DECJ           -08:11:08.887
F0             239.51996484444
F1             -4.55E-16
PEPOCH         54987
DM             55.311
BINARY         BT
PB             79.517379
ECC            5.38E-5
A1             9.332791
T0             54696.879781933
OM             49.8
BINARY2        BT
PB_2           1400.0
T0_2           54696.0
A1_2           120.0
OM_2           110.0
ECC_2          0.3
TZRMJD         54987
TZRFRQ         1400
TZRSITE        @
CLK            TT(TAI)
UNITS          TDB
EPHEM          DE405
"""

TRIPLE_PAR_ELL1 = """\
PSRJ           J0023_triple_ELL1
ELONG          9.07039380
ELAT           6.30910853
F0             327.8470205906107
F1             -1.22783E-15
PEPOCH         56567
DM             14.32810
BINARY         ELL1
PB             0.138799
A1             0.03484142
TASC           56567.02609362
EPS1           7.2E-6
EPS2           -4.0E-6
BINARY2        ELL1
PB_2           1400.0
A1_2           120.0
TASC_2         56567.0
EPS1_2         0.01
EPS2_2         0.02
TZRMJD         56567
TZRFRQ         1400
TZRSITE        @
CLK            TT(TAI)
UNITS          TDB
EPHEM          DE436
"""

FAMILY = {
    "DD": {
        "par": TRIPLE_PAR_DD,
        "params": ["A1DOT2", "A1DOT", "T0", "A1_2", "T0_2"],
    },
    "BT": {
        "par": TRIPLE_PAR_BT,
        "params": ["A1DOT2", "A1DOT", "T0", "A1_2", "T0_2"],
    },
    "ELL1": {
        "par": TRIPLE_PAR_ELL1,
        "params": ["A1DOT2", "A1DOT", "TASC", "A1_2", "TASC_2"],
    },
}

STEPS = {
    "A1DOT2": 1e-22,
    "A1DOT": 1e-15,
    "T0": 1e-6,
    "TASC": 1e-6,
    "A1_2": 1e-4,
    "T0_2": 1e-4,
    "TASC_2": 1e-4,
}


def _inner_only_par():
    """Return the triple parfile text with the BINARY2/outer lines removed."""
    lines = []
    with open(TRIPLE_PAR) as f:
        for line in f:
            key = line.split()[0] if line.split() else ""
            if key == "BINARY2" or key.endswith("_2"):
                continue
            lines.append(line)
    return "".join(lines)


def _load_family_model(family):
    par = FAMILY[family]["par"]
    if family == "DD":
        return mb.get_model(par)
    return mb.get_model(io.StringIO(par))


def _family_toas(model):
    return sim.make_fake_toas_uniform(
        model.PEPOCH.value - 200,
        model.PEPOCH.value + 800,
        50,
        model,
        freq=1400 * u.MHz,
        add_noise=False,
    )


@pytest.fixture(scope="module")
def triple_model():
    return mb.get_model(TRIPLE_PAR)


@pytest.fixture(scope="module")
def toas(triple_model):
    return sim.make_fake_toas_uniform(
        53400, 55000, 50, triple_model, freq=1400 * u.MHz, add_noise=False
    )


def test_two_binary_components_built(triple_model):
    """Both an inner and an outer binary component are present."""
    assert "BinaryDD" in triple_model.components
    assert "BinaryDD2" in triple_model.components
    assert triple_model.components["BinaryDD"].category == "pulsar_system"
    assert triple_model.components["BinaryDD2"].category == "pulsar_system_outer"
    assert triple_model.BINARY.value == "DD"
    assert triple_model.BINARY2.value == "DD"


def test_outer_ordered_before_inner(triple_model):
    """The outer binary must be evaluated before the inner one so that its
    delay propagates into the inner orbit."""
    order = [c.__class__.__name__ for c in triple_model.DelayComponent_list]
    assert order.index("BinaryDD2") < order.index("BinaryDD")


def test_parameters_resolve_to_correct_component(triple_model):
    """Canonical names resolve to the inner binary and ``_2`` names to the
    outer binary, with no cross-contamination."""
    assert np.isclose(triple_model.PB.quantity.to_value(u.day), 12.327171194774200418)
    assert triple_model.PB_2.quantity == 1400.0 * u.day
    assert np.isclose(triple_model.A1_2.value, 120.0)
    assert triple_model.OM_2.quantity == 110.0 * u.deg
    # The outer component exposes suffixed names only (canonical names removed).
    outer = triple_model.components["BinaryDD2"]
    assert "PB_2" in outer.params
    assert "PB" not in outer.params
    assert not hasattr(outer, "PB")


def test_parfile_roundtrip(triple_model):
    """``BINARY2`` and the ``_2`` parameters survive a parfile round-trip."""
    s = triple_model.as_parfile()
    par_lines = [line.split() for line in s.splitlines() if line.split()]
    assert ["BINARY", "DD"] in par_lines
    assert ["BINARY2", "DD"] in par_lines
    assert any(parts[0] == "PB_2" for parts in par_lines)
    assert any(parts[0] == "A1_2" for parts in par_lines)
    # BINARY/BINARY2 should each appear exactly once.
    assert sum(parts[0] == "BINARY" for parts in par_lines) == 1
    assert sum(parts[0] == "BINARY2" for parts in par_lines) == 1

    m2 = mb.get_model(io.StringIO(s))
    assert m2.PB_2.quantity == triple_model.PB_2.quantity
    assert m2.A1_2.quantity == triple_model.A1_2.quantity
    assert m2.BINARY2.value == "DD"


def test_residuals_finite(triple_model, toas):
    res = Residuals(toas, triple_model).time_resids
    assert np.all(np.isfinite(res.value))


def test_outer_orbit_affects_delay(triple_model, toas):
    """Switching the outer orbit on/off changes the total delay substantially."""
    d_on = triple_model.delay(toas)
    m_off = mb.get_model(TRIPLE_PAR)
    m_off.A1_2.value = 0.0
    d_off = m_off.delay(toas)
    # A1_2 = 120 ls means the outer Roemer delay reaches ~100 s.
    assert np.max(np.abs((d_on - d_off).to_value(u.s))) > 1.0


def test_outer_delay_propagates_into_inner(triple_model, toas):
    """The defining feature of a hierarchical triple: the inner binary is
    evaluated at an epoch shifted by the outer orbit's light-travel delay,
    rather than the two binaries being treated independently."""
    m_inner = mb.get_model(io.StringIO(_inner_only_par()))

    # Trigger a delay computation so each inner binary instance caches the
    # barycentric time it was evaluated at.
    triple_model.delay(toas)
    m_inner.delay(toas)

    t_triple = triple_model.components["BinaryDD"].binary_instance.t
    t_alone = m_inner.components["BinaryDD"].binary_instance.t

    shift = (t_triple - t_alone).to_value(u.s)
    # The inner orbit's evaluation epoch is shifted by the outer delay (seconds
    # scale), which is exactly the coupling that cures the apparent PBDOT etc.
    assert np.max(np.abs(shift)) > 1.0


def test_naive_sum_differs_from_coupled(triple_model, toas):
    """The coupled triple delay differs from naively adding an independent
    inner-binary delay and an independent outer-binary delay."""
    inner_comp = triple_model.components["BinaryDD"]
    outer_comp = triple_model.components["BinaryDD2"]

    # Coupled: inner sees the accumulated outer delay.
    coupled_total = triple_model.delay(toas)

    # Naive sum: evaluate each binary at the same (outer-free) accumulated delay.
    acc_before = triple_model.delay(
        toas, cutoff_component="BinaryDD2", include_last=False
    )
    outer_only = outer_comp.binarymodel_delay(toas, acc_before)
    inner_only = inner_comp.binarymodel_delay(toas, acc_before)
    naive_total = acc_before + outer_only + inner_only

    diff = np.max(np.abs((coupled_total - naive_total).to_value(u.s)))
    assert diff > 0.0


def test_outer_param_derivative(triple_model, toas):
    """Derivatives with respect to an outer (suffixed) parameter are available
    and non-trivial, so the outer orbit is fittable."""
    d = triple_model.d_delay_d_param(toas, "A1_2")
    assert np.all(np.isfinite(d.value))
    assert np.any(d.value != 0)


def test_outer_wrapper_classes():
    """The outer wrappers are configured for the BINARY2 tag and _2 suffix."""
    for cls in (BinaryDD2, BinaryBT2, BinaryELL12):
        outer = cls()
        assert outer.category == "pulsar_system_outer"
        assert outer.param_suffix == "_2"
        assert outer.binary_param_tag == "BINARY2"
        assert "PB_2" in outer.params
        assert "PB" not in outer.params

    ell = BinaryELL12()
    assert "TASC_2" in ell.params
    assert "TASC" not in ell.params

    # The inner DD model is unchanged.
    inner = BinaryDD()
    assert inner.category == "pulsar_system"
    assert inner.param_suffix == ""
    assert "PB" in inner.params


def test_derived_params_report_inner_binary(triple_model):
    """Summary / derived-parameter text should name the inner BINARY component."""
    text, info = triple_model.get_derived_params(returndict=True)
    assert info["Binary"] == "BinaryDD"
    assert "Binary model BinaryDD" in text
    assert "BinaryDD2" not in text.split("Binary model")[1].splitlines()[0]


def test_convert_binary_rejects_triple(triple_model):
    """convert_binary must not silently operate on a hierarchical triple."""
    import pint.binaryconvert

    with pytest.raises(ValueError, match="multiple binary components"):
        pint.binaryconvert.convert_binary(triple_model, "BT")


def test_outer_missing_parameter_uses_suffixed_name():
    """MissingParameter messages for outer orbits should cite T0_2, not T0."""
    from pint.exceptions import MissingParameter

    outer = BinaryDD2()
    # Leave T0_2 unset; give A1_2 a value so only T0_2 is reported missing.
    outer.A1_2.value = 1.0
    with pytest.raises(MissingParameter, match="T0_2") as exc_info:
        outer.validate()
    assert exc_info.value.param == "T0_2"


def test_a1dot2_changes_delay(toas):
    """A nonzero A1DOT2 (second derivative of the projected semi-major axis)
    changes the inner-binary delay."""
    m = mb.get_model(TRIPLE_PAR)
    d0 = m.delay(toas)
    m.A1DOT2.quantity = 1e-18 * u.lsec / u.s**2
    d1 = m.delay(toas)
    # 0.5 * A1DOT2 * tt0**2 with tt0 up to ~1e8 s gives a delay change of
    # order milliseconds; just require a clearly nonzero effect.
    assert np.max(np.abs((d1 - d0).to_value(u.s))) > 1e-9


@pytest.mark.parametrize(
    "family,param",
    [(fam, p) for fam, cfg in FAMILY.items() for p in cfg["params"]],
)
def test_delay_derivatives_match_numerical(family, param):
    """Analytic delay derivatives (including A1DOT2 and the chain rule through
    the outer->inner delay coupling) agree with central finite differences."""
    m = _load_family_model(family)
    toas = _family_toas(m)
    m.A1DOT.quantity = 3e-13 * u.lsec / u.s
    m.A1DOT2.quantity = 2e-21 * u.lsec / u.s**2

    ana = m.d_delay_d_param(toas, param)
    q = getattr(m, param)
    v0, h = q.value, STEPS[param]
    q.value = v0 + h
    dp = m.delay(toas)
    q.value = v0 - h
    dm = m.delay(toas)
    q.value = v0
    num = (dp - dm) / (2 * h * q.units)
    a = ana.to_value(num.unit)
    n = num.value
    scale = np.max(np.abs(n))
    assert scale > 0
    assert np.max(np.abs(a - n)) / scale < 1e-4


@pytest.mark.parametrize("family", ["BT", "ELL1"])
def test_outer_orbit_affects_delay_family(family):
    par = FAMILY[family]["par"]
    m = mb.get_model(io.StringIO(par))
    toas = sim.make_fake_toas_uniform(
        m.PEPOCH.value - 200,
        m.PEPOCH.value + 800,
        40,
        m,
        freq=1400 * u.MHz,
    )
    d0 = m.delay(toas)
    m.A1_2.quantity = 0 * u.lsec
    d1 = m.delay(toas)
    assert np.max(np.abs((d1 - d0).to_value(u.s))) > 1e-9


def _fb_triple_par(fb1=None):
    """Return the DD triple parfile text with the inner orbit expressed in
    orbital frequency (``FB0`` = 1/``PB``, optionally ``FB1``) instead of ``PB``."""
    lines = []
    with open(TRIPLE_PAR) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "PB":
                pb_s = np.longdouble(parts[1]) * 86400
                lines.append(f"FB0 {1 / pb_s:.25e} 1\n")
                if fb1 is not None:
                    lines.append(f"FB1 {fb1:.25e} 1\n")
                continue
            lines.append(line)
    return "".join(lines)


@pytest.mark.parametrize("fb1", [None, -3.0e-20])
def test_inner_orbit_fb_parameterization(triple_model, toas, fb1):
    """The inner orbit of a hierarchical triple can use the orbital-frequency
    (``FBn``) parameterization: ``FB0``/``FB1`` are owned by the inner
    ``BinaryDD``, the outer ``BinaryDD2`` has no ``FBn`` parameters, and the
    delay matches the equivalent ``PB``/``PBDOT`` triple to numerical noise."""
    m_fb = mb.get_model(io.StringIO(_fb_triple_par(fb1)))

    inner = m_fb.components["BinaryDD"]
    outer = m_fb.components["BinaryDD2"]
    assert "FB0" in inner.params
    assert inner.FB0.value is not None
    assert m_fb.PB.value is None
    assert not any(p.startswith("FB") for p in outer.params)
    assert "PB_2" in outer.params
    assert m_fb.PB_2.quantity == triple_model.PB_2.quantity

    d_fb = m_fb.delay(toas)
    assert np.all(np.isfinite(d_fb.value))

    # PB-equivalent reference: PBDOT = -FB1 / FB0**2.
    m_pb = mb.get_model(TRIPLE_PAR)
    if fb1 is not None:
        assert "FB1" in inner.params
        m_pb.PBDOT.quantity = (-m_fb.FB1.quantity / m_fb.FB0.quantity**2).to(
            u.dimensionless_unscaled
        )
    d_pb = m_pb.delay(toas)

    diff = np.max(np.abs((d_fb - d_pb).to_value(u.s)))
    assert diff < 1e-11

    # The FBn parameters of the inner orbit remain fittable inside the triple.
    for p in ["FB0"] + (["FB1"] if fb1 is not None else []):
        d = m_fb.d_delay_d_param(toas, p)
        assert np.all(np.isfinite(d.value))
        assert np.any(d.value != 0)
