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

import pint.models.model_builder as mb
import pint.simulation as sim
from pint.models.binary_bt import BinaryBT2
from pint.models.binary_dd import BinaryDD, BinaryDD2
from pint.residuals import Residuals
from pinttestdata import datadir

TRIPLE_PAR = os.path.join(datadir, "B1855+09_triple_DD.par")


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
    for cls in (BinaryDD2, BinaryBT2):
        outer = cls()
        assert outer.category == "pulsar_system_outer"
        assert outer.param_suffix == "_2"
        assert outer.binary_param_tag == "BINARY2"
        assert "PB_2" in outer.params
        assert "PB" not in outer.params

    # The inner DD model is unchanged.
    inner = BinaryDD()
    assert inner.category == "pulsar_system"
    assert inner.param_suffix == ""
    assert "PB" in inner.params
