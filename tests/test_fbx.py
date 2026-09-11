"""Various tests to assess the performance of the FBX model."""

import os

import astropy.units as u
import numpy as np
import pytest
import test_derivative_utils as tdu
from pinttestdata import datadir

from pint import fitter
from pint.models import get_model_and_toas
import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pint.models.tcb_conversion import convert_tcb_tdb
from pint.models.stand_alone_psr_binaries.DD_model import DDmodel
from pint.models.stand_alone_psr_binaries.binary_orbits import OrbitFBX
from pint.models.parameter import funcParameter
from pint.models.binary_ell1 import BinaryELL1
from pint.models.binary_dd import BinaryDDGR
import pint.simulation
from loguru import logger as log
from copy import deepcopy
from contextlib import contextmanager
import io

parfileJ0023 = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par")
parJ0023ell1 = os.path.join(datadir, "J0023+0923_ell1_simple.par")
timJ0023 = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.tim")


@pytest.fixture
def toasJ0023():
    return toa.get_TOAs(timJ0023, ephem="DE436", planets=False)


@pytest.fixture
def modelJ0023():
    return mb.get_model(parfileJ0023)


ltres, ltbindelay = np.genfromtxt(
    f"{parfileJ0023}.tempo2_test", skip_header=1, unpack=True
)


def test_J0023_binary_delay(modelJ0023, toasJ0023):
    # Calculate binary delays with PINT
    pint_binary_delay = modelJ0023.binarymodel_delay(toasJ0023, None)
    assert np.all(np.abs(pint_binary_delay.value + ltbindelay) < 1e-9)


@pytest.mark.xfail(reason="PINT has a more modern position for Arecibo than TEMPO2")
def test_J0023(modelJ0023, toasJ0023):
    pint_resids_us = Residuals(
        toasJ0023, modelJ0023, use_weighted_mean=False
    ).time_resids.to(u.s)
    assert np.all(np.abs(pint_resids_us.value - ltres) < 1e-8)


def test_derivative(modelJ0023, toasJ0023):
    testp = tdu.get_derivative_params(modelJ0023)
    delay = modelJ0023.delay(toasJ0023)
    for p in testp.keys():
        print("Runing derivative for %s", f"d_delay_d_{p}")
        if p in ["EPS2", "EPS1"]:
            testp[p] = 15
        ndf = modelJ0023.d_phase_d_param_num(toasJ0023, p, testp[p])
        adf = modelJ0023.d_phase_d_param(toasJ0023, delay, p)
        diff = adf - ndf
        if np.all(diff.value) != 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            if p in ["PMELONG", "ELONG"]:
                tol = 2e-2
            elif p in ["FB1"]:
                # paulr added this to make tests pass with oldest supported versions of numpy/astropy, but I don't know why it is needed
                # How should we decide what the acceptable tolerance is? This should not just be a random choice.
                tol = 0.002
            elif p in ["FB2", "FB3"]:
                tol = 0.08
            else:
                tol = 1e-3
            print(
                (
                    "derivative relative diff for %s, %lf"
                    % (f"d_delay_d_{p}", np.nanmax(relative_diff).value)
                )
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue


def test_summary_FB():
    m, t = get_model_and_toas(
        os.path.join(datadir, parJ0023ell1), os.path.join(datadir, timJ0023)
    )
    f = fitter.WLSFitter(toas=t, model=m)

    # Ensure print_summary runs without an exception for an ELL1 model with FBX
    f.print_summary()

    assert "PB" in f.get_summary()


# ---- hybrid/sparse FBX normalization ----

hybridbase = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
BINARY ELL1
A1 10
TASC 57000
EPS1 0
EPS2 0
"""

genericbase = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
"""

PB_DAYS = np.longdouble("0.5")
FB0_FROM_PB = 1 / (PB_DAYS * np.longdouble(86400))
DT = np.asarray([0, 1.0e5, 2.0e5], dtype=np.longdouble)


@contextmanager
def captured_warnings():
    messages = []
    sink = log.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield messages
    finally:
        log.remove(sink)


def orbit_count(model, dt_seconds):
    binary = model.components[
        next(name for name in model.components if name.startswith("Binary"))
    ].binary_instance
    reference_epoch = binary.TASC if hasattr(binary, "TASC") else binary.T0
    binary.t = reference_epoch + np.atleast_1d(dt_seconds) * u.s
    return binary.orbits_cls.orbits().to_value(u.dimensionless_unscaled)


def test_pb_fb2_inserts_zero_and_uses_fb2():
    fb2 = np.longdouble("1e-27")
    with captured_warnings() as messages:
        model = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS}\nFB2 {fb2}\n"))

    assert model.FB1.value == 0
    assert model.FB1.frozen
    expected = FB0_FROM_PB * DT + fb2 * DT**3 / 6
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)
    assert any("frozen zero" in message for message in messages)


def test_pb_pbdot_fb2_uses_both_derivatives():
    pbdot = np.longdouble("1e-12")
    fb2 = np.longdouble("1e-27")
    model = mb.get_model(
        io.StringIO(hybridbase + f"PB {PB_DAYS}\nPBDOT {pbdot}\nFB2 {fb2}\n")
    )

    fb1 = -pbdot * FB0_FROM_PB**2
    expected = FB0_FROM_PB * DT + fb1 * DT**2 / 2 + fb2 * DT**3 / 6
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)


def test_explicit_fb0_pbdot_is_not_ignored():
    pbdot = np.longdouble("1e-12")
    model = mb.get_model(
        io.StringIO(hybridbase + f"FB0 {FB0_FROM_PB}\nPBDOT {pbdot}\n")
    )

    expected_fb1 = -pbdot * FB0_FROM_PB**2
    assert np.isclose(np.longdouble(str(model.FB1.value)), expected_fb1, rtol=1e-14)
    expected = FB0_FROM_PB * DT + expected_fb1 * DT**2 / 2
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)


def test_pb_fb1_bridges_to_fb0():
    with captured_warnings() as messages:
        model = mb.get_model(
            io.StringIO(hybridbase + f"PB {PB_DAYS} 1 1e-10\nFB1 1e-21\n")
        )
    assert np.isclose(np.longdouble(str(model.FB0.value)), FB0_FROM_PB, rtol=1e-14)
    assert isinstance(model.PB, funcParameter)
    assert any("PB" in message and "FB0" in message for message in messages)


def test_pb_fb1_existing_pbdot_fb1_wins():
    with captured_warnings() as messages:
        model = mb.get_model(
            io.StringIO(hybridbase + f"PB {PB_DAYS}\nPBDOT 1e-12\nFB1 3e-21\n")
        )
    assert np.isclose(np.longdouble(str(model.FB1.value)), 3e-21, rtol=1e-14)
    assert any("PBDOT" in message and "FB1" in message for message in messages)
    expected_pbdot = -3e-21 / FB0_FROM_PB**2
    assert np.isclose(
        model.PBDOT.quantity.to_value(u.s / u.s), expected_pbdot, rtol=1e-10
    )


def test_pb_fb1_fb3_inserts_fb2_zero():
    with captured_warnings() as messages:
        model = mb.get_model(
            io.StringIO(hybridbase + f"PB {PB_DAYS}\nFB1 1e-21\nFB3 1e-40\n")
        )
    assert model.FB2.value == 0
    assert model.FB2.frozen
    assert any("frozen zero" in message for message in messages)
    expected = (
        FB0_FROM_PB * DT
        + np.longdouble("1e-21") * DT**2 / 2
        + np.longdouble("1e-40") * DT**4 / 24
    )
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)


def test_explicit_fb0_fb2_inserts_fb1_zero():
    model = mb.get_model(io.StringIO(hybridbase + f"FB0 {FB0_FROM_PB}\nFB2 1e-27\n"))
    assert model.FB1.value == 0
    assert model.FB1.frozen
    expected = FB0_FROM_PB * DT + np.longdouble("1e-27") * DT**3 / 6
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)


def test_hybrid_matches_manual_contiguous_fbx_phase():
    hybrid = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS}\nFB2 1e-27\n"))
    manual = mb.get_model(
        io.StringIO(hybridbase + f"FB0 {FB0_FROM_PB}\nFB1 0\nFB2 1e-27\n")
    )
    assert np.allclose(
        orbit_count(hybrid, DT), orbit_count(manual, DT), rtol=1e-14, atol=0
    )


def test_fb1_uncertainty_uses_pbdot_and_fb0_terms():
    pb = np.longdouble("0.5")
    sigma_pb = np.longdouble("2e-7")
    pbdot = np.longdouble("1e-12")
    sigma_pbdot = np.longdouble("3e-14")
    model = mb.get_model(
        io.StringIO(
            hybridbase
            + f"PB {pb} 1 {sigma_pb}\n"
            + f"PBDOT {pbdot} 1 {sigma_pbdot}\n"
            + "FB2 1e-27\n"
        )
    )

    fb0 = 1 / (pb * np.longdouble(86400))
    sigma_fb0 = sigma_pb / (pb**2 * np.longdouble(86400))
    expected = np.sqrt((fb0**2 * sigma_pbdot) ** 2 + (2 * pbdot * fb0 * sigma_fb0) ** 2)
    assert np.isclose(model.FB0.uncertainty_value, sigma_fb0, rtol=1e-12)
    assert np.isclose(model.FB1.uncertainty_value, expected, rtol=1e-12)
    assert not model.FB0.frozen
    assert not model.FB1.frozen


def test_pb_fit_flag_transfers_to_fb0():
    model = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS} 1 1e-10\nFB2 1e-27\n"))
    assert not model.FB0.frozen
    assert "FB0" in model.free_params
    assert "PB" not in model.free_params


def test_pure_pb_is_unchanged():
    model = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS} 1 1e-10\n"))
    assert not isinstance(model.PB, funcParameter)
    assert model.FB0.quantity is None
    assert "PB" in model.free_params


def test_higher_fbn_without_pb_or_fb0_raises():
    with pytest.raises(ValueError, match="FB0"):
        mb.get_model(io.StringIO(hybridbase + "FB1 1e-21\n"))


def test_nonpositive_hybrid_pb_raises():
    with pytest.raises(ValueError, match="PB must be positive"):
        mb.get_model(io.StringIO(hybridbase + "PB -0.5\nFB2 1e-27\n"))


def test_xpbdot_with_fbx_raises():
    component = BinaryDDGR()
    component.PB.value = np.longdouble("0.5")
    component.XPBDOT.value = np.longdouble("1e-12")
    component.add_param(component.FB0.new_param(2))
    component.FB2.value = np.longdouble("1e-27")
    with pytest.raises(ValueError, match=r"XPBDOT.*FBX"):
        component._setup_fbx_parameterization()


def test_fb0_pb_conflict_raises_before_mutation():
    component = BinaryELL1()
    component.PB.value = np.longdouble("0.5")
    component.FB0.value = np.longdouble("2.3148148148148148e-5")
    original_pb = component.PB
    original_fb0 = component.FB0.value
    original_params = list(component.params)

    with pytest.raises(
        ValueError, match="Model cannot have values for both FB0 and PB"
    ):
        component._setup_fbx_parameterization()

    assert component.PB is original_pb
    assert component.FB0.value == original_fb0
    assert component.params == original_params


def test_fbx_canonicalization_is_idempotent():
    model = mb.get_model(io.StringIO(hybridbase + "PB 0.5\nFB2 1e-27\n"))
    pb_object = model.PB
    pbdot_object = model.PBDOT
    params = list(model.params)
    values = {name: getattr(model, name).value for name in ("FB0", "FB1", "FB2")}

    with captured_warnings() as messages:
        model.setup()
        model.setup()

    assert model.PB is pb_object
    assert model.PBDOT is pbdot_object
    assert model.params == params
    assert {
        name: getattr(model, name).value for name in ("FB0", "FB1", "FB2")
    } == values
    assert not any("Converting PB" in message for message in messages)
    assert not any("frozen zero" in message for message in messages)


def test_all_fbx_models_expose_derived_pb():
    model = mb.get_model(io.StringIO(hybridbase + f"FB0 {FB0_FROM_PB}\nFB1 0\n"))
    assert isinstance(model.PB, funcParameter)
    assert np.isclose(model.PB.quantity.to_value(u.day), float(PB_DAYS), rtol=1e-14)


def test_fbx_roundtrip_preserves_derived_view_api():
    first = mb.get_model(io.StringIO(hybridbase + "PB 0.5\nPBDOT 1e-12\nFB2 1e-27\n"))
    second = mb.get_model(io.StringIO(first.as_parfile()))

    assert isinstance(first.PB, funcParameter)
    assert isinstance(second.PB, funcParameter)
    assert isinstance(first.PBDOT, funcParameter)
    assert isinstance(second.PBDOT, funcParameter)
    for name in ("FB0", "FB1", "FB2"):
        assert getattr(second, name).frozen == getattr(first, name).frozen
        assert np.isclose(
            getattr(second, name).value,
            getattr(first, name).value,
            rtol=1e-14,
        )
    assert "# PB" in first.as_parfile()
    assert "# PBDOT" in first.as_parfile()


def test_fbx_fit_uses_only_fbx_coefficients():
    model = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS} 1 1e-10\nFB2 1e-27\n"))
    toas = pint.simulation.make_fake_toas_uniform(
        54000, 56000, 80, model, add_noise=True
    )
    for name in model.free_params:
        if name != "FB0":
            getattr(model, name).frozen = True
    f = fitter.Fitter.auto(toas, model)
    f.fit_toas()
    assert "FB0" in f.model.free_params
    assert "PB" not in f.model.free_params
    assert np.isfinite(f.resids.calc_chi2())


def test_tcb_conversion_commutes_with_fbx_normalization():
    text = hybridbase + "UNITS TCB\nPB 0.5\nPBDOT 1e-12\nFB2 1e-27\n"
    raw = mb.get_model(io.StringIO(text), allow_tcb="raw")
    converted = deepcopy(raw)
    convert_tcb_tdb(converted)
    automatic = mb.get_model(io.StringIO(text), allow_tcb=True)

    for name in ("FB0", "FB1", "FB2", "PB", "PBDOT"):
        assert np.isclose(
            getattr(converted, name).value,
            getattr(automatic, name).value,
            rtol=1e-14,
        )
    assert isinstance(converted.PB, funcParameter)
    assert isinstance(converted.PBDOT, funcParameter)


def test_orbitfbx_rejects_caller_supplied_gap():
    parent = DDmodel()
    parent.FB0 = 1e-5 / u.s
    parent.add_binary_params("FB2", 1e-27 / u.s**3)
    with pytest.raises(
        ValueError, match="Indices must be 0 up to some number k without gaps"
    ):
        OrbitFBX(parent, ["FB0", "FB2"])


@pytest.mark.parametrize(
    "binary_block",
    [
        """
BINARY ELL1
A1 10
TASC 57000
EPS1 0
EPS2 0
""",
        """
BINARY DD
A1 10
T0 57000
ECC 0.1
OM 45
""",
        """
BINARY BT
A1 10
T0 57000
ECC 0.1
OM 45
""",
    ],
)
def test_fbx_normalization_binary_model_independent(binary_block):
    fb2 = np.longdouble("1e-27")
    model = mb.get_model(
        io.StringIO(genericbase + binary_block + f"\nPB {PB_DAYS}\nFB2 {fb2}\n")
    )
    expected = FB0_FROM_PB * DT + fb2 * DT**3 / 6
    assert np.allclose(orbit_count(model, DT), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("extra", ["FB2 1e-27", "FB1 -1e-18"])
def test_ddgr_fbx_rejected_before_mutation(extra):
    component = BinaryDDGR()
    component.PB.value = np.longdouble("0.5")
    component.M2.value = np.longdouble("0.3")
    component.MTOT.value = np.longdouble("1.7")
    name, value = extra.split()
    component.add_param(component.FB0.new_param(int(name[2:])))
    getattr(component, name).value = np.longdouble(value)

    original_pb = component.PB
    original_pbdot = component.PBDOT
    original_fb0 = component.FB0.value
    original_params = list(component.params)

    with pytest.raises(ValueError, match=r"FBX.*derived PBDOT"):
        component._setup_fbx_parameterization()

    assert component.PB is original_pb
    assert component.PBDOT is original_pbdot
    assert component.FB0.value == original_fb0
    assert component.params == original_params


def test_ddgr_pb_only_is_unchanged():
    # Use masses/A1 consistent with an inclination that keeps derived SINI in range.
    from pint import derived_quantities

    mp = 1.4 * u.Msun
    mc = 0.3 * u.Msun
    pb = 0.5 * u.day
    a1 = derived_quantities.a1sini(mp, mc, pb, 75 * u.deg)
    text = f"""
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
BINARY DDGR
A1 {a1.value}
T0 57000
ECC 0.1
OM 45
PB {pb.value}
M2 {mc.value}
MTOT {(mp + mc).value}
"""
    model = mb.get_model(io.StringIO(text))
    assert not isinstance(model.PB, funcParameter)
    assert model.FB0.quantity is None
    assert isinstance(model.PBDOT, funcParameter)


# ---- pure-FB0 consumer compatibility (API fallout from derived PB / absent PBDOT) ----

pure_fb0_ell1 = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
BINARY ELL1
A1 10
TASC 57000
EPS1 0
EPS2 0
FB0 2.3148148148148148e-5
"""


@pytest.mark.parametrize(
    "binary_block",
    [
        """
BINARY ELL1
A1 10
TASC 57000
EPS1 0
EPS2 0
""",
        """
BINARY DD
A1 10
T0 57000
ECC 0.1
OM 45
""",
        """
BINARY BT
A1 10
T0 57000
ECC 0.1
OM 45
""",
        """
BINARY BT_piecewise
A1 10
T0 57000
ECC 0.1
OM 45
""",
    ],
)
def test_pure_fb0_loads_without_writable_pbdot(binary_block):
    model = mb.get_model(
        io.StringIO(genericbase + binary_block + f"\nFB0 {FB0_FROM_PB}\n")
    )
    assert isinstance(model.PB, funcParameter)
    assert "PBDOT" not in model.params
    assert np.isclose(model.PB.quantity.to_value(u.day), float(PB_DAYS), rtol=1e-14)


def test_pure_fb0_pb_uses_fbx_coefficients():
    model = mb.get_model(io.StringIO(pure_fb0_ell1))
    period, _ = model.pb()
    assert np.isclose(period.to_value(u.s), 1 / float(FB0_FROM_PB), rtol=1e-14)


def test_hybrid_with_fb2_pb_uses_full_fb_series_not_pbdot_only():
    # pb() must use the FB Taylor branch so FB2 contributes at later times.
    model = mb.get_model(io.StringIO(hybridbase + f"PB {PB_DAYS}\nFB2 1e-20\n"))
    t = model.TASC.quantity + (2.0e7 * u.s)
    period, _ = model.pb(t)
    # Instantaneous period from dN/dt with FB0 and FB2 (FB1=0):
    # f(t) = FB0 + FB2 t^2 / 2 ; P = 1/f
    dt = 2.0e7
    freq = float(FB0_FROM_PB) + float(np.longdouble("1e-20")) * dt**2 / 2
    assert np.isclose(period.to_value(u.s), 1 / freq, rtol=1e-12)


def test_convert_binary_pure_fb0_ell1_to_dd():
    import pint.binaryconvert

    # Uncertainties on EPS1/EPS2 are required so ELL1→DD OM conversion is defined.
    par = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
BINARY ELL1
A1 10 0 1e-6
TASC 57000 0 1e-8
EPS1 -2e-5 0 1e-8
EPS2 2e-6 0 1e-8
SINI 0.9 0 0.01
M2 0.2 0 0.01
FB0 2.3148148148148148e-5 1 1e-18
"""
    model = mb.get_model(io.StringIO(par))
    assert isinstance(model.PB, funcParameter)
    assert "PBDOT" not in model.params
    converted = pint.binaryconvert.convert_binary(model, "DD")
    assert converted.BINARY.value == "DD"
    assert isinstance(converted.PB, funcParameter)
    # setup-before-validate must not invent PBDOT=0 → FB1=0.
    assert "FB1" not in converted.params
    assert "PBDOT" not in converted.params
    period, _ = converted.pb()
    assert np.isclose(period.to_value(u.d), model.pb()[0].to_value(u.d), rtol=1e-14)
