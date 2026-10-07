"""BINARY DDR wrap: load, composition, fit, FW, and cross-family delay differences.

Kernel identities live in ``test_ddr_kernel.py``. ``convert_binary`` rows live
in ``test_binconvert.py``.
"""

from __future__ import annotations

import warnings
from io import StringIO
from itertools import permutations
from types import SimpleNamespace

import astropy.coordinates as coords
import astropy.units as u
import numpy as np
import pytest

from pint.exceptions import InvalidModelParameters, TimingModelError
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.models.binary_ddr import (
    DDR_FB_KMAX,
    fw10_decode,
    fw10_encode,
    fw10_h3,
    fw10_h4,
    fw10_orbit_decode,
    fw10_orbit_encode,
    fw10_stigma,
)
from pint.models.parameter import funcParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.DDR_model import (
    _LD,
    DDRmodel,
    as_ld,
    inverse_timing,
    tsun_s,
    value,
)
from pint.simulation import make_fake_toas_uniform
from pint.utils import add_dummy_distance

_LD_EPS = float(np.finfo(np.longdouble).eps)
_IEEE_QUAD = _LD_EPS < 1e-30


def _ddr_rtol(tight, floor=2e-16):
    return max(float(tight), float(floor), 512.0 * _LD_EPS)


def _ddr_atol(tight, scale=1.0, floor=2e-16):
    mag = float(np.max(np.abs(np.asarray(scale, dtype=np.longdouble))))
    return max(float(tight), float(floor), 512.0 * _LD_EPS * (mag + 1.0))


def _ddr_fd_rtol(tight=5e-5, loose=3e-2):
    """Central-difference columns need more slop on 80-bit than on quad."""
    return float(tight) if _IEEE_QUAD else max(float(tight), float(loose))


def _ddr_peak_atol(scale, frac=2e-2, floor=1e-14):
    """Absolute floor relative to the peak of a Dual/FD column.

    Near a zero crossing, relative error is meaningless and 80-bit FD of an
    O(1) primitive (``d_I_au``) has an ulp/step noise floor around 1e-11.
    """
    mag = float(np.max(np.abs(np.asarray(scale, dtype=np.longdouble))))
    return max(_ddr_atol(floor, mag), frac * mag)


def _example_lines(**overrides):
    lines = {
        "PSRJ": "PSRJ            J0000+0000",
        "RAJ": "RAJ             00:00:00.0",
        "DECJ": "DECJ            +00:00:00.0",
        "F0": "F0              200",
        "PEPOCH": "PEPOCH          55000",
        "POSEPOCH": "POSEPOCH        55000",
        "DM": "DM              10",
        "PX": "PX              1.0",
        "PMRA": "PMRA            5.0",
        "PMDEC": "PMDEC           -3.0",
        "BINARY": "BINARY          DDR",
        "PB": "PB              2.0",
        "A1": "A1              1.963",
        "TASC": "TASC            55000",
        "EPS1": "EPS1            1.0e-6",
        "EPS2": "EPS2            -2.0e-6",
        "M2": "M2              0.2",
        "COSI": "COSI            0.5",
        "KOM": "KOM             30",
        "TGEO": "TGEO            55000",
        "DDRPK": "DDRPK           Y",
        "DDRPBDOT": "DDRPBDOT        kinematic",
        "DDRGEO": "DDRGEO          Y",
        "DDRKINE": "DDRKINE         Y",
        "UNITS": "UNITS           TDB",
    }
    for key, val in overrides.items():
        if val is None:
            lines.pop(key.upper(), None)
        else:
            name = key.upper()
            lines[name] = f"{name:16s}{val}"
    return lines


def example_par(**overrides):
    return "\n".join(_example_lines(**overrides).values()) + "\n"


def ecliptic_example_par(**overrides):
    defaults = dict(
        RAJ=None,
        DECJ=None,
        PMRA=None,
        PMDEC=None,
        ELONG="10.0",
        ELAT="20.0",
        PMELONG="5.0",
        PMELAT="-3.0",
        ECL="IERS2010",
    )
    defaults.update(overrides)
    return example_par(**defaults)


def _pheno(**overrides):
    defaults = dict(
        DDRPK="N",
        DDRPBDOT="absorb_gw",
        DDRGEO="N",
        DDRKINE="N",
        PBDOT="0",
        GGAMMA="0",
        OMDOT="0",
        PX=None,
        PMRA=None,
        PMDEC=None,
        KOM=None,
        TGEO=None,
    )
    defaults.update(overrides)
    return example_par(**defaults)


def _fbx(**overrides):
    defaults = dict(
        PB=None,
        PBDOT=None,
        FB0="5.787037037037037e-6",
        FB1="-1e-20",
        FB2="1e-28",
        FB3="-1e-36",
        FB4="1e-44",
        FB5="-1e-52",
    )
    defaults.update(overrides)
    return _pheno(**defaults)


# --- load / schema ---


def test_load_example_par():
    m = get_model(StringIO(example_par()))
    assert m.BINARY.value == "DDR"
    assert "BinaryDDR" in m.components
    assert m.TGEO.value == m.TASC.value
    assert m.TGEO.frozen
    assert isinstance(m.OMDOT, funcParameter)
    assert isinstance(m.PBDOT, funcParameter)
    assert isinstance(m.GAMMA, funcParameter)
    assert m.XPBDOT.value == 0
    par = m.as_parfile()
    assert "TGEO" in par
    assert m.ECC.value > 0


def test_tgeo_materialized_and_stable_after_tasc_change():
    m = get_model(StringIO(example_par(TGEO=None)))
    assert m.TGEO.value == pytest.approx(m.TASC.value)
    tgeo0 = m.TGEO.value
    m.TASC.value = 55100
    m.setup()
    assert m.TGEO.value == tgeo0
    assert "TGEO" in m.as_parfile()


@pytest.mark.parametrize("order", list(permutations(["DDRPK", "OMDOT", "GGAMMA"])))
def test_ddrpk_n_independent_omdot_ggamma_order(order):
    extras = {
        "DDRPK": "N",
        "OMDOT": "0.01 1",
        "GGAMMA": "0.002 1",
        "DDRPBDOT": "absorb_gw",
        "DDRGEO": "N",
        "DDRKINE": "N",
        "PX": None,
        "PMRA": None,
        "PMDEC": None,
        "KOM": None,
        "TGEO": None,
    }
    lines = _example_lines(**extras)
    keyed = dict(lines)
    body = []
    for name, line in lines.items():
        if name in order:
            if name == order[0]:
                for key in order:
                    body.append(keyed[key])
            continue
        body.append(line)
    m = get_model(StringIO("\n".join(body) + "\n"))
    assert not m.DDRPK.value
    assert not isinstance(m.OMDOT, funcParameter)
    assert m.OMDOT.value == pytest.approx(0.01)
    assert not m.OMDOT.frozen
    assert m.GGAMMA.value == pytest.approx(0.002)
    assert not m.GGAMMA.frozen


def test_ddrpk_y_rejects_explicit_omdot():
    with pytest.raises(TimingModelError, match="OMDOT"):
        get_model(StringIO(example_par(OMDOT="0.01 1")))


def test_ddrgeo_requires_ddrkine():
    with pytest.raises(TimingModelError, match="DDRKINE"):
        get_model(StringIO(example_par(DDRKINE="N")))


def test_kinematic_rejects_mismatched_pbdot():
    with pytest.raises(TimingModelError, match="PBDOT"):
        get_model(StringIO(example_par(PBDOT="1e-12")))


def test_kinematic_accepts_matching_pbdot():
    m = get_model(StringIO(example_par()))
    p = m.PBDOT.value
    m2 = get_model(StringIO(example_par(PBDOT=f"{p:.18e}")))
    assert m2.PBDOT.value == pytest.approx(p, rel=1e-10, abs=1e-20)


def test_om_dd_works_before_delay_on_kinematic_model():
    m = get_model(StringIO(example_par()))
    om = m.components["BinaryDDR"].om_dd()
    assert np.isfinite(om.to_value(u.deg))


def test_tcb_units_refused():
    with pytest.raises((TimingModelError, ValueError), match="TCB"):
        get_model(StringIO(example_par(UNITS="TCB")))


def test_nonzero_dr_dth_refused():
    with pytest.raises(TimingModelError, match="DR"):
        get_model(StringIO(example_par(DR="1e-6")))
    with pytest.raises(TimingModelError, match="DTH"):
        get_model(StringIO(example_par(DTH="1e-6")))


def test_nonzero_epsdot_edot_refused():
    with pytest.raises(TimingModelError, match="EPS1DOT"):
        get_model(StringIO(example_par(EPS1DOT="1e-12")))
    with pytest.raises(TimingModelError, match="EDOT"):
        get_model(StringIO(example_par(EDOT="1e-12")))


@pytest.mark.parametrize("name", ["EDOT", "EPS1DOT", "EPS2DOT", "DR", "DTH"])
def test_unsupported_zero_placeholder_must_be_frozen(name):
    with pytest.raises(TimingModelError, match=name):
        get_model(StringIO(_pheno(**{name: "0 1"})))


def test_orbwave_parameters_are_refused():
    with pytest.raises(TimingModelError, match="ORBWAVE"):
        get_model(
            StringIO(
                _pheno(
                    ORBWAVE_OM="1e-6",
                    ORBWAVE_EPOCH="55000",
                    ORBWAVEC0="1e-4",
                    ORBWAVES0="2e-4",
                )
            )
        )


def test_mode_flags_are_static_after_setup():
    m = get_model(StringIO(example_par()))
    omdot_type = type(m.OMDOT)
    pbdot_type = type(m.PBDOT)
    m.DDRPK.value = False
    with pytest.raises(TimingModelError, match="static"):
        m.setup()
    assert type(m.OMDOT) is omdot_type
    assert type(m.PBDOT) is pbdot_type


@pytest.mark.parametrize("name", ["DDRPK", "DDRPBDOT", "DDRGEO", "DDRKINE"])
def test_mode_flags_cannot_be_fitted(name):
    model = get_model(StringIO(example_par()))
    getattr(model, name).frozen = False
    with pytest.raises(TimingModelError, match=name):
        model.validate()


def test_fbx_chart_loads_through_fb5_and_registers_columns():
    m = get_model(
        StringIO(
            _fbx(
                **{
                    f"FB{j}": f"{(-1) ** j}e-{20 + 8 * max(j - 1, 0)} 1"
                    for j in range(1, 6)
                }
            )
        )
    )
    binary = m.components["BinaryDDR"]
    assert isinstance(m.PB, funcParameter)
    assert isinstance(m.PBDOT, funcParameter)
    assert "XPBDOT" not in binary._active_binary_independents()
    for j in range(6):
        name = f"FB{j}"
        assert name in binary.deriv_funcs
        if j:
            assert name in m.free_params
    toas = make_fake_toas_uniform(54500, 55500, 32, m, obs="@")
    binary.update_binary_object(toas, binary._upstream_delay(toas))
    for j in range(6):
        name = f"FB{j}"
        derivative = binary.binary_instance.d_delay_d_par(name)
        assert np.all(np.isfinite(derivative))
        assert np.any(derivative != 0)
        # FB3–FB5 delay changes at a 1e-4 relative step are ~1e-36 s: visible
        # on IEEE quad, invisible on 80-bit x87. Registration + finiteness
        # above are the portable wrap checks; kernel Dual covers the algebra.
        if j >= 3:
            continue
        parameter = getattr(m, name)
        original = np.longdouble(parameter.value)
        step = abs(original) * np.longdouble("1e-8" if j == 0 else "1e-4")
        parameter.value = original + step
        binary.update_binary_object(toas, binary._upstream_delay(toas))
        plus = binary.binary_instance.delay()
        parameter.value = original - step
        binary.update_binary_object(toas, binary._upstream_delay(toas))
        minus = binary.binary_instance.delay()
        parameter.value = original
        binary.update_binary_object(toas, binary._upstream_delay(toas))
        numeric = (plus - minus) / (2 * step)
        mask = np.abs(derivative) > np.max(np.abs(derivative)) * 1e-6
        np.testing.assert_allclose(
            derivative[mask],
            numeric[mask],
            rtol=_ddr_fd_rtol(1e-6),
            atol=_ddr_atol(1e-12, derivative[mask]),
        )


def test_fbx_discovers_fb12_numerically_not_lexically():
    m = get_model(
        StringIO(
            _fbx(
                FB10="0",
                FB12="1e-90 1",
            )
        )
    )
    binary = m.components["BinaryDDR"]
    mapping = binary._fbx_mapping()
    assert list(mapping) == list(range(13))
    lexical = sorted(mapping.values())
    numerical = [mapping[j] for j in sorted(mapping)]
    assert lexical.index("FB10") < lexical.index("FB2")
    assert numerical == [f"FB{j}" for j in range(13)]
    assert len(binary._fbx_coefficients()) == 13
    assert "FB12" in binary.deriv_funcs
    assert "FB10" in binary.deriv_funcs
    assert m.FB10.frozen
    assert not m.FB12.frozen
    toas = make_fake_toas_uniform(54500, 55500, 16, m, obs="@")
    binary.update_binary_object(toas, binary._upstream_delay(toas))
    derivative = binary.binary_instance.d_delay_d_par("FB12")
    assert np.all(np.isfinite(derivative))
    assert np.any(derivative != 0)


def test_fbx_order_is_static_after_setup():
    m = get_model(StringIO(_fbx()))
    binary = m.components["BinaryDDR"]
    binary._add_or_get_fbx(12)
    m.FB12.value = 1e-90
    with pytest.raises(TimingModelError, match="static"):
        m.setup()


def test_fbx_order_above_kmax_is_refused():
    extras = {f"FB{j}": "0" for j in range(1, DDR_FB_KMAX + 1)}
    extras[f"FB{DDR_FB_KMAX + 1}"] = "1e-90"
    with pytest.raises(TimingModelError, match="DDR_FB_KMAX"):
        get_model(StringIO(_fbx(**extras)))


def test_fbx_sparse_index_above_kmax_is_refused_before_fill():
    extras = {f"FB{j}": None for j in range(1, 6)}
    extras[f"FB{DDR_FB_KMAX + 8}"] = "1e-90"
    with pytest.raises(TimingModelError, match="DDR_FB_KMAX"):
        get_model(StringIO(_fbx(**extras)))


def test_validate_toas_refuses_fbx_slope_zero_inside_span():
    m = get_model(StringIO(_fbx(FB1="0", FB2=None, FB3=None, FB4=None, FB5=None)))
    toas = make_fake_toas_uniform(54000, 56000, 16, m, obs="@")
    m.validate_toas(toas)
    err = m.components["BinaryDDR"].inverse_timing_reference_error(toas)
    assert np.all(err >= 0)
    # FB1 = -FB0 / (400 d) drives λ̇ through zero ~400 days after TASC.
    m.FB1.quantity = (-m.FB0.quantity / (400 * u.d)).to(u.Hz / u.s)
    with pytest.raises(TimingModelError, match="slope"):
        m.validate_toas(toas)


def test_validate_toas_refuses_fbx_slope_minimum_between_grid_nodes():
    amplitude = np.longdouble("1e-4")
    m = get_model(
        StringIO(
            _fbx(
                TASC="55000",
                FB0=str(np.longdouble("0.09") * amplitude),
                FB1=str(-amplitude),
                FB2=str(2 * amplitude),
                FB3=None,
                FB4=None,
                FB5=None,
            )
        )
    )
    day = np.longdouble(86400)
    toas = SimpleNamespace(
        table={
            "tdbld": np.asarray(
                [
                    np.longdouble(55000),
                    np.longdouble(55000) + np.longdouble(64) / day,
                ],
                dtype=np.longdouble,
            )
        }
    )
    with pytest.raises(TimingModelError, match="slope"):
        m.components["BinaryDDR"]._validate_phase_slope_on_span(toas, SimpleNamespace())


def test_pb_plus_fb2_bridges_to_complete_fbx_chart():
    m = get_model(StringIO(_pheno(FB2="1e-28 1")))
    assert isinstance(m.PB, funcParameter)
    assert m.FB0.quantity is not None
    assert m.FB1.quantity is not None
    assert m.FB2.quantity is not None
    assert "FB2" in m.components["BinaryDDR"].deriv_funcs


def test_fbn_without_pb_or_fb0_is_refused():
    with pytest.raises(TimingModelError, match="FB0"):
        get_model(StringIO(_fbx(FB0=None, FB1=None, FB2="1e-28")))


def test_direct_fbx_with_pbdot_is_refused():
    with pytest.raises(TimingModelError, match="PBDOT"):
        get_model(StringIO(_fbx(PBDOT="1e-12")))


def test_pb_plus_fb2_with_nonzero_pbdot_is_refused():
    with pytest.raises(TimingModelError, match="PBDOT"):
        get_model(StringIO(_pheno(FB2="1e-28", PBDOT="1e-12")))


def test_reference_a1_must_be_nonnegative():
    with pytest.raises(TimingModelError, match="A1"):
        get_model(StringIO(_pheno(A1="-0.1", DDRPK="N", DDRPBDOT="absorb_gw")))


def test_inherited_binary_names_are_classified():
    ddr = get_model(StringIO(_pheno())).components["BinaryDDR"]
    base_names = set(PulsarBinary().params)
    active_or_mode = {
        "PB",
        "PBDOT",
        "A1",
        "A1DOT",
        "M2",
        "OMDOT",
    }
    placeholders = {"EDOT"}
    refused = {
        "FB0",
        "A1DOT2",
        "ORBWAVEC0",
        "ORBWAVES0",
        "ORBWAVE_OM",
        "ORBWAVE_EPOCH",
    }
    removed_or_display = {"ECC", "OM", "T0", "SINI"}
    assert base_names == active_or_mode | placeholders | refused | removed_or_display
    assert all(name in ddr.params or name == "T0" for name in base_names)


def test_ddrpk_n_gamma_zero_sets_ggamma():
    m = get_model(StringIO(_pheno(GAMMA="0")))
    assert m.GGAMMA.value == pytest.approx(0)


def test_ddrpk_y_refuses_explicit_gamma_zero():
    with pytest.raises(TimingModelError, match="GAMMA"):
        get_model(StringIO(example_par(GAMMA="0")))


def test_ddrpk_n_nonzero_gamma_initializes_regular_coefficient():
    model = get_model(StringIO(_pheno(GAMMA="1e-6 1", GGAMMA=None)))
    eccentricity = np.hypot(model.EPS1.value, model.EPS2.value)
    assert model.GGAMMA.value == pytest.approx(1e-6 / eccentricity)
    assert not model.GGAMMA.frozen


def test_ddrpk_n_nonzero_gamma_refuses_tiny_eccentricity():
    with pytest.raises(TimingModelError, match="eccentricity"):
        get_model(
            StringIO(
                _pheno(
                    GAMMA="1e-6",
                    GGAMMA=None,
                    EPS1="1e-7",
                    EPS2="0",
                )
            )
        )


def test_a1dot_with_geo_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_model(StringIO(example_par(A1DOT="1e-14")))
    assert any("A1DOT" in str(w.message) for w in caught)


def test_ecliptic_astrometry_loads():
    m = get_model(StringIO(ecliptic_example_par()))
    assert "AstrometryEcliptic" in m.components
    assert m.ELONG.value is not None
    assert isinstance(m.PBDOT, funcParameter)


def test_validate_toas_accepts_example():
    m = get_model(StringIO(example_par()))
    toas = make_fake_toas_uniform(54990, 55020, 8, m, obs="gbt")
    m.validate_toas(toas)


def test_validate_toas_domain_error():
    m = get_model(StringIO(example_par()))
    toas = make_fake_toas_uniform(54990, 55020, 8, m, obs="gbt")
    m.COSI.value = 1.0
    with pytest.raises(TimingModelError):
        m.validate_toas(toas)


@pytest.mark.parametrize("pk", ["Y", "N"])
@pytest.mark.parametrize("pbdot", ["kinematic", "absorb_gw"])
@pytest.mark.parametrize("geo", ["Y", "N"])
@pytest.mark.parametrize("kine", ["Y", "N"])
def test_valid_mode_combinations_load(pk, pbdot, geo, kine):
    if geo == "Y" and kine == "N":
        pytest.skip("DDRGEO Y requires DDRKINE Y")
    overrides = {
        "DDRPK": pk,
        "DDRPBDOT": pbdot,
        "DDRGEO": geo,
        "DDRKINE": kine,
    }
    if pk == "N":
        overrides["GGAMMA"] = "0"
        overrides["OMDOT"] = "0"
    if geo == "N":
        overrides["KOM"] = None
    if geo == "N" and kine == "N":
        overrides["PX"] = None
        overrides["PMRA"] = None
        overrides["PMDEC"] = None
    if pbdot == "absorb_gw":
        overrides["PBDOT"] = "0"
    m = get_model(StringIO(example_par(**overrides)))
    assert m.BINARY.value == "DDR"
    assert bool(m.DDRPK.value) == (pk == "Y")
    assert m.DDRPBDOT.value == pbdot
    if pk == "Y":
        assert isinstance(m.OMDOT, funcParameter)
        assert isinstance(m.GAMMA, funcParameter)
    else:
        assert not isinstance(m.OMDOT, funcParameter)
        assert m.GGAMMA.value == pytest.approx(0)
    if pbdot == "kinematic":
        assert isinstance(m.PBDOT, funcParameter)
    else:
        assert not isinstance(m.PBDOT, funcParameter)


def test_display_names_and_mp():
    m = get_model(StringIO(example_par()))
    assert m.SINI.value == pytest.approx((1 - 0.25) ** 0.5)
    assert m.KIN.value > 0
    assert m.H3.value > 0
    assert m.STIGMA.value > 0
    mp = m.components["BinaryDDR"].mp_from_mass_function()
    assert 1.0 < mp.value < 2.0
    t0 = m.components["BinaryDDR"].t0_from_tasc()
    assert t0 is not None


def test_change_binary_epoch_rejects_gr():
    m = get_model(StringIO(example_par()))
    with pytest.raises(TimingModelError, match="DDRPK"):
        m.components["BinaryDDR"].change_binary_epoch(55100)


def test_unfrozen_independents_registered_in_deriv_funcs():
    m = get_model(StringIO(example_par()))
    ddr = m.components["BinaryDDR"]
    for name in ("PB", "A1", "TASC", "EPS1", "EPS2", "M2", "COSI", "KOM", "PX"):
        assert name in ddr.deriv_funcs, name
    assert "DM" in ddr.deriv_funcs
    assert ddr.d_ddr_time_argument_correction in ddr.deriv_funcs["DM"]


# --- full-model composition ---


@pytest.fixture(scope="module")
def ddr_model_toas():
    model = get_model(StringIO(example_par()))
    toas = make_fake_toas_uniform(54990, 55020, 16, model, obs="gbt")
    return model, toas


def test_delay_finite(ddr_model_toas):
    model, toas = ddr_model_toas
    d = model.delay(toas)
    assert np.all(np.isfinite(d.to_value(u.s)))
    assert np.max(np.abs(d.to_value(u.s))) > 0.1


def _assert_phase_column(model, toas, param, step):
    analytic = model.d_phase_d_param(toas, delay=None, param=param)
    numeric = model.d_phase_d_param_num(toas, param, step=step)
    a = np.asarray(analytic.to_value(analytic.unit), dtype=np.longdouble)
    n = np.asarray(numeric.to_value(analytic.unit), dtype=np.longdouble)
    scale = np.max(np.abs(a)) + np.max(np.abs(n))
    np.testing.assert_allclose(
        a, n, rtol=5e-3, atol=1e-5 * scale + 1e-12, err_msg=param
    )
    return a, n


def _as_unit(q, unit):
    return np.asarray(
        q.to(unit, equivalencies=u.dimensionless_angles()).value, dtype=np.longdouble
    )


def test_upstream_only_dm_isolates_time_argument_correction(ddr_model_toas):
    model, toas = ddr_model_toas
    ddr = model.components["BinaryDDR"]
    A = ddr._upstream_d_delay_d_param(toas, "DM")
    corr = ddr.d_ddr_time_argument_correction(toas, "DM")
    full = model.d_delay_d_param(toas, "DM")
    unit = A.unit
    a = _as_unit(A, unit)
    c = _as_unit(corr, unit)
    leftover = _as_unit(full, unit) - a
    assert np.max(np.abs(c)) > 0
    assert np.max(np.abs(leftover)) / np.max(np.abs(a)) > 1e-6
    np.testing.assert_allclose(c, leftover, rtol=1e-6, atol=1e-18 * np.max(np.abs(a)))


def test_upstream_only_dm_phase_column(ddr_model_toas):
    model, toas = ddr_model_toas
    a, _n = _assert_phase_column(model, toas, "DM", 1e-4)
    assert np.max(np.abs(a)) > 0


def test_astrometric_columns_compose_B_and_time_argument(ddr_model_toas):
    model, toas = ddr_model_toas
    ddr = model.components["BinaryDDR"]
    for param in ("PX", "RAJ", "PMRA"):
        A = ddr._upstream_d_delay_d_param(toas, param)
        B = ddr.d_binary_delay_d_xxxx(toas, param)
        corr = ddr.d_ddr_time_argument_correction(toas, param)
        full = model.d_delay_d_param(toas, param)
        unit = A.unit
        combo = _as_unit(A, unit) + _as_unit(B, unit) + _as_unit(corr, unit)
        f = _as_unit(full, unit)
        np.testing.assert_allclose(
            f, combo, rtol=1e-5, atol=1e-16 * (np.max(np.abs(f)) + 1), err_msg=param
        )
        assert np.max(np.abs(_as_unit(corr, unit))) > 0, param


def test_raj_px_phase_columns(ddr_model_toas):
    model, toas = ddr_model_toas
    for param, step in (("PX", 1e-4), ("RAJ", 1e-8), ("PMRA", 1e-3)):
        _assert_phase_column(model, toas, param, step)


@pytest.mark.parametrize(
    ("par_text", "params"),
    [
        (
            example_par(
                RAJ="04:37:15",
                DECJ="+16:00:00",
                POSEPOCH="54000",
                TGEO="55000",
                PMRA="20",
                PMDEC="-10",
            ),
            ("RAJ", "DECJ", "PMRA", "PMDEC"),
        ),
        (
            ecliptic_example_par(
                ELONG="72.3",
                ELAT="-31.2",
                POSEPOCH="54000",
                TGEO="55000",
                PMELONG="20",
                PMELAT="-10",
            ),
            ("ELONG", "ELAT", "PMELONG", "PMELAT"),
        ),
    ],
)
def test_standalone_astrometric_columns_are_analytic(par_text, params):
    model = get_model(StringIO(par_text))
    toas = make_fake_toas_uniform(54900, 55265, 24, model, obs="gbt")
    binary = model.components["BinaryDDR"]
    upstream = binary._upstream_delay(toas)
    binary.update_binary_object(toas, upstream)
    obs_au = binary._obs_pos_au(toas)
    _state, derivatives = binary._analytic_astrometry(obs_au)
    # 1e-8 deg is below 80-bit ulp of O(1) AU primitives; 1e-6 keeps FD
    # above that noise floor while the space-motion map stays linear.
    steps = {"RAJ": 1e-6, "DECJ": 1e-6, "ELONG": 1e-6, "ELAT": 1e-6}
    primitive_keys = ("mu_I", "mu_J", "d_I_au", "d_J_au")
    for name in params:
        par = getattr(model, name)
        step = np.longdouble(steps.get(name, 1e-3))
        analytic = binary.d_binary_delay_d_xxxx(toas, name).to_value(u.s / par.units)
        kernel = np.asarray(
            binary.binary_instance.d_delay_d_par(name), dtype=np.longdouble
        )
        np.testing.assert_allclose(
            analytic,
            kernel,
            rtol=_ddr_rtol(1e-8),
            atol=_ddr_atol(1e-14, analytic),
            err_msg=name,
        )
        assert np.all(np.isfinite(analytic))
        original = np.longdouble(par.value)
        try:
            par.value = original + step
            plus_state, _ = binary._analytic_astrometry(obs_au)
            par.value = original - step
            minus_state, _ = binary._analytic_astrometry(obs_au)
        finally:
            par.value = original
        for key in primitive_keys:
            numeric = (plus_state[key] - minus_state[key]) / (2 * step)
            dual = derivatives[name][key.lower()]
            scale = max(np.max(np.abs(dual)), np.max(np.abs(numeric)), _LD("1e-30"))
            np.testing.assert_allclose(
                dual,
                numeric,
                rtol=_ddr_fd_rtol(2e-5),
                atol=_ddr_peak_atol(scale),
                err_msg=f"{name}:{key}",
            )


def test_zero_proper_motion_astrometric_columns_are_finite():
    model = get_model(StringIO(example_par(PMRA="0", PMDEC="0")))
    toas = make_fake_toas_uniform(54990, 55020, 16, model, obs="gbt")
    binary = model.components["BinaryDDR"]
    for name in ("PMRA", "PMDEC"):
        column = binary.d_binary_delay_d_xxxx(toas, name)
        assert np.all(np.isfinite(column.value))


def test_observer_projection_is_per_toa():
    model = get_model(StringIO(example_par(PMRA="0", PMDEC="0")))
    toas = make_fake_toas_uniform(54990, 55355, 24, model, obs="gbt")
    binary = model.components["BinaryDDR"]
    binary.binarymodel_delay(toas)
    obs_au = binary._obs_pos_au(toas)
    triad = binary._tgeo_triad()
    expected = obs_au @ np.asarray(triad["I0"], dtype=np.longdouble)
    got = np.asarray(binary.binary_instance.d_I_au, dtype=np.longdouble)
    assert got.shape == (toas.ntoas,)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert np.ptp(got) > 0.5


def test_zero_pm_geometry_delay_varies_annually():
    model_y = get_model(StringIO(example_par(PMRA="0", PMDEC="0")))
    model_n = get_model(
        StringIO(example_par(PMRA="0", PMDEC="0", DDRGEO="N", KOM=None))
    )
    toas = make_fake_toas_uniform(54990, 55355, 48, model_y, obs="gbt")
    difference = model_y.delay(toas).to_value(u.s) - model_n.delay(toas).to_value(u.s)
    assert np.max(np.abs(difference)) > 1e-8
    assert np.ptp(difference) > 0.5 * np.max(np.abs(difference))


def test_ddrgeo_changes_delay_at_ns_scale():
    model_y = get_model(StringIO(example_par()))
    model_n = get_model(StringIO(example_par(DDRGEO="N", KOM=None)))
    toas = make_fake_toas_uniform(54990, 55020, 16, model_y, obs="gbt")
    dy = model_y.delay(toas).to_value(u.s)
    dn = model_n.delay(toas).to_value(u.s)
    delta = np.max(np.abs(dy - dn))
    assert 1e-9 < delta < 1e-6


def test_posepoch_tgeo_tasc_triad_uses_tgeo_sky():
    """``n0`` is the TGEO direction, not POSEPOCH RA/DEC (analytics §5.1)."""
    m = get_model(
        StringIO(
            example_par(
                RAJ="04:37:15.0",
                DECJ="+16:00:00.0",
                POSEPOCH="54000",
                TGEO="55000",
                TASC="56000",
                PMRA="20.0",
                PMDEC="-10.0",
            )
        )
    )
    assert m.POSEPOCH.value != pytest.approx(m.TGEO.value)
    assert m.TGEO.value != pytest.approx(m.TASC.value)
    astrom = m.components["AstrometryEquatorial"]
    triad = m.components["BinaryDDR"]._tgeo_triad()

    def _n_and_ndot(coord):
        augmented = add_dummy_distance(coord)
        xyz = np.array(augmented.cartesian.xyz.to(u.au).value, dtype=np.longdouble)
        velocity = np.array(
            augmented.velocity.d_xyz.to(u.au / u.s).value, dtype=np.longdouble
        )
        norm = np.linalg.norm(xyz)
        n = xyz / norm
        ndot = (velocity - np.dot(velocity, n) * n) / norm
        return n, ndot

    n_tgeo, ndot_tgeo = _n_and_ndot(astrom.get_psr_coords(epoch=m.TGEO.quantity))
    n_pose, _ = _n_and_ndot(astrom.get_psr_coords(epoch=m.POSEPOCH.quantity))
    np.testing.assert_allclose(triad["n0"], n_tgeo, rtol=0, atol=1e-14)
    reconstructed_ndot = triad["mu_I"] * triad["I0"] + triad["mu_J"] * triad["J0"]
    np.testing.assert_allclose(reconstructed_ndot, ndot_tgeo, rtol=1e-14, atol=1e-30)
    assert np.max(np.abs(triad["n0"] - n_pose)) > 1e-8


def test_ecliptic_tgeo_triad_matches_space_motion():
    m = get_model(
        StringIO(
            ecliptic_example_par(
                ELONG="72.3",
                ELAT="-31.2",
                POSEPOCH="54000",
                TGEO="55000",
                TASC="56000",
                PMELONG="20.0",
                PMELAT="-10.0",
            )
        )
    )
    astrom = m.components["AstrometryEcliptic"]
    coord = add_dummy_distance(astrom.get_psr_coords(epoch=m.TGEO.quantity))
    xyz = np.asarray(coord.cartesian.xyz.to_value(u.au), dtype=np.longdouble)
    expected = xyz / np.linalg.norm(xyz)
    triad = m.components["BinaryDDR"]._tgeo_triad()
    np.testing.assert_allclose(triad["n0"], expected, rtol=0, atol=1e-14)


def test_binary_pb_phase_column(ddr_model_toas):
    model, toas = ddr_model_toas
    _assert_phase_column(model, toas, "PB", 1e-8)


def test_wrapper_reuses_one_batched_tangent_evaluation(ddr_model_toas, monkeypatch):
    model, toas = ddr_model_toas
    binary = model.components["BinaryDDR"]
    calls = 0
    original = binary.binary_instance.d_delay_d_pars

    def counted(names, t=None):
        nonlocal calls
        calls += 1
        return original(names, t)

    monkeypatch.setattr(binary.binary_instance, "d_delay_d_pars", counted)
    binary.d_binary_delay_d_xxxx(toas, "PB")
    binary.d_binary_delay_d_xxxx(toas, "A1")
    binary.d_binary_delay_d_xxxx(toas, "EPS1")
    assert calls == 1


def test_derivative_cache_key_ignores_longdouble_ulp(ddr_model_toas):
    model, toas = ddr_model_toas
    binary = model.components["BinaryDDR"]
    names = list(
        dict.fromkeys(binary._active_binary_independents() + binary._astrometry_in_B())
    )
    upstream = binary._upstream_delay(toas)
    key = binary._derivative_cache_key(toas, upstream, names)
    jitter = (
        np.finfo(np.longdouble).eps
        * np.maximum(np.abs(upstream.to_value(u.s)), 1.0)
        * u.s
    )
    assert key == binary._derivative_cache_key(toas, upstream + jitter, names)


@pytest.mark.parametrize("omdot", ["0", "0.01"])
def test_change_binary_epoch_p0_delay_invariance(omdot):
    model = get_model(StringIO(_pheno(OMDOT=omdot)))
    toas = make_fake_toas_uniform(54990, 55020, 12, model, obs="@")
    d0 = model.delay(toas).to_value(u.s)
    tgeo = model.TGEO.value
    model.components["BinaryDDR"].change_binary_epoch(55002)
    assert model.TGEO.value == tgeo
    d1 = model.delay(toas).to_value(u.s)
    np.testing.assert_allclose(d0, d1, rtol=0, atol=1e-14)


def test_change_binary_epoch_refuses_nonzero_rate():
    model = get_model(StringIO(_pheno(PBDOT="1")))
    with pytest.raises(TimingModelError, match="nonzero PBDOT"):
        model.components["BinaryDDR"].change_binary_epoch(55002)


def test_change_binary_epoch_fb0_only_uses_instantaneous_period():
    model = get_model(StringIO(_fbx(FB1=None, FB2=None, FB3=None, FB4=None, FB5=None)))
    toas = make_fake_toas_uniform(54990, 55020, 12, model, obs="@")
    d0 = model.delay(toas).to_value(u.s)
    model.components["BinaryDDR"].change_binary_epoch(55002)
    d1 = model.delay(toas).to_value(u.s)
    np.testing.assert_allclose(d0, d1, rtol=0, atol=1e-14)


def test_t0_from_tasc_matches_pb_chart_quadratic():
    model = get_model(StringIO(_pheno(PBDOT="0")))
    binary = model.components["BinaryDDR"]
    omega = np.mod(
        np.arctan2(np.longdouble(model.EPS1.value), np.longdouble(model.EPS2.value)),
        2 * np.longdouble(np.pi),
    )
    expected = np.longdouble(model.TASC.value) + np.longdouble(model.PB.value) * (
        omega / (2 * np.longdouble(np.pi))
    )
    np.testing.assert_allclose(
        binary.t0_from_tasc().to_value(u.d), expected, rtol=0, atol=2e-15
    )


def test_t0_from_tasc_expands_bracket_and_om_dd_is_reduced():
    angle = np.deg2rad(np.longdouble("359.9"))
    eccentricity = np.longdouble("0.001")
    model = get_model(
        StringIO(
            _pheno(
                PBDOT="1e9",
                EPS1=str(eccentricity * np.sin(angle)),
                EPS2=str(eccentricity * np.cos(angle)),
                OMDOT="0.01",
            )
        )
    )
    binary = model.components["BinaryDDR"]
    offset = np.longdouble(binary.t0_from_tasc().to_value(u.d)) - np.longdouble(
        model.TASC.value
    )
    assert offset > np.longdouble(model.PB.value)
    assert offset < np.longdouble("1.01") * np.longdouble(model.PB.value)
    assert 0 <= binary.om_dd().to_value(u.deg) < 360


def test_ecliptic_geometry_delay_finite():
    model = get_model(StringIO(ecliptic_example_par()))
    toas = make_fake_toas_uniform(54990, 55020, 12, model, obs="gbt")
    d = model.delay(toas).to_value(u.s)
    assert np.all(np.isfinite(d))
    assert np.max(np.abs(d)) > 0.1


def _kom_frame_fixture():
    return get_model(
        StringIO(
            example_par(
                RAJ="04:37:15.0",
                DECJ="+16:00:00.0",
                POSEPOCH="54000",
                TGEO="55000",
                TASC="56000",
                PMRA="20.0",
                PMDEC="-10.0",
            )
        )
    )


def _wrapped_deg(angle):
    return float(angle.to(u.deg).value) % 360.0


def _angle_delta_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _expected_ddr_kom_ecliptic(model, epoch, ecl="IERS2010"):
    source = model.components["AstrometryEquatorial"].get_psr_coords(epoch=epoch)
    node = coords.SkyCoord(
        ra=source.ra,
        dec=source.dec,
        obstime=epoch,
        pm_ra_cosdec=np.cos(model.KOM.quantity) * u.mas / u.yr,
        pm_dec=np.sin(model.KOM.quantity) * u.mas / u.yr,
        frame=coords.ICRS,
    ).transform_to(PulsarEcliptic(ecl=ecl))
    return _wrapped_deg(
        (np.arctan2(node.pm_lat.value, node.pm_lon_coslat.value) * u.rad)
    )


def test_ddr_geometry_is_frame_invariant_with_kom_rotation():
    model = _kom_frame_fixture()
    toas = make_fake_toas_uniform(54900, 55265, 32, model, obs="gbt")
    binary = model.components["BinaryDDR"]
    original = binary.binarymodel_delay(toas).to_value(u.s)
    ecliptic = model.as_ECL()
    transformed = ecliptic.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    np.testing.assert_allclose(transformed, original, rtol=0, atol=1e-12)
    roundtrip = ecliptic.as_ICRS()
    restored = roundtrip.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    np.testing.assert_allclose(restored, original, rtol=0, atol=1e-12)
    assert (
        _angle_delta_deg(
            _wrapped_deg(roundtrip.KOM.quantity), _wrapped_deg(model.KOM.quantity)
        )
        < 1e-14
    )


def test_as_ecl_kom_matches_tgeo_position_rotation():
    model = _kom_frame_fixture()
    ecliptic = model.as_ECL()
    tgeo_kom = _expected_ddr_kom_ecliptic(model, model.TGEO.quantity)
    posepoch_kom = _expected_ddr_kom_ecliptic(model, model.POSEPOCH.quantity)
    assert _angle_delta_deg(_wrapped_deg(ecliptic.KOM.quantity), tgeo_kom) < 1e-14
    assert _angle_delta_deg(tgeo_kom, posepoch_kom) > 1e-8


def test_as_ecl_leaves_unset_kom_unset():
    model = get_model(StringIO(_pheno()))
    assert model.KOM.quantity is None
    ecliptic = model.as_ECL()
    assert ecliptic.KOM.quantity is None
    assert ecliptic.as_ICRS().KOM.quantity is None


def test_as_icrs_leaves_unset_kom_unset_on_ecliptic_model():
    model = get_model(
        StringIO(
            ecliptic_example_par(
                DDRPK="N",
                DDRPBDOT="absorb_gw",
                DDRGEO="N",
                DDRKINE="N",
                PBDOT="0",
                GGAMMA="0",
                OMDOT="0",
                KOM=None,
            )
        )
    )
    assert model.KOM.quantity is None
    equatorial = model.as_ICRS()
    assert equatorial.KOM.quantity is None


# --- fit ---


def _truth_and_toas(**overrides):
    model = get_model(StringIO(example_par(F1="0", **overrides)))
    toas = make_fake_toas_uniform(54900, 55100, 200, model, obs="gbt", error=0.1 * u.us)
    return model, toas


def _perturb_and_fit(truth, toas, *, finite_e):
    fit = get_model(
        StringIO(
            example_par(
                F1="0",
                EPS1="1.0e-6" if finite_e else "0",
                EPS2="-2.0e-6" if finite_e else "0",
            )
        )
    )
    for name in ["PB", "A1", "TASC", "EPS1", "EPS2", "M2", "COSI", "F0", "F1"]:
        getattr(fit, name).frozen = False
    fit.PB.value = truth.PB.value * (1 + 1e-8)
    fit.A1.value = truth.A1.value * (1 + 1e-6)
    fit.TASC.value = truth.TASC.value + 1e-6
    fit.M2.value = truth.M2.value * (1 + 1e-4)
    fit.COSI.value = truth.COSI.value * (1 - 1e-4)
    fit.F0.value = truth.F0.value * (1 + 1e-12)
    fit.F1.value = 1e-16
    fit.EPS1.value = 1e-5 if finite_e else 1e-6
    fit.EPS2.value = -1e-5 if finite_e else -1e-6
    fitter = WLSFitter(toas, fit)
    fitter.fit_toas()
    return fitter, truth


def test_wls_recovers_circular_ddr():
    truth, toas = _truth_and_toas(EPS1="0", EPS2="0")
    fitter, truth = _perturb_and_fit(truth, toas, finite_e=False)
    assert fitter.resids.chi2 / fitter.resids.dof < 2
    for name in ("PB", "A1", "TASC", "M2", "COSI", "F0"):
        recovered = getattr(fitter.model, name)
        true = getattr(truth, name)
        pull = abs(recovered.value - true.value) / recovered.uncertainty.value
        assert pull < 1, f"{name} pull {pull}"


def test_wls_recovers_finite_e_ddr():
    truth, toas = _truth_and_toas()
    fitter, truth = _perturb_and_fit(truth, toas, finite_e=True)
    assert fitter.resids.chi2 / fitter.resids.dof < 2
    for name in ("PB", "A1", "EPS1", "EPS2", "F0"):
        recovered = getattr(fitter.model, name)
        true = getattr(truth, name)
        pull = abs(recovered.value - true.value) / recovered.uncertainty.value
        assert pull < 1, f"{name} pull {pull}"


def test_wls_recovers_fbx_coefficient():
    truth = get_model(StringIO(_fbx(F1="0", FB2="1e-28")))
    toas = make_fake_toas_uniform(54500, 55500, 120, truth, obs="@", error=0.1 * u.us)
    fit = get_model(StringIO(_fbx(F1="0", FB2="1e-28 1")))
    fit.FB2.value *= 1.01
    fitter = WLSFitter(toas, fit)
    fitter.fit_toas(maxiter=5)
    assert fitter.model.FB2.uncertainty is not None
    assert fitter.model.FB2.value == pytest.approx(
        truth.FB2.value, abs=5 * fitter.model.FB2.uncertainty_value
    )


def test_invalid_cosi_trial_raises_invalid_model_parameters():
    model = get_model(StringIO(example_par(F1="0")))
    toas = make_fake_toas_uniform(54990, 55020, 40, model, obs="gbt", error=0.1 * u.us)
    model.delay(toas)
    model.COSI.value = 1.0
    with pytest.raises(InvalidModelParameters):
        model.delay(toas)


def test_invalid_eccentricity_trial_raises_invalid_model_parameters():
    model = get_model(StringIO(example_par(F1="0")))
    toas = make_fake_toas_uniform(54990, 55020, 40, model, obs="gbt", error=0.1 * u.us)
    model.delay(toas)
    model.EPS1.value = 0.8
    model.EPS2.value = 0.8
    with pytest.raises(InvalidModelParameters):
        model.delay(toas)


# --- FW10 ---


def test_fw10_roundtrip_positive_cosi():
    m2 = 0.2 * u.Msun
    cosi = 0.5 * u.dimensionless_unscaled
    h3, stigma, h4 = fw10_encode(m2, cosi)
    m2_b, abs_c = fw10_decode(h3, stigma)
    np.testing.assert_allclose(m2_b.to_value(u.Msun), 0.2, rtol=1e-12)
    np.testing.assert_allclose(
        abs_c.to_value(u.dimensionless_unscaled), 0.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        h4.to_value(u.s), (h3 * stigma).to_value(u.s), rtol=1e-12
    )
    np.testing.assert_allclose(fw10_h3(m2, cosi).to_value(u.s), h3.to_value(u.s))
    np.testing.assert_allclose(fw10_h4(m2, cosi).to_value(u.s), h4.to_value(u.s))


def test_fw10_decode_drops_cosi_sign():
    m2 = 0.15 * u.Msun
    cosi = -0.3 * u.dimensionless_unscaled
    h3, stigma, _h4 = fw10_encode(m2, cosi)
    _m2_b, abs_c = fw10_decode(h3, stigma)
    np.testing.assert_allclose(
        abs_c.to_value(u.dimensionless_unscaled), 0.3, rtol=1e-12
    )
    s = fw10_stigma(cosi).to_value(u.dimensionless_unscaled)
    s_pos = fw10_stigma(0.3 * u.dimensionless_unscaled).to_value(
        u.dimensionless_unscaled
    )
    np.testing.assert_allclose(s, s_pos, rtol=1e-12)


@pytest.mark.parametrize(
    ("h3", "stigma"),
    [
        (1e-6 * u.s, 0 * u.dimensionless_unscaled),
        (1e-6 * u.s, 1.01 * u.dimensionless_unscaled),
        (0 * u.s, 0.5 * u.dimensionless_unscaled),
        (-1e-6 * u.s, 0.5 * u.dimensionless_unscaled),
    ],
)
def test_fw10_decode_enforces_domain(h3, stigma):
    with pytest.raises(InvalidModelParameters):
        fw10_decode(h3, stigma)


@pytest.mark.parametrize(
    "args",
    [
        (np.nan, 0.1, 0.1, 55000, 86400, 1e-6, 0.5),
        (10, np.nan, 0.1, 55000, 86400, 1e-6, 0.5),
        (10, 0.1, 0.1, 55000, np.nan, 1e-6, 0.5),
        (10, 0.1, 0.1, 55000, 86400, 1e-6, np.nan),
    ],
)
def test_fw10_orbit_decode_refuses_nonfinite_inputs(args):
    with pytest.raises(InvalidModelParameters, match="finite"):
        fw10_orbit_decode(*args)


def test_fw10_orbit_encode_decode_roundtrip():
    pb_s = np.longdouble(86400)
    r_s = np.longdouble("1e-6")
    stigma = np.longdouble("0.5")
    x, h, k, tasc = (
        np.longdouble(10),
        np.longdouble("0.01"),
        np.longdouble("-0.004"),
        np.longdouble(55000),
    )
    xa, ha, ka, tasc_a = fw10_orbit_encode(x, h, k, tasc, pb_s, r_s, stigma)
    x2, h2, k2, tasc2 = fw10_orbit_decode(xa, ha, ka, tasc_a, pb_s, r_s, stigma)
    np.testing.assert_allclose(x2, x, rtol=0, atol=1e-18)
    np.testing.assert_allclose(h2, h, rtol=0, atol=1e-18)
    np.testing.assert_allclose(k2, k, rtol=0, atol=1e-18)
    np.testing.assert_allclose(tasc2, tasc, rtol=0, atol=1e-18)
    assert xa > x


def test_ddr_reference_fixture_pieces():
    """Shared oracle (proposal §10.9). Inverse-timing is recorded, not independently derived here."""
    import json
    from pathlib import Path

    payload = json.loads(
        (
            Path(__file__).resolve().parent / "data" / "ddr_reference_fixtures.json"
        ).read_text()
    )
    p = payload["params"]
    m = DDRmodel(
        PB=_LD(p["PB"]),
        A1=_LD(p["A1"]),
        TASC=_LD(p["TASC"]),
        EPS1=_LD(p["EPS1"]),
        EPS2=_LD(p["EPS2"]),
        M2=_LD(p["M2"]),
        COSI=_LD(p["COSI"]),
        DDRPK=p["DDRPK"],
        DDRPBDOT=p["DDRPBDOT"],
        DDRGEO=p["DDRGEO"],
        DDRKINE=p["DDRKINE"],
        XPBDOT=_LD(p["XPBDOT"]),
    )
    t = np.array([_LD(x) for x in payload["t_mjd"]], dtype=np.longdouble)
    st = m.evaluate(t)
    atol = _LD(payload["conventions"]["longdouble_atol_s"])
    for key, got in (
        ("delay_s", st.delay),
        ("Delta_rom_s", st.Delta_rom),
        ("Delta_E_s", st.Delta_E),
        ("Delta_S_s", st.Delta_S),
        ("d_inv_s", st.d_inv),
    ):
        np.testing.assert_allclose(
            got,
            np.array([_LD(x) for x in payload[key]]),
            rtol=0,
            atol=atol,
            err_msg=key,
        )
    for par, key in (
        ("A1", "d_delay_d_A1"),
        ("EPS1", "d_delay_d_EPS1"),
        ("COSI", "d_delay_d_COSI"),
    ):
        d = m.d_delay_d_par(par, t)
        np.testing.assert_allclose(
            d,
            np.array([_LD(x) for x in payload[key]]),
            rtol=0,
            atol=_LD("1e-14"),
            err_msg=key,
        )


# --- cross-family delay *differences* (not identities except circular ELL1 / DD) ---


def test_circular_ell1_matches_ddr_inverse_timing():
    """e=0, no Shapiro: binary delays, not raw Roemer (analytics / proposal §10.3)."""
    par = """\
PSRJ J0000+0000
RAJ 00:00:00.0
DECJ +00:00:00.0
F0 200
PEPOCH 55000
DM 0
BINARY ELL1
PB 2.0
A1 1.963
TASC 55000
EPS1 0
EPS2 0
M2 0
SINI 0.5
PBDOT 0
UNITS TDB
"""
    ell1 = get_model(StringIO(par))
    ddr = get_model(StringIO(_pheno(EPS1="0", EPS2="0", M2="0", COSI="0.5", DM="0")))
    toas = make_fake_toas_uniform(54990, 55020, 24, ddr, obs="@")
    d_ell1 = ell1.components["BinaryELL1"].binarymodel_delay(toas).to_value(u.s)
    d_ddr = ddr.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    np.testing.assert_allclose(d_ddr, d_ell1, rtol=0, atol=5e-12)
    raw = 1.963 * np.sin(2 * np.pi * (toas.get_mjds().to_value(u.d) - 55000.0) / 2.0)
    assert np.max(np.abs(d_ddr - raw)) > 1e-6


def test_ell1h_circular_shapiro_constant_vs_physical():
    """``Δ_S,physical − Δ_29 = 2 r log(1+ς²)`` (analytics §10.2)."""
    stigma = _LD("0.5")
    c = (1 - stigma * stigma) / (1 + stigma * stigma)
    s = np.sqrt((1 - c) * (1 + c))
    r = _LD("1e-6")
    m2 = r / tsun_s()
    a1 = _LD(10)
    pb = _LD(1)
    tasc = _LD(55000)
    m = DDRmodel(
        PB=pb,
        A1=a1,
        TASC=tasc,
        EPS1=0,
        EPS2=0,
        M2=m2,
        COSI=c,
        DDRPK=False,
        DDRPBDOT="absorb_gw",
        PBDOT=0,
        GGAMMA=0,
        OMDOT=0,
        DDRGEO=False,
        DDRKINE=False,
    )
    t = tasc + as_ld(np.linspace(0, 1, 64)) * pb
    st = m.evaluate(t)
    phi = st.lam
    n_fw = 1 + stigma * stigma - 2 * stigma * np.sin(phi)
    d29 = -2 * r * np.log(n_fw)
    dphys = st.Delta_S
    const = 2 * r * np.log(1 + stigma * stigma)
    np.testing.assert_allclose(dphys - d29, const, rtol=0, atol=_LD("1e-18"))
    # Face-on boxed s check.
    np.testing.assert_allclose(
        s, 2 * stigma / (1 + stigma * stigma), rtol=0, atol=1e-15
    )


def test_ell1h_third_harmonic_amplitude_circular():
    """Absorbed first-order ELL1 vs circular DDR: third-harmonic ``3 n x r ς²``."""
    stigma = _LD("0.5")
    r = _LD("1e-6")
    x = _LD(10)
    pb = _LD(1)
    tasc = _LD(55000)
    n = _LD(2) * np.pi / (pb * _LD(86400))
    expected = 3 * n * x * r * stigma * stigma
    assert float(expected) == pytest.approx(5.454e-10, rel=1e-3)
    c = (1 - stigma * stigma) / (1 + stigma * stigma)
    m2 = r / tsun_s()
    m = DDRmodel(
        PB=pb,
        A1=x,
        TASC=tasc,
        EPS1=0,
        EPS2=0,
        M2=m2,
        COSI=c,
        DDRPK=False,
        DDRPBDOT="absorb_gw",
        PBDOT=0,
        GGAMMA=0,
        OMDOT=0,
        DDRGEO=False,
        DDRKINE=False,
    )
    t = tasc + as_ld(np.linspace(0, 1, 256, endpoint=False)) * pb
    st = m.evaluate(t)
    phi = st.lam
    n_fw = 1 + stigma * stigma - 2 * stigma * np.sin(phi)
    d28 = (
        -2
        * r
        * (np.log(n_fw) + 2 * stigma * np.sin(phi) - stigma * stigma * np.cos(2 * phi))
    )
    xa = x + 4 * r * stigma
    ha = 4 * r * stigma * stigma / xa
    ka = 8 * n * r * stigma * x / xa
    sphi = np.sin(phi)
    s2 = np.sin(2 * phi)
    cphi = np.cos(phi)
    c2 = np.cos(2 * phi)
    dre = xa * (sphi + _LD("0.5") * (ka * s2 - ha * c2))
    drep = xa * (cphi + ka * c2 + ha * s2)
    drepp = xa * (-sphi - 2 * ka * s2 + 2 * ha * c2)
    absorbed = value(inverse_timing(dre, drep, drepp, n, _LD(0), _LD(0))) + d28
    diff = np.asarray(st.delay - absorbed, dtype=np.longdouble)
    diff = diff - np.mean(diff)
    amp = 2 * np.abs(np.mean(diff * np.exp(-3j * np.asarray(phi, dtype=float))))
    np.testing.assert_allclose(amp, float(expected), rtol=0.15)
    assert amp > 0.4e-9


def test_ddh_via_fw_map_not_ell1h_identity():
    """DDH is orthometric-on-DD (full Shapiro). Compare DDR through the FW map.

    Residual identity with DDH is not the definition of correctness; ELL1H
    absorbed / truncated harmonics is the delay-*difference* test.
    """
    from pint.binaryconvert import convert_binary

    ddr = get_model(StringIO(_pheno(EPS1="0.01", EPS2="-0.004", DM="0")))
    ddh = convert_binary(ddr, "DDH")
    h3, stigma, _h4 = fw10_encode(ddr.M2.quantity, ddr.COSI.quantity)
    np.testing.assert_allclose(
        ddh.H3.quantity.to_value(u.s), h3.to_value(u.s), rtol=1e-8
    )
    np.testing.assert_allclose(
        ddh.STIGMA.quantity.to_value(u.dimensionless_unscaled),
        stigma.to_value(u.dimensionless_unscaled),
        rtol=1e-8,
    )
    ell1h = convert_binary(ddr, "ELL1H", useSTIGMA=True, NHARMS=3)
    toas = make_fake_toas_uniform(54990, 55020, 24, ddr, obs="@")
    dl = ddr.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    dh = ell1h.components["BinaryELL1H"].binarymodel_delay(toas).to_value(u.s)
    assert np.all(np.isfinite(dl)) and np.all(np.isfinite(dh))
    assert np.max(np.abs(dl - dh)) > 1e-10


def test_ddk_is_not_a_residual_identity():
    """First-order interior agreement only; annual geometry difference is nonzero."""
    from pint.binaryconvert import convert_binary

    ddr = get_model(
        StringIO(
            example_par(
                RAJ="04:37:15.0",
                DECJ="+16:00:00.0",
                DDRPK="N",
                DDRPBDOT="absorb_gw",
                PBDOT="0",
                GGAMMA="0",
                OMDOT="0",
            )
        )
    )
    ddk = convert_binary(ddr, "DDK")
    toas_ssb = make_fake_toas_uniform(54999, 55001, 24, ddr, obs="@")
    toas_ssb_year = make_fake_toas_uniform(54900, 55265, 48, ddr, obs="@")
    toas_gbt = make_fake_toas_uniform(54900, 55265, 48, ddr, obs="gbt")
    dl_ssb = ddr.components["BinaryDDR"].binarymodel_delay(toas_ssb).to_value(u.s)
    dd_ssb = ddk.components["BinaryDDK"].binarymodel_delay(toas_ssb).to_value(u.s)
    dl_gbt = ddr.components["BinaryDDR"].binarymodel_delay(toas_gbt).to_value(u.s)
    dd_gbt = ddk.components["BinaryDDK"].binarymodel_delay(toas_gbt).to_value(u.s)
    interior = np.max(np.abs(dl_ssb - dd_ssb))
    annual = np.max(np.abs(dl_gbt - dd_gbt))
    assert np.all(np.isfinite(dl_ssb)) and np.all(np.isfinite(dd_gbt))
    assert annual > 1e-10
    assert annual < 1e-3
    assert interior < annual
    bl = ddr.components["BinaryDDR"]
    bl.update_binary_object(toas_gbt, bl._upstream_delay(toas_gbt))
    sh_gbt = np.asarray(bl.binary_instance.evaluate().Delta_S, dtype=np.longdouble)
    bl.update_binary_object(toas_ssb_year, bl._upstream_delay(toas_ssb_year))
    sh_ssb = np.asarray(bl.binary_instance.evaluate().Delta_S, dtype=np.longdouble)
    # Projector in B_S: annual Shapiro is not a residual identity with DDK.
    assert np.max(np.abs(sh_gbt - sh_ssb)) > 1e-12
    assert ddk.T0.value != pytest.approx(ddr.TASC.value)
    assert ddk.KOM.value == pytest.approx(ddr.KOM.value)


def test_pb_resolves_the_tasc_epoch():
    """``pb()`` must not ask a TASC-based model for ``T0``.

    DDR carries ``TASC`` and no ``T0``, so the old
    ``binary_model_name.startswith("ELL1")`` dispatch raised ``AttributeError``
    here -- which took out anything that reached ``pb()``, including pyvela's
    default TASC prior, before any delay was evaluated. A populated ``PBDOT``
    is part of the case: without one, ``pb()`` returns early and never uses the
    epoch at all.
    """
    m = get_model(StringIO(example_par()))
    assert m.components["BinaryDDR"].binary_epoch_name == "TASC"
    assert np.isclose(m.pb()[0].to_value(u.d), m.PB.quantity.to_value(u.d))

    # `absorb_gw` makes PBDOT an ordinary parameter rather than the kinematic
    # view, which is the shape a released DDR par has.
    m2 = get_model(StringIO(example_par(DDRPBDOT="absorb_gw", PBDOT="1e-12")))
    assert np.isclose(m2.pb()[0].to_value(u.d), m2.PB.quantity.to_value(u.d))
