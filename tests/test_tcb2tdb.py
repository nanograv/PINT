"""Tests for `pint.models.tcb_conversion` and the `tcb2tdb` script."""

import os
import re
from copy import deepcopy
from io import StringIO

import astropy.units as u
import erfa
import numpy as np
import pytest
from astropy.table import Table

from pint import DMconst, dmu
from pint.config import examplefile
from pint.models.model_builder import ModelBuilder, get_model
from pint.models.parameter import AngleParameter
from pint.models.tcb_conversion import (
    TCB_TDB_F,
    TCB_TDB_K,
    _k_power_minus_one,
    convert_tcb_tdb,
)
from pint.pulsar_mjd import time_from_longdouble
from pint.scripts import tcb2tdb

simplepar = """
PSR              PSRTEST
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
BINARY              BT
T0                  53750
A1                  100.0 1 0.1
ECC                 1.0
OM                  0.0
PB                  10.0
FD1                 1e-3
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TCB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""

# Forward TCB->TDB on a single Time object matches Astropy to ~0.01 ns.
# Independent conversions and TCB<->TDB round-trips go through ERFA's
# two-part JD. On astropy 5.0.5 (oldest supported) that lands within a
# few ULPs of a day, ~0.05 ns, which is still well below the 1 ns
# no-refit bound.
_TCB_TDB_ROUNDTRIP_NS = 0.1


def _delta_seconds(t1, t2):
    """Difference of two ``Time`` objects in seconds, from two-part JD."""
    delta = t1 - t2
    day = np.longdouble(86400)
    return np.longdouble(delta.jd1) * day + np.longdouble(delta.jd2) * day


def _spin_phase_difference(f0_tcb, f1_tcb, f0_tdb, f1_tdb, dt_tcb, dt_tdb):
    """``φ_tdb - φ_tcb`` from increments, so ``F0 Δt`` does not cancel.

    ``F0 Δt`` is ~10^9 turns here. Subtracting two such phases spends the
    mantissa on the large term. Split ``F0_tdb = F0_tcb + dF0`` and
    ``Δt_tdb = Δt_tcb + dΔt`` instead; the two ~100-turn pieces are the
    TCB/TDB rate correction and they cancel at the size of that correction.
    """
    df0 = f0_tdb - f0_tcb
    df1 = f1_tdb - f1_tcb
    ddt = dt_tdb - dt_tcb
    dphi = f0_tcb * ddt + df0 * dt_tdb
    dphi += np.longdouble("0.5") * (
        f1_tcb * ddt * (dt_tdb + dt_tcb) + df1 * dt_tdb * dt_tdb
    )
    return dphi


@pytest.mark.parametrize("backwards", [True, False])
def test_convert_units(backwards):
    with pytest.raises(ValueError):
        m = ModelBuilder()(StringIO(simplepar))

    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    f0_tcb = m.F0.value
    pb_tcb = m.PB.value
    convert_tcb_tdb(m, backwards=backwards)
    assert m.UNITS.value == ("TCB" if backwards else "TDB")
    assert np.isclose(m.F0.value / f0_tcb, pb_tcb / m.PB.value)


def test_convert_units_roundtrip():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    m_ = deepcopy(m)
    convert_tcb_tdb(m, backwards=False)
    convert_tcb_tdb(m, backwards=True)

    for par in m.params:
        p = getattr(m, par)
        p_ = getattr(m_, par)
        if p.value is None:
            assert p_.value is None
        elif isinstance(p.value, str):
            assert getattr(m, par).value == getattr(m_, par).value
        else:
            assert np.isclose(getattr(m, par).value, getattr(m_, par).value)


def test_coordinate_epoch_uses_astropy_iau_tdb():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    original = time_from_longdouble(m.PEPOCH.value, "tcb")
    expected = original.tdb
    m.PEPOCH.uncertainty_value = 1

    report = convert_tcb_tdb(m)

    assert abs((m.PEPOCH.quantity - expected).to_value(u.ns)) < 0.01
    assert np.isclose(m.PEPOCH.uncertainty_value, TCB_TDB_F)
    assert m.PEPOCH.time_scale == "tdb"
    assert "PEPOCH" in report.converted

    convert_tcb_tdb(m, backwards=True)
    assert abs((m.PEPOCH.quantity - original).to_value(u.ns)) < _TCB_TDB_ROUNDTRIP_NS
    assert m.PEPOCH.time_scale == "tcb"


def test_prefix_coordinate_epoch_keeps_inner_time_scale_consistent():
    model = ModelBuilder()(
        StringIO(
            """
PSR TEST
F0 100
PEPOCH 55000
GLEP_1 55100
GLF0_1 1e-6
UNITS TDB
"""
        )
    )

    convert_tcb_tdb(model, backwards=True)

    assert model.GLEP_1.time_scale == "tcb"
    assert model.GLEP_1.param_comp.time_scale == "tcb"
    assert model.GLEP_1.quantity.scale == "tcb"
    value = model.GLEP_1.value
    model.GLEP_1.value = value
    assert model.GLEP_1.quantity.scale == "tcb"


def test_coordinate_epoch_includes_tdb0():
    reference_mjd = np.longdouble("43144.0003725")
    converted = time_from_longdouble(reference_mjd, "tcb").tdb
    numeric_offset = (converted.mjd_long - reference_mjd) * u.day

    assert np.isclose(numeric_offset.to_value(u.us), -65.5, atol=0.01)


def test_fixed_frequency_dm_and_fd_scaling():
    m = ModelBuilder()(
        StringIO(
            simplepar.replace(
                "DM              223.9  1",
                "DM              223.9  1\nDM1 1e-3\nDM2 2e-5\nDMEPOCH 53750",
            )
        ),
        allow_tcb="raw",
    )
    dm0, dm1, dm2 = m.DM.value, m.DM1.value, m.DM2.value
    frequencies = np.array([0.7, 1.4, 3.0]) * u.GHz
    fd_delay = m.FD_delay_frequency(frequencies)
    tcb_mjds = np.array([53750.0, 54000.0, 55000.0], dtype=np.longdouble)
    tdb_mjds = np.array(
        [time_from_longdouble(t, "tcb").tdb.mjd_long for t in tcb_mjds],
        dtype=np.longdouble,
    )
    dm_component = m.components["DispersionDM"]
    dm_tcb = dm_component.base_dm({"tdbld": tcb_mjds})
    dm_delay_tcb = dm_component.dispersion_time_delay(dm_tcb, frequencies)

    convert_tcb_tdb(m)

    assert np.isclose(m.DM.value / dm0, TCB_TDB_F)
    assert np.isclose(m.DM1.value / dm1, 1)
    assert np.isclose(m.DM2.value / dm2, TCB_TDB_K)
    fd_error = m.FD_delay_frequency(frequencies) - fd_delay * TCB_TDB_F
    assert np.max(np.abs(fd_error.to_value(u.ns))) < 0.01

    dm_tdb = dm_component.base_dm({"tdbld": tdb_mjds})
    dm_delay_tdb = dm_component.dispersion_time_delay(dm_tdb, frequencies)
    dm_error = dm_delay_tdb - dm_delay_tcb * TCB_TDB_F
    assert np.max(np.abs(dm_error.to_value(u.ns))) < 0.01


# A TCB par exercising every DM-family exponent: constant, two Taylor orders,
# and one DMX range.
_dmfamily_par = simplepar.replace(
    "DM              223.9  1",
    "DM              223.9  1\n"
    "DM1 1e-3\n"
    "DM2 2e-5\n"
    "DMEPOCH 53750\n"
    "DMX_0001 1e-2 1\n"
    "DMXR1_0001 53700\n"
    "DMXR2_0001 53800",
)


def _convert_with_dilatefreq(dilatefreq):
    """Read ``_dmfamily_par`` with the given DILATEFREQ and convert it."""
    par = re.sub(r"DILATEFREQ\s+\S+", f"DILATEFREQ {dilatefreq}", _dmfamily_par)
    m = ModelBuilder()(StringIO(par), allow_tcb="raw")
    before = {
        name: np.longdouble(m[name].value)
        for name in ("DM", "DM1", "DM2", "DMX_0001", "F0", "A1", "FD1")
    }
    report = convert_tcb_tdb(m)
    after = {name: np.longdouble(m[name].value) for name in before}
    return before, after, report


def test_undilated_source_uses_fixed_frequency_dm_exponents():
    # DILATEFREQ N on the TCB side: freqSSB is the same number in both unit
    # systems, so the DM family carries only the time dimension, K**(q-1).
    before, after, report = _convert_with_dilatefreq("N")
    assert report.convention == "iau2006-undilated-frequency"
    assert np.isclose(after["DM"] / before["DM"], TCB_TDB_F)
    assert np.isclose(after["DM1"] / before["DM1"], 1)
    assert np.isclose(after["DM2"] / before["DM2"], TCB_TDB_K)
    assert np.isclose(after["DMX_0001"] / before["DMX_0001"], TCB_TDB_F)


def test_dilated_source_uses_dilated_dm_exponents():
    # DILATEFREQ Y on the TCB side: freqSSB_tcb = freqSSB_tdb / K, so each
    # inverse-square frequency contributes K**2 on top, giving K**(q+1).
    before, after, report = _convert_with_dilatefreq("Y")
    assert report.convention == "iau2006-dilated-frequency"
    assert np.isclose(after["DM"] / before["DM"], TCB_TDB_K)
    assert np.isclose(after["DM1"] / before["DM1"], TCB_TDB_K**2)
    assert np.isclose(after["DM2"] / before["DM2"], TCB_TDB_K**3)
    assert np.isclose(after["DMX_0001"] / before["DMX_0001"], TCB_TDB_K)


def test_dilatefreq_moves_dm_by_k_squared():
    # The two conventions differ by K**2 - 1 = 3.1e-8 of the dispersion delay.
    # That is ~40 ns at 400 MHz for DM = 50, and it is chromatic, so unlike a
    # timescale offset no phase gauge absorbs it.
    _, after_n, _ = _convert_with_dilatefreq("N")
    _, after_y, _ = _convert_with_dilatefreq("Y")
    assert np.isclose(after_y["DM"] / after_n["DM"], TCB_TDB_K**2, rtol=1e-12)

    freq = 400 * u.MHz
    dm_error = (after_y["DM"] - after_n["DM"]) * dmu
    delay_error = (DMconst * dm_error / freq**2).to(u.ns)
    expected = (
        DMconst
        * (np.longdouble(after_n["DM"]) * float(TCB_TDB_K**2 - 1))
        * dmu
        / freq**2
    ).to(u.ns)
    assert np.isclose(delay_error.value, expected.value, rtol=1e-6)
    assert delay_error > 100 * u.ns  # DM is 223.9 here, so it is large


def test_dilatefreq_does_not_touch_non_dispersive_parameters():
    # Only the DM family declares a nonzero tcb2tdb_freq_power. FD and FDJUMP
    # are log-frequency terms whose dilation residue is ~1e-13 s, so they keep
    # the undilated exponent in both conventions.
    _, after_n, _ = _convert_with_dilatefreq("N")
    _, after_y, _ = _convert_with_dilatefreq("Y")
    for name in ("F0", "A1", "FD1"):
        assert after_y[name] == after_n[name]


def test_dilated_source_leaves_dispersion_components_unaudited():
    # The exponents are right, but PINT cannot certify them by closure: it
    # cannot evaluate a dilated model, and its undilated evaluation of the
    # converted par still differs from a dilated one by ~1e-9 of the
    # dispersion delay. Convert, then say so.
    _, _, report_n = _convert_with_dilatefreq("N")
    _, _, report_y = _convert_with_dilatefreq("Y")

    assert "DispersionDM" not in report_n.unaudited_components
    assert "DispersionDMX" not in report_n.unaudited_components

    assert "DispersionDM" in report_y.unaudited_components
    assert "DispersionDMX" in report_y.unaudited_components
    assert not report_y.accepted
    # Permissive: the DM family is still converted, not refused.
    assert "DM" in report_y.converted
    assert "DMX_0001" in report_y.converted
    assert "DM" not in report_y.unsupported


def test_backwards_conversion_ignores_the_source_dilatefreq():
    # Going TDB -> TCB the TCB side is PINT's own output, which is always
    # undilated, so DM must scale by K and not by F no matter what the input
    # file declared.
    par = re.sub(r"DILATEFREQ\s+\S+", "DILATEFREQ Y", simplepar).replace(
        "UNITS               TCB", "UNITS               TDB"
    )
    m = ModelBuilder()(StringIO(par), allow_tcb="raw")
    assert m.meta["tcb_source_dilatefreq"] is True

    dm0 = np.longdouble(m.DM.value)
    report = convert_tcb_tdb(m, backwards=True)

    assert report.convention == "iau2006-undilated-frequency"
    assert np.isclose(np.longdouble(m.DM.value) / dm0, TCB_TDB_K)


def test_k_power_minus_one_is_the_small_increment():
    # K-1 from L_B/(1-L_B), not from a float64 value sitting next to 1.
    # Use ERFA's L_B, the same primitive as the implementation. Reconstructing
    # L as ``1 - F`` is ``1 - (1 - L_B)`` and is not bit-identical to ``L_B``
    # on 80-bit x87 longdouble (CI), even though it is on IEEE quad.
    L = np.longdouble(erfa.ELB)
    f = TCB_TDB_F
    dk = L / f
    assert _k_power_minus_one(0) == 0
    assert _k_power_minus_one(1) == dk
    assert _k_power_minus_one(-1) == -L
    assert _k_power_minus_one(2) == L * (np.longdouble(1) + f) / (f * f)
    reconstructed = (np.longdouble(1) - f) / f
    naive = np.longdouble(np.float64(TCB_TDB_K) - np.float64(1))
    assert abs(_k_power_minus_one(1) - reconstructed) < abs(naive - reconstructed)


def test_spindown_phase_closes_without_refitting():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    tcb_times = [
        time_from_longdouble(np.longdouble(t), "tcb")
        for t in (53750.0, 54000.0, 55000.0)
    ]
    f0_tcb = np.longdouble(m.F0.value)
    f1_tcb = np.longdouble(m.F1.value)
    pepoch_tcb = time_from_longdouble(m.PEPOCH.value, "tcb")
    dt_tcb = np.array(
        [_delta_seconds(t, pepoch_tcb) for t in tcb_times], dtype=np.longdouble
    )

    convert_tcb_tdb(m)

    f0_tdb = np.longdouble(m.F0.value)
    f1_tdb = np.longdouble(m.F1.value)
    pepoch_tdb = m.PEPOCH.quantity
    dt_tdb = np.array(
        [_delta_seconds(t.tdb, pepoch_tdb) for t in tcb_times], dtype=np.longdouble
    )
    dphi = _spin_phase_difference(f0_tcb, f1_tcb, f0_tdb, f1_tdb, dt_tcb, dt_tdb)
    time_error_ns = np.abs(dphi / f0_tdb) * 1e9
    assert np.max(time_error_ns) < _TCB_TDB_ROUNDTRIP_NS


def test_astrometric_position_closes_at_same_physical_epoch():
    m = ModelBuilder()(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
PMRA 8
PMDEC -5
POSEPOCH 55000
F0 100
PEPOCH 55000
DM 10
UNITS TCB
"""
        ),
        allow_tcb="raw",
    )
    # A raw TCB model is intentionally not evaluable by PINT, so label the
    # source epoch explicitly for this component-level physical-instant test.
    m.POSEPOCH.time_scale = "tcb"
    epoch_tcb = time_from_longdouble(np.longdouble("57000"), "tcb")
    position_tcb = m.coords_as_ICRS(epoch=epoch_tcb)

    convert_tcb_tdb(m)

    position_tdb = m.coords_as_ICRS(epoch=epoch_tcb.tdb)
    assert position_tdb.separation(position_tcb).to_value(u.uas) < 1e-3


def test_solar_system_shapiro_closes_without_refitting():
    class FakeToas:
        def __init__(self, mjds):
            self.table = Table(
                {
                    "tdbld": mjds,
                    "obs_sun_pos": np.tile([1.0, 0.0, 0.0], (len(mjds), 1)) * u.au,
                }
            )

        def __len__(self):
            return len(self.table)

        def get_obss(self):
            return np.full(len(self), "gbt")

    model = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    model.PLANET_SHAPIRO.value = False
    tcb_mjds = np.array([53750.0, 54000.0, 55000.0], dtype=np.longdouble)
    tdb_mjds = np.array(
        [time_from_longdouble(t, "tcb").tdb.mjd_long for t in tcb_mjds],
        dtype=np.longdouble,
    )
    component = model.components["SolarSystemShapiro"]
    delay_tcb = component.solar_system_shapiro_delay(FakeToas(tcb_mjds))

    convert_tcb_tdb(model)

    delay_tdb = component.solar_system_shapiro_delay(FakeToas(tdb_mjds))
    error = delay_tdb - delay_tcb * TCB_TDB_F
    assert np.max(np.abs(error.to_value(u.ns))) < 0.01


def test_fdjump_coefficients_scale_as_time_at_fixed_frequency():
    m = ModelBuilder()(
        StringIO(simplepar + "\nFD1JUMP -sys backend 0.01\n"),
        allow_tcb="raw",
    )
    original = m.FD1JUMP1.value

    convert_tcb_tdb(m)

    assert np.isclose(m.FD1JUMP1.value / original, TCB_TDB_F)


def test_new_mask_parameters_preserve_conversion_metadata():
    model = ModelBuilder()(
        StringIO(
            simplepar
            + "\nFD1JUMP -sys backend1 0.01\n"
            + "FD1JUMP -sys backend2 0.02\n"
        ),
        allow_tcb="raw",
    )

    assert model.FD1JUMP2.tcb2tdb_scale_exponent == -1
    assert not model.FD1JUMP2.tcb2tdb_invariant


def test_angle_parameter_stores_conversion_metadata():
    parameter = AngleParameter(
        name="TEST",
        value=1,
        units="rad",
        convert_tcb2tdb=False,
        tcb2tdb_scale_factor=u.Quantity(1),
        tcb2tdb_scale_exponent=2,
        tcb2tdb_invariant=True,
    )

    assert parameter.tcb2tdb_scale_exponent == 2
    assert parameter.tcb2tdb_invariant


def test_utc_selectors_and_parallax_are_explicit_invariants():
    m = ModelBuilder()(
        StringIO(
            simplepar
            + """
PX 1.2
START 53000
FINISH 54000
DMXR1_0001 53500
DMXR2_0001 53600
DMX_0001 0.1
"""
        ),
        allow_tcb="raw",
    )
    m.add_DMX_ranges([53400], [53500], indices=[3], dmxs=[0.3])
    values = {
        name: m[name].value
        for name in (
            "PX",
            "START",
            "FINISH",
            "DMXR1_0001",
            "DMXR2_0001",
            "DMXR1_0003",
            "DMXR2_0003",
        )
    }

    report = convert_tcb_tdb(m)

    assert {name: m[name].value for name in values} == values
    assert set(values) <= set(report.invariant)
    assert np.isclose(m.DMX_0001.value, 0.1 * TCB_TDB_F)
    assert np.isclose(m.DMX_0003.value, 0.3 * TCB_TDB_F)


def test_report_accepts_only_audited_forward_components():
    m = ModelBuilder()(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
F0 100
PEPOCH 55000
DM 12.5
UNITS TCB
"""
        ),
        allow_tcb="raw",
    )
    assert convert_tcb_tdb(m).accepted

    binary = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    report = convert_tcb_tdb(binary)
    assert not report.accepted
    assert "BinaryBT" in report.unaudited_components


def test_noop_conversion_preserves_model_acceptance_status():
    model = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    first_report = convert_tcb_tdb(model)
    second_report = convert_tcb_tdb(model)

    assert not first_report.accepted
    assert not second_report.accepted
    assert "BinaryBT" in second_report.unaudited_components


def test_absolute_phase_is_not_certified_without_closure_test():
    model = ModelBuilder()(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
F0 100
PEPOCH 55000
DM 10
TZRMJD 55000
TZRSITE ssb
TZRFRQ 1400
UNITS TCB
"""
        ),
        allow_tcb="raw",
    )

    report = convert_tcb_tdb(model)

    assert "AbsPhase" in report.unaudited_components


def test_absolute_phase_loads_without_units():
    model = get_model(examplefile("test-wb-0.par"))
    assert model.TZRMJD.time_scale == "tdb"

    minimal = get_model(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
F0 100
PEPOCH 55000
DM 10
TZRMJD 55001
"""
        )
    )
    assert minimal.TZRSITE.value == "ssb"
    assert minimal.TZRMJD.time_scale == "tdb"


def test_effective_dimensionality():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb=True)
    assert m.PEPOCH.effective_dimensionality == 1
    assert m.F0.effective_dimensionality == -1
    assert m.F1.effective_dimensionality == -2

    assert m.POSEPOCH.effective_dimensionality == 1
    assert m.RAJ.effective_dimensionality == 0
    assert m.DECJ.effective_dimensionality == 0
    assert m.PMRA.effective_dimensionality == -1
    assert m.PMDEC.effective_dimensionality == -1
    assert m.PX.effective_dimensionality == -1

    assert m.DM.effective_dimensionality == -1

    assert m.T0.effective_dimensionality == 1
    assert m.A1.effective_dimensionality == 1
    assert m.ECC.effective_dimensionality == 0
    assert m.OM.effective_dimensionality == 0
    assert m.PB.effective_dimensionality == 1

    assert m.NE_SW.effective_dimensionality == -2


def test_dm_scaling_factor():
    m = get_model(
        StringIO(
            """
            PSR         TEST
            F0          100
            F1          -1e-14
            PEPOCH      55000
            DMEPOCH     55000
            DM          12.5
            DM1         -0.001
            DM2         1e-5
            DMXR1_0001  51000
            DMXR2_0001  51000
            DMX_0001    0.002
            DMWXEPOCH   55000
            DMWXFREQ_0001   0.001
            DMWXSIN_0001    0.0003
            DMWXCOS_0001    0.0002
            """
        )
    )

    for param in m.params:
        par = m[param]

        if hasattr(par, "units") and par.units == dmu:
            assert not par.convert_tcb2tdb or par.tcb2tdb_scale_factor == DMconst


def test_tcb2tdb(tmp_path):
    tmppar1 = tmp_path / "tmp1.par"
    tmppar2 = tmp_path / "tmp2.par"
    with open(tmppar1, "w") as f:
        f.write(simplepar)

    cmd = f"{tmppar1} {tmppar2}"
    tcb2tdb.main(cmd.split())

    assert os.path.isfile(tmppar2)

    m2 = ModelBuilder()(tmppar2)
    assert m2.UNITS.value == "TDB"
