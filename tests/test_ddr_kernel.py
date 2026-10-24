"""DDR kernel identities (Kepler, precession, Dual, Shapiro).

Proposal §8/§10: this is the one permitted extra test file besides
``test_ddr.py``. Import Dual/Kepler via ``DDR_model`` where practical;
``ddr_kepler`` is kernel-private.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

from pint.exceptions import InvalidModelParameters
from pint.models.stand_alone_psr_binaries import (
    DDR_model as geo,
    DDR_model as kep,
    DDR_model as ddr,
    DDR_model as prec,
)
from pint.models.stand_alone_psr_binaries.binary_orbits import OrbitFBX
from pint.models.stand_alone_psr_binaries.DDR_model import (
    _LD,
    Dual,
    DDRmodel,
    as_ld,
    roemer,
    sqrt,
    v_from_IJ,
    value,
)

_TWO_PI = _LD(2) * np.pi
_DAY = _LD(86400)
# 80-bit x87 longdouble (CI) has eps ~ 1e-19; IEEE quad (this aarch64
# container) has eps ~ 1e-34. Tolerances written on quad need a portable floor.
_LD_EPS = float(np.finfo(np.longdouble).eps)
_IEEE_QUAD = _LD_EPS < 1e-30


def _ddr_rtol(tight, floor=2e-16):
    return max(float(tight), float(floor), 512.0 * _LD_EPS)


def _ddr_atol(tight, scale=1.0, floor=2e-16):
    mag = float(np.max(np.abs(np.asarray(scale, dtype=np.longdouble))))
    return max(float(tight), float(floor), 512.0 * _LD_EPS * (mag + 1.0))


def _ddr_fd_rtol(tight=5e-5, loose=3e-2):
    """Central-difference columns need more slop on 80-bit than on quad.

    CI's 80-bit x87 longdouble (eps ~ 1e-19) plus a 1e-16 Kepler residual
    make wrap-level FD disagree with Dual at the percent level; IEEE quad
    stays at ``tight``.
    """
    return float(tight) if _IEEE_QUAD else max(float(tight), float(loose))


def _ld_grid():
    return as_ld(np.linspace(-3.0, 11.0, 64))


def _a1_from_masses(pb_d, m_p, m_c, c):
    pb_s = as_ld(pb_d) * _DAY
    n = _LD(2) * np.pi / pb_s
    s = np.sqrt((_LD(1) - c) * (_LD(1) + c))
    tsun = ddr.tsun_s()
    M = as_ld(m_p) + as_ld(m_c)
    return (
        tsun ** (_LD(1) / _LD(3))
        * (as_ld(m_c) * s)
        / (M ** (_LD(2) / _LD(3)) * n ** (_LD(2) / _LD(3)))
    )


def physical_params(**extra):
    pb = _LD(2)
    mc = _LD("0.2")
    c = _LD("0.5")
    mp = _LD("1.4")
    a1 = _a1_from_masses(pb, mp, mc, c)
    p = dict(
        PB=pb,
        A1=a1,
        TASC=_LD(53000),
        EPS1=_LD("0.001"),
        EPS2=_LD("-0.0004"),
        M2=mc,
        COSI=c,
        DDRPK=True,
        DDRPBDOT="kinematic",
        DDRGEO=False,
        DDRKINE=False,
        XPBDOT=_LD(0),
    )
    p.update(extra)
    return p


def _times(model, n=48):
    pb = model.PB
    return model.TASC + as_ld(np.linspace(-2.5, 4.0, n)) * pb


# --- Kepler / static orbit ---


def _ld_grid():
    return as_ld(np.linspace(-3.0, 11.0, 64))


def test_mean_longitude_is_longdouble_and_quadratic():
    dt = as_ld(np.array([0.0, 86400.0, 2 * 86400.0]))
    pb = _LD(86400)
    p = _LD("1e-8")
    lam = kep.mean_longitude(dt, pb, p)
    assert value(lam).dtype == np.longdouble
    np.testing.assert_array_equal(value(lam)[0], _LD(0))
    # p = 0 reduces to 2π dt/PB
    lam0 = kep.mean_longitude(dt, pb, _LD(0))
    np.testing.assert_allclose(
        value(lam0), _TWO_PI * dt / pb, rtol=0, atol=np.finfo(np.longdouble).eps
    )


def test_orbital_phase_pb_chart_matches_quadratic():
    dt = as_ld(np.linspace(-8, 8, 65)) * _DAY
    pb = _LD(2) * _DAY
    p = _LD("1e-12")
    old = kep.mean_longitude(dt, pb, p)
    new, slope, _curvature = kep.orbital_phase(dt, [1 / pb, -p / pb**2])
    np.testing.assert_allclose(value(new), value(old), rtol=0, atol=_LD("1e-16"))
    assert np.all(value(slope) > 0)


def test_orbital_phase_fbx_tangent_and_taylor_shift():
    coeffs = as_ld([1 / (2 * _DAY), -1e-20, 1e-28, -1e-36, 1e-44, -1e-52])
    dt = as_ld(np.linspace(-7.5, 7.5, 31)) * _LD("365.25") * _DAY
    lam, slope, _curvature = kep.orbital_phase(dt, coeffs)
    assert np.all(np.isfinite(value(lam)))
    assert np.all(value(slope) > 0)

    delta = _LD("1234567.89")
    shifted = kep.taylor_shift(coeffs, delta)
    old, _, _ = kep.orbital_phase(dt + delta, coeffs)
    origin, _, _ = kep.orbital_phase(delta, coeffs)
    new, _, _ = kep.orbital_phase(dt, shifted)
    np.testing.assert_allclose(
        value(new), value(old - origin), rtol=0, atol=_LD("5e-15")
    )

    seeded = [Dual(c, _LD(1) if j == 5 else _LD(0)) for j, c in enumerate(coeffs)]
    tangent, _, _ = kep.orbital_phase(dt, seeded)
    expected = _TWO_PI * dt**6 / _LD(720)
    np.testing.assert_allclose(tangent.d, expected, rtol=_ddr_rtol(2e-18), atol=0)


def test_orbital_phase_matches_orbitfbx_through_fb5():
    coeffs = as_ld([1 / (2 * _DAY), -1e-20, 1e-28, -1e-36, 1e-44, -1e-52])
    dt = as_ld(np.linspace(-7.5, 7.5, 61)) * _LD("365.25") * _DAY
    names = [f"FB{j}" for j in range(len(coeffs))]
    parent = SimpleNamespace(binary_params=names, tt0=dt * u.s)
    for j, coefficient in enumerate(coeffs):
        setattr(parent, f"FB{j}", coefficient * u.s ** (-(j + 1)))
    reference = OrbitFBX(parent, names).orbits().to_value(u.one)
    lam, _, _ = kep.orbital_phase(dt, coeffs)
    np.testing.assert_allclose(value(lam) / _TWO_PI, reference, rtol=0, atol=1e-15)


def test_phase_root_matches_known_fbx_offset():
    coeffs = as_ld([1 / (2 * _DAY), -1e-20, 1e-28])
    expected = _LD("43210.123")
    target, _, _ = kep.orbital_phase(expected, coeffs)
    got = kep.solve_phase_offset(coeffs, value(target), 0, 2 * _DAY)
    assert got == pytest.approx(expected, rel=0, abs=1e-12)


def test_orbital_phase_refuses_nonpositive_slope():
    with pytest.raises(InvalidModelParameters, match="slope"):
        kep.orbital_phase(as_ld([0, 2]), as_ld([1, -1]))


def test_phase_slope_sample_times_includes_inter_node_minimum():
    amplitude = _LD("1e-4")
    coeffs = as_ld([_LD("0.09") * amplitude, -amplitude, 2 * amplitude])
    samples = kep.phase_slope_sample_times(coeffs, _LD(0), _LD(64), grid_size=65)
    assert np.min(np.abs(samples - _LD("0.5"))) < _LD("1e-6")
    with pytest.raises(InvalidModelParameters, match="slope"):
        kep.orbital_phase(samples, coeffs)


def _alternating_fbx(k=12, years=_LD(5), term_cycles=_LD("0.02")):
    t_scale = years * _LD("365.25") * _DAY
    coeffs = [1 / (2 * _DAY)]
    for j in range(1, k + 1):
        coeffs.append(
            ((-1) ** j) * term_cycles * _LD(math.factorial(j + 1)) / t_scale ** (j + 1)
        )
    return as_ld(coeffs)


def test_orbital_phase_fb12_taylor_shift_and_tangent():
    coeffs = _alternating_fbx()
    dt = as_ld(np.linspace(-5, 5, 41)) * _LD("365.25") * _DAY
    lam, slope, _curvature = kep.orbital_phase(dt, coeffs)
    assert np.all(np.isfinite(value(lam)))
    assert np.all(value(slope) > 0)
    magnitude = np.maximum(np.abs(value(lam)), _LD(1))

    delta = _LD("-2.5") * _LD("365.25") * _DAY
    shifted = kep.taylor_shift(coeffs, delta)
    old, _, _ = kep.orbital_phase(dt + delta, coeffs)
    origin, _, _ = kep.orbital_phase(delta, coeffs)
    new, _, _ = kep.orbital_phase(dt, shifted)
    err = np.abs(value(new) - value(old - origin))
    np.testing.assert_array_less(err / magnitude, _LD("1e-14"))

    seeded = [Dual(c, _LD(1) if j == 12 else _LD(0)) for j, c in enumerate(coeffs)]
    tangent, _, _ = kep.orbital_phase(dt, seeded)
    expected = _TWO_PI * dt**13 / _LD(math.factorial(13))
    np.testing.assert_allclose(tangent.d, expected, rtol=_LD("2e-14"), atol=0)


def test_orbital_phase_matches_orbitfbx_through_fb12_mild_series():
    coeffs = as_ld(
        [1 / (2 * _DAY)]
        + [((-1) ** j) * _LD(10) ** -(20 + 8 * max(j - 1, 0)) for j in range(1, 13)]
    )
    dt = as_ld(np.linspace(-2, 5, 31)) * _LD("365.25") * _DAY
    names = [f"FB{j}" for j in range(len(coeffs))]
    parent = SimpleNamespace(binary_params=names, tt0=dt * u.s)
    for j, coefficient in enumerate(coeffs):
        setattr(parent, f"FB{j}", coefficient * u.s ** (-(j + 1)))
    reference = OrbitFBX(parent, names).orbits().to_value(u.one)
    lam, _, _ = kep.orbital_phase(dt, coeffs)
    phase = value(lam) / _TWO_PI
    magnitude = np.maximum(np.abs(reference), _LD("1e-12"))
    np.testing.assert_allclose(phase, reference, rtol=0, atol=1e-12)
    np.testing.assert_array_less(np.abs(phase - reference) / magnitude, 1e-12)


def test_kernel_fbx_centering_preserves_phase_and_reference_frequency():
    coeffs = _alternating_fbx()
    tasc = _LD(53000)
    model = DDRmodel(
        **physical_params(
            FB_COEFFS=coeffs,
            TASC=tasc,
            DDRPK=False,
            DDRPBDOT="absorb_gw",
            DDRKINE=False,
            PBDOT=0,
        )
    )
    times = tasc + as_ld(np.linspace(-2, 8, 17)) * _LD("365.25")
    state = model.evaluate(times)
    dt = (times - tasc) * _DAY
    lam, lamdot, _ = kep.orbital_phase(dt, coeffs)
    magnitude = np.maximum(np.abs(value(lam)), _LD(1))
    np.testing.assert_array_less(
        np.abs(state.lam - value(lam)) / magnitude, _LD("1e-14")
    )
    np.testing.assert_allclose(state.n, _TWO_PI * coeffs[0], rtol=0, atol=_LD("1e-18"))
    np.testing.assert_allclose(state.lamdot, value(lamdot), rtol=_LD("1e-14"), atol=0)
    assert np.all(state.inverse_timing_ref_error_s >= 0)

    tangent = model.evaluate(times, tangent_par="TASC")
    np.testing.assert_allclose(
        tangent.d_lam, -state.lamdot * _DAY, rtol=_LD("2e-12"), atol=0
    )


def test_kepler_circular_is_identity():
    lam = _ld_grid()
    F, D, c_e, s_e = kep.solve_F(lam, _LD(0), _LD(0))
    lam_red, _ = kep.reduce_longitude(lam)
    np.testing.assert_allclose(F, value(lam_red), rtol=0, atol=_LD("1e-18"))
    np.testing.assert_allclose(D, np.ones_like(F), rtol=0, atol=_LD("1e-18"))
    np.testing.assert_array_equal(c_e, np.zeros_like(F))
    np.testing.assert_array_equal(s_e, np.zeros_like(F))


def test_kepler_residual_and_root_error_scale():
    h, k = _LD("0.3"), _LD("0.4")  # e = 0.5
    e = np.sqrt(h * h + k * k)
    lam = _ld_grid()
    F, D, c_e, s_e = kep.solve_F(lam, h, k)
    lam_red, _ = kep.reduce_longitude(lam)
    R = F - k * np.sin(F) + h * np.cos(F) - value(lam_red)
    assert np.max(np.abs(R)) <= _LD("1e-16")
    assert np.max(np.abs(R) / (_LD(1) - e)) <= _LD("2e-16") / (_LD(1) - e) + _LD(
        "1e-18"
    )
    x = _LD("1.963")
    delay_scale = x * np.abs(R) / D
    assert np.max(delay_scale) < _LD("1e-15")


def test_kepler_refuses_oversize_eccentricity():
    with pytest.raises(InvalidModelParameters):
        kep.solve_F(_LD(0.1), _LD("0.8"), _LD("0.7"))  # e > 0.99


def test_static_XY_matches_polar_dd():
    rng = np.random.default_rng(0)
    lam = as_ld(rng.uniform(-20.0, 20.0, size=80))
    # Stay well inside e_max and away from the polar origin for atan2.
    e = as_ld(rng.uniform(1e-4, 0.9, size=80))
    om = as_ld(rng.uniform(-np.pi, np.pi, size=80))
    h = e * np.sin(om)
    k = e * np.cos(om)
    F, D, c_e, s_e = kep.solve_F(lam, h, k)
    X0, Y0, *_ = kep.static_XY(F, h, k)
    Xp, Yp = kep.polar_XY(F, h, k)
    np.testing.assert_allclose(value(X0), value(Xp), rtol=0, atol=_LD("2e-15"))
    np.testing.assert_allclose(value(Y0), value(Yp), rtol=0, atol=_LD("2e-15"))
    np.testing.assert_allclose(
        value(X0) ** 2 + value(Y0) ** 2,
        (_LD(1) - c_e) ** 2,
        rtol=0,
        atol=_LD("2e-15"),
    )


def test_circular_origin_static_jacobians():
    lam = _ld_grid()
    dX_dh, dX_dk, dY_dh, dY_dk = kep.circular_origin_dXY_dhk(lam)

    def _col(dh, dk):
        h = Dual(_LD(0), as_ld(dh))
        k = Dual(_LD(0), as_ld(dk))
        F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
        X0, Y0, *_ = kep.static_XY(F, h, k)
        return X0.d, Y0.d

    xh, yh = _col(1, 0)
    xk, yk = _col(0, 1)
    np.testing.assert_allclose(xh, dX_dh, rtol=0, atol=_LD("2e-15"))
    np.testing.assert_allclose(xk, dX_dk, rtol=0, atol=_LD("2e-15"))
    np.testing.assert_allclose(yh, dY_dh, rtol=0, atol=_LD("2e-15"))
    np.testing.assert_allclose(yk, dY_dk, rtol=0, atol=_LD("2e-15"))


def test_circular_origin_projections_are_trig():
    lam = _ld_grid()
    F, D, c_e, s_e = kep.solve_F(lam, 0, 0)
    X0, Y0, *_ = kep.static_XY(F, 0, 0)
    lam_red, _ = kep.reduce_longitude(lam)
    np.testing.assert_allclose(
        value(Y0), np.sin(value(lam_red)), rtol=0, atol=_LD("1e-18")
    )
    np.testing.assert_allclose(
        value(X0), np.cos(value(lam_red)), rtol=0, atol=_LD("1e-18")
    )


# --- Precession ---


def test_ce_se_invariant_under_precession_rotation():
    lam = as_ld(np.linspace(-40.0, 40.0, 50))
    h, k = _LD("0.2"), _LD("-0.15")
    F, D, c_e, s_e = kep.solve_F(lam, h, k)
    X0, Y0, *_ = kep.static_XY(F, h, k)
    e2 = h * h + k * k
    q = prec.q_nu_minus_M(c_e, s_e, e2)
    q_star = prec.q_at_tasc(h, k)
    kappa = _LD("0.001")
    delta = prec.precession_delta(lam, q, q_star, kappa)
    X, Y = prec.rotate_XY(X0, Y0, delta)
    # Rebuild from rotated (h,k) at the *new* F+δ would keep c_e,s_e;
    # rotating (h,k) at fixed F would not. Check the invariants of the
    # reference Kepler root are unchanged by the X,Y rotation.
    np.testing.assert_allclose(
        value(X) ** 2 + value(Y) ** 2,
        value(X0) ** 2 + value(Y0) ** 2,
        rtol=0,
        atol=_LD("2e-15"),
    )
    np.testing.assert_allclose(
        value(X) ** 2 + value(Y) ** 2,
        (_LD(1) - c_e) ** 2,
        rtol=0,
        atol=_LD("2e-15"),
    )


def test_circular_limit_Y_is_sin_F_plus_delta():
    lam = as_ld(np.linspace(-12.0, 30.0, 40))
    kappa = _LD("0.002")
    h = k = _LD(0)
    F, D, c_e, s_e = kep.solve_F(lam, h, k)
    X0, Y0, *_ = kep.static_XY(F, h, k)
    q = prec.q_nu_minus_M(c_e, s_e, _LD(0))
    q_star = prec.q_at_tasc(h, k)
    delta = prec.precession_delta(lam, q, q_star, kappa)
    X, Y = prec.rotate_XY(X0, Y0, delta)
    np.testing.assert_allclose(value(q), 0, atol=_LD("1e-18"))
    np.testing.assert_allclose(value(q_star), 0, atol=_LD("1e-18"))
    np.testing.assert_allclose(value(delta), kappa * lam, rtol=0, atol=_LD("1e-16"))
    np.testing.assert_allclose(
        value(Y), np.sin(F + value(delta)), rtol=0, atol=_LD("2e-15")
    )


def test_precession_no_jump_across_old_atan2_cut():
    # Crossing the polar ω = ±π cut must not jump in (X, Y).
    kappa = _LD("0.001")
    lam = as_ld(np.linspace(0.0, 8.0, 64))
    e = _LD("0.05")
    om = as_ld(np.linspace(np.pi - 0.2, np.pi + 0.2, 40))
    Ys = []
    for omi in om:
        h = e * np.sin(omi)
        k = e * np.cos(omi)
        F, D, c_e, s_e = kep.solve_F(lam, h, k)
        X0, Y0, *_ = kep.static_XY(F, h, k)
        q = prec.q_nu_minus_M(c_e, s_e, e * e)
        q_star = prec.q_at_tasc(h, k)
        delta = prec.precession_delta(lam, q, q_star, kappa)
        _X, Y = prec.rotate_XY(X0, Y0, delta)
        Ys.append(value(Y))
    Ys = np.stack(Ys)
    dY = np.max(np.abs(np.diff(Ys, axis=0)))
    assert dY < _LD("0.05")  # smooth in ω; no 2π unwrap jump


def test_many_positive_and_negative_orbits_secular_circular():
    """At e=0, secular δ=κλ over many signed orbits (proposal §10.2)."""
    pb_s = _LD(86400)
    n_orb = as_ld(np.array([-2000, -50, -1, 0, 1, 50, 2000]))
    dt = n_orb * pb_s
    lam = kep.mean_longitude(dt, pb_s, _LD(0))
    _red, n_rec = kep.reduce_longitude(lam)
    np.testing.assert_array_equal(n_rec, n_orb)
    h, k = _LD(0), _LD(0)
    F, D, c_e, s_e = kep.solve_F(lam, h, k)
    q = prec.q_nu_minus_M(c_e, s_e, _LD(0))
    q_star = prec.q_at_tasc(h, k)
    kappa = _LD("1e-4")
    delta = prec.precession_delta(lam, q, q_star, kappa)
    np.testing.assert_allclose(value(q), 0, atol=_LD("1e-18"))
    np.testing.assert_allclose(value(q_star), 0, atol=_LD("1e-18"))
    np.testing.assert_allclose(
        value(delta), kappa * value(lam), rtol=0, atol=_LD("1e-12")
    )


def test_circular_origin_d_delta_requires_live_q_star():
    lam = as_ld(np.linspace(0.2, 5.0, 24))
    kappa = _LD("0.003")
    boxed_h, boxed_k = prec.circular_origin_d_delta_dhk(lam, kappa)

    h = Dual(_LD(0), _LD(1))
    k = Dual(_LD(0), _LD(0))
    F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
    q = prec.q_nu_minus_M(c_e, s_e, h * h + k * k)
    q_star_live = prec.q_at_tasc(h, k)
    delta_live = prec.precession_delta(lam, q, q_star_live, kappa)
    np.testing.assert_allclose(delta_live.d, boxed_h, rtol=0, atol=_LD("2e-14"))

    # Freezing q(τ) at setup (tangent 0) must miss the boxed +2κ piece.
    q_star_frozen = Dual(value(q_star_live), np.zeros_like(value(q_star_live)))
    delta_frozen = prec.precession_delta(lam, q, q_star_frozen, kappa)
    assert np.max(np.abs(delta_frozen.d - boxed_h)) > _LD("1e-6")

    k = Dual(_LD(0), _LD(1))
    h = Dual(_LD(0), _LD(0))
    F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
    q = prec.q_nu_minus_M(c_e, s_e, h * h + k * k)
    q_star_live = prec.q_at_tasc(h, k)
    delta_live = prec.precession_delta(lam, q, q_star_live, kappa)
    np.testing.assert_allclose(delta_live.d, boxed_k, rtol=0, atol=_LD("2e-14"))


# --- Delay limits / Shapiro / PK ---


def test_physical_a1_scale():
    a1 = _a1_from_masses(2, 1.4, 0.2, 0.5)
    assert 1.5 < float(a1) < 2.5


def test_delay_longdouble_and_finite():
    m = DDRmodel(**physical_params())
    t = _times(m)
    d = m.delay(t)
    assert d.dtype == np.longdouble
    assert np.all(np.isfinite(d))
    # Amplitude of order x, not millisecond-scale from a tiny A1.
    assert np.max(np.abs(d)) > _LD("0.1")
    assert np.max(np.abs(d)) < _LD(10)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("PB", np.nan),
        ("A1", np.inf),
        ("EPS1", np.nan),
        ("EPS2", np.inf),
        ("M2", -0.1),
        ("COSI", np.nan),
        ("KOM", np.inf),
        ("TASC", np.nan),
        ("A1DOT", np.inf),
        ("GGAMMA", np.nan),
        ("OMDOT", np.inf),
        ("PBDOT", np.nan),
    ],
)
def test_kernel_refuses_invalid_primitive_on_every_evaluation(name, bad):
    model = DDRmodel(
        **physical_params(
            DDRPK=False,
            DDRPBDOT="absorb_gw",
            DDRKINE=False,
            **{name: bad},
        )
    )
    with pytest.raises(InvalidModelParameters):
        model.delay(as_ld([53000]))


def test_kernel_refuses_nonfinite_fbx_coefficient():
    model = DDRmodel(
        **physical_params(
            FB_COEFFS=as_ld([1 / (2 * _DAY), 0, np.nan]),
            DDRPK=False,
            DDRPBDOT="absorb_gw",
            DDRKINE=False,
        )
    )
    with pytest.raises(InvalidModelParameters, match="finite"):
        model.delay(as_ld([53000]))


def test_kernel_checks_reference_axis_independently_of_evolved_axis():
    model = DDRmodel(
        **physical_params(
            A1=-1,
            A1DOT=2 / _DAY,
            DDRPK=False,
            DDRPBDOT="absorb_gw",
            DDRKINE=False,
            PBDOT=0,
        )
    )
    with pytest.raises(InvalidModelParameters, match="x_star"):
        model.delay(as_ld([53001]))


def test_circular_einstein_vanishes_but_state_has_g_gamma():
    m = DDRmodel(**physical_params(EPS1=0, EPS2=0, DDRPBDOT="absorb_gw", PBDOT=0))
    st = m.evaluate(_times(m))
    np.testing.assert_allclose(st.Delta_E, 0, atol=_LD("1e-18"))
    assert st.g_gamma > 0


def test_kappa_independent_of_pb_at_fixed_mass_coords():
    base = physical_params(DDRPBDOT="absorb_gw", PBDOT=0)
    m = DDRmodel(**base)
    st0 = m.evaluate(m.TASC)
    m.update(PB=m.PB * _LD("1.01"))
    st1 = m.evaluate(m.TASC)
    # Mass inversion: κ = 3 T_⊙ m_c s / (x (1-e²)) has no P_B.
    np.testing.assert_allclose(st0.kappa, st1.kappa, rtol=0, atol=_LD("1e-18"))
    assert "pb" not in ddr.kappa_gr.__code__.co_varnames


def test_shapiro_squared_norm_matches_diff_away_from_conjunction():
    m = DDRmodel(**physical_params(DDRPBDOT="absorb_gw", PBDOT=0, COSI=_LD("0.2")))
    t = _times(m, n=80)
    st = m.evaluate(t)
    # Drop points near superior conjunction (small B_S).
    mask = st.B_S_diff > _LD("0.05")
    np.testing.assert_allclose(
        st.B_S[mask], st.B_S_diff[mask], rtol=_LD("1e-12"), atol=_LD("1e-15")
    )


def test_shapiro_near_conjunction_nonnegative_no_clip():
    # Nearly edge-on circular, sample near Y = 1.
    m = DDRmodel(
        **physical_params(
            EPS1=0,
            EPS2=0,
            COSI=_LD("0.02"),
            DDRPBDOT="absorb_gw",
            PBDOT=0,
        )
    )
    # λ = π/2 + ε → Y = sin λ ≈ 1
    pb_d = m.PB
    tasc = m.TASC
    eps = as_ld(np.linspace(-0.05, 0.05, 41))
    t = tasc + (0.25 + eps / (_LD(2) * np.pi)) * pb_d
    st = m.evaluate(t)
    assert np.all(st.B_S >= 0)
    assert np.all(np.isfinite(st.delay))


def test_shapiro_nonpositive_raises():
    # Force a pathological projector that can make B_S ≤ 0 by asking for
    # the helper directly at the point-mass singularity of a face-on skip:
    # s = 1, Y = 1, c_e = 0 → B_S = 0.
    B = ddr.shapiro_B_S_diff(_LD(0), _LD(1))
    with pytest.raises(InvalidModelParameters):
        ddr.shapiro_delay(_LD("1e-6"), B)


def test_mode_table_active_gr_terms():
    tasc = _LD(53000)
    t = tasc + as_ld([0.1, 0.7, 1.3])
    combos = [
        (True, "kinematic", True, True, True),
        (True, "absorb_gw", True, True, False),
        (False, "kinematic", False, False, True),
        (False, "absorb_gw", False, False, False),
    ]
    for pk_on, pbdot, ein, prec_on, quad in combos:
        extra = dict(DDRPK=pk_on, DDRPBDOT=pbdot, DDRGEO=False, DDRKINE=False)
        if not pk_on:
            extra.update(GGAMMA=0, OMDOT=0)
        if pbdot == "absorb_gw":
            extra["PBDOT"] = 0
        else:
            extra["XPBDOT"] = 0
        m = DDRmodel(**physical_params(**extra))
        st = m.evaluate(t)
        assert st.einstein_gr is ein
        assert st.precession_gr is prec_on
        assert st.quadrupole is quad
        if quad:
            assert np.all(st.p_gw != 0)
        else:
            np.testing.assert_allclose(st.p_gw, 0, atol=_LD("1e-30"))
        if pk_on:
            assert st.g_gamma > 0
            assert st.kappa > 0
        else:
            np.testing.assert_allclose(st.g_gamma, 0, atol=0)
            np.testing.assert_allclose(st.kappa, 0, atol=0)
        if pk_on or quad:
            assert st.m_p > 0


def test_ddrkine_n_omits_shk_gal():
    m = DDRmodel(
        **physical_params(
            DDRKINE=False,
            DDRGEO=False,
            DDRPBDOT="absorb_gw",
            PBDOT=_LD("1e-12"),
        )
    )
    st = m.evaluate(m.TASC + _LD(0.1))
    np.testing.assert_allclose(st.p_shk, 0, atol=0)
    np.testing.assert_allclose(st.p_gal, 0, atol=0)
    np.testing.assert_allclose(st.p, _LD("1e-12"), atol=_LD("1e-20"))


def test_galactic_vertical_vanishes_at_zero_distance():
    _a_pl, a_vert, a_los = ddr.galactic_acceleration_los(_LD(0), _LD(0.5), _LD(0.3))
    np.testing.assert_allclose(value(a_vert), 0, atol=_LD("1e-30"))
    # z_psr = z_odot + d sin b → z_odot at d = 0
    zsun = ddr.ZSUN_PC * ddr._PC
    z_psr = zsun  # d = 0
    np.testing.assert_allclose(
        value(ddr.a_z_softened_sheet(z_psr)),
        value(ddr.a_z_softened_sheet(zsun)),
        atol=_LD("1e-30"),
    )


def test_geometry_off_roemer_is_xY():
    m = DDRmodel(
        **physical_params(
            DDRPK=False, DDRPBDOT="absorb_gw", PBDOT=0, GGAMMA=0, OMDOT=0, M2=0
        )
    )
    st = m.evaluate(_times(m))
    np.testing.assert_allclose(st.Delta_rom, st.x * st.Y, rtol=0, atol=_LD("1e-18"))


def test_projector_matches_3d_dot_product():
    """``Δ_rom = n_app · a(Xp + Yq)``. Does not validate inverse timing."""
    x, c = _LD("1.963"), _LD("0.5")
    s = np.sqrt((_LD(1) - c) * (_LD(1) + c))
    I, J = _LD("3e-4"), _LD("-1e-4")
    Omega = _LD("0.4")
    X, Y = _LD("0.3"), _LD("-0.7")
    a = x / s
    d_proj = value(roemer(x, Y, c, s, I, J, X))
    sO, cO = np.sin(Omega), np.cos(Omega)
    v_I, v_J = v_from_IJ(I, J, Omega)
    Z = np.sqrt(_LD(1) + I * I + J * J)
    n_app = np.array([value(v_I) / Z, value(v_J) / Z, _LD(1) / Z], dtype=np.longdouble)
    R = a * np.array(
        [X * cO - Y * c * sO, X * sO + Y * c * cO, Y * s],
        dtype=np.longdouble,
    )
    np.testing.assert_allclose(d_proj, np.dot(n_app, R), rtol=0, atol=_LD("1e-18"))


def test_face_on_fixed_a_delay_is_finite():
    """Hold ``a = x/s`` fixed as ``c → 1``; Roemer stays finite."""
    a = _LD(3)
    c = _LD("0.999")
    s = np.sqrt((_LD(1) - c) * (_LD(1) + c))
    m = DDRmodel(
        **physical_params(
            A1=a * s,
            COSI=c,
            DDRPK=False,
            DDRGEO=True,
            DDRKINE=False,
            DDRPBDOT="absorb_gw",
            PBDOT=0,
            GGAMMA=0,
            OMDOT=0,
            I=_LD("1e-4"),
            J=_LD("-2e-4"),
            KOM=_LD(30),
        )
    )
    t = _times(m, n=24)
    d = m.delay(t)
    assert np.all(np.isfinite(d))
    assert np.max(np.abs(d)) < _LD(10)


def test_geometry_on_uses_projector():
    m = DDRmodel(
        **physical_params(
            DDRGEO=True,
            DDRKINE=False,
            DDRPBDOT="absorb_gw",
            PBDOT=0,
            KOM=_LD(30),
            I=_LD("1e-6"),
            J=_LD("-2e-6"),
        )
    )
    st = m.evaluate(_times(m, n=16))
    np.testing.assert_allclose(
        st.Delta_rom,
        st.a * st.P,
        rtol=_LD("1e-12"),
        atol=_LD("1e-18"),
    )
    assert np.all(st.Z > 1)


def test_dd_residual_identity_geometry_off_matching_pk():
    """DD comparison is allowed only for geometry off, deformations off,
    matching PK, matching inverse-timing, constant-P_B polynomial.
    """
    import astropy.units as u

    from pint.models.stand_alone_psr_binaries.DD_model import DDmodel

    e = _LD("0.01")
    om = _LD(1.2)  # rad
    pb = _LD(2)
    tasc = _LD(53000)
    sini = _LD("0.6")
    c = np.sqrt(_LD(1) - sini * sini)
    a1 = _LD("2.0")
    m2 = _LD("0.2")
    h = e * np.sin(om)
    k = e * np.cos(om)
    ggamma = _LD("1e-6")  # seconds
    gamma = e * ggamma
    ddr = DDRmodel(
        PB=pb,
        A1=a1,
        TASC=tasc,
        EPS1=h,
        EPS2=k,
        M2=m2,
        COSI=c,
        DDRPK=False,
        DDRPBDOT="absorb_gw",
        DDRGEO=False,
        DDRKINE=False,
        PBDOT=0,
        GGAMMA=ggamma,
        OMDOT=0,
    )
    n = _LD(2) * np.pi / (pb * _DAY)
    t0 = tasc + om / n / _DAY  # T0 in MJD
    dd = DDmodel()
    t = tasc + as_ld(np.linspace(0.0, 3.0, 40)) * pb
    dd.update_input(
        barycentric_toa=t,
        PB=pb * u.d,
        T0=t0 * u.d,
        A1=a1 * u.lsec,
        ECC=e * u.Unit(""),
        OM=(om * 180 / np.pi) * u.deg,
        OMDOT=0 * u.deg / u.yr,
        PBDOT=0 * u.Unit(""),
        M2=m2 * u.Msun,
        SINI=sini * u.Unit(""),
        GAMMA=gamma * u.s,
        A1DOT=0 * u.lsec / u.s,
    )
    d_ddr = ddr.delay(t)
    d_dd = np.longdouble(dd.DDdelay().to_value(u.s))
    # Same inverse-timing convention; not raw Roemer+Shapiro.
    np.testing.assert_allclose(d_ddr, d_dd, rtol=0, atol=_LD("5e-14"))


def test_update_mu_I_uses_kernel_attribute_case():
    m = DDRmodel(**physical_params())
    m.update(
        MU_I=_LD("1e-15"), MU_J=_LD("-2e-15"), D_I_AU=_LD(1), D_J_AU=_LD(0.5), PX=_LD(1)
    )
    assert m.mu_I == _LD("1e-15")
    assert m.d_I_au == _LD(1)
    assert not hasattr(m, "mu_i")
    assert not hasattr(m, "d_i_au")


def test_geometry_mu_d_path_changes_delay():
    """Parallax term ϖ d_I is ~10 ns at PX=1 mas, d_I=1 AU vs I=J=0."""
    common = physical_params(
        DDRGEO=True,
        DDRKINE=False,
        DDRPBDOT="absorb_gw",
        PBDOT=0,
        KOM=_LD(30),
        I=_LD(0),
        J=_LD(0),
    )
    m0 = DDRmodel(**common)
    t = _times(m0, n=24)
    d0 = m0.delay(t)
    m1 = DDRmodel(**common)
    m1.update(MU_I=_LD(0), MU_J=_LD(0), D_I_AU=_LD(1), D_J_AU=_LD(0), PX=_LD(1))
    d1 = m1.delay(t)
    delta = np.max(np.abs(d1 - d0))
    assert _LD("1e-9") < delta < _LD("1e-6")


def test_kernel_reference_fixture():
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
    np.testing.assert_allclose(
        st.delay,
        np.array([_LD(x) for x in payload["delay_s"]]),
        rtol=0,
        atol=_LD("1e-18"),
    )
    np.testing.assert_allclose(
        st.Delta_rom,
        np.array([_LD(x) for x in payload["Delta_rom_s"]]),
        rtol=0,
        atol=_LD("1e-18"),
    )


def test_kernel_fbx_reference_fixture():
    payload = json.loads(
        (
            Path(__file__).resolve().parent / "data" / "ddr_reference_fixtures.json"
        ).read_text()
    )["fbx"]
    p = payload["params"]
    coeffs = np.asarray([_LD(p[f"FB{j}"]) for j in range(6)], dtype=np.longdouble)
    model = DDRmodel(
        FB_COEFFS=coeffs,
        A1=_LD(p["A1"]),
        TASC=_LD(p["TASC"]),
        EPS1=_LD(p["EPS1"]),
        EPS2=_LD(p["EPS2"]),
        M2=_LD(p["M2"]),
        COSI=_LD(p["COSI"]),
        GGAMMA=_LD(p["GGAMMA"]),
        OMDOT=_LD(p["OMDOT"]),
        DDRPK=p["DDRPK"],
        DDRPBDOT=p["DDRPBDOT"],
        DDRGEO=p["DDRGEO"],
        DDRKINE=p["DDRKINE"],
    )
    times = np.asarray([_LD(x) for x in payload["t_mjd"]], dtype=np.longdouble)
    state = model.evaluate(times)
    for name in ("delay", "Delta_rom", "Delta_E", "Delta_S", "d_inv"):
        expected = np.asarray([_LD(x) for x in payload[f"{name}_s"]])
        np.testing.assert_allclose(
            getattr(state, name), expected, rtol=0, atol=_LD("1e-18")
        )
    for name in ("FB0", "FB2", "FB5"):
        expected = np.asarray([_LD(x) for x in payload[f"d_delay_d_{name}"]])
        column = model.d_delay_d_par(name, times)
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            column,
            expected,
            rtol=_ddr_rtol(2e-18),
            atol=_ddr_atol(1e-18, scale),
        )


# --- Standalone Dual vs FD ---


def _assert_match(analytic, fd, *, rtol=_LD("1e-6"), atol=None, scale=_LD(1)):
    analytic = as_ld(analytic)
    fd = as_ld(fd)
    if atol is None:
        atol = _LD("1e-10") * (np.max(np.abs(fd)) + scale)
    np.testing.assert_allclose(analytic, fd, rtol=float(rtol), atol=float(atol))


def _central_fd(model, par, t, eps):
    orig = getattr(model, par)
    d0 = model.d_delay_d_par(par, t)
    model.update(**{par: orig + eps})
    dp = model.delay(t)
    model.update(**{par: orig - eps})
    dm = model.delay(t)
    model.update(**{par: orig})
    fd = (dp - dm) / (as_ld(2) * as_ld(eps))
    return d0, fd


def test_einstein_circular_origin_partials():
    lam = as_ld(np.linspace(0.1, 5.0, 30))
    g = _LD("1.2e-6")
    h = Dual(_LD(0), _LD(1))
    k = Dual(_LD(0), _LD(0))
    F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
    dE = g * s_e
    np.testing.assert_allclose(dE.d, -g * np.cos(lam), rtol=0, atol=_LD("2e-18"))
    h = Dual(_LD(0), _LD(0))
    k = Dual(_LD(0), _LD(1))
    F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
    dE = g * s_e
    np.testing.assert_allclose(dE.d, g * np.sin(lam), rtol=0, atol=_LD("2e-18"))


def test_geometric_roemer_cosi_partial_at_fixed_XY():
    x = _LD("1.963")
    c = _LD("0.4")
    s = np.sqrt((_LD(1) - c) * (_LD(1) + c))
    I, J = _LD("3e-6"), _LD("-1e-6")
    X, Y = _LD("0.2"), _LD("-0.7")
    boxed = geo.d_roemer_d_c_geometric(x, c, s, I, J, X, Y)
    c_d = Dual(c, _LD(1))
    s_d = geo.sini_from_cosi(c_d)
    rom = geo.roemer(x, Y, c_d, s_d, I, J, X)
    np.testing.assert_allclose(rom.d, boxed, rtol=0, atol=_LD("1e-18"))


def test_delay_derivatives_vs_finite_difference():
    m = DDRmodel(**physical_params(DDRPBDOT="absorb_gw", PBDOT=_LD("1e-12")))
    t = _times(m, n=24)
    x = float(m.A1)
    cases = [
        ("EPS1", _LD("1e-8"), x),
        ("EPS2", _LD("1e-8"), x),
        ("A1", _LD("1e-8"), 1.0),
        ("M2", _LD("1e-8"), x),
        ("COSI", _LD("1e-8"), x),
        ("TASC", _LD("1e-8"), x),
        ("PB", _LD("1e-10"), x),
        ("PBDOT", _LD("1e-16"), x),
        ("A1DOT", _LD("1e-16"), x),
    ]
    for par, eps, scale in cases:
        analytic, fd = _central_fd(m, par, t, eps)
        _assert_match(analytic, fd, rtol=_ddr_fd_rtol(5e-5), scale=_LD(scale))


def test_full_cosi_column_includes_shapiro_and_gr():
    m = DDRmodel(**physical_params(DDRPBDOT="absorb_gw", PBDOT=0))
    t = _times(m, n=20)
    st = m.evaluate(t)
    geom = st.d_rom_d_c_geometric
    full, fd = _central_fd(m, "COSI", t, _LD("1e-8"))
    _assert_match(full, fd, rtol=_LD("5e-5"), scale=_LD(m.A1))
    # Full column is not the geometric Roemer-only boxed partial.
    assert np.max(np.abs(full - geom)) > _LD("1e-8")


def test_geometry_on_cosi_geometric_vs_full():
    m = DDRmodel(
        **physical_params(
            DDRGEO=True,
            DDRKINE=False,
            DDRPBDOT="absorb_gw",
            PBDOT=0,
            KOM=_LD(40),
            I=_LD("5e-6"),
            J=_LD("2e-6"),
        )
    )
    t = _times(m, n=16)
    st = m.evaluate(t)
    full = m.d_delay_d_par("COSI", t)
    assert np.max(np.abs(full - st.d_rom_d_c_geometric)) > _LD("1e-8")
    analytic, fd = _central_fd(m, "COSI", t, _LD("1e-8"))
    _assert_match(analytic, fd, rtol=_LD("5e-5"), scale=_LD(m.A1))


def test_kom_derivative_geometry_on():
    m = DDRmodel(
        **physical_params(
            DDRGEO=True,
            DDRKINE=False,
            DDRPBDOT="absorb_gw",
            PBDOT=0,
            KOM=_LD(25),
            I=_LD("4e-6"),
            J=_LD("-3e-6"),
        )
    )
    t = _times(m, n=16)
    analytic, fd = _central_fd(m, "KOM", t, _LD("1e-6"))
    _assert_match(analytic, fd, rtol=_LD("5e-5"), scale=_LD(m.A1))


def test_d_mp_d_params_matches_dual():
    pb_s = _LD(2) * _LD(86400)
    n0 = _LD(2) * np.pi / pb_s
    x0 = _LD("1.963")
    mc0 = _LD("0.2")
    c0 = _LD("0.5")
    mp0, s0 = geo.pulsar_mass(n0, x0, mc0, c0)
    d_x, d_pb, d_mc, d_c = geo.d_mp_d_params(n0, x0, mc0, c0, mp0, s0, pb_s)
    mp_x, _ = geo.pulsar_mass(n0, Dual(x0, _LD(1)), mc0, c0)
    np.testing.assert_allclose(mp_x.d, d_x, rtol=0, atol=_ddr_atol(1e-18, d_x))
    mp_mc, _ = geo.pulsar_mass(n0, x0, Dual(mc0, _LD(1)), c0)
    np.testing.assert_allclose(mp_mc.d, d_mc, rtol=0, atol=_ddr_atol(1e-18, d_mc))
    mp_c, _ = geo.pulsar_mass(n0, x0, mc0, Dual(c0, _LD(1)))
    np.testing.assert_allclose(mp_c.d, d_c, rtol=0, atol=_ddr_atol(1e-18, d_c))
    n_pb = Dual(_LD(2) * np.pi, _LD(0)) / Dual(pb_s, _LD(1))
    mp_pb, _ = geo.pulsar_mass(n_pb, x0, mc0, c0)
    np.testing.assert_allclose(mp_pb.d, d_pb, rtol=0, atol=_ddr_atol(1e-18, d_pb))


def test_dual_sqrt_zero_value_stays_finite():
    z = sqrt(Dual(_LD(0), _LD(0)))
    assert np.isfinite(z.v)
    assert np.isfinite(z.d)
    assert z.v == 0
    assert z.d == 0


def test_tcorr_derivative_matches_fd():
    m = DDRmodel(
        **physical_params(DDRPBDOT="absorb_gw", PBDOT=0, DDRPK=False, GGAMMA=0, OMDOT=0)
    )
    t = _times(m, n=24)
    bt = m.d_delay_d_tcorr(t)
    eps = _LD("1e-8")
    fd = (m.delay(t + eps) - m.delay(t - eps)) / (as_ld(2) * eps * _LD(86400))
    _assert_match(bt, fd, rtol=_LD("1e-6"), scale=_LD("1e-6"))
