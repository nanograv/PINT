"""Stand-alone Damour-Deruelle-Regular binary delay kernel (van Haasteren in prep.).

Not a ``PSR_BINARY`` subclass: no ``OrbitPB``, polar ``(ECC, OM, T0)``, or
Quantity on the hot path. Dual + Kepler live in ``ddr_kepler`` (kernel-private).
The PINT wrap ``BinaryDDR`` owns ``-B_t A_θ`` composition, epoch change, and
par-file loading. Prefer ``update_input`` / ``binary_delay`` /
``d_binarydelay_d_par`` at the wrap boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import astropy.constants as const
import astropy.units as u
import numpy as np
from erfa import DAYSEC as SECS_PER_DAY

from pint import Tsun
from pint.exceptions import InvalidModelParameters

from . import ddr_kepler as kep
from .ddr_kepler import (
    _LD,
    Dual,
    as_ld,
    atan2,
    circular_origin_dXY_dhk,
    cos,
    dual,
    kepler_with_implicit,
    log,
    mean_longitude,
    orbital_phase,
    phase_slope_sample_times,
    polar_XY,
    reduce_longitude,
    sin,
    solve_F,
    solve_phase_offset,
    sqrt,
    static_XY,
    taylor_shift,
    value,
)

__all__ = [
    "circular_origin_dXY_dhk",
    "mean_longitude",
    "orbital_phase",
    "phase_slope_sample_times",
    "polar_XY",
    "reduce_longitude",
    "solve_F",
    "solve_phase_offset",
    "static_XY",
    "taylor_shift",
]

_DAY = _LD(SECS_PER_DAY)
_TWO_PI = _LD(2) * np.pi
_JUL_YEAR = _DAY * _LD("365.25")
_DEG2RAD = np.pi / _LD(180)
_TWO = _LD(2)
_HALF = _LD("0.5")

# Galactic defaults (van Haasteren in prep. §8.2)
R0_KPC = _LD("8.178")
THETA0_KM_S = _LD("220")
RHO0_MSUN_PC3 = _LD("0.10")
Z0_PC = _LD("180")
ZSUN_PC = _LD("20")

_TSUN = _LD(Tsun.to_value(u.s))
_C_M_S = _LD(const.c.si.value)
_G = _LD(const.G.si.value)
_MSUN = _LD(const.M_sun.si.value)
_PC = _LD((1 * u.pc).to_value(u.m))
_KPC = _LD(1000) * _PC


# ---------------------------------------------------------------------------
# Regular ν−M periastron advance (van Haasteren in prep. §7)
# ---------------------------------------------------------------------------


def q_nu_minus_M(c_e, s_e, E2):
    """Regular ``q = ν − M``, finite at the circular origin. Dual-safe."""
    c_e = dual(c_e)
    s_e = dual(s_e)
    E2 = dual(E2)
    one = _LD(1)
    den = one + sqrt(one - E2) - c_e
    return s_e + _TWO * atan2(s_e, den)


def q_at_tasc(h, k):
    """``q(τ)`` from the Kepler solve at λ = 0. Recompute every parameter evaluation."""
    h = dual(h)
    k = dual(k)
    _F, _D, c_e, s_e = kepler_with_implicit(_LD(0), h, k)
    E2 = h * h + k * k
    return q_nu_minus_M(c_e, s_e, E2)


def precession_delta(lam, q, q_star, kappa):
    """``δ = κ [λ + q(t) − q(τ)]``. ``lam`` is the unreduced secular longitude."""
    return dual(kappa) * (dual(lam) + dual(q) - dual(q_star))


def rotate_XY(X0, Y0, delta):
    """Rotate static projections at fixed radial phase u. Dual-safe."""
    X0 = dual(X0)
    Y0 = dual(Y0)
    delta = dual(delta)
    cd = cos(delta)
    sd = sin(delta)
    X = X0 * cd - Y0 * sd
    Y = Y0 * cd + X0 * sd
    return X, Y


def circular_origin_d_delta_dhk(lam, kappa):
    """Exact circular-origin ``∂δ/∂(h,k)`` with live ``q(τ)`` (van Haasteren in prep. §7.2)."""
    lam = as_ld(lam)
    kappa = as_ld(kappa)
    d_dh = _TWO * kappa * (_LD(1) - np.cos(lam))
    d_dk = _TWO * kappa * np.sin(lam)
    return d_dh, d_dk


# ---------------------------------------------------------------------------
# Cartesian viewing projector (van Haasteren in prep. §5)
# ---------------------------------------------------------------------------


def sini_from_cosi(c):
    c = dual(c)
    return sqrt((_LD(1) - c) * (_LD(1) + c))


def IJ_from_v(v_I, v_J, Omega):
    v_I = dual(v_I)
    v_J = dual(v_J)
    Omega = dual(Omega)
    sO = sin(Omega)
    cO = cos(Omega)
    I = -v_I * sO + v_J * cO
    J = v_I * cO + v_J * sO
    return I, J


def v_from_IJ(I, J, Omega):
    I = dual(I)
    J = dual(J)
    Omega = dual(Omega)
    sO = sin(Omega)
    cO = cos(Omega)
    v_I = -I * sO + J * cO
    v_J = I * cO + J * sO
    return v_I, v_J


def v_from_mu_parallax(mu_I, mu_J, px_rad, d_I_au, d_J_au, dt_K_s):
    """``v_I = μ_I Δt_K − ϖ d_I`` with μ in rad/s, d in AU, ϖ in radians."""
    mu_I = dual(mu_I)
    mu_J = dual(mu_J)
    px_rad = dual(px_rad)
    d_I_au = dual(d_I_au)
    d_J_au = dual(d_J_au)
    dt_K_s = dual(dt_K_s)
    v_I = mu_I * dt_K_s - px_rad * d_I_au
    v_J = mu_J * dt_K_s - px_rad * d_J_au
    return v_I, v_J


def projector(X, Y, c, s, I, J):
    """Return ``U, V, Z, P, c_app`` for the normalized tangent law. Dual-safe."""
    X = dual(X)
    Y = dual(Y)
    c = dual(c)
    s = dual(s)
    I = dual(I)
    J = dual(J)
    Z = sqrt(_LD(1) + I * I + J * J)
    U = (s + c * I) / Z
    V = J / Z
    c_app = (c - s * I) / Z
    P = U * Y + V * X
    return U, V, Z, P, c_app


def roemer(x, Y, c, s, I, J, X):
    """``Δ_rom = (x Y + a c I Y + a J X) / Z`` with ``a = x/s``."""
    x = dual(x)
    Y = dual(Y)
    c = dual(c)
    s = dual(s)
    I = dual(I)
    J = dual(J)
    X = dual(X)
    Z = sqrt(_LD(1) + I * I + J * J)
    a = x / s
    return (x * Y + a * c * I * Y + a * J * X) / Z


def d_roemer_d_c_geometric(x, c, s, I, J, X, Y):
    """Geometric Roemer partial at fixed ``X, Y`` (van Haasteren in prep. §5.3 boxed)."""
    x = as_ld(x)
    c = as_ld(c)
    s = as_ld(s)
    I = as_ld(I)
    J = as_ld(J)
    X = as_ld(X)
    Y = as_ld(Y)
    a = x / s
    Z = np.sqrt(_LD(1) + I * I + J * J)
    return a / (s * s * Z) * (I * Y + c * J * X)


def shapiro_B_S_diff(c_e, P):
    """``B_S = 1 − c_e − P``."""
    return _LD(1) - dual(c_e) - dual(P)


def shapiro_B_S_squared_norm(c_e, X, Y, c, s, I, J, Omega):
    """``B_S = ρ/2 ||n_app − R̂||²``. Dual-safe."""
    c_e = dual(c_e)
    rho = _LD(1) - c_e
    Omega = dual(Omega)
    sO = sin(Omega)
    cO = cos(Omega)
    c = dual(c)
    s = dual(s)
    X = dual(X)
    Y = dual(Y)
    Rx = (X * cO - Y * c * sO) / rho
    Ry = (X * sO + Y * c * cO) / rho
    Rz = (Y * s) / rho
    I = dual(I)
    J = dual(J)
    Z = sqrt(_LD(1) + I * I + J * J)
    v_I, v_J = v_from_IJ(I, J, Omega)
    nax = v_I / Z
    nay = v_J / Z
    naz = _LD(1) / Z
    nrm2 = (nax - Rx) ** 2 + (nay - Ry) ** 2 + (naz - Rz) ** 2
    return _HALF * rho * nrm2


def shapiro_delay(r, B_S):
    """``Δ_S = −2 r log B_S``. Raises if primal ``B_S ≤ 0``."""
    r = dual(r)
    B_S = dual(B_S)
    if np.any(value(B_S) <= 0):
        raise InvalidModelParameters("DDR Shapiro argument B_S is not positive")
    return -_LD(2) * r * log(B_S)


# ---------------------------------------------------------------------------
# Mass-function inversion, GR maps, kinematic Ṗ_b (van Haasteren in prep. §§6, 8)
# ---------------------------------------------------------------------------


def tsun_s():
    return _TSUN


def mass_function_f(n, x_star):
    """``f = n² x_★³ / T_⊙``."""
    n = dual(n)
    x_star = dual(x_star)
    return n * n * x_star**3 / _TSUN


def pulsar_mass(n, x_star, m_c, c):
    """Leading-order mass-function inversion at reference ``(x_★, P_B)``."""
    s = sini_from_cosi(c)
    f = mass_function_f(n, x_star)
    if np.any(value(f) <= 0) or np.any(value(s) <= 0):
        raise InvalidModelParameters(
            "DDR mass-function map requires x>0, m_c>0, s>0, f>0"
        )
    if np.any(value(x_star) <= 0) or np.any(value(m_c) <= 0):
        raise InvalidModelParameters(
            "DDR mass-function map requires x>0, m_c>0, s>0, f>0"
        )
    m_c = dual(m_c)
    m_p = (m_c * s) ** (_LD("1.5")) / sqrt(f) - m_c
    if np.any(value(m_p) <= 0):
        raise InvalidModelParameters("DDR inferred pulsar mass is not positive")
    return m_p, s


def d_mp_d_params(n, x_star, m_c, c, m_p, s, pb_s):
    """Signed mass-function derivatives (van Haasteren in prep. §6). Primal values."""
    m_p = as_ld(m_p)
    m_c = as_ld(m_c)
    x_star = as_ld(x_star)
    s = as_ld(s)
    c = as_ld(c)
    pb_s = as_ld(pb_s)
    M = m_p + m_c
    d_x = -_LD("1.5") * M / x_star
    d_pb = M / pb_s
    d_mc = _LD("1.5") * M / m_c - _LD(1)
    d_c = -_LD("1.5") * c * M / (s * s)
    return d_x, d_pb, d_mc, d_c


def g_gamma_gr(pb_s, m_p, m_c):
    """Regular Einstein coefficient (γ/e), finite at e = 0. Dual-safe."""
    pb_s = dual(pb_s)
    m_p = dual(m_p)
    m_c = dual(m_c)
    M = m_p + m_c
    return (
        _TSUN ** (_LD(2) / _LD(3))
        * (pb_s / (_LD(2) * np.pi)) ** (_LD(1) / _LD(3))
        * m_c
        * (m_p + _LD(2) * m_c)
        / M ** (_LD(4) / _LD(3))
    )


def kappa_gr(x_star, m_c, s, e2):
    """``κ = 3 T_⊙ m_c s / (x_★ (1−e²))``. Independent of P_B at fixed x, m_c, c, h, k."""
    x_star = dual(x_star)
    m_c = dual(m_c)
    s = dual(s)
    e2 = dual(e2)
    return _LD(3) * _TSUN * m_c * s / (x_star * (_LD(1) - e2))


def pbdot_gw(pb_s, m_p, m_c, e2):
    """Dimensionless GR quadrupole Ṗ_b. Dual-safe."""
    pb_s = dual(pb_s)
    m_p = dual(m_p)
    m_c = dual(m_c)
    e2 = dual(e2)
    e4 = e2 * e2
    fe = (_LD(1) + _LD("73") / _LD("24") * e2 + _LD("37") / _LD("96") * e4) / (
        (_LD(1) - e2) ** (_LD("3.5"))
    )
    return (
        -_LD("192")
        * np.pi
        / _LD(5)
        * _TSUN ** (_LD(5) / _LD(3))
        * (pb_s / (_LD(2) * np.pi)) ** (-_LD(5) / _LD(3))
        * fe
        * m_p
        * m_c
        / (m_p + m_c) ** (_LD(1) / _LD(3))
    )


def pbdot_shklovskii(pb_s, mu_rad_s, distance_m):
    """Dimensionless Shklovskii Ṗ_b = P_B μ² d / c."""
    pb_s = dual(pb_s)
    mu = dual(mu_rad_s)
    d = dual(distance_m)
    if np.any(value(d) <= 0):
        raise InvalidModelParameters("DDR Shklovskii term requires positive distance")
    return pb_s * mu * mu * d / _C_M_S


def a_z_softened_sheet(z_m, rho0_msun_pc3=RHO0_MSUN_PC3, z0_pc=Z0_PC):
    """Vertical Galactic acceleration (softened sheet), m s⁻². Dual-safe."""
    rho0 = as_ld(rho0_msun_pc3) * _MSUN / (_PC**3)
    z0 = as_ld(z0_pc) * _PC
    z = dual(z_m)
    return -_LD(4) * np.pi * _G * rho0 * z0 * z / sqrt(z * z + z0 * z0)


def galactic_acceleration_los(
    distance_m,
    l_rad,
    b_rad,
    r0_kpc=R0_KPC,
    theta0_km_s=THETA0_KM_S,
    rho0_msun_pc3=RHO0_MSUN_PC3,
    z0_pc=Z0_PC,
    zsun_pc=ZSUN_PC,
):
    """Return ``(a_pl, a_vert, a_los)`` in m s⁻². Dual-safe in ``distance_m``."""
    d = dual(distance_m)
    l = dual(l_rad)
    b = dual(b_rad)
    R0 = as_ld(r0_kpc) * _KPC
    Theta0 = as_ld(theta0_km_s) * _LD(1000)
    cb = cos(b)
    sb = sin(b)
    cl = cos(l)
    sl = sin(l)
    beta = (d / R0) * cb - cl
    denom = sl * sl + beta * beta
    a_pl = -cb * (Theta0 * Theta0 / R0) * (cl + beta / denom)
    zsun = as_ld(zsun_pc) * _PC
    z_psr = zsun + d * sb
    a_vert = (
        a_z_softened_sheet(z_psr, rho0_msun_pc3, z0_pc)
        - a_z_softened_sheet(zsun, rho0_msun_pc3, z0_pc)
    ) * sb
    a_los = a_pl + a_vert
    return a_pl, a_vert, a_los


def pbdot_galactic(
    pb_s,
    distance_m,
    l_rad,
    b_rad,
    r0_kpc=R0_KPC,
    theta0_km_s=THETA0_KM_S,
    rho0_msun_pc3=RHO0_MSUN_PC3,
    z0_pc=Z0_PC,
    zsun_pc=ZSUN_PC,
):
    """Dimensionless Galactic Ṗ_b: planar Nice-Taylor + vertical softened sheet."""
    pb_s = dual(pb_s)
    _a_pl, _a_vert, a_los = galactic_acceleration_los(
        distance_m,
        l_rad,
        b_rad,
        r0_kpc=r0_kpc,
        theta0_km_s=theta0_km_s,
        rho0_msun_pc3=rho0_msun_pc3,
        z0_pc=z0_pc,
        zsun_pc=zsun_pc,
    )
    return pb_s / _C_M_S * a_los


def inverse_timing(d, dprime, dpp, n, c_e, s_e):
    """PINT DD inverse-timing polynomial with DDR intermediates (van Haasteren in prep. §9)."""
    one = _LD(1)
    nhat = dual(n) / (one - dual(c_e))
    d = dual(d)
    dprime = dual(dprime)
    dpp = dual(dpp)
    s_e = dual(s_e)
    c_e = dual(c_e)
    half = _LD("0.5")
    return d * (
        one
        - nhat * dprime
        + (nhat * dprime) ** 2
        + half * nhat**2 * d * dpp
        - half * s_e / (one - c_e) * nhat**2 * d * dprime
    )


def _as_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip().upper()
        if v in ("Y", "YES", "TRUE", "1"):
            return True
        if v in ("N", "NO", "FALSE", "0"):
            return False
        raise InvalidModelParameters(f"Unrecognized boolean flag {val!r}")
    return bool(val)


def _pbdot_mode(val, default="kinematic"):
    if val is None:
        return default
    v = str(val).strip().lower()
    if v not in ("kinematic", "absorb_gw"):
        raise InvalidModelParameters("DDRPBDOT must be 'kinematic' or 'absorb_gw'")
    return v


def _qty_value(x, unit, *, default=None):
    if x is None:
        return default
    if hasattr(x, "to"):
        return _LD(x.to_value(unit))
    return as_ld(x)


class DDRmodel:
    """Fixed-reference DDR delay at barycentric time.

    Parameters are PINT-like at the boundary (``PB`` in days, ``KOM`` in
    degrees, ``OMDOT`` in deg/yr) and converted to seconds/radians inside.
    The wrap supplies the TGEO astrometric triad as ``I``, ``J``, proper
    motions, and AU-projected observer coordinates; this kernel does not
    read SkyCoord.
    """

    binary_name = "DDR"

    def __init__(self, t=None, **params):
        self.t = None
        self._p_override = None
        self.set_defaults()
        if t is not None:
            self.t = np.atleast_1d(as_ld(t))
        if params:
            self.update(**params)

    def set_defaults(self):
        self.PB = _LD(1)  # days
        self.FB_COEFFS = None  # total anomalistic frequency Taylor coefficients
        self.A1 = _LD(0)  # light-seconds
        self.TASC = _LD(54000)
        self.EPS1 = _LD(0)
        self.EPS2 = _LD(0)
        self.M2 = _LD(0)
        self.COSI = _LD(0)
        self.KOM = _LD(0)  # deg
        self.TGEO = None
        self.A1DOT = _LD(0)  # lsec / s
        self.GGAMMA = _LD(0)  # s
        self.OMDOT = _LD(0)  # deg / yr
        self.PBDOT = _LD(0)
        self.XPBDOT = _LD(0)
        self.DDRPK = True
        self.DDRPBDOT = "kinematic"
        self.DDRGEO = False
        self.DDRKINE = False
        self.I = _LD(0)
        self.J = _LD(0)
        self.mu_I = None  # rad / s
        self.mu_J = None  # rad / s
        self.d_I_au = None
        self.d_J_au = None
        self.PX = None  # mas
        self.PM = None  # mas / yr (total)
        self.PM_SQUARED = None  # (mas / yr)^2, analytic at zero PM
        self.distance_m = None
        self.l_rad = None
        self.b_rad = None
        self.PRIMITIVE_DERIVS = {}
        self.DDRR0 = R0_KPC
        self.DDRTHETA0 = THETA0_KM_S
        self.DDRRHO0 = RHO0_MSUN_PC3
        self.DDRZ0 = Z0_PC
        self.DDRZSUN = ZSUN_PC

    def update(self, **params):
        if "barycentric_toa" in params:
            self.t = np.atleast_1d(as_ld(params.pop("barycentric_toa")))
        if "t" in params:
            self.t = np.atleast_1d(as_ld(params.pop("t")))
        if "p" in params:
            self._p_override = params.pop("p")
        for key, val in params.items():
            name = key.upper()
            if name in ("DDRPK", "DDRGEO", "DDRKINE"):
                setattr(self, name, _as_bool(val))
            elif name == "DDRPBDOT":
                self.DDRPBDOT = _pbdot_mode(val)
            elif name == "PB":
                self.PB = _qty_value(val, u.d)
            elif name == "FB_COEFFS":
                self.FB_COEFFS = np.asarray(val, dtype=_LD)
            elif name == "PRIMITIVE_DERIVS":
                self.PRIMITIVE_DERIVS = val
            elif name == "A1":
                self.A1 = _qty_value(val, u.lsec)
            elif name == "TASC":
                self.TASC = _qty_value(val, u.d)
            elif name == "TGEO":
                self.TGEO = _qty_value(val, u.d)
            elif name in ("EPS1", "EPS2", "COSI", "PBDOT", "XPBDOT"):
                setattr(self, name, as_ld(val.value if hasattr(val, "value") else val))
            elif name == "M2":
                self.M2 = _qty_value(val, u.Msun)
            elif name == "KOM":
                self.KOM = _qty_value(val, u.deg) if hasattr(val, "to") else as_ld(val)
            elif name == "A1DOT":
                self.A1DOT = (
                    _qty_value(val, u.lsec / u.s) if hasattr(val, "to") else as_ld(val)
                )
            elif name == "GGAMMA":
                self.GGAMMA = _qty_value(val, u.s) if hasattr(val, "to") else as_ld(val)
            elif name == "OMDOT":
                self.OMDOT = (
                    _qty_value(val, u.deg / u.yr) if hasattr(val, "to") else as_ld(val)
                )
            elif name in ("I", "J"):
                setattr(self, name, as_ld(val))
            elif name == "MU_I":
                self.mu_I = as_ld(val)
            elif name == "MU_J":
                self.mu_J = as_ld(val)
            elif name == "D_I_AU":
                self.d_I_au = as_ld(val)
            elif name == "D_J_AU":
                self.d_J_au = as_ld(val)
            elif name == "PX":
                self.PX = _qty_value(val, u.mas) if hasattr(val, "to") else as_ld(val)
            elif name == "PM":
                self.PM = (
                    _qty_value(val, u.mas / u.yr) if hasattr(val, "to") else as_ld(val)
                )
            elif name == "PM_SQUARED":
                self.PM_SQUARED = as_ld(val)
            elif name == "DISTANCE_M":
                self.distance_m = as_ld(val)
            elif name in ("L_RAD", "B_RAD"):
                setattr(self, name.lower(), as_ld(val))
            elif name in ("DDRR0", "DDRTHETA0", "DDRRHO0", "DDRZ0", "DDRZSUN"):
                setattr(self, name, as_ld(val))
            else:
                raise AttributeError(f"Unknown DDR parameter {key}")
        if self.TGEO is None:
            self.TGEO = self.TASC

    def _seed(self, name, tangent_par):
        if tangent_par is None:
            return _LD(0)
        if isinstance(tangent_par, str):
            return _LD(1) if tangent_par.upper() == name else _LD(0)
        names = [par.upper() for par in tangent_par]
        seed = np.zeros((len(names), 1), dtype=_LD)
        if name in names:
            seed[names.index(name), 0] = 1
        return seed

    def _primitive(self, name, value_, tangent_par):
        derivative = self._seed(name, tangent_par)
        if isinstance(tangent_par, str):
            derivative += self.PRIMITIVE_DERIVS.get(tangent_par.upper(), {}).get(
                name.lower(), _LD(0)
            )
        elif tangent_par is not None:
            extras = [
                self.PRIMITIVE_DERIVS.get(par.upper(), {}).get(name.lower(), _LD(0))
                for par in tangent_par
            ]
            shapes = [np.shape(extra) for extra in extras]
            target = np.broadcast_shapes(*shapes) if shapes else ()
            if target == ():
                target = (1,)
            derivative = derivative + np.stack(
                [np.broadcast_to(extra, target) for extra in extras]
            )
        return Dual(value_, derivative)

    def _pb_s(self, tangent_par):
        if self.FB_COEFFS is not None:
            return _LD(1) / Dual(self.FB_COEFFS[0], self._seed("FB0", tangent_par))
        return Dual(self.PB * _DAY, self._seed("PB", tangent_par) * _DAY)

    def _phase_coeffs(self, tangent_par):
        if self.FB_COEFFS is None:
            return None
        return [
            Dual(value, self._seed(f"FB{j}", tangent_par))
            for j, value in enumerate(self.FB_COEFFS)
        ]

    def _dt_s(self, t, tangent_par):
        t = as_ld(t)
        d_tasc = self._seed("TASC", tangent_par)
        d_t = self._seed("TCORR", tangent_par)
        return Dual((t - self.TASC) * _DAY, (d_t - d_tasc) * _DAY)

    def _fbx_orbital_phase(self, dt_s, phase_coeffs, t, tangent_par):
        """Evaluate the FBX phase with a computational origin, not a TASC move.

        Shifts the Taylor series to the span midpoint and adds back
        ``λ(τ_comp)`` so ``λ(TASC)=0``. Stored ``TASC``, ``A1``, ``(h,k)``,
        ``TGEO``, and the physical ``FB0`` used by mass maps stay put.
        """
        t_primal = as_ld(t)
        t_comp = _LD(0.5) * (np.min(t_primal) + np.max(t_primal))
        tasc = Dual(self.TASC, self._seed("TASC", tangent_par))
        delta_s = (t_comp - tasc) * _DAY
        if np.all(np.asarray(value(delta_s)) == 0):
            return kep.orbital_phase(dt_s, phase_coeffs)
        shifted = kep.taylor_shift(phase_coeffs, delta_s)
        dt_comp = Dual(
            (t_primal - t_comp) * _DAY,
            self._seed("TCORR", tangent_par) * _DAY,
        )
        lam_comp, lamdot, lamddot = kep.orbital_phase(
            dt_comp, shifted, check_slope=False
        )
        lam_origin, _, _ = kep.orbital_phase(delta_s, phase_coeffs, check_slope=False)
        lam = lam_comp + lam_origin
        if np.any(value(lamdot) <= 0):
            raise InvalidModelParameters(
                "DDR orbital phase slope is not positive on the evaluation span"
            )
        return lam, lamdot, lamddot

    def _dt_K_s(self, t, tangent_par):
        t = as_ld(t)
        tgeo = self.TASC if self.TGEO is None else self.TGEO
        d_t = self._seed("TCORR", tangent_par)
        return Dual((t - tgeo) * _DAY, d_t * _DAY)

    def _px_rad(self, tangent_par):
        if self.PX is None:
            return Dual(_LD(0), _LD(0))
        mas_to_rad = _LD("1e-3") * np.pi / (_LD(180) * _LD(3600))
        return Dual(
            as_ld(self.PX) * mas_to_rad,
            self._seed("PX", tangent_par) * mas_to_rad,
        )

    def _geometry_IJ(self, t, Omega, tangent_par):
        """Apparent ``(I, J)``. Dual-safe in ``t`` through ``Δt_K`` when μ, ϖ, d are set."""
        if self.mu_I is not None and self.d_I_au is not None:
            dt_K = self._dt_K_s(t, tangent_par)
            v_I, v_J = v_from_mu_parallax(
                self._primitive("mu_i", as_ld(self.mu_I), tangent_par),
                self._primitive("mu_j", as_ld(self.mu_J), tangent_par),
                self._px_rad(tangent_par),
                self._primitive("d_i_au", as_ld(self.d_I_au), tangent_par),
                self._primitive("d_j_au", as_ld(self.d_J_au), tangent_par),
                dt_K,
            )
            return IJ_from_v(v_I, v_J, Omega)
        return (
            Dual(as_ld(self.I), self._seed("I", tangent_par)),
            Dual(as_ld(self.J), self._seed("J", tangent_par)),
        )

    def _need_mass(self):
        if self.DDRPK:
            return True
        return self.DDRPBDOT == "kinematic"

    def _compose_p(self, pb_s, m_p, m_c, e2, tangent_par):
        if self._p_override is not None:
            return dual(self._p_override, self._seed("P", tangent_par))
        p = Dual(_LD(0), _LD(0))
        p_shk = Dual(_LD(0), _LD(0))
        p_gal = Dual(_LD(0), _LD(0))
        p_gw = Dual(_LD(0), _LD(0))
        if self.DDRPBDOT == "kinematic":
            p_gw = pbdot_gw(pb_s, m_p, m_c, e2)
            p = p + p_gw + Dual(self.XPBDOT, self._seed("XPBDOT", tangent_par))
        else:
            p = p + Dual(self.PBDOT, self._seed("PBDOT", tangent_par))
        if self.DDRKINE:
            dist = self._distance_m(tangent_par)
            if self.PM_SQUARED is None:
                mu = self._mu_rad_s(tangent_par)
                p_shk = pbdot_shklovskii(pb_s, mu, dist)
            else:
                mas_yr_to_rad_s = (
                    _LD("1e-3") * np.pi / (_LD(180) * _LD(3600)) / _JUL_YEAR
                )
                mu_squared = (
                    self._primitive("pm_squared", self.PM_SQUARED, tangent_par)
                    * mas_yr_to_rad_s**2
                )
                p_shk = pb_s * mu_squared * dist / _C_M_S
            if self.l_rad is None or self.b_rad is None:
                raise InvalidModelParameters("DDRKINE Y requires l_rad and b_rad")
            p_gal = pbdot_galactic(
                pb_s,
                dist,
                self._primitive("l_rad", self.l_rad, tangent_par),
                self._primitive("b_rad", self.b_rad, tangent_par),
                r0_kpc=self.DDRR0,
                theta0_km_s=self.DDRTHETA0,
                rho0_msun_pc3=self.DDRRHO0,
                z0_pc=self.DDRZ0,
                zsun_pc=self.DDRZSUN,
            )
            p = p + p_shk + p_gal
        return p, p_shk, p_gal, p_gw

    def _distance_m(self, tangent_par=None):
        if self.distance_m is not None:
            return dual(self.distance_m)
        if self.PX is None or value(dual(self.PX)) <= 0:
            raise InvalidModelParameters("DDR kinematics require PX>0 or distance_m")
        px_rad = self._px_rad(tangent_par)
        au_m = _LD((1 * u.au).to_value(u.m))
        return au_m / px_rad

    def _mu_rad_s(self, tangent_par=None):
        if self.PM is None:
            raise InvalidModelParameters("DDR Shklovskii term requires PM")
        mas_yr = self._primitive("pm", as_ld(self.PM), tangent_par)
        rad_yr = mas_yr * (_LD("1e-3") * np.pi / (_LD(180) * _LD(3600)))
        return rad_yr / _JUL_YEAR

    def evaluate(self, t=None, tangent_par=None):
        """Evaluate delay pieces. ``tangent_par`` seeds a Dual column."""
        if t is None:
            t = self.t
        if t is None:
            raise ValueError("No evaluation times: pass t or set barycentric_toa")
        t = np.atleast_1d(as_ld(t))
        tp = (
            None
            if tangent_par is None
            else (
                tangent_par.upper()
                if isinstance(tangent_par, str)
                else tuple(par.upper() for par in tangent_par)
            )
        )
        primitives = [
            t,
            self.A1,
            self.TASC,
            self.EPS1,
            self.EPS2,
            self.M2,
            self.COSI,
            self.KOM,
            self.A1DOT,
            self.GGAMMA,
            self.OMDOT,
            self.PBDOT,
            self.XPBDOT,
        ]
        if self.FB_COEFFS is None:
            primitives.append(self.PB)
        else:
            primitives.append(self.FB_COEFFS)
        primitives.extend(
            value_
            for value_ in (
                self.TGEO,
                self.PX,
                self.PM,
                self.PM_SQUARED,
                self.distance_m,
                self.l_rad,
                self.b_rad,
                self.mu_I,
                self.mu_J,
                self.d_I_au,
                self.d_J_au,
            )
            if value_ is not None
        )
        if any(np.any(~np.isfinite(as_ld(item))) for item in primitives):
            raise InvalidModelParameters("DDR primitive parameters must be finite")
        if self.FB_COEFFS is not None and (
            self.DDRPBDOT != "absorb_gw" or self.DDRKINE
        ):
            raise InvalidModelParameters(
                "DDR FBX chart requires DDRPBDOT absorb_gw and DDRKINE N"
            )

        pb_s = self._pb_s(tp)
        if np.any(value(pb_s) <= 0):
            raise InvalidModelParameters("DDR PB must be positive")
        dt_s = self._dt_s(t, tp)
        h = Dual(self.EPS1, self._seed("EPS1", tp))
        k = Dual(self.EPS2, self._seed("EPS2", tp))
        c = Dual(self.COSI, self._seed("COSI", tp))
        if np.any(np.abs(value(c)) >= 1):
            raise InvalidModelParameters("DDR |COSI| must be < 1")
        m_c = Dual(self.M2, self._seed("M2", tp))
        if np.any(value(m_c) < 0):
            raise InvalidModelParameters("DDR M2 must be nonnegative")
        x_star = Dual(self.A1, self._seed("A1", tp))
        a1dot = Dual(self.A1DOT, self._seed("A1DOT", tp))
        x = x_star + a1dot * dt_s
        e2 = h * h + k * k
        if np.any(value(e2) > np.longdouble("0.99") ** 2):
            raise InvalidModelParameters("DDR eccentricity must satisfy e <= 0.99")
        n = _TWO_PI / pb_s
        s = sini_from_cosi(c)

        m_p = Dual(_LD(0), _LD(0))
        if self._need_mass():
            if np.any(value(x_star) <= 0):
                raise InvalidModelParameters(
                    "DDR x_star must be positive when mass maps are active"
                )
            m_p, s = pulsar_mass(n, x_star, m_c, c)
            if np.any(value(x) <= 0):
                raise InvalidModelParameters(
                    "DDR x(t) must be positive when mass maps are active"
                )
        else:
            if np.any(value(x_star) < 0):
                raise InvalidModelParameters("DDR x_star must be nonnegative")
            if np.any(value(x) < 0):
                raise InvalidModelParameters("DDR x(t) is negative")

        if self.DDRPK:
            g_gamma = g_gamma_gr(pb_s, m_p, m_c)
            kappa = kappa_gr(x_star, m_c, s, e2)
        else:
            g_gamma = Dual(self.GGAMMA, self._seed("GGAMMA", tp))
            omdot_rad_s = Dual(
                self.OMDOT * _DEG2RAD / _JUL_YEAR,
                self._seed("OMDOT", tp) * _DEG2RAD / _JUL_YEAR,
            )
            kappa = omdot_rad_s / n

        phase_coeffs = self._phase_coeffs(tp)
        if phase_coeffs is not None:
            p = (
                -phase_coeffs[1] / (phase_coeffs[0] * phase_coeffs[0])
                if len(phase_coeffs) > 1
                else Dual(_LD(0), _LD(0))
            )
            p_shk = p_gal = p_gw = Dual(_LD(0), _LD(0))
        else:
            p_pack = self._compose_p(pb_s, m_p, m_c, e2, tp)
            if self._p_override is not None:
                p = p_pack
                p_shk = p_gal = p_gw = Dual(_LD(0), _LD(0))
            else:
                p, p_shk, p_gal, p_gw = p_pack

        if phase_coeffs is None:
            lam, lamdot, lamddot = kep.orbital_phase(
                dt_s, [1 / pb_s, -p / (pb_s * pb_s)]
            )
        else:
            lam, lamdot, lamddot = self._fbx_orbital_phase(dt_s, phase_coeffs, t, tp)
        lam_red, n_orb = kep.reduce_longitude(lam)
        F, D, c_e, s_e = kep.kepler_with_implicit(lam, h, k)
        X0, Y0, X0p, Y0p, X0pp, Y0pp = kep.static_XY(F, h, k)

        q = q_nu_minus_M(c_e, s_e, e2)
        q_star = q_at_tasc(h, k)
        delta = precession_delta(lam, q, q_star, kappa)
        X, Y = rotate_XY(X0, Y0, delta)
        Xp, Yp = rotate_XY(X0p, Y0p, delta)
        Xpp, Ypp = rotate_XY(X0pp, Y0pp, delta)

        Omega = Dual(self.KOM * _DEG2RAD, self._seed("KOM", tp) * _DEG2RAD)
        if self.DDRGEO:
            I, J = self._geometry_IJ(t, Omega, tp)
        else:
            I = Dual(_LD(0), _LD(0))
            J = Dual(_LD(0), _LD(0))

        if self.DDRGEO:
            U, V, Z, P, c_app = projector(X, Y, c, s, I, J)
            Delta_rom = roemer(x, Y, c, s, I, J, X)
            a = x / s
            Delta_romp = (x * Yp + a * c * I * Yp + a * J * Xp) / Z
            Delta_rompp = (x * Ypp + a * c * I * Ypp + a * J * Xpp) / Z
        else:
            # Geometry off: Δ_rom = x Y. Do not form a = x/s.
            U = s
            V = Dual(_LD(0), _LD(0))
            Z = Dual(_LD(1), _LD(0))
            P = s * Y
            c_app = c
            a = Dual(_LD(0), _LD(0))
            Delta_rom = x * Y
            Delta_romp = x * Yp
            Delta_rompp = x * Ypp

        Delta_E = g_gamma * s_e
        d = Delta_rom + Delta_E
        dprime = Delta_romp + g_gamma * c_e
        dpp = Delta_rompp - g_gamma * s_e
        d_inv = inverse_timing(d, dprime, dpp, n, c_e, s_e)
        nhat = n / (_LD(1) - c_e)
        Delta_Dop = nhat * dprime

        B_diff = shapiro_B_S_diff(c_e, P)
        B_S = shapiro_B_S_squared_norm(c_e, X, Y, c, s, I, J, Omega)
        r = tsun_s() * m_c
        Delta_S = shapiro_delay(r, B_S)
        delay = d_inv + Delta_S

        d_rom_dc_geom = d_roemer_d_c_geometric(
            value(x), value(c), value(s), value(I), value(J), value(X), value(Y)
        )

        return SimpleNamespace(
            t=t,
            delay=value(delay),
            d_delay=delay.d if isinstance(delay, Dual) else np.zeros_like(value(delay)),
            dt_s=value(dt_s),
            pb_s=value(pb_s),
            p=value(p),
            p_shk=value(p_shk),
            p_gal=value(p_gal),
            p_gw=value(p_gw),
            lamdot=value(lamdot),
            lamddot=value(lamddot),
            lam=value(lam),
            d_lam=lam.d if isinstance(lam, Dual) else np.zeros_like(value(lam)),
            lam_red=value(lam_red),
            N_orb=n_orb,
            F=value(F),
            D=value(D),
            c_e=value(c_e),
            s_e=value(s_e),
            X0=value(X0),
            Y0=value(Y0),
            X=value(X),
            Y=value(Y),
            Xp=value(Xp),
            Yp=value(Yp),
            Xpp=value(Xpp),
            Ypp=value(Ypp),
            delta=value(delta),
            d_delta=(
                delta.d if isinstance(delta, Dual) else np.zeros_like(value(delta))
            ),
            q=value(q),
            q_star=value(q_star),
            kappa=value(kappa),
            x=value(x),
            x_star=value(x_star),
            a=value(a),
            s=value(s),
            c=value(c),
            I=value(I),
            J=value(J),
            Z=value(Z),
            U=value(U),
            V=value(V),
            P=value(P),
            c_app=value(c_app),
            g_gamma=value(g_gamma),
            Delta_E=value(Delta_E),
            Delta_rom=value(Delta_rom),
            d=value(d),
            dprime=value(dprime),
            dpp=value(dpp),
            nhat=value(nhat),
            d_inv=value(d_inv),
            B_S=value(B_S),
            B_S_diff=value(B_diff),
            Delta_S=value(Delta_S),
            Delta_Dop=value(Delta_Dop),
            m_p=value(m_p),
            m_c=value(m_c),
            n=value(n),
            inverse_timing_ref_error_s=_HALF
            * value(x) ** 2
            * np.abs(value(lamdot) - value(n)),
            d_rom_d_c_geometric=d_rom_dc_geom,
            einstein_gr=bool(self.DDRPK),
            precession_gr=bool(self.DDRPK),
            quadrupole=self.DDRPBDOT == "kinematic",
        )

    def delay(self, t=None):
        """Binary delay in seconds (``np.longdouble``)."""
        return self.evaluate(t).delay

    def update_input(self, **params):
        """``PSR_BINARY``-shaped alias for :meth:`update`."""
        return self.update(**params)

    def d_delay_d_par(self, par, t=None):
        """Standalone ``∂Δ_B/∂par`` at fixed barycentric time."""
        return self.evaluate(t, tangent_par=par).d_delay

    def d_delay_d_pars(self, pars, t=None):
        """Batched standalone derivatives from one primal evaluation."""
        pars = tuple(pars)
        if not pars:
            times = self.t if t is None else t
            return np.empty((0, len(np.atleast_1d(times))), dtype=_LD)
        return self.evaluate(t, tangent_par=pars).d_delay

    def d_delay_d_tcorr(self, t=None):
        """``B_t = ∂Δ_B/∂t_corr`` in s/s. Holds parameters and TOA ``obs_pos`` fixed.

        Geometry still depends on ``t_corr`` through ``Δt_K``. This is not
        ``Δ_Dop``. Dual seed on ``TCORR`` is one day, so the raw tangent is
        converted to per-second.
        """
        return self.evaluate(t, tangent_par="TCORR").d_delay / _DAY

    def d_DDRdelay_d_tcorr(self, t=None):
        return self.d_delay_d_tcorr(t)

    def period_derivative(self, t=None):
        """Dimensionless total ``p`` from the current PK / kinematic composition."""
        if t is None:
            t = self.t
        if t is None:
            t = self.TASC
        st = self.evaluate(np.atleast_1d(as_ld(t)).reshape(-1)[:1])
        p = np.atleast_1d(st.p)
        return p[0]

    def DDRdelay(self):
        return self.delay() * u.s

    def binary_delay(self):
        return self.delay() * u.s

    def d_DDRdelay_d_par(self, par):
        return self.d_delay_d_par(par) * u.s / u.Unit("")

    def d_binarydelay_d_par(self, par):
        return self.d_DDRdelay_d_par(par)
