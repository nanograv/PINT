"""Damour-Deruelle-Regular binary (BINARY DDR) PINT wrap.

Anomalistic ``PB``, Laplace-Lagrange ``(EPS1, EPS2)`` at ``TASC``, frozen
``TGEO`` astrometric origin, DT92 ``KOM`` / ``COSI``, and full Shapiro. This is
not the ELL1 series and not stock DDK. The delay kernel is
:class:`~pint.models.stand_alone_psr_binaries.DDR_model.DDRmodel`
(van Haasteren in prep.).

``DDRPK`` / ``DDRPBDOT`` control GR Einstein + GR :math:`\\dot\\omega` and the
:math:`\\dot P_b` split; see the class docstring table.
"""

from __future__ import annotations

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, CartesianRepresentation, Galactic, SkyCoord

from pint import Tsun
from pint.exceptions import InvalidModelParameters, MissingParameter, TimingModelError
from pint.models.parameter import (
    MJDParameter,
    boolParameter,
    floatParameter,
    funcParameter,
    strParameter,
)
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries import ddr_kepler as kep
from pint.models.stand_alone_psr_binaries.ddr_kepler import (
    Dual,
    atan2 as dual_atan2,
    cos as dual_cos,
    sin as dual_sin,
    sqrt as dual_sqrt,
)
from pint.models.stand_alone_psr_binaries.DDR_model import (
    DDRmodel,
    g_gamma_gr,
    kappa_gr,
    pulsar_mass,
    value,
)
from pint.models.timing_model import Component
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.utils import add_dummy_distance, parse_time, pmtot, remove_dummy_distance

_DAY = np.longdouble(86400)
_TWO_PI = np.longdouble(2) * np.pi
_LD15 = np.longdouble("1.5")
DDR_FB_KMAX = 16
_GALAXY_NAMES = ("DDRR0", "DDRTHETA0", "DDRRHO0", "DDRZ0", "DDRZSUN")
_UNFREEZABLE = _GALAXY_NAMES + ("TGEO",)
_MODE_NAMES = ("DDRPK", "DDRPBDOT", "DDRGEO", "DDRKINE")
_ZERO_PLACEHOLDERS = ("EDOT", "EPS1DOT", "EPS2DOT", "DR", "DTH")
# ---------------------------------------------------------------------------
# Display / derived helpers (pickleable; ELL1 pattern)
# ---------------------------------------------------------------------------


def _eps_to_e(eps1, eps2):
    return np.sqrt(eps1**2 + eps2**2)


def _eps_to_om(eps1, eps2):
    om = np.arctan2(eps1, eps2)
    if om < 0:
        om += 360 * u.deg
    return om.to(u.deg)


def _cosi_to_sini(cosi):
    c = np.longdouble(cosi.to_value(u.dimensionless_unscaled))
    return np.sqrt((1 - c) * (1 + c)) * u.dimensionless_unscaled


def _cosi_to_kin(cosi):
    c = np.clip(np.longdouble(cosi.to_value(u.dimensionless_unscaled)), -1, 1)
    return (np.arccos(c) * u.rad).to(u.deg)


def _convert_kin(kin):
    return 180 * u.deg - kin


def _convert_kom(kom):
    return 90 * u.deg - kom


def fw10_stigma(cosi):
    """Folded FW10 :math:`\\varsigma = s / (1+|c|)` (van Haasteren in prep. §10)."""
    c = np.longdouble(cosi.to_value(u.dimensionless_unscaled))
    s = np.sqrt((1 - c) * (1 + c))
    return (s / (1 + np.abs(c))) * u.dimensionless_unscaled


def fw10_h3(m2, cosi):
    """FW10 :math:`H_3 = T_\\odot m_c \\varsigma^3`."""
    return Tsun * np.longdouble(m2.to_value(u.Msun)) * fw10_stigma(cosi) ** 3


def fw10_h4(m2, cosi):
    """FW10 :math:`H_4 = H_3 \\varsigma`."""
    return fw10_h3(m2, cosi) * fw10_stigma(cosi)


def fw10_encode(m2, cosi):
    """Return ``(H3, STIGMA, H4)`` from ``(M2, COSI)``."""
    stigma = fw10_stigma(cosi)
    h3 = fw10_h3(m2, cosi)
    return h3, stigma, h3 * stigma


def fw10_decode(h3, stigma):
    """Invert folded FW10 ``(H3, STIGMA)`` to ``(M2, |COSI|)``.

    ``STIGMA`` is folded, so the sign of ``COSI`` is not recovered.
    """
    h3_s = np.longdouble(h3.to_value(u.s))
    sig = np.longdouble(stigma.to_value(u.dimensionless_unscaled))
    if not np.isfinite(h3_s) or not np.isfinite(sig):
        raise InvalidModelParameters("FW10 decode inputs must be finite")
    if sig <= 0 or sig > 1:
        raise InvalidModelParameters("FW10 decode requires 0 < STIGMA <= 1")
    if h3_s <= 0:
        raise InvalidModelParameters("FW10 decode requires H3 > 0")
    m2 = np.longdouble((h3 / Tsun / stigma**3).to_value(u.one)) * u.Msun
    # ς = s/(1+|c|); s^2 + c^2 = 1 ⇒ |c| = (1-ς^2)/(1+ς^2)
    abs_c = (1 - sig * sig) / (1 + sig * sig)
    return m2, abs_c * u.dimensionless_unscaled


def fw10_orbit_decode(x_a, h_a, k_a, tasc_d, pb_s, r_s, stigma):
    """Absorbed ELL1H coordinates → intrinsic DDR (analytics §10.3).

    Complete map: the epoch already includes the ELL1 Roemer gauge
    ``3/2 x h``. Do not compose with a further TASC translation.
    ``x`` / ``r`` in seconds, ``TASC`` in days. Decode domain:
    ``x = x_a - 4 r ς > 0``, ``0 < ς ≤ 1``.
    """
    x_a = np.longdouble(x_a)
    h_a = np.longdouble(h_a)
    k_a = np.longdouble(k_a)
    tasc_d = np.longdouble(tasc_d)
    pb_s = np.longdouble(pb_s)
    r_s = np.longdouble(r_s)
    stigma = np.longdouble(stigma)
    if not np.all(np.isfinite([x_a, h_a, k_a, tasc_d, pb_s, r_s, stigma])):
        raise InvalidModelParameters("FW orbit decode inputs must be finite")
    if pb_s <= 0 or r_s < 0:
        raise InvalidModelParameters("FW orbit decode requires PB>0 and r>=0")
    if not (stigma > 0 and stigma <= 1):
        raise InvalidModelParameters("FW orbit decode requires 0 < STIGMA <= 1")
    n = _TWO_PI / pb_s
    x = x_a - 4 * r_s * stigma
    if not np.isfinite(x) or x <= 0:
        raise InvalidModelParameters("FW orbit decode requires x_a - 4 r ς > 0")
    h = (x_a * h_a - 4 * r_s * stigma * stigma) / x
    k = (x_a * k_a - 8 * n * r_s * stigma * x) / x
    tasc = tasc_d + (_LD15 * x * h + r_s * stigma * stigma) / _DAY
    return x, h, k, tasc


def fw10_orbit_encode(x, h, k, tasc_d, pb_s, r_s, stigma):
    """Intrinsic DDR coordinates → absorbed ELL1H (inverse of decode).

    Complete inverse map. Do not apply the reverse ELL1 TASC
    translation before calling this.
    """
    x = np.longdouble(x)
    h = np.longdouble(h)
    k = np.longdouble(k)
    tasc_d = np.longdouble(tasc_d)
    pb_s = np.longdouble(pb_s)
    r_s = np.longdouble(r_s)
    stigma = np.longdouble(stigma)
    if not np.all(np.isfinite([x, h, k, tasc_d, pb_s, r_s, stigma])):
        raise InvalidModelParameters("FW orbit encode inputs must be finite")
    if pb_s <= 0 or r_s < 0:
        raise InvalidModelParameters("FW orbit encode requires PB>0 and r>=0")
    if not (stigma > 0 and stigma <= 1):
        raise InvalidModelParameters("FW orbit encode requires 0 < STIGMA <= 1")
    n = _TWO_PI / pb_s
    xa = x + 4 * r_s * stigma
    if not np.isfinite(xa) or xa <= 0:
        raise InvalidModelParameters("FW orbit encode produced non-positive A1")
    ha = (x * h + 4 * r_s * stigma * stigma) / xa
    ka = (x * k + 8 * n * r_s * stigma * x) / xa
    tasc_a = tasc_d - (_LD15 * x * h + r_s * stigma * stigma) / _DAY
    return xa, ha, ka, tasc_a


def _ld_qty(q, unit):
    return np.longdouble(q.to_value(unit))


def _mass_sini(PB, A1, M2, COSI):
    pb_s = _ld_qty(PB, u.s)
    n = _TWO_PI / pb_s
    x = _ld_qty(A1, u.lsec)
    mc = _ld_qty(M2, u.Msun)
    c = np.longdouble(COSI.to_value(u.dimensionless_unscaled))
    return pulsar_mass(n, x, mc, c)


def _ddr_omdot_gr(PB, A1, M2, COSI, EPS1, EPS2):
    mp, s = _mass_sini(PB, A1, M2, COSI)
    h = np.longdouble(EPS1.to_value(u.dimensionless_unscaled))
    k = np.longdouble(EPS2.to_value(u.dimensionless_unscaled))
    e2 = h * h + k * k
    x = _ld_qty(A1, u.lsec)
    mc = _ld_qty(M2, u.Msun)
    kap = value(kappa_gr(x, mc, s, e2))
    n = _TWO_PI / _ld_qty(PB, u.s)
    return (kap * n * u.rad / u.s).to(u.deg / u.yr)


def _ddr_gamma_gr(PB, A1, M2, COSI, EPS1, EPS2):
    mp, _s = _mass_sini(PB, A1, M2, COSI)
    g = value(g_gamma_gr(_ld_qty(PB, u.s), mp, _ld_qty(M2, u.Msun)))
    e = np.sqrt(
        EPS1.to_value(u.dimensionless_unscaled) ** 2
        + EPS2.to_value(u.dimensionless_unscaled) ** 2
    )
    return (np.longdouble(e) * g) * u.s


def _ddr_gamma_from_ggamma(GGAMMA, EPS1, EPS2):
    e = np.sqrt(
        EPS1.to_value(u.dimensionless_unscaled) ** 2
        + EPS2.to_value(u.dimensionless_unscaled) ** 2
    )
    return np.longdouble(e) * GGAMMA


def _ddr_pbdot_from_kernel(**params):
    m = DDRmodel(**params)
    return m.period_derivative() * u.dimensionless_unscaled


def _ddr_pbdot_kinematic_nokine(PB, A1, M2, COSI, EPS1, EPS2, XPBDOT):
    return _ddr_pbdot_from_kernel(
        PB=_ld_qty(PB, u.d),
        A1=_ld_qty(A1, u.lsec),
        M2=_ld_qty(M2, u.Msun),
        COSI=np.longdouble(COSI.to_value(u.dimensionless_unscaled)),
        EPS1=np.longdouble(EPS1.to_value(u.dimensionless_unscaled)),
        EPS2=np.longdouble(EPS2.to_value(u.dimensionless_unscaled)),
        XPBDOT=np.longdouble(XPBDOT.to_value(u.dimensionless_unscaled)),
        DDRPK=True,
        DDRPBDOT="kinematic",
        DDRGEO=False,
        DDRKINE=False,
    )


def _galactic_lb_equatorial(RAJ, DECJ, PMRA, PMDEC, POSEPOCH, TGEO):
    c0 = SkyCoord(
        ra=RAJ,
        dec=DECJ,
        pm_ra_cosdec=PMRA,
        pm_dec=PMDEC,
        obstime=POSEPOCH,
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ck = remove_dummy_distance(
            add_dummy_distance(c0).apply_space_motion(new_obstime=TGEO)
        )
    gal = ck.galactic
    return np.longdouble(gal.l.to_value(u.rad)), np.longdouble(gal.b.to_value(u.rad))


def _ddr_pbdot_kinematic_ecliptic(
    PB,
    A1,
    M2,
    COSI,
    EPS1,
    EPS2,
    XPBDOT,
    PX,
    PMELONG,
    PMELAT,
    ELONG,
    ELAT,
    POSEPOCH,
    TGEO,
    ECL,
    DDRR0,
    DDRTHETA0,
    DDRRHO0,
    DDRZ0,
    DDRZSUN,
):
    ecl = ECL if isinstance(ECL, str) else str(ECL)
    c0 = SkyCoord(
        lon=ELONG,
        lat=ELAT,
        pm_lon_coslat=PMELONG,
        pm_lat=PMELAT,
        obstime=POSEPOCH,
        frame=PulsarEcliptic(ecl=ecl),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ck = remove_dummy_distance(
            add_dummy_distance(c0).apply_space_motion(new_obstime=TGEO)
        )
    gal = ck.galactic
    l_rad = np.longdouble(gal.l.to_value(u.rad))
    b_rad = np.longdouble(gal.b.to_value(u.rad))
    pm = np.sqrt(PMELONG**2 + PMELAT**2)
    return _ddr_pbdot_from_kernel(
        PB=_ld_qty(PB, u.d),
        A1=_ld_qty(A1, u.lsec),
        M2=_ld_qty(M2, u.Msun),
        COSI=np.longdouble(COSI.to_value(u.dimensionless_unscaled)),
        EPS1=np.longdouble(EPS1.to_value(u.dimensionless_unscaled)),
        EPS2=np.longdouble(EPS2.to_value(u.dimensionless_unscaled)),
        XPBDOT=np.longdouble(XPBDOT.to_value(u.dimensionless_unscaled)),
        PX=_ld_qty(PX, u.mas),
        PM=_ld_qty(pm, u.mas / u.yr),
        l_rad=l_rad,
        b_rad=b_rad,
        DDRR0=np.longdouble(DDRR0.value),
        DDRTHETA0=np.longdouble(DDRTHETA0.value),
        DDRRHO0=np.longdouble(DDRRHO0.value),
        DDRZ0=np.longdouble(DDRZ0.value),
        DDRZSUN=np.longdouble(DDRZSUN.value),
        DDRPK=True,
        DDRPBDOT="kinematic",
        DDRGEO=False,
        DDRKINE=True,
    )


def _ddr_pbdot_kinematic_equatorial(
    PB,
    A1,
    M2,
    COSI,
    EPS1,
    EPS2,
    XPBDOT,
    PX,
    PMRA,
    PMDEC,
    RAJ,
    DECJ,
    POSEPOCH,
    TGEO,
    DDRR0,
    DDRTHETA0,
    DDRRHO0,
    DDRZ0,
    DDRZSUN,
):
    l_rad, b_rad = _galactic_lb_equatorial(RAJ, DECJ, PMRA, PMDEC, POSEPOCH, TGEO)
    pm = np.sqrt(PMRA**2 + PMDEC**2)
    return _ddr_pbdot_from_kernel(
        PB=_ld_qty(PB, u.d),
        A1=_ld_qty(A1, u.lsec),
        M2=_ld_qty(M2, u.Msun),
        COSI=np.longdouble(COSI.to_value(u.dimensionless_unscaled)),
        EPS1=np.longdouble(EPS1.to_value(u.dimensionless_unscaled)),
        EPS2=np.longdouble(EPS2.to_value(u.dimensionless_unscaled)),
        XPBDOT=np.longdouble(XPBDOT.to_value(u.dimensionless_unscaled)),
        PX=_ld_qty(PX, u.mas),
        PM=_ld_qty(pm, u.mas / u.yr),
        l_rad=l_rad,
        b_rad=b_rad,
        DDRR0=np.longdouble(DDRR0.value),
        DDRTHETA0=np.longdouble(DDRTHETA0.value),
        DDRRHO0=np.longdouble(DDRRHO0.value),
        DDRZ0=np.longdouble(DDRZ0.value),
        DDRZSUN=np.longdouble(DDRZSUN.value),
        DDRPK=True,
        DDRPBDOT="kinematic",
        DDRGEO=False,
        DDRKINE=True,
    )


def _dual_vector(*items):
    return Dual(
        np.asarray([item.v for item in items], dtype=np.longdouble),
        np.asarray([item.d for item in items], dtype=np.longdouble),
    )


def _dual_item(vector, index):
    return Dual(vector.v[index], vector.d[index])


def _dual_dot(left, right):
    """Contract the last axis so ``obs_pos`` of shape ``(ntoa, 3)`` stays per-TOA."""
    return Dual(
        np.sum(left.v * right.v, axis=-1),
        np.sum(left.d * right.v + left.v * right.d, axis=-1),
    )


def _dual_cross(left, right):
    return Dual(
        np.cross(left.v, right.v),
        np.cross(left.d, right.v) + np.cross(left.v, right.d),
    )


def _dual_matvec(matrix, vector):
    return Dual(matrix @ vector.v, matrix @ vector.d)


def _frame_rotation(from_frame, to_frame):
    """Fixed Cartesian rotation from one Astropy sky frame to another."""
    basis = CartesianRepresentation(np.eye(3) * u.one, xyz_axis=0)
    transformed = SkyCoord(basis, frame=from_frame).transform_to(to_frame)
    return np.asarray(transformed.cartesian.xyz.value, dtype=np.longdouble)


_ICRS_TO_GAL = _frame_rotation(ICRS(), Galactic())


def _analytic_space_motion(
    lon,
    lat,
    pm_lon,
    pm_lat,
    dt_s,
    obs_pos,
    native_to_icrs,
    *,
    seed=None,
):
    """Closed-form TGEO triad and tangent for one astrometric parameter."""
    lon = Dual(lon, 1 if seed == "lon" else 0)
    lat = Dual(lat, 1 if seed == "lat" else 0)
    pm_lon = Dual(pm_lon, 1 if seed == "pm_lon" else 0)
    pm_lat = Dual(pm_lat, 1 if seed == "pm_lat" else 0)
    cl, sl = dual_cos(lon), dual_sin(lon)
    cb, sb = dual_cos(lat), dual_sin(lat)
    n_p = _dual_vector(cb * cl, cb * sl, sb)
    east_p = _dual_vector(-sl, cl, Dual(0))
    north_p = _dual_vector(-sb * cl, -sb * sl, cb)
    ndot_p = pm_lon * east_p + pm_lat * north_p
    r = n_p + ndot_p * np.longdouble(dt_s)
    rnorm = dual_sqrt(_dual_dot(r, r))
    n = r / rnorm
    ndot = (ndot_p - _dual_dot(ndot_p, n) * n) / rnorm

    nx, ny = _dual_item(n, 0), _dual_item(n, 1)
    rho = dual_sqrt(nx * nx + ny * ny)
    east = _dual_vector(-ny / rho, nx / rho, Dual(0))
    north = _dual_cross(n, east)

    n_icrs = _dual_matvec(native_to_icrs, n)
    n_gal = _dual_matvec(_ICRS_TO_GAL, n_icrs)
    gx, gy, gz = (_dual_item(n_gal, j) for j in range(3))
    gal_lon = dual_atan2(gy, gx)
    gal_lat = dual_atan2(gz, dual_sqrt(gx * gx + gy * gy))

    obs_pos = np.asarray(obs_pos, dtype=np.longdouble)
    pm_squared = pm_lon * pm_lon + pm_lat * pm_lat
    return {
        "I0": east,
        "J0": north,
        "n0": n,
        "mu_I": _dual_dot(east, ndot),
        "mu_J": _dual_dot(north, ndot),
        "d_I_au": _dual_dot(Dual(obs_pos), east),
        "d_J_au": _dual_dot(Dual(obs_pos), north),
        "l_rad": gal_lon,
        "b_rad": gal_lat,
        "pm_squared_rad_s2": pm_squared,
    }


class BinaryDDR(PulsarBinary):
    """Damour-Deruelle-Regular binary model (``BINARY DDR``).

    Regular: delay nonsingular at ``e=0``. Not a Laplace-Lagrange rename
    of DD/ELL1.

    Coordinates are the anomalistic period ``PB``, projected axis ``A1`` at
    ``TASC`` (mean longitude zero; catalog description matches ELL1),
    Laplace-Lagrange ``(EPS1, EPS2)`` at ``TASC``, companion mass ``M2``, and
    DT92 ``COSI``. ``TGEO`` is the frozen astrometric origin of the Cartesian
    projector (materialized once from ``TASC`` if omitted). ``KOM`` is DT92,
    east through north in the model's sky frame.

    This is the exact Kepler + projector delay (van Haasteren in prep.), not
    the ELL1 Fourier series and not stock DDK annual-orbital parallax. Shapiro
    uses the physical ``r, B_S`` form.

    Mode flags:

    ===========  ============  ===============================================
    ``DDRPK``    ``DDRPBDOT``  GR / :math:`\\dot P_b` in the delay
    ===========  ============  ===============================================
    Y            kinematic     Einstein, GR precession, quadrupole GW
    Y            absorb_gw     Einstein, GR precession; free ``PBDOT``
    N            kinematic     quadrupole only; free ``GGAMMA``, ``OMDOT``
    N            absorb_gw     none of those three
    ===========  ============  ===============================================

    ``DDRKINE Y`` adds Shklovskii + Galactic :math:`\\dot P_b`. ``DDRGEO Y``
    turns on the Cartesian projector and requires ``KOM``, ``PX>0``, and
    ``DDRKINE Y``. The FBX chart accepts the complete ``FB0``-``FBK`` prefix
    (general order, refused above ``DDR_FB_KMAX``); it is not truncated at
    FB5. ``EDOT`` / ``EPS*DOT`` are unsupported zero placeholders.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ddr.BinaryDDR
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "DDR"
        self.binary_epoch_name = "TASC"
        self.binary_model_class = DDRmodel
        self.warn_default_params = []
        # DDR applies -B_t A_θ through d_ddr_time_argument_correction on
        # upstream parameters. The kernel is not a PSR_BINARY, so the generic
        # delay_deriv_wrt_prev_delay chain would both crash and double-count.
        self.delay_deriv_wrt_prev_delay_funcs = []
        self._tgeo_materialized = False
        self._schema_finalized = False
        self._finalized_modes = None
        self._batched_derivative_cache = None

        self.add_param(
            MJDParameter(
                name="TASC",
                # Catalog text must match ELL1; DDR's TASC is λ=0 (class docstring).
                description="Epoch of ascending node",
                time_scale="tdb",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            MJDParameter(
                name="TGEO",
                description="Frozen astrometric origin of the DDR projector",
                time_scale="tdb",
                frozen=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="EPS1",
                units="",
                description="First Laplace-Lagrange parameter, ECC*sin(OM)",
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="EPS2",
                units="",
                description="Second Laplace-Lagrange parameter, ECC*cos(OM)",
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="COSI",
                units="",
                description="Cosine of inclination (DT92)",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="KOM",
                units="deg",
                description="The longitude of the ascending node",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="GGAMMA",
                units="second",
                description="Regular Einstein coefficient g_γ (γ/e)",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="GAMMA",
                units="second",
                description="Time dilation & gravitational redshift",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="XPBDOT",
                units=u.day / u.day,
                description="Excess Orbital period derivative respect to time compared to GR",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            boolParameter(
                name="DDRPK",
                value=True,
                description="GR Einstein delay and GR periastron advance",
            )
        )
        self.add_param(
            strParameter(
                name="DDRPBDOT",
                value="kinematic",
                description="Pbdot mode: kinematic or absorb_gw",
            )
        )
        self.add_param(
            boolParameter(
                name="DDRGEO",
                value=True,
                description="Cartesian annual-geometry projector",
            )
        )
        self.add_param(
            boolParameter(
                name="DDRKINE",
                value=True,
                description="Shklovskii and Galactic Pbdot (and geometry distance)",
            )
        )
        self.add_param(
            floatParameter(
                name="DDRR0",
                units="kpc",
                value=8.178,
                description="Galactic R0 (frozen)",
                frozen=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="DDRTHETA0",
                units="km/s",
                value=220.0,
                description="Galactic Θ0 (frozen)",
                frozen=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="DDRRHO0",
                units="Msun/pc^3",
                value=0.10,
                description="Local dark-matter density (frozen)",
                frozen=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="DDRZ0",
                units="pc",
                value=180.0,
                description="Galactic disk scale height (frozen)",
                frozen=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="DDRZSUN",
                units="pc",
                value=20.0,
                description="Sun's height above the Galactic plane (frozen)",
                frozen=True,
                convert_tcb2tdb=False,
            )
        )

        for name in ("ECC", "OM", "T0", "SINI", "EDOT"):
            self.remove_param(name)

        self.add_param(
            floatParameter(
                name="EDOT",
                units="1/s",
                description="Eccentricity derivative respect to time",
                frozen=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="EPS1DOT",
                units="1e-12/s",
                description="First derivative of first Laplace-Lagrange parameter",
                frozen=True,
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="EPS2DOT",
                units="1e-12/s",
                description="Second derivative of first Laplace-Lagrange parameter",
                frozen=True,
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="DR",
                units="",
                description="Relativistic deformation of the orbit",
                frozen=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="DTH",
                units="",
                aliases=["DTHETA"],
                description="Relativistic deformation of the orbit",
                frozen=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.add_param(
            funcParameter(
                name="ECC",
                units="",
                aliases=["E"],
                description="Eccentricity",
                params=("EPS1", "EPS2"),
                func=_eps_to_e,
            )
        )
        self.add_param(
            funcParameter(
                name="OM",
                units=u.deg,
                description="Longitude of periastron",
                long_double=True,
                params=("EPS1", "EPS2"),
                func=_eps_to_om,
            )
        )
        self.add_param(
            funcParameter(
                name="SINI",
                units="",
                description="Sine of inclination angle",
                params=("COSI",),
                func=_cosi_to_sini,
            )
        )
        self.add_param(
            funcParameter(
                name="KIN",
                units="deg",
                description="Inclination angle",
                params=("COSI",),
                func=_cosi_to_kin,
            )
        )
        self.add_param(
            funcParameter(
                name="KINIAU",
                units="deg",
                description="Inclination angle in the IAU convention",
                params=("KIN",),
                func=_convert_kin,
            )
        )
        self.add_param(
            funcParameter(
                name="KOMIAU",
                units="deg",
                description="The longitude of the ascending node in the IAU convention",
                params=("KOM",),
                func=_convert_kom,
            )
        )
        self.add_param(
            funcParameter(
                name="STIGMA",
                units="",
                description="Shapiro delay parameter STIGMA as in Freire and Wex 2010 Eq(12)",
                aliases=["VARSIGMA", "STIG"],
                params=("COSI",),
                func=fw10_stigma,
            )
        )
        self.add_param(
            funcParameter(
                name="H3",
                units="s",
                description="Shapiro delay parameter H3 as in Freire and Wex 2010 Eq(20)",
                params=("M2", "COSI"),
                func=fw10_h3,
            )
        )
        self.add_param(
            funcParameter(
                name="H4",
                units="s",
                description="Shapiro delay parameter H4 as in Freire and Wex 2010 Eq(21)",
                params=("M2", "COSI"),
                func=fw10_h4,
            )
        )
        self.internal_params += ["PMLONG_DDR", "PMLAT_DDR"]

    @property
    def PMLONG_DDR(self):
        if "AstrometryEquatorial" in self._parent.components:
            return self._parent.PMRA
        if "AstrometryEcliptic" in self._parent.components:
            return self._parent.PMELONG
        raise TimingModelError(
            "No valid AstrometryEcliptic or AstrometryEquatorial component found"
        )

    @property
    def PMLAT_DDR(self):
        if "AstrometryEquatorial" in self._parent.components:
            return self._parent.PMDEC
        if "AstrometryEcliptic" in self._parent.components:
            return self._parent.PMELAT
        raise TimingModelError(
            "No valid AstrometryEcliptic or AstrometryEquatorial component found"
        )

    def _is_writable(self, name):
        return name in self.params and not isinstance(
            getattr(self, name), funcParameter
        )

    def _replace_with_func(self, name, func_par):
        if name in self.params:
            existing = getattr(self, name)
            if isinstance(existing, funcParameter) and existing._func is func_par._func:
                return
            self.remove_param(name)
        self.add_param(func_par)

    def _pbdot_mode(self):
        v = str(self.DDRPBDOT.value).strip().lower()
        if v not in ("kinematic", "absorb_gw"):
            raise TimingModelError("DDRPBDOT must be 'kinematic' or 'absorb_gw'")
        return v

    def _using_fbx(self):
        return any(
            getattr(self, name).quantity is not None
            for name in self._fbx_mapping().values()
        )

    def _fbx_order(self):
        if not self._using_fbx():
            return None
        mapping = self._fbx_mapping()
        return max(mapping) if mapping else 0

    def _fbx_coefficients(self):
        mapping = self._fbx_mapping()
        return np.asarray(
            [
                np.longdouble(
                    getattr(self, mapping[j]).quantity.to_value(u.s ** (-(j + 1)))
                )
                for j in range(max(mapping) + 1)
            ],
            dtype=np.longdouble,
        )

    def _mode_signature(self):
        return (
            bool(self.DDRPK.value),
            self._pbdot_mode(),
            bool(self.DDRGEO.value),
            bool(self.DDRKINE.value),
            "fbx" if self._using_fbx() else "pb",
            self._fbx_order(),
        )

    def _check_static_modes(self):
        if self._finalized_modes is None:
            return
        if self._mode_signature() != self._finalized_modes:
            raise TimingModelError(
                "DDR mode flags are static after setup; build a new component"
            )

    def _refuse_orbwave(self):
        names = ["ORBWAVE_OM", "ORBWAVE_EPOCH"]
        names += list(self.get_prefix_mapping_component("ORBWAVEC").values())
        names += list(self.get_prefix_mapping_component("ORBWAVES").values())
        for name in names:
            par = getattr(self, name, None)
            if par is not None and par.quantity is not None:
                raise TimingModelError(f"BINARY DDR does not support {name}")

    def _apply_conflicts(self):
        pk = bool(self.DDRPK.value)
        mode = self._pbdot_mode()
        if pk:
            if self._is_writable("OMDOT") and self.OMDOT.quantity is not None:
                raise TimingModelError(
                    "DDRPK Y does not accept an independent OMDOT; omit OMDOT or set DDRPK N"
                )
            if self._is_writable("GGAMMA") and self.GGAMMA.quantity is not None:
                raise TimingModelError(
                    "DDRPK Y does not accept an independent GGAMMA; omit GGAMMA or set DDRPK N"
                )
            if self._is_writable("GAMMA") and self.GAMMA.quantity is not None:
                raise TimingModelError(
                    "DDRPK Y does not accept an independent GAMMA; omit GAMMA or set DDRPK N"
                )
        else:
            if (
                self._is_writable("GAMMA")
                and self.GAMMA.quantity is not None
                and np.longdouble(self.GAMMA.value) == 0
                and self.GGAMMA.quantity is None
            ):
                self.GGAMMA.value = 0
            elif (
                self._is_writable("GAMMA")
                and self.GAMMA.quantity is not None
                and np.longdouble(self.GAMMA.value) != 0
                and self.GGAMMA.quantity is None
            ):
                h = np.longdouble(self.EPS1.value or 0)
                k = np.longdouble(self.EPS2.value or 0)
                eccentricity = np.hypot(h, k)
                if eccentricity <= np.longdouble("1e-6"):
                    raise TimingModelError(
                        "DDRPK N cannot regularize nonzero GAMMA at eccentricity <= 1e-6"
                    )
                gamma = self.GAMMA
                self.GGAMMA.value = np.longdouble(gamma.value) / eccentricity
                self.GGAMMA.frozen = gamma.frozen
                if gamma.uncertainty is not None:
                    self.GGAMMA.uncertainty_value = (
                        np.longdouble(gamma.uncertainty.to_value(u.s)) / eccentricity
                    )
        if mode == "kinematic":
            if self._is_writable("PBDOT") and self.PBDOT.quantity is not None:
                self._explicit_pbdot = np.longdouble(self.PBDOT.value)
            else:
                self._explicit_pbdot = None
        else:
            self._explicit_pbdot = None
            if self._is_writable("XPBDOT") and self.XPBDOT.quantity is not None:
                x = np.longdouble(self.XPBDOT.value)
                if x != 0:
                    raise TimingModelError(
                        "XPBDOT is only used with DDRPBDOT kinematic"
                    )

    def _kinematic_p_total(self):
        inst = DDRmodel()
        inst.update(**self._kernel_param_dict())
        return np.longdouble(inst.period_derivative())

    def _check_pbdot_identity(self):
        """Accept explicit kinematic ``PBDOT`` only if it matches ``p_base+XPBDOT``.

        Tolerance is analytics/proposal §7:
        ``|Δp| ≤ 1e-18 + 1e-6 |PBDOT|``. Ordinary load never rewrites ``XPBDOT``.
        """
        got = getattr(self, "_explicit_pbdot", None)
        if got is None:
            return
        if self._parent is None:
            return
        expected = self._kinematic_p_total()
        tol = np.longdouble("1e-18") + np.longdouble("1e-6") * abs(got)
        if abs(got - expected) > tol:
            raise TimingModelError(
                "DDRPBDOT kinematic: explicit PBDOT does not match "
                f"p_base+XPBDOT ({got} vs {expected})"
            )
        self._explicit_pbdot = None

    def _install_pbdot_view(self):
        if self._using_fbx():
            return
        if self._pbdot_mode() != "kinematic":
            if self.PBDOT.quantity is None:
                self.PBDOT.value = 0
            return
        if self.XPBDOT.quantity is None:
            self.XPBDOT.value = 0
            self.XPBDOT.frozen = True
        kine = bool(self.DDRKINE.value)
        if kine and self._parent is not None:
            if "AstrometryEquatorial" in self._parent.components:
                self._replace_with_func(
                    "PBDOT",
                    funcParameter(
                        name="PBDOT",
                        units=u.day / u.day,
                        description="Total kinematic Pbdot",
                        unit_scale=True,
                        scale_factor=1e-12,
                        scale_threshold=1e-7,
                        params=(
                            "PB",
                            "A1",
                            "M2",
                            "COSI",
                            "EPS1",
                            "EPS2",
                            "XPBDOT",
                            "PX",
                            "PMRA",
                            "PMDEC",
                            "RAJ",
                            "DECJ",
                            "POSEPOCH",
                            "TGEO",
                            "DDRR0",
                            "DDRTHETA0",
                            "DDRRHO0",
                            "DDRZ0",
                            "DDRZSUN",
                        ),
                        func=_ddr_pbdot_kinematic_equatorial,
                    ),
                )
            else:
                self._replace_with_func(
                    "PBDOT",
                    funcParameter(
                        name="PBDOT",
                        units=u.day / u.day,
                        description="Total kinematic Pbdot",
                        unit_scale=True,
                        scale_factor=1e-12,
                        scale_threshold=1e-7,
                        params=(
                            "PB",
                            "A1",
                            "M2",
                            "COSI",
                            "EPS1",
                            "EPS2",
                            "XPBDOT",
                            "PX",
                            "PMELONG",
                            "PMELAT",
                            "ELONG",
                            "ELAT",
                            "POSEPOCH",
                            "TGEO",
                            "ECL",
                            "DDRR0",
                            "DDRTHETA0",
                            "DDRRHO0",
                            "DDRZ0",
                            "DDRZSUN",
                        ),
                        func=_ddr_pbdot_kinematic_ecliptic,
                    ),
                )
        else:
            self._replace_with_func(
                "PBDOT",
                funcParameter(
                    name="PBDOT",
                    units=u.day / u.day,
                    description="GW+XPBDOT kinematic Pbdot",
                    unit_scale=True,
                    scale_factor=1e-12,
                    scale_threshold=1e-7,
                    params=("PB", "A1", "M2", "COSI", "EPS1", "EPS2", "XPBDOT"),
                    func=_ddr_pbdot_kinematic_nokine,
                ),
            )

    def _finalize_pk_schema(self):
        pk = bool(self.DDRPK.value)
        if pk:
            self._replace_with_func(
                "OMDOT",
                funcParameter(
                    name="OMDOT",
                    units="deg/year",
                    description="GR periastron advance",
                    long_double=True,
                    params=("PB", "A1", "M2", "COSI", "EPS1", "EPS2"),
                    func=_ddr_omdot_gr,
                ),
            )
            self._replace_with_func(
                "GAMMA",
                funcParameter(
                    name="GAMMA",
                    units="second",
                    description="Display Einstein parameter e g_γ^GR",
                    params=("PB", "A1", "M2", "COSI", "EPS1", "EPS2"),
                    func=_ddr_gamma_gr,
                ),
            )
        else:
            self._replace_with_func(
                "GAMMA",
                funcParameter(
                    name="GAMMA",
                    units="second",
                    description="Display Einstein parameter e GGAMMA",
                    params=("GGAMMA", "EPS1", "EPS2"),
                    func=_ddr_gamma_from_ggamma,
                ),
            )
            if self.GGAMMA.quantity is None:
                self.GGAMMA.value = 0
            if self.OMDOT.quantity is None:
                self.OMDOT.value = 0

    def setup(self):
        """Finalize mode-dependent schema and register delay derivatives.

        TimingModel inserts delay components in ``DEFAULT_ORDER``, so
        DispersionDM runs before BinaryDDR. Upstream ``-B_t A_θ`` names
        are registered here once (not on the delay hot path). Prefix
        parameters such as ``DMX_*`` that appear after this ``setup()``
        are not auto-registered until a later ``setup()``.
        """
        if self.DDRPBDOT.value is not None:
            self.DDRPBDOT.value = str(self.DDRPBDOT.value).strip().lower()
        self._check_static_modes()
        self._refuse_orbwave()
        if not self._schema_finalized:
            if (
                self._using_fbx()
                and self._is_writable("PBDOT")
                and self.PBDOT.quantity is not None
            ):
                # FB0 + any PBDOT is a mixed chart. PB + FBn≥2 + nonzero PBDOT
                # would otherwise be silently rewritten as an (FB0, FB1)
                # truncation of the higher coefficients.
                if (
                    self.FB0.quantity is not None
                    or np.longdouble(self.PBDOT.value) != 0
                ):
                    raise TimingModelError(
                        "DDR FBX chart does not accept PBDOT; encode the total phase in FBn"
                    )
            if self._using_fbx():
                order = self._fbx_order()
                if order is not None and order > DDR_FB_KMAX:
                    raise TimingModelError(
                        f"DDR FB order K={order} exceeds DDR_FB_KMAX={DDR_FB_KMAX}"
                    )
            try:
                self._setup_fbx_parameterization()
            except ValueError as exc:
                raise TimingModelError(str(exc)) from exc
        if self.TGEO.value is None and self.TASC.value is not None:
            self.TGEO.quantity = self.TASC.quantity
            self._tgeo_materialized = True
        if not self._schema_finalized:
            if self._using_fbx():
                if self._pbdot_mode() != "absorb_gw":
                    raise TimingModelError("DDR FBX chart requires DDRPBDOT absorb_gw")
                if self.DDRKINE.value:
                    raise TimingModelError("DDR FBX chart requires DDRKINE N")
            self._apply_conflicts()
            self._finalize_pk_schema()
            self._schema_finalized = True
            self._finalized_modes = self._mode_signature()
        self._check_pbdot_identity()
        self._install_pbdot_view()
        self.binary_instance = DDRmodel()
        self._register_ddr_derivs()
        Component.setup(self)

    def _active_binary_independents(self):
        names = ["A1", "TASC", "EPS1", "EPS2", "M2", "COSI"]
        if self._using_fbx():
            names.extend(
                name
                for _, name in sorted(self._fbx_mapping().items())
                if getattr(self, name).quantity is not None
            )
        else:
            names.append("PB")
        if self.A1DOT.quantity is not None:
            names.append("A1DOT")
        if self.DDRGEO.value:
            names.append("KOM")
        if not self.DDRPK.value:
            names.extend(["GGAMMA", "OMDOT"])
        if not self._using_fbx():
            if self._pbdot_mode() == "kinematic":
                names.append("XPBDOT")
            else:
                names.append("PBDOT")
        return [
            n
            for n in names
            if n in self.params and not isinstance(getattr(self, n), funcParameter)
        ]

    def _astrometry_in_B(self):
        if self._parent is None:
            return []
        if not (self.DDRGEO.value or self.DDRKINE.value):
            return []
        names = ["PX"]
        if "AstrometryEquatorial" in self._parent.components:
            names += ["RAJ", "DECJ", "PMRA", "PMDEC"]
        elif "AstrometryEcliptic" in self._parent.components:
            names += ["ELONG", "ELAT", "PMELONG", "PMELAT"]
        return names

    def _register_name(self, func, name):
        # PulsarBinary.setup() would register d_binary_delay_d_xxxx for every
        # inherited name, including inactive and derived ones. BinaryDDR
        # therefore installs only the active analytic columns itself.
        if name not in self.deriv_funcs:
            self.deriv_funcs[name] = [func]
        elif func not in self.deriv_funcs[name]:
            self.deriv_funcs[name] = self.deriv_funcs[name] + [func]

    def _register_ddr_derivs(self):
        for name in self._active_binary_independents():
            self._register_name(self.d_binary_delay_d_xxxx, name)
        if self._parent is None:
            return
        for name in self._astrometry_in_B():
            self._register_name(self.d_binary_delay_d_xxxx, name)
        self._register_upstream_time_args()

    def _register_upstream_time_args(self):
        """Register ``-B_t A_θ`` for upstream delay parameters.

        Called from ``setup()`` only. TimingModel builds delay components in
        ``DEFAULT_ORDER`` (dispersion before ``pulsar_system``). A fixed
        candidate list from ``parent.params`` covers names whose
        ``deriv_funcs`` may not be populated yet. Prefix families such as
        ``DMX_*`` that appear after this ``setup()`` are missed until a
        later ``setup()``.
        """
        if self._parent is None:
            return
        binary_only = set(self._active_binary_independents())
        candidates = []
        for dc in self._parent.DelayComponent_list:
            if dc is self:
                break
            candidates.extend(dc.deriv_funcs.keys())
        for name in (
            "RAJ",
            "DECJ",
            "ELONG",
            "ELAT",
            "PMRA",
            "PMDEC",
            "PMELONG",
            "PMELAT",
            "PX",
            "DM",
        ):
            if name in self._parent:
                candidates.append(name)
        for name in dict.fromkeys(candidates):
            if name in binary_only or name in _UNFREEZABLE:
                continue
            self._register_name(self.d_ddr_time_argument_correction, name)

    def validate(self):
        for p in ("EPS1", "EPS2"):
            pm = getattr(self, p)
            if pm.value is None:
                pm.value = 0
        if self.A1DOT.quantity is None:
            self.A1DOT.value = 0
        if self.A1.value is not None and self.A1.value < 0:
            raise TimingModelError("DDR A1 must be nonnegative")
        super().validate()
        self.check_required_params(["PB", "A1", "TASC", "M2", "COSI"])
        self._check_static_modes()
        self._refuse_orbwave()
        if self._parent is not None and self._parent.UNITS.value == "TCB":
            raise TimingModelError(
                "BINARY DDR is TDB-only (UNITS TCB is not supported)"
            )
        for name in _ZERO_PLACEHOLDERS:
            par = getattr(self, name, None)
            if par is None or par.quantity is None:
                continue
            if np.longdouble(par.value) != 0:
                raise TimingModelError(f"BINARY DDR does not support nonzero {name}")
            if not par.frozen:
                raise TimingModelError(f"BINARY DDR does not support unfrozen {name}")
        mode = self._pbdot_mode()
        using_fbx = self._using_fbx()
        if using_fbx and mode != "absorb_gw":
            raise TimingModelError("DDR FBX chart requires DDRPBDOT absorb_gw")
        if using_fbx and self.DDRKINE.value:
            raise TimingModelError("DDR FBX chart requires DDRKINE N")
        if self.DDRGEO.value and not using_fbx and not self.DDRKINE.value:
            raise TimingModelError("DDRGEO Y requires DDRKINE Y")
        if self.DDRGEO.value:
            if self.KOM.quantity is None:
                raise MissingParameter("DDR", "KOM", "KOM is required when DDRGEO Y")
        if self.DDRGEO.value or self.DDRKINE.value:
            if self._parent is None or not hasattr(self._parent, "PX"):
                raise MissingParameter("DDR", "PX", "DDRGEO/DDRKINE require PX")
            if self._parent.PX.value is None or self._parent.PX.value <= 0:
                raise TimingModelError("DDRGEO/DDRKINE require PX>0")
            _ = self.PMLONG_DDR.quantity
            _ = self.PMLAT_DDR.quantity
        if self.COSI.value is not None and abs(self.COSI.value) >= 1:
            raise TimingModelError("DDR |COSI| must be < 1")
        if self.COSI.value is not None and abs(self.COSI.value) > 1 - 1e-6:
            if not self.COSI.frozen:
                raise TimingModelError("Fitted DDR |COSI| must be <= 1 - 1e-6")
        e2 = np.longdouble(self.EPS1.value) ** 2 + np.longdouble(self.EPS2.value) ** 2
        if e2 > np.longdouble("0.99") ** 2:
            raise TimingModelError("DDR eccentricity must satisfy e <= 0.99")
        for name in _UNFREEZABLE:
            if hasattr(self, name) and not getattr(self, name).frozen:
                raise TimingModelError(f"{name} is frozen and not fittable in DDR v1")
        for name in _MODE_NAMES:
            if not getattr(self, name).frozen:
                raise TimingModelError(f"{name} is a static mode flag and not fittable")
        if (
            self.DDRGEO.value
            and self.A1DOT.quantity is not None
            and self.A1DOT.value != 0
        ):
            warnings.warn("Using A1DOT with DDRGEO Y is not advised.")
        need_mass = bool(self.DDRPK.value) or (not using_fbx and mode == "kinematic")
        if need_mass:
            if self.A1.value is None or self.A1.value <= 0:
                raise TimingModelError("DDR mass maps require A1>0")
            if self.M2.value is None or self.M2.value <= 0:
                raise TimingModelError("DDR mass maps require M2>0")
            try:
                mp, _s = pulsar_mass(
                    _TWO_PI / _ld_qty(self.PB.quantity, u.s),
                    _ld_qty(self.A1.quantity, u.lsec),
                    _ld_qty(self.M2.quantity, u.Msun),
                    np.longdouble(self.COSI.value),
                )
                if value(mp) <= 0:
                    raise TimingModelError("DDR inferred pulsar mass is not positive")
            except InvalidModelParameters as exc:
                raise TimingModelError(str(exc)) from exc
        self._validate_inherited_names()

    def _validate_inherited_names(self):
        """Refuse populated inherited parameters that DDR does not evaluate."""
        active = set(self._active_binary_independents())
        allowed = active | set(_MODE_NAMES) | set(_UNFREEZABLE)
        allowed |= {
            "GAMMA",
            "ECC",
            "OM",
            "SINI",
            "KIN",
            "KINIAU",
            "KOMIAU",
            "H3",
            "H4",
            "STIGMA",
        }
        allowed |= set(_ZERO_PLACEHOLDERS)
        for name in self.params:
            par = getattr(self, name)
            if isinstance(par, funcParameter) or name in allowed:
                continue
            if par.quantity is not None or not par.frozen:
                raise TimingModelError(
                    f"BINARY DDR does not evaluate populated parameter {name}"
                )

    def validate_toas(self, toas):
        super().validate_toas(toas)
        A = self._upstream_delay(toas)
        self.update_binary_object(toas, A)
        try:
            state = self.binary_instance.evaluate()
        except InvalidModelParameters as exc:
            raise TimingModelError(str(exc)) from exc
        self._validate_phase_slope_on_span(toas, state)
        self.last_inverse_timing_reference_error_max_s = np.max(
            np.atleast_1d(state.inverse_timing_ref_error_s)
        )

    def _validate_phase_slope_on_span(self, toas, state):
        """Refuse a non-positive ``λ̇`` anywhere on the TOA span, not only at TOAs.

        Samples endpoints, real critical points of the frequency
        polynomial, and a 65-point grid so a dip between grid nodes is
        not missed.
        """
        tdb = np.asarray(toas.table["tdbld"], dtype=np.longdouble)
        tmin = np.min(tdb)
        tmax = np.max(tdb)
        if tmin == tmax:
            return
        tasc = np.longdouble(self.TASC.value)
        if self._using_fbx():
            coeffs = self._fbx_coefficients()
        else:
            pb_s = np.longdouble(np.atleast_1d(state.pb_s).reshape(-1)[0])
            p = np.longdouble(np.atleast_1d(state.p).reshape(-1)[0])
            coeffs = [1 / pb_s, -p / (pb_s * pb_s)]
        dt_s = kep.phase_slope_sample_times(
            coeffs, (tmin - tasc) * _DAY, (tmax - tasc) * _DAY
        )
        try:
            kep.orbital_phase(dt_s, coeffs)
        except InvalidModelParameters as exc:
            raise TimingModelError(str(exc)) from exc

    def inverse_timing_reference_error(self, toas=None):
        """Leading circular inverse-timing discrepancy ``(x²/2)|λ̇ − n_ref|``.

        Uses the named prescription ``n = 2π FB0`` (or ``2π/PB``). Does not
        change the delay; reports the scale on the current evaluation times.
        """
        if toas is not None:
            A = self._upstream_delay(toas)
            self.update_binary_object(toas, A)
        state = self.binary_instance.evaluate()
        return np.asarray(state.inverse_timing_ref_error_s, dtype=np.longdouble)

    def _astrometry(self):
        if "AstrometryEquatorial" in self._parent.components:
            return self._parent.components["AstrometryEquatorial"]
        if "AstrometryEcliptic" in self._parent.components:
            return self._parent.components["AstrometryEcliptic"]
        raise TimingModelError("DDR requires an astrometry component")

    def _tgeo_triad(self):
        """``(I0, J0, n0)`` and ``(μ_I, μ_J)`` at ``TGEO`` (van Haasteren in prep. §5)."""
        state, _derivatives = self._analytic_astrometry(np.zeros((1, 3)))
        return {
            "I0": state["I0"],
            "J0": state["J0"],
            "n0": state["n0"],
            "mu_I": state["mu_I"],
            "mu_J": state["mu_J"],
        }

    def _galactic_lb(self):
        state, _derivatives = self._analytic_astrometry(np.zeros((1, 3)))
        return state["l_rad"], state["b_rad"]

    def _analytic_astrometry(self, obs_pos):
        """Analytic TGEO space-motion primitives and their parameter tangents."""
        dt_s = (
            np.longdouble(self.TGEO.value) - np.longdouble(self._parent.POSEPOCH.value)
        ) * _DAY
        if "AstrometryEquatorial" in self._parent.components:
            names = ("RAJ", "DECJ", "PMRA", "PMDEC")
            roles = ("lon", "lat", "pm_lon", "pm_lat")
            frame = ICRS()
        elif "AstrometryEcliptic" in self._parent.components:
            names = ("ELONG", "ELAT", "PMELONG", "PMELAT")
            roles = ("lon", "lat", "pm_lon", "pm_lat")
            frame = PulsarEcliptic(ecl=self._parent.ECL.value)
        else:
            raise TimingModelError("DDR requires an astrometry component")

        lon = getattr(self._parent, names[0]).quantity.to_value(u.rad)
        lat = getattr(self._parent, names[1]).quantity.to_value(u.rad)
        pm_lon = getattr(self._parent, names[2]).quantity.to_value(u.rad / u.s)
        pm_lat = getattr(self._parent, names[3]).quantity.to_value(u.rad / u.s)
        native_to_icrs = _frame_rotation(frame, ICRS())
        args = (
            np.longdouble(lon),
            np.longdouble(lat),
            np.longdouble(pm_lon),
            np.longdouble(pm_lat),
            dt_s,
            obs_pos,
            native_to_icrs,
        )
        primal = _analytic_space_motion(*args)
        rad_s_to_mas_yr = (u.rad / u.s).to(u.mas / u.yr)
        state = {
            key: (
                np.asarray(item.v, dtype=np.longdouble)
                if item.v.ndim
                else np.longdouble(item.v)
            )
            for key, item in primal.items()
        }
        state["pm_squared"] = (
            np.longdouble(primal["pm_squared_rad_s2"].v)
            * np.longdouble(rad_s_to_mas_yr) ** 2
        )
        state["pm"] = np.sqrt(state["pm_squared"])

        derivatives = {}
        for name, role in zip(names, roles):
            tangent_state = _analytic_space_motion(*args, seed=role)
            par = getattr(self._parent, name)
            canonical = u.rad if role in ("lon", "lat") else u.rad / u.s
            scale = np.longdouble((1 * par.units).to_value(canonical))
            derivatives[name] = {
                key.lower(): np.asarray(item.d, dtype=np.longdouble) * scale
                for key, item in tangent_state.items()
                if key not in ("I0", "J0", "n0", "pm_squared_rad_s2")
            }
            derivatives[name]["pm_squared"] = (
                np.longdouble(tangent_state["pm_squared_rad_s2"].d)
                * scale
                * np.longdouble(rad_s_to_mas_yr) ** 2
            )
        return state, derivatives

    def _obs_pos_au(self, toas):
        tbl = toas.table
        if "AstrometryEquatorial" in self._parent.components:
            pos = tbl["ssb_obs_pos"].quantity.to(u.AU)
            return np.asarray(pos.value, dtype=np.longdouble)
        obs_pos = SkyCoord(
            tbl["ssb_obs_pos"].quantity,
            representation_type="cartesian",
            frame="icrs",
        )
        xyz = obs_pos.transform_to(
            PulsarEcliptic(ecl=self._parent.ECL.value)
        ).cartesian.xyz.transpose()
        return np.asarray(xyz.to(u.AU).value, dtype=np.longdouble)

    def _kernel_param_dict(self, *, astrometry=True):
        def _val(name, unit=None):
            par = getattr(self, name)
            if par.quantity is None:
                return None
            if unit is None:
                return par.value
            return par.quantity.to(unit)

        updates = dict(
            A1=_val("A1", u.lsec),
            TASC=np.longdouble(self.TASC.value),
            EPS1=_val("EPS1"),
            EPS2=_val("EPS2"),
            M2=_val("M2", u.Msun),
            COSI=_val("COSI"),
            A1DOT=_val("A1DOT", u.lsec / u.s),
            DDRPK=bool(self.DDRPK.value),
            DDRPBDOT=self._pbdot_mode(),
            DDRGEO=bool(self.DDRGEO.value),
            DDRKINE=bool(self.DDRKINE.value),
            DDRR0=np.longdouble(self.DDRR0.value),
            DDRTHETA0=np.longdouble(self.DDRTHETA0.value),
            DDRRHO0=np.longdouble(self.DDRRHO0.value),
            DDRZ0=np.longdouble(self.DDRZ0.value),
            DDRZSUN=np.longdouble(self.DDRZSUN.value),
        )
        if self._using_fbx():
            updates["FB_COEFFS"] = self._fbx_coefficients()
        else:
            updates["PB"] = _val("PB", u.d)
        if self.TGEO.value is not None:
            updates["TGEO"] = np.longdouble(self.TGEO.value)
        if self.KOM.quantity is not None:
            updates["KOM"] = _val("KOM", u.deg)
        if not self.DDRPK.value:
            updates["GGAMMA"] = _val("GGAMMA", u.s)
            if self._is_writable("OMDOT"):
                updates["OMDOT"] = _val("OMDOT", u.deg / u.yr)
        if self._pbdot_mode() == "kinematic":
            updates["XPBDOT"] = (
                _val("XPBDOT") if self.XPBDOT.quantity is not None else 0
            )
        elif self._is_writable("PBDOT"):
            updates["PBDOT"] = _val("PBDOT")
        if self._parent is not None:
            if self.DDRGEO.value or self.DDRKINE.value:
                if hasattr(self._parent, "PX") and self._parent.PX.quantity is not None:
                    updates["PX"] = self._parent.PX.quantity
            if astrometry and self.DDRKINE.value:
                updates["PM"] = pmtot(self._parent)
                l_rad, b_rad = self._galactic_lb()
                updates["l_rad"] = l_rad
                updates["b_rad"] = b_rad
        return {k: v for k, v in updates.items() if v is not None}

    def _upstream_delay(self, toas):
        return self._parent.delay(
            toas, cutoff_component="BinaryDDR", include_last=False
        )

    def _upstream_d_delay_d_param(self, toas, param):
        par = getattr(self._parent, param)
        result = np.zeros(toas.ntoas) << (u.s / par.units)
        for dc in self._parent.DelayComponent_list:
            if dc is self:
                break
            for func in dc.deriv_funcs.get(param, []):
                result += func(toas, param, None).to(
                    result.unit, equivalencies=u.dimensionless_angles()
                )
        return result

    def update_binary_object(self, toas, acc_delay=None):
        updates = self._kernel_param_dict(astrometry=False)
        if toas is not None:
            tbl = toas.table
            if acc_delay is None:
                acc_delay = self._upstream_delay(toas)
            self.barycentric_time = tbl["tdbld"] * u.day - acc_delay
            updates["barycentric_toa"] = np.asarray(
                self.barycentric_time.to_value(u.day), dtype=np.longdouble
            )
        if self.DDRGEO.value or self.DDRKINE.value:
            obs_au = self._obs_pos_au(toas) if toas is not None else np.zeros((1, 3))
            state, derivatives = self._analytic_astrometry(obs_au)
            updates["PM"] = state["pm"]
            updates["PM_SQUARED"] = state["pm_squared"]
            updates["l_rad"] = state["l_rad"]
            updates["b_rad"] = state["b_rad"]
            updates["PRIMITIVE_DERIVS"] = derivatives
            if self.DDRGEO.value:
                updates["mu_I"] = state["mu_I"]
                updates["mu_J"] = state["mu_J"]
                updates["d_I_au"] = state["d_I_au"]
                updates["d_J_au"] = state["d_J_au"]
        self.binary_instance.update(**updates)

    def binarymodel_delay(self, toas, acc_delay=None):
        A = acc_delay if acc_delay is not None else self._upstream_delay(toas)
        self.update_binary_object(toas, A)
        self._batched_derivative_cache = None
        return self.binary_instance.delay() * u.s

    def _derivative_cache_key(self, toas, A, analytic_names):
        # float64 tokens, not longdouble bytes / repr: 80-bit x87 (CI) can
        # recompute the same upstream delay at a different ulp, which would
        # otherwise miss the batch cache and re-run Dual for every column.
        values = tuple(
            (name, float(np.float64(getattr(self._parent, name).value)))
            for name in analytic_names
        )
        a_values = np.asarray(A.to_value(u.s), dtype=np.float64)
        tdb = np.asarray(toas.table["tdbld"], dtype=np.float64)
        return (tdb.tobytes(), values, a_values.shape, a_values.tobytes())

    def _ensure_eval_cache(self, toas, A):
        analytic_names = list(
            dict.fromkeys(self._active_binary_independents() + self._astrometry_in_B())
        )
        cache_key = self._derivative_cache_key(toas, A, analytic_names)
        if (
            self._batched_derivative_cache is None
            or self._batched_derivative_cache[0] != cache_key
        ):
            self.update_binary_object(toas, A)
            batch = self.binary_instance.d_delay_d_pars(analytic_names)
            bt = np.asarray(self.binary_instance.d_delay_d_tcorr(), dtype=np.longdouble)
            self._batched_derivative_cache = (
                cache_key,
                dict(zip(analytic_names, batch)),
                bt,
            )
        return self._batched_derivative_cache

    def d_binary_delay_d_xxxx(self, toas, param, acc_delay=None):
        """``∂B/∂θ`` at fixed ``t_corr``. Dual for binary independents and PX."""
        A = self._upstream_delay(toas)
        par = getattr(self._parent, param)
        cache = self._ensure_eval_cache(toas, A)
        if param in cache[1]:
            d = cache[1][param]
            return (np.asarray(d, dtype=np.longdouble) * u.s) / par.units
        raise TimingModelError(f"No analytic DDR derivative registered for {param}")

    def d_ddr_time_argument_correction(self, toas, param, acc_delay=None):
        A = self._upstream_delay(toas)
        bt = self._ensure_eval_cache(toas, A)[2]
        a_theta = self._upstream_d_delay_d_param(toas, param)
        return -bt * a_theta

    def _prepare_epoch_kernel(self):
        """Refresh the kernel without leftover per-TOA geometry arrays."""
        inst = self.binary_instance
        inst.d_I_au = None
        inst.d_J_au = None
        inst.t = None
        inst.update(**self._kernel_param_dict())
        return inst

    def t0_from_tasc(self):
        """First periastron after TASC from the active phase polynomial."""
        h = np.longdouble(self.EPS1.value)
        k = np.longdouble(self.EPS2.value)
        tasc = np.longdouble(self.TASC.value)
        if h == 0 and k == 0:
            return tasc * u.d
        omega = np.mod(np.arctan2(h, k), _TWO_PI)
        if self._using_fbx():
            coeffs = self._fbx_coefficients()
        else:
            pb_s = np.longdouble(self.PB.quantity.to_value(u.s))
            inst = self._prepare_epoch_kernel()
            p = np.longdouble(inst.period_derivative())
            # Analytics §11.4 quadratic on the PB chart.
            psi = omega / _TWO_PI
            disc = 1 - 2 * p * psi
            if disc >= 0 and (1 + np.sqrt(disc)) != 0:
                return (tasc + (2 * (pb_s / _DAY) * psi) / (1 + np.sqrt(disc))) * u.d
            coeffs = np.asarray([1 / pb_s, -p / pb_s**2], dtype=np.longdouble)
        period_s = 1 / coeffs[0]
        hi_s = period_s
        for _ in range(32):
            try:
                dt_s = kep.solve_phase_offset(coeffs, omega, 0, hi_s)
                break
            except InvalidModelParameters as exc:
                if "not bracketed" not in str(exc):
                    raise TimingModelError(str(exc)) from exc
                hi_s *= np.longdouble("2")
        else:
            raise TimingModelError("DDR phase root is not bracketed near TASC")
        return (tasc + dt_s / _DAY) * u.d

    def om_dd(self):
        """``ω_star + δ(T0)`` in degrees (analytics §11.4)."""
        t0 = np.longdouble(self.t0_from_tasc().to_value(u.d))
        inst = self._prepare_epoch_kernel()
        st = inst.evaluate(np.array([t0], dtype=np.longdouble))
        omega = np.arctan2(
            np.longdouble(self.EPS1.value), np.longdouble(self.EPS2.value)
        )
        return (np.mod(omega + st.delta[0], _TWO_PI) * u.rad).to(u.deg)

    def mp_from_mass_function(self):
        mp, _s = pulsar_mass(
            _TWO_PI / _ld_qty(self.PB.quantity, u.s),
            _ld_qty(self.A1.quantity, u.lsec),
            _ld_qty(self.M2.quantity, u.Msun),
            np.longdouble(self.COSI.value),
        )
        return value(mp) * u.Msun

    def print_par(self, format="pint"):
        result = super().print_par(format=format)
        try:
            t0 = self.t0_from_tasc()
            result += f"# T0 {np.longdouble(t0.to_value(u.d)):.18f}\n"
        except (TimingModelError, InvalidModelParameters):
            pass
        return result

    def change_binary_epoch(self, new_epoch):
        """Move ``TASC`` to a nearby ``λ+δ = 2π m`` root (analytics §11.3).

        Permitted only for phenomenological ``p=0`` (``DDRPK N``,
        ``DDRKINE N``, ``DDRPBDOT absorb_gw``). ``TGEO`` is unchanged.
        """
        if self.DDRPK.value:
            raise TimingModelError(
                "DDR change_binary_epoch is not supported for DDRPK Y"
            )
        if self.DDRKINE.value:
            raise TimingModelError(
                "DDR change_binary_epoch is not supported for DDRKINE Y"
            )
        if self._pbdot_mode() == "kinematic":
            raise TimingModelError(
                "DDR change_binary_epoch is not supported for DDRPBDOT kinematic"
            )
        if self._using_fbx():
            mapping = self._fbx_mapping()
            nonzero_rates = [
                name
                for index, name in mapping.items()
                if index >= 1
                and getattr(self, name).quantity is not None
                and np.longdouble(getattr(self, name).value) != 0
            ]
            if nonzero_rates:
                raise TimingModelError(
                    "DDR change_binary_epoch is not supported for nonzero FB derivatives"
                )
            p = np.longdouble(0)
        else:
            p = np.longdouble(0 if self.PBDOT.quantity is None else self.PBDOT.value)
        if p != 0:
            raise TimingModelError(
                "DDR change_binary_epoch is not supported for nonzero PBDOT"
            )
        new_epoch = parse_time(new_epoch, scale="tdb", precision=9)
        t_req = np.longdouble(new_epoch.tdb.mjd_long)
        inst = self._prepare_epoch_kernel()
        st0 = inst.evaluate(np.array([t_req], dtype=np.longdouble))
        m_orb = np.round((st0.lam[0] + st0.delta[0]) / _TWO_PI)
        t = t_req
        if self._using_fbx():
            pb = (
                np.longdouble(1)
                / np.longdouble(self.FB0.quantity.to_value(u.Hz))
                / _DAY
            )
        else:
            pb = np.longdouble(self.PB.quantity.to_value(u.d))
        for _ in range(30):
            st = inst.evaluate(np.array([t], dtype=np.longdouble), tangent_par="TCORR")
            f = st.lam[0] + st.delta[0] - _TWO_PI * m_orb
            fp = st.d_lam[0] + st.d_delta[0]
            if fp == 0:
                break
            t_next = t - f / fp
            if abs(t_next - t) < np.longdouble("1e-16"):
                t = t_next
                break
            t = t_next
        if abs(t - t_req) > 5 * pb:
            raise TimingModelError(
                "DDR change_binary_epoch failed to find a nearby λ+δ = 2π m root"
            )
        st = inst.evaluate(np.array([t], dtype=np.longdouble))
        delta_star = st.delta[0]
        h = np.longdouble(self.EPS1.value)
        k = np.longdouble(self.EPS2.value)
        cd = np.cos(delta_star)
        sd = np.sin(delta_star)
        self.EPS1.value = h * cd + k * sd
        self.EPS2.value = k * cd - h * sd
        if self.A1DOT.quantity is not None and self.A1DOT.value != 0:
            dt_s = (t - np.longdouble(self.TASC.value)) * _DAY
            self.A1.value = (
                np.longdouble(self.A1.value)
                + np.longdouble(self.A1DOT.quantity.to_value(u.lsec / u.s)) * dt_s
            )
        if self._is_writable("OMDOT") and self.OMDOT.quantity is not None:
            # At p=0 (or FB0-only), f0 and therefore OMDOT = κ 2π f0 are unchanged.
            assert np.longdouble(pb) > 0
        self.TASC.quantity = t
        return t
