"""Internal Dual arithmetic and Laplace-Lagrange Kepler solve for DDR.

Canonical units: radians and dimensionless. See van Haasteren in prep. §§2-3.

Kepler roots are solved in primal arithmetic; Dual values are injected
through the implicit differential of that root. Do not differentiate the
Newton/bisection predicates.
"""

from __future__ import annotations

import numpy as np

from pint.exceptions import InvalidModelParameters

_LD = np.longdouble
_TWO_PI = _LD(2) * np.pi
_RESIDUAL_TOL = _LD("1e-16")
_MAX_ITER = 64
_EMAX = _LD("0.99")


def as_ld(x):
    """Return ``x`` as an ``np.longdouble`` ndarray (0-d for scalars)."""
    if isinstance(x, Dual):
        return x.v
    return np.asarray(x, dtype=_LD)


class Dual:
    """Value plus directional derivative, broadcasting like ``ndarray``."""

    __slots__ = ("v", "d")
    __array_ufunc__ = None
    __array_priority__ = 1000

    def __init__(self, v, d=None):
        self.v = np.asarray(v, dtype=_LD)
        if d is None:
            self.d = np.zeros_like(self.v, dtype=_LD)
        else:
            self.d = np.asarray(d, dtype=_LD)
        if self.d.ndim > self.v.ndim:
            shape = np.broadcast_shapes(self.d.shape, self.v.shape)
            self.d = np.broadcast_to(self.d, shape)
        else:
            self.v, self.d = np.broadcast_arrays(self.v, self.d)
            self.v = np.asarray(self.v, dtype=_LD)
        self.d = np.asarray(self.d, dtype=_LD)

    def _coerce(self, other):
        if isinstance(other, Dual):
            return other
        return Dual(other, np.zeros_like(as_ld(other), dtype=_LD))

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __add__(self, other):
        other = self._coerce(other)
        return Dual(self.v + other.v, self.d + other.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        return Dual(self.v - other.v, self.d - other.d)

    def __rsub__(self, other):
        other = self._coerce(other)
        return Dual(other.v - self.v, other.d - self.d)

    def __mul__(self, other):
        other = self._coerce(other)
        return Dual(self.v * other.v, self.d * other.v + self.v * other.d)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        return Dual(
            self.v / other.v,
            (self.d * other.v - self.v * other.d) / (other.v * other.v),
        )

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, p):
        p = _LD(p)
        if p == 0:
            return Dual(np.ones_like(self.v), np.zeros_like(self.d))
        return Dual(self.v**p, p * self.v ** (p - _LD(1)) * self.d)


def dual(x, dx=None):
    if isinstance(x, Dual):
        if dx is None:
            return x
        return Dual(x.v, dx)
    return Dual(x, dx)


def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.v), np.cos(x.v) * x.d)
    return np.sin(as_ld(x))


def cos(x):
    if isinstance(x, Dual):
        return Dual(np.cos(x.v), -np.sin(x.v) * x.d)
    return np.cos(as_ld(x))


def sqrt(x):
    if isinstance(x, Dual):
        s = np.sqrt(x.v)
        deriv = np.zeros_like(x.d, dtype=_LD)
        nonzero = s != 0
        np.divide(x.d, _LD(2) * s, out=deriv, where=nonzero)
        return Dual(s, deriv)
    return np.sqrt(as_ld(x))


def log(x):
    if isinstance(x, Dual):
        return Dual(np.log(x.v), x.d / x.v)
    return np.log(as_ld(x))


def atan2(y, x):
    if isinstance(y, Dual) or isinstance(x, Dual):
        y = dual(y)
        x = dual(x)
        n = x.v * x.v + y.v * y.v
        return Dual(np.arctan2(y.v, x.v), (x.v * y.d - y.v * x.d) / n)
    return np.arctan2(as_ld(y), as_ld(x))


def hypot(x, y):
    return sqrt(dual(x) * dual(x) + dual(y) * dual(y))


def value(x):
    return x.v if isinstance(x, Dual) else as_ld(x)


def _primal_scale(x):
    """Positive Horner scale from the primal magnitude of ``x``."""
    xv = np.abs(np.atleast_1d(as_ld(x)))
    peak = np.max(xv)
    if not np.isfinite(peak) or peak <= 0:
        return _LD(1)
    return _LD(peak)


def _scaled_coeffs(coeffs, scale):
    """Return ``a_j = f_j s^j`` so Horner in ``u = dt/s`` is Dual-safe."""
    scaled = []
    power = _LD(1)
    for coeff in coeffs:
        scaled.append(dual(coeff) * power)
        power = power * scale
    return scaled


def _taylor_horner(x, coeffs, deriv_order=0):
    """Dual-safe ``d^n/dx^n`` of ``sum c_i x^i / i!``."""
    coeffs = [dual(c) for c in coeffs]
    x = dual(x)
    if deriv_order < 0:
        raise ValueError("deriv_order must be non-negative")
    if deriv_order >= len(coeffs):
        return dual(x) * 0
    der = coeffs[deriv_order:]
    result = dual(0)
    fact = _LD(len(der))
    for coeff in der[::-1]:
        result = result * x / fact + coeff
        fact -= _LD(1)
    return result


def phase_slope_sample_times(coeffs, lo_s, hi_s, *, grid_size=65):
    """Times that must be checked for ``λ̇ > 0`` on ``[lo_s, hi_s]``.

    Includes the endpoints, real critical points of the anomalistic
    frequency (roots of ``df/dt``), and a uniform grid. The critical
    points catch a dip between grid nodes; the grid covers a missed
    companion-matrix root at high order.
    """
    lo = _LD(lo_s)
    hi = _LD(hi_s)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise InvalidModelParameters("DDR orbital phase span must be finite")
    if lo == hi:
        return np.asarray([lo], dtype=_LD)
    if hi < lo:
        lo, hi = hi, lo
    samples = [lo, hi]
    primal = [_LD(np.asarray(value(c)).reshape(-1)[0]) for c in coeffs]
    if len(primal) >= 2:
        scale = max(abs(lo), abs(hi), _LD(1))
        power = scale
        fact = _LD(1)
        monomials = []
        for j in range(1, len(primal)):
            monomials.append(float(primal[j] * power / fact))
            power = power * scale
            fact = fact * _LD(j)
        if any(term != 0 for term in monomials):
            for root in np.polynomial.polynomial.polyroots(
                np.asarray(monomials, dtype=np.float64)
            ):
                if abs(np.imag(root)) > 1e-8:
                    continue
                trial = _LD(np.real(root)) * scale
                if lo < trial < hi:
                    samples.append(trial)
    grid = np.linspace(lo, hi, int(grid_size), dtype=_LD)
    return np.unique(np.concatenate((np.asarray(samples, dtype=_LD), grid)))


def orbital_phase(dt_s, coeffs, *, check_slope=True):
    """Taylor orbital phase and its first two time derivatives.

    ``coeffs[j]`` is the anomalistic-frequency derivative ``FBj`` in
    ``s**-(j+1)`` at ``TASC``. Arguments may be :class:`Dual`. Evaluation
    uses scaled Horner in ``Δt / max|Δt|``, valid on both sides of
    ``TASC`` through high FB order.
    """
    dt_s = dual(dt_s)
    coeffs = [dual(c) for c in coeffs]
    if not coeffs:
        raise InvalidModelParameters("DDR orbital phase requires FB0")

    scale = _primal_scale(dt_s)
    u = dt_s / scale
    scaled = _scaled_coeffs(coeffs, scale)
    frequency = _taylor_horner(u, scaled, 0)
    phase_cycles = scale * _taylor_horner(u, [dual(0)] + scaled, 0)
    frequency_dot = _taylor_horner(u, scaled, 1) / scale

    if check_slope and np.any(value(frequency) <= 0):
        raise InvalidModelParameters(
            "DDR orbital phase slope is not positive on the evaluation span"
        )
    return _TWO_PI * phase_cycles, _TWO_PI * frequency, _TWO_PI * frequency_dot


def taylor_shift(coeffs, delta_s):
    """Re-reference orbital-frequency Taylor coefficients by ``delta_s``.

    Dual-safe scaled Horner. Primal inputs return an ``np.longdouble``
    array; Dual inputs return a list of :class:`Dual`.
    """
    coeffs = list(coeffs)
    if not coeffs:
        return np.asarray([], dtype=_LD)
    any_dual = isinstance(delta_s, Dual) or any(isinstance(c, Dual) for c in coeffs)
    scale = _primal_scale(delta_s)
    u = dual(delta_s) / scale
    scaled = _scaled_coeffs(coeffs, scale)
    inv_power = _LD(1)
    shifted = []
    for j in range(len(coeffs)):
        shifted.append(_taylor_horner(u, scaled[j:], 0) * inv_power)
        inv_power = inv_power / scale
    if any_dual:
        return shifted
    return np.asarray(
        [_LD(np.asarray(value(c)).reshape(-1)[0]) for c in shifted], dtype=_LD
    )


def solve_phase_offset(coeffs, target_rad, lo_s, hi_s):
    """Solve the monotone phase polynomial for a time offset."""
    coeffs = np.asarray(coeffs, dtype=_LD)
    target = _LD(target_rad)
    lo = _LD(lo_s)
    hi = _LD(hi_s)

    def phase_and_slope(dt):
        lam, slope, _ = orbital_phase(_LD(dt), coeffs)
        return _LD(value(lam) - target), _LD(value(slope))

    flo, _ = phase_and_slope(lo)
    fhi, _ = phase_and_slope(hi)
    if flo > 0 or fhi < 0:
        raise InvalidModelParameters("DDR phase root is not bracketed")
    x = (lo + hi) / _LD(2)
    for _ in range(128):
        fx, slope = phase_and_slope(x)
        if abs(fx) <= _LD("1e-18"):
            return x
        if fx < 0:
            lo = x
        else:
            hi = x
        trial = x - fx / slope
        x = trial if lo < trial < hi else (lo + hi) / _LD(2)
        if abs(hi - lo) <= _LD("1e-15"):
            return x
    raise InvalidModelParameters("DDR phase-root solver did not converge")


def mean_longitude(dt_s, pb_s, p):
    """Quadratic anomalistic mean longitude λ at constant parameter P_B.

    ``dt_s`` is t − TASC in seconds, ``pb_s`` the anomalistic period in
    seconds, ``p`` the single dimensionless Ṗ_b. Arguments may be Dual.
    """
    pb_s = dual(pb_s)
    p = dual(p)
    lam, _lamdot, _lamddot = orbital_phase(dt_s, [1 / pb_s, -p / (pb_s * pb_s)])
    return lam


def reduce_longitude(lam):
    """Return ``(lam_red, N_orb)`` with ``lam = lam_red + 2π N_orb``.

    Reduction uses the primal value; Dual tangents pass through unchanged
    (dλ_red = dλ).
    """
    lam = dual(lam)
    n_orb = np.round(lam.v / _TWO_PI)
    return lam - n_orb * _TWO_PI, n_orb


def _kepler_residual(F, h, k, lam):
    return F - k * np.sin(F) + h * np.cos(F) - lam


def solve_F(lam, h, k, residual_tol=_RESIDUAL_TOL, max_iter=_MAX_ITER):
    """Solve ``F - k sin F + h cos F = λ`` with safeguarded Newton.

    Primal only. Dual arguments are accepted; only ``.v`` is solved.
    Returns primal ``F, D, c_e, s_e`` at the **reduced** longitude.

    Exit is on equation residual ``|R| ≤ residual_tol`` in long double.
    """
    lam_v = as_ld(lam)
    h_v = as_ld(h)
    k_v = as_ld(k)
    e2 = h_v * h_v + k_v * k_v
    if np.any(e2 > _EMAX * _EMAX):
        raise InvalidModelParameters("DDR eccentricity exceeds e_max = 0.99")
    lam_red, _ = reduce_longitude(lam_v)
    lam_red_v = as_ld(lam_red)
    lo = lam_red_v - _LD(1)
    hi = lam_red_v + _LD(1)
    F = lam_red_v + k_v * np.sin(lam_red_v) - h_v * np.cos(lam_red_v)
    F = np.minimum(np.maximum(F, lo), hi)

    residual_tol = _LD(residual_tol)
    for _ in range(int(max_iter)):
        R = _kepler_residual(F, h_v, k_v, lam_red_v)
        if np.all(np.abs(R) <= residual_tol):
            break
        D = _LD(1) - k_v * np.cos(F) - h_v * np.sin(F)
        F_try = F - R / D
        good = np.isfinite(F_try) & (F_try >= lo) & (F_try <= hi)
        R_lo = _kepler_residual(lo, h_v, k_v, lam_red_v)
        go_hi = R_lo * R > 0
        lo_n = np.where(go_hi, F, lo)
        hi_n = np.where(go_hi, hi, F)
        F_bisect = _LD("0.5") * (lo_n + hi_n)
        F = np.where(good, F_try, F_bisect)
        lo = np.where(good, lo, lo_n)
        hi = np.where(good, hi, hi_n)
    else:
        R = _kepler_residual(F, h_v, k_v, lam_red_v)
        if np.any(np.abs(R) > residual_tol):
            raise InvalidModelParameters(
                "DDR Kepler solver failed to meet residual tolerance"
            )

    D = _LD(1) - k_v * np.cos(F) - h_v * np.sin(F)
    c_e = k_v * np.cos(F) + h_v * np.sin(F)
    s_e = k_v * np.sin(F) - h_v * np.cos(F)
    return F, D, c_e, s_e


def kepler_with_implicit(lam, h, k, residual_tol=_RESIDUAL_TOL):
    """Kepler root with Dual tangents from the implicit differential of F.

    Does not differentiate Newton/bisection predicates.
    """
    lam = dual(lam)
    h = dual(h)
    k = dual(k)
    F_v, D_v, _, _ = solve_F(lam.v, h.v, k.v, residual_tol=residual_tol)
    dF = (lam.d + np.sin(F_v) * k.d - np.cos(F_v) * h.d) / D_v
    F = Dual(F_v, dF)
    D = _LD(1) - k * cos(F) - h * sin(F)
    c_e = k * cos(F) + h * sin(F)
    s_e = k * sin(F) - h * cos(F)
    return F, D, c_e, s_e


def static_XY(F, h, k, dr=_LD(0), dth=_LD(0)):
    """Regular static projections ``X0, Y0`` and frozen-element F-derivatives.

    Dual-safe. v1 delay uses ``dr = dth = 0``. Deformed formulae are kept
    for tests against polar DD, not for the v1 delay.
    """
    F = dual(F)
    h = dual(h)
    k = dual(k)
    dr = dual(dr)
    dth = dual(dth)
    E2 = h * h + k * k
    one = _LD(1)
    eta = sqrt(one - (one + dth) ** 2 * E2)
    b = (one + dth) ** 2 / (one + eta)
    sF = sin(F)
    cF = cos(F)
    Y0 = (one - b * k * k) * sF + b * h * k * cF - (one + dr) * h
    X0 = (one - b * h * h) * cF + b * h * k * sF - (one + dr) * k
    Y0p = (one - b * k * k) * cF - b * h * k * sF
    Y0pp = -(one - b * k * k) * sF - b * h * k * cF
    X0p = -(one - b * h * h) * sF + b * h * k * cF
    X0pp = -(one - b * h * h) * cF - b * h * k * sF
    return X0, Y0, X0p, Y0p, X0pp, Y0pp


def polar_XY(F, h, k):
    """Independent polar DD projections (undeformed) for tests only.

    Uses ``atan2``; not part of the delay kernel.
    """
    F = dual(F)
    h = dual(h)
    k = dual(k)
    e = sqrt(h * h + k * k)
    om = atan2(h, k)
    u = F - om
    sq = sqrt(_LD(1) - e * e)
    cu = cos(u)
    su = sin(u)
    som = sin(om)
    com = cos(om)
    Y = som * (cu - e) + sq * com * su
    X = com * (cu - e) - sq * som * su
    return X, Y


def circular_origin_dXY_dhk(lam):
    """Exact static circular-origin Jacobians (van Haasteren in prep. §3.1)."""
    lam = as_ld(lam)
    c2 = np.cos(_LD(2) * lam)
    s2 = np.sin(_LD(2) * lam)
    dY_dh = -_LD("1.5") - _LD("0.5") * c2
    dY_dk = _LD("0.5") * s2
    dX_dh = _LD("0.5") * s2
    dX_dk = -_LD("1.5") + _LD("0.5") * c2
    return dX_dh, dX_dk, dY_dh, dY_dk
