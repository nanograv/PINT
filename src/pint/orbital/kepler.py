"""Functions for working with Keplerian orbits

All times are in days, distances in light-seconds, and masses in solar masses.
"""
import collections

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import newton

# FIXME: can I import this from somewhere?
G = 36768.59290949113  # Based on standard gravitational parameter


def true_from_eccentric(e, eccentric_anomaly):
    """Compute the true anomaly from the eccentric anomaly.

    Parameters
    ----------
    e : float
        the eccentricity
    eccentric_anomaly : float
        the eccentric anomaly

    Returns
    -------
    true_anomaly : float
        the true anomaly
    true_anomaly_de : float
        derivative of true anomaly with respect to e
    true_anomaly_prime : float
        derivative of true anomaly with respect to eccentric anomaly
    """
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2),
    )
    true_anomaly_de = np.sin(eccentric_anomaly) / (
        np.sqrt(1 - e**2) * (1 - e * np.cos(eccentric_anomaly))
    )
    true_anomaly_prime = np.sqrt(1 - e**2) / (1 - e * np.cos(eccentric_anomaly))
    return true_anomaly, true_anomaly_de, true_anomaly_prime


def eccentric_from_mean(e, mean_anomaly):
    """Compute the eccentric anomaly from the mean anomaly.

    Parameters
    ----------
    e : float
        the eccentricity
    mean_anomaly : float
        the mean anomaly

    Returns
    -------
    eccentric_anomaly : float
        the true anomaly
    derivatives : float
        pair of derivatives with respect to the two inputs
    """
    eccentric_anomaly = newton(
        lambda E: E - e * np.sin(E) - mean_anomaly,
        mean_anomaly,
        lambda E: 1 - e * np.cos(E),
    )
    eccentric_anomaly_de = np.sin(eccentric_anomaly) / (
        1 - e * np.cos(eccentric_anomaly)
    )
    eccentric_anomaly_prime = (1 - e * np.cos(eccentric_anomaly)) ** (-1)
    return eccentric_anomaly, [eccentric_anomaly_de, eccentric_anomaly_prime]


def mass(a, pb):
    """Compute the mass of a particle in a Kepler orbit.

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    return 4 * np.pi**2 * a**3 * pb ** (-2) / G


def mass_partials(a, pb):
    """Compute the mass of a particle in a Kepler orbit, with partials.

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    m = mass(a, pb)
    return m, np.array([3 * m / a, -2 * m / pb])


def btx_parameters(asini, pb, eps1, eps2, tasc):
    """Attempt to convert parameters from ELL1 to BTX."""
    e = np.hypot(eps1, eps2)
    om = np.arctan2(eps1, eps2)
    true_anomaly = -om  # True anomaly at the ascending node
    eccentric_anomaly = np.arctan2(
        np.sqrt(1 - e**2) * np.sin(true_anomaly), e + np.cos(true_anomaly)
    )
    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)
    t0 = tasc - mean_anomaly * pb / (2 * np.pi)
    return asini, pb, e, om, t0


class Kepler2DParameters(
    collections.namedtuple("Kepler2DParameters", "a pb eps1 eps2 t0")
):
    """Parameters to describe a one-object 2D Keplerian orbit.

    Parameters
    ----------
    a : float
        semi-major axis
    pb : float
        binary period
    eps1 : float
    eps2 : float
        eccentricity parameters
    t0 : float
        time of the ascending node
    """

    __slots__ = ()


def kepler_2d(params, t):
    """Position and velocity of a particle in a Kepler orbit.

    The orbit has semi-major axis a, period pb, and eccentricity
    parametrized by eps1=e*sin(om) and eps2=e*cos(om), and the
    particle is on the x axis at time t0, while the values
    are computed for time t.

    The function returns a pair (xv, p), where xv is of length
    four and consists of (x,y,v_x,v_y), and p is of shape (4,5)
    and cell (i,j) contains the the partial derivative of the
    ith element of xv with respect to the jth orbital parameter.

    The zero of time is when the particle is on the positive x
    axis. (Which will be the ascending node in a three-dimensional
    model.)
    """
    a = params.a
    pb = params.pb
    eps1 = params.eps1
    eps2 = params.eps2
    t = t - params.t0
    # if eps1 == 0 and eps2 == 0:
    #     eps1 = 1e-50
    e = np.hypot(eps1, eps2)
    if e == 0:
        d_e = np.array([0, 0, 0, 0, 0, 0])
    else:
        d_e = np.array([0, 0, eps1 / e, eps2 / e, 0, 0])
    # return e, d_e

    om = np.arctan2(eps1, eps2)
    if e == 0:
        d_om = np.array([0, 0, 0, 0, 0, 0])
    else:
        d_om = np.array([0, 0, eps2 / e**2, -eps1 / e**2, 0, 0])
    # return om, d_om

    true_anomaly_0 = -om
    d_true_anomaly_0 = -d_om

    eccentric_anomaly_0 = np.arctan2(
        np.sqrt(1 - e**2) * np.sin(true_anomaly_0), e + np.cos(true_anomaly_0)
    )
    d_eccentric_anomaly_0 = (
        d_e
        * (
            -(1 + e * np.cos(true_anomaly_0))
            * np.sin(true_anomaly_0)
            / (np.sqrt(1 - e**2) * (e * np.cos(true_anomaly_0) + 1) ** 2)
        )
        + d_true_anomaly_0
        * (np.sqrt(1 - e**2) * (1 + e * np.cos(true_anomaly_0)))
        / (e * np.cos(true_anomaly_0) + 1) ** 2
    )

    mean_anomaly_0 = eccentric_anomaly_0 - e * np.sin(eccentric_anomaly_0)
    d_mean_anomaly_0 = (
        d_eccentric_anomaly_0
        - d_e * np.sin(eccentric_anomaly_0)
        - e * np.cos(eccentric_anomaly_0) * d_eccentric_anomaly_0
    )
    # return mean_anomaly_0, d_mean_anomaly_0

    mean_anomaly = 2 * np.pi * t / pb + mean_anomaly_0
    d_mean_anomaly = (
        2 * np.pi * np.array([0, -t / pb**2, 0, 0, -(pb ** (-1)), pb ** (-1)])
        + d_mean_anomaly_0
    )
    # return mean_anomaly, d_mean_anomaly

    mean_anomaly_dot = 2 * np.pi / pb
    d_mean_anomaly_dot = 2 * np.pi * np.array([0, -(pb ** (-2)), 0, 0, 0, 0])
    # return ([mean_anomaly, mean_anomaly_dot],
    #        [d_mean_anomaly, d_mean_anomaly_dot])

    (
        eccentric_anomaly,
        (eccentric_anomaly_de, eccentric_anomaly_prime),
    ) = eccentric_from_mean(e, mean_anomaly)
    eccentric_anomaly_dot = eccentric_anomaly_prime * mean_anomaly_dot

    d_eccentric_anomaly = (
        eccentric_anomaly_de * d_e + eccentric_anomaly_prime * d_mean_anomaly
    )
    d_eccentric_anomaly_prime = (
        np.cos(eccentric_anomaly) / (1 - e * np.cos(eccentric_anomaly)) ** 2 * d_e
        - e
        * np.sin(eccentric_anomaly)
        / (1 - e * np.cos(eccentric_anomaly)) ** 2
        * d_eccentric_anomaly
    )
    d_eccentric_anomaly_dot = (
        d_eccentric_anomaly_prime * mean_anomaly_dot
        + eccentric_anomaly_prime * d_mean_anomaly_dot
    )
    # return eccentric_anomaly, d_eccentric_anomaly
    # return eccentric_anomaly_prime, d_eccentric_anomaly_prime
    # return eccentric_anomaly_dot, d_eccentric_anomaly_dot

    true_anomaly, true_anomaly_de, true_anomaly_prime = true_from_eccentric(
        e, eccentric_anomaly
    )
    true_anomaly_dot = true_anomaly_prime * eccentric_anomaly_dot

    d_true_anomaly = true_anomaly_de * d_e + true_anomaly_prime * d_eccentric_anomaly
    d_true_anomaly_prime = (
        (np.cos(eccentric_anomaly) - e)
        / (np.sqrt(1 - e**2) * (1 - e * np.cos(eccentric_anomaly)) ** 2)
    ) * d_e - e * np.sqrt(1 - e**2) * np.sin(eccentric_anomaly) / (
        1 - e * np.cos(eccentric_anomaly)
    ) ** 2 * d_eccentric_anomaly
    d_true_anomaly_dot = (
        d_true_anomaly_prime * eccentric_anomaly_dot
        + true_anomaly_prime * d_eccentric_anomaly_dot
    )
    # return true_anomaly, d_true_anomaly
    # return true_anomaly_prime, d_true_anomaly_prime
    # return true_anomaly_dot, d_true_anomaly_dot

    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    r_prime = (
        a
        * e
        * (1 - e**2)
        * np.sin(true_anomaly)
        / (1 + e * np.cos(true_anomaly)) ** 2
    )
    r_dot = r_prime * true_anomaly_dot
    d_a = np.array([1, 0, 0, 0, 0, 0])
    d_r = (
        d_a * r / a
        - a
        * d_e
        * ((1 + e**2) * np.cos(true_anomaly) + 2 * e)
        / (1 + e * np.cos(true_anomaly)) ** 2
        + r_prime * d_true_anomaly
    )
    d_r_prime = (
        d_a * r_prime / a
        + a
        * d_e
        * (-e * (1 + e**2) * np.cos(true_anomaly) - 3 * e**2 + 1)
        * np.sin(true_anomaly)
        / (1 + e * np.cos(true_anomaly)) ** 3
        + a
        * e
        * (1 - e**2)
        * (e * (np.sin(true_anomaly) ** 2 + 1) + np.cos(true_anomaly))
        / (1 + e * np.cos(true_anomaly)) ** 3
        * d_true_anomaly
    )
    d_r_dot = d_r_prime * true_anomaly_dot + r_prime * d_true_anomaly_dot
    # return r, d_r
    # return r_prime, d_r_prime
    # return r_dot, d_r_dot

    xyv = np.zeros(4)
    xyv[0] = r * np.cos(true_anomaly + om)
    xyv[1] = r * np.sin(true_anomaly + om)
    xyv[2] = r_dot * np.cos(true_anomaly + om) - r * true_anomaly_dot * np.sin(
        true_anomaly + om
    )
    xyv[3] = r_dot * np.sin(true_anomaly + om) + r * true_anomaly_dot * np.cos(
        true_anomaly + om
    )

    partials = np.zeros((4, 6))

    partials[0, :] = d_r * np.cos(true_anomaly + om) - (
        d_true_anomaly + d_om
    ) * r * np.sin(true_anomaly + om)
    partials[1, :] = d_r * np.sin(true_anomaly + om) + (
        d_true_anomaly + d_om
    ) * r * np.cos(true_anomaly + om)
    partials[2, :] = (
        d_r_dot * np.cos(true_anomaly + om)
        - (d_true_anomaly + d_om) * r_dot * np.sin(true_anomaly + om)
        - d_r * true_anomaly_dot * np.sin(true_anomaly + om)
        - r * d_true_anomaly_dot * np.sin(true_anomaly + om)
        - r * true_anomaly_dot * np.cos(true_anomaly + om) * (d_true_anomaly + d_om)
    )
    partials[3, :] = (
        d_r_dot * np.sin(true_anomaly + om)
        + (d_true_anomaly + d_om) * r_dot * np.cos(true_anomaly + om)
        + d_r * true_anomaly_dot * np.cos(true_anomaly + om)
        + r * d_true_anomaly_dot * np.cos(true_anomaly + om)
        - r * true_anomaly_dot * np.sin(true_anomaly + om) * (d_true_anomaly + d_om)
    )

    return xyv, partials


def inverse_kepler_2d(xv, m, t):
    """Compute the Keplerian parameters for the osculating orbit.

    No partial derivatives are computed (even though it would be much easier)
    because you can use the partials for kepler_2d and invert the matrix.

    The value of t0 computed is the value within one half-period of t.
    """
    mu = G * m
    # a_guess = np.hypot(xv[0], xv[1])
    h = xv[0] * xv[3] - xv[1] * xv[2]
    r = np.hypot(xv[0], xv[1])
    eps2, eps1 = np.array([xv[3], -xv[2]]) * h / mu - xv[:2] / r
    e = np.hypot(eps1, eps2)
    p = h**2 / mu
    a = p / (1 - e**2)
    pb = 2 * np.pi * (a**3 / mu) ** (0.5)

    om = np.arctan2(eps1, eps2)
    true_anomaly = np.arctan2(xv[1], xv[0]) - om
    eccentric_anomaly = np.arctan2(
        np.sqrt(1 - e**2) * np.sin(true_anomaly), e + np.cos(true_anomaly)
    )
    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)

    true_anomaly_0 = -om
    eccentric_anomaly_0 = np.arctan2(
        np.sqrt(1 - e**2) * np.sin(true_anomaly_0), e + np.cos(true_anomaly_0)
    )
    mean_anomaly_0 = eccentric_anomaly_0 - e * np.sin(eccentric_anomaly_0)

    return Kepler2DParameters(
        a=a,
        pb=pb,
        eps1=eps1,
        eps2=eps2,
        t0=t - (mean_anomaly - mean_anomaly_0) * pb / (2 * np.pi),
    )
    # mean_anomaly*pb/(2*np.pi)


class Kepler3DParameters(
    collections.namedtuple("Kepler3DParameters", "a pb eps1 eps2 i lan t0")
):
    """Parameters to describe a one-object 3D Keplerian orbit.

    Parameters
    ----------
    a : float
        semi-major axis
    pb : float
        binary period
    eps1 : float
    eps2 : float
        eccentricity parameters
    i : float
        inclination angle
    lan : float
        longitude of the ascending node
    t0 : float
        time of the ascending node
    """

    __slots__ = ()


def kepler_3d(params, t):
    """One-body Kepler problem in 3D.

    This function simply uses kepler_2d and rotates it into 3D.
    """
    a = params.a
    pb = params.pb
    eps1 = params.eps1
    eps2 = params.eps2
    i = params.i
    lan = params.lan

    p2 = Kepler2DParameters(a=a, pb=pb, eps1=eps1, eps2=eps2, t0=params.t0)
    xv, jac = kepler_2d(p2, t)
    xyv = np.zeros(6)
    xyv[:2] = xv[:2]
    xyv[3:5] = xv[2:]

    jac2 = np.zeros((6, 8))
    t = np.zeros((6, 6))
    t[:2] = jac[:2]
    t[3:5] = jac[2:]
    jac2[:, :4] = t[:, :4]
    jac2[:, -2:] = t[:, -2:]

    r_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    d_r_i = np.array(
        [[0, 0, 0], [0, -np.sin(i), -np.cos(i)], [0, np.cos(i), -np.sin(i)]]
    )
    r_i_6 = block_diag(r_i, r_i)
    d_r_i_6 = block_diag(d_r_i, d_r_i)
    xyv3 = np.dot(r_i_6, xyv)
    jac3 = np.dot(r_i_6, jac2)
    jac3[:, 4] += np.dot(d_r_i_6, xyv)

    r_lan = np.array(
        [[np.cos(lan), np.sin(lan), 0], [-np.sin(lan), np.cos(lan), 0], [0, 0, 1]]
    )
    d_r_lan = np.array(
        [[-np.sin(lan), np.cos(lan), 0], [-np.cos(lan), -np.sin(lan), 0], [0, 0, 0]]
    )
    r_lan_6 = block_diag(r_lan, r_lan)
    d_r_lan_6 = block_diag(d_r_lan, d_r_lan)
    xyv4 = np.dot(r_lan_6, xyv3)
    jac4 = np.dot(r_lan_6, jac3)
    jac4[:, 5] += np.dot(d_r_lan_6, xyv3)

    return xyv4, jac4


def inverse_kepler_3d(xyv, m, t):
    """Inverse Kepler one-body calculation."""
    L = np.cross(xyv[:3], xyv[3:])
    i = np.arccos(L[2] / np.sqrt(np.dot(L, L)))
    lan = (-np.arctan2(L[0], -L[1])) % (2 * np.pi)

    r_lan = np.array(
        [[np.cos(lan), np.sin(lan), 0], [-np.sin(lan), np.cos(lan), 0], [0, 0, 1]]
    )
    r_lan_6 = block_diag(r_lan, r_lan)
    xyv2 = np.dot(r_lan_6.T, xyv)

    r_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    r_i_6 = block_diag(r_i, r_i)
    xyv3 = np.dot(r_i_6.T, xyv2)

    xv = xyv3[np.array([True, True, False, True, True, False])]
    p2 = inverse_kepler_2d(xv, m, t)

    return Kepler3DParameters(
        a=p2.a, pb=p2.pb, eps1=p2.eps1, eps2=p2.eps2, i=i, lan=lan, t0=p2.t0
    )


# Two body forward and back
# Additional parameters: mass_ratio, cm_x[3], cm_v[3]
# FIXME: define when center of mass is at x_cm
class KeplerTwoBodyParameters(
    collections.namedtuple(
        "KeplerTwoBodyParameters",
        "a pb eps1 eps2 i lan q x_cm y_cm z_cm vx_cm vy_cm vz_cm tasc",
    )
):
    """Parameters to describe a one-object 3D Keplerian orbit.

    Parameters
    ----------
    a : float
        semimajor axis
    pb : float
        binary period
    eps1, eps2 : float
        eccentricity parameters
    i : float
        inclination angle
    lan : float
        longitude of the ascending node
    q : float
        mass ratio of bodies (companion over primary)
    x_cm : float
    y_cm : float
    z_cm : float
        position of the center of mass
    vx_cm : float
    vy_cm : float
    vz_cm : float
        velocity of the center of mass
    tasc : float
        time of the ascending node
    """

    __slots__ = ()


def kepler_two_body(params, t):
    """Set up two bodies in a Keplerian orbit

    Most orbital parameters describe the orbit of the
    primary; the secondary's parameters are inferred
    from the fact that its mass is q times that of the
    primary. x_cm and v_cm are the position and velocity
    of the center of mass of the system.

    The system is observed at time t, and tasc is the
    the time of the ascending node.

    Includes derivatives.
    """
    a = params.a
    pb = params.pb
    eps1 = params.eps1
    eps2 = params.eps2
    i = params.i
    lan = params.lan
    q = params.q
    x_cm = np.array([params.x_cm, params.y_cm, params.z_cm])
    v_cm = np.array([params.vx_cm, params.vy_cm, params.vz_cm])
    tasc = params.tasc

    e = np.eye(15)
    (d_a, d_pb, d_eps1, d_eps2, d_i, d_lan, d_q) = e[:7]
    d_x_cm = e[7:10]
    d_v_cm = e[10:13]
    d_tasc = e[13]
    d_t = e[14]

    a_c = a / q
    a_tot = a + a_c
    d_a_c = d_a / q - a * d_q / q**2
    d_a_tot = d_a + d_a_c

    m_tot, m_tot_prime = mass_partials(a_tot, pb)
    m = m_tot / (1 + q)
    m_c = q * m
    d_m_tot = m_tot_prime[0] * d_a_tot + m_tot_prime[1] * d_pb
    d_m = d_m_tot / (1 + q) - m_tot * d_q / (1 + q) ** 2
    d_m_c = d_q * m + q * d_m

    p2 = Kepler3DParameters(
        a=a_tot,
        pb=params.pb,
        eps1=params.eps1,
        eps2=params.eps2,
        i=params.i,
        lan=params.lan,
        t0=params.tasc,
    )
    xv_tot, jac_one = kepler_3d(p2, t)
    d_xv_tot = np.dot(
        jac_one, np.array([d_a_tot, d_pb, d_eps1, d_eps2, d_i, d_lan, d_tasc, d_t])
    )

    xv = xv_tot / (1 + 1.0 / q)
    d_xv = d_xv_tot / (1 + 1.0 / q) + xv_tot[:, None] * d_q[None, :] / (1 + q) ** 2

    xv_c = -xv / q
    d_xv_c = -d_xv / q + xv[:, None] * d_q[None, :] / q**2

    xv[:3] += x_cm  # FIXME: when, if t is actually t0?
    xv[3:] += v_cm
    xv_c[:3] += x_cm
    xv_c[3:] += v_cm
    d_xv[:3] += d_x_cm
    d_xv[3:] += d_v_cm
    d_xv_c[:3] += d_x_cm
    d_xv_c[3:] += d_v_cm

    total_state = np.zeros(14)
    total_state[:6] = xv
    total_state[6] = m
    total_state[7:13] = xv_c
    total_state[13] = m_c
    d_total_state = np.zeros((14, 15))
    d_total_state[:6] = d_xv
    d_total_state[6] = d_m
    d_total_state[7:13] = d_xv_c
    d_total_state[13] = d_m_c

    return total_state, d_total_state


def inverse_kepler_two_body(total_state, t):
    x_p = total_state[:3]
    v_p = total_state[3:6]
    m_p = total_state[6]

    x_c = total_state[7:10]
    v_c = total_state[10:13]
    m_c = total_state[13]

    x_cm = (m_p * x_p + m_c * x_c) / (m_c + m_p)
    v_cm = (m_p * v_p + m_c * v_c) / (m_c + m_p)

    x = x_p - x_c
    v = v_p - v_c

    xv = np.concatenate((x, v))

    p2 = inverse_kepler_3d(xv, m_c + m_p, t)
    a_tot, pb, eps1, eps2, i, lan, t0 = p2
    q = m_c / m_p
    a = a_tot / (1 + 1.0 / q)

    return KeplerTwoBodyParameters(
        a=a,
        pb=pb,
        eps1=eps1,
        eps2=eps2,
        i=i,
        lan=lan,
        q=q,
        x_cm=x_cm[0],
        y_cm=x_cm[1],
        z_cm=x_cm[2],
        vx_cm=v_cm[0],
        vy_cm=v_cm[1],
        vz_cm=v_cm[2],
        tasc=t0,
    )
