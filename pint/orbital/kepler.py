"""Functions for working with Keplerian orbits

All times are in days, distances in light-seconds, and masses in solar masses.
"""
import collections
import numpy as np
from scipy.optimize import newton, fsolve
from scipy.linalg import block_diag
import scipy.linalg


# FIXME: can I import this from somewhere?
G = 36768.59290949113 # Based on standard gravitational parameter

def true_from_eccentric(e, eccentric_anomaly):
    """Compute the true anomaly from the eccentric anomaly.

    Inputs:
        e - the eccentricity
        eccentric_anomaly - the eccentric anomaly

    Outputs:
        true_anomaly - the true anomaly
        true_anomaly_de - derivative of true anomaly with respect to e
        true_anomaly_prime - derivative of true anomaly with respect to
            eccentric anomaly
    """
    true_anomaly = 2*np.arctan2(np.sqrt(1+e)*np.sin(eccentric_anomaly/2),
                                np.sqrt(1-e)*np.cos(eccentric_anomaly/2))
    true_anomaly_de = (np.sin(eccentric_anomaly)/
            (np.sqrt(1-e**2)*(1-e*np.cos(eccentric_anomaly))))
    true_anomaly_prime = (np.sqrt(1-e**2)/(1-e*np.cos(eccentric_anomaly)))
    return true_anomaly, true_anomaly_de, true_anomaly_prime

def eccentric_from_mean(e, mean_anomaly):
    """Compute the eccentric anomaly from the mean anomaly.

    Inputs:
        e - the eccentricity
        mean_anomaly - the mean anomaly

    Outputs:
        eccentric_anomaly - the true anomaly
        derivatives - pair of derivatives with respect to the two inputs
    """
    eccentric_anomaly = newton(
            lambda E: E-e*np.sin(E)-mean_anomaly,
            mean_anomaly,
            lambda E: 1-e*np.cos(E))
    eccentric_anomaly_de = (np.sin(eccentric_anomaly)
                             /(1-e*np.cos(eccentric_anomaly)))
    eccentric_anomaly_prime = (1-e*np.cos(eccentric_anomaly))**(-1)
    return eccentric_anomaly, [eccentric_anomaly_de, eccentric_anomaly_prime]



def mass(a, pb):
    """Compute the mass of a particle in a Kepler orbit.

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    return 4*np.pi**2*a**3*pb**(-2)/G
def mass_partials(a, pb):
    """Compute the mass of a particle in a Kepler orbit, with partials.

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    m = mass(a, pb)
    return m, np.array([3*m/a, -2*m/pb])


def btx_parameters(asini, pb, eps1, eps2, tasc):
    """Attempt to convert parameters from ELL1 to BTX.

    """
    e = np.hypot(eps1, eps2)
    om = np.arctan2(eps1, eps2)
    true_anomaly = -om # True anomaly at the ascending node
    eccentric_anomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly),
                                   e+np.cos(true_anomaly))
    mean_anomaly = eccentric_anomaly - e*np.sin(eccentric_anomaly)
    t0 = tasc-mean_anomaly*pb/(2*np.pi)
    return asini, pb, e, om, t0



class Kepler2DParameters(
        collections.namedtuple("Kepler2DParameters",
                               "a pb eps1 eps2 t0")):
    """Parameters to describe a one-object 2D Keplerian orbit.

    a - semimajor axis
    pb - binary period
    eps1, eps2 - eccentricity parameters
    t0 - time of the ascending node
    """
    __slots__ = ()

def kepler_2d(params, t):
    """Position and velocity of a particle in a Kepler orbit.

    The orbit has semimajor axis a, period pb, and eccentricity
    paramerized by eps1=e*sin(om) and eps2=e*cos(om), and the
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
    t = t-params.t0
    if eps1 == 0 and eps2 == 0:
        eps1 = 1e-50
    e = np.hypot(eps1, eps2)
    if e == 0:
        d_e = np.array([0, 0, 0, 0, 0, 0])
    else:
        d_e = np.array([0, 0, eps1/e, eps2/e, 0, 0])
    #return e, d_e

    om = np.arctan2(eps1, eps2)
    if e == 0:
        d_om = np.array([0, 0, 0, 0, 0, 0])
    else:
        d_om = np.array([0, 0, -eps1/e**2, eps2/e**2, 0, 0])
    #return om, d_om

    true_anomaly_0 = -om
    d_true_anomaly_0 = -d_om

    eccentric_anomaly_0 = np.arctan2(
            np.sqrt(1-e**2)*np.sin(true_anomaly_0),
            e + np.cos(true_anomaly_0))
    d_eccentric_anomaly_0 = (
        d_e*(-(1+e*np.cos(true_anomaly_0))*np.sin(true_anomaly_0)/
            (np.sqrt(1-e**2)*(e*np.cos(true_anomaly_0)+1)**2)) +
        d_true_anomaly_0
            *(np.sqrt(1-e**2)*(1+e*np.cos(true_anomaly_0)))
            /(e*np.cos(true_anomaly_0)+1)**2)

    mean_anomaly_0 = eccentric_anomaly_0 - e*np.sin(eccentric_anomaly_0)
    d_mean_anomaly_0 = (d_eccentric_anomaly_0
            -d_e*np.sin(eccentric_anomaly_0)
            -e*np.cos(eccentric_anomaly_0)*d_eccentric_anomaly_0)

    mean_anomaly = 2*np.pi*t/pb + mean_anomaly_0
    d_mean_anomaly = \
      (2*np.pi*np.array([0, -t/pb**2, 0, 0, -pb**(-1), pb**(-1)])
            + d_mean_anomaly_0)

    mean_anomaly_dot = 2*np.pi/pb
    d_mean_anomaly_dot = 2*np.pi*np.array([0, -pb**(-2), 0, 0, 0, 0])
    #return ([mean_anomaly, mean_anomaly_dot],
    #        [d_mean_anomaly, d_mean_anomaly_dot])
    #return mean_anomaly, d_mean_anomaly

    eccentric_anomaly, (eccentric_anomaly_de, eccentric_anomaly_prime) = \
       eccentric_from_mean(e, mean_anomaly)
    eccentric_anomaly_dot = eccentric_anomaly_prime*mean_anomaly_dot

    d_eccentric_anomaly = (eccentric_anomaly_de*d_e
            +eccentric_anomaly_prime*d_mean_anomaly)
    d_eccentric_anomaly_prime = \
       (np.cos(eccentric_anomaly)/
        (1-e*np.cos(eccentric_anomaly))**2*d_e
                -e*np.sin(eccentric_anomaly)/
                (1-e*np.cos(eccentric_anomaly))**2*d_eccentric_anomaly)
    d_eccentric_anomaly_dot = (d_eccentric_anomaly_prime*mean_anomaly_dot
            +eccentric_anomaly_prime*d_mean_anomaly_dot)
    #return eccentric_anomaly, d_eccentric_anomaly
    #return eccentric_anomaly_prime, d_eccentric_anomaly_prime
    #return eccentric_anomaly_dot, d_eccentric_anomaly_dot

    true_anomaly, true_anomaly_de, true_anomaly_prime = \
      true_from_eccentric(e, eccentric_anomaly)
    true_anomaly_dot = true_anomaly_prime*eccentric_anomaly_dot

    d_true_anomaly = (true_anomaly_de*d_e
                      + true_anomaly_prime*d_eccentric_anomaly)
    d_true_anomaly_prime = (
            ((np.cos(eccentric_anomaly)-e)/
             (np.sqrt(1-e**2)*(1-e*np.cos(eccentric_anomaly))**2))*d_e
            -e*np.sqrt(1-e**2)*np.sin(eccentric_anomaly)
             /(1-e*np.cos(eccentric_anomaly))**2*d_eccentric_anomaly)
    d_true_anomaly_dot = (d_true_anomaly_prime*eccentric_anomaly_dot
                         +true_anomaly_prime*d_eccentric_anomaly_dot)
    #return true_anomaly, d_true_anomaly
    #return true_anomaly_prime, d_true_anomaly_prime
    #return true_anomaly_dot, d_true_anomaly_dot

    r = a*(1-e**2)/(1+e*np.cos(true_anomaly))
    r_prime = (a*e*(1-e**2)*np.sin(true_anomaly)
            /(1+e*np.cos(true_anomaly))**2)
    r_dot = r_prime*true_anomaly_dot
    d_a = np.array([1, 0, 0, 0, 0, 0])
    d_r = (d_a*r/a
          -a*d_e*((1+e**2)*np.cos(true_anomaly)+2*e)
          /(1+e*np.cos(true_anomaly))**2
          +r_prime*d_true_anomaly)
    d_r_prime = (d_a*r_prime/a
                +a*d_e*(-e*(1+e**2)*np.cos(true_anomaly)-3*e**2+1)
                *np.sin(true_anomaly)/(1+e*np.cos(true_anomaly))**3
                +a*e*(1-e**2)
                *(e*(np.sin(true_anomaly)**2+1)+np.cos(true_anomaly))
                /(1+e*np.cos(true_anomaly))**3*d_true_anomaly)
    d_r_dot = d_r_prime*true_anomaly_dot + r_prime*d_true_anomaly_dot
    #return r, d_r
    #return r_prime, d_r_prime
    #return r_dot, d_r_dot

    xyv = np.zeros(4)
    xyv[0] = r*np.cos(true_anomaly+om)
    xyv[1] = r*np.sin(true_anomaly+om)
    xyv[2] = (r_dot*np.cos(true_anomaly+om)
            -r*true_anomaly_dot*np.sin(true_anomaly+om))
    xyv[3] = (r_dot*np.sin(true_anomaly+om)
            +r*true_anomaly_dot*np.cos(true_anomaly+om))

    partials = np.zeros((4, 6))

    partials[0, :] = (d_r*np.cos(true_anomaly+om)
                     -(d_true_anomaly+d_om)*r*np.sin(true_anomaly+om))
    partials[1, :] = (d_r*np.sin(true_anomaly+om)
                     +(d_true_anomaly+d_om)*r*np.cos(true_anomaly+om))
    partials[2, :] = (d_r_dot*np.cos(true_anomaly+om)
                     -(d_true_anomaly+d_om)*r_dot*np.sin(true_anomaly+om)
                     -d_r*true_anomaly_dot*np.sin(true_anomaly+om)
                     -r*d_true_anomaly_dot*np.sin(true_anomaly+om)
                     -r*true_anomaly_dot*np.cos(true_anomaly+om)
                         *(d_true_anomaly+d_om))
    partials[3, :] = (d_r_dot*np.sin(true_anomaly+om)
                     +(d_true_anomaly+d_om)*r_dot*np.cos(true_anomaly+om)
                     +d_r*true_anomaly_dot*np.cos(true_anomaly+om)
                     +r*d_true_anomaly_dot*np.cos(true_anomaly+om)
                     -r*true_anomaly_dot*np.sin(true_anomaly+om)
                         *(d_true_anomaly+d_om))

    return xyv, partials

def inverse_kepler_2d(xv, m, t):
    """Compute the Keplerian parameters for the osculating orbit.

    No partial derivatives are computed (even though it would be much easier)
    because you can use the partials for kepler_2d and invert the matrix.
    """
    mu = G*m
    #a_guess = np.hypot(xv[0], xv[1])
    h = (xv[0]*xv[3]-xv[1]*xv[2])
    r = np.hypot(xv[0], xv[1])
    eps2, eps1 = np.array([xv[3], -xv[2]])*h/mu - xv[:2]/r
    e = np.hypot(eps1, eps2)
    p = h**2/mu
    a = p/(1-e**2)
    pb = 2*np.pi*(a**3/mu)**(0.5)

    om = np.arctan2(eps1, eps2)
    true_anomaly = np.arctan2(xv[1], xv[0])-om
    eccentric_anomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly),
                                   e+np.cos(true_anomaly))
    mean_anomaly = eccentric_anomaly - e*np.sin(eccentric_anomaly)

    true_anomaly_0 = -om
    eccentric_anomaly_0 = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly_0),
                                   e+np.cos(true_anomaly_0))
    mean_anomaly_0 = eccentric_anomaly_0 - e*np.sin(eccentric_anomaly_0)

    return Kepler2DParameters(a=a, pb=pb, eps1=eps1, eps2=eps2,
                              t0=(mean_anomaly-mean_anomaly_0)*pb/(2*np.pi))
    #mean_anomaly*pb/(2*np.pi)
