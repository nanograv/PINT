"""Functions to compute various derived quantities from pulsar spin parameters, masses, etc.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

import pint

__all__ = [
    "a1sini",
    "companion_mass",
    "dr",
    "dth",
    "gamma",
    "mass_funct",
    "mass_funct2",
    "omdot",
    "omdot_to_mtot",
    "p_to_f",
    "pbdot",
    "pferrs",
    "pulsar_B",
    "pulsar_B_lightcyl",
    "pulsar_age",
    "pulsar_edot",
    "pulsar_mass",
    "shklovskii_factor",
    "sini",
]


@u.quantity_input(
    p=[u.Hz, u.s], pd=[u.Hz / u.s, u.s / u.s], pdd=[u.Hz / u.s**2, u.s / u.s**2]
)
def p_to_f(p, pd, pdd=None):
    """Converts P, Pdot to F, Fdot (or vice versa)

    Convert period, period derivative and period second
    derivative (if supplied) to the equivalent frequency counterparts.
    Will also convert from F to P.

    Parameters
    ----------
    p : astropy.units.Quantity
        pulsar period (or frequency), :math:`P` (or :math:`f`)
    pd : astropy.units.Quantity
        period derivative (or frequency derivative),
        :math:`\dot P` (or :math:`\dot f`)
    pdd : astropy.units.Quantity, optional
        period second derivative (or frequency second derivative),
        :math:`\ddot P` (or :math:`\ddot f`)

    Returns
    -------
    f : astropy.units.Quantity
        pulsar frequency (or period), :math:`f` (or :math:`P`)
    fd : astropy.units.Quantity
        pulsar frequency derivative (or period derivative),
        :math:`\dot f` (or :math:`\dot P`)
    fdd : astropy.units.Quantity
        if `pdd` is supplied, then frequency second derivative
        (or period second derivative), :math:`\ddot f` (or :math:`\ddot P`)
    """
    f = 1.0 / p
    fd = -pd / (p * p)
    if pdd is None:
        return [f, fd]
    fdd = (
        0.0 * f.unit / (u.s**2)
        if pdd == 0.0
        else 2.0 * pd * pd / (p**3.0) - pdd / (p * p)
    )
    return [f, fd, fdd]


@u.quantity_input(
    porf=[u.Hz, u.s],
    porferr=[u.Hz, u.s],
    pdorfd=[u.Hz / u.s, u.s / u.s],
    pdorfderr=[u.Hz / u.s, u.s / u.s],
)
def pferrs(porf, porferr, pdorfd=None, pdorfderr=None):
    """Convert P, Pdot to F, Fdot with uncertainties (or vice versa).

    Calculate the period or frequency errors and
    the Pdot or fdot errors from the opposite ones.

    Parameters
    ----------
    porf : astropy.units.Quantity
        pulsar period (or frequency), :math:`P` or :math:`f`
    porferr : astropy.units.Quantity
        pulsar period uncertainty (or frequency uncertainty),
        :math:`\sigma_P` or :math:`\sigma_f`
    pdorfd : astropy.units.Quantity, optional
        pulsar period derivative (or frequency derivative),
        :math:`\dot P` or :math:`\dot f`
    pdorfderr : astropy.units.Quantity, optional
        pulsar period derivative uncertainty
        (or frequency derivative uncertainty)  :math:`\sigma_{\dot P}`
        or :math:`\sigma_{\dot f}`

    Returns
    -------
    forp : astropy.units.Quantity
        pulsar frequency (or period) :math:`f` or :math:`P`
    forperr : astropy.units.Quantity
        pulsar frequency uncertainty (or period uncertainty)
        :math:`\sigma_f` or :math:`\sigma_P`
    fdorpd : astropy.units.Quantity
        pulsar frequency derivative (or period derivative) if pdorfd supplied
        :math:`\dot f` or :math:`\dot P`
    fdorpderr : astropy.units.Quantity
        if `pdorfd` supplied, then pulsar frequency derivative uncertainty
        (or period derivative uncertainty),
        :math:`\sigma_{\dot f}` or :math:`\sigma_{\dot P}`
    """
    if pdorfd is None:
        return [1.0 / porf, porferr / porf**2.0]
    forperr = porferr / porf**2.0
    fdorpderr = np.sqrt(
        (4.0 * pdorfd**2.0 * porferr**2.0) / porf**6.0
        + pdorfderr**2.0 / porf**4.0
    )
    [forp, fdorpd] = p_to_f(porf, pdorfd)
    return [forp, forperr, fdorpd, fdorpderr]


@u.quantity_input(fo=u.Hz)
def pulsar_age(f: u.Hz, fdot: u.Hz / u.s, n=3, fo=1e99 * u.Hz):
    """Compute pulsar characteristic age

    Return the age of a pulsar given the spin frequency
    and frequency derivative.  By default, the characteristic age
    is returned (assuming a braking index `n` =3 and an initial
    spin frequency :math:`f_0 \gg f`).  But `n` and `fo` can be set.

    Parameters
    ----------
    f : astropy.units.Quantity
        pulsar frequency
    fdot : astropy.units.Quantity
        frequency derivative :math:`\dot f`
    n : int, optional
        braking index (default = 3)
    fo : astropy.units.Quantity, optional
        initial frequency :math:`f_0`

    Returns
    -------
    age : astropy.units.Quantity
        pulsar age in ``u.yr``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        \\tau = \\frac{f}{(n-1)\dot f}\\left(1-\\left(\\frac{f}{f_0}\\right)^{n-1}\\right)
    """
    return (-f / ((n - 1.0) * fdot) * (1.0 - (f / fo) ** (n - 1.0))).to(u.yr)


@u.quantity_input(I=u.g * u.cm**2)
def pulsar_edot(f: u.Hz, fdot: u.Hz / u.s, I=1.0e45 * u.g * u.cm**2):
    """Compute pulsar spindown energy loss rate

    Return the pulsar `Edot` (:math:`\dot E`, in erg/s) given the spin frequency `f` and
    frequency derivative `fdot`. The NS moment of inertia is assumed to be
    `I` = 1.0e45 g cm^2 by default.

    Parameters
    ----------
    f : astropy.units.Quantity
        pulsar frequency
    fdot : astropy.units.Quantity
        frequency derivative :math:`\dot f`
    I : astropy.units.Quantity, optional
        pulsar moment of inertia, default of 1e45 g*cm**2

    Returns
    -------
    Edot : astropy.units.Quantity
        pulsar spin-down luminosity in ``u.erg/u.s``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates :math:`\dot E = -4\pi^2  I  f  \dot f`
    """
    return (-4.0 * np.pi**2 * I * f * fdot).to(u.erg / u.s)


@u.quantity_input
def pulsar_B(f: u.Hz, fdot: u.Hz / u.s):
    """Compute pulsar surface magnetic field

    Return the estimated pulsar surface magnetic field strength
    given the spin frequency and frequency derivative.

    Parameters
    ----------
    f : astropy.units.Quantity
        pulsar frequency
    fdot : astropy.units.Quantity
        frequency derivative :math:`\dot f`

    Returns
    -------
    B : astropy.units.Quantity
        pulsar dipole magnetic field in ``u.G``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates :math:`B=3.2\\times 10^{19}\\,{\\rm  G}\\sqrt{ f \dot f^{-3}}`
    """
    # This is a hack to use the traditional formula by stripping the units.
    # It would be nice to improve this to a  proper formula with units
    return 3.2e19 * u.G * np.sqrt(-fdot.to_value(u.Hz / u.s) / f.to_value(u.Hz) ** 3.0)


@u.quantity_input
def pulsar_B_lightcyl(f: u.Hz, fdot: u.Hz / u.s):
    """Compute pulsar magnetic field at the light cylinder

    Return the estimated pulsar magnetic field strength at the
    light cylinder given the spin frequency and
    frequency derivative.

    Parameters
    ----------
    f : astropy.units.Quantity
        pulsar frequency
    fdot : astropy.units.Quantity
        frequency derivative :math:`\dot f`

    Returns
    -------
    Blc : astropy.units.Quantity
        pulsar dipole magnetic field at the light cylinder in ``u.G``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates :math:`B_{LC} = 2.9\\times 10^8\\,{\\rm G} P^{-5/2} \dot P^{1/2}`
    """
    p, pd = p_to_f(f, fdot)
    # This is a hack to use the traditional formula by stripping the units.
    # It would be nice to improve this to a  proper formula with units
    return (
        2.9e8
        * u.G
        * p.to_value(u.s) ** (-5.0 / 2.0)
        * np.sqrt(pd.to(u.dimensionless_unscaled).value)
    )


@u.quantity_input
def mass_funct(pb: u.d, x: u.cm):
    """Compute binary mass function from period and semi-major axis

    Can handle scalar or array inputs.

    Parameters
    ----------
    pb : astropy.units.Quantity
        Binary period
    x : astropy.units.Quantity
        Semi-major axis, A1SINI, in units of ``pint.ls``

    Returns
    -------
    f_m : astropy.units.Quantity
        Mass function in ``u.solMass``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        f(m_p, m_c) = \\frac{4\pi^2 x^3}{G P_b^2}

    See [1]_

    .. [1] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.34 (RHS)
    """
    fm = 4.0 * np.pi**2 * x**3 / (const.G * pb**2)
    return fm.to(u.solMass)


@u.quantity_input
def mass_funct2(mp: u.Msun, mc: u.Msun, i: u.deg):
    """Compute binary mass function from masses and inclination

    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        Pulsar mass, typically in ``u.solMass``
    mc : astropy.units.Quantity
        Companion mass, typically in ``u.solMass``
    i : astropy.coordinates.Angle or astropy.units.Quantity
        Inclination angle, in ``u.deg`` or ``u.rad``

    Returns
    -------
    f_m : astropy.units.Quantity
        Mass function in ``u.solMass``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Inclination is such that edge on is ``i = 90*u.deg``
    An 'average' orbit has cos(i) = 0.5, or ``i = 60*u.deg``

    Calculates

    .. math::
        f(m_p, m_c) = \\frac{m_c^3\sin^3 i}{(m_c + m_p)^2}

    See [2]_

    .. [2] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.34 (LHS)

    """
    return (mc * np.sin(i)) ** 3.0 / (mc + mp) ** 2.0


@u.quantity_input
def pulsar_mass(pb: u.d, x: u.cm, mc: u.Msun, i: u.deg):
    """Compute pulsar mass from orbital parameters

    Return the pulsar mass (in solar mass units) for a binary.
    Can handle scalar or array inputs.

    Parameters
    ----------
    pb : astropy.units.Quantity
        Binary orbital period
    x : astropy.units.Quantity
        Projected pulsar semi-major axis (aka ASINI) in ``pint.ls``
    mc : astropy.units.Quantity
        Companion mass in ``u.solMass``
    i : astropy.coordinates.Angle or astropy.units.Quantity
        Inclination angle, in ``u.deg`` or ``u.rad``

    Returns
    -------
    mass : astropy.units.Quantity
        In ``u.solMass``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Example
    -------
    >>> import pint
    >>> import pint.derived_quantities
    >>> from astropy import units as u
    >>> print(pint.derived_quantities.pulsar_mass(2*u.hr, .2*pint.ls, 0.5*u.Msun, 60*u.deg))
    7.6018341985817885 solMass


    Notes
    -------
    This forms a quadratic equation of the form:
    :math:`a M_p^2 + b M_p + c = 0``

    with:

    - :math:`a = f(P_b,x)` (the mass function)
    - :math:`b = 2 f(P_b,x) M_c`
    - :math:`c = f(P_b,x)  M_c^2 - M_c\sin^3 i`

    except the discriminant simplifies to:
    :math:`4f(P_b,x) M_c^3 \sin^3 i`

    solve it directly
    this has to be the positive branch of the quadratic
    because the vertex is at :math:`-M_c`, so
    the negative branch will always be < 0
    """
    massfunct = mass_funct(pb, x)

    sini = np.sin(i)
    ca = massfunct
    cb = 2 * massfunct * mc

    return ((-cb + np.sqrt(4 * massfunct * mc**3 * sini**3)) / (2 * ca)).to(u.Msun)


@u.quantity_input(inc=u.deg, mpsr=u.solMass)
def companion_mass(pb: u.d, x: u.cm, i=60.0 * u.deg, mp=1.4 * u.solMass):
    """Commpute the companion mass from the orbital parameters

    Compute companion mass for a binary system from orbital mechanics,
    not Shapiro delay.
    Can handle scalar or array inputs.

    Parameters
    ----------
    pb : astropy.units.Quantity
        Binary orbital period
    x : astropy.units.Quantity
        Projected pulsar semi-major axis (aka ASINI) in ``pint.ls``
    i : astropy.coordinates.Angle or astropy.units.Quantity, optional
        Inclination angle, in ``u.deg`` or ``u.rad.`` Default is 60 deg.
    mp : astropy.units.Quantity, optional
        Pulsar mass in ``u.solMass``. Default is 1.4 Msun

    Returns
    -------
    mass : astropy.units.Quantity
        In ``u.solMass``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Example
    -------
    >>> import pint
    >>> import pint.derived_quantities
    >>> from astropy import units as u
    >>> print(pint.derived_quantities.companion_mass(1*u.d, 2*pint.ls, inc=30*u.deg, mpsr=1.3*u.Msun))
    0.6363138973397279 solMass

    Notes
    -----
    This ends up as a a cubic equation of the form:
    :math:`a M_c^3 + b M_c^2 + c M_c + d = 0`

    - :math:`a = \sin^3(inc)`
    - :math:`b = -{\\rm massfunct}`
    - :math:`c = -2 M_p {\\rm massfunct}`
    - :math:`d = -{\\rm massfunct} M_p^2`

    To solve it we can use a direct calculation of the cubic roots [3]_.


    It's useful to look at the discriminant to understand the nature of the roots
    and make sure we get the right one [4]_.


    :math:`\Delta = (b^2 c^2 - 4ac^3-4b^3d-27a^2d^2+18abcd)`

    if :math:`\delta< 0` then there is only 1 real root,
    and I think we do it correctly below
    and this should be < 0
    since this reduces to :math:`-27\sin^6 i f(P_b,x)^2 M_p^4 -4\sin^3 i f(P_b,x)^3 M_p^3`
    (where :math:`f(P_b,x)` is the mass function) so there is just 1 real root.

    .. [3] https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    .. [4] https://en.wikipedia.org/wiki/Discriminant#Degree_3

    """
    massfunct = mass_funct(pb, x)

    # solution
    sini = np.sin(i)
    a = sini**3
    # delta0 = b ** 2 - 3 * a * c
    # delta0 is always > 0
    delta0 = massfunct**2 + 6 * mp * massfunct * a
    # delta1 is always <0
    # delta1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d
    delta1 = (
        -2 * massfunct**3
        - 18 * a * mp * massfunct**2
        - 27 * a**2 * massfunct * mp**2
    )
    # Q**2 is always > 0, so this is never a problem
    # in terms of complex numbers
    # Q = np.sqrt(delta1**2 - 4*delta0**3)
    Q = np.sqrt(
        108 * a**3 * mp**3 * massfunct**3
        + 729 * a**4 * mp**4 * massfunct**2
    )
    # this could be + or - Q
    # pick the - branch since delta1 is <0 so that delta1 - Q is never near 0
    Ccubed = 0.5 * (delta1 + Q)
    # try to get the real root
    C = np.sign(Ccubed) * np.fabs(Ccubed) ** (1.0 / 3)
    # note that the difference b**2 - 3*a*c should be strictly positive
    # so those shouldn't cancel
    # and then all three terms should have the same signs
    # since a>0, b<0, C<0, and delta0>0
    # the denominator will be near 0 only when sin(i) is ~0, but that's already a known problem
    x1 = massfunct / 3.0 / a - C / 3.0 / a - delta0 / 3.0 / a / C
    return x1.to(u.Msun)


@u.quantity_input
def pbdot(mp: u.Msun, mc: u.Msun, pb: u.d, e: u.dimensionless_unscaled):
    """Post-Keplerian orbital decay pbdot, assuming general relativity.

    pbdot (:math:`\dot P_B`) is the change in the binary orbital period
    due to emission of gravitational waves.
    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period
    e : astropy.units.Quantity or float
        orbital eccentricity

    Returns
    -------
    pbdot : astropy.units.Quantity
        (dimensionless)

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::
        \dot P_b = -\\frac{192\pi}{5}T_{\odot}^{5/3} \\left(\\frac{P_b}{2\pi}\\right)^{-5/3}
        f(e)\\frac{m_p m_c}{(m_p+m_c)^{1/3}}

    with

    .. math::
        f(e)=\\frac{1+(73/24)e^2+(37/96)e^4}{(1-e^2)^{7/2}}

    and :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [5]_.

    .. [5] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.52

    """
    f = (1 + (73.0 / 24) * e**2 + (37.0 / 96) * e**4) / (1 - e**2) ** (7.0 / 2)
    value = (
        (const.G / const.c**3) ** (5.0 / 3)
        * (pb / (2 * np.pi)) ** (-5.0 / 3)
        * (-192 * np.pi / 5)
        * f
        * (mp * mc)
        / (mp + mc) ** (1.0 / 3)
    )
    return value.to(u.s / u.s)


@u.quantity_input
def gamma(mp: u.Msun, mc: u.Msun, pb: u.d, e: u.dimensionless_unscaled):
    """Post-Keplerian time dilation and gravitational redshift gamma, assuming general relativity.

    gamma (:math:`\gamma`) is the amplitude of the modification in arrival times caused by the varying
    gravitational redshift of the companion and time dilation in an elliptical orbit.  The time delay is
    :math:`\gamma \sin E`, where :math:`E` is the eccentric anomaly.
    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period
    e : astropy.units.Quantity or float
        orbital eccentricity

    Returns
    -------
    gamma : astropy.units.Quantity
        in ``u.s``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::
        \gamma = T_{\odot}^{2/3} \\left(\\frac{P_b}{2\pi}\\right)^{1/3} e \\frac{m_c(m_p+2m_c)}{(m_p+m_c)^{4/3}}

    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [6]_

    .. [6] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.49

    """
    value = (
        (const.G / const.c**3) ** (2.0 / 3)
        * (pb / (2 * np.pi)) ** (1.0 / 3)
        * e
        * (mc * (mp + 2 * mc))
        / (mp + mc) ** (4.0 / 3)
    )
    return value.to(u.s)


@u.quantity_input
def omdot(mp: u.Msun, mc: u.Msun, pb: u.d, e: u.dimensionless_unscaled):
    """Post-Keplerian longitude of periastron precession rate omdot, assuming general relativity.

    omdot (:math:`\dot \omega`) is the relativistic advance of periastron.
    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period
    e : astropy.units.Quantity or float
        orbital eccentricity

    Returns
    -------
    omdot : astropy.units.Quantity
        In ``u.deg/u.yr``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        \dot \omega = 3T_{\odot}^{2/3} \\left(\\frac{P_b}{2\pi}\\right)^{-5/3}
        \\frac{1}{1-e^2}(m_p+m_c)^{2/3}

    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [7]_.

    .. [7] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.48

    """
    value = (
        3
        * (pb / (2 * np.pi)) ** (-5.0 / 3)
        * (1 / (1 - e**2))
        * (const.G * (mp + mc) / const.c**3) ** (2.0 / 3)
    )
    return value.to(u.deg / u.yr, equivalencies=u.dimensionless_angles())


@u.quantity_input
def sini(mp: u.Msun, mc: u.Msun, pb: u.d, x: u.cm):
    """Post-Keplerian sine of inclination, assuming general relativity.

    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period
    x : astropy.units.Quantity
        Semi-major axis, A1SINI, in units of ``pint.ls``

    Returns
    -------
    sini : astropy.units.Quantity

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        s = T_{\odot}^{-1/3} \\left(\\frac{P_b}{2\pi}\\right)^{-2/3}
        \\frac{(m_p+m_c)^{2/3}}{m_c}

    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [11]_.

    .. [11] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.51

    """

    return (
        (const.G) ** (-1.0 / 3)
        * (pb / 2 / np.pi) ** (-2.0 / 3)
        * x
        * (mp + mc) ** (2.0 / 3)
        / mc
    ).decompose()


@u.quantity_input
def dr(mp: u.Msun, mc: u.Msun, pb: u.d):
    """Post-Keplerian Roemer delay term

    dr (:math:`\delta_r`) is part of the relativistic deformation of the orbit

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period

    Returns
    -------
    dr : astropy.units.Quantity

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        \delta_r = T_{\odot}^{2/3} \\left(\\frac{P_b}{2\pi}\\right)^{2/3}
        \\frac{3 m_p^2+6 m_p m_c +2m_c^2}{(m_p+m_c)^{4/3}}

    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [12]_.

    .. [12] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.54

    """
    return (
        (const.G / const.c**3) ** (2.0 / 3)
        * (2 * np.pi / pb) ** (2.0 / 3)
        * (3 * mp**2 + 6 * mp * mc + 2 * mc**2)
        / (mp + mc) ** (4 / 3)
    ).decompose()


@u.quantity_input
def dth(mp: u.Msun, mc: u.Msun, pb: u.d):
    """Post-Keplerian Roemer delay term

    dth (:math:`\delta_{\\theta}`) is part of the relativistic deformation of the orbit

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period

    Returns
    -------
    dth : astropy.units.Quantity

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        \delta_{\\theta} = T_{\odot}^{2/3} \\left(\\frac{P_b}{2\pi}\\right)^{2/3}
        \\frac{3.5 m_p^2+6 m_p m_c +2m_c^2}{(m_p+m_c)^{4/3}}

    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [13]_.

    .. [13] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.55

    """
    return (
        (const.G / const.c**3) ** (2.0 / 3)
        * (2 * np.pi / pb) ** (2.0 / 3)
        * (3.5 * mp**2 + 6 * mp * mc + 2 * mc**2)
        / (mp + mc) ** (4 / 3)
    ).decompose()


@u.quantity_input
def omdot_to_mtot(omdot: u.deg / u.yr, pb: u.d, e: u.dimensionless_unscaled):
    """Determine total mass from Post-Keplerian longitude of periastron precession rate omdot,
    assuming general relativity.

    omdot (:math:`\dot \omega`) is the relativistic advance of periastron.  It relates to the total
    system mass (assuming GR).
    Can handle scalar or array inputs.

    Parameters
    ----------
    omdot : astropy.units.Quantity
        relativistic advance of periastron
    pb : astropy.units.Quantity
        Binary orbital period
    e : astropy.units.Quantity or float
        orbital eccentricity

    Returns
    -------
    mtot : astropy.units.Quantity
        In ``u.Msun``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Inverts

    .. math::

        \dot \omega = 3T_{\odot}^{2/3} \\left(\\frac{P_b}{2\pi}\\right)^{-5/3}
        \\frac{1}{1-e^2}(m_p+m_c)^{2/3}

    to calculate :math:`m_{\\rm tot} = m_p + m_c`,
    with :math:`T_\odot = GM_\odot c^{-3}`.

    More details in :ref:`Timing Models`.  Also see [9]_.

    .. [9] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.48
    """
    return (
        (
            (
                omdot
                / (
                    3
                    * (const.G / const.c**3) ** (2.0 / 3)
                    * (pb / (2 * np.pi)) ** (-5.0 / 3)
                    * (1 - e**2) ** (-1)
                )
            )
        )
        ** (3.0 / 2)
    ).to(u.Msun, equivalencies=u.dimensionless_angles())


@u.quantity_input(pb=u.d, mp=u.Msun, mc=u.Msun, i=u.deg)
def a1sini(mp, mc, pb, i=90 * u.deg):
    """Pulsar's semi-major axis.

    The full semi-major axis is given by Kepler's third law.  This is the
    projection (:math:`\sin i`) of just the pulsar's orbit (:math:`m_c/(m_p+m_c)`
    times the full semi-major axis), which is what pulsar timing measures.
    Can handle scalar or array inputs.

    Parameters
    ----------
    mp : astropy.units.Quantity
        pulsar mass
    mc : astropy.units.Quantity
        companion mass
    pb : astropy.units.Quantity
        Binary orbital period
    i : astropy.coordinates.Angle or astropy.units.Quantity
        orbital inclination

    Returns
    -------
    a1sini : astropy.units.Quantity
        Projected semi-major axis of pulsar's orbit in ``pint.ls``

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities

    Notes
    -----
    Calculates

    .. math::

        \\frac{a_p \sin i}{c} = \\frac{m_c \sin i}{(m_p+m_c)^{2/3}}
        G^{1/3}\\left(\\frac{P_b}{2\pi}\\right)^{2/3}

    More details in :ref:`Timing Models`.  Also see [8]_

    .. [8] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.21, 8.22, 8.27

    """
    return (
        (mc * np.sin(i))
        * (const.G * (pb / (2 * np.pi)) ** 2 / (mp + mc) ** 2) ** (1.0 / 3)
    ).to(pint.ls)


@u.quantity_input
def shklovskii_factor(pmtot: u.mas / u.yr, D: u.kpc):
    """Compute magnitude of Shklovskii correction factor.

    Computes the Shklovskii correction factor, as defined in Eq 8.12 of Lorimer & Kramer (2005) [10]_
    This is the factor by which :math:`\dot P /P` is increased due to the transverse velocity.
    Note that this affects both the measured spin period and the orbital period.
    If we call this Shklovskii acceleration :math:`a_s`, then

    .. math::

        \dot P_{\\rm intrinsic} = \dot P_{\\rm observed} - a_s P

    Parameters
    ----------
    pmtot : astropy.units.Quantity
        typically units of u.mas/u.yr
        Total proper motion of the pulsar :math:`\mu` (system)
    D : astropy.units.Quantity
        typically in units of u.kpc or u.pc
        Distance to the pulsar

    Returns
    -------
    acceleration : astropy.units.Quantity
        Shklovskii acceleration

    Notes
    -----
    .. [10] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.12
    """
    # This uses the small angle approximation that sin(x) = x, so we need to
    # make our angle dimensionless.
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        a_s = (D * pmtot**2 / const.c).to(u.s**-1)
    return a_s


@u.quantity_input
def dispersion_slope(dm: pint.dmu):
    """Compute the dispersion slope.

    This is equal to DMconst * DM.
    See https://nanograv-pint.readthedocs.io/en/latest/explanation.html#dispersion-measure
    for an explanation on why this is relevant.

    Parameters
    ----------
    dm: astropy.units.Quantity
        Dispersion measure

    Returns
    -------
    dsl: astropy.units.Quantity
        Dispersion slope

    Raises
    ------
    astropy.units.UnitsError
        If the input data are not appropriate quantities
    TypeError
        If the input data are not quantities
    """

    return (dm * pint.DMconst).decompose()
