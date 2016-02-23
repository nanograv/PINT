"""This model provides the DD (Damour & Deruelle 1986, AIHS, 44, 263-292) model.
    Based on DDmodel.C in TEMPO2
    """
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
from .parameter import Parameter, MJDParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import Phase
from ..utils import time_from_mjd_string, time_to_longdouble
from ..orbital.kepler import eccentric_from_mean
import numpy as np
import time


def eccentric_anomaly(eccentricity, mean_anomaly):
    """
    eccentric_anomaly(mean_anomaly):
    Return the eccentric anomaly in radians, given a set of mean_anomalies
    in radians.
    """
    TWOPI = 2 * np.pi
    ma = np.fmod(mean_anomaly, TWOPI)
    ma = np.where(ma < 0.0, ma+TWOPI, ma)
    ecc_anom_old = ma
    ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)
    # This is a simple iteration to solve Kepler's Equation
    while (np.maximum.reduce(np.fabs(ecc_anom-ecc_anom_old)) > 5e-15):
        ecc_anom_old = ecc_anom[:]
        ecc_anom = ma + eccentricity * np.sin(ecc_anom_old)
    return ecc_anom

def DD_delay_func(t, PB, T0, A1, OM=0.0, ECC=0.0, EDOT=0.0, PBDOT=0.0, XDOT=0.0, \
        OMDOT=0.0, am2=1.0, GAMMA=0.0, kin=None, sini=None, XPBDOT=0.0):
    """
    Delay due to pulsar binary motion. Model:
    Damour & Deruelle (1986), AIHS, 44, 263-292
    Based on DDmodel.C in Tempo2. DD Equations listed as comments
    
    @param t:       Time (tdbld)
    @param PB:      Binary period
    @param T0:      Epoch of periastron passage
    @param A1:      Projected semi-major axis of orbit on line-of-sight
    @param OM:      Longitude of periastron (omega)
    @param ECC:     Eccentricity of the orbit [0.0]
    @param EDOT:    Time-derivative of ECC [0.0]
    @param PBDOT:   Time-derivative of PB [0.0]
    @param XDOT:    Time-derivative of a*sin(i)  [0.0]
    @param OMDOT:   Time-derivative of OMEGA [0.0]
    @param am2:     Companion mass (in solar mass) [1.0]
    @param XPBDOT:  Rate of change of orbital period minus GR prediction
    @param GAMMA:   Time dilation and gravitational redshift
    @param kin:     (overrides sini), Inclination angle
    @param sini:    Sine of inclination angle
    """
    # TODO (RvH): How do I get the solMass in seconds using astropy?
    SUNMASS         = np.longdouble('4.925490947e-6')

    # TODO (RvH):
    # - XPBDOT is completely covariant with PBDOT. Why is it included in Tempo2?
    # - dr and dth are not used in DDmodel (set to 0). Why not have them as
    #   separate parameters?
    dr, dth = 0.0, 0.0

    # Sin i
    if kin is not None:
        si = np.sin(kin)
    elif sini is not None:
        si = sini
    else:
        si = 0.0

    if si > 1.0:
        print("Sin I > 1.0. Setting to 1: should probably use DDS model")
        si = 1.0

    # Unit transformations where necessary
    m2 = am2 * SUNMASS
    pb = PB * SECS_PER_DAY
    an = 2.0*np.pi / pb
    k = OMDOT * np.pi/ (180.0*365.25*SECS_PER_DAY*an)
    t0 = T0
    ct = t
    tt0 = (ct - time_to_longdouble(t0)) * SECS_PER_DAY
    omz = OM
    xdot = XDOT
    pbdot = PBDOT
    edot = EDOT
    xpbdot = XPBDOT
    gamma = GAMMA
    x = A1 + xdot*tt0
    ecc = ECC + edot*tt0
    er, eth = ecc*(1.0+dr), ecc*(1.0+dth)

    assert np.all(np.logical_and(ecc >= 0, ecc <= 1)), \
        "DD model: Eccentricity goes out of range"

    orbits = tt0/pb - 0.5*(pbdot+xpbdot)*(tt0/pb)**2
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)
    u = eccentric_anomaly(ecc, phase)

    # DD equations: 17b, 17c, 29, and 46 through 52
    su = np.sin(u)
    cu = np.cos(u)
    onemecu = 1.0-ecc*cu
    cae = (cu-ecc)/onemecu
    sae = np.sqrt(1.0-pow(ecc,2))*su/onemecu
    ae = np.arctan2(sae, cae)
    ae[ae<0.0] += 2.0*np.pi
    ae = 2.0 * np.pi * orbits + ae - phase
    omega = omz  + k * ae
    sw = np.sin(omega)
    cw = np.cos(omega)
    alpha = x * sw
    beta = x * np.sqrt(1-eth**2) * cw
    bg = beta + gamma
    dre = alpha * (cu-er) + bg * su
    drep = -alpha * su + bg * cu
    drepp = -alpha * cu - bg * su
    anhat = an / onemecu

    # DD equations: 26, 27, 57:
    sqr1me2 = np.sqrt(1-ecc**2)
    cume = cu - ecc
    brace = onemecu - si*(sw*cume+sqr1me2*cw*su)
    dlogbr = np.log(brace)
    ds = -2*m2*dlogbr

    #  Now compute d2bar, the orbital time correction in DD equation 42
    d2bar = dre*(1-anhat*drep+(anhat**2)*(drep**2 + 0.5*dre*drepp - \
                      0.5*ecc*su*dre*drep/onemecu)) + ds
    return -d2bar




class DD(TimingModel):
    """This class provides an implementation of the  DD model
        by Damour & Deruelle 1986, AIHS, 44, 263-292.
        Based on DDmodel.C in TEMPO2
    """
    def __init__(self):
        super(DD, self).__init__()

        # TO DO: add commonly used parameters such as T90, TASC with clear,
        # documented usage.

        # Parameters are mostly defined as numpy doubles.
        # Some might become long doubles in the future.
        self.BinaryModelName = 'DD'
        self.add_param(Parameter(name="PB",
            units="s",
            description="Orbital period",
            parse_value=np.double,
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="A1",
            units="lt-s",
            description="Projected semi-major axis of orbit",
            parse_value=np.double))

        self.add_param(Parameter(name="E",
            units="",
            aliases = ["ECC"],
            description="Eccentricity",
            parse_value=np.double))

        self.add_param(Parameter(name="OM",
            units="deg",
            description="Longitude of periastron passage",
            parse_value=lambda x: np.double(x) / 180 * np.pi))

        self.add_param(MJDParameter(name="T0",
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb'),
            description="Epoch of periastron passage"))

        self.add_param(Parameter(name="PBDOT",
            units="s/s",
            description="First derivative of orbital period",
            parse_value=np.double))

        self.add_param(Parameter(name="OMDOT",
            units="deg/yr",
            description="Periastron advance",
            parse_value=np.double))

        self.add_param(Parameter(name="XDOT",
            units="s/s",
            description="Rate of change of semi-major axis",
            aliases = ["A1DOT"],
            parse_value=np.double))

        self.add_param(Parameter(name="EDOT",
            units="s^-1",
            description="Rate of change of eccentricity",
            parse_value=np.double))

        self.add_param(Parameter(name="M2",
            units="solMass",
            description="Mass of companion",
            parse_value=np.double))

        # TODO: not a DD parameter
        #self.add_param(Parameter(name="XPBDOT",
        #    units="s/s",
        #    description="Rate of change of orbital period minus GR prediction",
        #    parse_value=np.double))

        self.add_param(Parameter(name="GAMMA",
            units="s",
            description="Time dilation and gravitational redshift",
            parse_value=np.double))

        self.add_param(Parameter(name="KIN",
            units="deg",
            description="Inclination angle",
            parse_value=np.double))

        self.add_param(Parameter(name="SINI",
            units="",
            description="sin(inclination angle)",
            parse_value=np.double))

        self.delay_funcs += [self.DD_delay,]
#        self.phase_funcs += [self.DD_phase,]
#
    def setup(self):
        super(DD, self).setup()

        # If any necessary parameter is missing, raise MissingParameter.
        # This will probably be updated after ELL1 model is added.
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("DD", p,
                                       "%s is required for DD" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "XDOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("Spindown", "T0",
                        "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.E.value == 0 or self.E.value is None:
            for p in ("E", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

    def DD_delay(self, toas):
        """Actual delay calculation of the DD model
           Performed in DD_delay_func. Prepare model here
            toas are really toas.table"""

        t = np.array([ti for ti in toas['tdbld']], dtype=np.longdouble)

        # Apply all the delay terms, except for the binary model itself
        for df in self.delay_funcs:
            if df != self.DD_delay:
                t -= df(toas) / SECS_PER_DAY

        # Tempo2 uses some unit conversions, with the following comments:
        # /* Check units: DO BETTER JOB */  :(
        edot = self.EDOT.value
        if edot > 1.0e-7:
            edot *= 1.0e-12

        pbdot = self.PBDOT.value
        if pbdot > 1.0e-7:
            pbdot *= 1.0e-12

        xdot = self.XDOT.value
        if xdot > 1.0e-7:
            xdot *= 1.0e-12

        return -DD_delay_func(t, self.PB.value, self.T0.value, self.A1.value, \
                self.OM.value, self.E.value, edot, \
                pbdot, xdot, self.OMDOT.value, \
                self.M2.value, self.GAMMA.value, \
                sini=self.SINI.value, kin=self.KIN.value, XPBDOT=0.0)
