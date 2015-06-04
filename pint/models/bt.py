"""This model provides the BT (Blandford & Teukolsky 1976, ApJ, 205, 580) model.
    Based on BTmodel.C in TEMPO2
    """
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25

from .parameter import Parameter, MJDParameter
from .timing_model import Cache, TimingModel, MissingParameter
from ..phase import Phase
from ..utils import time_from_mjd_string, time_to_longdouble
from ..orbital.kepler import eccentric_from_mean
import numpy as np
import time

class BT(TimingModel):
    """This class provides an implementation of the BT model
        by Blandford & Teukolsky 1976, ApJ, 205, 580.
        Based on BTmodel.C in TEMPO2
    """
    def __init__(self):
        super(BT, self).__init__()

        # TO DO: add commonly used parameters such as T90, TASC with clear,
        # documented usage.

        # Parameters are mostly defined as numpy doubles.
        # Some might become long doubles in the future.

        self.add_param(Parameter(name="PB",
            units="s",
            description="Orbital period",
            parse_value=np.double,
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="A1",
            units="lt-s",
            description="Projected semi-major axis",
            parse_value=np.double))

        self.add_param(Parameter(name="E",
            units="",
            aliases = ["ECC"],
            description="Eccentricity",
            parse_value=np.double))

        self.add_param(Parameter(name="OM",
            units="deg",
            description="Longitude of periastron",
            parse_value=lambda x: np.double(x)))

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
            description="Orbital spin-down rate",
            parse_value=np.double))

        self.add_param(Parameter(name="EDOT",
            units="s^-1",
            description="Orbital spin-down rate",
            parse_value=np.double))

        self.add_param(Parameter(name="GAMMA",
            units="s",
            description="Time dilation & gravitational redshift",
            parse_value=np.double))

        self.delay_funcs += [self.BT_delay,]

    def setup(self):
        super(BT, self).setup()

        # If any necessary parameter is missing, raise MissingParameter.
        # This will probably be updated after ELL1 model is added.
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p,
                                       "%s is required for BT" % p)

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

    def eccentric_anomaly(self, eccentricity, mean_anomaly):
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

    @Cache.use_cache
    def BT_delay(self, toas):
        """Actual delay calculation of the BT model (equation 5 of Taylor &
            Weisberg, 1989, ApJ, 345, 434-450)
            From BTmodel.C in TEMPO2, and so in turn from bnrybt.f in TEMPO.
            toas are really toas.table
            """
        tt0 = np.array([t for t in toas['tdbld']], dtype=np.longdouble) * SECS_PER_DAY

        # Apply all the delay terms, except for the binary model itself
        for df in self.delay_funcs:
            if df != self.BT_delay:
                tt0 -= df(toas)

        tt0 -= time_to_longdouble(self.T0.value) * SECS_PER_DAY

        pb = self.PB.value * SECS_PER_DAY
        edot = self.EDOT.value
        ecc = self.E.value + edot * tt0

        # TODO: Check this assertion. This mechanism is probably too strong
        # Probably better to create a BadParameter signal to raise,
        # catch it and give a new value to eccentricity?
        assert np.all(np.logical_and(ecc >= 0, ecc <= 1)), \
            "BT model: Eccentricity goes out of range"

        pbdot = self.PBDOT.value
        xdot = self.XDOT.value
        asini  = self.A1.value + xdot * tt0

        # XPBDOT exists in other models, not BT. In Tempo2 it is set to 0.
        # Check if it even makes sense to keep it here.
        xpbdot = 0
        omdot = self.OMDOT.value
        omega0 = self.OM.value
        omega = np.radians(omega0 + omdot*tt0/SECS_PER_JUL_YEAR)
        gamma = self.GAMMA.value

        orbits = tt0 / pb - 0.5 * (pbdot + xpbdot) * (tt0 / pb) ** 2
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = 2 * np.pi * (orbits - norbits)
        bige = self.eccentric_anomaly(ecc, phase)

        tt = 1.0 - ecc ** 2
        som = np.sin(omega)
        com = np.cos(omega)

        alpha = asini * som
        beta = asini * com * np.sqrt(tt)
        sbe = np.sin(bige)
        cbe = np.cos(bige)
        q = alpha * (cbe - ecc) + (beta + gamma) * sbe
        r = -alpha * sbe + beta * cbe
        s = 1.0 / (1.0 - ecc * cbe)

        return q - (2 * np.pi / pb) * q * r * s
