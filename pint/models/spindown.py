"""This module implements a simple spindown model for an isolated pulsar.
"""
# spindown.py
# Defines Spindown timing model class
import numpy
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
from .parameter import Parameter, MJDParameter,prefixParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble, taylor_horner,\
                    time_from_longdouble

# The maximum number of spin frequency derivs we allow
maxderivs = 20

class Spindown(TimingModel):
    """This class provides a simple timing model for an isolated pulsar."""
    def __init__(self):
        super(Spindown, self).__init__()

        # The number of terms in the taylor exapansion of spin freq (F0...FN)
        self.num_spin_terms = maxderivs

        self.add_param(Parameter(name="F0",
            units="Hz",
            description="Spin frequency",
            aliases=["F"],
            parse_value=str2longdouble,
            print_value=repr))

        self.add_param(Parameter(name="F1",
            units="Hz/s", value=0.0,
            description="Spin-down rate"))

        for ii in range(2, self.num_spin_terms + 1):
            self.add_param(prefixParameter(name="F%d"%ii,
                units="Hz/s^%s"%ii, value=0.0,
                unitTplt = lambda x: "Hz/s^%s"%x,
                description="Spin-frequency %d derivative"%ii,
                descriptionTplt = lambda x: "Spin-frequency %d derivative"%x))

        self.add_param(MJDParameter(name="TZRMJD",
            description="Reference epoch for phase = 0.0",
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb'),
            get_value = lambda x: time_from_longdouble(x,'tdb')))

        self.add_param(MJDParameter(name="PEPOCH",
            description="Reference epoch for spin-down",
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb'),
            get_value = lambda x: time_from_longdouble(x,'tdb')))

        self.prefix_params+= ['F',]
        self.phase_funcs += [self.spindown_phase,]

    def setup(self):
        super(Spindown, self).setup()
        # Check for required params
        for p in ("F0",):
            if getattr(self, p).value is None:
                raise MissingParameter("Spindown", p)
        # If F1 is set, we need PEPOCH
        if self.F1.value != 0.0:
            if self.PEPOCH.value is None:
                raise MissingParameter("Spindown", "PEPOCH",
                        "PEPOCH is required if F1 or higher are set")
        # Remove all unused freq derivs
        for ii in range(self.num_spin_terms, -1, -1):
            term = "F%d"%ii
            if hasattr(self, term) and \
                    getattr(self, term).value==0.0 and \
                    getattr(self, term).uncertainty is None:
                delattr(self, term)
                self.params.remove(term)
                if ii>1:
                    self.num_prefix_params['F']-=1
            else:
                break
        # Add a shortcut for the number of spin terms there are
        if hasattr(self,'F1'):
            self.num_spin_terms = self.num_prefix_params['F'] + 2
        else:
            self.num_spin_terms = 1
    def get_spin_terms(self):
        """Return a list of the spin term values in the model: [F0, F1, ..., FN]
        """
        return [getattr(self, "F%d"%ii).value for ii in range(self.num_spin_terms)]

    def spindown_phase(self, toas, delay):
        """Spindown phase function.

        delay is the time delay from the TOA to time of pulse emission
          at the pulsar, in seconds.

        This routine should implement Eq 120 of the Tempo2 Paper II (2006, MNRAS 372, 1549)

        returns an array of phases in long double
        """
        # If TZRMJD is not defined, use the first time as phase reference
        # NOTE, all of this ignores TZRSITE and TZRFRQ for the time being.
        # TODO: TZRMJD should be set by default somewhere in a standard place,
        #       after the TOAs are loaded (RvH -- June 2, 2015)
        # NOTE: Should we be using barycentric arrival times, instead of TDB?
        if self.TZRMJD.value is None:
            self.TZRMJD.value = toas['tdb'][0] - delay[0]*u.s
        # Warning(paulr): This looks wrong.  You need to use the
        # TZRFREQ and TZRSITE to compute a proper TDB reference time.
        if not hasattr(self, "TZRMJDld"):
            self.TZRMJDld = time_to_longdouble(self.TZRMJD.value)

        # Add the [0.0] because that is the constant phase term
        fterms = [0.0] + self.get_spin_terms()

        dt_tzrmjd = (toas['tdbld'] - self.TZRMJDld) * SECS_PER_DAY - delay
        # TODO: what timescale should we use for pepoch calculation? Does this even matter?
        dt_pepoch = (time_to_longdouble(self.PEPOCH.value) - self.TZRMJDld) * SECS_PER_DAY

        phs_tzrmjd = taylor_horner(dt_tzrmjd-dt_pepoch, fterms)
        phs_pepoch = taylor_horner(-dt_pepoch, fterms)
        return phs_tzrmjd - phs_pepoch

    def d_phase_d_F0(self, toas):
        """Calculate the derivative wrt F0"""
        # NOTE: Should we be using barycentric arrival times, instead of TDB?
        # TODO: toas should have units from the table
        tdb = toas['tdbld'].quantity * u.day
        dt_pepoch = time_to_longdouble(self.PEPOCH.value) * u.day
        delay = self.delay(toas) * u.s
        dpdF0 = -(tdb - dt_pepoch) - delay
        return dpdF0.decompose()

    def d_phase_d_F1(self, toas):
        """Calculate the derivative wrt F1"""
        # NOTE: Should we be using barycentric arrival times, instead of TDB?
        # TODO: what timescale should we use for pepoch calculation? Does this even matter?
        tdb = toas['tdbld'] * u.day
        delay = self.delay(toas) * u.s
        dt_pepoch = time_to_longdouble(self.PEPOCH.value) * u.day
        dt = tdb - dt_pepoch - delay
        dpdF1 = -0.5 * dt ** 2
        return dpdF1.decompose()
