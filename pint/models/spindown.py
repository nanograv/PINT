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
import parameter as p
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble, taylor_horner,\
                    time_from_longdouble


class Spindown(TimingModel):
    """This class provides a simple timing model for an isolated pulsar."""
    def __init__(self):
        super(Spindown, self).__init__()

        # The number of terms in the taylor exapansion of spin freq (F0...FN)
        #self.num_spin_terms = maxderivs

        self.add_param(p.floatParameter(name="F0", value=0.0, units="Hz",
                       description="Spin-frequency", long_double=True))

        self.add_param(p.prefixParameter(name="F1", value=0.0, units='Hz/s^1',
                       description="Spindown-rate",
                       unitTplt=self.F_unit,
                       descriptionTplt=self.F_description,
                       type_match='float',long_double=True))

        self.add_param(p.MJDParameter(name="TZRMJD",
                       description="Reference epoch for phase = 0.0",
                       time_scale='tdb'))

        self.add_param(p.MJDParameter(name="PEPOCH",
                       description="Reference epoch for spin-down",
                       time_scale='tdb'))


        self.phase_funcs += [self.spindown_phase,]

    def setup(self):
        super(Spindown, self).setup()
        # Check for required params
        for p in ("F0",):
            if getattr(self, p).value is None:
                raise MissingParameter("Spindown", p)

        # Check continuity
        F_terms = self.get_prefix_mapping('F').keys()
        F_terms.sort()
        F_in_order = range(1,max(F_terms)+1)
        if not F_terms == F_in_order:
            diff = list(set(F_in_order) - set(F_terms))
            raise MissingParameter("Spindown", "F%d"%diff[0])
        # If F1 is set, we need PEPOCH
        if self.F1.value != 0.0:
            if self.PEPOCH.value is None:
                raise MissingParameter("Spindown", "PEPOCH",
                        "PEPOCH is required if F1 or higher are set")

        self.num_spin_terms = len(self.get_prefix_mapping('F')) + 1

    def F_description(self, x):
        """Template function for description"""
        if x <1:
            return "Spin-frequency"
        else:
            return "Spin-frequency %d derivative"%x

    def F_unit(self,x):
        """Template function for unit"""
        if x <1:
            return "Hz"
        else:
            return "Hz/s^%d"%x

    def get_spin_terms(self):
        """Return a list of the spin term values in the model: [F0, F1, ..., FN]
        """
        return [getattr(self, "F%d"%ii).num_value for ii in range(self.num_spin_terms)]

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
