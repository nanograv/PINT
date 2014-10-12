"""This module implements a simple spindown model for an isolated pulsar.
"""
# spindown.py
# Defines Spindown timing model class
import mpmath
import astropy.units as u
from astropy.time.core import SECS_PER_DAY
from .parameter import Parameter, MJDParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_to_mjd_mpf, time_from_mjd_string
from pint import utils
import numpy

class Spindown(TimingModel):
    """This class provides a simple timing model for an isolated pulsar."""
    @mpmath.workdps(20)
    def __init__(self):
        super(Spindown, self).__init__()

        self.add_param(Parameter(name="F0",
            units="Hz",
            description="Spin frequency",
            aliases=["F"],
            parse_value=mpmath.mpf,
            print_value=lambda x: '%.15f'%x,longdoubleV = True))

        self.add_param(Parameter(name="F1",
            units="Hz/s", value=0.0,
            description="Spin-down rate",longdoubleV = True))

        self.add_param(MJDParameter(name="TZRMJD",
            description="Reference epoch for phase",longdoubleV = True))

        self.add_param(MJDParameter(name="PEPOCH",
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb'),
            description="Reference epoch for spin-down",longdoubleV = True))

        self.phase_funcs += [self.simple_spindown_phase,]
        self.phase_funcs_ld += [self.simple_spindown_phase_ld,]
        self.phase_funcs_table += [self.simple_spindown_phase_table,]
        self.dt = []
        self.dt_ld = None
        self.dt_pepoch_ld = None
        self.dt_pepoch = []

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
                        "PEPOCH is required if F1 is set")

    @mpmath.workdps(20)
    def simple_spindown_phase(self, toa, delay):
        """
        Placeholder function for simple spindown phase.

        toa is a single toa object

        delay is the time delay from the TOA to time of pulse emission
          at the pulsar, in seconds.

        returns a Phase object

        TODO:
          make delay input have astropy units?
          mpmath used internally but need to check for precision issues
        """
        # If TZRMJD is not defined, use the first time as phase reference
        # NOTE, all of this ignores TZRSITE and TZRFRQ for the time being.
        if self.TZRMJD.value is None:
            self.TZRMJD.value = toa['mjd'] - delay*u.s
        toaTDBld = toa['tdbld']
        TZRMJDtdbld = self.TZRMJD.longd_value
        dt = (toaTDBld - TZRMJDtdbld) * SECS_PER_DAY
        dt -= delay
        self.dt.append(dt)
        # print "dt not long double array",dt       
        # TODO: what timescale should we use for pepoch calculation?
        # Does this even matter?
        PEPOCHtdbld = self.PEPOCH.longd_value
        dt_pepoch = (PEPOCHtdbld - TZRMJDtdbld) * SECS_PER_DAY
        self.F0.value = numpy.longdouble(self.F0.value)
        self.dt_pepoch.append(dt_pepoch)
        phase = (self.F0.value + 0.5*self.F1.value*(dt-2.0*dt_pepoch))*dt
        return Phase(phase)
    
    def simple_spindown_phase_ld(self, TOAs, delay_array):
        """
        ld doubld arry version of simple_spindow_phase()
        """
        if self.TZRMJD.value is None:
            self.TZRMJD.value = TOAs.tdbld[0]*u.d -\
                                (delay_array[0]*u.s).to(u.day)

        dt = ((TOAs.tdbld-self.TZRMJD.longd_value)*u.day).to(u.s).value
        dt-=delay_array
        self.dt_ld = dt
        dt_pepoch = ((self.PEPOCH.longd_value - self.TZRMJD.longd_value)\
                    *u.day).to(u.s).value
        self.dt_pepoch_ld = dt_pepoch
      #  print "dt Long Double Array",dt
        phase = (self.F0.longd_value + 0.5*self.F1.longd_value*(dt-2.0*dt_pepoch))*dt
                
        return Phase_array(phase)
        
    def simple_spindown_phase_table(self, TOAs, delay_array):
        """
        Table version of simple_spindow_phase()
        """
        if self.TZRMJD.value is None:
            self.TZRMJD.value = TOAs.dataTable['tdb_ld'][0]*u.day -\
                                (delay_array[0]*u.s).to(u.day)
        dt = ((TOAs.dataTable['tdb_ld']-      
                self.TZRMJD.longd_value)*u.day).to(u.s).value    
                                
        dt-=delay_array
        self.total_delay = dt
        dt_pepoch = ((self.PEPOCH.longd_value - self.TZRMJD.longd_value)\
                    *u.day).to(u.s).value
        phase = (self.F0.longd_value + 0.5*self.F1.longd_value*(dt-2.0*dt_pepoch))*dt

        return Phase_array(phase)
             







