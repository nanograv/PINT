# spindown.py
# Defines Spindown timing model class
import mpmath
import astropy.units as u
from astropy.time.core import SECS_PER_DAY
from .timing_model import Parameter, MJDParameter, TimingModel, MissingParameter
from ..phase import Phase
from ..utils import timedelta_to_mpf_sec, time_to_mjd_mpf

class Spindown(TimingModel):

    @mpmath.workdps(20)
    def __init__(self):
        super(Spindown,self).__init__()

        self.add_param(Parameter(name="F0",
            units="Hz", 
            description="Spin frequency",
            aliases=["F"],
            parse_value=mpmath.mpf,
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="F1",
            units="Hz/s", value=0.0, 
            description="Spin-down rate"))

        self.add_param(MJDParameter(name="TZRMJD",
            description="Reference epoch for phase"))

        self.add_param(MJDParameter(name="PEPOCH",
            description="Reference epoch for spin-down"))

        self.phase_funcs += [self.simple_spindown_phase,]

    def setup(self):
        super(Spindown,self).setup()
        # Check for required params
        for p in ("F0",):
            if getattr(self,p).value is None:
                raise MissingParameter("Spindown",p)
        # If F1 is set, we need PEPOCH
        if self.F1.value!=0.0:
            if self.PEPOCH.value is None:
                raise MissingParameter("Spindown","PEPOCH",
                        "PEPOCH is required if F1 is set")

    @mpmath.workdps(20)
    def simple_spindown_phase(self,toa,delay):
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
            self.TZRMJD.value = toa.mjd - delay*u.s
        dt = (time_to_mjd_mpf(toa.mjd.tdb) 
                - time_to_mjd_mpf(self.TZRMJD.value.tdb)) * SECS_PER_DAY
        dt -= delay
        # TODO: what timescale should we use for pepoch calculation?
        # Does this even matter?
        dt_pepoch = timedelta_to_mpf_sec(self.PEPOCH.value-self.TZRMJD.value)
        phase = (self.F0.value + 0.5*self.F1.value*(dt-2.0*dt_pepoch))*dt
        return Phase(phase)
