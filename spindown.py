# spindown.py
# Defines Spindown timing model class
from timing_model import Parameter, TimingModel, MissingParameter

class Spindown(TimingModel):

    def __init__(self):
        super(Spindown,self).__init__()

        self.add_param(Parameter(name="F0",
            units="Hz", 
            description="Spin frequency",
            aliases=["F"],
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="F1",
            units="Hz/s", value=0.0, 
            description="Spin-down rate"))

        self.add_param(Parameter(name="TZRMJD",
            units="MJD", 
            description="Reference epoch for phase"))

        self.add_param(Parameter(name="PEPOCH",
            units="MJD", 
            description="Reference epoch for spin-down"))

        self.phase_funcs += [self.simple_spindown_phase,]

    def setup(self):
        super(Spindown,self).setup()
        # Check for required params
        for p in ("F0",):
            if getattr(self,p).value==None:
                raise MissingParameter("Spindown",p)
        # If F1 is set, we need PEPOCH
        if self.F1.value!=0.0:
            if self.PEPOCH.value==None:
                raise MissingParameter("Spindown","PEPOCH",
                        "PEPOCH is required if F1 is set")

    def simple_spindown_phase(self,t_pulsar):
        """
        Placeholder function for simple spindown phase.

        Note t_pulsar is the time of emission at the pulsar, not
        the time of arrival at earth.

        Still need to figure out data types for times, make
        sure the right precision is in use, etc.  For now, this is 
        here to show the structure of how this will work.
        """
        # If TZRMJD is not defined, use the first time as phase reference
        if self.TZRMJD.value==None:
            self.TZRMJD.value = t_pulsar
        dt = t_pulsar - self.TZRMJD.value
        dt_pepoch = self.PEPOCH.value - self.TZRMJD.value
        phase = self.F0.value*dt + 0.5*self.F1.value*dt*(dt-2.0*dt_pepoch)
        return phase
