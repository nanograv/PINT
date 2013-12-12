# spindown.py
# Defines Spindown timing model class
from timing_model import Parameter, TimingModel

class Spindown(TimingModel):

    def __init__(self):
        super(Spindown,self).__init__()

        self.add_param(Parameter(name="F0",
            units="Hz", 
            description="Spin frequency",
            aliases=["F"],
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="F1",
            units="Hz/s", 
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
        print "Spindown setup"

    def simple_spindown_phase(self,toa):
        """
        Placeholder function for simple spindown phase.
        Still need to figure out data types for toa, mjd, make
        sure the right precision is in use, etc.  For now, this is 
        here to show the structure of how this will work.
        """
        # If TZRMJD is not defined, use this toa for phase reference
        if self.TZRMJD.value==None:
            self.TZRMJD.value = toa
        dt = toa - self.TZRMJD.value
        phase = dt*self.F0.value + 0.5*dt*dt*self.F1.value
        return phase
