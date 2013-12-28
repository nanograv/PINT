# solar_system_shapiro.py
# Add in Shapiro delays due to solar system objects
from warnings import warn
import astropy.units as u
from timing_model import Parameter, TimingModel, MissingParameter

class SolarSystemShapiro(TimingModel):

    def __init__(self):
        super(SolarSystemShapiro, self).__init__()

        self.add_param(Parameter(name="PLANET_SHAPIRO",
            units=None, value=True,
            description="Include planetary Shapiro delays (Y/N)",
            parse_value=lambda x: x.upper()=='Y',
            print_value=lambda x: 'Y' if x else 'N'))

        self.delay_funcs += [self.solar_system_shapiro_delay,]

    def setup(self):
        super(SolarSystemShapiro,self).setup()

    def solar_system_shapiro_delay(self,toa):
        # TODO implement the calculation! ;)
        return 0.0

