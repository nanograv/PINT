# solar_system_shapiro.py
# Add in Shapiro delays due to solar system objects
from warnings import warn
import numpy
import astropy.units as u
import astropy.constants as const
from .timing_model import Parameter, TimingModel, MissingParameter

# TODO: define this in a single place somewhere
ls = u.def_unit('ls', const.c * 1.0 * u.s)

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
        # Just test with sun for now:
        psr_dir = self.ssb_to_psb_xyz(epoch=toa.mjd)
        obs_sun = -toa.obs_sun_pvs.pos
        r = numpy.sqrt(obs_sun.dot(obs_sun))
        rcostheta = obs_sun.dot(psr_dir)
        Tsun = 4.925490947e-6 # define this somewhere
        # TODO: figure out best way to use units here:
        delay = -2.0*(Tsun)*numpy.log((r+rcostheta)/const.au).value
        return delay

