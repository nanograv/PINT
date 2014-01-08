# solar_system_shapiro.py
# Add in Shapiro delays due to solar system objects
from warnings import warn
import numpy
import astropy.constants as const
from .parameter import Parameter
from .timing_model import TimingModel, MissingParameter
from .. import Tsun, Tmercury, Tvenus, Tearth, Tmars, \
        Tjupiter, Tsaturn, Turanus, Tneptune

class SolarSystemShapiro(TimingModel):

    def __init__(self):
        super(SolarSystemShapiro, self).__init__()

        self.add_param(Parameter(name="PLANET_SHAPIRO",
            units=None, value=False,
            description="Include planetary Shapiro delays (Y/N)",
            parse_value=lambda x: x.upper()=='Y',
            print_value=lambda x: 'Y' if x else 'N'))

        self.delay_funcs += [self.solar_system_shapiro_delay,]

    def setup(self):
        super(SolarSystemShapiro,self).setup()


    # Put masses in a convenient dictionary
    _ss_mass_sec = {"sun": Tsun.value,
                   "mercury": Tmercury.value,
                   "venus": Tvenus.value,
                   "earth": Tearth.value,
                   "mars": Tmars.value,
                   "jupiter": Tjupiter.value,
                   "saturn": Tsaturn.value,
                   "uranus": Turanus.value,
                   "neptune": Tneptune.value}

    @staticmethod
    def ss_obj_shapiro_delay(obj_pos, psr_dir, T_obj):
        """
        ss_obj_shapiro_delay(obj_pos, psr_dir, T_obj)

        returns Shapiro delay in seconds for a solar system object.

        Inputs:
          obj_pos : position vector from Earth to SS object, with Units
          psr_dir : unit vector in direction of pulsar
          T_obj : mass of object in seconds (GM/c^3)
        """
        r = numpy.sqrt(obj_pos.dot(obj_pos))
        rcostheta = obj_pos.dot(psr_dir)
        # This formula copied from tempo2 code.  The sign of the
        # cos(theta) term has been changed since we are using the
        # opposite convention for object position vector (from 
        # observatory to object in this code).
        return -2.0 * T_obj * numpy.log((r-rcostheta)/const.au).value

    def solar_system_shapiro_delay(self, toa):
        """
        Returns total shapiro delay to due solar system objects.
        If the PLANET_SHAPIRO model param is set to True then 
        planets are included, otherwise only the value for the
        Sun in calculated.

        Requires Astrometry or similar model that provides the
        ssb_to_psb_xyz method for direction to pulsar.  

        If planets are to be included, TOAs.compute_posvels() must
        have been called with the planets=True argument.
        """
        psr_dir = self.ssb_to_psb_xyz(epoch=toa.mjd)
        delay = 0.0

        # Sun
        delay += self.ss_obj_shapiro_delay(toa.obs_sun_pvs.pos, 
                                           psr_dir,
                                           self._ss_mass_sec['sun'])
        
        if self.PLANET_SHAPIRO.value:
            for pl in ('jupiter','saturn','venus','uranus'):
                delay += self.ss_obj_shapiro_delay(
                    getattr(toa, 'obs_'+pl+'_pvs').pos,
                    psr_dir,
                    self._ss_mass_sec[pl])
                
        return delay

