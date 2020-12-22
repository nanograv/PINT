"""Solar system Shapiro delay."""
# solar_system_shapiro.py
# Add in Shapiro delays due to solar system objects
from __future__ import absolute_import, division, print_function

import astropy.constants as const
import astropy.units as u
import numpy
from astropy import log

from pint import (
    Tearth,
    Tjupiter,
    Tmars,
    Tmercury,
    Tneptune,
    Tsaturn,
    Tsun,
    Turanus,
    Tvenus,
)
from pint.models.parameter import boolParameter
from pint.models.timing_model import DelayComponent


class SolarSystemShapiro(DelayComponent):
    """Shapiro delay due to light bending near Solar System objects."""

    register = True
    category = "solar_system_shapiro"

    def __init__(self):
        super(SolarSystemShapiro, self).__init__()
        self.add_param(
            boolParameter(
                name="PLANET_SHAPIRO",
                value=False,
                description="Include planetary Shapiro delays (Y/N)",
            )
        )
        self.delay_funcs_component += [self.solar_system_shapiro_delay]

    def setup(self):
        super(SolarSystemShapiro, self).setup()

    def validate(self):
        super(SolarSystemShapiro, self).validate()

    # Put masses in a convenient dictionary
    _ss_mass_sec = {
        "sun": Tsun.value,
        "mercury": Tmercury.value,
        "venus": Tvenus.value,
        "earth": Tearth.value,
        "mars": Tmars.value,
        "jupiter": Tjupiter.value,
        "saturn": Tsaturn.value,
        "uranus": Turanus.value,
        "neptune": Tneptune.value,
    }

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
        # TODO: numpy.sum currently loses units in some cases...
        r = (numpy.sqrt(numpy.sum(obj_pos ** 2, axis=1))) * obj_pos.unit
        rcostheta = numpy.sum(obj_pos * psr_dir, axis=1)
        # This is the 2nd to last term from Eqn 4.6 in Backer &
        # Hellings, ARAA, 1986 with gamma = 1 (as defined by GR).  We
        # have the opposite sign of the cos(theta) term, since our
        # object position vector is defined from the observatory to
        # the solar system object, rather than from the object to the
        # pulsar (as described after Eqn 4.3 in the paper).
        # See also https://en.wikipedia.org/wiki/Shapiro_time_delay
        # where \Delta t = \frac{2GM}{c^3}\log(1-\vec{R}\cdot\vec{x})
        return -2.0 * T_obj * numpy.log((r - rcostheta) / const.au).value

    def solar_system_shapiro_delay(self, toas, acc_delay=None):
        """
        Returns total shapiro delay to due solar system objects.
        If the PLANET_SHAPIRO model param is set to True then
        planets are included, otherwise only the value for the
        Sun is calculated.

        Requires Astrometry or similar model that provides the
        ssb_to_psb_xyz method for direction to pulsar.

        If planets are to be included, TOAs.compute_posvels() must
        have been called with the planets=True argument.
        """
        # Start out with 0 delay with units of seconds
        tbl = toas.table
        delay = numpy.zeros(len(tbl))
        for ii, key in enumerate(tbl.groups.keys):
            grp = tbl.groups[ii]
            # obs = tbl.groups.keys[ii]["obs"]
            loind, hiind = tbl.groups.indices[ii : ii + 2]
            if key["obs"].lower() == "barycenter":
                log.debug("Skipping Shapiro delay for Barycentric TOAs")
                continue
            psr_dir = self._parent.ssb_to_psb_xyz_ICRS(
                epoch=grp["tdbld"].astype(numpy.float64)
            )
            delay[loind:hiind] += self.ss_obj_shapiro_delay(
                grp["obs_sun_pos"], psr_dir, self._ss_mass_sec["sun"]
            )
            if self.PLANET_SHAPIRO.value:
                for pl in ("jupiter", "saturn", "venus", "uranus", "neptune"):
                    delay[loind:hiind] += self.ss_obj_shapiro_delay(
                        grp["obs_" + pl + "_pos"], psr_dir, self._ss_mass_sec[pl]
                    )
        return delay * u.second
