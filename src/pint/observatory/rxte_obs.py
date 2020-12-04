# special_locations.py
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from scipy.interpolate import interp1d

from pint.observatory.nicer_obs import load_FPorbit
from pint.observatory.special_locations import SpacecraftObs
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel

# Special "site" location for RXTE satellite


class RXTEObs(SpacecraftObs):
    """Observatory-derived class for the RXTE photon data.

    Note that this must be instantiated once to be put into the Observatory registry.

    Parameters
    ----------

    name: str
        Observatory name
    ft2name: str
        File name to read spacecraft position information from
    """

    def __init__(self, name, FPorbname):

        self.FPorb = load_FPorbit(FPorbname)
        # Now build the interpolator here:
        self.X = interp1d(self.FPorb["MJD_TT"], self.FPorb["X"])
        self.Y = interp1d(self.FPorb["MJD_TT"], self.FPorb["Y"])
        self.Z = interp1d(self.FPorb["MJD_TT"], self.FPorb["Z"])
        self.Vx = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vx"])
        self.Vy = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vy"])
        self.Vz = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vz"])
        super(RXTEObs, self).__init__(name=name)

    def get_gcrs(self, t, ephem=None, grp=None):
        """Return position vector of RXTE in GCRS
        t is an astropy.Time or array of astropy.Time objects
        Returns a 3-vector of Quantities representing the position
        in GCRS coordinates.
        """
        return (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FPorb["X"].unit
        )

    def posvel(self, t, ephem):
        """Return position and velocity vectors of RXTE.

        t is an astropy.Time or array of astropy.Times
        """
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)
        # Now add vector from Earth to RXTE
        rxte_pos_geo = (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FPorb["X"].unit
        )
        rxte_vel_geo = (
            np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])
            * self.FPorb["Vx"].unit
        )
        rxte_posvel = PosVel(rxte_pos_geo, rxte_vel_geo, origin="earth", obj="rxte")
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + rxte_posvel
