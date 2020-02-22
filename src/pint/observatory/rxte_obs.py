# special_locations.py
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from scipy.interpolate import interp1d

from pint.observatory.nicer_obs import load_FPorbit
from pint.observatory.special_locations import SpecialLocation
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel

# Special "site" location for RXTE satellite


class RXTEObs(SpecialLocation):
    """Observatory-derived class for the RXTE photon data.

    Note that this must be instantiated once to be put into the Observatory registry.

    Parameters
    ----------

    name: str
        Observatory name
    ft2name: str
        File name to read spacecraft position information from
    tt2tdb_mode: str
        Selection for mode to use for TT to TDB conversion.
        'none' = Give no position to astropy.Time()
        'pint' = Use PINT routines for TT to TDB conversion.
        'geo' = Give geocenter position to astropy.Time()
        'astropy' = Give spacecraft ITRF position to astropy.Time()
    """

    def __init__(self, name, FPorbname, tt2tdb_mode="pint"):

        self.FPorb = load_FPorbit(FPorbname)
        # Now build the interpolator here:
        self.X = interp1d(self.FPorb["MJD_TT"], self.FPorb["X"])
        self.Y = interp1d(self.FPorb["MJD_TT"], self.FPorb["Y"])
        self.Z = interp1d(self.FPorb["MJD_TT"], self.FPorb["Z"])
        self.Vx = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vx"])
        self.Vy = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vy"])
        self.Vz = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vz"])
        super(RXTEObs, self).__init__(name=name, tt2tdb_mode=tt2tdb_mode)

    @property
    def timescale(self):
        return "tt"

    def earth_location_itrf(self, time=None):
        """Return RXTE spacecraft location in ITRF coordinates"""

        if self.tt2tdb_mode.lower().startswith("pint"):
            log.debug("Using location=None for TT to TDB conversion")
            return None
        elif self.tt2tdb_mode.lower().startswith("astropy"):
            # First, interpolate ECI geocentric location from orbit file.
            # These are inertial coorinates aligned with ICRF
            log.debug("Performing GCRS to ITRS transformation")
            pos_gcrs = GCRS(
                CartesianRepresentation(
                    self.X(time.tt.mjd) * u.m,
                    self.Y(time.tt.mjd) * u.m,
                    self.Z(time.tt.mjd) * u.m,
                ),
                obstime=time,
            )

            # Now transform ECI (GCRS) to ECEF (ITRS)
            # By default, this uses the WGS84 ellipsoid
            pos_ITRS = pos_gcrs.transform_to(ITRS(obstime=time))

            # Return geocentric ITRS coordinates as an EarthLocation object
            return pos_ITRS.earth_location
        else:
            log.error("Unknown tt2tdb_mode %s, using None" % self.tt2tdb_mode)
            return None

    @property
    def tempo_code(self):
        return None

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
