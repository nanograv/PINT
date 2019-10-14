"""nuSTAR as an observatory."""
from __future__ import absolute_import, division, print_function

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.table import Table
from scipy.interpolate import interp1d

from pint.fits_utils import read_fits_event_mjds
from pint.observatory.special_locations import SpecialLocation
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel

# Special "site" location for the NuSTAR satelite


def load_orbit(orb_filename):
    """Load data from a NuSTAR orbit file

    Parameters
    ----------
    orb_filename : str
        Name of file to load

    Returns
    -------
    astropy.table.Table
        containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load photon times from FT1 file

    if "_orb" in orb_filename:
        log.warning(
            "The NuSTAR orbit file you are providing is known to give"
            "a solution precise only to the ~0.5ms level. Use the "
            "pipeline-produced attitude-orbit file ('*.attorb.gz') for"
            "better precision."
        )

    hdulist = pyfits.open(orb_filename)
    orb_hdr = hdulist[1].header
    orb_dat = hdulist[1].data

    log.info("Opened orb FITS file {0}".format(orb_filename))
    # TIMESYS should be 'TT'
    # TIMEREF should be 'LOCAL', since no delays are applied
    timesys = orb_hdr["TIMESYS"]
    log.info("orb TIMESYS {0}".format(timesys))
    try:
        timeref = orb_hdr["TIMEREF"]
    except KeyError:
        timeref = "LOCAL"

    log.info("orb TIMEREF {0}".format(timeref))

    # The X, Y, Z position are for the START time
    mjds_TT = read_fits_event_mjds(hdulist[1])
    mjds_TT = mjds_TT * u.d
    # SC_POS is in meters in X,Y,Z Earth-centered Inertial (ECI) coordinates
    SC_POS = orb_dat.field("POSITION")
    X = SC_POS[:, 0] * u.km
    Y = SC_POS[:, 1] * u.km
    Z = SC_POS[:, 2] * u.km
    SC_VEL = orb_dat.field("VELOCITY")
    Vx = SC_VEL[:, 0] * u.km / u.s
    Vy = SC_VEL[:, 1] * u.km / u.s
    Vz = SC_VEL[:, 2] * u.km / u.s

    log.info(
        "Building orb table covering MJDs {0} to {1}".format(
            mjds_TT.min(), mjds_TT.max()
        )
    )
    orb_table = Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "orb"},
    )
    return orb_table


class NuSTARObs(SpecialLocation):
    """Observatory-derived class for the NuSTAR photon data.

    Note that this must be instantiated once to be put into the Observatory registry.

    Parameters
    ----------
    name: str
        Observatory name
    ft2name: str
        File name to read spacecraft position information from
    tt2tdb_mode: str
        Selection for mode to use for TT to TDB conversion.

        none
            Give no position to `astropy.time.Time`
        pint
            Use PINT routines for TT to TDB conversion.
        geo
            Give geocenter position to `astropy.time.Time`
        astropy
            Give spacecraft ITRF position to `astropy.time.Time`
    """

    def __init__(self, name, FPorbname, tt2tdb_mode="pint"):

        self.FPorb = load_orbit(FPorbname)
        # Now build the interpolator here:
        self.X = interp1d(self.FPorb["MJD_TT"], self.FPorb["X"])
        self.Y = interp1d(self.FPorb["MJD_TT"], self.FPorb["Y"])
        self.Z = interp1d(self.FPorb["MJD_TT"], self.FPorb["Z"])
        self.Vx = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vx"])
        self.Vy = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vy"])
        self.Vz = interp1d(self.FPorb["MJD_TT"], self.FPorb["Vz"])
        super(NuSTARObs, self).__init__(name=name, tt2tdb_mode=tt2tdb_mode)

    @property
    def timescale(self):
        return "tt"

    def earth_location_itrf(self, time=None):
        """Return NuSTAR spacecraft location in ITRF coordinates"""

        if self.tt2tdb_mode.lower().startswith("pint"):
            # log.warning('Using location=None for TT to TDB conversion')
            return None
        elif self.tt2tdb_mode.lower().startswith("astropy"):
            # First, interpolate ECI geocentric location from orbit file.
            # These are inertial coorinates aligned with ICRF
            log.warning("Performing GCRS to ITRS transformation")
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
        """Return position vector of NuSTAR in GCRS
        t is an astropy.Time or array of astropy.Time objects
        Returns a 3-vector of Quantities representing the position
        in GCRS coordinates.
        """
        return (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FPorb["X"].unit
        )

    def posvel(self, t, ephem):
        """Return position and velocity vectors of NuSTAR.
        t is an astropy.Time or array of astropy.Times
        """
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)
        # Now add vector from Earth to NuSTAR
        nustar_pos_geo = (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FPorb["X"].unit
        )
        nustar_vel_geo = (
            np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])
            * self.FPorb["Vx"].unit
        )
        nustar_posvel = PosVel(
            nustar_pos_geo, nustar_vel_geo, origin="earth", obj="nustar"
        )
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + nustar_posvel
