# special_locations.py
from __future__ import absolute_import, division, print_function

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline

from pint.fits_utils import read_fits_event_mjds
from pint.observatory.special_locations import SpacecraftObs
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel

# Special "site" location for Fermi satelite


def load_FT2(ft2_filename):
    """Load data from a Fermi FT2 file

        The contents of the FT2 file are described here:
        https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html#SpacecraftFile
        The coordinates are X, Y, Z in the ECI (Earth-Centered Inertial)
        frame. I (@paulray) **believe** this is the same as astropy's GCRS
        <http://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>,
        but this should be confirmed.

        Parameters
        ----------
        ft2_filename : str
            Name of file to load

        Returns
        -------
        astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load photon times from FT1 file
    hdulist = pyfits.open(ft2_filename)
    FT2_hdr = hdulist[1].header
    FT2_dat = hdulist[1].data

    log.info("Opened FT2 FITS file {0}".format(ft2_filename))
    # TIMESYS should be 'TT'
    # TIMEREF should be 'LOCAL', since no delays are applied
    timesys = FT2_hdr["TIMESYS"]
    log.info("FT2 TIMESYS {0}".format(timesys))
    timeref = FT2_hdr["TIMEREF"]
    log.info("FT2 TIMEREF {0}".format(timeref))

    # The X, Y, Z position are for the START time
    mjds_TT = read_fits_event_mjds(hdulist[1], timecolumn="START")
    mjds_TT = mjds_TT * u.d
    # SC_POS is in meters in X,Y,Z Earth-centered Inertial (ECI) coordinates
    SC_POS = FT2_dat.field("SC_POSITION")
    X = SC_POS[:, 0] * u.m
    Y = SC_POS[:, 1] * u.m
    Z = SC_POS[:, 2] * u.m
    try:
        # If available, get the velocities from the FT2 file
        SC_VEL = FT2_dat.field("SC_VELOCITY")
        Vx = SC_VEL[:, 0] * u.m / u.s
        Vy = SC_VEL[:, 1] * u.m / u.s
        Vz = SC_VEL[:, 2] * u.m / u.s
    except:
        # Otherwise, compute velocities by differentiation because FT2 does not have velocities
        # This is not the best way. Should fit an orbit and determine velocity from that.
        dt = mjds_TT[1] - mjds_TT[0]
        log.info("FT2 spacing is " + str(dt.to(u.s)))
        # Use "spacing" argument for gradient to handle nonuniform entries
        tt = mjds_TT.to(u.s).value
        Vx = np.gradient(X.value, tt) * u.m / u.s
        Vy = np.gradient(Y.value, tt) * u.m / u.s
        Vz = np.gradient(Z.value, tt) * u.m / u.s
    log.info(
        "Building FT2 table covering MJDs {0} to {1}".format(
            mjds_TT.min(), mjds_TT.max()
        )
    )
    FT2_table = Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "FT2"},
    )
    return FT2_table


class FermiObs(SpacecraftObs):
    """Observatory-derived class for the Fermi FT1 data.

    Note that this must be instantiated once to be put into the Observatory registry.

    Parameters
    ----------

    name: str
        Observatory name
    ft2name: str
        File name to read spacecraft position information from
    """

    def __init__(self, name, ft2name):
        self.FT2 = load_FT2(ft2name)
        # Now build the interpolator here:
        tt = self.FT2["MJD_TT"]
        self.X = InterpolatedUnivariateSpline(tt, self.FT2["X"])
        self.Y = InterpolatedUnivariateSpline(tt, self.FT2["Y"])
        self.Z = InterpolatedUnivariateSpline(tt, self.FT2["Z"])
        self.Vx = InterpolatedUnivariateSpline(tt, self.FT2["Vx"])
        self.Vy = InterpolatedUnivariateSpline(tt, self.FT2["Vy"])
        self.Vz = InterpolatedUnivariateSpline(tt, self.FT2["Vz"])
        super(FermiObs, self).__init__(name=name)

    def get_gcrs(self, t, ephem=None, grp=None):
        """Return position vector of Fermi in GCRS
        t is an astropy.Time or array of astropy.Time objects
        Returns a 3-vector of Quantities representing the position
        in GCRS coordinates.
        """
        return (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FT2["X"].unit
        )

    def posvel(self, t, ephem):
        """Return position and velocity vectors of Fermi, wrt SSB.

        These positions and velocites are in inertial coordinates
        (i.e. aligned with ICRS)

        t is an astropy.Time or array of astropy.Times
        """
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)
        # Now add vector from Earth to Fermi
        fermi_pos_geo = (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FT2["X"].unit
        )
        log.debug("fermi_pos_geo {0}".format(fermi_pos_geo[:, 0]))
        fermi_vel_geo = (
            np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])
            * self.FT2["Vx"].unit
        )
        fermi_posvel = PosVel(fermi_pos_geo, fermi_vel_geo, origin="earth", obj="Fermi")
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + fermi_posvel
