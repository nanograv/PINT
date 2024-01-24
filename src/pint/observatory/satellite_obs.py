"""Observatories at special (non-Earth) locations."""


import astropy.constants as const
import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.table import Table, vstack
from loguru import logger as log
from scipy.interpolate import InterpolatedUnivariateSpline

from pint.fits_utils import read_fits_event_mjds
from pint.observatory import bipm_default
from pint.observatory.special_locations import SpecialLocation
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel


def load_Fermi_FT2(ft2_filename):
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
        log.info(f"FT2 spacing is {str(dt.to(u.s))}")
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
    return Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "FT2"},
    )


def load_FPorbit(orbit_filename):
    """Load data from an (RXTE or NICER) FPorbit file

    Reads a FPorbit FITS file

    Parameters
    ----------
    orbit_filename : str
        Name of file to load

    Returns
    -------
    astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load orbit FITS file
    hdulist = pyfits.open(orbit_filename)
    # log.info('orb file HDU name is {0}'.format(hdulist[1].name))
    if hdulist[1].name not in ("ORBIT", "XTE_PE"):
        log.error(
            "NICER orb file first extension is {0}. It should be ORBIT".format(
                hdulist[1].name
            )
        )
    FPorbit_hdr = hdulist[1].header
    FPorbit_dat = hdulist[1].data

    log.info("Opened FPorbit FITS file {0}".format(orbit_filename))
    # TIMESYS should be 'TT'

    # TIMEREF should be 'LOCAL', since no delays are applied

    if "TIMESYS" not in FPorbit_hdr:
        log.warning("Keyword TIMESYS is missing. Assuming TT")
        timesys = "TT"
    else:
        timesys = FPorbit_hdr["TIMESYS"]
        log.debug("FPorbit TIMESYS {0}".format(timesys))

    if "TIMEREF" not in FPorbit_hdr:
        log.warning("Keyword TIMESYS is missing. Assuming TT")
        timeref = "LOCAL"
    else:
        timeref = FPorbit_hdr["TIMEREF"]
        log.debug("FPorbit TIMEREF {0}".format(timeref))

    mjds_TT = read_fits_event_mjds(hdulist[1])

    mjds_TT = mjds_TT * u.d
    log.debug("FPorbit spacing is {0}".format((mjds_TT[1] - mjds_TT[0]).to(u.s)))
    X = FPorbit_dat.field("X") * u.m
    Y = FPorbit_dat.field("Y") * u.m
    Z = FPorbit_dat.field("Z") * u.m
    Vx = FPorbit_dat.field("Vx") * u.m / u.s
    Vy = FPorbit_dat.field("Vy") * u.m / u.s
    Vz = FPorbit_dat.field("Vz") * u.m / u.s
    log.info(
        "Building FPorbit table covering MJDs {0} to {1}".format(
            mjds_TT.min(), mjds_TT.max()
        )
    )
    FPorbit_table = Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "FPorbit"},
    )
    # Make sure table is sorted by time
    log.debug("Sorting FPorbit table")
    FPorbit_table.sort("MJD_TT")

    good = np.diff(FPorbit_table["MJD_TT"]) > 0
    if not np.all(good):
        log.warning("The orbit table has duplicate entries. Please check.")
        good = np.concatenate((good, [True]))
        FPorbit_table = FPorbit_table[good]

    # Now delete any bad entries where the positions are 0.0
    idx = np.where(
        np.logical_and(FPorbit_table["X"] != 0.0, FPorbit_table["Y"] != 0.0)
    )[0]
    if len(idx) != len(FPorbit_table):
        log.warning(
            "Dropping {0} zero entries from FPorbit table".format(
                len(FPorbit_table) - len(idx)
            )
        )
        FPorbit_table = FPorbit_table[idx]
    return FPorbit_table


def load_nustar_orbit(orb_filename):
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
    return Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "orb"},
    )


def load_orbit(obs_name, orb_filename):
    """Generalized function to load one or more orbit files.

    Parameters
    ----------
    obs_name : str
        Observatory name. (Fermi, NICER, RXTE, and NuSTAR are valid.)
    orb_filename : str
        An FT2-like file tabulating orbit position.  If the first character
        is @, interpreted as a metafile listing multiple orbit files.

    Returns
    -------
    orb_table: astropy.table.Table
        A table containing entries MJD_TT, X, Y, Z, Vx, Vy, Vz
    """

    if str(orb_filename).startswith("@"):
        # Read multiple orbit files names
        fnames = [ll.strip() for ll in open(orb_filename[1:]).readlines()]
        orb_list = [load_orbit(obs_name, fn) for fn in fnames]
        full_orb = vstack(orb_list)
        # Make sure full table is sorted
        full_orb.sort("MJD_TT")
        return full_orb

    lower_name = obs_name.lower()
    if "fermi" in lower_name:
        return load_Fermi_FT2(orb_filename)
    elif "nicer" in lower_name:
        return load_FPorbit(orb_filename)
    elif "ixpe" in lower_name:
        return load_FPorbit(orb_filename)
    elif "xte" in lower_name:
        return load_FPorbit(orb_filename)
    elif "nustar" in lower_name:
        return load_nustar_orbit(orb_filename)
    else:
        raise ValueError(f"Unrecognized satellite observatory {obs_name}.")


class SatelliteObs(SpecialLocation):
    """Generalized class for high-energy photon data and tabulated position/velocity.

    Note that this must be instantiated once for each satellite to be put into the Observatory registry.

    Parameters
    ----------

    name: str
        Observatory name [Fermi, NICER, RXTE, NuSTAR]
    ft2name: str
        File name to read spacecraft position information from
    maxextrap: float
        Maximum minutes between a time and the closest S/C measurement.
    overwrite: bool
        Replace the entry in the observatory table.
    """

    def __init__(
        self,
        name,
        ft2name,
        maxextrap=2,
        include_gps=True,
        include_bipm=True,
        bipm_version=bipm_default,
        overwrite=False,
    ):
        super().__init__(
            name,
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            overwrite=overwrite,
        )
        self.FT2 = load_orbit(name, ft2name)

        # Now build the interpolator.  This extrapolation will fail quickly,
        # which is where maxextrap comes in.
        tt = self.FT2["MJD_TT"]
        self.X = InterpolatedUnivariateSpline(tt, self.FT2["X"], ext="extrapolate")
        self.Y = InterpolatedUnivariateSpline(tt, self.FT2["Y"], ext="extrapolate")
        self.Z = InterpolatedUnivariateSpline(tt, self.FT2["Z"], ext="extrapolate")
        self.Vx = InterpolatedUnivariateSpline(tt, self.FT2["Vx"], ext="extrapolate")
        self.Vy = InterpolatedUnivariateSpline(tt, self.FT2["Vy"], ext="extrapolate")
        self.Vz = InterpolatedUnivariateSpline(tt, self.FT2["Vz"], ext="extrapolate")
        self._geocenter = EarthLocation.from_geocentric(0.0 * u.m, 0.0 * u.m, 0.0 * u.m)
        self._maxextrap = maxextrap

    @property
    def timescale(self):
        return "tt"

    @property
    def tempo_code(self):
        return None

    def earth_location_itrf(self, time=None):
        return self._geocenter

    def _check_bounds(self, t):
        """Ensure t is within maxextrap of the closest S/C measurement.

        The purpose is to catch cases where there is missing S/C orbital
        information.  A common case would be providing an "FT2" file that
        is shorter than the photon data, or building an FT2 file that is
        missing a chunk.

        Parameters
        ----------
        t: an astropy.Time or array of astropy.Times
            Times to ensure are valid relative to S/C information.
        """
        ft2_tt = self.FT2["MJD_TT"]
        in_tt = np.atleast_1d(t.tt.mjd)
        i0 = np.searchsorted(ft2_tt, in_tt)
        i0 = np.clip(i0, 1, len(ft2_tt) - 1, out=i0)
        dright = np.abs(ft2_tt[i0] - in_tt)
        dleft = np.abs(ft2_tt[i0 - 1] - in_tt)
        min_duration = np.minimum(dright, dleft)
        if np.any(min_duration > (self._maxextrap / (60 * 24))):
            log.error(
                "Extrapolating S/C position by more than %d minutes!" % self._maxextrap
            )
            raise ValueError("Bad extrapolation of S/C file.")

    def _get_TDB_default(self, t, ephem):
        # Add in correction term to t.tdb equal to r.v / c^2
        vel = objPosVel_wrt_SSB("earth", t, ephem).vel
        pos = self.get_gcrs(t, ephem=ephem)
        dnom = const.c * const.c

        corr = ((pos[0] * vel[0] + pos[1] * vel[1] + pos[2] * vel[2]) / dnom).to(u.s)
        log.debug("\tTopocentric Correction:\t%s" % corr)

        return t.tdb + corr

    def get_gcrs(self, t, ephem=None):
        """Return position vector of S/C in GCRS.

        Returns a 3-vector of Quantities representing the position
        in GCRS coordinates.

        Parameters
        ----------
        t: an astropy.Time or array of astropy.Times
        """
        self._check_bounds(t)
        return (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FT2["X"].unit
        )

    def posvel(self, t, ephem, group=None):
        """Return position and velocity vectors of satellite, wrt SSB.

        These positions and velocites are in inertial coordinates
        (i.e. aligned with ICRS)

        t is an astropy.Time or array of astropy.Times
        """
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)
        # Now add vector from Earth to satellite
        sat_posvel = self.posvel_gcrs(t, ephem)
        return geo_posvel + sat_posvel

    def posvel_gcrs(self, t, ephem=None):
        """Return GCRS position and velocity vectors of S/C.

        t is an astropy.Time or array of astropy.Times
        """
        self._check_bounds(t)
        # Compute vector from Earth to satellite
        sat_pos_geo = (
            np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])
            * self.FT2["X"].unit
        )
        # log.debug("[{0}] sat_pos_geo {1}".format(self.name, sat_pos_geo[:, 0]))
        sat_vel_geo = (
            np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])
            * self.FT2["Vx"].unit
        )
        return PosVel(sat_pos_geo, sat_vel_geo, origin="earth", obj=self.name)


def get_satellite_observatory(name, ft2name, **kwargs):
    """Factory to get/instantiate a SatelliteObs.""

    Parameters
    ----------
    name: str
        Observatory name [Fermi, NICER, RXTE, NuSTAR]
    ft2name: str
        File name to read spacecraft position information from.
    """
    # Default maximum extrapolation is 2 minutes, which is suitable for
    # recognized observatories.  This factory can be used to set appropriate
    # values as new observatories are added.
    if "maxextrap" not in kwargs:
        kwargs["maxextrap"] = 2

    return SatelliteObs(name, ft2name, **kwargs)
