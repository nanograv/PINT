"""Astrometric models for describing pulsar sky positions."""

import copy
import sys
import warnings
from typing import Optional, List, Union

import astropy.constants as const
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.time import Time

from loguru import logger as log

from erfa import ErfaWarning, pmsafe
from pint import ls
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    floatParameter,
    strParameter,
)
import pint.toa
from pint.models.timing_model import DelayComponent, MissingParameter
from pint.pulsar_ecliptic import OBL, PulsarEcliptic
from pint.utils import add_dummy_distance, remove_dummy_distance

astropy_version = sys.modules["astropy"].__version__
mas_yr = u.mas / u.yr

__all__ = [
    "AstrometryEquatorial",
    "AstrometryEcliptic",
    "Astrometry",
]


class Astrometry(DelayComponent):
    """Common tools for astrometric calculations."""

    register = False
    category = "astrometry"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="POSEPOCH",
                description="Reference epoch for position",
                time_scale="tdb",
            )
        )

        self.add_param(
            floatParameter(name="PX", units="mas", value=0.0, description="Parallax")
        )

        self.delay_funcs_component += [self.solar_system_geometric_delay]
        self.register_deriv_funcs(self.d_delay_astrometry_d_PX, "PX")

    def ssb_to_psb_xyz_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> u.Quantity:
        """Returns unit vector(s) from SSB to pulsar system barycenter under ICRS.

        If epochs (MJD) are given, proper motion is included in the calculation.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        np.ndarray :
            (len(epoch), 3) array of unit vectors
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?

        # this is somewhat slow, since it repeatedly created different SkyCoord Objects
        # but for consistency only change the method in the subclasses below
        return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

    def ssb_to_psb_xyz_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = None
    ) -> u.Quantity:
        """Returns unit vector(s) from SSB to pulsar system barycenter under Ecliptic coordinates.

        If epochs (MJD) are given, proper motion is included in the calculation.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed
        ecl : str, optional
            Obliquity (IERS2010 by default)

        Returns
        -------
        np.ndarray :
            (len(epoch), 3) array of unit vectors
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.coords_as_ECL(epoch=epoch, ecl=ecl).cartesian.xyz.transpose()

    def sun_angle(
        self, toas: pint.toa.TOAs, heliocenter: bool = True, also_distance: bool = False
    ) -> np.ndarray:
        """Compute the pulsar-observatory-Sun angle.

        This is the angle between the center of the Sun and the direction to
        the pulsar, as seen from the observatory (for each TOA).

        This angle takes into account the motion of the Sun around the solar system barycenter.

        Parameters
        ----------
        toas: :class:`pint.toa.TOAs`
            The pulse arrival times at which to evaluate the sun angle.
        heliocenter: bool
            Whether to use the Sun's actual position (the heliocenter) or
            the solar system barycenter. The latter may be useful for
            comparison with other software.
        also_distance: bool
            If True, also return the observatory-Sun distance as a Quantity

        Returns
        -------
        array
            The angle in radians
        """
        tbl = toas.table

        if heliocenter:
            osv = tbl["obs_sun_pos"].quantity.copy()
        else:
            osv = -tbl["ssb_obs_pos"].quantity.copy()
        psr_vec = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"])
        r = (osv**2).sum(axis=1) ** 0.5
        osv /= r[:, None]
        cos = (osv * psr_vec).sum(axis=1)
        return (np.arccos(cos), r) if also_distance else np.arccos(cos)

    def barycentric_radio_freq(self, toas):
        raise NotImplementedError

    def solar_system_geometric_delay(
        self, toas: pint.toa.TOAs, acc_delay=None
    ) -> u.Quantity:
        """Returns geometric delay (in sec) due to position of site in
        solar system.  This includes Roemer delay and parallax.

        NOTE: currently assumes XYZ location of TOA relative to SSB is
        available as 3-vector toa.xyz, in units of light-seconds.
        """
        tbl = toas.table
        delay = np.zeros(len(toas))
        # c selects the non-barycentric TOAs that need actual calculation
        c = np.logical_and.reduce(tbl["ssb_obs_pos"] != 0, axis=1)
        if np.any(c):
            L_hat = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"][c].astype(np.float64))
            re_dot_L = np.sum(tbl["ssb_obs_pos"][c] * L_hat, axis=1)
            delay[c] = -re_dot_L.to(ls).value
            if self.PX.value != 0.0:
                L = (1.0 / self.PX.value) * u.kpc
                # TODO: np.sum currently loses units in some cases...
                re_sqr = (
                    np.sum(tbl["ssb_obs_pos"][c] ** 2, axis=1)
                    * tbl["ssb_obs_pos"].unit ** 2
                )
                delay[c] += (
                    (0.5 * (re_sqr / L) * (1.0 - re_dot_L**2 / re_sqr)).to(ls).value
                )
        return delay * u.second

    def get_d_delay_quantities(self, toas: pint.toa.TOAs) -> dict:
        """Calculate values needed for many d_delay_d_param functions"""
        # TODO: Should delay not have units of u.second?
        delay = self._parent.delay(toas)

        # TODO: tbl['tdbld'].quantity should have units of u.day
        # NOTE: Do we need to include the delay here?
        tbl = toas.table
        rd = {"epoch": tbl["tdbld"].quantity * u.day}
        # Distance from SSB to observatory, and from SSB to psr
        ssb_obs = tbl["ssb_obs_pos"].quantity
        ssb_psr = self.ssb_to_psb_xyz_ICRS(epoch=np.array(rd["epoch"]))

        # Cartesian coordinates, and derived quantities
        rd["ssb_obs_r"] = np.sqrt(np.sum(ssb_obs**2, axis=1))
        rd["ssb_obs_z"] = ssb_obs[:, 2]
        rd["ssb_obs_xy"] = np.sqrt(ssb_obs[:, 0] ** 2 + ssb_obs[:, 1] ** 2)
        rd["ssb_obs_x"] = ssb_obs[:, 0]
        rd["ssb_obs_y"] = ssb_obs[:, 1]
        rd["in_psr_obs"] = np.sum(ssb_obs * ssb_psr, axis=1)

        # Earth right ascension and declination
        rd["earth_dec"] = np.arctan2(rd["ssb_obs_z"], rd["ssb_obs_xy"])
        rd["earth_ra"] = np.arctan2(rd["ssb_obs_y"], rd["ssb_obs_x"])

        return rd

    def get_params_as_ICRS(self):
        raise NotImplementedError

    def get_psr_coords(self, epoch=None):
        raise NotImplementedError

    def d_delay_astrometry_d_PX(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt PX

        Roughly following Smart, 1977, chapter 9.

        px_r:   Extra distance to Earth, wrt SSB, from pulsar
        r_e:    Position of earth (vector) wrt SSB
        u_p:    Unit vector from SSB pointing to pulsar
        t_d:    Parallax delay
        c:      Speed of light
        delta:  Parallax

        The parallax delay is due to a distance orthogonal to the line of sight
        to the pulsar from the SSB:

        px_r = sqrt( r_e**2 - (r_e.u_p)**2 ),

        with delay

        t_d = 0.5 * px_r * delta'/ c,  and delta = delta' * px_r / (1 AU)

        """
        rd = self.get_d_delay_quantities(toas)

        px_r = np.sqrt(rd["ssb_obs_r"] ** 2 - rd["in_psr_obs"] ** 2)
        dd_dpx = 0.5 * (px_r**2 / (u.AU * const.c)) * (u.mas / u.radian)

        # We want to return sec / mas
        return dd_dpx.decompose(u.si.bases) / u.mas

    def d_delay_astrometry_d_POSEPOCH(self, toas, param="", acc_delay=None):
        """Calculate the derivative wrt POSEPOCH"""
        pass

    def change_posepoch(self, new_epoch):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: float or `astropy.Time` object
            The new POSEPOCH value.
        """
        raise NotImplementedError

    def as_ECL(self, epoch=None, ecl="IERS2010"):
        raise NotImplementedError

    def as_ICRS(self, epoch=None, ecl="IERS2010"):
        raise NotImplementedError


class AstrometryEquatorial(Astrometry):
    """Astrometry in equatorial coordinates.

    Parameters supported:

    .. paramtable::
        :class: pint.models.astrometry.AstrometryEquatorial
    """

    register = True

    def __init__(self):
        super().__init__()
        self.add_param(
            AngleParameter(
                name="RAJ",
                units="H:M:S",
                description="Right ascension (J2000)",
                aliases=["RA"],
            )
        )

        self.add_param(
            AngleParameter(
                name="DECJ",
                units="D:M:S",
                description="Declination (J2000)",
                aliases=["DEC"],
            )
        )

        self.add_param(
            floatParameter(
                name="PMRA",
                units="mas/year",
                description="Proper motion in RA",
                value=0.0,
            )
        )

        self.add_param(
            floatParameter(
                name="PMDEC",
                units="mas/year",
                description="Proper motion in DEC",
                value=0.0,
            )
        )
        self.set_special_params(["RAJ", "DECJ", "PMRA", "PMDEC"])
        for param in ["RAJ", "DECJ", "PMRA", "PMDEC"]:
            deriv_func_name = f"d_delay_astrometry_d_{param}"
            func = getattr(self, deriv_func_name)
            self.register_deriv_funcs(func, param)

    def validate(self):
        """Validate the input parameter."""
        super().validate()
        # RA/DEC are required
        for p in ("RAJ", "DECJ"):
            if getattr(self, p).value is None:
                raise MissingParameter("Astrometry", p)
        # Check for POSEPOCH
        if (
            np.any(self.PMRA.quantity != 0) or np.any(self.PMDEC.quantity != 0)
        ) and self.POSEPOCH.quantity is None:
            if self._parent.PEPOCH.quantity is None:
                raise MissingParameter(
                    "AstrometryEquatorial",
                    "POSEPOCH",
                    "POSEPOCH or PEPOCH are required if PM is set.",
                )
            else:
                self.POSEPOCH.quantity = self._parent.PEPOCH.quantity

    def print_par(self, format: str = "pint") -> str:
        result = ""
        print_order = ["RAJ", "DECJ", "PMRA", "PMDEC", "PX", "POSEPOCH"]
        for p in print_order:
            par = getattr(self, p)
            if par.quantity is not None:
                result += getattr(self, p).as_parfile_line(format=format)
        return result

    def barycentric_radio_freq(self, toas: pint.toa.TOAs) -> u.Quantity:
        """Return radio frequencies (MHz) of the toas corrected for Earth motion"""
        tbl = toas.table
        L_hat = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"].astype(np.float64))
        v_dot_L_array = np.sum(tbl["ssb_obs_vel"] * L_hat, axis=1)
        return tbl["freq"] * (1.0 - v_dot_L_array / const.c)

    def get_psr_coords(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Returns pulsar sky coordinates as an astropy ICRS object instance.

        Parameters
        ----------
        epoch: `astropy.time.Time` or Float, optional
            new epoch for position.  If Float, MJD(TDB) is assumed

        Returns
        -------
        position
            ICRS SkyCoord object optionally with proper motion applied

        If epoch (MJD) is specified, proper motion is included to return
        the position at the given epoch.

        """
        if epoch is None or (self.PMRA.value == 0.0 and self.PMDEC.value == 0.0):
            return coords.SkyCoord(
                ra=self.RAJ.quantity,
                dec=self.DECJ.quantity,
                pm_ra_cosdec=self.PMRA.quantity,
                pm_dec=self.PMDEC.quantity,
                obstime=self.POSEPOCH.quantity,
                frame=coords.ICRS,
            )
        newepoch = (
            epoch if isinstance(epoch, Time) else Time(epoch, scale="tdb", format="mjd")
        )
        position_now = add_dummy_distance(self.get_psr_coords())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            # for the most part the dummy distance should remove any potential erfa warnings
            # but for some very large proper motions that does not quite work
            # so we catch the warnings
            position_then = position_now.apply_space_motion(new_obstime=newepoch)
            position_then = remove_dummy_distance(position_then)

        return position_then

    def coords_as_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Return the pulsar's ICRS coordinates as an astropy coordinate object.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        return self.get_psr_coords(epoch)

    def coords_as_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = None
    ) -> coords.SkyCoord:
        """Return the pulsar's ecliptic coordinates as an astropy coordinate object.

        The value used for the obliquity of the ecliptic can be controlled with the
        `ecl` keyword, which should be one of the codes listed in `ecliptic.dat`.
        If `ecl` is left unspecified, the global default IERS2010 will be used.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed
        ecl : str

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        if ecl is None:
            log.debug("ECL not specified; using IERS2010.")
            ecl = "IERS2010"

        pos_icrs = self.get_psr_coords(epoch=epoch)
        return pos_icrs.transform_to(PulsarEcliptic(ecl=ecl))

    def coords_as_GAL(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Return the pulsar's galactic coordinates as an astropy coordinate object.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        pos_icrs = self.get_psr_coords(epoch=epoch)
        return pos_icrs.transform_to(coords.Galactic)

    def get_params_as_ICRS(self) -> dict:
        return {
            "RAJ": self.RAJ.quantity,
            "DECJ": self.DECJ.quantity,
            "PMRA": self.PMRA.quantity,
            "PMDEC": self.PMDEC.quantity,
        }

    def ssb_to_psb_xyz_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> u.Quantity:
        """Returns unit vector(s) from SSB to pulsar system barycenter under ICRS.

        If epochs (MJD) are given, proper motion is included in the calculation.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        np.ndarray :
            (len(epoch), 3) array of unit vectors
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?

        # this was somewhat slow, since it repeatedly created different SkyCoord Objects
        # return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

        # Instead look at what https://docs.astropy.org/en/stable/_modules/astropy/coordinates/sky_coordinate.html#SkyCoord.apply_space_motion
        # does, which is to use https://github.com/liberfa/erfa/blob/master/src/starpm.c
        # and then just use the relevant pieces of that
        if epoch is None or (self.PMRA.quantity == 0 and self.PMDEC.quantity == 0):
            return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

        if isinstance(epoch, Time):
            jd1 = epoch.jd1
            jd2 = epoch.jd2
        elif isinstance(epoch, u.Quantity):
            epoch = Time(epoch, format="mjd", scale="tdb")
            jd1 = epoch.jd1
            jd2 = epoch.jd2
        else:
            # assume MJD
            jd1 = 2400000.5
            jd2 = epoch
        # compared to the general case above we can assume that the coordinates are ICRS
        # so just access those components
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            # note that starpm wants mu_alpha not mu_alpha * cos(delta)
            starpmout = pmsafe(
                self.RAJ.quantity.to_value(u.radian),
                self.DECJ.quantity.to_value(u.radian),
                self.PMRA.quantity.to_value(u.radian / u.yr)
                / np.cos(self.DECJ.quantity).value,
                self.PMDEC.quantity.to_value(u.radian / u.yr),
                self.PX.quantity.to_value(u.arcsec),
                0.0,
                self.POSEPOCH.quantity.jd1,
                self.POSEPOCH.quantity.jd2,
                jd1,
                jd2,
            )
        # ra,dec now in radians
        ra, dec = starpmout[0], starpmout[1]
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
        return u.Quantity([x, y, z]).T

    def d_delay_astrometry_d_RAJ(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt RAJ

        For the RAJ and DEC derivatives, use the following approximate model for
        the pulse delay. (Inner-product between two Cartesian vectors):

            - de = Earth declination (wrt SSB)
            - ae = Earth right ascension
            - dp = pulsar declination
            - aa = pulsar right ascension
            - r = distance from SSB to Earh
            - c = speed of light

        delay = r*[cos(de)*cos(dp)*cos(ae-aa)+sin(de)*sin(dp)]/c
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        geom = (
            np.cos(rd["earth_dec"]) * np.cos(psr_dec) * np.sin(psr_ra - rd["earth_ra"])
        )
        dd_draj = rd["ssb_obs_r"] * geom / (const.c * u.radian)

        return dd_draj.decompose(u.si.bases)

    def d_delay_astrometry_d_DECJ(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt DECJ

        Definitions as in d_delay_d_RAJ
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        geom = np.cos(rd["earth_dec"]) * np.sin(psr_dec) * np.cos(
            psr_ra - rd["earth_ra"]
        ) - np.sin(rd["earth_dec"]) * np.cos(psr_dec)
        dd_ddecj = rd["ssb_obs_r"] * geom / (const.c * u.radian)

        return dd_ddecj.decompose(u.si.bases)

    def d_delay_astrometry_d_PMRA(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt PMRA

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar RA
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity

        te = rd["epoch"] - self.POSEPOCH.quantity.tdb.mjd_long * u.day
        geom = np.cos(rd["earth_dec"]) * np.sin(psr_ra - rd["earth_ra"])

        deriv = rd["ssb_obs_r"] * geom * te / (const.c * u.radian)
        dd_dpmra = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmra.decompose(u.si.bases) / (u.mas / u.year)

    def d_delay_astrometry_d_PMDEC(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt PMDEC

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar DEC
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        te = rd["epoch"] - self.POSEPOCH.quantity.tdb.mjd_long * u.day
        geom = np.cos(rd["earth_dec"]) * np.sin(psr_dec) * np.cos(
            psr_ra - rd["earth_ra"]
        ) - np.cos(psr_dec) * np.sin(rd["earth_dec"])

        deriv = rd["ssb_obs_r"] * geom * te / (const.c * u.radian)
        dd_dpmdec = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmdec.decompose(u.si.bases) / (u.mas / u.year)

    def change_posepoch(self, new_epoch: Union[float, u.Quantity, Time]):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: `astropy.time.Time` or float or astropy.units.Quantity
            If float or Quantity, MJD(TDB) is assumed.
            Note that uncertainties are not adjusted.
            The new POSEPOCH value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        if self.POSEPOCH.value is None:
            raise ValueError("POSEPOCH is not currently set.")
        new_coords = self.get_psr_coords(new_epoch.mjd_long)
        self.RAJ.value = new_coords.ra
        self.DECJ.value = new_coords.dec
        self.POSEPOCH.value = new_epoch

    def as_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> "AstrometryEquatorial":
        """Return pint.models.astrometry.Astrometry object in ICRS frame.

        Parameters
        ----------
        epoch : `astropy.time.Time` or float or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.

        Returns
        -------
        pint.models.astrometry.AstrometryEquatorial
        """
        m = copy.deepcopy(self)
        if epoch is not None:
            m.change_posepoch(epoch)
        return m

    def as_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = "IERS2010"
    ) -> "AstrometryEcliptic":
        """Return pint.models.astrometry.Astrometry object in PulsarEcliptic frame.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.
        ecl : str, optional
            Obliquity for PulsarEcliptic frame

        Returns
        -------
        pint.models.astrometry.AstrometryEcliptic
        """
        m_ecl = AstrometryEcliptic()

        # transfer over parallax and POSEPOCH: don't need to change
        m_ecl.PX = self.PX
        m_ecl.POSEPOCH = self.POSEPOCH
        # get ELONG, ELAT, PM
        c = self.coords_as_ECL(epoch=epoch, ecl=ecl)
        m_ecl.ELONG.quantity = c.lon
        m_ecl.ELAT.quantity = c.lat
        m_ecl.PMELONG.quantity = c.pm_lon_coslat
        m_ecl.PMELAT.quantity = c.pm_lat
        m_ecl.ECL.value = ecl

        # use fake proper motions to convert uncertainties on ELONG, ELAT
        # assume that ELONG uncertainty does not include cos(ELAT)
        # and that the RA uncertainty does not include cos(DEC)
        # put it in here as pm_ra_cosdec since astropy complains otherwise
        dt = 1 * u.yr
        c = coords.SkyCoord(
            ra=self.RAJ.quantity,
            dec=self.DECJ.quantity,
            obstime=self.POSEPOCH.quantity,
            pm_ra_cosdec=(
                self.RAJ.uncertainty * np.cos(self.DECJ.quantity) / dt
                if self.RAJ.uncertainty is not None
                else 0 * self.RAJ.units / dt
            ),
            pm_dec=(
                self.DECJ.uncertainty / dt
                if self.DECJ.uncertainty is not None
                else 0 * self.DECJ.units / dt
            ),
            frame=coords.ICRS,
        )
        c_ECL = c.transform_to(PulsarEcliptic(ecl=ecl))
        m_ecl.ELONG.uncertainty = c_ECL.pm_lon_coslat * dt / np.cos(c_ECL.lat)
        m_ecl.ELAT.uncertainty = c_ECL.pm_lat * dt
        # use fake proper motions to convert uncertainties on proper motion
        # assume that the PM_RA _does_ include cos(DEC)
        c = coords.SkyCoord(
            ra=self.RAJ.quantity,
            dec=self.DECJ.quantity,
            obstime=self.POSEPOCH.quantity,
            pm_ra_cosdec=(
                self.PMRA.uncertainty
                if self.PMRA.uncertainty is not None
                else 0 * self.PMRA.units
            ),
            pm_dec=(
                self.PMDEC.uncertainty
                if self.PMDEC.uncertainty is not None
                else 0 * self.PMDEC.units
            ),
            frame=coords.ICRS,
        )
        c_ECL = c.transform_to(PulsarEcliptic(ecl=ecl))
        m_ecl.PMELONG.uncertainty = c_ECL.pm_lon_coslat
        m_ecl.PMELAT.uncertainty = c_ECL.pm_lat
        # freeze comparable parameters
        m_ecl.ELONG.frozen = self.RAJ.frozen
        m_ecl.ELAT.frozen = self.DECJ.frozen
        m_ecl.PMELONG.frozen = self.PMRA.frozen
        m_ecl.PMELAT.frozen = self.PMDEC.frozen

        return m_ecl


class AstrometryEcliptic(Astrometry):
    """Astrometry in ecliptic coordinates.

    Parameters supported:

    .. paramtable::
        :class: pint.models.astrometry.AstrometryEcliptic
    """

    register = True

    def __init__(self):
        super().__init__()
        self.add_param(
            AngleParameter(
                name="ELONG",
                units="deg",
                description="Ecliptic longitude",
                aliases=["LAMBDA"],
            )
        )

        self.add_param(
            AngleParameter(
                name="ELAT",
                units="deg",
                description="Ecliptic latitude",
                aliases=["BETA"],
            )
        )

        self.add_param(
            floatParameter(
                name="PMELONG",
                units="mas/year",
                description="Proper motion in ecliptic longitude",
                aliases=["PMLAMBDA"],
                value=0.0,
            )
        )

        self.add_param(
            floatParameter(
                name="PMELAT",
                units="mas/year",
                description="Proper motion in ecliptic latitude",
                aliases=["PMBETA"],
                value=0.0,
            )
        )

        self.add_param(
            strParameter(
                name="ECL",
                value="IERS2010",
                description="Obliquity of the ecliptic (reference)",
            )
        )

        self.set_special_params(["ELONG", "ELAT", "PMELONG", "PMELAT"])
        for param in ["ELAT", "ELONG", "PMELAT", "PMELONG"]:
            deriv_func_name = f"d_delay_astrometry_d_{param}"
            func = getattr(self, deriv_func_name)
            self.register_deriv_funcs(func, param)

    def validate(self):
        """Validate Ecliptic coordinate parameter inputs."""
        super().validate()
        # ELONG/ELAT are required
        for p in ("ELONG", "ELAT"):
            if getattr(self, p).value is None:
                raise MissingParameter("AstrometryEcliptic", p)
        # Check for POSEPOCH
        if (
            np.any(self.PMELONG.value != 0) or np.any(self.PMELAT.value != 0)
        ) and self.POSEPOCH.quantity is None:
            if self._parent.PEPOCH.quantity is None:
                raise MissingParameter(
                    "Astrometry",
                    "POSEPOCH",
                    "POSEPOCH or PEPOCH are required if PM is set.",
                )
            else:
                self.POSEPOCH.quantity = self._parent.PEPOCH.quantity

    def barycentric_radio_freq(self, toas: pint.toa.TOAs) -> u.Quantity:
        """Return radio frequencies (MHz) of the toas corrected for Earth motion"""
        if "ssb_obs_vel_ecl" not in toas.table.colnames:
            obliquity = OBL[self.ECL.value]
            toas.add_vel_ecl(obliquity)
        tbl = toas.table
        L_hat = self.ssb_to_psb_xyz_ECL(epoch=tbl["tdbld"].astype(np.float64))
        v_dot_L_array = np.sum(tbl["ssb_obs_vel_ecl"] * L_hat, axis=1)
        return tbl["freq"] * (1.0 - v_dot_L_array / const.c)

    def get_psr_coords(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Returns pulsar sky coordinates as an astropy ecliptic coordinate instance.

        Parameters
        ----------
        epoch: `astropy.time.Time` or float or astropy.units.Quantity, optional
            new epoch for position.  If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        position
            PulsarEcliptic SkyCoord object optionally with proper motion applied

        If epoch (MJD) is specified, proper motion is included to return
        the position at the given epoch.
        """
        try:
            obliquity = OBL[self.ECL.value]
        except KeyError as e:
            raise ValueError(
                f"No obliquity {str(self.ECL.value)} provided. Check your pint/datafile/ecliptic.dat file."
            ) from e
        if epoch is None or (self.PMELONG.value == 0.0 and self.PMELAT.value == 0.0):
            # Compute only once
            return coords.SkyCoord(
                obliquity=obliquity,
                lon=self.ELONG.quantity,
                lat=self.ELAT.quantity,
                pm_lon_coslat=self.PMELONG.quantity,
                pm_lat=self.PMELAT.quantity,
                obstime=self.POSEPOCH.quantity,
                frame=PulsarEcliptic,
            )
            # Compute for each time because there is proper motion
        newepoch = (
            epoch if isinstance(epoch, Time) else Time(epoch, scale="tdb", format="mjd")
        )
        position_now = add_dummy_distance(self.get_psr_coords())
        with warnings.catch_warnings():
            # This is a fake position, no point ERFA warning the user it's bogus
            warnings.filterwarnings("ignore", r".*distance overridden", ErfaWarning)
            position_then = position_now.apply_space_motion(new_obstime=newepoch)
        return remove_dummy_distance(position_then)

    def coords_as_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Return the pulsar's ICRS coordinates as an astropy coordinate object.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        pos_ecl = self.get_psr_coords(epoch=epoch)
        return pos_ecl.transform_to(coords.ICRS)

    def coords_as_GAL(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> coords.SkyCoord:
        """Return the pulsar's galactic coordinates as an astropy coordinate object.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        pos_ecl = self.get_psr_coords(epoch=epoch)
        return pos_ecl.transform_to(coords.Galactic)

    def coords_as_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = None
    ) -> coords.SkyCoord:
        """Return the pulsar's ecliptic coordinates as an astropy coordinate object.

        The value used for the obliquity of the ecliptic can be controlled with the
        `ecl` keyword, which should be one of the codes listed in `ecliptic.dat`.
        If `ecl` is left unspecified, the model's ECL parameter will be used.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed
        ecl : str

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        pos_ecl = self.get_psr_coords(epoch=epoch)
        if ecl is not None:
            pos_ecl = pos_ecl.transform_to(PulsarEcliptic(ecl=ecl))
        return pos_ecl

    def ssb_to_psb_xyz_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = None
    ) -> u.Quantity:
        """Returns unit vector(s) from SSB to pulsar system barycenter under ECL.

        If epochs (MJD) are given, proper motion is included in the calculation.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed
        ecl : str, optional
            Obliquity (IERS2010 by default)

        Returns
        -------
        np.ndarray :
            (len(epoch), 3) array of unit vectors
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?

        # this was somewhat slow, since it repeatedly created different SkyCoord Objects
        # return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

        # Instead look at what https://docs.astropy.org/en/stable/_modules/astropy/coordinates/sky_coordinate.html#SkyCoord.apply_space_motion
        # does, which is to use https://github.com/liberfa/erfa/blob/master/src/starpm.c
        # and then just use the relevant pieces of that

        # but we need to check that the obliquity is the same
        if ecl is not None and ecl != self.ECL.quantity:
            return super().ssb_to_psb_xyz_ECL(epoch=epoch, ecl=ecl)

        if ecl is None:
            log.debug("ECL not specified; using IERS2010.")
            ecl = "IERS2010"
        if epoch is None or (self.PMELONG.value == 0 and self.PMELAT.value == 0):
            return self.coords_as_ECL(epoch=epoch, ecl=ecl).cartesian.xyz.transpose()
        if isinstance(epoch, Time):
            jd1 = epoch.jd1
            jd2 = epoch.jd2
        elif isinstance(epoch, u.Quantity):
            epoch = Time(epoch, format="mjd", scale="tdb")
            jd1 = epoch.jd1
            jd2 = epoch.jd2
        else:
            jd1 = 2400000.5
            jd2 = epoch
        # compared to the general case above we can assume that the coordinates are ECL
        # so just access those components
        lon = self.ELONG.quantity.to_value(u.radian)
        lat = self.ELAT.quantity.to_value(u.radian)
        pm_lon = (
            self.PMELONG.quantity.to_value(u.radian / u.yr)
            / np.cos(self.ELAT.quantity).value
        )
        pm_lat = self.PMELAT.quantity.to_value(u.radian / u.yr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            # note that pmsafe wants mu_lon not mu_lon * cos(lat)
            starpmout = pmsafe(
                lon,
                lat,
                pm_lon,
                pm_lat,
                self.PX.quantity.to_value(u.arcsec),
                0.0,
                self.POSEPOCH.quantity.jd1,
                self.POSEPOCH.quantity.jd2,
                jd1,
                jd2,
            )
        # lon,lat now in radians
        lon, lat = starpmout[0], starpmout[1]
        x = np.cos(lon) * np.cos(lat)
        y = np.sin(lon) * np.cos(lat)
        z = np.sin(lat)
        return u.Quantity([x, y, z]).T

    def get_d_delay_quantities_ecliptical(self, toas: pint.toa.TOAs) -> u.Quantity:
        """Calculate values needed for many d_delay_d_param functions."""
        # TODO: Move all these calculations in a separate class for elegance

        # From the earth_ra dec to earth_elong and elat
        try:
            obliquity = OBL[self.ECL.value]
        except KeyError as e:
            raise ValueError(
                (
                    f"No obliquity {self.ECL.value}" + " provided. "
                    "Check your pint/datafile/ecliptic.dat file."
                )
            ) from e

        rd = self.get_d_delay_quantities(toas)
        coords_icrs = coords.ICRS(ra=rd["earth_ra"], dec=rd["earth_dec"])
        coords_elpt = coords_icrs.transform_to(PulsarEcliptic(obliquity=obliquity))
        rd["earth_elong"] = coords_elpt.lon
        rd["earth_elat"] = coords_elpt.lat

        return rd

    def get_params_as_ICRS(self) -> dict:
        pv_ECL = self.get_psr_coords()
        pv_ICRS = pv_ECL.transform_to(coords.ICRS)
        return {
            "RAJ": pv_ICRS.ra.to(u.hourangle),
            "DECJ": pv_ICRS.dec,
            "PMRA": pv_ICRS.pm_ra_cosdec,
            "PMDEC": pv_ICRS.pm_dec,
        }

    def d_delay_astrometry_d_ELONG(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt RAJ.

        For the RAJ and DEC derivatives, use the following approximate model for
        the pulse delay. (Inner-product between two Cartesian vectors)

        de = Earth declination (wrt SSB)
        ae = Earth right ascension
        dp = pulsar declination
        aa = pulsar right ascension
        r = distance from SSB to Earth
        c = speed of light

        delay = r*[cos(de)*cos(dp)*cos(ae-aa)+sin(de)*sin(dp)]/c

        elate = Earth elat (wrt SSB)
        elonge = Earth elong
        elatp = pulsar elat
        elongp = pulsar elong
        r = distance from SSB to Earth
        c = speed of light

        delay = r*[cos(elate)*cos(elatp)*cos(elonge-elongp)+sin(elate)*sin(elatp)]/c
        """
        rd = self.get_d_delay_quantities_ecliptical(toas)

        psr_elong = self.ELONG.quantity
        psr_elat = self.ELAT.quantity

        geom = (
            np.cos(rd["earth_elat"])
            * np.cos(psr_elat)
            * np.sin(psr_elong - rd["earth_elong"])
        )
        dd_delong = rd["ssb_obs_r"] * geom / (const.c * u.radian)

        return dd_delong.decompose(u.si.bases)

    def d_delay_astrometry_d_ELAT(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt DECJ

        Definitions as in d_delay_d_RAJ
        """
        rd = self.get_d_delay_quantities_ecliptical(toas)

        psr_elong = self.ELONG.quantity
        psr_elat = self.ELAT.quantity

        geom = np.cos(rd["earth_elat"]) * np.sin(psr_elat) * np.cos(
            psr_elong - rd["earth_elong"]
        ) - np.sin(rd["earth_elat"]) * np.cos(psr_elat)
        dd_delat = rd["ssb_obs_r"] * geom / (const.c * u.radian)

        return dd_delat.decompose(u.si.bases)

    def d_delay_astrometry_d_PMELONG(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt PMRA

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar RA
        """
        rd = self.get_d_delay_quantities_ecliptical(toas)

        psr_elong = self.ELONG.quantity
        psr_elat = self.ELAT.quantity

        te = rd["epoch"] - self.POSEPOCH.quantity.tdb.mjd_long * u.day
        geom = np.cos(rd["earth_elat"]) * np.sin(psr_elong - rd["earth_elong"])

        deriv = rd["ssb_obs_r"] * geom * te / (const.c * u.radian)
        dd_dpmelong = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmelong.decompose(u.si.bases) / (u.mas / u.year)

    def d_delay_astrometry_d_PMELAT(
        self, toas: pint.toa.TOAs, param="", acc_delay=None
    ) -> u.Quantity:
        """Calculate the derivative wrt PMDEC

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar DEC
        """
        rd = self.get_d_delay_quantities_ecliptical(toas)

        psr_elong = self.ELONG.quantity
        psr_elat = self.ELAT.quantity

        te = rd["epoch"] - self.POSEPOCH.quantity.tdb.mjd_long * u.day
        geom = np.cos(rd["earth_elat"]) * np.sin(psr_elat) * np.cos(
            psr_elong - rd["earth_elong"]
        ) - np.cos(psr_elat) * np.sin(rd["earth_elat"])

        deriv = rd["ssb_obs_r"] * geom * te / (const.c * u.radian)
        dd_dpmelat = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmelat.decompose(u.si.bases) / (u.mas / u.year)

    def print_par(self, format: str = "pint") -> str:
        result = ""
        print_order = ["ELONG", "ELAT", "PMELONG", "PMELAT", "PX", "ECL", "POSEPOCH"]
        for p in print_order:
            par = getattr(self, p)
            if par.quantity is not None:
                result += getattr(self, p).as_parfile_line(format=format)
        return result

    def change_posepoch(self, new_epoch: Union[float, u.Quantity, Time]):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: float or `astropy.Time` or `astropy.units.Quantity` object
            The new POSEPOCH value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        if self.POSEPOCH.value is None:
            raise ValueError("POSEPOCH is not currently set.")
        new_coords = self.get_psr_coords(new_epoch.mjd_long)
        self.ELONG.value = new_coords.lon
        self.ELAT.value = new_coords.lat
        self.POSEPOCH.value = new_epoch

    def as_ECL(
        self, epoch: Union[float, u.Quantity, Time] = None, ecl: str = "IERS2010"
    ) -> "AstrometryEcliptic":
        """Return pint.models.astrometry.Astrometry object in PulsarEcliptic frame.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.
        ecl : str, optional
            Obliquity for PulsarEcliptic frame

        Returns
        -------
        pint.models.astrometry.AstrometryEcliptic
        """

        # change epoch only
        if ecl == self.ECL.value:
            m = copy.deepcopy(self)
            if epoch is not None:
                m.change_posepoch(epoch)
            return m

        m_ecl = AstrometryEcliptic()

        # transfer over parallax and POSEPOCH: don't need to change
        m_ecl.PX = self.PX
        m_ecl.POSEPOCH = self.POSEPOCH
        # get ELONG, ELAT, PM
        c = self.coords_as_ECL(epoch=epoch, ecl=ecl)
        m_ecl.ELONG.quantity = c.lon
        m_ecl.ELAT.quantity = c.lat
        m_ecl.PMELONG.quantity = c.pm_lon_coslat
        m_ecl.PMELAT.quantity = c.pm_lat
        m_ecl.ECL.value = ecl

        # use fake proper motions to convert uncertainties on ELONG, ELAT
        # assume that ELONG uncertainty does not include cos(ELAT)
        # and that the RA uncertainty does not include cos(DEC)
        # put it in here as pm_ra_cosdec since astropy complains otherwise
        dt = 1 * u.yr
        c = coords.SkyCoord(
            lon=self.ELONG.quantity,
            lat=self.ELAT.quantity,
            obliquity=OBL[self.ECL.value],
            obstime=self.POSEPOCH.quantity,
            pm_lon_coslat=(
                self.ELONG.uncertainty * np.cos(self.ELAT.quantity) / dt
                if self.ELONG.uncertainty is not None
                else 0 * self.ELONG.units / dt
            ),
            pm_lat=(
                self.ELAT.uncertainty / dt
                if self.ELAT.uncertainty is not None
                else 0 * self.ELAT.units / dt
            ),
            frame=PulsarEcliptic,
        )
        c_ECL = c.transform_to(PulsarEcliptic(ecl=ecl))
        m_ecl.ELONG.uncertainty = c_ECL.pm_lon_coslat * dt / np.cos(c_ECL.lat)
        m_ecl.ELAT.uncertainty = c_ECL.pm_lat * dt
        # use fake proper motions to convert uncertainties on proper motion
        # assume that PMELONG uncertainty includes cos(DEC)
        c = coords.SkyCoord(
            lon=self.ELONG.quantity,
            lat=self.ELAT.quantity,
            obliquity=OBL[self.ECL.value],
            obstime=self.POSEPOCH.quantity,
            pm_lon_coslat=(
                self.PMELONG.uncertainty
                if self.PMELONG.uncertainty is not None
                else 0 * self.PMELONG.units
            ),
            pm_lat=(
                self.PMELAT.uncertainty
                if self.PMELAT.uncertainty is not None
                else 0 * self.PMELAT.units
            ),
            frame=PulsarEcliptic,
        )
        c_ECL = c.transform_to(PulsarEcliptic(ecl=ecl))
        m_ecl.PMELONG.uncertainty = c_ECL.pm_lon_coslat
        m_ecl.PMELAT.uncertainty = c_ECL.pm_lat
        # freeze comparable parameters
        m_ecl.ELONG.frozen = self.ELONG.frozen
        m_ecl.ELAT.frozen = self.ELAT.frozen
        m_ecl.PMELONG.frozen = self.PMELONG.frozen
        m_ecl.PMELAT.frozen = self.PMELAT.frozen

        return m_ecl

    def as_ICRS(
        self, epoch: Union[float, u.Quantity, Time] = None
    ) -> "AstrometryEquatorial":
        """Return pint.models.astrometry.Astrometry object in ICRS frame.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.

        Returns
        -------
        pint.models.astrometry.AstrometryEquatorial
        """
        m_eq = AstrometryEquatorial()
        # transfer over parallax and POSEPOCH: don't need to change
        m_eq.PX = self.PX
        m_eq.POSEPOCH = self.POSEPOCH
        # get RA, DEC, PM
        c = self.coords_as_ICRS(epoch=epoch)
        m_eq.RAJ.quantity = c.ra
        m_eq.DECJ.quantity = c.dec
        m_eq.PMRA.quantity = c.pm_ra_cosdec
        m_eq.PMDEC.quantity = c.pm_dec

        # use fake proper motions to convert uncertainties on RA,Dec
        # assume that RA uncertainty does not include cos(Dec)
        # and neither does the ELONG uncertainty
        # put it in as pm_lon_coslat since astropy complains otherwise
        dt = 1 * u.yr
        c = coords.SkyCoord(
            lon=self.ELONG.quantity,
            lat=self.ELAT.quantity,
            obliquity=OBL[self.ECL.value],
            obstime=self.POSEPOCH.quantity,
            pm_lon_coslat=(
                self.ELONG.uncertainty * np.cos(self.ELAT.quantity) / dt
                if self.ELONG.uncertainty is not None
                else 0 * self.ELONG.units / dt
            ),
            pm_lat=(
                self.ELAT.uncertainty / dt
                if self.ELAT.uncertainty is not None
                else 0 * self.ELAT.units / dt
            ),
            frame=PulsarEcliptic,
        )
        c_ICRS = c.transform_to(coords.ICRS)

        m_eq.RAJ.uncertainty = np.abs(c_ICRS.pm_ra_cosdec * dt / np.cos(c_ICRS.dec))
        m_eq.DECJ.uncertainty = np.abs(c_ICRS.pm_dec * dt)
        # use fake proper motions to convert uncertainties on proper motion
        # assume that PMELONG uncertainty includes cos(DEC)
        c = coords.SkyCoord(
            lon=self.ELONG.quantity,
            lat=self.ELAT.quantity,
            obliquity=OBL[self.ECL.value],
            obstime=self.POSEPOCH.quantity,
            pm_lon_coslat=(
                self.PMELONG.uncertainty
                if self.PMELONG.uncertainty is not None
                else 0 * self.PMELONG.units
            ),
            pm_lat=(
                self.PMELAT.uncertainty
                if self.PMELAT.uncertainty is not None
                else 0 * self.PMELAT.units
            ),
            frame=PulsarEcliptic,
        )
        c_ICRS = c.transform_to(coords.ICRS)
        m_eq.PMRA.uncertainty = np.abs(c_ICRS.pm_ra_cosdec)
        m_eq.PMDEC.uncertainty = np.abs(c_ICRS.pm_dec)
        # freeze comparable parameters
        m_eq.RAJ.frozen = self.ELONG.frozen
        m_eq.DECJ.frozen = self.ELAT.frozen
        m_eq.PMRA.frozen = self.PMELONG.frozen
        m_eq.PMDEC.frozen = self.PMELAT.frozen

        return m_eq
