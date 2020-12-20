# astrometry.py
# Defines Astrometry timing model class
from __future__ import absolute_import, division, print_function

import sys

import astropy.constants as const
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy import log
from astropy.time import Time

from pint import ls
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    floatParameter,
    strParameter,
)
from pint.models.timing_model import DelayComponent, MissingParameter
from pint.pulsar_ecliptic import OBL, PulsarEcliptic
from pint.utils import add_dummy_distance, remove_dummy_distance

astropy_version = sys.modules["astropy"].__version__
mas_yr = u.mas / u.yr


class Astrometry(DelayComponent):
    register = False
    category = "astrometry"

    def __init__(self):
        super(Astrometry, self).__init__()
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

    def ssb_to_psb_xyz_ICRS(self, epoch=None):
        """Returns unit vector(s) from SSB to pulsar system barycenter under ICRS.

        If epochs (MJD) are given, proper motion is included in the calculation.
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

    def ssb_to_psb_xyz_ECL(self, epoch=None):
        """Returns unit vector(s) from SSB to pulsar system barycenter under Ecliptic coordinates.

        If epochs (MJD) are given, proper motion is included in the calculation.
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.coords_as_ECL(epoch=epoch).cartesian.xyz.transpose()

    def sun_angle(self, toas, heliocenter=True, also_distance=False):
        """Compute the pulsar-observatory-Sun angle.

        This is the angle between the center of the Sun and the direction to
        the pulsar, as seen from the observatory (for each TOA).

        This angle takes into account the motion of the Sun around the solar system barycenter.

        Parameters
        ----------
        toas: :class:`pint.toas.TOAs`
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
        r = (osv ** 2).sum(axis=1) ** 0.5
        osv /= r[:, None]
        cos = (osv * psr_vec).sum(axis=1)
        if also_distance:
            return np.arccos(cos), r
        else:
            return np.arccos(cos)

    def barycentric_radio_freq(self, toas):
        raise NotImplementedError

    def solar_system_geometric_delay(self, toas, acc_delay=None):
        """Returns geometric delay (in sec) due to position of site in
        solar system.  This includes Roemer delay and parallax.

        NOTE: currently assumes XYZ location of TOA relative to SSB is
        available as 3-vector toa.xyz, in units of light-seconds.
        """
        tbl = toas.table
        L_hat = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"].astype(np.float64))
        re_dot_L = np.sum(tbl["ssb_obs_pos"] * L_hat, axis=1)
        delay = -re_dot_L.to(ls).value
        if self.PX.value != 0.0 and np.count_nonzero(tbl["ssb_obs_pos"]) > 0:
            L = (1.0 / self.PX.value) * u.kpc
            # TODO: np.sum currently loses units in some cases...
            re_sqr = (
                np.sum(tbl["ssb_obs_pos"] ** 2, axis=1) * tbl["ssb_obs_pos"].unit ** 2
            )
            delay += (0.5 * (re_sqr / L) * (1.0 - re_dot_L ** 2 / re_sqr)).to(ls).value
        return delay * u.second

    def get_d_delay_quantities(self, toas):
        """Calculate values needed for many d_delay_d_param functions """
        # TODO: Move all these calculations in a separate class for elegance
        rd = dict()

        # TODO: Should delay not have units of u.second?
        delay = self._parent.delay(toas)

        # TODO: tbl['tdbld'].quantity should have units of u.day
        # NOTE: Do we need to include the delay here?
        tbl = toas.table
        rd["epoch"] = tbl["tdbld"].quantity * u.day  # - delay * u.second

        # Distance from SSB to observatory, and from SSB to psr
        ssb_obs = tbl["ssb_obs_pos"].quantity
        ssb_psr = self.ssb_to_psb_xyz_ICRS(epoch=np.array(rd["epoch"]))

        # Cartesian coordinates, and derived quantities
        rd["ssb_obs_r"] = np.sqrt(np.sum(ssb_obs ** 2, axis=1))
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

    def d_delay_astrometry_d_PX(self, toas, param="", acc_delay=None):
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
        dd_dpx = 0.5 * (px_r ** 2 / (u.AU * const.c)) * (u.mas / u.radian)

        # We want to return sec / mas
        return dd_dpx.decompose(u.si.bases) / u.mas

    def d_delay_astrometry_d_POSEPOCH(self, toas, param="", acc_delay=None):
        """Calculate the derivative wrt POSEPOCH
        """
        pass

    def change_posepoch(self, new_epoch):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: float or `astropy.Time` object
            The new POSEPOCH value.
        """
        raise NotImplementedError


class AstrometryEquatorial(Astrometry):
    register = True

    def __init__(self):
        super(AstrometryEquatorial, self).__init__()
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
                value=0.0,
                description="Proper motion in RA",
            )
        )

        self.add_param(
            floatParameter(
                name="PMDEC",
                units="mas/year",
                value=0.0,
                description="Proper motion in DEC",
            )
        )
        self.set_special_params(["RAJ", "DECJ", "PMRA", "PMDEC"])
        for param in ["RAJ", "DECJ", "PMRA", "PMDEC"]:
            deriv_func_name = "d_delay_astrometry_d_" + param
            func = getattr(self, deriv_func_name)
            self.register_deriv_funcs(func, param)

    def setup(self):
        super(AstrometryEquatorial, self).setup()

    def validate(self):
        """ Validate the input parameter.
        """
        super(AstrometryEquatorial, self).validate()
        # RA/DEC are required
        for p in ("RAJ", "DECJ"):
            if getattr(self, p).value is None:
                raise MissingParameter("Astrometry", p)
        # Check for POSEPOCH
        if self.POSEPOCH.quantity is None:
            if self._parent.PEPOCH.quantity is None:
                raise MissingParameter(
                    "AstrometryEquatorial",
                    "POSEPOCH",
                    "POSEPOCH or PEPOCH are required if PM is set.",
                )
            else:
                log.warning("POSEPOCH not found; using PEPOCH unless set explicitly!")
                self.POSEPOCH.quantity = self._parent.PEPOCH.quantity

    def print_par(self):
        result = ""
        print_order = ["RAJ", "DECJ", "PMRA", "PMDEC", "PX", "POSEPOCH"]
        for p in print_order:
            par = getattr(self, p)
            if par.quantity is not None:
                result += getattr(self, p).as_parfile_line()
        return result

    def barycentric_radio_freq(self, toas):
        """Return radio frequencies (MHz) of the toas corrected for Earth motion"""
        tbl = toas.table
        L_hat = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"].astype(np.float64))
        v_dot_L_array = np.sum(tbl["ssb_obs_vel"] * L_hat, axis=1)
        return tbl["freq"] * (1.0 - v_dot_L_array / const.c)

    def get_psr_coords(self, epoch=None):
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
        else:
            if isinstance(epoch, Time):
                newepoch = epoch
            else:
                newepoch = Time(epoch, scale="tdb", format="mjd")
            position_now = add_dummy_distance(self.get_psr_coords())
            position_then = remove_dummy_distance(
                position_now.apply_space_motion(new_obstime=newepoch)
            )
            return position_then

    def coords_as_ICRS(self, epoch=None):
        """Return the pulsar's ICRS coordinates as an astropy coordinate object."""
        return self.get_psr_coords(epoch)

    def coords_as_ECL(self, epoch=None, ecl=None):
        """Return the pulsar's ecliptic coordinates as an astropy coordinate object.

        The value used for the obliquity of the ecliptic can be controlled with the
        `ecl` keyword, which should be one of the codes listed in `ecliptic.dat`.
        If `ecl` is left unspecified, the global default IERS2010 will be used.
        """
        if ecl is None:
            log.info("ECL not specified; using IERS2010.")
            ecl = "IERS2010"

        pos_icrs = self.get_psr_coords(epoch=epoch)
        return pos_icrs.transform_to(PulsarEcliptic(ecl=ecl))

    def coords_as_GAL(self, epoch=None):
        """Return the pulsar's galactic coordinates as an astropy coordinate object.

        """
        pos_icrs = self.get_psr_coords(epoch=epoch)
        return pos_icrs.transform_to(coords.Galactic)

    def get_params_as_ICRS(self):
        result = {
            "RAJ": self.RAJ.quantity,
            "DECJ": self.DECJ.quantity,
            "PMRA": self.PMRA.quantity,
            "PMDEC": self.PMDEC.quantity,
        }
        return result

    def d_delay_astrometry_d_RAJ(self, toas, param="", acc_delay=None):
        """Calculate the derivative wrt RAJ

        For the RAJ and DEC derivatives, use the following approximate model for
        the pulse delay. (Inner-product between two Cartesian vectors)

        de = Earth declination (wrt SSB)
        ae = Earth right ascension
        dp = pulsar declination
        aa = pulsar right ascension
        r = distance from SSB to Earh
        c = speed of light

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

    def d_delay_astrometry_d_DECJ(self, toas, param="", acc_delay=None):
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

    def d_delay_astrometry_d_PMRA(self, toas, param="", acc_delay=None):
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

    def d_delay_astrometry_d_PMDEC(self, toas, param="", acc_delay=None):
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

    def change_posepoch(self, new_epoch):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: float or `astropy.Time` object
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


class AstrometryEcliptic(Astrometry):
    register = True

    def __init__(self):
        super(AstrometryEcliptic, self).__init__()
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
                value=0.0,
                description="Proper motion in ecliptic longitude",
                aliases=["PMLAMBDA"],
            )
        )

        self.add_param(
            floatParameter(
                name="PMELAT",
                units="mas/year",
                value=0.0,
                description="Proper motion in ecliptic latitude",
                aliases=["PMBETA"],
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
            deriv_func_name = "d_delay_astrometry_d_" + param
            func = getattr(self, deriv_func_name)
            self.register_deriv_funcs(func, param)

    def setup(self):
        super(AstrometryEcliptic, self).setup()

    def validate(self):
        """ Validate Ecliptic coordinate parameter inputs.
        """
        super(AstrometryEcliptic, self).validate()
        # ELONG/ELAT are required
        for p in ("ELONG", "ELAT"):
            if getattr(self, p).value is None:
                raise MissingParameter("AstrometryEcliptic", p)
        # Check for POSEPOCH
        if self.POSEPOCH.quantity is None:
            if self._parent.PEPOCH.quantity is None:
                raise MissingParameter(
                    "Astrometry",
                    "POSEPOCH",
                    "POSEPOCH or PEPOCH are required if PM is set.",
                )
            else:
                log.warning("POSEPOCH not found; using PEPOCH unless set explicitly!")
                self.POSEPOCH.quantity = self._parent.PEPOCH.quantity

    def barycentric_radio_freq(self, toas):
        """Return radio frequencies (MHz) of the toas corrected for Earth motion"""
        if "ssb_obs_vel_ecl" not in toas.table.colnames:
            obliquity = OBL[self.ECL.value]
            toas.add_vel_ecl(obliquity)
        tbl = toas.table
        L_hat = self.ssb_to_psb_xyz_ECL(epoch=tbl["tdbld"].astype(np.float64))
        v_dot_L_array = np.sum(tbl["ssb_obs_vel_ecl"] * L_hat, axis=1)
        return tbl["freq"] * (1.0 - v_dot_L_array / const.c)

    def get_psr_coords(self, epoch=None):
        """Returns pulsar sky coordinates as an astropy ecliptic coordinate instance.

        Parameters
        ----------
        epoch: `astropy.time.Time` or Float, optional
            new epoch for position.  If Float, MJD(TDB) is assumed

        Returns
        -------
        position
            PulsarEcliptic SkyCoord object optionally with proper motion applied

        If epoch (MJD) is specified, proper motion is included to return
        the position at the given epoch.

        """
        try:
            obliquity = OBL[self.ECL.value]
        except KeyError:
            raise ValueError(
                "No obliquity " + str(self.ECL.value) + " provided. "
                "Check your pint/datafile/ecliptic.dat file."
            )
        if epoch is None or (self.PMELONG.value == 0.0 and self.PMELAT.value == 0.0):
            return coords.SkyCoord(
                obliquity=obliquity,
                lon=self.ELONG.quantity,
                lat=self.ELAT.quantity,
                pm_lon_coslat=self.PMELONG.quantity,
                pm_lat=self.PMELAT.quantity,
                obstime=self.POSEPOCH.quantity,
                frame=PulsarEcliptic,
            )
        else:
            if isinstance(epoch, Time):
                newepoch = epoch
            else:
                newepoch = Time(epoch, scale="tdb", format="mjd")
            position_now = add_dummy_distance(self.get_psr_coords())
            position_then = remove_dummy_distance(
                position_now.apply_space_motion(new_obstime=newepoch)
            )
            return position_then

    def coords_as_ICRS(self, epoch=None):
        """Return the pulsar's ICRS coordinates as an astropy coordinate object."""
        pos_ecl = self.get_psr_coords(epoch=epoch)
        return pos_ecl.transform_to(coords.ICRS)

    def coords_as_GAL(self, epoch=None):
        """Return the pulsar's galactic coordinates as an astropy coordinate object.

        """
        pos_ecl = self.get_psr_coords(epoch=epoch)
        return pos_ecl.transform_to(coords.Galactic)

    def coords_as_ECL(self, epoch=None, ecl=None):
        """Return the pulsar's ecliptic coordinates as an astropy coordinate object.

        The value used for the obliquity of the ecliptic can be controlled with the
        `ecl` keyword, which should be one of the codes listed in `ecliptic.dat`.
        If `ecl` is left unspecified, the model's ECL parameter will be used.
        """
        pos_ecl = self.get_psr_coords(epoch)
        if ecl is not None:
            pos_ecl = pos_ecl.transform_to(PulsarEcliptic(ecl=ecl))
        return pos_ecl

    def get_d_delay_quantities_ecliptical(self, toas):
        """Calculate values needed for many d_delay_d_param functions """
        # TODO: Move all these calculations in a separate class for elegance
        rd = dict()
        # From the earth_ra dec to earth_elong and elat
        try:
            obliquity = OBL[self.ECL.value]
        except KeyError:
            raise ValueError(
                "No obliquity " + self.ECL.value + " provided. "
                "Check your pint/datafile/ecliptic.dat file."
            )

        rd = self.get_d_delay_quantities(toas)
        coords_icrs = coords.ICRS(ra=rd["earth_ra"], dec=rd["earth_dec"])
        coords_elpt = coords_icrs.transform_to(PulsarEcliptic(obliquity=obliquity))
        rd["earth_elong"] = coords_elpt.lon
        rd["earth_elat"] = coords_elpt.lat

        return rd

    def get_params_as_ICRS(self):
        result = dict()
        pv_ECL = self.get_psr_coords()
        pv_ICRS = pv_ECL.transform_to(coords.ICRS)
        result["RAJ"] = pv_ICRS.ra.to(u.hourangle)
        result["DECJ"] = pv_ICRS.dec
        result["PMRA"] = pv_ICRS.pm_ra_cosdec
        result["PMDEC"] = pv_ICRS.pm_dec
        return result

    def d_delay_astrometry_d_ELONG(self, toas, param="", acc_delay=None):
        """Calculate the derivative wrt RAJ

        For the RAJ and DEC derivatives, use the following approximate model for
        the pulse delay. (Inner-product between two Cartesian vectors)

        de = Earth declination (wrt SSB)
        ae = Earth right ascension
        dp = pulsar declination
        aa = pulsar right ascension
        r = distance from SSB to Earh
        c = speed of light

        delay = r*[cos(de)*cos(dp)*cos(ae-aa)+sin(de)*sin(dp)]/c

        elate = Earth elat (wrt SSB)
        elonge = Earth elong
        elatp = pulsar elat
        elongp = pulsar elong
        r = distance from SSB to Earh
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

    def d_delay_astrometry_d_ELAT(self, toas, param="", acc_delay=None):
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

    def d_delay_astrometry_d_PMELONG(self, toas, param="", acc_delay=None):
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

    def d_delay_astrometry_d_PMELAT(self, toas, param="", acc_delay=None):
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

    def print_par(self):
        result = ""
        print_order = ["ELONG", "ELAT", "PMELONG", "PMELAT", "PX", "ECL", "POSEPOCH"]
        for p in print_order:
            par = getattr(self, p)
            if par.quantity is not None:
                result += getattr(self, p).as_parfile_line()
        return result

    def change_posepoch(self, new_epoch):
        """Change POSEPOCH to a new value and update the position accordingly.

        Parameters
        ----------
        new_epoch: float or `astropy.Time` object
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
