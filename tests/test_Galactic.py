import logging
import os
import pytest

import astropy.units as u

import pint.models.model_builder as mb
from pinttestdata import datadir
from pint import utils

import astropy.coordinates
import astropy.time


class TestGalactic:
    """Test conversion from equatorial/ecliptic -> Galactic coordinates as astropy objects"""

    @classmethod
    def setup_class(cls):
        # J0613 is in equatorial
        cls.parfileJ0613 = os.path.join(
            datadir, "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
        )
        cls.modelJ0613 = mb.get_model(cls.parfileJ0613)

        # B1855+09 is in ecliptic
        cls.parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par")
        cls.modelB1855 = mb.get_model(cls.parfileB1855)

        cls.log = logging.getLogger("TestGalactic")

    def test_proper_motion(self):
        """
        use the PINT and astropy proper motion calculations and compare
        """

        # make a test SkyCoord object
        # make sure it has obstime and distance supplied
        # to use it for conversions as well
        J0613_icrs = self.modelJ0613.coords_as_ICRS()
        J0613_icrs_now = utils.add_dummy_distance(J0613_icrs)
        newepoch = self.modelJ0613.POSEPOCH.quantity.mjd + 100
        # now do it for a future epoch
        J0613_icrs = self.modelJ0613.coords_as_ICRS(epoch=newepoch)
        # and use the coordinates now but use astropy's space motion
        print(
            J0613_icrs_now.apply_space_motion(
                new_obstime=astropy.time.Time(newepoch, scale="tdb", format="mjd")
            )
        )
        J0613_icrs_now_to_then = utils.remove_dummy_distance(
            J0613_icrs_now.apply_space_motion(
                new_obstime=astropy.time.Time(newepoch, scale="tdb", format="mjd")
            )
        )
        sep = J0613_icrs.separation(J0613_icrs_now_to_then)
        msg = (
            "Applying proper motion for +100d failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg

        # make sure it can support newepoch supplied as a Time object
        newepoch = astropy.time.Time(newepoch, format="mjd")
        J0613_icrs = self.modelJ0613.coords_as_ICRS(epoch=newepoch)
        J0613_icrs_now_to_then = utils.remove_dummy_distance(
            J0613_icrs_now.apply_space_motion(new_obstime=newepoch)
        )
        sep = J0613_icrs.separation(J0613_icrs_now_to_then)
        msg = (
            "Applying proper motion for +100d failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg

    def test_proper_motion_identity(self):
        # sanity check that evaluation at POSEPOCH returns something very close to 0
        J0613_icrs = self.modelJ0613.coords_as_ICRS()
        J0613_icrs_alt = self.modelJ0613.coords_as_ICRS(
            epoch=self.modelJ0613.POSEPOCH.quantity.mjd
        )
        sep = J0613_icrs_alt.separation(J0613_icrs)
        assert sep < 1e-11 * u.arcsec

    def test_equatorial_to_galactic(self):
        """
        start with a pulsar in equatorial coordinates
        convert to Galactic and make sure the positions are consistent

        then apply the space motion to the equatorial object & convert to Galactic
        compare that to the Galactic object w/ space motion

        """

        # make a test SkyCoord object
        # make sure it has obstime and distance supplied
        # to use it for conversions as well
        J0613_icrs = self.modelJ0613.coords_as_ICRS()
        J0613_icrs_now = utils.add_dummy_distance(J0613_icrs)

        J0613_galactic = self.modelJ0613.coords_as_GAL()
        J0613_galactic_now = utils.add_dummy_distance(J0613_galactic)

        newepoch = self.modelJ0613.POSEPOCH.quantity.mjd + 100

        # what I get converting within astropy
        J0613_galactic_comparison = J0613_icrs_now.transform_to(
            astropy.coordinates.Galactic
        )
        sep = J0613_galactic_now.separation(J0613_galactic_comparison)
        msg = (
            "Equatorial to Galactic conversion for now failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg

        J0613_icrs = self.modelJ0613.coords_as_ICRS(epoch=newepoch)
        # what I get converting within astropy
        J0613_galactic_comparison = J0613_icrs.transform_to(
            astropy.coordinates.Galactic
        )
        J0613_galactic_then = utils.remove_dummy_distance(
            J0613_galactic_now.apply_space_motion(
                new_obstime=astropy.time.Time(newepoch, scale="tdb", format="mjd")
            )
        )
        sep = J0613_galactic_then.separation(J0613_galactic_comparison)
        msg = (
            "Equatorial to Galactic conversion for +100d failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg

    def test_ecliptic_to_galactic(self):
        """
        start with a pulsar in ecliptic coordinates
        convert to Galactic and make sure the positions are consistent

        then apply the space motion to the ecliptic object & convert to Galactic
        compare that to the Galactic object w/ space motion

        """

        # make a test SkyCoord object
        # make sure it has obstime and distance supplied
        # to use it for conversions as well
        B1855_ECL = self.modelB1855.coords_as_ECL()
        B1855_ECL_now = utils.add_dummy_distance(B1855_ECL)

        B1855_galactic = self.modelB1855.coords_as_GAL()
        B1855_galactic_now = utils.add_dummy_distance(B1855_galactic)

        newepoch = self.modelB1855.POSEPOCH.quantity.mjd + 100

        # what I get converting within astropy
        B1855_galactic_comparison = B1855_ECL_now.transform_to(
            astropy.coordinates.Galactic
        )
        sep = B1855_galactic_now.separation(B1855_galactic_comparison)
        msg = (
            "Ecliptic to Galactic conversion for now failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg

        B1855_ECL = self.modelB1855.coords_as_ECL(epoch=newepoch)
        # what I get converting within astropy
        B1855_galactic_comparison = B1855_ECL.transform_to(astropy.coordinates.Galactic)
        B1855_galactic_then = utils.remove_dummy_distance(
            B1855_galactic_now.apply_space_motion(
                new_obstime=astropy.time.Time(newepoch, scale="tdb", format="mjd")
            )
        )
        sep = B1855_galactic_then.separation(B1855_galactic_comparison)
        msg = (
            "Ecliptic to Galactic conversion for +100d failed with separation %.1e arcsec"
            % sep.arcsec
        )
        assert sep < 1e-9 * u.arcsec, msg
