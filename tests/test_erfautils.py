from __future__ import absolute_import, print_function, division
import unittest

from nose.tools import raises
import numpy as np
from astropy import log
import astropy.table as table
from astropy.time import Time
import astropy.units as u
from astropy.utils.iers import IERS_Auto, IERS_B

from pint.observatory import Observatory
from pint import toa, utils, erfautils
from pinttestdata import testdir, datadir


def test_simpler_erfa_import():
    import astropy._erfa as erfa

@unittest.skip
def test_compare_erfautils_astropy():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    mjds = np.linspace(50000,58000,512)
    t = Time(mjds, scale="tdb", format="mjd")
    posvel = erfautils.gcrs_posvel_from_itrf(loc, t, obsname=o)
    astropy_posvel = erfautils.astropy_gcrs_posvel_from_itrf(
        loc, t, obsname=o)
    dopv = astropy_posvel - posvel
    dpos = np.sqrt((dopv.pos**2).sum(axis=0))
    dvel = np.sqrt((dopv.vel**2).sum(axis=0))
    assert len(dpos)==len(mjds)
    # This is just above the level of observed difference
    assert dpos.max()<0.05*u.m, "position difference of %s" % dpos.max().to(u.m)
    # This level is what is permitted as a velocity difference from tempo2 in test_times.py
    assert dvel.max()<0.02*u.mm/u.s, "velocity difference of %s" % dvel.max().to(u.mm/u.s)

def test_iers_discrepancies():
    iers_auto = IERS_Auto.open()
    iers_b = IERS_B.open()
    for mjd in [56000,56500,57000]:
        t = Time(mjd, scale="tdb", format="mjd")
        b_x, b_y = iers_b.pm_xy(t)
        a_x, a_y = iers_auto.pm_xy(t)
        assert abs(a_x-b_x)<1*u.marcsec
        assert abs(a_y-b_y)<1*u.marcsec

def test_scalar():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    t = Time(56000, scale="tdb", format="mjd")
    posvel = erfautils.gcrs_posvel_from_itrf(loc, t, obsname=o)
    assert posvel.pos.shape == (3,)

@raises(ValueError)
def test_matrix():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    t = Time(56000*np.ones((4,5)), scale="tdb", format="mjd")
    posvel = erfautils.gcrs_posvel_from_itrf(loc, t, obsname=o)
