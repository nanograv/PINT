import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from astropy.utils.data import download_file
from astropy.utils.iers import IERS_B, IERS_B_FILE, IERS_B_URL, IERS_Auto
from numpy.testing import assert_allclose, assert_equal

from pint import erfautils
from pint.observatory import Observatory


def test_simpler_erfa_import():
    import erfa

    erfa


@pytest.mark.xfail(
    reason="astropy doesn't include up-to-date IERS B - "
    "if this starts passing we can ditch the "
    "implementation in erfautils"
)
def test_compare_erfautils_astropy():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    mjds = np.linspace(50000, 58000, 512)
    t = Time(mjds, scale="tdb", format="mjd")
    posvel = erfautils.old_gcrs_posvel_from_itrf(loc, t, obsname=o)
    astropy_posvel = erfautils.astropy_gcrs_posvel_from_itrf(loc, t, obsname=o)
    dopv = astropy_posvel - posvel
    dpos = np.sqrt((dopv.pos**2).sum(axis=0))
    dvel = np.sqrt((dopv.vel**2).sum(axis=0))
    assert len(dpos) == len(mjds)
    # This is just above the level of observed difference
    assert dpos.max() < 0.05 * u.m, f"position difference of {dpos.max().to(u.m)}"
    # This level is what is permitted as a velocity difference from tempo2 in test_times.py
    assert dvel.max() < 0.02 * u.mm / u.s, "velocity difference of %s" % dvel.max().to(
        u.mm / u.s
    )


def test_iers_discrepancies():
    iers_auto = IERS_Auto.open()
    iers_b = IERS_B.open()
    for mjd in [56000, 56500, 57000]:
        t = Time(mjd, scale="tdb", format="mjd")
        b_x, b_y = iers_b.pm_xy(t)
        a_x, a_y = iers_auto.pm_xy(t)
        assert abs(a_x - b_x) < 1 * u.marcsec
        assert abs(a_y - b_y) < 1 * u.marcsec


def test_scalar():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    t = Time(56000, scale="tdb", format="mjd")
    posvel = erfautils.gcrs_posvel_from_itrf(loc, t, obsname=o)
    assert posvel.pos.shape == (3,)


@pytest.mark.skip(
    "I don't know why this test exists but it no longer fails after changing to astropy version of gcrs_posvel_from_itrf() -- paulr"
)
def test_matrix():
    """Confirm higher-dimensional arrays raise an exception"""
    with pytest.raises(ValueError):
        o = "Arecibo"
        loc = Observatory.get(o).earth_location_itrf()
        t = Time(56000 * np.ones((4, 5)), scale="tdb", format="mjd")
        erfautils.gcrs_posvel_from_itrf(loc, t, obsname=o)


# The below explore why astropy might disagree with PINT internal code


@pytest.mark.xfail(reason="astropy doesn't include up-to-date IERS B")
def test_IERS_B_all_in_IERS_Auto():
    B = IERS_B.open(download_file(IERS_B_URL, cache=True))
    mjd = B["MJD"].to(u.day).value
    A = IERS_Auto.open()
    A.pm_xy(mjd)  # ensure that data is available for this date
    i_A = np.searchsorted(A["MJD"].to(u.day).value, mjd)
    assert_equal(A["dX_2000A_B"][i_A], B["dX_2000A"])


@pytest.mark.xfail(reason="IERS changes old values in new versions of the B table")
def test_IERS_B_agree_with_IERS_Auto_dX():
    A = IERS_Auto.open()
    B = IERS_B.open(download_file(IERS_B_URL, cache=True))
    mjd = B["MJD"].to(u.day).value
    A.pm_xy(mjd)  # ensure that data is available for this date

    # Let's get rid of some trouble values and see if they agree on a restricted subset
    # IERS Auto ends with a bunch of 1e20 values meant as sentinels (?)
    ok_A = abs(A["dX_2000A_B"]) < 1e6 * u.marcsec
    # For some reason IERS B starts with zeros up to MJD 45700? IERS Auto doesn't match this
    # ok_A &= A['MJD'] > 45700*u.d
    # Maybe the old values are bogus?
    ok_A &= A["MJD"] > 50000 * u.d

    mjds_A = A["MJD"][ok_A].to(u.day).value
    i_B = np.searchsorted(B["MJD"].to(u.day).value, mjds_A)
    assert np.all(np.diff(i_B) == 1), "Valid region not contiguous"
    assert_equal(A["MJD"][ok_A], B["MJD"][i_B], "MJDs don't make sense")
    for tag in ["dX_2000A", "dY_2000A"]:
        assert_allclose(
            A[f"{tag}_B"][ok_A].to(u.marcsec).value,
            B[tag][i_B].to(u.marcsec).value,
            atol=1e-5,
            rtol=1e-3,
            err_msg=f"IERS A-derived IERS B {tag} values don't match current IERS B values",
        )


@pytest.mark.xfail(
    reason="IERS changes old values in new versions of the B table in astropy 4"
)
def test_IERS_B_agree_with_IERS_Auto():
    A = IERS_Auto.open()
    B = IERS_B.open(download_file(IERS_B_URL, cache=True))
    mjd = B["MJD"].to(u.day).value
    A.pm_xy(mjd)  # ensure that data is available for this date

    # Let's get rid of some trouble values and see if they agree on a restricted subset
    # IERS Auto ends with a bunch of 1e20 values meant as sentinels (?)
    ok_A = abs(A["PM_X_B"]) < 1e6 * u.marcsec
    # Maybe the old values are bogus?
    ok_A &= A["MJD"] > 50000 * u.d

    mjds_A = A["MJD"][ok_A].to(u.day).value
    i_B = np.searchsorted(B["MJD"].to(u.day).value, mjds_A)
    assert np.all(np.diff(i_B) == 1), "Valid region not contiguous"
    assert_equal(A["MJD"][ok_A], B["MJD"][i_B], "MJDs don't make sense")
    for atag, btag, unit in [
        ("UT1_UTC_B", "UT1_UTC", u.s),  # s, six decimal places
        ("PM_X_B", "PM_x", u.arcsec),
        ("PM_Y_B", "PM_y", u.arcsec),
    ]:
        assert_allclose(
            A[atag][ok_A].to(unit).value,
            B[btag][i_B].to(unit).value,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"Inserted IERS B {atag} values don't match IERS_B_URL {btag} values",
        )


# @pytest.mark.xfail(reason="disagreement in current astropy")
def test_IERS_B_builtin_agree_with_IERS_Auto_dX():
    A = IERS_Auto.open()
    B = IERS_B.open(IERS_B_FILE)
    mjd = B["MJD"].to(u.day).value
    A.pm_xy(mjd)  # ensure that data is available for these dates

    # We're going to look up the OK auto values in the B table
    ok_A = A["MJD"] < B["MJD"][-1]
    # Let's get rid of some trouble values and see if they agree on a restricted subset
    # IERS Auto ends with a bunch of 1e20 values meant as sentinels (?)
    ok_A &= abs(A["dX_2000A_B"]) < 1e6 * u.marcsec
    # For some reason IERS B starts with zeros up to MJD 45700? IERS Auto doesn't match this
    ok_A &= A["MJD"] > 45700 * u.d
    # Maybe the old values are bogus?
    ok_A &= A["MJD"] > 50000 * u.d

    mjds_A = A["MJD"][ok_A].to(u.day).value
    i_B = np.searchsorted(B["MJD"].to(u.day).value, mjds_A)

    assert np.all(np.diff(i_B) == 1), "Valid region not contiguous"
    assert_equal(A["MJD"][ok_A], B["MJD"][i_B], "MJDs don't make sense")
    assert_allclose(
        A["dX_2000A_B"][ok_A].to(u.marcsec).value,
        B["dX_2000A"][i_B].to(u.marcsec).value,
        atol=1e-5,
        rtol=1e-3,
        err_msg="IERS B values included in IERS A (dX_2000A) don't match IERS_B_FILE values",
    )


@pytest.mark.skip
def test_IERS_B_builtin_agree_with_IERS_Auto():
    """The UT1-UTC, PM_X, and PM_Y values are correctly copied"""
    A = IERS_Auto.open()
    B = IERS_B.open(IERS_B_FILE)
    mjd = B["MJD"].to(u.day).value
    A.pm_xy(mjd)  # ensure that data is available for these dates

    # We're going to look up the OK auto values in the B table
    ok_A = A["MJD"] < B["MJD"][-1]
    # Let's get rid of some trouble values and see if they agree on a restricted subset
    # IERS Auto ends with a bunch of 1e20 values meant as sentinels (?)
    ok_A &= abs(A["PM_X_B"]) < 1e6 * u.marcsec
    # For some reason IERS B starts with zeros up to MJD 45700? IERS Auto doesn't match this
    ok_A &= A["MJD"] > 45700 * u.d
    # Maybe the old values are bogus?
    ok_A &= A["MJD"] > 50000 * u.d

    mjds_A = A["MJD"][ok_A].to(u.day).value
    i_B = np.searchsorted(B["MJD"].to(u.day).value, mjds_A)

    assert np.all(np.diff(i_B) == 1), "Valid region not contiguous"
    assert_equal(A["MJD"][ok_A], B["MJD"][i_B], "MJDs don't make sense")
    for atag, btag, unit in [
        ("UT1_UTC_B", "UT1_UTC", u.s),
        ("PM_X_B", "PM_x", u.arcsec),
        ("PM_Y_B", "PM_y", u.arcsec),
    ]:
        assert_allclose(
            A[atag][ok_A].to(unit).value,
            B[btag][i_B].to(unit).value,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"Inserted IERS B {atag} values don't match IERS_B_FILE {btag} values",
        )


copy_columns = [
    ("UT1_UTC", "UT1_UTC_B"),
    ("PM_x", "PM_X_B"),
    ("PM_y", "PM_Y_B"),
    pytest.param(
        "dX_2000A", "dX_2000A_B", marks=pytest.mark.xfail(reason="Bug in astropy")
    ),
    pytest.param(
        "dY_2000A", "dY_2000A_B", marks=pytest.mark.xfail(reason="Bug in astropy")
    ),
]


@pytest.mark.skip
@pytest.mark.parametrize("b_name,a_name", copy_columns)
def test_IERS_B_parameters_loaded_into_IERS_Auto(b_name, a_name):
    A = IERS_Auto.open()
    A[a_name]
    B = IERS_B.open(IERS_B_FILE)

    ok_A = A["MJD"] < B["MJD"][-1]

    mjds_A = A["MJD"][ok_A].to(u.day).value
    i_B = np.searchsorted(B["MJD"].to(u.day).value, mjds_A)

    assert_equal(np.diff(i_B), 1, err_msg="Valid region not contiguous")
    assert_equal(A["MJD"][ok_A], B["MJD"][i_B], err_msg="MJDs don't make sense")
    assert_equal(
        A[a_name][ok_A],
        B[b_name][i_B],
        err_msg=f"IERS B parameter {b_name} not copied over IERS A parameter {a_name}",
    )
