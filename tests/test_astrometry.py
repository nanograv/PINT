import os.path
from io import StringIO
from packaging import version

import astropy.units as u
import erfa
import hypothesis.strategies as st
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude
from hypothesis import example, given
from pint.models import get_model
from pinttestdata import datadir


@pytest.fixture
def model():
    parfile = os.path.join(datadir, "J1600-3053_test.par")
    return get_model(parfile)


def test_ecliptic_to_icrs(model):
    astrometry_component = model.components["AstrometryEcliptic"]
    params_icrs = astrometry_component.get_params_as_ICRS()

    # Values from NANOGrav 11-year timing paper (Arzoumanian et al. 2018)
    # Ecliptic coords in test par file are from the same source.
    ref_ra = Longitude("16:00:51.903178", u.hourangle)
    ref_dec = Latitude("-30:53:49.3919", u.deg)

    assert np.abs(params_icrs["RAJ"] - ref_ra) < 40 * u.uas
    assert np.abs(params_icrs["DECJ"] - ref_dec) < 40 * u.uas


par_basic_ecliptic = """
PSR J1234+5678
F0 1
PEPOCH 57000
ELAT 0
ELONG 0
DM 10
"""

par_basic_equatorial = """
PSR J1234+5678
F0 1
PEPOCH 57000
RAJ 10:23:47.67
DECJ 00:38:41.2
DM 10
"""


def test_pm_unset():
    m = get_model(StringIO(par_basic_ecliptic))
    assert m.PMELAT.value == 0
    assert m.PMELONG.value == 0
    assert m.POSEPOCH.quantity is None


def test_pm_unset_equatorial():
    m = get_model(StringIO(par_basic_equatorial))
    assert m.PMRA.value == 0
    assert m.PMDEC.value == 0
    assert m.POSEPOCH.quantity is None


def test_pm_acquires_posepoch():
    m = get_model(StringIO(par_basic_ecliptic))
    assert m.POSEPOCH.quantity is None
    m.PMELAT.value = 7
    m.validate()
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity


def test_pm_acquires_posepoch_equatorial():
    m = get_model(StringIO(par_basic_equatorial))
    assert m.POSEPOCH.quantity is None
    m.PMRA.value = 7
    m.validate()
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity


def test_pm_one_set_other_not():
    m = get_model(StringIO("\n".join([par_basic_equatorial, "PMRA 7"])))
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity


parICRS = """PSR              1748-2021E
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
PMRA     1000
PMDEC    -500
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE421
CLK              UTC(NIST)
UNITS               TDB
TIMEEPH             FB90
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      N
DILATEFREQ          N
TZRMJD  53801.38605118223
TZRFRQ            1949.609
TZRSITE                  1
"""


@given(
    st.floats(0, 360),
    st.floats(-89.9, 89.9),
    st.floats(0, 10),
    st.floats(0, 10),
)
@example(ra=1.0, dec=1.0, pmra=2.0, pmdec=4.345847379899e-311)
@pytest.mark.xfail(
    version.parse(erfa.__version__) < version.parse("2.0.1"),
    reason="Old version of pyerfa had a bug",
)
def test_ssb_accuracy_ICRS(ra, dec, pmra, pmdec):
    m = get_model(StringIO(parICRS), RAJ=ra, DECJ=dec, PMRA=pmra, PMDEC=pmdec)
    t0 = m.POSEPOCH.quantity
    t0.format = "pulsar_mjd"
    t = t0 + np.linspace(0, 20) * u.yr
    xyzastropy = m.coords_as_ICRS(epoch=t).cartesian.xyz.transpose()
    xyznew = m.ssb_to_psb_xyz_ICRS(epoch=t)
    assert np.allclose(xyznew, xyzastropy)


parECL = """PSR              1748-2021E
ELONG       300.0
ELAT      -55.0
PMELONG     1000
PMELAT    -500
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE421
CLK              UTC(NIST)
UNITS               TDB
TIMEEPH             FB90
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      N
DILATEFREQ          N
TZRMJD  53801.38605118223
TZRFRQ            1949.609
TZRSITE                  1
"""


@given(
    st.floats(0, 360),
    st.floats(-89.9, 89.9),
    st.floats(0, 10),
    st.floats(0, 10),
)
def test_ssb_accuracy_ICRS_ECLmodel(elong, elat, pmelong, pmelat):
    m = get_model(
        StringIO(parECL), ELONG=elong, ELAT=elat, PMELONG=pmelong, PMELAT=pmelat
    )
    t0 = m.POSEPOCH.quantity
    t0.format = "pulsar_mjd"
    t = t0 + np.linspace(0, 20) * u.yr
    xyzastropy = m.coords_as_ICRS(epoch=t).cartesian.xyz.transpose()
    xyznew = m.ssb_to_psb_xyz_ICRS(epoch=t)
    assert np.allclose(xyznew, xyzastropy)


@given(
    st.floats(0, 360),
    st.floats(-89.9, 89.9),
    st.floats(0, 10),
    st.floats(0, 10),
)
@example(1.0, 1.0, 0.5, 4.345847379899e-311)
@pytest.mark.xfail(
    version.parse(erfa.__version__) < version.parse("2.0.1"),
    reason="Old version of pyerfa had a bug",
)
def test_ssb_accuracy_ECL_ECLmodel(elong, elat, pmelong, pmelat):
    m = get_model(
        StringIO(parECL), ELONG=elong, ELAT=elat, PMELONG=pmelong, PMELAT=pmelat
    )
    t0 = m.POSEPOCH.quantity
    t0.format = "pulsar_mjd"
    t = t0 + np.linspace(0, 20) * u.yr
    xyzastropy = m.coords_as_ECL(epoch=t).cartesian.xyz.transpose()
    xyznew = m.ssb_to_psb_xyz_ECL(epoch=t)
    assert np.allclose(xyznew, xyzastropy)
