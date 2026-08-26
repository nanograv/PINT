from astropy import units as u, constants as c
from astropy.time import Time
import numpy as np
import pint.solar_system_ephemerides
from pint.models import get_model
import pint.residuals
import pint.simulation
import io
import pytest


@pytest.mark.parametrize("ephem", ["de434", "de430", "de436", "de440"])
def test_tt_tdb(ephem):

    t = Time(np.arange(55000, 59000, 10), format="mjd")

    tdb_tt_inpop = pint.solar_system_ephemerides.get_tdb_tt_ephem_geocenter(
        t.tt, "inpop21a"
    )
    tdb_tt_jpl = pint.solar_system_ephemerides.get_tdb_tt_ephem_geocenter(
        t.tt, ephem.upper() + "t"
    )
    assert np.allclose(tdb_tt_inpop, tdb_tt_jpl, atol=1e-7)


@pytest.mark.parametrize("ephem", ["de434", "de430", "de436", "de440"])
def test_pv(ephem):
    t = Time(np.arange(55000, 59000, 10), format="mjd")
    pv_inpop = pint.solar_system_ephemerides.objPosVel_wrt_SSB(
        "Earth", t.tdb, "inpop21a"
    )
    pv_jpl = pint.solar_system_ephemerides.objPosVel_wrt_SSB(
        "Earth", t.tdb, ephem.upper()
    )

    assert np.allclose(pv_inpop.pos.to_value(u.km), pv_jpl.pos.to_value(u.km), atol=200)
    assert np.allclose(
        pv_inpop.vel.to_value(u.km / u.s), pv_jpl.vel.to_value(u.km / u.s), atol=1e-7
    )


def test_residuals():

    model = """PSR              1748-2021E
    RAJ       17:48:52.75  1
    DECJ      -20:21:29.0  1
    F0       61.485476554  1
    F1         -1.181D-15  1
    PEPOCH        53750.000000
    POSEPOCH      53750.000000
    DM              223.9  1
    SOLARN0               0.00
    EPHEM               DE421
    CLK              TT(BIPM2023)
    UNITS               TDB
    TIMEEPH             FB90
    T2CMETHOD           TEMPO
    CORRECT_TROPOSPHERE N
    PLANET_SHAPIRO      N
    DILATEFREQ          N
    TZRMJD  53801.38605118223
    TZRFRQ            1949.609
    TZRSITE                  1
    """

    m1 = get_model(io.StringIO(model))
    t = pint.simulation.make_fake_toas_uniform(
        55000, 56000, 100, model=m1, add_noise=False
    )
    r1 = pint.residuals.Residuals(t, m1)

    m2 = get_model(io.StringIO(model), EPHEM="INPOP21a")
    t.compute_posvels(m2.EPHEM.value)
    r2 = pint.residuals.Residuals(t, m2)
    assert np.allclose(r1.phase_resids, r2.phase_resids, atol=1e-4)
