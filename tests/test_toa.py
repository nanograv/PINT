import unittest
import astropy.units as u
from astropy.time import Time
from pint.toa import TOA, TOAs
from pint.observatory import get_observatory


class TestTOA(unittest.TestCase):
    """Test of TOA class
    """

    def setUp(self):
        self.MJD = 57000

    def test_units(self):
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD * u.m)
        with self.assertRaises(u.UnitConversionError):
            t = TOA((self.MJD * u.m, 0))
        t = TOA((self.MJD * u.day).to(u.s))
        with self.assertRaises(u.UnitConversionError):
            t = TOA((self.MJD * u.day, 0))
        t = TOA((self.MJD * u.day, 0 * u.day))
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD, error=1 * u.m)
        t = TOA(self.MJD, freq=100 * u.kHz)
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD, freq=100 * u.s)

    def test_precision_mjd(self):
        t = TOA(self.MJD)
        self.assertEqual(t.mjd.precision, 9)

    def test_precision_time(self):
        t = TOA(Time("2008-08-19", format="iso", precision=1))
        self.assertEqual(t.mjd.precision, 9)

    def test_typo(self):
        TOA(self.MJD, errror=1)
        with self.assertRaises(TypeError):
            TOA(self.MJD, errror=1, flags={})


class TestTOAs(unittest.TestCase):
    """Test of TOAs class
    """

    def setUp(self):
        self.freq = 1440.012345678 * u.MHz
        self.obs = "gbt"
        self.MJD = 57000
        self.error = 3.0

    def test_make_TOAs(self):
        t = TOA(self.MJD, freq=self.freq, obs=self.obs, error=self.error)
        t_list = [t, t]
        assert t_list[0].mjd.precision == 9
        assert t_list[1].mjd.precision == 9
        assert t_list[0].mjd.location is not None
        assert t_list[1].mjd.location is not None
        # Check information in the TOAs table
        toas = TOAs(toalist=t_list)
        assert toas.table[1]["freq"] == self.freq.to_value(u.MHz)
        assert toas.table["freq"].unit == self.freq.unit
        assert toas.table[1]["obs"] == self.obs
        assert toas.table[1]["error"] == self.error
        assert toas.table["error"].unit == u.us
        assert toas.table["mjd"][0].precision == 9
        assert toas.table["mjd"][0].location is not None

    def test_mulit_obs(self):
        obs1 = "gbt"
        obs2 = "ao"
        obs3 = "barycenter"
        site1 = get_observatory(obs1)
        site2 = get_observatory(obs2)
        site3 = get_observatory(obs3)
        t1 = TOA(self.MJD, freq=self.freq, obs=obs1, error=self.error)
        t2 = TOA(self.MJD + 1.0, freq=self.freq, obs=obs2, error=self.error)
        t3 = TOA(self.MJD + 1.0, freq=self.freq, obs=obs3, error=self.error)

        toas = TOAs(toalist=[t1, t2, t3])

        # Table will be grouped by observatories, and will be sorted by the
        # observatory so the TOA order will be different
        assert toas.table["obs"][0] == site2.name
        assert toas.table["mjd"][0] == t2.mjd
        assert toas.table["obs"][1] == site3.name
        assert toas.table["mjd"][1] == t3.mjd
        assert toas.table["obs"][2] == site1.name
        assert toas.table["mjd"][2] == t1.mjd

        # obs in time object
        assert toas.table["mjd"][0].location == site2.earth_location_itrf()
        assert toas.table["mjd"][1].location == site3.earth_location_itrf()
        assert toas.table["mjd"][2].location == site1.earth_location_itrf()
