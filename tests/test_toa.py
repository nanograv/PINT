import unittest

import astropy.units as u
from astropy.time import Time
import astropy.units as u
from pint.toa import TOA, TOAs
from pint.observatory import get_observatory


class TestTOA(unittest.TestCase):
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
    def setUp(self):
        self.freq = 1440.012345678 * u.MHz
        self.obs = 'gbt'
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
        assert toas.table[1]['freq'] == self.freq.to_value(u.MHz)
        assert toas.table['freq'].unit == self.freq.unit
        assert toas.table[1]['obs'] == self.obs
        assert toas.table[1]['error'] == self.error
        assert toas.table['error'].unit == u.us
        assert toas.table['mjd'][0].precision == 9
        assert toas.table['mjd'][0].location is not None
   
