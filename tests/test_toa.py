import unittest

import astropy.units as u
from astropy.time import Time

from pint.toa import TOA

class TestTOA(unittest.TestCase):
    def setUp(self):
        self.MJD = 57000
    def test_units(self):
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD*u.m)
        with self.assertRaises(u.UnitConversionError):
            t = TOA((self.MJD*u.m,0))
        t = TOA((self.MJD*u.day).to(u.s))
        with self.assertRaises(u.UnitConversionError):
            t = TOA((self.MJD*u.day,0))
        t = TOA((self.MJD*u.day,0*u.day))
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD, error=1*u.m)
        t = TOA(self.MJD,freq=100*u.kHz)
        with self.assertRaises(u.UnitConversionError):
            t = TOA(self.MJD,freq=100*u.s)
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

