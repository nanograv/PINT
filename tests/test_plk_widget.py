import astropy.units as u
import unittest
from pint.pintk.plk import PlkWidget
from tkinter import Frame


class TestPlkWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # use __new__ to prevent an actual tkinter window from popping up
        cls.widget = Frame.__new__(PlkWidget)

    def test_determine_yaxis_units(self):
        # test interval where no unit conversion happens
        miny = 5 * u.us
        maxy = 10 * u.us
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, miny.value)
        self.assertEqual(newmin.unit, miny.unit)
        self.assertAlmostEqual(newmax.value, maxy.value)
        self.assertEqual(newmax.unit, maxy.unit)

        # test interval where u.us converted to u.ms
        maxy = 10000 * u.us
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, 0.005)
        self.assertEqual(newmin.unit, u.ms)
        self.assertAlmostEqual(newmax.value, 10)
        self.assertEqual(newmax.unit, u.ms)

        # test interval where u.us converted to u.s
        maxy = 210000 * u.us
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, 0.000005)
        self.assertEqual(newmin.unit, u.s)
        self.assertAlmostEqual(newmax.value, 0.21)
        self.assertEqual(newmax.unit, u.s)

        # test u.ms to u.ms
        miny = 5 * u.ms
        maxy = 10 * u.ms
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, miny.value)
        self.assertEqual(newmin.unit, miny.unit)
        self.assertAlmostEqual(newmax.value, maxy.value)
        self.assertEqual(newmax.unit, maxy.unit)

        # test u.ms to u.s
        maxy = 10000 * u.ms
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, 0.005)
        self.assertEqual(newmin.unit, u.s)
        self.assertAlmostEqual(newmax.value, 10)
        self.assertEqual(newmax.unit, u.s)

        # test u.ms to u.us
        maxy = 6 * u.ms
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, 5000)
        self.assertEqual(newmin.unit, u.us)
        self.assertAlmostEqual(newmax.value, 6000)
        self.assertEqual(newmax.unit, u.us)

        # test u.s to u.s
        miny = 5 * u.s
        maxy = 10 * u.s
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, miny.value)
        self.assertEqual(newmin.unit, miny.unit)
        self.assertAlmostEqual(newmax.value, maxy.value)
        self.assertEqual(newmax.unit, maxy.unit)

        # test u.s to u.ms
        maxy = 5.5 * u.us
        newmin, newmax = self.widget.determine_yaxis_units(maxy, miny)

        self.assertAlmostEqual(newmin.value, miny.value)
        self.assertEqual(newmin.unit, miny.unit)
        self.assertAlmostEqual(newmax.value, maxy.value)
        self.assertEqual(newmax.unit, maxy.unit)
