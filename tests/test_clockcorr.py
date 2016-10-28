from pint.observatory import Observatory
from pint.observatory.clock_file import ClockFile
import astropy.units as u
import numpy
import unittest

class TestClockcorrection(unittest.TestCase):
    # Note, these tests currently depend on external data (TEMPO clock
    # files, which could potentially change.  Values here are taken
    # from tempo version 2016-01-27 2bb9277
    def test_Parkes(self):
        obs = Observatory.get('Parkes')
        cf = ClockFile.read(obs.clock_fullpath, format=obs.clock_fmt,
                obscode=obs.tempo_code)
        mjd = cf.time.mjd
        corr = cf.clock.to(u.us).value
        
        assert numpy.isclose(mjd.min(), 44000.0)

        idx = numpy.where(numpy.isclose(mjd,49990.0))[0][0]
        assert numpy.isclose(corr[idx],-2.506)

        idx = numpy.where(numpy.isclose(mjd,55418.27))[0][0]
        assert numpy.isclose(corr[idx],-0.586)
