"""Tests for jump model component """
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir
import test_derivative_utils as tdu
import logging

os.chdir(datadir)


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = 'B1855+09_NANOGrav_dfg+12_TAI.par'
        self.timf = 'B1855+09_NANOGrav_dfg+12.tim'
        self.JUMPm = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf, ephem="DE405", 
                                 planets=False, include_bipm=False)
        # libstempo calculation
        self.ltres = np.genfromtxt(self.parf + '.tempo_test', unpack=True, names=True, dtype='float128')
    def test_jump(self):
        presids_s = resids(self.toas, self.JUMPm, False).time_resids.to(u.s)
        assert np.all(np.abs(presids_s.value - self.ltres['residuals']) < 1e-7), "JUMP test failed."

    def test_derivative(self):
        log= logging.getLogger( "Jump delay test")
        p = 'JUMP2'
        log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
        ndf = self.JUMPm.d_delay_d_param_num(self.toas.table, p)
        adf = self.JUMPm.d_delay_d_param(self.toas.table, p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf+ndf)/2.0
            relative_diff = np.abs(diff)/np.abs(mean_der)
            #print "Diff Max is :", np.abs(diff).max()
            msg = 'Derivative test failed at d_delay_d_%s with max relative difference %lf' % (p, np.nanmax(relative_diff).value)
            assert np.nanmax(relative_diff) < 0.001, msg
if __name__ == '__main__':
    pass
