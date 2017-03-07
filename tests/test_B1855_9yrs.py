"""Various tests to assess the performance of the B1855+09."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
import test_derivative_utils as tdu
import logging

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestB1855(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileB1855 = 'B1855+09_NANOGrav_9yv1.gls.par'
        self.timB1855 = 'B1855+09_NANOGrav_9yv1.tim'
        self.toasB1855 = toa.get_TOAs(self.timB1855, ephem="DE421", 
                                      planets=False, include_bipm=False)
        self.modelB1855 = mb.get_model(self.parfileB1855)
        # tempo result
        self.ltres= np.genfromtxt(self.parfileB1855 + \
                                  '.tempo2_test',skip_header=1, unpack=True)

    def test_B1855(self):
        pint_resids_us = resids(self.toasB1855, self.modelB1855, False).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'B1855 residuals test failed.'

    def test_derivative(self):
        log= logging.getLogger( "TestB1855.derivative_test")
        testp = tdu.get_derivative_params(self.modelB1855)
        delay = self.modelB1855.delay(self.toasB1855.table)
        for p in testp.keys():
            log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
            ndf = self.modelB1855.d_phase_d_param_num(self.toasB1855.table, p, testp[p])
            adf = self.modelB1855.d_phase_d_param(self.toasB1855.table, delay, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_delay_d_%s with max relative difference %lf' % (p, np.nanmax(relative_diff).value)
                if p in ['SINI']:
                    tol = 0.7
                else:
                    tol = 1e-3
                log.debug( "derivative relative diff for %s, %lf"%('d_delay_d_'+p, np.nanmax(relative_diff).value))
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

if __name__ == '__main__':
    pass
