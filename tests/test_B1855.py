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
        self.parfileB1855 = 'B1855+09_NANOGrav_dfg+12_TAI_FB90.par'
        self.timB1855 = 'B1855+09_NANOGrav_dfg+12.tim'
        self.toasB1855 = toa.get_TOAs(self.timB1855, ephem="DE405", planets=False)
        self.modelB1855 = mb.get_model(self.parfileB1855)
        # tempo result
        self.ltres= np.genfromtxt(self.parfileB1855 + \
                                  '.tempo2_test',skip_header=1, unpack=True)
        print(self.ltres)
    # def test_B1855_binary_delay(self):
    #     # Calculate delays with PINT
    #     pint_binary_delay = self.modelB1855.binarymodel_delay(self.toasB1855.table)
    #     assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8), 'B1855 binary delay test failed.'

    def test_B1855(self):
        pint_resids_us = resids(self.toasB1855, self.modelB1855).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'B1855 residuals test failed.'

    def test_derivative(self):
        dd, pd, nd = tdu.get_derivative_funcs(self.modelB1855)
        log= logging.getLogger( "TestB1855.derivative_test")
        for p in dd.keys():
            log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
            ndf = tdu.num_diff_delay(self.toasB1855.table, p, self.modelB1855)
            adf = self.modelB1855.d_delay_d_param(self.toasB1855.table, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_delay_d_%s with max relative difference %lf' % (p, np.nanmax(relative_diff).value)
                assert np.nanmax(relative_diff) < 0.001, msg
            else:
                continue

        for pp in self.modelB1855.params:
            par = getattr(self.modelB1855, pp)
            if par.frozen is True:
                continue
            log.debug( "Runing derivative for %s", 'd_phase_d_'+pp)
            delay = self.modelB1855.delay(self.toasB1855.table)
            ndf = tdu.num_diff_phase(self.toasB1855.table, pp, self.modelB1855)
            adf = self.modelB1855.d_phase_d_param(self.toasB1855.table, delay, pp)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                if pp in ['SINI', 'PX']:
                    tol = 0.07
                else:
                    tol = 0.001
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_phase_d_%s with max relative difference %lf' % (pp, relative_diff.max().value)
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

if __name__ == '__main__':
    pass
