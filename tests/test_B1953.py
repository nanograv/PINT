"""Various tests to assess the performance of the B1953+29."""
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

class TestB1953(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileB1953 = 'B1953+29_NANOGrav_dfg+12_TAI_FB90.par'
        self.timB1953 = 'B1953+29_NANOGrav_dfg+12.tim'
        self.toasB1953 = toa.get_TOAs(self.timB1953, ephem="DE405", planets=False)
        self.modelB1953 = mb.get_model(self.parfileB1953)
        # tempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileB1953 + \
                                     '.tempo2_test',skip_header=1, unpack=True)
        print(self.ltres)
    def test_B1953_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1953.binarymodel_delay(self.toasB1953.table)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8), 'B1953 binary delay test failed.'

    def test_B1953(self):
        pint_resids_us = resids(self.toasB1953, self.modelB1953).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'B1953 residuals test failed.'

    def test_derivative(self):
        dd, pd, nd = tdu.get_derivative_funcs(self.modelB1953)
        log= logging.getLogger( "TestB1953.derivative_test")
        for p in dd.keys():
            log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
            ndf = tdu.num_diff_delay(self.toasB1953.table, p, self.modelB1953)
            adf = self.modelB1953.d_delay_d_param(self.toasB1953.table, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_delay_d_%s with max relative difference %lf' % (p, np.nanmax(relative_diff).value)
                if p in ['EDOT', 'ECC']: # Due to the no d_delayR_d_ECC, this is not as accurate as numerical derivative
                    tol = 20.0
                else:
                    tol = 0.001
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

        for p in pd.keys():
            log.debug( "Runing derivative for %s", 'd_phase_d_'+p)
            delay = self.modelB1953.delay(self.toasB1953.table)
            ndf = tdu.num_diff_phase(self.toasB1953.table, p, self.modelB1953)
            adf = pd[p](self.toasB1953.table, delay)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_phase_d_%s with max relative difference %lf' % (p, relative_diff.max().value)
                assert np.nanmax(relative_diff) < 0.001, msg
            else:
                continue

if __name__ == '__main__':
    pass
