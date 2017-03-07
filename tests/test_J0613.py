"""Various tests to assess the performance of the J0623-0200."""
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

class TestJ0613(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileJ0613 = 'J0613-0200_NANOGrav_dfg+12_TAI_FB90.par'
        self.timJ0613 = 'J0613-0200_NANOGrav_dfg+12.tim'
        self.toasJ0613 = toa.get_TOAs(self.timJ0613, ephem="DE405", 
                                      planets=False, include_bipm=False)
        self.modelJ0613 = mb.get_model(self.parfileJ0613)
        # tempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileJ0613 + \
                                     '.tempo2_test',skip_header=1, unpack=True)
        print(self.ltres)
    def test_J0613_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ0613.binarymodel_delay(self.toasJ0613.table)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8), 'J0613 binary delay test failed.'

    def test_J0613(self):
        pint_resids_us = resids(self.toasJ0613, self.modelJ0613, False).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'J0613 residuals test failed.'

    def test_derivative(self):
        log= logging.getLogger( "TestJ0613.derivative_test")
        self.modelJ0613.PBDOT.value = 0.0 # For test PBDOT
        self.modelJ0613.EPS1DOT.value = 0.0
        self.modelJ0613.EPS2DOT.value = 0.0
        self.modelJ0613.A1DOT.value = 0.0
        testp = tdu.get_derivative_params(self.modelJ0613)
        delay = self.modelJ0613.delay(self.toasJ0613.table)
        # Change parameter test step
        testp['EPS1'] = 1
        testp['EPS2'] = 1
        testp['PMDEC'] = 1
        testp['PMRA'] = 1
        for p in testp.keys():
            log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
            ndf = self.modelJ0613.d_phase_d_param_num(self.toasJ0613.table, p, testp[p])
            adf = self.modelJ0613.d_phase_d_param(self.toasJ0613.table, delay, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf+ndf)/2.0
                relative_diff = np.abs(diff)/np.abs(mean_der)
                #print "Diff Max is :", np.abs(diff).max()
                msg = 'Derivative test failed at d_delay_d_%s with max relative difference %lf' % (p, np.nanmax(relative_diff).value)
                if p in ['EPS1DOT', 'EPS1']:
                    tol = 0.05
                else:
                    tol = 1e-3
                log.debug( "derivative relative diff for %s, %lf"%('d_delay_d_'+p, np.nanmax(relative_diff).value))
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

if __name__ == '__main__':
    pass
