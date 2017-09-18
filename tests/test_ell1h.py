"""Tests of ELL1H model """
import pint.models as model
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
import test_derivative_utils as tdu
import logging

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestELL1H(unittest.TestCase):
    """Compare delays from the ELL1 model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileJ1853 = "J1853+1303_NANOGrav_11yv0.gls.par"
        self.timfileJ1853 = "J1853+1303_NANOGrav_11yv0.tim"
        self.toasJ1853 = toa.get_TOAs(self.timfileJ1853, ephem="DE421",
                                      planets=False)
        self.modelJ1853 = model.get_model(self.parfileJ1853)
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileJ1853 + \
                                     '.tempo2_test',skip_header=1, unpack=True)
    def test_J1853(self):
        pint_resids_us = resids(self.toasJ1853, self.modelJ1853, False).time_resids.to(u.s)
        # Due to PINT has higher order of ELL1 model, Tempo2 gives a difference around 3e-8
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'J1853 residuals test failed.'

    def test_J1853_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ1853.binarymodel_delay(self.toasJ1853.table, None)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 3e-8), 'J1853 binary delay test failed.'


    def test_derivative(self):
        log= logging.getLogger( "TestJ1853.derivative_test")
        test_params = ['H3', 'H4', 'STIGMA']
        self.modelJ1853.H4.value = 0.0 # For test PBDOT
        self.modelJ1853.STIGMA.value = 0.0
        testp = tdu.get_derivative_params(self.modelJ1853)
        delay = self.modelJ1853.delay(self.toasJ1853.table)
        # Change parameter test step
        testp['H3'] = 5e-1
        testp['H4'] = 1e-2
        testp['STIGMA'] = 1e-2
        for p in test_params:
            log.debug( "Runing derivative for %s", 'd_delay_d_'+p)
            ndf = self.modelJ1853.d_phase_d_param_num(self.toasJ1853.table, p, testp[p])
            adf = self.modelJ1853.d_phase_d_param(self.toasJ1853.table, delay, p)
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
