"""Tests of ELL1H model """
import logging
import os
import unittest
import pytest

import astropy.units as u
import numpy as np

import pint.fitter as ff
import pint.models as model
import pint.toa as toa
import test_derivative_utils as tdu
from pint.residuals import Residuals
from pinttestdata import datadir
from io import StringIO

simple_par = """
PSR    J0613-0200
LAMBDA 93.7990065496191  1 0.0000000158550
BETA   -25.4071326875232  1 0.0000000369013
PMLAMBDA 2.1192  1 0.0174
PMBETA -10.3422  1              0.0433
PX 0.9074  1              0.1509
F0 326.6005670972169810  1  0.0000000000066373
F1 -1.022985317101D-15  1  6.219122230955D-20
PEPOCH        54890.000000
DM         38.778683
BINARY ELL1H
A1 1.091442190  1  0.000000598
PB 1.19851255667964  1 0.00000000001332
TASC 54889.991808565  1 0.000000012
EPS1 0.0000025554  1 0.0000002783
EPS2 0.0000036160  1 0.0000000847
H3 2.7507208E-7  1       1.5114416E-7
H4 2.0262048E-7  1       1.1276173E-7
"""

os.chdir(datadir)


class TestELL1H:
    """Compare delays from the ELL1 model with tempo and PINT"""

    def setup(self):
        self.parfileJ1853 = "J1853+1303_NANOGrav_11yv0.gls.par"
        self.timfileJ1853 = "J1853+1303_NANOGrav_11yv0.tim"
        self.toasJ1853 = toa.get_TOAs(self.timfileJ1853, ephem="DE421", planets=False)
        self.modelJ1853 = model.get_model(self.parfileJ1853)
        self.ltres, self.ltbindelay = np.genfromtxt(
            self.parfileJ1853 + ".tempo2_test", skip_header=1, unpack=True
        )

        self.parfileJ0613 = "J0613-0200_NANOGrav_9yv1_ELL1H.gls.par"
        self.timfileJ0613 = "J0613-0200_NANOGrav_9yv1.tim"
        self.modelJ0613 = model.get_model(self.parfileJ0613)
        self.toasJ0613 = toa.get_TOAs(self.timfileJ0613, ephem="DE421", planets=False)
        self.parfileJ0613_STIG = "J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par"
        self.modelJ0613_STIG = model.get_model(self.parfileJ0613_STIG)

    def test_J1853(self):
        pint_resids_us = Residuals(
            self.toasJ1853, self.modelJ1853, use_weighted_mean=False
        ).time_resids.to(u.s)
        # Due to PINT has higher order of ELL1 model, Tempo2 gives a difference around 3e-8
        # Changed to 4e-8 since modification to get_PSR_freq() makes this 3.1e-8
        log = logging.getLogger("TestJ1853.J1853_residuals")
        diffs = np.abs(pint_resids_us.value - self.ltres)
        log.debug("Diffs: %s\nMax: %s" % (diffs, np.max(diffs)))
        assert np.all(diffs < 4e-8), "J1853 residuals test failed."

    def test_J1853_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ1853.binarymodel_delay(self.toasJ1853, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 3e-8
        ), "J1853 binary delay test failed."

    def test_derivative(self):
        log = logging.getLogger("TestJ1853.derivative_test")
        test_params = ["H3", "H4", "STIGMA"]
        self.modelJ1853.H4.value = 0.0  # For test PBDOT
        self.modelJ1853.STIGMA.value = 0.0
        testp = tdu.get_derivative_params(self.modelJ1853)
        delay = self.modelJ1853.delay(self.toasJ1853)
        # Change parameter test step
        testp["H3"] = 6e-1
        testp["H4"] = 1e-2
        testp["STIGMA"] = 1e-2
        for p in test_params:
            log.debug("Runing derivative for %s", "d_delay_d_" + p)
            ndf = self.modelJ1853.d_phase_d_param_num(self.toasJ1853, p, testp[p])
            adf = self.modelJ1853.d_phase_d_param(self.toasJ1853, delay, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf + ndf) / 2.0
                relative_diff = np.abs(diff) / np.abs(mean_der)
                # print "Diff Max is :", np.abs(diff).max()
                msg = (
                    "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                    % (p, np.nanmax(relative_diff).value)
                )
                if p in ["EPS1DOT", "EPS1"]:
                    tol = 0.05
                else:
                    tol = 1e-3
                log.debug(
                    "derivative relative diff for %s, %lf"
                    % ("d_delay_d_" + p, np.nanmax(relative_diff).value)
                )
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

    def test_J0613_H4(self):
        log = logging.getLogger("TestJ0613.fit_tests")
        f = ff.GLSFitter(self.toasJ0613, self.modelJ0613)
        f.fit_toas()
        f.model.free_params = ("H3", "H4")
        for pn, p in f.model.get_params_dict("free", "quantity").items():
            op = getattr(f.model_init, pn)
            diff = np.abs(p.value - op.value)
            sigma = diff / op.uncertainty_value
            # Fit th
            assert (
                sigma < 0.7
            ), "refit %s is %lf sigma different from original value" % (pn, sigma)

    def test_J0613_STIG(self):
        log = logging.getLogger("TestJ0613.fit_tests_stig")
        f = ff.GLSFitter(self.toasJ0613, self.modelJ0613_STIG)
        f.fit_toas()
        f.model.free_params = ("H3", "STIGMA")
        for pn, p in f.get_params_dict("free", "quantity").items():
            op = getattr(f.model_init, pn)
            diff = np.abs(p.value - op.value)
            sigma = diff / op.uncertainty_value
            # Fit th
            assert (
                sigma < 0.7
            ), "refit %s is %lf sigma different from original value" % (pn, sigma)

    def test_no_H3_H4(self):
        """ Test no H3 and H4 in model.
        """
        no_H3_H4 = simple_par.replace("H4 2.0262048E-7  1       1.1276173E-7", "")
        no_H3_H4 = no_H3_H4.replace("H3 2.7507208E-7  1       1.5114416E-7", "")
        print(no_H3_H4)
        no_H3_H4_model = model.get_model(StringIO(no_H3_H4))
        assert no_H3_H4_model.H3.value == None
        assert no_H3_H4_model.H4.value == None
        test_toas = self.toasJ0613[::20]
        f = ff.WLSFitter(test_toas, no_H3_H4_model)
        f.fit_toas()

    def test_H3_H4_pairs(self):
        """ Testing if the different H3, H4 combination breaks the fitting. the
        fitting result will not be checked here.
        """
        simple_model = model.get_model(StringIO(simple_par))

        test_toas = self.toasJ0613[::20]
        f = ff.WLSFitter(test_toas, simple_model)
        f.fit_toas()

        # Zero H4
        H4_zero = simple_par.replace(
            "H4 2.0262048E-7  1       1.1276173E-7", "H4 0  1  0"
        )
        H4_zero_model = model.get_model(StringIO(H4_zero))
        assert H4_zero_model.H4.value == 0.0
        assert H4_zero_model.H3.value != 0.0
        f = ff.WLSFitter(test_toas, H4_zero_model)
        f.fit_toas()

        # Zero H3
        H3_zero = simple_par.replace(
            "H3 2.7507208E-7  1       1.5114416E-7", "H3 0  1  0"
        )
        H3_zero_model = model.get_model(StringIO(H3_zero))
        assert H3_zero_model.H3.value == 0.0
        assert H3_zero_model.H4.value != 0.0
        f = ff.WLSFitter(test_toas, H3_zero_model)
        with pytest.raises(ValueError):
            f.fit_toas()

        # Zero H3 and H4 and fit for H3 and H4
        H3H4_zero = H4_zero.replace(
            "H3 2.7507208E-7  1       1.5114416E-7", "H3 0  1  0"
        )
        H3H4_zero_model = model.get_model(StringIO(H3H4_zero))
        assert H3H4_zero_model.H3.value == 0.0
        assert "H3" in H3H4_zero_model.free_params
        assert H3H4_zero_model.H4.value == 0.0
        assert "H4" in H3H4_zero_model.free_params
        f = ff.WLSFitter(test_toas, H3H4_zero_model)
        # Fitting H4 with H3 == 0, will give a ValueError
        with pytest.raises(ValueError):
            f.fit_toas()
        # Not fitting H4
        H3H4_zero2 = H3H4_zero.replace("H4 0  1  0", "H4 0  0  0")
        H3H4_zero2_model = model.get_model(StringIO(H3H4_zero2))
        assert H3H4_zero2_model.H3.value == 0.0
        assert "H3" in H3H4_zero2_model.free_params
        assert H3H4_zero2_model.H4.value == 0.0
        assert "H4" not in H3H4_zero2_model.free_params
        f = ff.WLSFitter(test_toas, H3H4_zero2_model)
        # This should work
        f.fit_toas()


if __name__ == "__main__":
    pass
