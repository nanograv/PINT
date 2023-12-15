"""Various tests to assess the performance of the DD model."""

import copy
import logging
import os
import pytest
from io import StringIO
import warnings

import astropy.units as u
import numpy as np
import pytest
import test_derivative_utils as tdu
from pinttestdata import datadir
from utils import verify_stand_alone_binary_parameter_updates

import pint.models.model_builder as mb
from pint.models import get_model
import pint.simulation
import pint.toa as toa
from pint.models.parameter import boolParameter
from pint.models.timing_model import MissingParameter
from pint.residuals import Residuals
import pint.fitter

temp_par_str = """
    PSR  J1713+0747
    LAMBDA 256.66  1 0.001
    BETA 30.70036  1 0.001
    PMLAMBDA 5.2671  1  0.0021
    PMBETA  -3.4428  1  0.0043
    PX  0.8211  1  0.0258
    F0  218.81  1  0.01
    PEPOCH  55391.0
    BINARY  DDK
    A1 32.34  1  0.001
    E  0.074  1  0.001
    T0 55388.836  1  0.0002
    PB 67.825129  1  0.0001
    OM 176.19845  1  0.0013
    M2  0.283395  1  0.0104
    KOM   83.100  1  1.800
    K96  1
"""


class TestDDK:
    """Compare delays from the ddk model with libstempo and PINT"""

    @classmethod
    def setup_class(cls):
        cls.parfileJ1713 = "J1713+0747_NANOGrav_11yv0_short.gls.par"
        cls.ICRSparfileJ1713 = "J1713+0747_NANOGrav_11yv0_short.gls.ICRS.par"
        cls.timJ1713 = "J1713+0747_NANOGrav_11yv0_short.tim"
        cls.toasJ1713 = toa.get_TOAs(
            os.path.join(datadir, cls.timJ1713), ephem="DE421", planets=False
        )
        index = cls.toasJ1713.table["index"]
        cls.ECLmodelJ1713 = mb.get_model(os.path.join(datadir, cls.parfileJ1713))
        cls.ICRSmodelJ1713 = mb.get_model(os.path.join(datadir, cls.ICRSparfileJ1713))
        # libstempo result
        # calculate using: datafile/make_J1713_libstempo.py
        # make sure to sort to agree with the grouped TOAs
        (
            cls.ECLltt,
            cls.ECLltdt,
            cls.ECLltf,
            cls.ECLltres,
            cls.ECLltbindelay,
        ) = np.genfromtxt(
            os.path.join(datadir, f"{cls.parfileJ1713}.libstempo"), unpack=True
        )[
            :, index
        ]
        (
            cls.ICRSltt,
            cls.ICRSltdt,
            cls.ICRSltf,
            cls.ICRSltres,
            cls.ICRSltbindelay,
        ) = np.genfromtxt(
            os.path.join(datadir, f"{cls.ICRSparfileJ1713}.libstempo"), unpack=True
        )[
            :, index
        ]

    def test_j1713_ecl_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.ECLmodelJ1713.binarymodel_delay(self.toasJ1713, None)
        print(f"{np.abs(pint_binary_delay.value + self.ECLltbindelay).max()}")
        assert np.all(np.abs(pint_binary_delay.value + self.ECLltbindelay) < 5e-6), (
            "DDK J1713 ECL BINARY DELAY TEST FAILED: max difference is %e"
            % np.abs(pint_binary_delay.value + self.ECLltbindelay).max()
        )

    def test_j1713_icrs_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.ICRSmodelJ1713.binarymodel_delay(self.toasJ1713, None)
        print(f"{np.abs(pint_binary_delay.value + self.ECLltbindelay).max()}")
        assert np.all(np.abs(pint_binary_delay.value + self.ECLltbindelay) < 6e-6), (
            "DDK J1713 ICRS BINARY DELAY TEST FAILED: max difference is %e"
            % np.abs(pint_binary_delay.value + self.ICRSltbindelay).max()
        )

    def test_j1713_ecl(self):
        pint_resids_us = Residuals(
            self.toasJ1713, self.ECLmodelJ1713, use_weighted_mean=False
        ).time_resids.to(u.s)
        diff = pint_resids_us.value - self.ECLltres
        print("Max diff %e" % np.abs(diff - diff.mean()).max())
        assert np.all(np.abs(diff - diff.mean()) < 2e-8), (
            "DDK J1713 ECL RESIDUAL TEST FAILED: max difference is %e"
            % np.abs(diff - diff.mean()).max()
        )

    def test_j1713_icrs(self):
        pint_resids_us = Residuals(
            self.toasJ1713, self.ICRSmodelJ1713, use_weighted_mean=False
        ).time_resids.to(u.s)
        diff = pint_resids_us.value - self.ICRSltres
        print("Max diff %e" % np.abs(diff - diff.mean()).max())
        assert np.all(np.abs(diff - diff.mean()) < 2e-8), (
            "DDK J1713 ICRS RESIDUAL TEST FAILED: max difference is %e"
            % np.abs(diff - diff.mean()).max()
        )

    def test_change_px(self):
        self.ECLmodelJ1713.update_binary_object(toas=self.toasJ1713)
        assert (
            self.ECLmodelJ1713.binary_instance.PX.value == self.ECLmodelJ1713.PX.value
        )
        bdelay0 = self.ECLmodelJ1713.binary_instance.binary_delay()
        b_time0 = self.ECLmodelJ1713.binary_instance.t
        # Change PX value
        self.ECLmodelJ1713.PX.value = 0.1
        self.ECLmodelJ1713.update_binary_object(None)
        b_time1 = self.ECLmodelJ1713.binary_instance.t
        assert self.ECLmodelJ1713.binary_instance.PX.value == 0.1
        # The stand alone binary model's input time should not change
        assert np.all(b_time0 == b_time1)
        # Check if the time residual changed
        bdelay1 = self.ECLmodelJ1713.binary_instance.binary_delay()
        diff = bdelay0 - bdelay1
        assert np.all(diff != 0)

    def test_j1713_deriv(self):
        testp = tdu.get_derivative_params(self.ECLmodelJ1713)
        delay = self.ECLmodelJ1713.delay(self.toasJ1713)
        for p in testp.keys():
            # Only check the binary parameters
            if p not in self.ECLmodelJ1713.binary_instance.binary_params:
                continue
            if p in ["PX", "PMRA", "PMDEC"]:
                continue
            par = getattr(self.ECLmodelJ1713, p)
            if isinstance(par, boolParameter):
                continue
            print(f"Runing derivative for d_phase_d_{p}")
            ndf = self.ECLmodelJ1713.d_phase_d_param_num(self.toasJ1713, p, testp[p])
            adf = self.ECLmodelJ1713.d_phase_d_param(self.toasJ1713, delay, p)
            diff = adf - ndf
            if np.all(diff.value) != 0.0:
                mean_der = (adf + ndf) / 2.0
                relative_diff = np.abs(diff) / np.abs(mean_der)
                # print "Diff Max is :", np.abs(diff).max()
                msg = (
                    "Derivative test failed at d_phase_d_%s with max relative difference %lf"
                    % (p, np.nanmax(relative_diff).value)
                )
                if p in ["SINI", "KIN"]:
                    tol = 0.7
                elif p in ["KOM"]:
                    tol = 0.04
                else:
                    tol = 1e-3
                print(
                    (
                        "derivative relative diff for %s, %lf"
                        % (f"d_phase_d_{p}", np.nanmax(relative_diff).value)
                    )
                )
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

    def test_k96(self):
        modelJ1713 = copy.deepcopy(self.ECLmodelJ1713)
        log = logging.getLogger("TestJ1713 Switch of K96")
        modelJ1713.K96.value = False
        res = Residuals(
            self.toasJ1713, modelJ1713, use_weighted_mean=False
        ).time_resids.to(u.s)
        delay = self.ECLmodelJ1713.delay(self.toasJ1713)
        testp = tdu.get_derivative_params(modelJ1713)
        for p in testp.keys():
            self.ECLmodelJ1713.d_phase_d_param(self.toasJ1713, delay, p)


def test_ddk_ECL_ICRS():
    mECL = get_model(StringIO(temp_par_str + "\n KIN  71.969  1               0.562"))
    print(
        f"Simulated TOAs in ECL with (KIN,KOM) = ({mECL.KIN.quantity},{mECL.KOM.quantity})"
    )
    tECL = pint.simulation.make_fake_toas_uniform(
        50000, 60000, 100, mECL, error=0.5 * u.us, add_noise=True
    )
    prefit = Residuals(tECL, mECL)
    print(f"Prefit chi^2 in ECL {prefit.calc_chi2()}")
    # get proper motion vector and normalize
    pm_ECL = np.array(
        [mECL.coords_as_ECL().pm_lon_coslat.value, mECL.coords_as_ECL().pm_lat.value]
    )
    pm_ECL /= np.sqrt(np.dot(pm_ECL, pm_ECL))

    mICRS = mECL.as_ICRS()
    pm_ICRS = np.array(
        [mICRS.coords_as_ICRS().pm_ra_cosdec.value, mICRS.coords_as_ICRS().pm_dec.value]
    )
    pm_ICRS /= np.sqrt(np.dot(pm_ICRS, pm_ICRS))
    # get the angle between proper motion vectors, which should be the difference between KOMs
    angle = np.arccos(np.dot(pm_ECL, pm_ICRS)) * u.rad

    new_KOM = mICRS.KOM.quantity
    mICRS.KOM.quantity = mECL.KOM.quantity
    rICRS = Residuals(tECL, mICRS)
    print(
        f"Prefit chi^2 in ICRS with the same KOM ({mICRS.KOM.quantity}) {rICRS.calc_chi2()}"
    )
    mICRS_newKOM = mECL.as_ICRS()
    rICRS_newKOM = Residuals(tECL, mICRS_newKOM)
    print(
        f"Change KOM by {angle.to(u.deg)} to {mICRS_newKOM.KOM.quantity}, now chi^2 in ICRS is {rICRS_newKOM.calc_chi2()}"
    )
    # fitting with the wrong KOM should be bad
    assert rICRS.calc_chi2() - prefit.calc_chi2() > 10
    # and with the new KOM they should be close
    assert np.isclose(rICRS_newKOM.calc_chi2(), prefit.calc_chi2())

    # check that round-trip is OK
    mECL_transformed = mICRS_newKOM.as_ECL()
    assert np.isclose(mECL_transformed.KOM.quantity, mECL.KOM.quantity)


def test_stand_alone_model_params_updates():
    test_par_str = temp_par_str + "\n KIN  71.969  1  0.562"
    m = mb.get_model(StringIO(test_par_str))
    # Check if KIN exists in the pint facing object and stand alone binary
    # models.
    assert hasattr(m.binary_instance, "KIN")
    assert hasattr(m, "KIN")
    verify_stand_alone_binary_parameter_updates(m)


def test_zero_PX():
    zero_px_str = temp_par_str.replace("PX  0.8211", "PX  0.0")
    with pytest.raises(ValueError):
        mb.get_model(StringIO(zero_px_str))


def test_remove_PX():
    test_par_str = temp_par_str + "\n KIN  71.969  1  0.562"
    m = mb.get_model(StringIO(test_par_str))
    m.remove_param("PX")
    with pytest.raises(MissingParameter):
        m.validate()


def test_A1dot_warning():
    # should not have a warning
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Using A1DOT")
        m = mb.get_model(StringIO(temp_par_str))

    # and this should have a warning
    with pytest.warns(
        UserWarning, match=r"Using A1DOT with a DDK model is not advised."
    ):
        m = mb.get_model(StringIO(temp_par_str + "\nA1DOT 2\n"))


def test_alternative_solutions():
    mECL = get_model(StringIO(temp_par_str + "\n KIN  71.969  1               0.562"))
    assert len(mECL.components["BinaryDDK"].alternative_solutions()) == 4
