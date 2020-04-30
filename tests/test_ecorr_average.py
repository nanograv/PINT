#! /usr/bin/env python
import json
import os
import unittest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
from pint import toa
from pint.fitter import GLSFitter
from pinttestdata import datadir

# This function can be used to recreate the data
# for this test if needed.
def _gen_data(par, tim):
    t = toa.get_TOAs(tim, ephem="DE436")
    m = mb.get_model(par)
    gls = GLSFitter(t, m)
    gls.fit_toas()

    mjds = t.get_mjds().to(u.d).value
    freqs = t.get_freqs().to(u.MHz).value
    res = gls.resids.time_resids.to(u.us).value
    err = m.scaled_sigma(t).to(u.us).value
    info = t.get_flag_value("f")

    fout = open(par + ".resids", "w")
    iout = open(par + ".info", "w")
    for i in range(t.ntoas):
        line = "%.10f %.4f %+.8e %.3e 0.0 %s" % (
            mjds[i],
            freqs[i],
            res[i],
            err[i],
            info[i],
        )
        fout.write(line + "\n")
        iout.write(info[i] + "\n")
    fout.close()
    iout.close()

    # Requires res_avg in path
    cmd = "cat %s.resids | res_avg -r -t0.0001 -E%s -i%s.info > %s.resavg" % (
        par,
        par,
        par,
        par,
    )
    print(cmd)
    # os.system(cmd)


class TestEcorrAverage(unittest.TestCase):
    """Compare epoch-averaging of residuals with that done by tempo's
    res_avg utility."""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.par = "J0023+0923_NANOGrav_11yv0.gls.par"
        cls.tim = "J0023+0923_NANOGrav_11yv0.tim"
        cls.m = mb.get_model(cls.par)
        cls.t = toa.get_TOAs(cls.tim, ephem="DE436")
        cls.f = GLSFitter(cls.t, cls.m)
        # Get comparison resids and uncertainties
        mjd, freq, res, err, ophase, chi2, info = np.genfromtxt(
            cls.par + ".resavg", unpack=True
        )
        cls.resavg_mjd = mjd * u.d
        cls.resavg_freq = freq * u.MHz
        cls.resavg_res = res * u.us
        cls.resavg_err = err * u.us
        cls.resavg_chi2 = chi2

    def test_ecorr_average(self):
        self.f.fit_toas()
        self.avg = self.f.resids.ecorr_average()
        # The comparison data always come out time-sorted
        # so we need to sort here.
        ii = np.argsort(self.avg["mjds"])
        self.mjd_diff = self.avg["mjds"][ii] - self.resavg_mjd
        self.res_diff = self.avg["time_resids"][ii] - self.resavg_res
        self.err_ratio = self.avg["errors"][ii] / self.resavg_err
        assert np.abs(self.mjd_diff).max() < 1e-9 * u.d
        assert np.abs(self.res_diff).max() < 5 * u.ns
        assert np.abs(self.err_ratio - 1.0).max() < 5e-4
