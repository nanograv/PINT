import os
import pytest

import astropy.units as u

import pint.models
import pint.residuals
import pint.toa
import pint.fitter
from pinttestdata import datadir

# Not included in the test here, but as a sanity check I used this same
# ephemeris to phase up Fermi data, and it looks good.

parfile = os.path.join(datadir, "j0007_ifunc.par")
timfile = os.path.join(datadir, "j0007_ifunc.tim")


class TestIFunc:
    @classmethod
    def setup_class(cls):
        cls.m = pint.models.get_model(parfile)
        cls.t = pint.toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)

    def test_j0007(self):
        print("Test RMS of a PSR J0007+7303 ephemeris with IFUNCs(2).")
        self.m.SIFUNC.quantity = 2
        rs = pint.residuals.Residuals(self.t, self.m)
        rms = rs.time_resids.to(u.us).std()
        chi2 = rs.reduced_chi2
        emsg = f"RMS of {str(rms.value)} is too big."
        assert rms < 2700.0 * u.us, emsg
        emsg = f"reduced chi^2 of {str(chi2)} is too big."
        assert chi2 < 1.1, emsg

        # test a fit
        f = pint.fitter.WLSFitter(self.t, self.m)
        f.fit_toas()
        rs = f.resids
        rms = rs.time_resids.to(u.us).std()
        chi2 = rs.reduced_chi2
        emsg = f"RMS of {str(rms.value)} is too big."
        assert rms < 2700.0 * u.us, emsg
        emsg = f"reduced chi^2 of {str(chi2)} is too big."
        assert chi2 < 1.1, emsg

        # the residuals are actually terrible when using linear interpolation,
        # so this test just makes sure there are no access errors
        print("Test RMS of a PSR J0007+7303 ephemeris with IFUNCs(0).")
        self.m.SIFUNC.quantity = 0
        rs = pint.residuals.Residuals(self.t, self.m)
