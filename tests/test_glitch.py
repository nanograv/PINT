from io import StringIO
import os
import pytest

import astropy.units as u
import numpy as np
import pytest

from pint.exceptions import MissingParameter
import pint.fitter
from pint.models import get_model
import pint.residuals
import pint.toa
from pinttestdata import datadir

parfile = os.path.join(datadir, "prefixtest.par")
timfile = os.path.join(datadir, "prefixtest.tim")

basepar = """
    PSRJ           J0835-4510
    RAJ            08:35:20.61149
    DECJ           -45:10:34.8751
    F0             11.18965156782
    PEPOCH         55305                       
    DM             67.99
    UNITS          TDB
"""

good = """
    GLEP_1         55555
    GLPH_1         0
    GLF0_1         1.0e-6
    GLF1_1         -1.0e-12
    GLF0D_1        1.0e-6
    GLTD_1         10.0
"""

# no exponential decay set
bad1 = """
    GLEP_1         55555
    GLF0_1         1.0e-6
    GLF1_1         -1.0e-12
    GLF0D_1        1.0e-6
    GLTD_1         0.0
"""

# no epoch set
bad2 = """
    GLPH_1         0
    GLF0_1         1.0e-6
    GLF1_1         -1.0e-12
"""

# fitting both epoch and glitch phase
bad3 = """
    GLEP_1         55555 1 0
    GLPH_1         0 1 0
    GLF0_1         1.0e-6
    GLF1_1         -1.0e-12
"""


class TestGlitch:
    @classmethod
    def setup_class(cls):
        cls.m = get_model(parfile)
        cls.t = pint.toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)
        cls.f = pint.fitter.WLSFitter(cls.t, cls.m)

    def test_glitch(self):
        print("Test prefix parameter via a glitch model")
        rs = pint.residuals.Residuals(self.t, self.m).phase_resids
        # Now do the fit
        print("Fitting...")
        self.f.fit_toas()
        emsg = f"RMS of {self.m.PSR.value} is too big."
        assert self.f.resids.time_resids.std().to(u.us).value < 950.0, emsg

    @pytest.mark.filterwarnings("ignore:invalid value")
    def test_glitch_der(self):
        delay = self.m.delay(self.t)
        for pf in self.m.glitch_prop:
            for idx in set(self.m.glitch_indices):
                if pf in ["GLF0D_", "GLTD_"]:
                    getattr(self.m, f"GLF0D_{idx}").value = 1.0
                    getattr(self.m, f"GLTD_{idx}").value = 100
                else:
                    getattr(self.m, f"GLF0D_{idx}").value = 0.0
                    getattr(self.m, f"GLTD_{idx}").value = 0.0
                param = pf + str(idx)
                adf = self.m.d_phase_d_param(self.t, delay, param)
                param_obj = getattr(self.m, param)
                # Get numerical derivative steps.
                h = 1e-8 if param_obj.units == u.day else 1e-2
                ndf = self.m.d_phase_d_param_num(self.t, param, h)
                diff = adf - ndf
                mean = (adf + ndf) / 2.0
                r_diff = diff / mean
                errormsg = f"Derivatives for {param} is not accurate, max relative difference is"
                errormsg += " %lf" % np.nanmax(np.abs(r_diff.value))
                assert np.nanmax(np.abs(r_diff.value)) < 1e-3, errormsg

    def test_bad_input(self):
        get_model(StringIO(basepar + good))
        with pytest.raises(MissingParameter):
            get_model(StringIO(basepar + bad1))
        with pytest.raises(MissingParameter):
            m = get_model(StringIO(basepar + bad2))
            print(m)
        with pytest.raises(ValueError):
            get_model(StringIO(basepar + bad3))
