""" Various of tests on the wideband DM data
"""

import os
import numpy as np
import pytest

from pint.models import get_model
from pint.toa import get_TOAs
from pinttestdata import datadir


os.chdir(datadir)


class TestDMData:
    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")

    def test_data_reading(self):
        dm_data_raw, valid = self.toas.get_flag_value("pp_dm")
        # For this input, the DM number should be the same with the TOA number.
        dm_data = np.array(dm_data_raw)[valid]
        assert len(valid) == self.toas.ntoas
        assert len(dm_data) == self.toas.ntoas
        assert dm_data.mean != 0.0

    def test_dm_modelcomponent(self):
        assert "DispersionJump" in self.model.components.keys()
        assert "ScaleDmError" in self.model.components.keys()

    def test_dm_jumps(self):
        pass

    def test_dm_noise(self):
        pass
