""" Test for pint design matrix"""

import pytest

from pint.models import get_model
from pint.toa import get_TOAs
from pint.pint_matrix import  DesignMatrix


class TestDesignMatrix:

    def setup(self):
        self.par_file = "J1614-2230_NANOGrav_12yv3.wb.gls.par"
        self.tim_file = "J1614-2230_NANOGrav_12yv3.wb.tim"
        self.model = get_model(self.par_file)
        self.toas = get_TOAs(self.tim_file)

    def test_make_design_matrix(self):
        pass
