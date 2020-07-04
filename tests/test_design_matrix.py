""" Test for pint design matrix"""

import pytest

from pint.models import get_model
from pint.toa import get_TOAs
from pint.pint_matrix import  DesignMatrix
import astropy.units as u


class TestDesignMatrix:

    def setup(self):
        self.par_file = "J1614-2230_NANOGrav_12yv3.wb.gls.par"
        self.tim_file = "J1614-2230_NANOGrav_12yv3.wb.tim"
        self.model = get_model(self.par_file)
        self.toas = get_TOAs(self.tim_file)
        self.default_test_param = []
        for p in self.model.params:
            if not getattr(self.model, p).frozen:
                self.default_test_param.append(p)

    def test_make_phase_designmatrix(self):
        quantity = 'phase'
        phase_designmatrix = DesignMatrix(self.toas, self.model, quantity,
                                          u.Unit(''), self.default_test_param)

    def test_make_dm_designmatrix(self):
        quantity = 'dm'
        phase_designmatrix = DesignMatrix(self.toas, self.model, quantity,
                                          u.Unit(''), self.default_test_param)
