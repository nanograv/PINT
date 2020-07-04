""" Test for pint design matrix"""
import os
import pytest

from pint.models import get_model
from pint.toa import get_TOAs
from pint.pint_matrix import  DesignMatrix, PhaseDesignMatrix
import astropy.units as u
from pinttestdata import datadir


class TestDesignMatrix:

    def setup(self):
        os.chdir(datadir)
        self.par_file = "J1614-2230_NANOGrav_12yv3.wb.gls.par"
        self.tim_file = "J1614-2230_NANOGrav_12yv3.wb.tim"
        self.model = get_model(self.par_file)
        self.toas = get_TOAs(self.tim_file)
        self.default_test_param = []
        for p in self.model.params:
            if not getattr(self.model, p).frozen:
                self.default_test_param.append(p)

    def test_make_phase_designmatrix(self):
        phase_designmatrix = PhaseDesignMatrix(self.toas, self.model,
                                               self.default_test_param)

    # def test_make_dm_designmatrix(self):
    #     quantity = 'dm'
    #     phase_designmatrix = DesignMatrix(self.toas, self.model, quantity,
    #                                       u.Unit(''), self.default_test_param)
