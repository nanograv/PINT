""" Various of tests for the general data fitter using wideband TOAs.
"""

import pytest
import os

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import GeneralDataFitter
from pinttestdata import datadir

os.chdir(datadir)

class TestGeneralDataFitter:

    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")
        self.fit_data_name = ['phase', 'dm']
        self.fit_params_lite = ['F0', 'F1', 'ELONG', 'ELAT', 'DMJUMP1',
                                'DMX_0022']

    def test_fitter_init(self):
        fitter = GeneralDataFitter(self.model, self.fit_data_name,
                                   [self.toas,], additional_args={})

        # test making residuals
        assert len(fitter.resids.resids) == 2 * self.toas.ntoas
        # test additional args
        add_args = {}
        add_args['phase'] = {'subtract_mean': False}
        fitter2= GeneralDataFitter(self.model, self.fit_data_name,
                                   [self.toas,], add_args)
        assert fitter2.resids.residual_objs[0].subtract_mean == False

    def test_fitter_designmatrix(self):
        fitter = GeneralDataFitter(self.model, self.fit_data_name,
                                   [self.toas,], additional_args={})
        fitter.set_fitparams(self.fit_params_lite)
        assert set(fitter.get_fitparams()) == set(self.fit_params_lite)
        # test making design matrix
        d_matrix = fitter.get_designmatrix()
        assert d_matrix.shape == (2 * self.toas.ntoas,
                                 len(self.fit_params_lite) + 1)
        assert [lb[0] for lb in d_matrix.labels[0]] == ['phase', 'dm']
        assert d_matrix.derivative_params == (['Offset'] +
            list(fitter.get_fitparams().keys()))

    def test_fitting(self):
        fitter = GeneralDataFitter(self.model, self.fit_data_name,
                                   [self.toas,], additional_args={})
        fitter.fit_toas()
