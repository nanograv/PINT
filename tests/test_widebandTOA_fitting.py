""" Various of tests for the general data fitter using wideband TOAs.
"""

import pytest
import os

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
from pinttestdata import datadir

os.chdir(datadir)

class TestWidebandTOAFitter:

    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")
        self.fit_data_name = ['toa', 'dm']
        self.fit_params_lite = ['F0', 'F1', 'ELONG', 'ELAT', 'DMJUMP1',
                                'DMX_0022']

    def test_fitter_init(self):
        fitter = WidebandTOAFitter(self.model, [self.toas,], additional_args={})

        # test making residuals
        assert len(fitter.resids.resids) == 2 * self.toas.ntoas
        # test additional args
        add_args = {}
        add_args['toa'] = {'subtract_mean': False}
        fitter2= WidebandTOAFitter(self.model, [self.toas,],
                                   additional_args=add_args)
        assert fitter2.resids.residual_objs[0].subtract_mean == False

    def test_fitter_designmatrix(self):
        fitter = WidebandTOAFitter(self.model, [self.toas,], additional_args={})
        fitter.set_fitparams(self.fit_params_lite)
        assert set(fitter.get_fitparams()) == set(self.fit_params_lite)
        # test making design matrix
        d_matrix = fitter.get_designmatrix()
        assert d_matrix.shape == (2 * self.toas.ntoas,
                                 len(self.fit_params_lite) + 1)
        assert [lb[0] for lb in d_matrix.labels[0]] == ['toa', 'dm']
        assert d_matrix.derivative_params == (['Offset'] +
            list(fitter.get_fitparams().keys()))

    def test_fitting_no_full_cov(self):
        fitter = WidebandTOAFitter(self.model, [self.toas,], additional_args={})
        rms_pre = fitter.resids_init.rms_weighted()
        fitter.fit_toas()
        assert fitter.resids.rms_weighted() - rms_pre < 1e-9

    def test_fitting_full_cov(self):
        fitter = WidebandTOAFitter(self.model, [self.toas,], additional_args={})
        rms_pre = fitter.resids_init.rms_weighted()
        fitter.fit_toas(full_cov=True)
        assert fitter.resids.rms_weighted() - rms_pre < 1e-9
