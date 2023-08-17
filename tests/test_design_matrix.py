""" Test for pint design matrix"""
import os
import numpy as np

from pint.models import get_model
from pint.toa import get_TOAs
from pint.pint_matrix import (
    DesignMatrixMaker,
    combine_design_matrices_by_quantity,
    combine_design_matrices_by_param,
)
import astropy.units as u
from pinttestdata import datadir


class TestDesignMatrix:
    def setup_method(self):
        os.chdir(datadir)
        self.par_file = "J1614-2230_NANOGrav_12yv3.wb.gls.par"
        self.tim_file = "J1614-2230_NANOGrav_12yv3.wb.tim"
        self.model = get_model(self.par_file)
        self.toas = get_TOAs(self.tim_file)
        self.default_test_param = []
        for p in self.model.params:
            if not getattr(self.model, p).frozen:
                self.default_test_param.append(p)
        self.test_param_lite = ["F0", "ELONG", "ELAT", "DMX_0023", "JUMP1", "DMJUMP2"]
        self.phase_designmatrix_maker = DesignMatrixMaker("phase", u.Unit(""))
        self.toa_designmatrix_maker = DesignMatrixMaker("toa", u.s)
        self.dm_designmatrix_maker = DesignMatrixMaker("dm", u.pc / u.cm**3)
        self.noise_designmatrix_maker = DesignMatrixMaker("toa_noise", u.s)

    def test_make_phase_designmatrix(self):
        phase_designmatrix = self.phase_designmatrix_maker(
            self.toas, self.model, self.test_param_lite
        )

        assert phase_designmatrix.ndim == 2
        assert phase_designmatrix.shape == (
            self.toas.ntoas,
            len(self.test_param_lite) + 1,
        )

        # Test labels
        labels = phase_designmatrix.labels
        assert len(labels) == 2
        assert len(labels[0]) == 1
        assert len(labels[1]) == len(self.test_param_lite) + 1
        assert [l[0] for l in labels[1]] == ["Offset"] + self.test_param_lite

    def test_make_dm_designmatrix(self):
        test_param = ["DMX_0001", "DMX_0010", "DMJUMP1"]
        phase_designmatrix = self.dm_designmatrix_maker(
            self.toas, self.model, test_param
        )

    def test_combine_designmatrix_quantity(self):
        phase_designmatrix = self.phase_designmatrix_maker(
            self.toas, self.model, self.test_param_lite
        )
        dm_designmatrix = self.dm_designmatrix_maker(
            self.toas, self.model, self.test_param_lite, offset=True, offset_padding=0.0
        )

        combined = combine_design_matrices_by_quantity(
            [phase_designmatrix, dm_designmatrix]
        )
        # dim1 includes parameter lite and offset
        assert combined.shape == (2 * self.toas.ntoas, len(self.test_param_lite) + 1)
        assert len(combined.get_axis_labels(0)) == 2
        dim0_labels = [x[0] for x in combined.get_axis_labels(0)]
        assert dim0_labels == ["phase", "dm"]
        dim1_labels = [x[0] for x in combined.get_axis_labels(1)]
        assert dim1_labels == ["Offset"] + self.test_param_lite

    def test_toa_noise_designmatrix(self):
        toas = get_TOAs("B1855+09_NANOGrav_9yv1.tim")
        model = get_model("B1855+09_NANOGrav_9yv1.gls.par")
        noise_designmatrix = self.noise_designmatrix_maker(toas, model)
        assert noise_designmatrix.shape[0] == toas.ntoas
        assert noise_designmatrix.derivative_quantity == ["toa"]
        assert noise_designmatrix.derivative_params == ["toa_noise_params"]

    def test_combine_designmatrix_all(self):
        toas = get_TOAs("B1855+09_NANOGrav_12yv3.wb.tim")
        model = get_model("B1855+09_NANOGrav_12yv3.wb.gls.par")
        noise_designmatrix = self.noise_designmatrix_maker(toas, model)

        toa_designmatrix = self.toa_designmatrix_maker(
            toas, model, self.test_param_lite
        )
        dm_designmatrix = self.dm_designmatrix_maker(
            toas, model, self.test_param_lite, offset=True, offset_padding=0.0
        )
        combined_quantity = combine_design_matrices_by_quantity(
            [toa_designmatrix, dm_designmatrix]
        )
        combined_param = combine_design_matrices_by_param(
            combined_quantity, noise_designmatrix
        )

        assert combined_param.shape == (
            toa_designmatrix.shape[0] + dm_designmatrix.shape[0],
            toa_designmatrix.shape[1] + noise_designmatrix.shape[1],
        )

        assert np.all(
            combined_param.matrix[
                toas.ntoas : toas.ntoas * 2, toa_designmatrix.shape[1] : :
            ]
            == 0.0
        )

    def test_param_order(self):
        params_dm = self.model.designmatrix(self.toas, incoffset=False)[1]
        params_free = self.model.free_params
        assert params_dm == params_free
