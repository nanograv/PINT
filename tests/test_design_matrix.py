""" Test for pint design matrix"""
import warnings

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

from pint.models import get_model
from pint.pint_matrix import (
    DesignMatrixMaker,
    combine_design_matrices_by_param,
    combine_design_matrices_by_quantity,
)
from pint.toa import get_TOAs


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()
    s.par_file = datadir / "J1614-2230_NANOGrav_12yv3.wb.gls.par"
    s.tim_file = datadir / "J1614-2230_NANOGrav_12yv3.wb.tim"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        s.model = get_model(s.par_file)
    s.toas = get_TOAs(s.tim_file, picklefilename=pickle_dir)
    s.default_test_param = []
    for p in s.model.params:
        if not getattr(s.model, p).frozen:
            s.default_test_param.append(p)
    s.test_param_lite = ["F0", "ELONG", "ELAT", "DMX_0023", "JUMP1", "DMJUMP2"]
    s.phase_designmatrix_maker = DesignMatrixMaker("phase", u.Unit(""))
    s.toa_designmatrix_maker = DesignMatrixMaker("toa", u.s)
    s.dm_designmatrix_maker = DesignMatrixMaker("dm", u.pc / u.cm**3)
    s.noise_designmatrix_maker = DesignMatrixMaker("toa_noise", u.s)
    return s


def test_make_phase_designmatrix(setup):
    phase_designmatrix = setup.phase_designmatrix_maker(
        setup.toas, setup.model, setup.test_param_lite
    )

    assert phase_designmatrix.ndim == 2
    assert phase_designmatrix.shape == (
        setup.toas.ntoas,
        len(setup.test_param_lite) + 1,
    )

    # Test labels
    labels = phase_designmatrix.labels
    assert len(labels) == 2
    assert len(labels[0]) == 1
    assert len(labels[1]) == len(setup.test_param_lite) + 1
    assert [l[0] for l in labels[1]] == ["Offset"] + setup.test_param_lite


def test_make_dm_designmatrix(setup):
    test_param = ["DMX_0001", "DMX_0010", "DMJUMP1"]
    phase_designmatrix = setup.dm_designmatrix_maker(
        setup.toas, setup.model, test_param
    )


def test_combine_designmatrix_quantity(setup):
    phase_designmatrix = setup.phase_designmatrix_maker(
        setup.toas, setup.model, setup.test_param_lite
    )
    dm_designmatrix = setup.dm_designmatrix_maker(
        setup.toas, setup.model, setup.test_param_lite, offset=True, offset_padding=0.0
    )

    combined = combine_design_matrices_by_quantity(
        [phase_designmatrix, dm_designmatrix]
    )
    # dim1 includes parameter lite and offset
    assert combined.shape == (2 * setup.toas.ntoas, len(setup.test_param_lite) + 1)
    assert len(combined.get_axis_labels(0)) == 2
    dim0_labels = [x[0] for x in combined.get_axis_labels(0)]
    assert dim0_labels == ["phase", "dm"]
    dim1_labels = [x[0] for x in combined.get_axis_labels(1)]
    assert dim1_labels == ["Offset"] + setup.test_param_lite


def test_toa_noise_designmatrix(setup, pickle_dir):
    toas = get_TOAs(datadir / "B1855+09_NANOGrav_9yv1.tim", picklefilename=pickle_dir)
    model = get_model(datadir / "B1855+09_NANOGrav_9yv1.gls.par")
    noise_designmatrix = setup.noise_designmatrix_maker(toas, model)
    assert noise_designmatrix.shape[0] == toas.ntoas
    assert noise_designmatrix.derivative_quantity == ["toa"]
    assert noise_designmatrix.derivative_params == ["toa_noise_params"]


def test_combine_designmatrix_all(setup, pickle_dir):
    toas = get_TOAs(
        datadir / "B1855+09_NANOGrav_12yv3.wb.tim", picklefilename=pickle_dir
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        model = get_model(datadir / "B1855+09_NANOGrav_12yv3.wb.gls.par")
    noise_designmatrix = setup.noise_designmatrix_maker(toas, model)

    toa_designmatrix = setup.toa_designmatrix_maker(toas, model, setup.test_param_lite)
    dm_designmatrix = setup.dm_designmatrix_maker(
        toas, model, setup.test_param_lite, offset=True, offset_padding=0.0
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
