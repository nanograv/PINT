from pint.models import get_model_and_toas
from pint.residuals import Residuals, WidebandTOAResiduals
from pinttestdata import datadir
import pytest
import numpy as np


@pytest.fixture
def resids_nb():
    m, t = get_model_and_toas(
        f"{datadir}/J0613-0200_NANOGrav_9yv1.gls.par",
        f"{datadir}/J0613-0200_NANOGrav_9yv1.tim",
    )
    return Residuals(t, m)


@pytest.fixture
def resids_wb():
    m, t = get_model_and_toas(
        f"{datadir}/B1855+09_NANOGrav_12yv3.wb.gls.par",
        f"{datadir}/B1855+09_NANOGrav_12yv3.wb.tim",
    )
    return WidebandTOAResiduals(t, m)


def test_dtype_nb(resids_nb):
    m = resids_nb.model
    t = resids_nb.toas

    for x in [
        resids_nb.calc_time_resids(),
        resids_nb.calc_phase_resids(),
        resids_nb.get_data_error(),
        m.scaled_toa_uncertainty(t),
        m.designmatrix(t)[0],
        m.noise_model_designmatrix(t),
        m.full_designmatrix(t)[0],
        m.noise_model_basis_weight(t),
    ]:
        assert x.dtype is np.dtype("float")


def test_dtype_wb(resids_wb):
    m = resids_wb.model
    t = resids_wb.toas

    for x in [
        resids_wb.toa.calc_time_resids(),
        resids_wb.toa.calc_phase_resids(),
        resids_wb.dm.calc_resids(),
        resids_wb.calc_wideband_resids(),
        resids_wb.toa.get_data_error(),
        resids_wb.dm.get_data_error(),
        m.scaled_toa_uncertainty(t),
        m.scaled_dm_uncertainty(t),
        m.scaled_wideband_uncertainty(t),
        m.designmatrix(t)[0],
        m.dm_designmatrix(t)[0],
        m.noise_model_designmatrix(t),
        m.noise_model_dm_designmatrix(t),
        m.full_designmatrix(t)[0],
        m.full_wideband_designmatrix(t)[0],
        m.noise_model_basis_weight(t),
    ]:
        assert x.dtype is np.dtype("float")
