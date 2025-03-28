import numpy as np
from os.path import join

import astropy.units as u
import pytest
from pinttestdata import datadir

import pint
from pint.fitter import (
    DownhillGLSFitter,
    DownhillWLSFitter,
    GLSFitter,
    WLSFitter,
    WidebandDownhillFitter,
    WidebandTOAFitter,
)
from pint.models.model_builder import get_model
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform


@pytest.fixture
def wls():
    m = get_model(join(datadir, "NGC6440E.par"))
    t = get_TOAs(join(datadir, "NGC6440E.tim"), ephem="DE421")

    wls = WLSFitter(t, m)
    wls.fit_toas()

    return wls


@pytest.fixture
def wb():
    m = get_model(join(datadir, "NGC6440E.par"))
    t = make_fake_toas_uniform(
        55000, 58000, 20, model=m, freq=1400 * u.MHz, wideband=True
    )

    wb = WidebandTOAFitter(t, m)
    wb.fit_toas()

    return wb


def test_wls_covmatrix(wls):
    uncertainties = wls.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            wls.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)


def test_gls_covmatrix(wls):
    gls = GLSFitter(wls.toas, wls.model_init)
    gls.fit_toas()

    assert np.isclose(
        wls.parameter_covariance_matrix.matrix, gls.parameter_covariance_matrix.matrix
    ).all()

    uncertainties = gls.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            gls.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)


def test_downhillwls_covmatrix(wls):
    dh_wls = DownhillWLSFitter(wls.toas, wls.model_init)
    dh_wls.fit_toas()

    assert np.isclose(
        wls.parameter_covariance_matrix.matrix,
        dh_wls.parameter_covariance_matrix.matrix,
    ).all()

    uncertainties = dh_wls.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            dh_wls.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)


def test_downhillgls_covmatrix(wls):
    dh_gls = DownhillGLSFitter(wls.toas, wls.model_init)
    dh_gls.fit_toas()

    assert np.isclose(
        wls.parameter_covariance_matrix.matrix,
        dh_gls.parameter_covariance_matrix.matrix,
    ).all()

    uncertainties = dh_gls.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            dh_gls.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)


def test_wb_covmatrix(wb):
    uncertainties = wb.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            wb.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)


def test_downhillwb_covmatrix(wb):
    dwb = WidebandDownhillFitter(wb.toas, wb.model_init)
    dwb.fit_toas()

    assert np.isclose(
        wb.parameter_covariance_matrix.matrix, dwb.parameter_covariance_matrix.matrix
    ).all()

    uncertainties = dwb.model.get_params_dict("free", "uncertainty")
    for param in uncertainties:
        # uncertainty based on the diagonals of the covariance matrix
        matrix_uncertainty = np.sqrt(
            dwb.parameter_covariance_matrix.get_label_matrix([param]).matrix[0, 0]
        )
        assert np.isclose(uncertainties[param], matrix_uncertainty)
