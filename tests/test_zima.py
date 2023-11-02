import os
import sys
from io import StringIO

import numpy as np
import pytest
from pinttestdata import datadir

import matplotlib
import pint.scripts.zima as zima
from pint.models import get_model_and_toas, get_model
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals
from pint.fitter import DownhillGLSFitter
from astropy import units as u


@pytest.mark.parametrize("addnoise", ["", "--addnoise"])
def test_result(tmp_path, addnoise):
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"{parfile} {timfile} {addnoise}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout

    model, toas = get_model_and_toas(parfile, timfile)
    res = Residuals(toas, model)
    redchisq = res.reduced_chi2

    if addnoise == "":
        assert np.isclose(redchisq, 0, atol=1e-2)
    else:
        assert redchisq > 0.5 and redchisq < 2


def test_result_inputtim(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    inputtim = os.path.join(datadir, "NGC6440E.tim")
    timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"{parfile} {timfile} --addnoise --inputtim {inputtim}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_wb_result_with_noise(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = tmp_path / "fake_testzima_wb.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"{parfile} {timfile} --addnoise --wideband --dmerror 1e-5"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout

    model, toas = get_model_and_toas(parfile, timfile)
    assert toas.is_wideband()

    res = Residuals(toas, model)
    redchiq = res.reduced_chi2
    assert redchiq < 2 and redchiq > 0.5


def test_zima_plot(tmp_path):
    matplotlib.use("Agg")

    parfile = os.path.join(datadir, "NGC6440E.par")
    output_timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"--plot {parfile} {output_timfile}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_zima_fuzzdays(tmp_path):
    matplotlib.use("Agg")

    parfile = os.path.join(datadir, "NGC6440E.par")
    output_timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"--fuzzdays 1 {parfile} {output_timfile}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_simulate_corrnoise(tmp_path):
    parfile = datadir / "B1855+09_NANOGrav_9yv1.gls.par"

    m = get_model(parfile)

    # Simulated TOAs won't have the correct flags for some of these to work.
    m.remove_component("ScaleToaError")
    m.remove_component("EcorrNoise")
    m.remove_component("DispersionDMX")
    m.remove_component("PhaseJump")
    m.remove_component("FD")
    m.PLANET_SHAPIRO.value = False

    t = make_fake_toas_uniform(
        m.START.value,
        m.FINISH.value,
        1000,
        m,
        add_noise=True,
        add_correlated_noise=True,
    )

    # Check if the created TOAs can be whitened using
    # the original timing model. This won't work if the
    # noise is not realized correctly.
    ftr = DownhillGLSFitter(t, m)
    ftr.fit_toas()
    rc = sum(ftr.resids.noise_resids.values())
    r = ftr.resids.time_resids
    rw = r - rc
    sigma = ftr.resids.get_data_error()

    # This should be independent and standard-normal distributed.
    x = (rw / sigma).to_value("")
    assert np.isclose(np.std(x), 1, atol=0.2)
    assert np.isclose(np.mean(x), 0, atol=0.01)


@pytest.mark.parametrize("multifreq", [True, False])
def test_simulate_uniform_multifreq(multifreq):
    parfile = os.path.join(datadir, "NGC6440E.par")
    m = get_model(parfile)

    ntoas = 100

    freqs = np.array([500, 1400]) * u.MHz
    t = make_fake_toas_uniform(
        50000,
        51000,
        ntoas,
        m,
        add_noise=True,
        freq=freqs,
        multi_freqs_in_epoch=multifreq,
    )
    assert len(t) == ntoas

    freqs = np.array([500, 750, 1400]) * u.MHz
    t = make_fake_toas_uniform(
        50000,
        51000,
        ntoas,
        m,
        add_noise=True,
        freq=freqs,
        multi_freqs_in_epoch=multifreq,
    )
    assert len(t) == ntoas
