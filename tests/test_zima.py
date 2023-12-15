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


def test_zima_multifreq(tmp_path):
    matplotlib.use("Agg")

    parfile = os.path.join(datadir, "NGC6440E.par")
    output_timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"--freq 1400 500 --multifreq {parfile} {output_timfile}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout
