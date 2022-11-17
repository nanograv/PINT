#!/usr/bin/env python
import os
import sys

from io import StringIO

import pint.scripts.zima as zima
from pinttestdata import datadir


def test_zima(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    output_timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = "{0} {1}".format(parfile, output_timfile)
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_zima_inputtim(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    input_timfile = os.path.join(datadir, "NGC6440E.tim")
    output_timfile = tmp_path / "fake_testzima.tim"

    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"--inputtim {input_timfile} {parfile} {output_timfile}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_zima_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    parfile = os.path.join(datadir, "NGC6440E.par")
    output_timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = "--plot {0} {1}".format(parfile, output_timfile)
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout
