import os
import sys

from io import StringIO

from pint.scripts import pintempo
from pinttestdata import datadir
import matplotlib
import pytest

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")


@pytest.mark.parametrize("gls", ["", "--gls"])
def test_pintempo(gls):
    matplotlib.use("Agg")
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    cmd = f"{parfile} {timfile} --plot {gls} --plotfile _test_pintempo.pdf --outfile _test_pintempo.out"
    pintempo.main(cmd.split())
    lines = sys.stdout.getvalue()
    v = 999.0
    # This line is in the output:
    # Prefit residuals Wrms = 1090.580262221985 us, Postfit residuals Wrms = 21.182038051610704 us
    for l in lines.split("\n"):
        if l.startswith("Prefit residuals"):
            v = float(l.split()[-2])
    # Check that RMS is less than 30 microseconds
    assert v < 30.0
    sys.stdout = saved_stdout
