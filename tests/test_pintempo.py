#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys

from six import StringIO

from pint.scripts import pintempo
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")


def test_result():
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    cmd = "{0} {1}".format(parfile, timfile)
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
