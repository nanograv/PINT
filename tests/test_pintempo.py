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
    for l in lines.split("\n"):
        if l.startswith("RMS in time is"):
            v = float(l.split()[4])
    # Check that RMS is less than 34 microseconds
    from astropy import log

    log.warning("%f" % v)
    assert v < 34.0
    sys.stdout = saved_stdout
