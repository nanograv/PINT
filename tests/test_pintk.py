#!/usr/bin/env python
from __future__ import division, print_function

import os
import unittest

import numpy as np
import pytest
from six import StringIO

import pint.scripts.pintk as pintk
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
class TestPintk(unittest.TestCase):
    def test_result(self):
        saved_stdout, pintk.sys.stdout = pintk.sys.stdout, StringIO("_")
        cmd = "--test {0} {1}".format(parfile, timfile)
        pintk.main(cmd.split())
        lines = pintk.sys.stdout.getvalue()
        pintk.sys.stdout = saved_stdout


if __name__ == "__main__":
    unittest.main()
