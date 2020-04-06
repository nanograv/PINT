#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
import unittest

import numpy as np
from six import StringIO

import pint.scripts.zima as zima
from pinttestdata import datadir, testdir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "fake_testzima.tim")


class TestZima(unittest.TestCase):
    def test_result(self):
        saved_stdout, sys.stdout = sys.stdout, StringIO("_")
        cmd = "{0} {1}".format(parfile, timfile)
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
        sys.stdout = saved_stdout


if __name__ == "__main__":
    unittest.main()
