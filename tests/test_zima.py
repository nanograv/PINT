#!/usr/bin/env python
import os
import sys
import unittest
from io import StringIO

import numpy as np
from pinttestdata import datadir, testdir

import pint.scripts.zima as zima

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
