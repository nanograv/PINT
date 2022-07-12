#!/usr/bin/env python
import os
import sys
import unittest
from io import StringIO

import numpy as np
from pinttestdata import datadir, testdir

import pint.scripts.zima as zima


def test_result(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = "{0} {1}".format(parfile, timfile)
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout
