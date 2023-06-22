import os
import pytest

import pytest
from io import StringIO

import pint.scripts.pintk as pintk
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
class TestPintk:
    def test_result(self):
        saved_stdout, pintk.sys.stdout = pintk.sys.stdout, StringIO("_")
        cmd = "--test {0} {1}".format(parfile, timfile)
        pintk.main(cmd.split())
        lines = pintk.sys.stdout.getvalue()
        pintk.sys.stdout = saved_stdout
