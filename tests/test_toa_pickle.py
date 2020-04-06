#!/usr/bin/env python
import os
import unittest

from pint import toa
from pinttestdata import datadir


class TestTOAReader(unittest.TestCase):
    def setUp(self):
        os.chdir(datadir)
        # First, read the TOAs from the tim file.
        # This should also create the pickle file.
        try:
            os.remove("test1.tim.pickle.gz")
            os.remove("test1.tim.pickle")
        except OSError:
            pass
        tt = toa.get_TOAs("test1.tim", usepickle=False, include_bipm=False)
        self.numtoas = tt.ntoas
        del tt
        # Now read them from the pickle
        self.t = toa.get_TOAs("test1.tim", usepickle=True, include_bipm=False)

    def test_pickle(self):
        # Initially this just checks that the same number
        # of TOAs came out of the pickle as went in.
        assert self.t.ntoas == self.numtoas
