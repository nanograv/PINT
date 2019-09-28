#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import unittest
import os
import numpy as np



class TestNumpy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_str2longdouble(self):
        print("You are using numpy %s" % np.__version__)
        a = np.longdouble("0.12345678901234567890")
        b = np.float("0.12345678901234567890")
        # If numpy is converting to longdouble without loss of
        # precision these two numbers should be different
        assert not (a == b), (
            "numpy losing precision in str to longdouble conversion! "
            "This is fixed in 1.11.0 and later."
        )
