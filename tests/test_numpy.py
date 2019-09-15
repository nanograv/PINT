#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import unittest
import os
import numpy as np

from pint.utils import extended_precision


class TestNumpy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_str2extended_precision(self):
        print("You are using numpy %s" % np.__version__)
        a = extended_precision("0.12345678901234567890")
        b = np.float64("0.12345678901234567890")
        # If numpy is converting to extended_precision without loss of
        # precision these two numbers should be different
        assert not (a == b), (
            "numpy losing precision in str to extended_precision conversion! "
            "This is fixed in 1.11.0 and later."
        )
