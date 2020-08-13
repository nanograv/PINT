#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np


def test_str2longdouble():
    print("You are using numpy %s" % np.__version__)
    a = np.longdouble("0.12345678901234567890")
    b = np.float("0.12345678901234567890")
    # If numpy is converting to longdouble without loss of
    # precision these two numbers should be different
    assert not (a == b), (
        "numpy losing precision in str to longdouble conversion! "
        "This is fixed in 1.11.0 and later."
    )
