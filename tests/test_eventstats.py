#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from numpy.testing import assert_allclose, assert_equal

import pint.eventstats as es

def test_sig2sigma():
    sig = [1.0E-1, 1.0e-10, 1.0e-16, 1.0E-120]
    res = es.sig2sigma(sig)
    ans = [ 1.64485363,  6.46695109,  8.30478543, 23.36370742]
    assert_allclose(res,ans,atol=1.0e-7)
