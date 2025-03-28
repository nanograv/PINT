import numpy as np
from astropy.time import Time

# This test is to make sure that astropy is correctly including recent leap seconds.
# It should be updated whenever a new leap second occurs. Just add a new check.


def test_leapsec_2008_dec_31():
    # Test that 2008 Dec 31 leap second is correctly included
    t1 = Time("2008-12-31T23:59:00", scale="utc")
    t2 = Time("2009-01-01T00:00:00", scale="utc")
    assert np.isclose((t2 - t1).sec, 61.0)


def test_leapsec_2016_dec_31():
    # Test that 2016 Dec 31 leap second is correctly included
    t1 = Time("2016-12-31T23:59:00", scale="utc")
    t2 = Time("2017-01-01T00:00:00", scale="utc")
    assert np.isclose((t2 - t1).sec, 61.0)
