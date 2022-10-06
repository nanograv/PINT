import os
import os.path
from io import StringIO

import astropy.units as u
import numpy as np
from pinttestdata import datadir

from pint.models import get_model
from pint.observatory import get_observatory
from pint.toa import get_TOAs

parfile = os.path.join(datadir, "NGC6440E.par")

tim = """FORMAT 1
fake 1400 58512.921979 1.0 mk -flag thing1
fake 1400 58514.619895 1.0 mk -flag thing2
"""


def test_get_TOAs():
    tt = get_TOAs(StringIO(tim), ephem="DE421")
    # Check the site clock correction by itself
    site = get_observatory("meerkat", include_gps=False, include_bipm=False)
    clock_corr = site.clock_corrections(tt.table["mjd"][0])
    assert np.isclose(clock_corr.to_value(u.us), 0.40802, atol=0.002)  # from mk2utc.clk
    # Now check barycentering
    mm = get_model(parfile)
    # The "correct" value is from PRESTO's bary command run like this:
    # > bary MK 17:48:52.75 -20:21:29.0 1400 223.9 DE421
    # 58512.921979
    # .
    # Note that the DM is important!
    bts = mm.get_barycentric_toas(tt)
    # This checks to see if they are the same to within 50us.
    assert np.fabs((bts[0].value - 58512.9185136816) * 86400.0) < 50e-6
