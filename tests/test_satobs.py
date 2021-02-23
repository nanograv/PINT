import os
import pytest

from astropy.time import Time
import numpy as np

from pint.observatory.satellite_obs import get_satellite_observatory
from pinttestdata import datadir


def test_good_calls():
    # ensure that valid entries are accepted
    ft2file = os.path.join(datadir, "lat_spacecraft_weekly_w323_p202_v001.fits")
    fermi_obs = get_satellite_observatory("Fermi", ft2file, overwrite=True)
    tt_mjd = fermi_obs.FT2["MJD_TT"]
    test_mjds = tt_mjd[:: len(tt_mjd) // 10] + (np.random.rand(1) * 4 - 2) / (60 * 24)
    # add an explicit entry 20s after the last FT2 point; should pass with
    # default settings
    test_mjds = np.append(test_mjds, tt_mjd[-1] + 20.0 / 86400)
    assert test_mjds[-1] > tt_mjd[-1]
    good_times = Time(test_mjds, format="mjd", scale="tt")
    # NB this also tests calls with a vector Time
    fermi_obs._check_bounds(good_times)


def test_bad_calls():
    # find an SAA passage, which will be internal
    ft2file = os.path.join(datadir, "lat_spacecraft_weekly_w323_p202_v001.fits")
    fermi_obs = get_satellite_observatory("Fermi", ft2file, overwrite=True)
    tt_mjd = fermi_obs.FT2["MJD_TT"]
    saa_idx = np.argmax((tt_mjd[1:] - tt_mjd[:-1]))
    bad_time_saa = Time(tt_mjd[saa_idx] + 3.0 / (60 * 24), format="mjd", scale="tt")
    # and test extrapolation from the ends
    # NB this also tests calls with a scalar Time
    bad_time_end = Time(tt_mjd[-1] + 3.0 / (60 * 24), format="mjd", scale="tt")
    bad_time_beg = Time(tt_mjd[0] - 3.0 / (60 * 24), format="mjd", scale="tt")
    with pytest.raises(ValueError):
        fermi_obs._check_bounds(bad_time_saa)
    with pytest.raises(ValueError):
        fermi_obs._check_bounds(bad_time_end)
    with pytest.raises(ValueError):
        fermi_obs._check_bounds(bad_time_beg)
