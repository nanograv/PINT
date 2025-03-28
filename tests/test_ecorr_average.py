import os

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.models.model_builder as mb
from pint import toa
from pint.fitter import GLSFitter


# This function can be used to recreate the data
# for this test if needed.
def _gen_data(par, tim):
    t = toa.get_TOAs(tim, ephem="DE436")
    m = mb.get_model(par)
    gls = GLSFitter(t, m)
    gls.fit_toas()

    mjds = t.get_mjds().to(u.d).value
    freqs = t.get_freqs().to(u.MHz).value
    res = gls.resids.time_resids.to(u.us).value
    err = m.scaled_sigma(t).to(u.us).value
    info = t.get_flag_value("f")

    with open(f"{par}.resids", "w") as fout:
        with open(f"{par}.info", "w") as iout:
            for i in range(t.ntoas):
                line = "%.10f %.4f %+.8e %.3e 0.0 %s" % (
                    mjds[i],
                    freqs[i],
                    res[i],
                    err[i],
                    info[i],
                )
                fout.write(line + "\n")
                iout.write(info[i] + "\n")

    # Requires res_avg in path
    cmd = f"cat {par}.resids | res_avg -r -t0.0001 -E{par} -i{par}.info > {par}.resavg"
    print(cmd)
    # os.system(cmd)


@pytest.mark.skip(reason="Slow and also xfail")
@pytest.mark.xfail(reason="PINT has a more modern position for Arecibo than TEMPO2")
def test_ecorr_average():
    par = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par")
    tim = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.tim")
    m = mb.get_model(par)
    t = toa.get_TOAs(tim, ephem="DE436")
    f = GLSFitter(t, m)
    # Get comparison resids and uncertainties
    mjd, freq, res, err, ophase, chi2, info = np.genfromtxt(
        f"{par}.resavg", unpack=True
    )
    resavg_mjd = mjd * u.d
    # resavg_freq = freq * u.MHz
    resavg_res = res * u.us
    resavg_err = err * u.us
    # resavg_chi2 = chi2

    f.fit_toas()
    avg = f.resids.ecorr_average()
    # The comparison data always come out time-sorted
    # so we need to sort here.
    ii = np.argsort(avg["mjds"])
    mjd_diff = avg["mjds"][ii] - resavg_mjd
    res_diff = avg["time_resids"][ii] - resavg_res
    err_ratio = avg["errors"][ii] / resavg_err
    assert np.abs(mjd_diff).max() < 1e-9 * u.d
    assert np.abs(res_diff).max() < 7 * u.ns
    assert np.abs(err_ratio - 1.0).max() < 5e-4
