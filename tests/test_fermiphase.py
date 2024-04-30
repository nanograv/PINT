import os
import sys
from io import StringIO
from pathlib import Path

import pytest

import numpy as np

from astropy.io import fits
from astropy import units as u

import pint.models
import pint.scripts.fermiphase as fermiphase
from pint.fermi_toas import get_Fermi_TOAs, _default_uncertainty
from pint.observatory.satellite_obs import get_satellite_observatory
from pinttestdata import datadir

parfile = datadir / "PSRJ0030+0451_psrcat.par"
eventfile = (
    datadir / "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
)

eventfileraw = datadir / "J0030+0451_w323_ft1weights.fits"
ft2file = datadir / "lat_spacecraft_weekly_w323_p202_v001.fits"


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_GEO_Htest_result(tmp_path):
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    p = Path.cwd()
    try:
        os.chdir(tmp_path)
        cmd = "--maxMJD 55000 --plot --plotfile fermitest.png --outfile fermitest.fits {0} {1} CALC".format(
            eventfile, parfile
        )
        fermiphase.main(cmd.split())
        lines = sys.stdout.getvalue()
        v = 0
        for l in lines.split("\n"):
            if l.startswith("Htest"):
                v = float(l.split()[2])
        # Check that H-test falls in right range for this portion of data
        assert (v > 550) and (v < 600)
    finally:
        sys.stdout = saved_stdout
        os.chdir(p)


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_process_and_accuracy():
    # Checks that instantiating an observatory and phasing of
    # topocentric times works.  Verifies accuracy to the sub-mus
    # level by comparison with stored Tempo2 "Fermi plugin" results.

    modelin = pint.models.get_model(parfile)
    get_satellite_observatory("Fermi", ft2file)
    ts = get_Fermi_TOAs(
        eventfileraw,
        weightcolumn="PSRJ0030+0451",
        include_gps=False,
        include_bipm=False,
        planets=False,
        ephem="DE405",
    )
    iphss, phss = modelin.phase(ts, abs_phase=True)
    ph_pint = phss % 1

    with fits.open(eventfileraw) as f:
        ph_tempo2 = f[1].data.field("pulse_phase")

    dphi = ph_pint - ph_tempo2
    dphi[dphi < -0.1] += 1
    dphi[dphi > 0.1] -= 1
    resids_mus = dphi / modelin.F0.value * 1e6
    # if the "2 mus" problem exists, the scatter in these values will
    # be 4-5 mus, whereas if everything is OK it should be few 100 ns
    # require range in TOAs to be less than 200ns
    assert (resids_mus.max() - resids_mus.min()) < 0.2
    # require absolute phase to be within 500 ns; NB this relies on
    # GBT clock corrections since the TZR is referenced there
    assert max(abs(resids_mus)) < 0.5


def test_for_toa_errors_default():
    get_satellite_observatory("Fermi", ft2file, overwrite=True)
    ts = get_Fermi_TOAs(
        eventfileraw,
        weightcolumn="PSRJ0030+0451",
        include_gps=False,
        include_bipm=False,
        planets=False,
        ephem="DE405",
    )
    assert np.all(ts.get_errors() == _default_uncertainty)


@pytest.mark.parametrize("errors", [2, 2 * u.us])
def test_for_toa_errors_manual(errors):
    get_satellite_observatory("Fermi", ft2file, overwrite=True)
    ts = get_Fermi_TOAs(
        eventfileraw,
        weightcolumn="PSRJ0030+0451",
        include_gps=False,
        include_bipm=False,
        planets=False,
        ephem="DE405",
        errors=errors,
    )
    assert np.all(ts.get_errors() == 2 * u.us)
