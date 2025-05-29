import os
import pytest
from tempfile import NamedTemporaryFile

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import numpy as np

from pint import plot_utils


def test_phaseogram():

    # just make up some data
    mjds = np.random.rand(1000) * 1000 + 58000
    ph_pulse = np.random.rand(500) * 0.1 + 0.4
    ph_offpulse = np.random.rand(500)
    we_pulse = np.random.rand(500) * 0.25 + 0.75
    we_offpulse = np.random.rand(500) * 0.25 + 0.25
    ph = np.append(ph_pulse, ph_offpulse)
    we = np.append(we_pulse, we_offpulse)

    with NamedTemporaryFile("w") as plotfile:
        # test with different time format inputs
        plot_utils.phaseogram(
            mjds, ph, weights=we, title="test", plotfile=plotfile.name
        )
        plot_utils.phaseogram(
            mjds * u.d, ph, weights=we, plotfile=plotfile.name, bins=200
        )
        plot_utils.phaseogram(
            Time(mjds * u.d, format="mjd"), ph, weights=we, plotfile=plotfile.name
        )
        plot_utils.phaseogram_binned(
            mjds, ph, weights=we, plotfile=plotfile.name, bins=20
        )
