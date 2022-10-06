"""Test basic functionality of the :module:`pint.plot_utils`."""
import numpy as np
import astropy.units as u
from astropy.time import Time
from pint.plot_utils import phaseogram, phaseogram_binned
import tempfile
import os


def test_phaseogram():
    mjds_float = np.arange(58000, 58500, 1.0)
    mjds_quantity = mjds_float * u.d
    mjds_time = Time(mjds_quantity, format="mjd", scale="tdb")

    phases = np.random.rand(len(mjds_float))

    with tempfile.TemporaryDirectory() as tmpdir:
        outf = os.path.join(tmpdir, "plotfile.png")

        # Test that phaseogram can be called with each supported input type without exceptions
        phaseogram(mjds_float, phases, plotfile=outf)
        phaseogram(mjds_quantity, phases, plotfile=outf)
        phaseogram(mjds_time, phases, plotfile=outf)

        # Test that phaseogram_binned can be called with each supported input type without exceptions
        phaseogram_binned(mjds_float, phases, plotfile=outf)
        phaseogram_binned(mjds_quantity, phases, plotfile=outf)
        phaseogram_binned(mjds_time, phases, plotfile=outf)
