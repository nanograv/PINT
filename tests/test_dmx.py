import logging
import os
import pytest
import io
import pytest

import astropy.units as u
import numpy as np

import pint.toa as toa
from pint import residuals
from pint.models import model_builder as mb
from pint.models import get_model
import pint.simulation
from pinttestdata import datadir


par = """
PSR J1234+5678
F0 1
DM 10
ELAT 10
ELONG 0
PEPOCH 54000
DMXR1_0001 54000
DMXR2_0001 54500
DMX_0001 1
"""
dmxpar = """
DMXR1_0002 54500
DMXR2_0002 55000
DMX_0002 2
DMXR1_0003 55500
DMXR2_0003 56000
DMX_0003 3
"""


class TestDMX:
    @classmethod
    def setup_class(cls):
        cls.parf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_DMX.par")
        cls.timf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")
        cls.DMXm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(cls.timf, ephem="DE405", include_bipm=False)

    def test_dmx(self):
        print("Testing DMX module.")
        rs = (
            residuals.Residuals(self.toas, self.DMXm, use_weighted_mean=False)
            .time_resids.to(u.s)
            .value
        )
        ltres, _ = np.genfromtxt(f"{self.parf}.tempo_test", unpack=True)
        resDiff = rs - ltres
        assert np.all(
            np.abs(resDiff) < 2e-8
        ), "PINT and tempo Residual difference is too big."

    def test_derivative(self):
        log = logging.getLogger("DMX.derivative_test")
        p = "DMX_0002"
        log.debug("Running derivative for %s", f"d_delay_d_{p}")
        ndf = self.DMXm.d_delay_d_param_num(self.toas, p)
        adf = self.DMXm.d_delay_d_param(self.toas, p)
        diff = adf - ndf
        if np.all(diff.value) != 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            tol = 0.7 if p in {"SINI"} else 1e-3
            log.debug(
                (
                    "derivative relative diff for %s, %lf"
                    % (f"d_delay_d_{p}", np.nanmax(relative_diff).value)
                )
            )
            assert np.nanmax(relative_diff) < tol, msg


def test_dmx_overlap():
    model = get_model(io.StringIO(par))
    toas = pint.simulation.make_fake_toas_uniform(
        54000, 56000, 100, model=model, obs="gbt"
    )
    dmx = model.components["DispersionDMX"].dmx_dm(toas)

    # add in new overlapping DMX
    model.add_DMX_range(54500, 55000, dmx=0, frozen=True)
    newdmx = model.components["DispersionDMX"].dmx_dm(toas)
    assert np.all(dmx == newdmx)


def test_multiple_dmxs():
    model = get_model(io.StringIO(par))
    toas = pint.simulation.make_fake_toas_uniform(
        54000, 56000, 100, model=model, obs="gbt"
    )
    dmxmodel = get_model(io.StringIO(par + dmxpar))
    indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], dmxs=[2, 3])
    assert np.all(np.array(indices) == np.array([2, 3]))
    assert np.all(
        model.components["DispersionDMX"].dmx_dm(toas)
        == dmxmodel.components["DispersionDMX"].dmx_dm(toas)
    )


def test_multiple_dmxs_broadcast_frozens():
    model = get_model(io.StringIO(par))
    indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], frozens=False)
    for index in indices:
        assert getattr(model, f"DMX_{index:04d}").frozen == False


def test_multiple_dmxs_broadcast_dmxs():
    model = get_model(io.StringIO(par))
    indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], dmxs=2)
    for index in indices:
        assert getattr(model, f"DMX_{index:04d}").value == 2


def test_multiple_dmxs_wrong_ends():
    model = get_model(io.StringIO(par))
    with pytest.raises(ValueError):
        indices = model.add_DMX_ranges([54500, 55500], [55000], dmxs=[2, 3])


def test_multiple_dmxs_wrong_starts():
    model = get_model(io.StringIO(par))
    with pytest.raises(ValueError):
        indices = model.add_DMX_ranges([54500], [55000, 56000], dmxs=[2, 3])


def test_multiple_dmxs_wrong_dmxs():
    model = get_model(io.StringIO(par))
    with pytest.raises(ValueError):
        indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], dmxs=[2, 3, 4])


def test_multiple_dmxs_wrong_frozens():
    model = get_model(io.StringIO(par))
    with pytest.raises(ValueError):
        indices = model.add_DMX_ranges(
            [54500, 55500], [55000, 56000], frozens=[True, False, False]
        )


def test_multiple_dmxs_explicit_indices():
    model = get_model(io.StringIO(par))
    indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], indices=[2, 3])
    assert np.all(np.array(indices) == np.array([2, 3]))


def test_multiple_dmxs_explicit_indices_duplicate():
    model = get_model(io.StringIO(par))
    with pytest.raises(ValueError):
        indices = model.add_DMX_ranges([54500, 55500], [55000, 56000], indices=[1, 3])
