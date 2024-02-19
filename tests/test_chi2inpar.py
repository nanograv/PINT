from astropy import units as u
import pytest
from pint.models import get_model_and_toas, get_model
import pint.fitter
import pint.simulation
from pinttestdata import datadir
import os
import io


@pytest.fixture
def wb():
    m = get_model(os.path.join(datadir, "NGC6440E.par"))
    t = pint.simulation.make_fake_toas_uniform(
        55000, 58000, 20, model=m, freq=1400 * u.MHz, wideband=True, add_noise=True
    )

    return m, t


@pytest.fixture
def nb():
    m = get_model(os.path.join(datadir, "NGC6440E.par"))
    t = pint.simulation.make_fake_toas_uniform(
        55000,
        58000,
        20,
        model=m,
        freq=1400 * u.MHz,
        wideband=False,
        add_noise=True,
    )

    return m, t


@pytest.mark.parametrize(
    "ft",
    [
        pint.fitter.WLSFitter,
        pint.fitter.GLSFitter,
        pint.fitter.DownhillWLSFitter,
        pint.fitter.DownhillGLSFitter,
    ],
)
def test_nbfitters(ft, nb):
    m, t = nb
    f = ft(t, m)
    f.fit_toas()
    for p in ["CHI2", "CHI2R", "TRES"]:
        matched_line = [
            line for line in f.model.as_parfile().split("\n") if line.startswith(p)
        ]
        assert matched_line
        assert float(matched_line[0].split()[-1]) > 0
    assert not [
        line for line in f.model.as_parfile().split("\n") if line.startswith("DMRES")
    ]
    m2 = get_model(io.StringIO(f.model.as_parfile()))


@pytest.mark.parametrize(
    "ft", [pint.fitter.WidebandTOAFitter, pint.fitter.WidebandDownhillFitter]
)
def test_wbfitters(ft, wb):
    m, t = wb
    f = ft(t, m)
    f.fit_toas()
    for p in ["CHI2", "CHI2R", "TRES", "DMRES"]:
        matched_line = [
            line for line in f.model.as_parfile().split("\n") if line.startswith(p)
        ]
        assert matched_line
        assert float(matched_line[0].split()[-1]) > 0
    m2 = get_model(io.StringIO(f.model.as_parfile()))
