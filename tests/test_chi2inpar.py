import pytest
from pint.models import get_model_and_toas, get_model
import pint.fitter
from pinttestdata import datadir
import os
import io


@pytest.mark.parametrize(
    "ft",
    [
        pint.fitter.WLSFitter,
        pint.fitter.GLSFitter,
        pint.fitter.DownhillWLSFitter,
        pint.fitter.DownhillGLSFitter,
    ],
)
def test_nbfitters(ft):
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )
    f = ft(t, m)
    f.fit_toas()
    matched_line = [
        line for line in f.model.as_parfile().split("\n") if line.startswith("CHI2")
    ]
    assert matched_line
    assert float(matched_line[0].split()[-1]) > 0
    m2 = get_model(io.StringIO(f.model.as_parfile()))


@pytest.mark.parametrize(
    "ft", [pint.fitter.WidebandTOAFitter, pint.fitter.WidebandDownhillFitter]
)
def test_wbfitters(ft):
    m, t = get_model_and_toas(
        "tests/datafile/B1855+09_NANOGrav_12yv3.wb.gls.par",
        "tests/datafile/B1855+09_NANOGrav_12yv3.wb.tim",
    )
    f = ft(t, m)
    f.fit_toas()
    matched_line = [
        line for line in f.model.as_parfile().split("\n") if line.startswith("CHI2")
    ]
    assert matched_line
    assert float(matched_line[0].split()[-1]) > 0
    m2 = get_model(io.StringIO(f.model.as_parfile()))
