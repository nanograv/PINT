import pytest
import tempfile

from pint.models import get_model
import pint.simulation as simulation


stigma_template = """
PSR              FAKE
LAMBDA   270
BETA       2
POSEPOCH        58133.0000
F0     79
F1     -4e-16
PEPOCH        58133.000000
DM              149
EPHEM               DE436
ECL                 IERS2010
CLK                 TT(BIPM2017)
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      N
DILATEFREQ          N
BINARY            ELL1H
A1             3.7
PB        0.69
TASC       58133
EPS1          0.000005
EPS2          0.000002
H3          0.000002
STIGMA  {}  1  0.01
"""


def parfile_name(tmpdir, contents):
    fh, fn = tempfile.mkstemp(dir=tmpdir)
    with open(fn, "wt") as f:
        f.write(contents)
    return fn


def test_stigma_zero(tmpdir):
    with pytest.raises(ValueError):
        get_model(parfile_name(tmpdir, stigma_template.format(0)))
    # if STIGMA is zero everything goes wrong.
    # with np.errstate(invalid="raise"):
    #    simulation.make_fake_toas_uniform(58000, 59000, 10, model=m)


def test_stigma_nonzero(tmpdir):
    m = get_model(parfile_name(tmpdir, stigma_template.format(0.5)))
    simulation.make_fake_toas_uniform(58000, 59000, 10, model=m)
