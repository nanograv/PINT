import io
import pint.models.model_builder as mb
from pint.scripts import compare_parfiles
import pytest


@pytest.fixture()
def model_ECL():
    parfile = """
        PSR              J1713+0747
        LAMBDA   256.6686952405695  1     0.0000000023351
        BETA      30.7003604937060  1     0.0000000041466
        PMLAMBDA            5.2671  1              0.0021
        PMBETA             -3.4428  1              0.0043
        PX                  0.8211  1              0.0258
        POSEPOCH        56553.0000
        F0    218.8118437960826270  1  0.0000000000000988
        F1     -4.083888637248D-16  1  1.433249826455D-21
        PEPOCH        55391.000000
        DM               15.917131
        EPHEM               DE421
        ECL                 IERS2003
        CLK                 TT(BIPM2015)
        UNITS               TDB
        TIMEEPH             FB90
    """
    return mb.get_model(io.StringIO(parfile))


@pytest.fixture()
def model_ICRS():
    parfile = """
        PSR                            J1713+0747
        MODE                                    1
        EPHEM                               DE421
        CLK                          TT(BIPM2015)
        UNITS                                 TDB
        TIMEEPH                              FB90
        RAJ                     17:13:49.53355031 1 0.00000057727246830927
        DECJ                     7:47:37.48847685 1 0.00001419450415241618
        PMRA                    4.925763856775273 1 0.002489055022551827
        PMDEC                 -3.9156180453775624 1 0.004087126752953659
        PX                                 0.8211 1 0.0258
        POSEPOCH           55391.0000000000000000
        F0                    218.811843796082627 1 9.88e-14
        F1                    -4.083888637248e-16 1 1.433249826455e-21
        PEPOCH             55391.0000000000000000
    """
    return mb.get_model(io.StringIO(parfile))


def test_model_compare(model_ECL, model_ICRS):
    comparison1 = model_ECL.compare(model_ICRS)
    comparison2 = model_ICRS.compare(model_ECL)

    assert isinstance(comparison1, str) and isinstance(comparison2, str)


def test_model_compare_convert_coord_y(model_ECL, model_ICRS):
    comparison1y = model_ECL.compare(model_ICRS, convertcoordinates=True)
    comparison2y = model_ICRS.compare(model_ECL, convertcoordinates=True)
    assert isinstance(comparison1y, str) and isinstance(comparison2y, str)


def test_model_compare_convert_coord_n(model_ECL, model_ICRS):
    comparison1n = model_ECL.compare(model_ICRS, convertcoordinates=False)
    comparison2n = model_ICRS.compare(model_ECL, convertcoordinates=False)
    assert isinstance(comparison1n, str) and isinstance(comparison2n, str)


def test_compare_parfile_script(model_ECL, model_ICRS):
    parfile1 = "par_a.par"
    parfile2 = "par_b.par"

    with open(parfile1, "w") as par1:
        par1.write(str(model_ECL))

    with open(parfile2, "w") as par2:
        par2.write(str(model_ICRS))

    argv = f"{parfile1} {parfile2}".split()
    compare_parfiles.main(argv)

    argv = f"--convertcoordinates {parfile1} {parfile2}".split()
    compare_parfiles.main(argv)
