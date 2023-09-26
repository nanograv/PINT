import os
import pytest
from io import StringIO

import pytest
from hypothesis import given
from hypothesis.strategies import sampled_from

from pint.models import get_model, get_model_and_toas
from pint import fitter
from pinttestdata import datadir


def test_SWM():
    """Should be present in PINT, not in TEMPO/TEMPO2"""

    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    assert (
        "SWM" in m.as_parfile()
        and "SWM" not in m.as_parfile(format="tempo")
        and "SWM" not in m.as_parfile(format="tempo2")
    )


def test_CHI2():
    """Should be present after fit in PINT, not in TEMPO/TEMPO2"""
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )

    f = fitter.WLSFitter(toas=t, model=m)
    f.fit_toas()
    assert "CHI2" in f.model.as_parfile()
    assert "CHI2" in f.model.as_parfile(format="tempo2")
    assert "CHI2" in f.model.as_parfile(format="tempo")


def test_T2CMETHOD():
    """Should be commented out in TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par"))
    for l in m.as_parfile().split("\n"):
        if "T2CMETHOD" in l:
            assert not (l.startswith("#"))
    for l in m.as_parfile(format="tempo").split("\n"):
        if "T2CMETHOD" in l:
            assert not (l.startswith("#"))
    for l in m.as_parfile(format="tempo2").split("\n"):
        if "T2CMETHOD" in l:
            assert l.startswith("#")


def test_MODE1():
    """Should start TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par"))
    assert (
        not (m.as_parfile(include_info=False).startswith("MODE 1"))
        and not (m.as_parfile(include_info=False, format="tempo").startswith("MODE 1"))
        and (m.as_parfile(include_info=False, format="tempo2").startswith("MODE 1"))
    )


def test_STIGMA():
    """Should get changed to VARSIGMA for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par"))
    assert (
        "STIGMA" in m.as_parfile()
        and "VARSIGMA" not in m.as_parfile()
        and "STIGMA" not in m.as_parfile(format="tempo")
        and "VARSIGMA" in m.as_parfile(format="tempo")
        and "STIGMA" not in m.as_parfile(format="tempo2")
        and "VARSIGMA" in m.as_parfile(format="tempo2")
    )


def test_A1DOT():
    """Should get changed to XDOT for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J1600-3053_test.par"))
    assert (
        "A1DOT" in m.as_parfile()
        and "XDOT" not in m.as_parfile()
        and "A1DOT" not in m.as_parfile(format="tempo")
        and "XDOT" in m.as_parfile(format="tempo")
        and "A1DOT" not in m.as_parfile(format="tempo2")
        and "XDOT" in m.as_parfile(format="tempo2")
    )


def test_ECL():
    """Should be only IERS2003 for TEMPO2"""
    m = get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1.gls.par"))
    for l in m.as_parfile().split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2010"
    for l in m.as_parfile(format="tempo").split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2010"
    for l in m.as_parfile(format="tempo2").split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2003"


def test_DMDATA_N():
    """Should be an integer for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J0030+0451_post.par"))
    for l in m.as_parfile(format="tempo").split("\n"):
        if "DMDATA" in l:
            # this should be a 0
            dmdata = int(l.split()[-1])
            assert dmdata == 0


def test_DMDATA_Y():
    """Should be an integer for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_12yv3.wb.gls.par"))
    for l in m.as_parfile(format="tempo").split("\n"):
        if "DMDATA" in l:
            # this should be a  1
            dmdata = int(l.split()[-1])
            assert dmdata == 1


def test_formats():
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    with pytest.raises(ValueError):
        s = m.as_parfile(format="nottempo")


def test_EFAC():
    """Should become T2EFAC in TEMPO/TEMPO2"""
    model = get_model(
        StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EFAC mjd 57000 58000 2
        """
        )
    )
    assert any(
        l.startswith("EFAC") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("EFAC") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert not any(
        l.startswith("EFAC") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="tempo2").split("\n")
    )


def test_EQUAD():
    """Should become T2EQUAD in TEMPO/TEMPO2"""
    model = get_model(
        StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EQUAD mjd 57000 58000 2
        """
        )
    )
    assert any(
        l.startswith("EQUAD") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("EQUAD") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert not any(
        l.startswith("EQUAD") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="tempo2").split("\n")
    )


def test_DM001_vs_DM1_pint_tempo_read_write():
    """Ensure all parfile formats write out DMn the way it was read in. May want to change later."""
    model = get_model(
        StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        DM001 0 1 0
        DM2 0 1 0
        DM0010 0 1 0
        """
        )
    )
    assert "DM1" in model.params
    assert "DM10" in model.params
    assert any(
        l.startswith("DM0010") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("DM0010") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(
        l.startswith("DM0010") for l in model.as_parfile(format="pint").split("\n")
    )
    assert any(
        l.startswith("DM2") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("DM2") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(l.startswith("DM2") for l in model.as_parfile(format="pint").split("\n"))


def test_NHARMS():
    model = get_model(
        StringIO(
            """
        PSR                            J1802-2124
        EPHEM                               DE440
        CLOCK                        TT(BIPM2019)
        UNITS                                 TDB
        START              57682.9164254267206481
        FINISH             58945.5403619002087731
        DILATEFREQ                              N
        DMDATA                                  N
        ELONG                 270.486526147922007 1 0.00000012082526951556
        ELAT                    2.037373400788321 1 0.00000314425302204301
        PMELONG                -1.826189745727688 1 0.4777024218759009
        PMELAT                -2.9704295220972257 1 12.55798968405999
        PX                     0.5286751048529988 1 1.2501655070063251
        ECL                              IERS2010
        POSEPOCH           58314.0000000000000000
        F0                    79.0664240384491762 1 2.7971394286742147137e-12
        F1               -4.55647355155766584e-16 1 2.5152536793477603086e-19
        PEPOCH             58314.0000000000000000
        CORRECT_TROPOSPHERE                         Y
        PLANET_SHAPIRO                          Y
        NE_SW                                 0.0
        SWM                                   0.0
        DM                  149.60232712555309087
        DM1                                   0.0
        DMEPOCH            58314.0000000000000000
        BINARY ELL1H
        PB                  0.6988892433285496519 1 2.306950233730408576e-11
        A1                      3.718865580010016 1 5.10696441177697e-07
        TASC               58314.1068677136653058 1 1.02392354114256444625e-08
        EPS1             4.117234830314865184e-06 1 2.8610710701285065444e-07
        EPS2            1.9600104585898805536e-06 1 2.9783944064651076693e-07
        H3              2.4037642640660087724e-06 1 3.31127323135470378e-07
        H4              1.7027113002634860964e-06 1 2.88248121675800648e-07
        NHARMS                                  7.0
        FD1                2.2042642599772845e-05 1 1.2254082126867467e-06
        TZRMJD             58133.7623761140993056
        TZRSITE                                GB
        TZRFRQ                           1455.314
        """
        )
    )
    for line in model.as_parfile(format="tempo").split("\n"):
        if line.startswith("NHARMS"):
            d = line.split()
            # it should be an integer
            assert "." not in d[-1]


sensible_pars = [
    "B1855+09_NANOGrav_9yv1.gls.par",
    "NGC6440E.par",
    "B1855+09_NANOGrav_dfg+12_TAI.par",
    "B1855+09_NANOGrav_dfg+12_TAI.par",
    "J0613-0200_NANOGrav_9yv1.gls.par",
]


@given(sampled_from(sensible_pars), sampled_from(["pint", "tempo", "tempo2"]))
def test_roundtrip(par, format):
    m = get_model(os.path.join(datadir, par))
    m2 = get_model(StringIO(m.as_parfile(format=format)))
    # FIXME: check some things aren't changed
    # for p in m.params:
    #    assert getattr(m, p).value == getattr(m2, p).value
