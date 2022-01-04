import os
import unittest
import pytest
from io import StringIO

from pint.models import get_model, get_model_and_toas
from pint import fitter
from pinttestdata import datadir


def test_SWM():
    """Should be present in PINT, not in TEMPO/TEMPO2"""

    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    assert (
        ("SWM" in m.as_parfile())
        and not ("SWM" in m.as_parfile(format="tempo"))
        and not ("SWM" in m.as_parfile(format="tempo2"))
    )


def test_CHI2():
    """Should be present after fit in PINT, not in TEMPO/TEMPO2"""
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )

    f = fitter.WLSFitter(toas=t, model=m)
    assert "CHI2" in f.model.as_parfile()
    assert not ("CHI2" in f.model.as_parfile(format="tempo2"))
    assert not ("CHI2" in f.model.as_parfile(format="tempo"))


def test_T2CMETHOD():
    """Should be commented out in TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par"))
    for l in m.as_parfile().split():
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
        ("STIGMA" in m.as_parfile())
        and not ("VARSIGMA" in m.as_parfile())
        and not ("STIGMA" in m.as_parfile(format="tempo"))
        and ("VARSIGMA" in m.as_parfile(format="tempo"))
        and not ("STIGMA" in m.as_parfile(format="tempo2"))
        and ("VARSIGMA" in m.as_parfile(format="tempo2"))
    )


def test_A1DOT():
    """Should get changed to XDOT for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J1600-3053_test.par"))
    assert (
        ("A1DOT" in m.as_parfile())
        and not ("XDOT" in m.as_parfile())
        and not ("A1DOT" in m.as_parfile(format="tempo"))
        and ("XDOT" in m.as_parfile(format="tempo"))
        and not ("A1DOT" in m.as_parfile(format="tempo2"))
        and ("XDOT" in m.as_parfile(format="tempo2"))
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


def test_formats():
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    with pytest.raises(AssertionError):
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
