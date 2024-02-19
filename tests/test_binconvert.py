from astropy import units as u, constants as c
import astropy.time
import numpy as np
import io
import os
import copy
import pytest

from pint.models import get_model
from pint import derived_quantities
import pint.simulation
import pint.fitter
import pint.binaryconvert

parDD = """
PSRJ           1855+09
RAJ             18:57:36.3932884         0  0.00002602730280675029
DECJ           +09:43:17.29196           0  0.00078789485676919773
F0             186.49408156698235146     0  0.00000000000698911818
F1             -6.2049547277487420583e-16 0  1.7380934373573401505e-20
PEPOCH         49453
POSEPOCH       49453
DMEPOCH        49453
DM             13.29709
PMRA           -2.5054345161030380639    0  0.03104958261053317181
PMDEC          -5.4974558631993817232    0  0.06348008663748286318
PX             1.2288569063263405232     0  0.21243361289239687251
T0             49452.940695077335647     0  0.00169031830532837251
OM             276.55142180589701234     0  0.04936551005019605698
ECC            0.1 0  0.00000004027191312623
START          53358.726464889485214
FINISH         55108.922917417192366
TZRMJD         54177.508359343262555
TZRFRQ         424
TZRSITE        ao
TRES           0.395
EPHVER         5
CLK            TT(TAI)
MODE 1
UNITS          TDB
T2CMETHOD      TEMPO
#NE_SW          0.000
CORRECT_TROPOSPHERE  N
EPHEM          DE405
NITS           1
NTOA           702
CHI2R          2.1896 637
SOLARN0        00.00
TIMEEPH        FB90
PLANET_SHAPIRO N
EDOT       2e-10 1 2e-12
"""
Mp = 1.4 * u.Msun
Mc = 1.1 * u.Msun
i = 85 * u.deg
PB = 0.5 * u.day
A1 = derived_quantities.a1sini(Mp, Mc, PB, i)

parELL1 = """PSR              B1855+09
LAMBDA   286.8634874826803  1     0.0000000103957
BETA      32.3214851782886  1     0.0000000165796
PMLAMBDA           -3.2697  1              0.0079
PMBETA             -5.0683  1              0.0154
PX                  0.7135  1              0.1221
POSEPOCH        55637.0000
F0    186.4940812354533364  1  0.0000000000087885
F1     -6.204846776906D-16  1  4.557200069514D-20
PEPOCH        55637.000000
START            53358.726
FINISH           57915.276
DM               13.313704
OLARN0               0.00
EPHEM               DE436
ECL                 IERS2010
CLK                 TT(BIPM2017)                    
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      N
DILATEFREQ          N
NTOA                   313
TRES                  2.44
TZRMJD  55638.45920097834544
TZRFRQ            1389.800
TZRSITE                  AO
MODE                     1
NITS 1
DMDATA                   1
INFO -f                              
BINARY            ELL1    
A1             9.230780257  1         0.000000172
PB       12.32717119177539  1    0.00000000014613
TASC       55631.710921347  1         0.000000017
EPS1         -0.0000215334  1        0.0000000194
EPS2          0.0000024177  1        0.0000000127
SINI              0.999185  1            0.000190
M2                0.246769  1            0.009532
EPS1DOT           1e-10 1 1e-11
EPS2DOT           -1e-10 1 1e-11
"""

parELL1FB0 = """PSR              B1855+09
LAMBDA   286.8634874826803  1     0.0000000103957
BETA      32.3214851782886  1     0.0000000165796
PMLAMBDA           -3.2697  1              0.0079
PMBETA             -5.0683  1              0.0154
PX                  0.7135  1              0.1221
POSEPOCH        55637.0000
F0    186.4940812354533364  1  0.0000000000087885
F1     -6.204846776906D-16  1  4.557200069514D-20
PEPOCH        55637.000000
START            53358.726
FINISH           57915.276
DM               13.313704
OLARN0               0.00
EPHEM               DE436
ECL                 IERS2010
CLK                 TT(BIPM2017)                    
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      N
DILATEFREQ          N
NTOA                   313
TRES                  2.44
TZRMJD  55638.45920097834544
TZRFRQ            1389.800
TZRSITE                  AO
MODE                     1
NITS 1
DMDATA                   1
INFO -f                              
BINARY            ELL1    
A1             9.230780257  1         0.000000172
#PB       12.32717119177539  1    0.00000000014613
FB0   9.389075477264583e-07 1  1.1130092850564776e-17
TASC       55631.710921347  1         0.000000017
EPS1         -0.0000215334  1        0.0000000194
EPS2          0.0000024177  1        0.0000000127
SINI              0.999185  1            0.000190
M2                0.246769  1            0.009532
EPS1DOT           1e-10 1 1e-11
EPS2DOT           -1e-10 1 1e-11
"""

kwargs = {"ELL1H": {"NHARMS": 3, "useSTIGMA": True}, "DDK": {"KOM": 0 * u.deg}}


@pytest.mark.parametrize(
    "output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK", "DDH"]
)
def test_ELL1(output):
    m = get_model(io.StringIO(parELL1))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    assert mout.BINARY.value == output
    assert f"Binary{output}" in mout.components


@pytest.mark.parametrize(
    "output1", ["ELL1", "ELL1H", "ELL1k", "DD", "DDS", "DDK", "DDH"]
)
@pytest.mark.parametrize(
    "output2", ["ELL1", "ELL1H", "ELL1k", "DD", "DDS", "DDK", "DDH"]
)
def test_matrix(output1, output2):
    m = get_model(io.StringIO(parELL1))
    mout = pint.binaryconvert.convert_binary(m, output1, **kwargs.get(output1, {}))
    mout2 = pint.binaryconvert.convert_binary(mout, output2, **kwargs.get(output2, {}))


@pytest.mark.parametrize(
    "output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK", "DDH"]
)
def test_ELL1_roundtrip(output):
    m = get_model(io.StringIO(parELL1))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    mback = pint.binaryconvert.convert_binary(mout, "ELL1")
    for p in m.params:
        if output == "BT" and p in ["M2", "SINI"]:
            # these are not in BT
            continue
        if getattr(m, p).value is None:
            continue
        if not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time)):
            assert np.isclose(
                getattr(m, p).value, getattr(mback, p).value
            ), f"{p}: {getattr(m, p).value} does not match {getattr(mback, p).value}"
            if getattr(m, p).uncertainty is not None:
                # some precision may be lost in uncertainty conversion
                assert np.isclose(
                    getattr(m, p).uncertainty_value,
                    getattr(mback, p).uncertainty_value,
                    rtol=0.2,
                ), f"{p} uncertainty: {getattr(m, p).uncertainty_value} does not match {getattr(mback, p).uncertainty_value}"
        else:
            assert (
                getattr(m, p).value == getattr(mback, p).value
            ), f"{p}: {getattr(m, p).value} does not match {getattr(mback, p).value}"


@pytest.mark.parametrize("output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK"])
def test_ELL1FB0(output):
    m = get_model(io.StringIO(parELL1FB0))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    assert mout.BINARY.value == output
    assert f"Binary{output}" in mout.components


@pytest.mark.parametrize("output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK"])
def test_ELL1_roundtripFB0(output):
    m = get_model(io.StringIO(parELL1FB0))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    mback = pint.binaryconvert.convert_binary(mout, "ELL1")
    for p in m.params:
        if output == "BT" and p in ["M2", "SINI"]:
            # these are not in BT
            continue
        if getattr(m, p).value is None:
            continue
        if not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time)):
            assert np.isclose(
                getattr(m, p).value, getattr(mback, p).value
            ), f"{p}: {getattr(m, p).value} does not match {getattr(mback, p).value}"
            if getattr(m, p).uncertainty is not None:
                # some precision may be lost in uncertainty conversion
                assert np.isclose(
                    getattr(m, p).uncertainty_value,
                    getattr(mback, p).uncertainty_value,
                    rtol=0.2,
                ), f"{p} uncertainty: {getattr(m, p).uncertainty_value} does not match {getattr(mback, p).uncertainty_value}"
        else:
            assert (
                getattr(m, p).value == getattr(mback, p).value
            ), f"{p}: {getattr(m, p).value} does not match {getattr(mback, p).value}"


@pytest.mark.parametrize(
    "output", ["ELL1", "ELL1k", "ELL1H", "DD", "BT", "DDS", "DDH", "DDK"]
)
def test_DD(output):
    m = get_model(
        io.StringIO(
            f"{parDD}\nBINARY DD\nSINI {np.sin(i).value}\nA1 {A1.value}\nPB {PB.value}\nM2 {Mc.value}\n"
        )
    )
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    assert mout.BINARY.value == output
    assert f"Binary{output}" in mout.components


@pytest.mark.parametrize(
    "output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDH", "DDK"]
)
def test_DD_roundtrip(output):
    s = f"{parDD}\nBINARY DD\nSINI {np.sin(i).value} 1 0.01\nA1 {A1.value}\nPB {PB.value} 1 0.1\nM2 {Mc.value} 1 0.01\n"
    if output not in ["ELL1", "ELL1H"]:
        s += "OMDOT       1e-10 1 1e-12"

    m = get_model(io.StringIO(s))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    mback = pint.binaryconvert.convert_binary(mout, "DD")
    for p in m.params:
        if output == "BT" and p in ["M2", "SINI"]:
            # these are not in BT
            continue
        if getattr(m, p).value is None:
            continue
        # print(getattr(m, p), getattr(mback, p))
        if not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time)):
            assert np.isclose(getattr(m, p).value, getattr(mback, p).value)
            if getattr(m, p).uncertainty is not None:
                # some precision may be lost in uncertainty conversion
                if output in ["ELL1", "ELL1H", "ELL1k"] and p in ["ECC"]:
                    # we lose precision on ECC since it also contains a contribution from OM now
                    continue
                if output in ["ELL1H", "DDH"] and p == "M2":
                    # this also loses precision
                    continue
                assert np.isclose(
                    getattr(m, p).uncertainty_value,
                    getattr(mback, p).uncertainty_value,
                    rtol=0.2,
                ), f"Parameter '{p}' failed: initial uncertainty {getattr(m, p).uncertainty_value} but returned {getattr(mback, p).uncertainty_value}"
        else:
            assert getattr(m, p).value == getattr(mback, p).value


@pytest.mark.parametrize("output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK"])
def test_DDGR(output):
    m = get_model(
        io.StringIO(
            f"{parDD}\nBINARY DDGR\nA1 {A1.value} 0 0.01\nPB {PB.value} 0 0.02\nM2 {Mc.value} \nMTOT {(Mp+Mc).value}\n"
        )
    )
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))


@pytest.mark.parametrize(
    "output",
    [
        "ELL1",
        "ELL1k",
        "ELL1H",
        "DD",
        "BT",
        "DDS",
        "DDK",
    ],
)
def test_DDFB0(output):
    m = get_model(
        io.StringIO(
            f"{parDD}\nBINARY DD\nSINI {np.sin(i).value}\nA1 {A1.value}\nFB0 {(1/PB).to_value(u.Hz)}\nM2 {Mc.value}\n"
        )
    )
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))


@pytest.mark.parametrize("output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK"])
def test_DDFB0_roundtrip(output):
    s = f"{parDD}\nBINARY DD\nSINI {np.sin(i).value} 1 0.01\nA1 {A1.value}\nFB0 {(1/PB).to_value(u.Hz)} 1 0.1\nM2 {Mc.value} 1 0.01\n"
    if output not in ["ELL1", "ELL1H"]:
        s += "OMDOT       1e-10 1 1e-12"

    m = get_model(io.StringIO(s))
    mout = pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))
    mback = pint.binaryconvert.convert_binary(mout, "DD")
    for p in m.params:
        if output == "BT" and p in ["M2", "SINI"]:
            # these are not in BT
            continue
        if getattr(m, p).value is None:
            continue
        # print(getattr(m, p), getattr(mback, p))
        if not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time)):
            assert np.isclose(getattr(m, p).value, getattr(mback, p).value)
            if getattr(m, p).uncertainty is not None:
                # some precision may be lost in uncertainty conversion
                if output in ["ELL1", "ELL1H", "ELL1k"] and p in ["ECC"]:
                    # we lose precision on ECC since it also contains a contribution from OM now
                    continue
                if output == "ELL1H" and p == "M2":
                    # this also loses precision
                    continue
                assert np.isclose(
                    getattr(m, p).uncertainty_value,
                    getattr(mback, p).uncertainty_value,
                    rtol=0.2,
                )
        else:
            assert getattr(m, p).value == getattr(mback, p).value
