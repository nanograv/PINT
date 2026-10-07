import copy
import io
import json
from pathlib import Path

import astropy.time
import numpy as np
import pytest
from astropy import units as u

import pint.binaryconvert
import pint.fitter
import pint.simulation
from pint import derived_quantities
from pint.exceptions import InvalidModelParameters, TimingModelError
from pint.models import get_model
from pint.models.binary_ddr import fw10_orbit_decode

_LD_EPS = float(np.finfo(np.longdouble).eps)


def _ddr_rtol(tight, floor=2e-16):
    return max(float(tight), float(floor), 512.0 * _LD_EPS)


def _ddr_atol(tight, scale=1.0, floor=2e-16):
    mag = float(np.max(np.abs(np.asarray(scale, dtype=np.longdouble))))
    return max(float(tight), float(floor), 512.0 * _LD_EPS * (mag + 1.0))


def _assert_roundtrip_value(p, m, mback):
    """Compare one parameter after a binary-model roundtrip.

    Numeric parameters use ``np.isclose``.  MJD/``Time`` parameters cannot use
    exact ``==``: ``convert_binary`` propagates TASC↔T0 through
    ``uncertainties`` (float64), and platforms with IEEE binary128
    ``numpy.longdouble`` (e.g. Linux aarch64) expose sub-ns residuals that
    still matched under 80-bit x87 longdouble bit-equality by coincidence.
    Allow about one float64 ulp at the MJD magnitude.
    """
    q = getattr(m, p).quantity
    v0, v1 = getattr(m, p).value, getattr(mback, p).value
    if isinstance(q, (str, bool)):
        assert v0 == v1, f"{p}: {v0} does not match {v1}"
    elif isinstance(q, astropy.time.Time):
        a = np.longdouble(v0)
        b = np.longdouble(v1)
        atol = np.finfo(np.float64).eps * max(abs(float(a)), abs(float(b)), 1.0)
        assert np.isclose(a, b, rtol=0.0, atol=atol), f"{p}: {v0} does not match {v1}"
    else:
        assert np.isclose(v0, v1), f"{p}: {v0} does not match {v1}"


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
    pint.binaryconvert.convert_binary(mout, output2, **kwargs.get(output2, {}))


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
        _assert_roundtrip_value(p, m, mback)
        if (
            not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time))
            and getattr(m, p).uncertainty is not None
        ):
            # some precision may be lost in uncertainty conversion
            assert np.isclose(
                getattr(m, p).uncertainty_value,
                getattr(mback, p).uncertainty_value,
                rtol=0.2,
            ), f"{p} uncertainty: {getattr(m, p).uncertainty_value} does not match {getattr(mback, p).uncertainty_value}"


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
        _assert_roundtrip_value(p, m, mback)
        if (
            not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time))
            and getattr(m, p).uncertainty is not None
        ):
            # some precision may be lost in uncertainty conversion
            assert np.isclose(
                getattr(m, p).uncertainty_value,
                getattr(mback, p).uncertainty_value,
                rtol=0.2,
            ), f"{p} uncertainty: {getattr(m, p).uncertainty_value} does not match {getattr(mback, p).uncertainty_value}"


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
        _assert_roundtrip_value(p, m, mback)
        if (
            not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time))
            and getattr(m, p).uncertainty is not None
        ):
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


@pytest.mark.parametrize("output", ["ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDS", "DDK"])
def test_DDGR(output):
    m = get_model(
        io.StringIO(
            f"{parDD}\nBINARY DDGR\nA1 {A1.value} 0 0.01\nPB {PB.value} 0 0.02\nM2 {Mc.value} \nMTOT {(Mp+Mc).value}\n"
        )
    )
    pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))


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
    pint.binaryconvert.convert_binary(m, output, **kwargs.get(output, {}))


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
        _assert_roundtrip_value(p, m, mback)
        if (
            not isinstance(getattr(m, p).quantity, (str, bool, astropy.time.Time))
            and getattr(m, p).uncertainty is not None
        ):
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


def test_ELL1_ELL1H():
    m = get_model(io.StringIO(parELL1))
    mout = pint.binaryconvert.convert_binary(m, "ELL1H", useSTIGMA=True)
    assert "STIGMA" in mout.components["BinaryELL1H"].binary_instance.fit_params

    mout = pint.binaryconvert.convert_binary(m, "ELL1H", useSTIGMA=False)
    assert "H4" in mout.components["BinaryELL1H"].binary_instance.fit_params
    assert mout.NHARMS.value == 7

    mout = pint.binaryconvert.convert_binary(m, "ELL1H", useSTIGMA=False, NHARMS=3)
    assert "H4" not in mout.components["BinaryELL1H"].binary_instance.fit_params
    assert mout.NHARMS.value == 3


parELL1_ddr = """\
PSRJ J0000+0000
RAJ 00:00:00.0
DECJ +00:00:00.0
F0 200
PEPOCH 55000
POSEPOCH 55000
DM 10
PX 1.0
PMRA 5.0
PMDEC -3.0
BINARY ELL1
PB 2.0
A1 1.963
TASC 55000
EPS1 1.0e-6
EPS2 -2.0e-6
M2 0.2
SINI 0.8660254037844386
PBDOT 0
UNITS TDB
"""

parDD_ddr = """\
PSRJ J0000+0000
RAJ 00:00:00.0
DECJ +00:00:00.0
F0 200
PEPOCH 55000
POSEPOCH 55000
DM 10
PX 1.0
PMRA 5.0
PMDEC -3.0
BINARY DD
PB 2.0
A1 1.963
T0 55000.0005
OM 30
ECC 0.001
M2 0.2
SINI 0.8660254037844386
PBDOT 0
OMDOT 0
GAMMA 0
UNITS TDB
"""

parDDK_ddr = """\
PSRJ J0000+0000
RAJ 00:00:00.0
DECJ +00:00:00.0
F0 200
PEPOCH 55000
POSEPOCH 55000
DM 10
PX 1.0
PMRA 5.0
PMDEC -3.0
BINARY DDK
PB 2.0
A1 1.963
T0 54990
OM 30
ECC 0.001
M2 0.2
KIN 60
KOM 40
PBDOT 0
UNITS TDB
"""


def test_ell1_to_ddr_and_back_coordinates():
    m = get_model(io.StringIO(parELL1_ddr))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert ddr.BINARY.value == "DDR"
    assert not ddr.DDRPK.value
    assert ddr.DDRPBDOT.value == "absorb_gw"
    assert not ddr.DDRGEO.value
    assert not ddr.DDRKINE.value
    report = ddr.binary_conversion_report
    assert "gauge_transfer" in report["coordinates"]
    assert isinstance(report["pk"], list)
    back = pint.binaryconvert.convert_binary(ddr, "ELL1")
    assert back.BINARY.value == "ELL1"
    np.testing.assert_allclose(back.A1.value, m.A1.value, rtol=1e-8)
    np.testing.assert_allclose(back.EPS1.value, m.EPS1.value, rtol=1e-5)


def test_ell1_to_ddr_is_phenomenological_by_default():
    m = get_model(io.StringIO(parELL1_ddr))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert not ddr.DDRPK.value
    ddr2 = pint.binaryconvert.convert_binary(m, "DDR", ddrpk=True, ddrpbdot="kinematic")
    assert ddr2.DDRPK.value
    assert ddr2.DDRPBDOT.value == "kinematic"
    assert "replaced_by_ddrpk" in ddr2.binary_conversion_report["pk"]


def test_missing_px_disables_geometry_and_kinematics():
    m = get_model(io.StringIO(parELL1_ddr.replace("PX 1.0\n", "")))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert not ddr.DDRGEO.value
    assert not ddr.DDRKINE.value
    with pytest.raises(TimingModelError, match="PX"):
        pint.binaryconvert.convert_binary(m, "DDR", ddrgeo=True, KOM=30 * u.deg)


def test_dd_to_ddr_polar_map():
    m = get_model(io.StringIO(parDD_ddr))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert ddr.BINARY.value == "DDR"
    np.testing.assert_allclose(ddr.A1.value, m.A1.value)
    dd = pint.binaryconvert.convert_binary(ddr, "DD")
    assert dd.BINARY.value == "DD"
    assert "BinaryDDK" not in dd.components
    np.testing.assert_allclose(dd.ECC.value, m.ECC.value, rtol=1e-6)


def test_dd_edge_on_sini_maps_to_zero_cosi():
    source = get_model(
        io.StringIO(parDD_ddr.replace("SINI 0.8660254037844386", "SINI 1"))
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert converted.COSI.value == pytest.approx(0)


def test_ddk_preserves_kin_epoch_as_tgeo():
    m = get_model(io.StringIO(parDDK_ddr))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert ddr.TGEO.value == pytest.approx(m.T0.value)
    assert ddr.DDRGEO.value
    assert ddr.KOM.value == pytest.approx(m.KOM.value)
    assert ddr.binary_conversion_report["geometry"] == ["from_ddk"]
    total = ddr.components["BinaryDDR"]._kinematic_p_total()
    np.testing.assert_allclose(total, 0, rtol=0, atol=1e-18)
    assert ddr.PBDOT.value != 0
    assert "pbdot_total_preserved" in ddr.binary_conversion_report["secular"]


def test_ddr_to_dd_is_dd_not_ddk():
    from test_ddr import example_par

    m = get_model(
        io.StringIO(
            example_par(
                DDRPK="N",
                DDRPBDOT="absorb_gw",
                DDRGEO="N",
                DDRKINE="N",
                PBDOT="0",
                GGAMMA="0",
                OMDOT="0",
                PX=None,
                PMRA=None,
                PMDEC=None,
                KOM=None,
            )
        )
    )
    dd = pint.binaryconvert.convert_binary(m, "DD")
    assert dd.BINARY.value == "DD"
    assert "BinaryDD" in dd.components
    assert "BinaryDDK" not in dd.components


def test_ell1k_ddgr_refuse_and_fbx_converts_to_ddr():
    m = get_model(io.StringIO(parELL1_ddr))
    mk = pint.binaryconvert.convert_binary(m, "ELL1k")
    with pytest.raises(TimingModelError):
        pint.binaryconvert.convert_binary(mk, "DDR")
    mdot = get_model(io.StringIO(parELL1))
    with pytest.raises(TimingModelError, match="secular"):
        pint.binaryconvert.convert_binary(mdot, "DDR")
    mfb = get_model(
        io.StringIO(
            parELL1FB0.replace("EPS1DOT", "#EPS1DOT").replace("EPS2DOT", "#EPS2DOT")
        )
    )
    ddr = pint.binaryconvert.convert_binary(mfb, "DDR")
    assert ddr.binary_conversion_report["chart"] == "fbx"
    assert ddr.FB0.quantity == mfb.FB0.quantity
    assert isinstance(ddr.PB, pint.models.parameter.funcParameter)


@pytest.mark.parametrize("family", ["ELL1", "DD"])
def test_fbx_coefficients_are_shifted_and_round_trip(family):
    base = parELL1_ddr if family == "ELL1" else parDD_ddr
    text = base.replace(
        "PB 2.0",
        "FB0 5.787037037037037e-6\nFB1 -1e-20\nFB2 1e-28",
    ).replace("PBDOT 0", "")
    source = get_model(io.StringIO(text))
    ddr = pint.binaryconvert.convert_binary(source, "DDR")
    assert ddr.binary_conversion_report["chart"] == "fbx"
    for index in range(3):
        assert getattr(ddr, f"FB{index}").quantity is not None
    back = pint.binaryconvert.convert_binary(ddr, family)
    for index in range(3):
        np.testing.assert_allclose(
            getattr(back, f"FB{index}").value,
            getattr(source, f"FB{index}").value,
            rtol=3e-13,
            atol=1e-40,
        )


def test_fbx_high_order_coefficients_round_trip_without_truncation():
    lines = "\n".join(f"FB{j} {(-1) ** j}e-{20 + 8 * max(j - 1, 0)}" for j in range(13))
    text = parELL1_ddr.replace("PB 2.0", lines).replace("PBDOT 0", "")
    source = get_model(io.StringIO(text))
    ddr = pint.binaryconvert.convert_binary(source, "DDR")
    assert ddr.binary_conversion_report["chart"] == "fbx"
    mapping = ddr.components["BinaryDDR"]._fbx_mapping()
    assert list(mapping) == list(range(13))
    for index in range(13):
        np.testing.assert_allclose(
            getattr(ddr, f"FB{index}").value,
            getattr(source, f"FB{index}").value,
            rtol=0,
            atol=0,
        )
    back = pint.binaryconvert.convert_binary(ddr, "ELL1")
    for index in range(13):
        np.testing.assert_allclose(
            getattr(back, f"FB{index}").value,
            getattr(source, f"FB{index}").value,
            rtol=3e-13,
            atol=1e-40,
        )


def test_dd_fbx_epoch_root_requires_full_polynomial():
    text = (
        parDD_ddr.replace(
            "PB 2.0",
            "FB0 5.787037037037037e-6\nFB1 -1e-20\nFB2 1e-24",
        )
        .replace("PBDOT 0", "")
        .replace("PX 1.0", "PX 0")
    )
    source = get_model(io.StringIO(text))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    _chart, coefficients, _names = pint.binaryconvert._phase_chart(source)
    omega = np.mod(
        np.arctan2(converted.EPS1.value, converted.EPS2.value),
        2 * np.longdouble(np.pi),
    )
    wrong_delta = pint.binaryconvert._monotone_phase_root(
        coefficients[:2],
        -omega / (2 * np.longdouble(np.pi)),
        -omega / (2 * np.longdouble(np.pi) * coefficients[0]),
    )
    wrong = copy.deepcopy(converted)
    wrong.TASC.value = np.longdouble(source.T0.value) + wrong_delta / 86400
    wrong_coefficients = pint.binaryconvert._shift_phase_coeffs(
        coefficients, wrong_delta
    )
    for index, coefficient in enumerate(wrong_coefficients):
        getattr(wrong, f"FB{index}").quantity = coefficient * u.s ** (-(index + 1))

    toas = pint.simulation.make_fake_toas_uniform(54990, 55010, 80, source, obs="@")
    source_delay = source.components["BinaryDD"].binarymodel_delay(toas).to_value(u.s)
    exact_delay = (
        converted.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    wrong_delay = wrong.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    exact_error = np.max(np.abs(source_delay - exact_delay))
    wrong_error = np.max(np.abs(source_delay - wrong_delay))
    assert exact_error < 5e-12
    assert wrong_error > exact_error
    assert np.max(np.abs(wrong_delay - exact_delay)) > 5e-12


def test_total_pbdot_is_preserved_without_flag():
    m = get_model(io.StringIO(parELL1_ddr))
    ddr = pint.binaryconvert.convert_binary(
        m, "DDR", ddrpk=True, ddrpbdot="kinematic", ddrkine=True
    )
    assert ddr.DDRPBDOT.value == "kinematic"
    np.testing.assert_allclose(ddr.PBDOT.value, m.PBDOT.value, rtol=0, atol=1e-18)
    assert "pbdot_total_preserved" in ddr.binary_conversion_report["secular"]


def test_rescale_pb_preserves_rescaled_total_rate():
    source = get_model(io.StringIO(parELL1_ddr.replace("PBDOT 0", "PBDOT 1")))
    converted = pint.binaryconvert.convert_binary(
        source,
        "DDR",
        ddrpk=True,
        ddrpbdot="kinematic",
        rescale_pb=True,
    )
    scale = np.longdouble(converted.PB.value) / np.longdouble(source.PB.value)
    np.testing.assert_allclose(
        converted.PBDOT.value,
        scale * np.longdouble(source.PBDOT.value),
        rtol=2e-14,
        atol=1e-24,
    )
    assert (
        "pb_azimuthal_to_anomalistic"
        in converted.binary_conversion_report["orbital_law"]
    )
    assert "pbdot_total_preserved" in converted.binary_conversion_report["secular"]


def test_kom_none_and_zero_are_distinct():
    m = get_model(io.StringIO(parDD_ddr))
    without = pint.binaryconvert.convert_binary(m, "DDR", KOM=None)
    with_zero = pint.binaryconvert.convert_binary(m, "DDR", KOM=0 * u.deg)
    assert not without.DDRGEO.value
    assert with_zero.DDRGEO.value
    assert with_zero.KOM.value == 0


def test_ell1_gauge_shift_uses_longdouble_mjd():
    m = get_model(io.StringIO(parELL1_ddr))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    expected_s = (
        np.longdouble("1.5") * np.longdouble(m.A1.value) * np.longdouble(m.EPS1.value)
    )
    actual_s = (np.longdouble(ddr.TASC.value) - np.longdouble(m.TASC.value)) * 86400
    np.testing.assert_allclose(actual_s, expected_s, rtol=0, atol=2e-10)
    np.testing.assert_allclose(
        ddr.binary_conversion_report["constant_offset_s"],
        -expected_s,
        rtol=0,
        atol=1e-18,
    )


@pytest.mark.parametrize(("eps1", "missing_shift_ns"), [(1e-6, 0.21), (1e-4, 21.0)])
def test_ell1_gauge_shift_closes_periodic_delay_term(eps1, missing_shift_ns):
    text = (
        parELL1_ddr.replace("EPS1 1.0e-6", f"EPS1 {eps1}")
        .replace("EPS2 -2.0e-6", "EPS2 0")
        .replace("M2 0.2", "M2 0")
        .replace("SINI 0.8660254037844386", "SINI 0.5")
        .replace("PX 1.0", "PX 0")
    )
    source = get_model(io.StringIO(text))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    no_gauge = copy.deepcopy(converted)
    no_gauge.TASC.quantity = source.TASC.quantity
    toas = pint.simulation.make_fake_toas_uniform(54990, 55010, 80, source, obs="@")
    source_delay = source.components["BinaryELL1"].binarymodel_delay(toas).to_value(u.s)

    converted_delay = (
        converted.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    difference = source_delay - converted_delay
    difference -= np.mean(difference)
    # 5e-15 is reachable on IEEE quad; 80-bit x87 lands around 1e-14.
    bound = max(5e-15, 2e-14)
    assert np.max(np.abs(difference)) < bound

    no_gauge_delay = (
        no_gauge.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    negative_control = source_delay - no_gauge_delay
    negative_control -= np.mean(negative_control)
    assert np.max(np.abs(negative_control)) * 1e9 == pytest.approx(
        missing_shift_ns, rel=0.1
    )


def test_dd_secular_transfer_preserves_kappa_and_total_rate():
    text = (
        parDD_ddr.replace("PBDOT 0", "PBDOT 1e-12")
        .replace("OMDOT 0", "OMDOT 0.01")
        .replace("GAMMA 0", "GAMMA 2e-6")
        .replace("A1 1.963", "A1 1.963\nA1DOT 1e-12")
    )
    m = get_model(io.StringIO(text))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    delta_s = (np.longdouble(ddr.TASC.value) - np.longdouble(m.T0.value)) * 86400
    expected_p = (
        np.longdouble(m.PBDOT.value)
        / (1 - np.longdouble(m.PBDOT.value) * delta_s / m.PB.quantity.to_value(u.s))
        ** 2
    )
    np.testing.assert_allclose(ddr.PBDOT.value, expected_p, rtol=2e-12, atol=1e-24)
    source_kappa = (
        m.OMDOT.quantity.to_value(u.rad / u.s)
        * m.PB.quantity.to_value(u.s)
        / (2 * np.pi)
    )
    target_kappa = (
        ddr.OMDOT.quantity.to_value(u.rad / u.s)
        * ddr.PB.quantity.to_value(u.s)
        / (2 * np.pi)
    )
    np.testing.assert_allclose(target_kappa, source_kappa, rtol=2e-14)
    np.testing.assert_allclose(
        ddr.A1.value,
        m.A1.value + m.A1DOT.quantity.to_value(u.lsec / u.s) * delta_s,
        rtol=0,
        atol=2e-15,
    )
    assert "transferred" in ddr.binary_conversion_report["secular"]
    back = pint.binaryconvert.convert_binary(ddr, "DD")
    np.testing.assert_allclose(back.PBDOT.value, m.PBDOT.value, rtol=2e-12)
    np.testing.assert_allclose(back.OMDOT.value, m.OMDOT.value, rtol=2e-12)


def test_dd_secular_transfer_delay_difference():
    text = (
        parDD_ddr.replace("PBDOT 0", "PBDOT 1e-12")
        .replace("OMDOT 0", "OMDOT 0.01")
        .replace("GAMMA 0", "GAMMA 2e-6")
        .replace("A1 1.963", "A1 1.963\nA1DOT 1e-12")
        .replace("PX 1.0", "PX 0")
    )
    source = get_model(io.StringIO(text))
    target = pint.binaryconvert.convert_binary(source, "DDR")
    toas = pint.simulation.make_fake_toas_uniform(52000, 58000, 80, source, obs="@")
    source_delay = source.components["BinaryDD"].binarymodel_delay(toas).to_value(u.s)
    target_delay = target.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    np.testing.assert_allclose(source_delay, target_delay, rtol=0, atol=5e-12)


def test_dd_secular_transfer_matches_shared_oracle():
    fixture = json.loads(
        (Path(__file__).parent / "data" / "ddr_reference_fixtures.json").read_text()
    )["dd_secular_transfer"]
    text = (
        parDD_ddr.replace("PBDOT 0", "PBDOT 1e-12")
        .replace("OMDOT 0", "OMDOT 0.01")
        .replace("GAMMA 0", "GAMMA 2e-6")
        .replace("A1 1.963", "A1 1.963\nA1DOT 1e-12")
        .replace("PX 1.0", "PX 0")
    )
    converted = pint.binaryconvert.convert_binary(get_model(io.StringIO(text)), "DDR")
    for name, expected in fixture["expected_ddr"].items():
        np.testing.assert_allclose(
            np.longdouble(getattr(converted, name).value),
            np.longdouble(expected),
            rtol=2e-16,
            atol=2e-18,
            err_msg=name,
        )


def test_dd_orientation_transfer_negative_controls():
    text = parDD_ddr.replace("OMDOT 0", "OMDOT 0.01").replace("PX 1.0", "PX 0")
    source = get_model(io.StringIO(text))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    toas = pint.simulation.make_fake_toas_uniform(52000, 58000, 400, source, obs="@")
    source_delay = source.components["BinaryDD"].binarymodel_delay(toas).to_value(u.s)
    _chart, coefficients, _names = pint.binaryconvert._phase_chart(source)
    source_f0 = np.longdouble(coefficients[0])
    kappa = np.longdouble(source.OMDOT.quantity.to_value(u.rad / u.s)) / (
        2 * np.longdouble(np.pi) * source_f0
    )

    def difference_for(omega):
        candidate = copy.deepcopy(converted)
        delta_s = pint.binaryconvert._monotone_phase_root(
            coefficients,
            -omega / (2 * np.longdouble(np.pi)),
            -omega / (2 * np.longdouble(np.pi) * source_f0),
        )
        shifted = pint.binaryconvert._shift_phase_coeffs(coefficients, delta_s)
        candidate.TASC.value = np.longdouble(source.T0.value) + delta_s / 86400
        candidate.PB.quantity = (1 / shifted[0]) * u.s
        candidate.PBDOT.value = -shifted[1] / shifted[0] ** 2
        candidate.EPS1.value = np.longdouble(source.ECC.value) * np.sin(omega)
        candidate.EPS2.value = np.longdouble(source.ECC.value) * np.cos(omega)
        candidate.OMDOT.quantity = (
            np.longdouble(source.OMDOT.quantity.to_value(u.rad / u.s))
            * shifted[0]
            / source_f0
            * u.rad
            / u.s
        )
        delay = candidate.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
        difference = source_delay - delay
        return np.max(np.abs(difference - np.mean(difference)))

    om_dd = np.longdouble(source.OM.quantity.to_value(u.rad))
    shortcut_error = difference_for(om_dd)
    seed_error = difference_for(om_dd / (1 + kappa))
    assert shortcut_error > 1e-7
    assert seed_error > 1e-10


@pytest.mark.parametrize(
    ("om_deg", "omdot"),
    [(0.0001, 0.01), (359.9999, 0.01), (0.0001, -0.01)],
)
def test_dd_orientation_transfer_unwrapped_branches(om_deg, omdot):
    text = (
        parDD_ddr.replace("OM 30", f"OM {om_deg}")
        .replace("OMDOT 0", f"OMDOT {omdot}")
        .replace("PX 1.0", "PX 0")
    )
    source = get_model(io.StringIO(text))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    toas = pint.simulation.make_fake_toas_uniform(52000, 58000, 80, source, obs="@")
    source_delay = source.components["BinaryDD"].binarymodel_delay(toas).to_value(u.s)
    target_delay = (
        converted.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    np.testing.assert_allclose(source_delay, target_delay, rtol=0, atol=2e-12)
    roundtrip = pint.binaryconvert.convert_binary(converted, "DD")
    roundtrip_delay = (
        roundtrip.components["BinaryDD"].binarymodel_delay(toas).to_value(u.s)
    )
    np.testing.assert_allclose(roundtrip_delay, source_delay, rtol=0, atol=2e-12)


def test_dd_fit_flags_uncertainties_and_report():
    text = (
        parDD_ddr.replace("T0 55000.0005", "T0 55000.0005 1 1e-7")
        .replace("OM 30", "OM 30 1 1e-4")
        .replace("ECC 0.001", "ECC 0.001 1 1e-7")
        .replace("M2 0.2", "M2 0.2 1 0.01")
        .replace("SINI 0.8660254037844386", "SINI 0.8660254037844386 1 1e-4")
    )
    m = get_model(io.StringIO(text))
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    for name in ("EPS1", "EPS2", "TASC", "M2", "COSI"):
        assert not getattr(ddr, name).frozen
    assert ddr.M2.uncertainty is not None
    assert ddr.EPS1.uncertainty is not None
    assert ddr.binary_conversion_report["fit_space"] == ["preserved"]
    assert ddr.binary_conversion_report["free_params"]["source"]
    assert ddr.binary_conversion_report["free_params"]["target"]


def test_dd_fit_space_shrinks_when_only_eccentricity_is_free():
    source = get_model(io.StringIO(parDD_ddr.replace("ECC 0.001", "ECC 0.001 1 1e-7")))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert converted.EPS1.frozen
    assert converted.EPS2.frozen
    assert converted.binary_conversion_report["fit_space"] == ["shrunk: EPS1, EPS2"]


def test_dd_frozen_t0_keeps_tasc_frozen_without_shrinking_element_space():
    source = get_model(
        io.StringIO(
            parDD_ddr.replace("ECC 0.001", "ECC 0.001 1 1e-7").replace(
                "OM 30", "OM 30 1 1e-4"
            )
        )
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert converted.TASC.frozen
    assert not converted.EPS1.frozen
    assert not converted.EPS2.frozen
    assert converted.binary_conversion_report["fit_space"] == ["preserved"]


def test_dd_conversion_accepts_covariance_and_propagates_it():
    text = (
        parDD_ddr.replace("T0 55000.0005", "T0 55000.0005 1")
        .replace("OM 30", "OM 30 1")
        .replace("ECC 0.001", "ECC 0.001 1")
        .replace("M2 0.2", "M2 0.2 1")
        .replace("SINI 0.8660254037844386", "SINI 0.8660254037844386 1")
    )
    source = get_model(io.StringIO(text))
    sigmas = {
        "T0": 1e-7,
        "OM": 1e-4,
        "ECC": 1e-7,
        "M2": 0.01,
        "SINI": 1e-4,
    }
    covariance = np.diag([sigmas[name] ** 2 for name in source.free_params])
    target = pint.binaryconvert.convert_binary(source, "DDR", covariance=covariance)
    assert target.binary_conversion_report["uncertainty_propagation"] == "covariance"
    for name in ("EPS1", "EPS2", "TASC", "M2", "COSI"):
        assert getattr(target, name).uncertainty is not None


@pytest.mark.parametrize(
    "name,value",
    [
        ("DR", "1e-4"),
        ("DTH", "1e-4"),
        ("EDOT", "1e-20"),
        ("A0", "1e-6"),
    ],
)
def test_ddr_import_refuses_untransferred_active_terms(name, value):
    m = get_model(io.StringIO(parDD_ddr + f"\n{name} {value}\n"))
    with pytest.raises(TimingModelError, match=name):
        pint.binaryconvert.convert_binary(m, "DDR")


def test_ddr_import_refuses_populated_zero_orbwave():
    source = get_model(
        io.StringIO(
            parELL1_ddr
            + "\nORBWAVE_OM 1e-6\nORBWAVE_EPOCH 55000\nORBWAVEC0 0\nORBWAVES0 0\n"
        )
    )
    with pytest.raises(TimingModelError, match="ORBWAVE"):
        pint.binaryconvert.convert_binary(source, "DDR")


parELL1H_ddr = """\
PSRJ J0000+0000
RAJ 00:00:00.0
DECJ +00:00:00.0
F0 200
PEPOCH 55000
DM 0
BINARY ELL1H
PB 1.0
A1 10
TASC 55000
EPS1 0
EPS2 0
H3 1.25e-7
STIGMA 0.5
PBDOT 0
UNITS TDB
"""


def test_ell1h_absorbed_applies_orbit_decode():
    m = get_model(io.StringIO(parELL1H_ddr), ell1h_shapiro="absorbed")
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert "full_from_absorbed" in ddr.binary_conversion_report["shapiro"]
    # x = x_a - 4 r ς; r = H3 / ς³ = 1e-6, 4 r ς = 2e-6
    np.testing.assert_allclose(ddr.A1.value, 10.0 - 2e-6, rtol=0, atol=1e-12)


def test_ell1h_absorbed_public_conversion_is_decode_only():
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("PBDOT 0", "PBDOT 1e-12")),
        ell1h_shapiro="absorbed",
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    decoded = fw10_orbit_decode(
        source.A1.value,
        source.EPS1.value,
        source.EPS2.value,
        source.TASC.value,
        source.PB.quantity.to_value(u.s),
        1e-6,
        0.5,
    )
    assert np.longdouble(converted.TASC.value) == pytest.approx(decoded[3], abs=2e-15)
    assert converted.A1.value == pytest.approx(decoded[0], abs=2e-14)

    toas = pint.simulation.make_fake_toas_uniform(54999, 55001, 100, source, obs="@")
    source_delay = (
        source.components["BinaryELL1H"].binarymodel_delay(toas).to_value(u.s)
    )
    target_delay = (
        converted.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    difference = source_delay - target_delay
    difference -= np.mean(difference)
    assert 1e-10 < np.max(np.abs(difference)) < 2e-9

    roundtrip = pint.binaryconvert.convert_binary(
        converted, "ELL1H", useSTIGMA=True, ell1h_shapiro="absorbed"
    )
    for name in ("A1", "EPS1", "EPS2", "TASC", "PBDOT"):
        np.testing.assert_allclose(
            getattr(roundtrip, name).value,
            getattr(source, name).value,
            rtol=2e-12,
            atol=2e-14,
        )


@pytest.mark.parametrize(("eps1", "doubled_ns"), [(1e-6, 10.9), (1e-4, 1090.0)])
def test_ell1h_absorbed_nonzero_h_is_decode_not_double_gauge(eps1, doubled_ns):
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("EPS1 0", f"EPS1 {eps1}")),
        ell1h_shapiro="absorbed",
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    decoded = fw10_orbit_decode(
        source.A1.value,
        source.EPS1.value,
        source.EPS2.value,
        source.TASC.value,
        source.PB.quantity.to_value(u.s),
        1e-6,
        0.5,
    )
    assert np.longdouble(converted.TASC.value) == pytest.approx(decoded[3], abs=2e-15)

    toas = pint.simulation.make_fake_toas_uniform(54999, 55001, 400, source, obs="@")
    source_delay = (
        source.components["BinaryELL1H"].binarymodel_delay(toas).to_value(u.s)
    )
    target_delay = (
        converted.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    difference = source_delay - target_delay
    difference -= np.mean(difference)
    assert np.max(np.abs(difference)) < 1e-9

    doubled = copy.deepcopy(converted)
    doubled.TASC.value = decoded[3] + (
        np.longdouble("1.5") * decoded[0] * decoded[1] / 86400
    )
    doubled_delay = (
        doubled.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    )
    negative = source_delay - doubled_delay
    negative -= np.mean(negative)
    assert np.max(np.abs(negative)) * 1e9 == pytest.approx(doubled_ns, rel=0.15)


def test_ell1_gauge_copies_phase_chart_and_a1():
    source = get_model(
        io.StringIO(parELL1_ddr.replace("PBDOT 0", "PBDOT 1e-12") + "\nA1DOT 1e-14\n")
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    np.testing.assert_allclose(
        converted.A1.quantity.to_value(u.lsec),
        source.A1.quantity.to_value(u.lsec),
        rtol=0,
        atol=_ddr_atol(0, source.A1.quantity.to_value(u.lsec)),
    )
    np.testing.assert_allclose(
        converted.PB.quantity.to_value(u.s),
        source.PB.quantity.to_value(u.s),
        rtol=0,
        atol=_ddr_atol(1e-18, source.PB.quantity.to_value(u.s)),
    )
    np.testing.assert_allclose(
        converted.PBDOT.value,
        source.PBDOT.value,
        rtol=_ddr_rtol(0),
        atol=_ddr_atol(0, source.PBDOT.value),
    )
    expected_s = (
        np.longdouble("1.5")
        * np.longdouble(source.A1.value)
        * np.longdouble(source.EPS1.value)
    )
    actual_s = (
        np.longdouble(converted.TASC.value) - np.longdouble(source.TASC.value)
    ) * 86400
    np.testing.assert_allclose(actual_s, expected_s, rtol=0, atol=2e-10)


def test_ell1h_absorbed_refuses_nonzero_a1dot():
    source = get_model(
        io.StringIO(parELL1H_ddr + "\nA1DOT 1e-12\n"),
        ell1h_shapiro="absorbed",
    )
    with pytest.raises(TimingModelError, match="A1DOT"):
        pint.binaryconvert.convert_binary(source, "DDR")


def test_ell1_roundtrip_preserves_phase_fit_flags():
    source = get_model(io.StringIO(parELL1_ddr))
    source.PB.frozen = False
    source.PBDOT.frozen = False
    source.A1.frozen = False
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert not converted.PB.frozen
    assert not converted.PBDOT.frozen
    assert not converted.A1.frozen
    back = pint.binaryconvert.convert_binary(converted, "ELL1")
    assert not back.PB.frozen
    assert not back.PBDOT.frozen
    assert not back.A1.frozen
    assert back.binary_conversion_report["fit_space"] == ["preserved"]


def test_rescale_pb_refused_for_dd_source():
    source = get_model(io.StringIO(parDD_ddr))
    with pytest.raises(TimingModelError, match="rescale_pb"):
        pint.binaryconvert.convert_binary(source, "DDR", ddrpk=True, rescale_pb=True)


def test_ell1h_full_does_not_shift_a1():
    m = get_model(io.StringIO(parELL1H_ddr), ell1h_shapiro="full")
    ddr = pint.binaryconvert.convert_binary(m, "DDR")
    assert "full_from_harmonics" in ddr.binary_conversion_report["shapiro"]
    np.testing.assert_allclose(ddr.A1.value, 10.0, rtol=0, atol=1e-14)


def test_ell1h_h3_only_uses_wd_mass_prior():
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("STIGMA 0.5", "")),
        ell1h_shapiro="full",
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert "h3_only_prior" in converted.binary_conversion_report["orientation"]
    assert converted.M2.frozen
    assert converted.M2.value == pytest.approx(0.2, rel=1e-12)
    declared = pint.binaryconvert.convert_binary(
        source, "DDR", cosi=0.5 * u.dimensionless_unscaled
    )
    assert declared.COSI.value == pytest.approx(0.5)
    assert "declared_cosi" in declared.binary_conversion_report["orientation"]
    with pytest.raises(TimingModelError, match="cosi= or stigma="):
        pint.binaryconvert.convert_binary(source, "DDR", h3_only="require")


def test_ell1h_without_h3_treated_as_ell1():
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("H3 1.25e-7\nSTIGMA 0.5\n", "")),
        ell1h_shapiro="full",
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert "ell1h_without_h3" in converted.binary_conversion_report["shapiro"]
    assert "absent_off" in converted.binary_conversion_report["shapiro"]
    assert converted.M2.value == pytest.approx(0.0)
    assert converted.M2.frozen


def test_ell1h_negative_h3_is_refused():
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("H3 1.25e-7", "H3 -1.25e-7")),
        ell1h_shapiro="full",
    )
    with pytest.raises(TimingModelError, match="H3 > 0"):
        pint.binaryconvert.convert_binary(source, "DDR")


@pytest.mark.parametrize("par", [parELL1_ddr, parDD_ddr])
def test_incomplete_m2_sini_keeps_shapiro_off(par):
    text = (
        "\n".join(line for line in par.splitlines() if not line.startswith("SINI "))
        + "\n"
    )
    source = get_model(io.StringIO(text))
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert "absent_off" in converted.binary_conversion_report["shapiro"]
    assert converted.M2.value == pytest.approx(0.0)
    assert converted.M2.frozen
    assert converted.COSI.frozen
    assert converted.COSI.value == pytest.approx(0.5)


def test_bt_imports_as_dd_like_upgrade():
    source = get_model(
        io.StringIO(
            parDD_ddr.replace("BINARY DD", "BINARY BT")
            .replace("M2 0.2\n", "")
            .replace("SINI 0.8660254037844386\n", "")
        )
    )
    converted = pint.binaryconvert.convert_binary(source, "DDR")
    assert converted.BINARY.value == "DDR"
    assert "bt_to_dd" in converted.binary_conversion_report["orbital_law"]
    assert "absent_off" in converted.binary_conversion_report["shapiro"]
    with pytest.raises(TimingModelError, match="not supported"):
        pint.binaryconvert.convert_binary(converted, "BT")


def test_drop_edot_allows_dd_import():
    source = get_model(io.StringIO(parDD_ddr + "\nEDOT 1e-20\n"))
    with pytest.raises(TimingModelError, match="EDOT"):
        pint.binaryconvert.convert_binary(source, "DDR")
    converted = pint.binaryconvert.convert_binary(source, "DDR", drop_edot=True)
    assert "edot_dropped" in converted.binary_conversion_report["secular"]
    assert converted.BINARY.value == "DDR"


def test_ell1h_invalid_stigma_refuses():
    source = get_model(
        io.StringIO(parELL1H_ddr.replace("STIGMA 0.5", "STIGMA 2")),
        ell1h_shapiro="full",
    )
    with pytest.raises((InvalidModelParameters, TimingModelError), match="STIGMA"):
        pint.binaryconvert.convert_binary(source, "DDR")


def test_ddr_to_ell1h_absorbed_applies_encode():
    from test_ddr import _pheno

    ddr = get_model(io.StringIO(_pheno(EPS1="0", EPS2="0", DM="0")))
    ell1h = pint.binaryconvert.convert_binary(
        ddr, "ELL1H", useSTIGMA=True, ell1h_shapiro="absorbed"
    )
    assert ell1h.components["BinaryELL1H"].ell1h_shapiro == "absorbed"
    assert "full_from_absorbed" in ell1h.binary_conversion_report["shapiro"]
    assert ell1h.A1.value > ddr.A1.value


def test_ddr_exports_dss_ddh_and_ell1h_secular_terms():
    from test_ddr import _pheno

    ddr = get_model(io.StringIO(_pheno(PBDOT="1e-12", OMDOT="0.01", GGAMMA="0.002")))
    dds = pint.binaryconvert.convert_binary(ddr, "DDS")
    np.testing.assert_allclose(dds.SHAPMAX.value, -np.log(1 - ddr.SINI.value))
    ddh = pint.binaryconvert.convert_binary(ddr, "DDH")
    np.testing.assert_allclose(ddh.GAMMA.value, ddr.GAMMA.value, rtol=2e-14)
    assert ddh.OMDOT.value != 0

    ell1_source = get_model(io.StringIO(_pheno(PBDOT="1e-12", OMDOT="0", GGAMMA="0")))
    ell1h = pint.binaryconvert.convert_binary(ell1_source, "ELL1H")
    np.testing.assert_allclose(ell1h.PBDOT.value, 1e-12, rtol=2e-13)


def test_ddrpk_export_keeps_derived_pk_frozen():
    from test_ddr import example_par

    ddr = get_model(
        io.StringIO(
            example_par(
                DDRGEO="N",
                DDRKINE="N",
                PX=None,
                PMRA=None,
                PMDEC=None,
                KOM=None,
                XPBDOT="0",
            )
        )
    )
    dd = pint.binaryconvert.convert_binary(ddr, "DD")
    assert dd.OMDOT.frozen
    assert dd.GAMMA.frozen
    assert dd.binary_conversion_report["fit_space"] == [
        "constraint_released: OMDOT, GAMMA, PBDOT"
    ]


def test_ddr_to_ddk_rereferences_inclination_omega_and_axis():
    from test_ddr import example_par

    ddr = get_model(
        io.StringIO(
            example_par(
                TASC="58652.5",
                TGEO="55000",
                POSEPOCH="55000",
                PMRA="0",
                PMDEC="10",
                KOM="0",
            )
        )
    )
    ddk = pint.binaryconvert.convert_binary(ddr, "DDK")
    dt = (np.longdouble(ddk.T0.value) - np.longdouble(ddr.TGEO.value)) * u.d
    delta_i = (10 * u.mas / u.yr * dt).to_value(
        u.rad, equivalencies=u.dimensionless_angles()
    )
    expected_axis = np.longdouble(ddr.A1.value) * (
        1 + delta_i / np.tan(ddr.KIN.quantity.to_value(u.rad))
    )
    np.testing.assert_allclose(ddk.A1.value, expected_axis, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        ddk.KIN.quantity.to_value(u.rad),
        ddr.KIN.quantity.to_value(u.rad) + delta_i,
        rtol=0,
        atol=2e-15,
    )
    assert "to_ddk_linearized" in ddk.binary_conversion_report["geometry"]
    assert "ddk_orbital_parallax" in ddk.binary_conversion_report["geometry"]


def test_ddr_to_ddk_axis_term_delay_control():
    from test_ddr import example_par

    ddr = get_model(
        io.StringIO(
            example_par(
                A1="2.0",
                TASC="58652.5",
                TGEO="55000",
                POSEPOCH="55000",
                PMRA="0",
                PMDEC="10",
                KOM="0",
                DDRPK="N",
                DDRPBDOT="absorb_gw",
                PBDOT="0",
                OMDOT="0",
                GGAMMA="0",
            )
        )
    )
    ddk = pint.binaryconvert.convert_binary(ddr, "DDK")
    dt = (np.longdouble(ddk.T0.value) - np.longdouble(ddr.TGEO.value)) * u.d
    delta_i = (10 * u.mas / u.yr * dt).to_value(
        u.rad, equivalencies=u.dimensionless_angles()
    )
    axis = (
        np.longdouble(2.0) * delta_i / np.tan(np.arccos(np.longdouble(ddr.COSI.value)))
    )
    toas = pint.simulation.make_fake_toas_uniform(58600, 58700, 48, ddr, obs="@")
    ddr_delay = ddr.components["BinaryDDR"].binarymodel_delay(toas).to_value(u.s)
    ddk_delay = ddk.components["BinaryDDK"].binarymodel_delay(toas).to_value(u.s)
    no_axis = copy.deepcopy(ddk)
    no_axis.A1.value = np.longdouble(ddk.A1.value) - axis
    no_delay = no_axis.components["BinaryDDK"].binarymodel_delay(toas).to_value(u.s)
    with_term = ddr_delay - ddk_delay
    without_term = ddr_delay - no_delay
    with_term -= np.mean(with_term)
    without_term -= np.mean(without_term)
    assert np.max(np.abs(without_term)) * 1e9 == pytest.approx(560, rel=0.25)
    assert np.max(np.abs(with_term)) < 0.2 * np.max(np.abs(without_term))


@pytest.mark.parametrize("output", ["BT", "ELL1k"])
def test_ddr_refuses_unsupported_exports(output):
    from test_ddr import _pheno

    ddr = get_model(io.StringIO(_pheno()))
    with pytest.raises(TimingModelError, match=output):
        pint.binaryconvert.convert_binary(ddr, output)
