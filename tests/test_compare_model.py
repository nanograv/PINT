from io import StringIO
from copy import deepcopy

import pytest

from pint.models import get_model
from pint.scripts import compare_parfiles

par_basic = """
PSR J1234+5678
ELAT 0 1 1e-6
ELONG 0 1 1e-6
PEPOCH 57000
F0 1 1 1e-6
"""

par_bin = (
    par_basic
    + """
BINARY ELL1
PB 1 1 1e-6
A1 10 1 1e-6
TASC 57000
EPS1 0 1 1e-6
EPS2 0 1 1e-6
"""
)


@pytest.mark.parametrize(
    "par, line",
    [
        (par_basic, ""),
        (par_basic, "F1 0"),
        (par_basic, "F1 0 1"),
        (par_bin, "PBDOT 0 0"),
    ],
)
def test_unset(par, line):
    m1 = get_model(StringIO(par_basic))
    m2 = get_model(StringIO("\n".join([par, line])))
    m1.compare(m2)
    m2.compare(m1)


par_15yr_a = """
PSR                            J1738+0333
EPHEM                               DE440
CLOCK                        TT(BIPM2019)
UNITS                                 TDB
ELONG                 264.094916455970292 1 0.00000001627066523287
ELAT                   26.884240630647529 1 0.00000003355674214485
PMELONG                 6.866999380457335 1 0.023068982613321045
PMELAT                  5.260024482574339 1 0.0492892020311819
PX                     0.6513416871655758 1 0.25848239903639775
POSEPOCH           57054.5451898281098693
F0                  170.93737241242573821 1 2.2675806693272832601e-13
F1              -7.0473358332142274913e-16 1 4.6499232274848548092e-21
PEPOCH             57054.5451898281098693
DM                   34.22159583883240564
BINARY ELL1
PB                 0.35479073432202298626 1 8.966975233400924148e-12
A1                      0.343429812939532 1 1.2378761170297296e-07
TASC               56524.9399563860504259 1 2.6584108234670461195e-08
EPS1            1.1161957981429708052e-06 1 5.6759295647382127554e-07
EPS2            -4.0214633261943368967e-07 1 5.8457540084782386455e-07
EPS1DOT                                 0 1
EPS2DOT                                 0 1
"""

par_15yr_b = """
PSR                            J1738+0333
EPHEM                               DE440
CLOCK                        TT(BIPM2019)
UNITS                                 TDB
ELONG                 264.094916439827557 1 0.00000006366078510011
ELAT                   26.884240649512481 1 0.00000013703616696038
PMELONG                 6.879358744725808 1 0.0868512038839615
PMELAT                  5.245175629568009 1 0.19452101016523687
PX                     0.8645790061281238 1 1.0853234778579688
POSEPOCH           57054.5451898281098693
F0                  170.93737241242648785 1 9.222762418705808598e-13
F1              -7.047472146156761276e-16 1 1.8774119548217020186e-20
PEPOCH             57054.5451898281098693
DM                  33.738993039399929004
BINARY ELL1
PB                  0.3547907343419840973 1 3.2425515183801981837e-11
A1                      0.343429789774511 1 4.788436999216635e-07
TASC               56524.9399563066790186 1 8.8313377079315856625e-08
EPS1            -1.1245449642203993914e-06 1 1.97301995395510912e-06
EPS2            -1.8943477531941200777e-06 1 2.097611442502435212e-06
"""


def test_known_problem():
    m1 = get_model(StringIO(par_15yr_a))
    m2 = get_model(StringIO(par_15yr_b))
    m1.compare(m2)
    m2.compare(m1)


def test_compare_parfile_script():
    parfile1 = "par_15yr_a.par"
    parfile2 = "par_15yr_b.par"

    args = ""

    with open(parfile1, "w") as par1:
        par1.write(par_15yr_a)

    with open(parfile2, "w") as par2:
        par2.write(par_15yr_b)

    argv = f"{args} {parfile1} {parfile2}".split()
    compare_parfiles.main(argv)


def test_compare_model_binary():
    m1 = get_model(StringIO(par_bin))

    m2 = deepcopy(m1)
    m2.TASC.value = 57001

    m1.compare(m2)
    m2.compare(m1)
