"""Tests for par file parsing, pintify parfile and value fill up
"""

import pytest
from io import StringIO

from pint.models.model_builder import ModelBuilder, parse_parfile, TimingModelError
from pint.models.model_builder import guess_binary_model

from pint.models import get_model

base_par = """
PSR J1234+5678
ELAT 0 1 2
ELONG 0 1 3
F0 1 1 5
DM 10 0 3
PEPOCH 57000
"""


def test_parse_parfile():
    par_dict = parse_parfile(StringIO(base_par))
    assert len(par_dict) == len((base_par.strip()).split("\n"))
    assert par_dict["ELAT"] == ["0 1 2"]


def test_parse_parfile_repeatline():
    repeat_par = base_par + "DM 1 1 3\nF0 2 1 3"
    par_dict = parse_parfile(StringIO(repeat_par))
    assert len(par_dict) == len((repeat_par.strip()).split("\n")) - 2
    assert par_dict["DM"] == ["10 0 3", "1 1 3"]
    assert par_dict["F0"] == ["1 1 5", "2 1 3"]


def test_pintify_parfile_from_string():
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(base_par))
    assert len(pint_dict) == len((base_par.strip()).split("\n"))
    assert len(unknow) == 0


def test_pintify_parfile_from_parfile_dict():
    m = ModelBuilder()
    par_dict = parse_parfile(StringIO(base_par))
    pint_dict, original_name, unknow = m._pintify_parfile(par_dict)
    assert len(pint_dict) == len((base_par.strip()).split("\n"))
    assert len(unknow) == 0


def test_pintify_parfile_repeat_line_with_nonrepeat_param():
    repeat_par = base_par + "DM 1 1 3\nF0 2 1 3"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(repeat_par))


def test_pintify_parfile_repeat_line_with_repeat_param():
    repeat_par = base_par + "JUMP -fe 1 1 3\nJUMP -fe 2 1 3"
    repeat_par += "\nECORR -fe 1 1 2\nECORR -fe 2 1 2"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(repeat_par))
    assert len(pint_dict["JUMP1"]) == 2
    assert len(pint_dict["ECORR1"]) == 2


def test_pintify_parfile_repeat_line_with_repeat_aliases():
    repeat_par = base_par + "T2EQUAD -fe 1 1 3\nT2EQUAD -fe 2 1 3"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(repeat_par))
    assert len(pint_dict["EQUAD1"]) == 2
    assert "T2EQUAD" not in pint_dict.keys()


def test_pintify_parfile_repeat_line_mix_alises():
    repeat_par = base_par + "EQUAD -fe 1 1 3\nT2EQUAD -fe 2 1 3\nEQUAD -fe 4 2 2"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(repeat_par))
    pint_dict, original_name, unknow = m._pintify_parfile(
        StringIO(repeat_par), allow_name_mixing=True
    )
    # The list order is swaped since parse_parfile will pack EQUADs first and
    # then the pintify will add T2ECORR to the list.
    assert pint_dict["EQUAD1"] == ["-fe 1 1 3", "-fe 4 2 2", "-fe 2 1 3"]
    assert "T2EQUAD" not in pint_dict.keys()


def test_parse_parfile_index_param():
    indexed_par = base_par + "DMX_0001 1 1 2\nDMX_0025 2 1 3"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(indexed_par))
    assert pint_dict["DMX_0001"] == ["1 1 2"]
    assert pint_dict["DMX_0025"] == ["2 1 3"]


dm_deriv_lines = """
            PSR J1234+5678
            ELAT 0 1 2
            ELONG 0 1 3
            F0 1 1 5
            DM 10 0 3
            PEPOCH 57000
            DM001 0 1 0
            DM002 1 1 0
            DM3 2 1 0
            DM010 3 1 0
            DMX_0001 0 1 0
            DMXR1_0001 59000
            DMXR2_0001 59001
            """


def test_parse_dm_derivs():
    dm_deriv_par = StringIO(dm_deriv_lines)
    m = get_model(dm_deriv_par)
    assert m.DM.value == 10.0
    assert m.DM1.value == 0.0
    assert m.DM3.value == 2.0
    assert m.DM10.value == 3.0
    assert m.DMX_0001.value == 0.0


def test_name_check_dm_derivs():
    dm_deriv_par = StringIO(dm_deriv_lines)
    mb = ModelBuilder()
    pint_dict, original_name, unknow = mb._pintify_parfile(dm_deriv_par)
    assert original_name["DM1"] == "DM001"
    assert original_name["DM3"] == "DM3"
    assert original_name["DM10"] == "DM010"


def test_pintify_parfile_alises():
    aliases_par = base_par.replace("ELONG", "LAMBDA")
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(aliases_par))
    assert pint_dict["ELONG"] == ["0 1 3"]
    assert len(unknow) == 0
    assert "LAMBDA" not in pint_dict.keys()


def test_pintify_parfile_nonrepeat_with_alise_repeating():
    aliases_par = f"{base_par}LAMBDA 2 1 3"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(aliases_par))


def test_pintify_parfile_unrecognize():
    wrong_par = f"{base_par}UNKNOWN 2 1 1"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(wrong_par))
    assert unknow == {"UNKNOWN": ["2 1 1"]}


def test_pintify_parfile_duplicate_binary():
    dup_par = base_par + "BINARY DD\nBINARY BT"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(dup_par))


ddk_par = """
PSR J1234+5678
ELAT 0 1 2
ELONG 0 1 3
F0 1 1 5
DM 10 0 3
PEPOCH 57000
BINARY T2
PB 60
T0 54300
A1 30
OM 150
ECC 8.0e-05
PBDOT 2.0e-14
OMDOT 0.0001
M2 0.3
KOM 90
KIN 70
SINI KIN
#SINI 0.95
"""

ell1h_par = """
PSR J1234+5678
ELAT 0 1 2
ELONG 0 1 3
F0 1 1 5
DM 10 0 3
PEPOCH 57000
BINARY T2
PB 60
A1 1
PBDOT 2.0e-14
TASC 54300
EPS1 1.0e-06
EPS2 1.0e-6
H3 1.0e-07
H4 1.0e-07
"""

nobinarymodel_par = """
PSR J1234+5678
ELAT 0 1 2
ELONG 0 1 3
F0 1 1 5
DM 10 0 3
PEPOCH 57000
BINARY T2
PB 60
A1 1
PBDOT 2.0e-14
TASC 54300
T0 54300
EPS1 1.0e-06
EPS2 1.0e-6
H3 1.0e-07
H4 1.0e-07
"""


def test_guess_binary_model():
    parfiles = [base_par, ddk_par, ell1h_par, nobinarymodel_par]
    binary_models = ["Isolated", "DDK", "ELL1H", None]
    trip_raise = [False, True, True, True]

    for parfile, binary_model, trip in zip(parfiles, binary_models, trip_raise):
        par_dict = parse_parfile(StringIO(parfile))

        binary_model_guesses = guess_binary_model(par_dict)

        if binary_model:
            assert binary_model_guesses[0] == binary_model
        else:
            assert binary_model_guesses == []

        if trip:
            with pytest.raises(TimingModelError):
                m = get_model(StringIO(parfile))
