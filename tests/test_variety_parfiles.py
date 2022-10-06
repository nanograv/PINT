"""Various test for the bad par files"""
import pytest
from io import StringIO

from pint.models.timing_model import (
    TimingModelError,
    UnknownBinaryModel,
    MissingBinaryError,
    MissingParameter,
)
from pint.models.model_builder import get_model


base_par = """
PSR J1234+5678
ELAT 0
ELONG 0
F0 1
DM 10
PEPOCH 57000
"""


def test_duplicate_nonrepeat_param_name():
    dup_par = base_par + "\nDM 2"
    with pytest.raises(TimingModelError):
        get_model(StringIO(dup_par))


def test_dulicate_with_alise():
    dup_par = base_par + "\nLAMBDA 1"
    with pytest.raises(TimingModelError):
        get_model(StringIO(dup_par))


def test_mix_alise():
    mixed_par = base_par + "\nEQUAD -fe 430 1\nT2EQUAD -fe guppi 2"
    m = get_model(StringIO(mixed_par), allow_name_mixing=True)
    assert hasattr(m, "EQUAD2")
    assert m.EQUAD2.value == 2
    assert m.EQUAD2.key == "-fe"
    assert m.EQUAD2.key_value == ["guppi"]


def test_conflict_position():
    conf_par = base_par + "\nRAJ 12:01:02" + "\nDECJ 12:01:02"
    with pytest.raises(TimingModelError):
        get_model(StringIO(conf_par))


def test_conflict_pos_pm():
    conf_par = base_par + "\nPMRA -2.1" + "\nPMDECJ -2.2"
    with pytest.raises(TimingModelError):
        get_model(StringIO(conf_par))


def test_bad_binary_name():
    bad_binary = base_par + "\nBINARY BADBINARY"
    with pytest.raises(UnknownBinaryModel):
        get_model(StringIO(bad_binary))


def test_dulicate_binary():
    bad_binary = base_par + "\nBINARY DD\nBINARY BT"
    with pytest.raises(TimingModelError):
        get_model(StringIO(bad_binary))


def test_no_binary():
    bad_binary = base_par + "\nECC 0.5 1 1\nT0 55000"
    with pytest.raises(MissingBinaryError):
        get_model(StringIO(bad_binary))


def test_meaningless_line():
    broken_line = "\nThis is a meaningless line"
    meaningless = base_par + broken_line
    with pytest.warns(UserWarning):
        get_model(StringIO(meaningless))


def test_broken_parameter_line_fit():
    broken_line1 = "\nPMELONG 3 SC 0.8"
    broken1 = base_par + broken_line1
    with pytest.raises(
        ValueError, match="Unidentified string 'SC' in parfile " "line PMELONG 3 SC 0.8"
    ):
        get_model(StringIO(broken1))


def test_broken_parameter_line_value():
    broken_line1 = "\nPMELONG SC 1 0.8"
    broken1 = base_par + broken_line1
    with pytest.raises(ValueError, match="could not convert string to float: 'SC'"):
        get_model(StringIO(broken1))


def test_broken_parameter_line_uncertainty():
    broken_line1 = "\nPMELONG 1 1 SC"
    broken1 = base_par + broken_line1
    with pytest.raises(ValueError, match="could not convert string to float: 'SC'"):
        get_model(StringIO(broken1))


def test_illegal_mixtrues():
    mix_line = "\nBINARY BT\nPB 1 2 3\nFB0 2 3 4"
    mix_par = base_par + mix_line
    with pytest.raises(
        ValueError, match="Model cannot have values for both FB0 and PB"
    ):
        get_model(StringIO(mix_par))


def test_wrong_parameter_for_model():
    wrong_parameter = "\nBINARY DDK\nSINI 0.9"
    wrong_p_model = base_par + wrong_parameter
    with pytest.raises(TimingModelError):
        get_model(StringIO(wrong_p_model))


def test_missing_parameter_for_model():
    missing_param_line = "\nBINARY DD\nECC 1 2 3"
    missing_model = base_par + missing_param_line
    with pytest.raises(MissingParameter):
        get_model(StringIO(missing_model))
