"""Various test for the bad par files"""
import pytest
from io import StringIO

from pint.models.timing_model import TimingModelError
from pint.models.model_builder import get_model


base_par="""
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
    m = get_model(StringIO(mixed_par))
    assert hasattr(m, 'EQUAD2')
    assert m.EQUAD2.value == 2
    assert m.EQUAD2.key == 'fe'
    assert m.EQUAD2.key_value == ['guppi']

def test_conflict_position():
    conf_par = base_par + "\nRAJ 12:01:02" + "\nDECJ 12:01:02"
    #get_model(StringIO(conf_par))


def test_conflict_pos_pm():
    conf_par = base_par + "\nPMRA -2.1" + "\nPMDECJ -2.2"
    #get_model(StringIO(conf_par))

def test_bad_binary_name():
    pass

def test_dulicate_binary():
    pass

def test_meaningless_line():
    pass

#def test_
