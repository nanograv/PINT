#! /usr/bin/env python
"""Tests for par file parsing, pintify parfile and value fill up
"""
import os
import tempfile

import pytest
from io import StringIO

from pint.models.model_builder import (
    ModelBuilder,
    parse_parfile,
    TimingModelError
)


base_par="""
PSR J1234+5678
ELAT 0 1 2
ELONG 0 1 3
F0 1 1 5
DM 10 0 3
PEPOCH 57000
"""


def test_parse_parfile():
    par_dict = parse_parfile(StringIO(base_par))
    assert len(par_dict) == len((base_par.strip()).split('\n'))
    assert par_dict['ELAT'] == ['0 1 2']


def test_parse_parfile_repeatline():
    repeat_par = base_par + "DM 1 1 3\nF0 2 1 3"
    par_dict = parse_parfile(StringIO(repeat_par))
    assert len(par_dict) == len((repeat_par.strip()).split('\n')) - 2
    assert par_dict['DM'] == ['10 0 3', '1 1 3']
    assert par_dict['F0'] == ['1 1 5', '2 1 3']


def test_pintify_parfile_from_string():
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(base_par))
    assert len(pint_dict) == len((base_par.strip()).split('\n'))
    assert len(unknow) == 0


def test_pintify_parfile_from_parfile_dict():
    m = ModelBuilder()
    par_dict = parse_parfile(StringIO(base_par))
    pint_dict, original_name, unknow = m._pintify_parfile(par_dict)
    assert len(pint_dict) == len((base_par.strip()).split('\n'))
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
    assert len(pint_dict['JUMP1']) == 2
    assert len(pint_dict['ECORR1']) == 2


def test_pintify_parfile_repeat_line_with_repeat_aliases():
    repeat_par = base_par + "T2EQUAD -fe 1 1 3\nT2EQUAD -fe 2 1 3"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(repeat_par))
    assert len(pint_dict['EQUAD1']) == 2
    assert "T2EQUAD" not in pint_dict.keys()


def test_pintify_parfile_repeat_line_mix_alises():
    repeat_par = base_par + "EQUAD -fe 1 1 3\nT2EQUAD -fe 2 1 3\nEQUAD -fe 4 2 2"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(repeat_par))
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(repeat_par), allow_name_mixing=True)
    # The list order is swaped since parse_parfile will pack EQUADs first and
    # then the pintify will add T2ECORR to the list.
    assert pint_dict['EQUAD1'] == ['-fe 1 1 3', '-fe 4 2 2', '-fe 2 1 3']
    assert "T2EQUAD" not in pint_dict.keys()


def test_parse_parfile_index_param():
    indexed_par = base_par + "DMX_0001 1 1 2\nDMX_0025 2 1 3"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(indexed_par))
    assert pint_dict['DMX_0001'] == ['1 1 2']
    assert pint_dict['DMX_0025'] == ['2 1 3']

def test_pintify_parfile_alises():
    aliases_par = base_par.replace("ELONG", "LAMBDA")
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(aliases_par))
    assert pint_dict['ELONG'] == ['0 1 3']
    assert len(unknow) == 0
    assert "LAMBDA" not in pint_dict.keys()


def test_pintify_parfile_nonrepeat_with_alise_repeating():
    aliases_par = base_par + "LAMBDA 2 1 3"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(aliases_par))


def test_pintify_parfile_unrecognize():
    wrong_par =  base_par + "UNKNOWN 2 1 1"
    m = ModelBuilder()
    pint_dict, original_name, unknow = m._pintify_parfile(StringIO(wrong_par))
    assert unknow == {'UNKNOWN': ["2 1 1"]}


def test_pintify_parfile_duplicate_binary():
    dup_par = base_par + "BINARY DD\nBINARY BT"
    m = ModelBuilder()
    with pytest.raises(TimingModelError):
        m._pintify_parfile(StringIO(dup_par))
