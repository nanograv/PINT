#! /usr/bin/env python
import os
import tempfile

import pytest
from io import StringIO

from pint.models.model_builder import (
    ModelBuilder,
    parse_parfile
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

def test_pintify_parfile():
    pass
