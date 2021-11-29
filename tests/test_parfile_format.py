"""Verious of tests for the parfile format"""
import os
import pytest
import yaml
import numpy as np

import astropy.units as u
from pint import ls
from pint.io.parfile_format import ParfileTranslator
import pint.models.parameter as p


def load_config(config_file):
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    return config


@pytest.fixture
def test_format_config():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(file_path), 'src/pint/io')
    return os.path.join(file_path, 'parfile_format.yml')

@pytest.fixture
def test_format_config_dict():
    config = {'test':{'TESTPARAM': {'to_name': 'NEWNAME',
                                    'to_value': 222.5,
                                    'to_type': 'int',
                                    'to_unit': 'km / s'},
                      'TESTPARAM2': {'to_value': 301.3}
                      }
             }
    return config

@pytest.fixture
def example_translator(test_format_config_dict):
    return ParfileTranslator(test_format_config_dict, 'test')


@pytest.mark.parametrize('format', ['tempo', 'tempo2'])
def test_translate_class_build_from_file(test_format_config, format):
    par_trans = ParfileTranslator(test_format_config, format)
    assert par_trans.format_name == format


def test_unkown_config_type():
    with pytest.raises(ValueError, match='Unacceptable'):
        par_trans = ParfileTranslator(['test'], 'test')


def test_unknow_format(test_format_config_dict):
    with pytest.raises(KeyError, match=r"does not have the format"):
        par_trans = ParfileTranslator(test_format_config_dict, 'test2')


def test_get_param_entry_raises(test_format_config):
    par_trans = ParfileTranslator(test_format_config, 'tempo2')
    with pytest.raises(KeyError, match=r"Parameter .* is not in the .* translator"):
        par_trans._get_param_entry("WRONGNAME")


def test_name_change(test_format_config):
    par_trans = ParfileTranslator(test_format_config, 'tempo2')
    pint_name = 'A1DOT'
    pint_param = p.floatParameter(name=pint_name, value=0.1, units=ls / u.s)
    p_entry = par_trans._get_param_entry(pint_name)
    new_par = par_trans.to_name(pint_param, p_entry)
    raw_config = load_config(test_format_config)
    assert new_par.name == raw_config['tempo2'][pint_name]['to_name']


def test_value_change(example_translator, test_format_config_dict):
    pint_name = 'TESTPARAM'
    pint_param = p.floatParameter(name=pint_name, value=233.45, units=ls / u.s)
    p_entry = example_translator._get_param_entry(pint_name)
    new_param = example_translator.to_value(pint_param, p_entry)
    assert new_param.value == test_format_config_dict['test'][pint_name]['to_value']


def test_unit_change(example_translator, test_format_config_dict):
    pint_name = 'TESTPARAM'
    pint_param = p.floatParameter(name=pint_name, value=233.45, units=ls / u.s)
    p_entry = example_translator._get_param_entry(pint_name)
    new_param = example_translator.to_unit(pint_param, p_entry)
    assert new_param.quantity.unit == u.Unit(test_format_config_dict['test'][pint_name]['to_unit'])


def test_type_change(example_translator, test_format_config_dict):
    pint_name = 'TESTPARAM'
    pint_param = p.floatParameter(name=pint_name, value=233.45, units=ls / u.s)
    p_entry = example_translator._get_param_entry(pint_name)
    new_param = example_translator.to_type(pint_param, p_entry)
    assert new_param.quantity.dtype == test_format_config_dict['test'][pint_name]['to_type']
    assert np.modf(new_param.value)[0] == 0


def test_all_change(example_translator, test_format_config_dict):
    pint_name = 'TESTPARAM'
    pint_param = p.floatParameter(name=pint_name, value=233.45, units=ls / u.s)
    new_param = example_translator.to_format(pint_param)
    assert new_param.value == test_format_config_dict['test'][pint_name]['to_value']
    assert new_param.quantity.unit == u.Unit(test_format_config_dict['test'][pint_name]['to_unit'])
    assert new_param.quantity.dtype == test_format_config_dict['test'][pint_name]['to_type']
    assert new_param.name == test_format_config_dict['test'][pint_name]['to_name']
