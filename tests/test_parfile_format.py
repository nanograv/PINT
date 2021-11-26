"""Verious of tests for the parfile format"""
import os
import pytest

from pint.io.parfile_format import ParfileFormat

@pytest.fixture
def test_format_config():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(file_path), 'src/pint/io')
    return os.path.join(file_path, 'parfile_format.yml')


@pytest.mark.parametrize('format', ['tempo', 'tempo2'])
def test_format_class_build(test_format_config, format):
    par_format = ParfileFormat(test_format_config, format)
    assert par_format.format_name == format
