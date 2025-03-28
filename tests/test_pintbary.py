import sys
import pytest

import numpy as np
from io import StringIO

import pint.scripts.pintbary as pintbary


class TestPintBary:
    def test_result(self):
        saved_stdout, sys.stdout = sys.stdout, StringIO("_")
        cmd = "56000.0 --ra 12h22m33.2s --dec 19d21m44.2s --obs gbt --ephem DE405"
        pintbary.main(cmd.split())
        v = sys.stdout.getvalue()
        # Check that last value printed is the barycentered time
        assert np.isclose(float(v.split()[-1]), 56000.0061691189)
        sys.stdout = saved_stdout


# Below are some examples of types of tests
#     def test_main_returns_nonzero_on_error(self):
#         self.assertNotEqual(plus_one.main(['test']), 0)

#     def test_get_number_returns_second_list_element_as_integer(self):
#         self.assertEquals(plus_one.get_number(['anything', 42]), 42)

#     def test_get_number_raises_value_error_with_string(self):
#         self.assertRaises(ValueError, plus_one.get_number, ['something',
#                                                             'forty-two'])

#     def test_get_number_raises_index_error_with_too_few_arguments(self):
#         self.assertRaises(IndexError, plus_one.get_number, ['nothing'])

#     def test_plus_one_adds_one_to_number(self):
#         self.assertEquals(plus_one.plus_one(1), 2)

#    def test_output_prints_input(self):
#        saved_stdout, pintbary.sys.stdout = pintbary.sys.stdout, StringIO('_')
#        pintbary.output('some_text')
#        self.assertEquals(plus_one.sys.stdout.getvalue(), 'some_text\n')
#        plus_one.sys.stdout = saved_stdout
