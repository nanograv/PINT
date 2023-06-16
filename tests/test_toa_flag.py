"""Various tests to assess the performance of TOA get_flag_value."""


import os
import pytest
import io
import numpy as np
import pytest

import pint.toa as toa
from pinttestdata import datadir


class TestToaFlag:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        cls.tim = "B1855+09_NANOGrav_dfg+12.tim"
        cls.toas = toa.get_TOAs(
            os.path.join(datadir, cls.tim),
            ephem="DE405",
            planets=False,
            include_bipm=False,
        )

    def test_flag_value_float(self):
        flag_value, valid = self.toas.get_flag_value("to")
        assert len(flag_value) == self.toas.ntoas
        for v in set(flag_value):
            assert v in {-7.89e-07, -8.39e-07, None}
        count1 = {}
        for flag_dict in self.toas.get_flags():
            fv1 = flag_dict.get("to")
            if fv1 in list(count1.keys()):
                count1[fv1] += 1
            else:
                count1[fv1] = 1

        count2 = {}
        for fv2 in flag_value:
            if fv2 in list(count2.keys()):
                count2[fv2] += 1
            else:
                count2[fv2] = 1
        assert count1 == count2

    def test_flag_value_fill(self):
        flag_value, valid = self.toas.get_flag_value("to", fill_value=-9999)
        for v in set(flag_value):
            assert v in {-7.89e-07, -8.39e-07, -9999}

    def test_flag_value_str(self):
        flag_value, valid = self.toas.get_flag_value("be")
        assert len(flag_value) == self.toas.ntoas
        for v in set(flag_value):
            assert v in {"ASP", "PUPPI"}


s = """FORMAT 1
MODE 1
uwl_191009_015952.0000_0000.rf.pTF 1102.00000000 58765.09374976854768846 22.04600 pks
uwl_191010_000912.0000_0000.rf.pTF 1102.00000000 58766.02430580012498496 23.74000 pks
uwl_191012_013412.0000_0000.rf.pTF 1102.00000000 58768.08628513784793768 16.29100 pks
uwl_191013_013912.0000_0000.rf.pTF 1102.00000000 58769.08628494704621659 18.55400 pks
uwl_191014_011651.0000_0000.rf.pTF 1102.00000000 58770.06840417512788122 21.57900 pks
uwl_191015_063631.0000_0000.rf.pTF 1102.00000000 58771.28385332518061546 23.07400 pks
uwl_191029_231442.0000_0000.rf.pTF 1102.00000000 58785.98420216310909581 18.68600 pks
uwl_191111_020722.0000_0000.rf.pTF 1102.00000000 58798.10659743353399165 17.39200 pks
uwl_191128_000422.0000_0000.rf.pTF 1102.00000000 58815.01493200936268835 20.28800 pks
uwl_191207_024252.0000_0000.rf.pTF 1102.00000000 58824.11892727732292840 25.10700 pks
"""


def test_flag_set_scalar():
    t = toa.get_TOAs(io.StringIO(s))
    t["test"] = str(1)
    for i in range(t.ntoas):
        assert t["flags"][i]["test"] == "1"


def test_flag_set_array():
    t = toa.get_TOAs(io.StringIO(s))
    t["test"] = np.arange(len(t))
    for i in range(t.ntoas):
        assert float(t["test", i]) == i


def test_flag_set_wrongarray():
    t = toa.get_TOAs(io.StringIO(s))
    with pytest.raises(ValueError):
        t["test"] = np.arange(2 * len(t))


def test_flag_set_partialarray():
    t = toa.get_TOAs(io.StringIO(s))
    # only set a few elements and make sure that only those are set
    t[:2, "test"] = "1"
    for i in range(t.ntoas):
        if i < 2:
            assert float(t["test", i]) == 1
        else:
            assert "test" not in t["flags"][i]


def test_flag_delete():
    t = toa.get_TOAs(io.StringIO(s))
    t["test"] = "1"
    t["test"] = ""
    for i in range(t.ntoas):
        assert "test" not in t["flags"][i]


def test_flag_partialdelete():
    t = toa.get_TOAs(io.StringIO(s))
    t["test"] = "1"
    t[:2, "test"] = ""
    for i in range(t.ntoas):
        if i < 2:
            assert "test" not in t["flags"][i]
        else:
            assert float(t["test", i]) == 1
