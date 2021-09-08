import astropy.units as u
import pytest
from pint.pintk.plk import PlkWidget
from tkinter import Frame


@pytest.fixture
def widget():
    return Frame.__new__(PlkWidget)


@pytest.mark.parametrize(
    "in_min,in_max,out_min,out_max",
    [
        (0 * u.us, 190 * u.us, 0 * u.us, 190 * u.us),
        (0 * u.us, 201 * u.us, 0 * u.ms, 0.201 * u.ms),
        (0 * u.us, 200001 * u.us, 0 * u.s, 0.200001 * u.s),
        (0 * u.ms, 2.1 * u.ms, 0 * u.ms, 2.1 * u.ms),
        (0 * u.ms, 201 * u.ms, 0 * u.s, 0.201 * u.s),
        (0 * u.ms, 0.19 * u.ms, 0 * u.us, 190 * u.us),
        (0 * u.s, 0.21 * u.s, 0 * u.s, 0.21 * u.s),
        (0 * u.s, 0.19 * u.s, 0 * u.ms, 190 * u.ms),
        (0 * u.s, 0.00019 * u.s, 0 * u.us, 190 * u.us),
    ],
)
def test_determine_yaxis_units(in_min, in_max, out_min, out_max, widget):
    newmin, newmax = widget.determine_yaxis_units(in_min, in_max)

    assert newmin.value == pytest.approx(out_min.value)
    assert newmin.unit == out_min.unit
    assert newmax.value == pytest.approx(out_max.value)
    assert newmax.unit == out_max.unit
