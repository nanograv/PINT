"""Test combine_design_matrices_by_param function"""

import astropy.units as u
import pytest
from pint.pint_matrix import DesignMatrixMaker, combine_design_matrices_by_param
from pint.config import examplefile
from pint.models import get_model_and_toas


@pytest.fixture()
def model_and_toas():
    parfile = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    timfile = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    return model, toas


def test_combine_design_matrices_by_param(model_and_toas):
    model, toas = model_and_toas
    dmm = DesignMatrixMaker("phase", u.Quantity(1, ""))
    dm1 = dmm(toas, model, ["F0"], offset=True)
    dm2 = dmm(toas, model, ["F1"], offset=True)

    with pytest.raises(ValueError):
        dm12 = combine_design_matrices_by_param(dm1, dm2)

    dm1_no_offset = dmm(toas, model, ["F0"], offset=False)
    dm12 = combine_design_matrices_by_param(dm1_no_offset, dm2)  # This should work.
