"""Tests of PINT generic binary model """

from os.path import basename, join
from glob import glob
import pytest

from pint.models.model_builder import get_model
from pint.models.timing_model import MissingParameter
from utils import verify_stand_alone_binary_parameter_updates
from pinttestdata import datadir


bad_trouble = ["J1923+2515_NANOGrav_9yv1.gls.par", "J1744-1134.basic.ecliptic.par"]


@pytest.mark.parametrize("parfile", glob(join(datadir, "*.par")))
def test_if_stand_alone_binary_model_get_updated_from_PINT_model(parfile):
    if basename(parfile) in bad_trouble:
        pytest.skip("This parfile is unclear")
    try:
        m = get_model(parfile)
    except (ValueError, IOError, MissingParameter) as e:
        pytest.skip(f"Existing code raised an exception {e}")
    verify_stand_alone_binary_parameter_updates(m)
