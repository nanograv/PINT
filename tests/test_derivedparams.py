import glob
import pytest
from pint.models import get_model
from pinttestdata import datadir


@pytest.mark.parametrize("parname", list(glob.glob(str(datadir) + "/*.par")))
def test_derived_params(parname):
    try:
        m = get_model(parname)
        readin = True
    except:
        readin = False
    if readin:
        out = m.get_derived_params()
