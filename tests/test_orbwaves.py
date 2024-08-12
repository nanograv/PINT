import os
import astropy.units as u
import pytest

import pint.models as tm
from pint import fitter, toa, simulation
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls
from pint.models import get_model, get_model_and_toas


def test_orbwaves_fit():
    m, t = get_model_and_toas(
        os.path.join(datadir, "J1048+2339_orbwaves.par"),
        os.path.join(datadir, "J1048+2339_3PC_fake.tim"),
    )

    f = fitter.WLSFitter(toas=t, model=m)

    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 29.2 and rms < 29.3

    f.fit_toas()
    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 21.1 and rms < 21.2


def test_invalid_parfiles():
    with pytest.raises(Exception):
        m1 = get_model(os.path.join(datadir, "J1048+2339_orbwaves_invalid1.par"))

    with pytest.raises(Exception):
        m2 = get_model(os.path.join(datadir, "J1048+2339_orbwaves_invalid2.par"))

    with pytest.raises(Exception):
        m3 = get_model(os.path.join(datadir, "J1048+2339_orbwaves_invalid3.par"))


if __name__ == "__main__":
    test_orbwaves_fit()
    test_invalid_parfiles()
