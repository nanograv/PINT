from collections import deque
from io import StringIO
import os
import pytest

import astropy.units as u
import pint.models as tm
from pint import fitter, toa, simulation
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls
from pint.models import get_model, get_model_and_toas


def test_orbwaves_fit():
    m, t = get_model_and_toas(
        datadir / "J1048+2339_orbwaves.par", datadir / "J1048+2339_3PC_fake.tim"
    )

    f = fitter.WLSFitter(toas=t, model=m)

    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 29.2 and rms < 29.3

    f.fit_toas()
    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 21.1 and rms < 21.2


def test_orbwaves_DD_fit():
    m, t = get_model_and_toas(
        datadir / "J1048+2339_orbwaves_DD.par", datadir / "J1048+2339_3PC_fake.tim"
    )

    f = fitter.WLSFitter(toas=t, model=m)

    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 29.2 and rms < 29.3

    f.fit_toas()
    rms = f.resids.rms_weighted().to_value("us")

    assert rms > 21.1 and rms < 21.2


def test_invalid_parfiles():
    parlines = open(datadir / "J1048+2339_orbwaves.par").readlines()

    def _delete_line(labels):
        lines = deque()
        for line in parlines:
            label = line.split()[0]
            if label in labels:
                continue
            lines.append(line)
        return StringIO(lines)

    deleted_lines = [["ORBWAVEC0", "ORBWAVES0"], ["ORBWAVEC3"], ["ORBWAVES4"]]

    for lines in deleted_lines:
        with pytest.raises(Exception):
            m = get_model(_delete_line(lines))


if __name__ == "__main__":
    test_orbwaves_fit()
    test_invalid_parfiles()
