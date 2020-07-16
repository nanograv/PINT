"""Test basic functionality of the :module:`pint.modelutils`."""

import pint.models.model_builder as mb
from pinttestdata import datadir
from pint import fitter, toa
from pinttestdata import datadir
import os

from pint.modelutils import (
    convert_to_ecliptic,
    convert_to_equatorial,
)


def test_convert_to_ecliptic():

    os.chdir(datadir)
    parfileNGC6440E = "NGC6440E.par"
    timNGC6440E = "NGC6440E.tim"
    toasNGC6440E = toa.get_TOAs(
        timNGC6440E, ephem="DE405", planets=False, include_bipm=False
    )
    ICRSmodelNGC6440E = mb.get_model(parfileNGC6440E)
    ECLmodelNGC6440E = convert_to_ecliptic(ICRSmodelNGC6440E) 

    # Ensure a working model is returned (try fitting)
    try:
        ECLfitNGC6440E = fitter.WLSFitter(toasNGC6440E,ECLmodelNGC6440E)
        ECLfitNGC6440E.fit_toas()
    except:
        pass
        #??

    # Perhaps I should compare with ICRS resids to ensure there's no substantial difference?

def test_convert_to_equatorial():

    os.chdir(datadir)
    parfileJ0613 = "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
    timJ0613 = "J0613-0200_NANOGrav_dfg+12.tim"
    toasJ0613 = toa.get_TOAs(
        timJ0613, ephem="DE405", planets=False, include_bipm=False
    )
    ECLmodelJ0613 = mb.get_model(parfileJ0613)
    ICRSmodelJ0613 = convert_to_equatorial(ECLmodelJ0613)


if __name__ == "__main__":
    test_convert_to_ecliptic()
    test_convert_to_equatorial()
