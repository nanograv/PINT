#!/usr/bin/env python
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy.utils.iers import IERS_A_URL

try:
    from urllib.error import HTTPError

    iers_a = IERS_A.open(IERS_A_URL)
    URL = IERS_A_URL
except HTTPError:
    try:
        from astropy.utils.iers import IERS_A_URL_MIRROR

        iers_a = IERS_A.open(IERS_A_URL_MIRROR)
        URL = IERS_A_URL_MIRROR
    except ImportError:
        URL = "NO_IERS_URL_A_OR_MIRROR_FOUND"

download_file(IERS_A_URL, cache=True)
