#!/usr/bin/env python
from __future__ import print_function, division
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy.utils.iers import IERS_A_URL
from urllib.error import HTTPError, URLError
from astropy import log

try:

    iers_a = IERS_A.open(IERS_A_URL)
    URL = IERS_A_URL
except HTTPError:
    try:
        from astropy.utils.iers import IERS_A_URL_MIRROR

        iers_a = IERS_A.open(IERS_A_URL_MIRROR)
        URL = IERS_A_URL_MIRROR
    except (ImportError, URLError):
        URL = "NO_IERS_URL_A_OR_MIRROR_FOUND"

try:
    download_file(IERS_A_URL, cache=True)
except HTTPError:
    log.warning("IERS A file download failed. This may cause problems.")
