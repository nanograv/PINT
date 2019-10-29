#!/usr/bin/env python
from __future__ import print_function, division

import sys

from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy.utils.iers import IERS_A_URL

from astropy import log

if sys.version_info.major < 3:
    try:
        iers_a = IERS_A.open(IERS_A_URL)
        URL = IERS_A_URL
    except:
        URL = "NO_IERS_URL_A_OR_MIRROR_FOUND"
    try:
        download_file(URL, cache=True)
    except:
        log.error("IERS A file download failed. This may cause problems.")
else:
    from urllib.error import HTTPError, URLError
    from http.client import RemoteDisconnected

    try:
        iers_a = IERS_A.open(IERS_A_URL)
        URL = IERS_A_URL
    except (HTTPError, URLError, RemoteDisconnected):
        try:
            from astropy.utils.iers import IERS_A_URL_MIRROR

            iers_a = IERS_A.open(IERS_A_URL_MIRROR)
            URL = IERS_A_URL_MIRROR
        except (ImportError, URLError):
            URL = "NO_IERS_URL_A_OR_MIRROR_FOUND"

    try:
        download_file(URL, cache=True)
    except (HTTPError, ValueError):
        log.error("IERS A file download failed. This may cause problems.")
