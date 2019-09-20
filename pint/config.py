"""Functions related to PINT configuration."""
from __future__ import absolute_import, division, print_function

import os

from .extern import appdirs

# Values for appdirs calls
_app = "pint"
_auth = "pint"

# PINT install dir
_install_dir = os.path.abspath(os.path.dirname(__file__))


def datapath(fname):
    """Returns the full path to the requested data file.

    Will first search the appdirs user_data_dir (typically
    $HOME/.local/share/pint on linux) then the installed data files dir
    (__file__/datafiles).  If the file is not found, returns None.

    """

    # List of directories to search, in order
    search_dirs = [
        appdirs.user_data_dir(_app, _auth),
        os.path.join(_install_dir, "datafiles"),
    ]

    for d in search_dirs:
        full_fname = os.path.join(d, fname)
        if os.path.exists(full_fname):
            return full_fname

    raise ValueError("Unable to find {} in directories {}".format(fname, search_dirs))
