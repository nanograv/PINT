"""Standard observatories read in from observatories.json

These observatories are registered when this file is imported. As a result it
cannot be imported until TopoObs has successfully been imported.
"""

import json
import os

import pint.config
from pint.observatory.topo_obs import TopoObs
from pint.observatory import bipm_default

observatories_json = pint.config.runtimefile("observatories.json")

pint_env_var = "PINT_OBS_OVERRIDE"

__all__ = ["observatories_json", "read_observatories"]


def get_default_value(entry, key, default=None):
    """Return default value if key is not in entry.  Otherwise return entry[key]

    Parameters
    ----------
    entry : dict
    key : str
    default :

    Returns
    -------
        value for assignment
    """

    return default if not key in entry else entry[key]


# Default values for instantiating a new TopoObs object
# They will be overridden by any values set in the JSON file
default_keywords = {
    "tempo_code": None,
    "itoa_code": None,
    "aliases": None,
    "itrf_xyz": None,
    "clock_file": "",
    "clock_dir": "PINT",
    "clock_fmt": "tempo",
    "include_gps": True,
    "include_bipm": True,
    "bipm_version": bipm_default,
    "origin": None,
    "bogus_last_correction": False,
}


def read_observatories(filename=observatories_json, overwrite=False):
    """Read observatory definitions from JSON and create TopoObs objects, registering them

    Set `overwrite` to ``True`` if you want to re-read a file with updated definitions

    Parameters
    ----------
    filename : str, optional
    overwrite : bool, optional
        Whether a new instance of an existing observatory should overwrite the existing one.


    """
    # read in the JSON file
    observatories = json.load(open(filename))
    for obsname in observatories:
        keywords = {}
        for keyword in default_keywords:
            keywords[keyword] = get_default_value(
                observatories[obsname], keyword, default_keywords[keyword]
            )
        if overwrite:
            keywords["overwrite"] = True
        # create the object, which will also register it
        TopoObs(name=obsname, **keywords)


# read the observaatories, with the potential to override the filename from the environment variable
if pint_env_var in os.environ:
    read_observatories(os.environ[pint_env_var])
else:
    read_observatories()
