"""Standard observatories read in from observatories.json

These observatories are registered when this file is imported. 
"""

import json
import os
from pathlib import Path

import pint.config
import pint.observatory
from pint.observatory.topo_obs import TopoObs

observatories_json = pint.config.runtimefile("observatories.json")

pint_env_var = "PINT_OBS_OVERRIDE"

__all__ = ["observatories_json", "read_observatories"]


def read_observatories(filename=observatories_json, overwrite=False):
    """Read observatory definitions from JSON and create :class:`pint.observatory.topo_obs.TopoObs` objects, registering them

    Set `overwrite` to ``True`` if you want to re-read a file with updated definitions.
    If `overwrite` is ``False`` and you attempt to add an existing observatory, an exception is raised.

    Parameters
    ----------
    filename : str or file-like object, optional
    overwrite : bool, optional
        Whether a new instance of an existing observatory should overwrite the existing one.

    Raises
    ------
    ValueError
        If an attempt is made to add an existing observatory with ``overwrite=False``

    """
    # read in the JSON file
    if isinstance(filename, (str, Path)):
        f = open(filename, "r")
    elif hasattr(filename, "read"):
        f = filename
    contents = f.read()
    observatories = json.loads(contents)

    for obsname, obsdict in observatories.items():

        if overwrite:
            obsdict["overwrite"] = True
        # create the object, which will also register it
        TopoObs(name=obsname, **obsdict)


def read_observatories_from_usual_locations():
    """Clear observatory registry, and then re-read from the default JSON file as well as $PINT_OBS_OVERRIDE"""
    pint.observatory.Observatory.clear_registry()
    # read the observatories
    read_observatories()
    # potentially override any defined here
    if pint_env_var in os.environ:
        read_observatories(os.environ[pint_env_var], overwrite=True)


read_observatories_from_usual_locations()
