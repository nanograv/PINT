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


def read_observatories(filename=observatories_json, overwrite=False):
    """Read observatory definitions from JSON and create :class:`pint.observatory.topo_obs.TopoObs` objects, registering them

    Set `overwrite` to ``True`` if you want to re-read a file with updated definitions.
    If `overwrite` is ``False`` and you attempt to add an existing observatory, an exception is raised.

    Parameters
    ----------
    filename : str, optional
    overwrite : bool, optional
        Whether a new instance of an existing observatory should overwrite the existing one.

    """
    # read in the JSON file
    observatories = json.load(open(filename))
    for obsname, obsdict in observatories.items():

        if overwrite:
            obsdict["overwrite"] = True
        # create the object, which will also register it
        TopoObs(name=obsname, **obsdict)


# read the observatories
read_observatories()
# potentially override any defined here
if pint_env_var in os.environ:
    read_observatories(os.environ[pint_env_var], overwrite=True)
