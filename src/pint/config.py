"""Functions related to PINT configuration."""

import importlib.resources
import os

__all__ = ["datadir", "examplefile", "runtimefile"]


# location of tim, par files installed via pkg_resources
def datadir() -> str:
    """Location of the PINT data (par and tim files)

    Returns
    -------
    str
        Directory of PINT data files
    """
    return os.path.join(importlib.resources.files("pint"), "data/")


def examplefile(filename: str) -> str:
    """Full path to the requested PINT example data file

    Parameters
    ----------
    filename : str

    Returns
    -------
    str
        Full path to the requested file

    Notes
    -----
    This is **not** for files needed at runtime. Those are located by :func:`pint.config.runtimefile`.
    This is for files needed for the example notebooks.
    """
    return os.path.join(importlib.resources.files("pint"), f"data/examples/{filename}")


def runtimefile(filename: str) -> str:
    """Full path to the requested PINT runtime (clock etc) data file

    Parameters
    ----------
    filename : str

    Returns
    -------
    str
        Full path to the requested file

    Notes
    -----
    This **is**  for files needed at runtime. Files needed for the example notebooks
    are found via :func:`pint.config.examplefile`.
    """
    return os.path.join(importlib.resources.files("pint"), f"data/runtime/{filename}")
