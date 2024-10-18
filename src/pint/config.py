"""Functions related to PINT configuration."""

import os
import importlib.resources

__all__ = ["datadir", "examplefile", "runtimefile"]


# location of tim, par files installed via pkg_resources
def datadir() -> str:
    """Location of the PINT data (par and tim files)

    Returns
    -------
    str
        Directory of PINT data files
    """
    return importlib.resources.path("pint", "data/")


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
    return importlib.resources.path("pint", "data", "examples", filename)


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
    return importlib.resources.path("pint", "data", "runtime", filename)
