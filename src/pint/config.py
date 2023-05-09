"""Functions related to PINT configuration."""

import os
import pkg_resources

__all__ = ["datadir", "examplefile", "runtimefile"]


# location of tim, par files installed via pkg_resources
def datadir():
    """Location of the PINT data (par and tim files)

    Returns
    -------
    str
        Directory of PINT data files
    """
    return pkg_resources.resource_filename(__name__, "data/")


def examplefile(filename):
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
    return pkg_resources.resource_filename(
        __name__, os.path.join("data/examples/", filename)
    )


def runtimefile(filename):
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
    return pkg_resources.resource_filename(
        __name__, os.path.join("data/runtime/", filename)
    )
