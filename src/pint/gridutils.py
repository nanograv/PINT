"""Tools for building chi-squared grids."""
import copy
import multiprocessing
import os
import logging

# from multiprocessing import Process, Queue

import astropy.constants as const
import astropy.units as u
import numpy as np

import pint.utils

log = logging.getLogger(__name__)

__all__ = ["grid_chisq", "grid_chisq_derived", "plot_grid_chisq"]


def _grid_docol(ftr, par1_name, par1, par2_name, par2_grid):
    """Worker process that computes one row of the chisq grid"""
    chisq = np.zeros(len(par2_grid))
    for jj, par2 in enumerate(par2_grid):
        # Make a full copy of the fitter to work with
        myftr = copy.deepcopy(ftr)
        # Freeze the two params we are going to grid over and set their values
        # All other unfrozen parameters will be fitted for at each grid point
        getattr(myftr.model, par1_name).frozen = True
        getattr(myftr.model, par2_name).frozen = True
        getattr(myftr.model, par1_name).quantity = par1
        getattr(myftr.model, par2_name).quantity = par2
        chisq[jj] = myftr.fit_toas()
    return chisq


def _grid_doonefit(ftr, parnames, parvalues):
    """Worker process that computes one fit for the chisq grid"""
    # Make a full copy of the fitter to work with
    myftr = copy.deepcopy(ftr)
    for parname, parvalue in zip(parnames, parvalues):
        # Freeze the  params we are going to grid over and set their values
        # All other unfrozen parameters will be fitted for at each grid point
        getattr(myftr.model, parname).frozen = True
        getattr(myftr.model, parname).quantity = parvalue
    return myftr.fit_toas()


def grid_chisq(ftr, parnames, parvalues, ncpu=None, printprogress=True):
    """Compute chisq over a grid of two parameters, serial version

    Computation of chisq over 2-D grid of parameters.

    Parameters
    ----------
    ftr
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be array of Quantity)
    ncpu : int, optional
        Number of processes to use in parallel. Default is number of CPUs available.
        If `ncpu` is 1, then use single-threaded version
    printprogress : bool, optional
        Print indications of progress

    Returns
    -------
    array : 2-D array of chisq values

    Notes
    -----
    Uses pathos's multiprocessing package to do a parallel computation of
    chisq over 2-D grid of parameters.  Need this instead of stock python because
    of unpicklable objects in python >=3.8
    """
    if ncpu is None or ncpu > 1:
        try:
            from pathos.multiprocessing import ProcessingPool as Pool

            if ncpu is None:
                # Use al available CPUs
                ncpu = multiprocessing.cpu_count()
            pool = Pool(ncpu)
        except ImportError:
            log.warning("pathos module not found; using single processor version")
            ncpu = 1

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the  params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    # All other unfrozen parameters will be fitted for at each grid point
    out = np.meshgrid(*parvalues)
    chi2 = np.zeros(out[0].shape)
    it = np.nditer(out[0], flags=["multi_index"])
    if ncpu > 1:
        # First create all the processes and put them in a pool
        results = pool.map(
            _grid_doonefit,
            (ftr,) * len(out[0].flatten()),
            (parnames,) * len(out[0].flatten()),
            list(zip(*[x.flatten() for x in out])),
        )
        for j, x in enumerate(it):
            chi2[it.multi_index] = results[j]
    else:
        for x in it:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[parnum][it.multi_index]
            chi2[it.multi_index] = ftr.fit_toas()
            if printprogress:
                print(".", end="")

        if printprogress:
            print("")

    # Restore saved model
    ftr.model = savemod
    return chi2


def grid_chisq_derived(
    ftr, parnames, parfuncs, gridvalues, printprogress=True,
):
    """Compute chisq over a grid of two parameters, serial version

    Single-threaded computation of chisq over 2-D grid of parameters.

    Parameters
    ----------
    ftr
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parfuncs : list
        List of functions to convert `gridvalues` to quantities accessed through `parnames`
    gridvalues : list
        List of underlying grid values to grid over (each should be array of Quantity)
    printprogress : bool, optional
        Print indications of progress

    Returns
    -------
    array : 2-D array of chisq values
    """

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    # All other unfrozen parameters will be fitted for at each grid point
    grid = np.meshgrid(*gridvalues)
    chi2 = np.zeros(grid[0].shape)
    it = np.nditer(grid[0], flags=["multi_index"])
    out = []
    # convert the gridded values to the actual parameter values
    for j in range(len(gridvalues)):
        out.append(parfuncs[j](*grid))

    for x in it:
        for parnum, parname in enumerate(parnames):
            getattr(ftr.model, parname).quantity = out[parnum][it.multi_index]
        chi2[it.multi_index] = ftr.fit_toas()
        if printprogress:
            print(".", end="")

    if printprogress:
        print("")

    # Restore saved model
    ftr.model = savemod
    return chi2, out


def plot_grid_chisq(
    par1_name, par1_grid, par2_name, par2_grid, chi2, title="Chisq Heatmap"
):
    """Plot results of chi2 grid

    Parameters
    ----------
    ftr
        The base fitter to use.
    par1_name : str
        Name of the first parameter to grid over
    par1_grid : array, Quantity
        Array of par1 values for column of the output matrix
    par2_name : str
        Name of the second parameter to grid over
    par2_grid : array, Quantity
        Array of par2 values for column of the output matrix
    title : str, optional
        Title for plot
    """

    import matplotlib.pyplot as plt

    # Compute chi2 difference from minimum
    delchi2 = chi2 - chi2.min()
    fig, ax = plt.subplots(figsize=(9, 9))
    delta_par1 = (par1_grid[1] - par1_grid[0]).value
    delta_par2 = (par2_grid[1] - par2_grid[0]).value
    ax.imshow(
        delchi2,
        origin="lower",
        extent=(
            par1_grid[0].value - delta_par1 / 2,
            par1_grid[-1].value + delta_par1 / 2,
            par2_grid[0] - delta_par2 / 2,
            par2_grid[-1] + delta_par2 / 2,
        ),
        aspect="auto",
        cmap="Blues_r",
        interpolation="bicubic",
        vmin=0,
        vmax=10,
    )
    levels = np.arange(4) + 1
    ax.contour(
        delchi2,
        levels=levels,
        colors="red",
        extent=(
            par1_grid[0].value - delta_par1 / 2,
            par1_grid[-1].value + delta_par1 / 2,
            par2_grid[0] - delta_par2 / 2,
            par2_grid[-1] + delta_par2 / 2,
        ),
    )
    ax.set_xlabel(par1_name)
    ax.set_ylabel(par2_name)
    ax.grid(True)
    ax.set_title(title)
    return
