"""Tools for building chi-squared grids."""
import concurrent.futures
import copy
import logging
import multiprocessing
import os

import astropy.constants as const
import astropy.units as u
import numpy as np

import pint.utils

log = logging.getLogger(__name__)

__all__ = ["doonefit", "grid_chisq", "grid_chisq_derived", "plot_grid_chisq"]


def doonefit(ftr, parnames, parvalues):
    """Worker process that computes one fit with specified parameters fixed

    Parameters
    ----------
    ftr : pint.fitter.Fitter
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be 1D array of astropy.units.Quantity)
    """
    # Make a full copy of the fitter to work with
    myftr = copy.deepcopy(ftr)
    for parname, parvalue in zip(parnames, parvalues):
        # Freeze the  params we are going to grid over and set their values
        # All other unfrozen parameters will be fitted for at each grid point
        getattr(myftr.model, parname).frozen = True
        getattr(myftr.model, parname).quantity = parvalue
    myftr.fit_toas()
    return myftr.resids.chi2


def grid_chisq(
    ftr,
    parnames,
    parvalues,
    ncpu=None,
    executor=concurrent.futures.ProcessPoolExecutor,
    chunksize=1,
    printprogress=True,
):
    """Compute chisq over a grid of parameters

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be 1D array of astropy.units.Quantity)
    ncpu : int, optional
        Number of processes to use in parallel. Default is number of CPUs available.
        If `ncpu` is 1, then use single-threaded version
    executor : concurrent.futures.Executor
        Executor object to run multiple processes in parallel (if `ncpu` is None or >1)
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress for single processor only

    Returns
    -------
    np.ndarray : array of chisq values
    """
    if ncpu is None or ncpu > 1:
        if ncpu is None:
            # Use al available CPUs
            ncpu = multiprocessing.cpu_count()

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
    if ncpu > 1:
        with executor(max_workers=ncpu) as e:
            result = e.map(
                doonefit,
                (ftr,) * len(out[0].flatten()),
                (parnames,) * len(out[0].flatten()),
                list(zip(*[x.flatten() for x in out])),
                chunksize=chunksize,
            )
        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r
    else:
        it = np.ndindex(out[0].shape)
        for i in it:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[parnum][i]
            ftr.fit_toas()
            chi2[i] = ftr.resids.chi2
            if printprogress:
                print(".", end="")

        if printprogress:
            print("")

    # Restore saved model
    ftr.model = savemod
    return chi2


def grid_chisq_derived(
    ftr,
    parnames,
    parfuncs,
    gridvalues,
    ncpu=None,
    executor=concurrent.futures.ProcessPoolExecutor,
    chunksize=1,
    printprogress=True,
):
    """Compute chisq over a grid of derived parameters

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parfuncs : list
        List of functions to convert `gridvalues` to quantities accessed through `parnames`
    gridvalues : list
        List of underlying grid values to grid over (each should be 1D array of astropy.units.Quantity)
    ncpu : int, optional
        Number of processes to use in parallel. Default is number of CPUs available.
        If `ncpu` is 1, then use single-threaded version
    executor : concurrent.futures.Executor
        Executor object to run multiple processes in parallel (if `ncpu` is None or >1)
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress for single-processor only

    Returns
    -------
    np.ndarray : array of chisq values
    parvalues : list of np.ndarray
        Parameter values computed from `gridvalues` and `parfuncs`
    """

    if ncpu is None or ncpu > 1:
        if ncpu is None:
            # Use al available CPUs
            ncpu = multiprocessing.cpu_count()

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
    out = []
    # convert the gridded values to the actual parameter values
    for j in range(len(parfuncs)):
        out.append(parfuncs[j](*grid))

    if ncpu > 1:
        with executor(max_workers=ncpu) as e:
            result = e.map(
                doonefit,
                (ftr,) * len(out[0].flatten()),
                (parnames,) * len(out[0].flatten()),
                list(zip(*[x.flatten() for x in out])),
                chunksize=chunksize,
            )
        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r
    else:
        it = np.ndindex(grid[0].shape)
        for i in it:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[parnum][i]
            ftr.fit_toas()
            chi2[i] = ftr.resids.chi2
            if printprogress:
                print(".", end="")

        if printprogress:
            print("")

    # Restore saved model
    ftr.model = savemod
    return chi2, out


##############################
# WIP
##############################
def _grid_docolfit(ftr, par1_name, par1_values, parnames, parvalues):
    """Worker process that computes one row of the chisq grid"""
    chisq = np.zeros(len(par1_values))
    for jj, par1 in enumerate(par1_values):
        # Make a full copy of the fitter to work with
        myftr = copy.deepcopy(ftr)
        for parname, parvalue in zip(parnames, parvalues):
            getattr(myftr.model, parname).frozen = True
            getattr(myftr.model, parname).quantity = parvalue
        getattr(myftr.model, par1_name).frozen = True
        getattr(myftr.model, par1_name).quantity = par1
        chisq[jj] = myftr.fit_toas()
    return chisq


def grid_chisq_col(ftr, parnames, parvalues, ncpu=None):
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
    out = np.meshgrid(*parvalues[:-1])
    chi2 = np.zeros(out[0].shape + (len(parvalues[-1]),))
    print(out[0].shape)
    print(chi2.shape)
    print(np.meshgrid(*parvalues)[0].shape)
    nrep = len(out[0].flatten())
    results = pool.map(
        _grid_docolfit,
        (ftr,) * nrep,
        (parnames[-1],) * nrep,
        (parvalues[-1],) * nrep,
        (parnames[:-1],) * nrep,
        list(zip(*[x.flatten() for x in out])),
    )
    for i, j in enumerate(np.ndindex(out[0].shape)):
        chi2[j] = results[i]
    return chi2


##############################
# OLD
##############################
from multiprocessing import Process, Queue


def _grid_docol(ftr, par1_name, par1, par2_name, par2_grid, ii, q):
    """Worker process that computes one row of the chisq grid"""
    for jj, par2 in enumerate(par2_grid):
        # Make a full copy of the fitter to work with
        myftr = copy.deepcopy(ftr)
        # Freeze the two params we are going to grid over and set their values
        # All other unfrozen parameters will be fitted for at each grid point
        getattr(myftr.model, par1_name).frozen = True
        getattr(myftr.model, par2_name).frozen = True
        getattr(myftr.model, par1_name).quantity = par1
        getattr(myftr.model, par2_name).quantity = par2
        chisq = myftr.fit_toas()
        # print(".", end="")
        q.put([ii, jj, chisq])


def old_grid_chisq_mp(ftr, par1_name, par1_grid, par2_name, par2_grid, ncpu=None):
    """Compute chisq over a grid of two parameters, multiprocessing version
    Use Python's multiprocessing package to do a parallel computation of
    chisq over 2-D grid of parameters.
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
    ncpu : int, optional
        Number of processes to use in parallel. Default is number of CPUs available
    Returns
    -------
    array : 2-D array of chisq values with par1 varying in columns and par2 varying in rows
    """

    if ncpu is None:
        # Use al available CPUs
        ncpu = multiprocessing.cpu_count()

    # Instantiate a Queue for getting return values from the worker processes
    q = Queue()

    chi2 = np.zeros((len(par1_grid), len(par2_grid)))
    # First create all the processes and put them in a list
    processes = []
    # Want par1 on X-axis and par2 on y-axis
    for ii, par1 in enumerate(par1_grid):
        # ii indexes rows, now for have each column done by a different process
        proc = Process(
            target=_grid_docol, args=(ftr, par1_name, par1, par2_name, par2_grid, ii, q)
        )
        processes.append(proc)

    # Now consume the list of processes by starting up to ncpu processes at a time
    while len(processes):
        # Start up to ncpu processes
        started = []
        for cpunum in range(ncpu):
            if len(processes):
                proc = processes.pop()
                proc.start()
                started.append(proc)
        # Collect all the results from the started processes
        for proc in started * len(par2_grid):
            ii, jj, ch = q.get()
            # Array index here is rownum, colnum so translates to y, x
            chi2[jj, ii] = ch

        # Now join each of those that are done to close them out
        # This will be inefficient if the processes have large differences in runtime, since there
        # is a synchronization point. In this case it should not be a problem.
        for proc in started:
            proc.join()
            # print("|", end="")
        # print("")

    return chi2


def old_grid_chisq(ftr, par1_name, par1_grid, par2_name, par2_grid):
    """Compute chisq over a grid of two parameters, serial version
    Single-threaded computation of chisq over 2-D grid of parameters.
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
    Returns
    -------
    array : 2-D array of chisq values with par1 varying in columns and par2 varying in rows
    """

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the two params we are going to grid over
    getattr(ftr.model, par1_name).frozen = True
    getattr(ftr.model, par2_name).frozen = True

    # All other unfrozen parameters will be fitted for at each grid point

    chi2 = np.zeros((len(par1_grid), len(par2_grid)))
    # Want par1 on X-axis and par2 on y-axis
    for ii, par1 in enumerate(par1_grid):
        getattr(ftr.model, par1_name).quantity = par1
        for jj, par2 in enumerate(par2_grid):
            getattr(ftr.model, par2_name).quantity = par2
            # Array index here is rownum, colnum so translates to y, x
            chi2[jj, ii] = ftr.fit_toas()
            # print(".", end="")
        # print("")

    # Restore saved model
    ftr.model = savemod
    return chi2


##############################


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
