from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import multiprocessing
from multiprocessing import Process, Queue

import astropy.units as u
import numpy as np
from astropy import log
import astropy.constants as const
import pint.utils

__all__ = ["grid_chisq", "grid_chisq_mp", "plot_grid_chisq"]


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
        print(".", end="")
        q.put([ii, jj, chisq])


def grid_chisq_mp(ftr, par1_name, par1_grid, par2_name, par2_grid, ncpu=None):
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
            print("|", end="")
        print("")

    return chi2


def grid_chisq(ftr, par1_name, par1_grid, par2_name, par2_grid):
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
    savemod = self.model
    gridmod = copy.deepcopy(self.model)
    self.model = gridmod

    # Freeze the two params we are going to grid over
    getattr(self.model, par1_name).frozen = True
    getattr(self.model, par2_name).frozen = True

    # All other unfrozen parameters will be fitted for at each grid point

    chi2 = np.zeros((len(par1_grid), len(par2_grid)))
    # Want par1 on X-axis and par2 on y-axis
    for ii, par1 in enumerate(par1_grid):
        getattr(self.model, par1_name).quantity = par1
        for jj, par2 in enumerate(par2_grid):
            getattr(self.model, par2_name).quantity = par2
            # Array index here is rownum, colnum so translates to y, x
            chi2[jj, ii] = self.fit_toas()
            print(".", end="")
        print("")

    # Restore saved model
    self.model = savemod
    return chi2


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
