"""Tools for building chi-squared grids."""
import concurrent.futures
import copy
import multiprocessing
import os
import subprocess
import functools

import astropy.constants as const
import astropy.units as u
import numpy as np
from loguru import logger as log

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from astropy.utils.console import ProgressBar

from pint import fitter
import pint.utils


__all__ = ["doonefit", "grid_chisq", "grid_chisq_derived", "plot_grid_chisq"]


def hostinfo():
    return subprocess.check_output("uname -a", shell=True)


class WrappedFitter:
    """Worker class to compute one fit with specified parameters fixed but passing other parameters to fit_toas()"""

    def __init__(self, ftr, **fitargs):
        """Worker class to computes one fit with specified parameters fixed

        Parameters
        ----------
        ftr : pint.fitter.Fitter
        fitargs :
            additional arguments pass to fit_toas()
        """
        self.ftr = ftr
        self.fitargs = fitargs

    def doonefit(self, parnames, parvalues, extraparnames=[]):
        """Worker process that computes one fit with specified parameters fixed

        Parameters
        ----------
        parnames : list
            Names of the parameters to grid over
        parvalues : list
            List of parameter values to grid over (each should be 1D array of astropy.units.Quantity)
        extraparnames : list, optional
            Names of other parameters to return

        Returns
        -------
        chi2 : float
        extraparvalues : list
        """
        # Make a full copy of the fitter to work with
        myftr = copy.deepcopy(self.ftr)
        parstrings = []
        for parname, parvalue in zip(parnames, parvalues):
            # Freeze the  params we are going to grid over and set their values
            # All other unfrozen parameters will be fitted for at each grid point
            getattr(myftr.model, parname).frozen = True
            getattr(myftr.model, parname).quantity = parvalue
            parstrings.append(f"{parname} = {parvalue}")
            log.debug(f"Running for {','.join(parstrings)} on {hostinfo()}")
        try:
            myftr.fit_toas(**self.fitargs)
            chi2 = myftr.resids.chi2
        except (fitter.InvalidModelParameters, fitter.StepProblem):
            log.warning(
                f"Fit may not be converged for {','.join(parstrings)}, but returning anyway"
            )
            chi2 = myftr.resids.chi2
        except fitter.MaxiterReached:
            log.warning(
                f"Max iterations reached for {','.join(parstrings)}: returning NaN"
            )
            chi2 = np.NaN
        except Exception as e:
            log.warning(
                f"Unexpected exception {e} for {','.join(parstrings)}: returning NaN"
            )
            chi2 = np.NaN
        log.debug(
            f"Computed chi^2={myftr.resids.chi2} for {','.join(parstrings)} on {hostinfo()}"
        )
        extraparvalues = []
        for extrapar in extraparnames:
            extraparvalues.append(getattr(myftr.model, extrapar).quantity)
        return chi2, extraparvalues


def doonefit(ftr, parnames, parvalues, extraparnames=[]):
    """Worker process that computes one fit with specified parameters fixed

    Parameters
    ----------
    ftr : pint.fitter.Fitter
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be 1D array of astropy.units.Quantity)
    extraparnames : list, optional
        Names of other parameters to return

    Returns
    -------
    chi2 : float
    extraparvalues : list
    """
    # Make a full copy of the fitter to work with
    myftr = copy.deepcopy(ftr)
    parstrings = []
    for parname, parvalue in zip(parnames, parvalues):
        # Freeze the  params we are going to grid over and set their values
        # All other unfrozen parameters will be fitted for at each grid point
        getattr(myftr.model, parname).frozen = True
        getattr(myftr.model, parname).quantity = parvalue
        parstrings.append(f"{parname} = {parvalue}")
    log.debug(f"Running for {','.join(parstrings)} on {hostinfo()} at {log.name}")
    try:
        myftr.fit_toas()
    except (fitter.InvalidModelParameters, fitter.StepProblem):
        log.warning(
            f"Fit may not be converged for {','.join(parstrings)}, but returning anyway"
        )
    except fitter.MaxiterReached:
        log.warning(f"Max iterations reached for {','.join(parstrings)}: returning NaN")
        return np.NaN
    log.debug(
        f"Computed chi^2={myftr.resids.chi2} for {','.join(parstrings)} on {hostinfo()}"
    )
    extraparvalues = []
    for extrapar in extraparnames:
        extraparvalues.append(getattr(myftr.model, extrapar).quantity)

    return myftr.resids.chi2, extraparvalues


def grid_chisq(
    ftr,
    parnames,
    parvalues,
    extraparnames=[],
    executor=None,
    ncpu=None,
    chunksize=1,
    printprogress=True,
    **fitargs,
):
    """Compute chisq over a grid of parameters

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be 1D array of :class:`astropy.units.Quantity`)
    extraparnames : list, optional
        Names of other parameters to return
    executor : concurrent.futures.Executor or None, optional
        Executor object to run multiple processes in parallel
        If None, will use default :class:`concurrent.futures.ProcessPoolExecutor`, unless overridden by ``ncpu=1``
    ncpu : int, optional
        If an existing Executor is not supplied, one will be created with this number of workers.
        If 1, will run single-processor version
        If None, will use :func:`multiprocessing.cpu_count`
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress (requires :mod:`tqdm` for `ncpu`>1)
    fitargs :
        additional arguments pass to fit_toas()

    Returns
    -------
    np.ndarray : array of chisq values
    extraout : dict of np.ndarray
        Parameter values computed at each grid point for `extraparnames`

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.config
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model, get_model_and_toas
    # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    # We'll do something like 3-sigma around the best-fit values of  F0
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,25)
    >>> chi2_F0,_ = pint.gridutils.grid_chisq(f, ("F0",), (F0,))

    Notes
    -----
    By default, it will create :class:`~concurrent.futures.ProcessPoolExecutor`
    with ``max_workers`` equal to the desired number of cpus.
    However, if you are running this as a script you may need something like::

        import multiprocessing

        if __name__ == "__main__":
            multiprocessing.freeze_support()
            ...
            grid_chisq(...)

    If an instantiated :class:`~concurrent.futures.Executor` is passed instead, it will be used as-is.

    The behavior for different combinations of `executor` and `ncpu` is:
    +-----------------+--------+------------------------+
    | `executor`      | `ncpu` | result                 |
    +=================+========+========================+
    | existing object | any    | uses existing executor |
    +-----------------+--------+------------------------+
    | None	      | 1      | uses single-processor  |
    +-----------------+--------+------------------------+
    | None	      | None   | creates default        |
    |                 |        | executor with          |
    |                 |        | ``cpu_count`` workers  |
    +-----------------+--------+------------------------+
    | None	      | >1     | creates default        |
    |                 |        | executor with desired  |
    |                 |        | number of workers      |
    +-----------------+--------+------------------------+

    Other ``Executors`` can be found for different computing environments:
    * [1]_ for MPI
    * [2]_ for SLURM or Condor

    .. [1] https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor
    .. [2] https://github.com/sampsyo/clusterfutures
    """
    if isinstance(executor, concurrent.futures.Executor):
        # the executor has already been created
        executor = executor
    elif executor is None and (ncpu is None or ncpu > 1):
        # make the default type of Executor
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the  params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    wftr = WrappedFitter(ftr, **fitargs)

    # All other unfrozen parameters will be fitted for at each grid point
    out = np.meshgrid(*parvalues)
    chi2 = np.zeros(out[0].shape)

    extraout = {}
    for extrapar in extraparnames:
        extraout[extrapar] = (
            np.zeros(out[0].shape, dtype=getattr(ftr.model, extrapar).quantity.dtype)
            * getattr(ftr.model, extrapar).quantity.unit
        )

    # at this point, if the executor is None then run single-processor version
    if executor is not None:
        with executor as e:
            if printprogress and tqdm is not None:
                result = list(
                    tqdm(
                        e.map(
                            wftr.doonefit,
                            (parnames,) * len(out[0].flatten()),
                            list(zip(*[x.flatten() for x in out])),
                            (extraparnames,) * len(out[0].flatten()),
                            chunksize=chunksize,
                        ),
                        total=len(out[0].flatten()),
                        ascii=True,
                    )
                )
            else:
                result = e.map(
                    wftr.doonefit,
                    (parnames,) * len(out[0].flatten()),
                    list(zip(*[x.flatten() for x in out])),
                    (extraparnames,) * len(out[0].flatten()),
                    chunksize=chunksize,
                )
        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r[0]
            for extrapar, extraparvalue in zip(extraparnames, r[1]):
                extraout[extrapar][i] = extraparvalue
    else:
        indices = list(np.ndindex(out[0].shape))
        if printprogress:
            if tqdm is not None:
                indices = tqdm(indices, ascii=True)
            else:
                indices = ProgressBar(indices)

        for i in indices:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[parnum][i]
            ftr.fit_toas(**fitargs)
            chi2[i] = ftr.resids.chi2
            for extrapar in extraparnames:
                extraout[extrapar] = getattr(ftr.model, extrapar).quantity

    # Restore saved model
    ftr.model = savemod
    return chi2, extraout


def grid_chisq_derived(
    ftr,
    parnames,
    parfuncs,
    gridvalues,
    extraparnames=[],
    executor=None,
    ncpu=None,
    chunksize=1,
    printprogress=True,
    **fitargs,
):
    """Compute chisq over a grid of derived parameters

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters (available in `ftr`) to grid over
    parfuncs : list
        List of functions to convert `gridvalues` to quantities accessed through `parnames`
    gridvalues : list
        List of underlying grid values to grid over (each should be 1D array of astropy.units.Quantity)
    extraparnames : list, optional
        Names of other parameters to return
    executor : concurrent.futures.Executor or None, optional
        Executor object to run multiple processes in parallel
        If None, will use default :class:`concurrent.futures.ProcessPoolExecutor`, unless overridden by ``ncpu=1``
    ncpu : int, optional
        If an existing Executor is not supplied, one will be created with this number of workers.
        If 1, will run single-processor version
        If None, will use :func:`multiprocessing.cpu_count`
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress (requires :mod:`tqdm` for `ncpu`>1)
    fitargs :
        additional arguments pass to fit_toas()

    Returns
    -------
    np.ndarray : array of chisq values
    parvalues : list of np.ndarray
        Parameter values computed from `gridvalues` and `parfuncs`
    extraout : dict of np.ndarray
        Parameter values computed at each grid point for `extraparnames`

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.config
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model, get_model_and_toas
    # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    # do a grid for F0 and tau
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,15,)
    >>> tau = np.linspace(8.1, 8.3, 13) * 100 * u.Myr
    >>> chi2grid_tau, params = pint.gridutils.grid_chisq_derived(f,("F0", "F1"),(lambda x, y: x, lambda x, y: -x / 2 / y),(F0, tau))

    Notes
    -----
    By default, it will create :class:`~concurrent.futures.ProcessPoolExecutor`
    with ``max_workers`` equal to the desired number of cpus.
    However, if you are running this as a script you may need something like::

    import multiprocessing
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        ...
        grid_chisq_derived(...)

    If an instantiated :class:`~concurrent.futures.Executor` is passed instead, it will be used as-is.

    The behavior for different combinations of `executor` and `ncpu` is:
    +-----------------+--------+------------------------+
    | `executor`      | `ncpu` | result                 |
    +=================+========+========================+
    | existing object | any    | uses existing executor |
    +-----------------+--------+------------------------+
    | None	      | 1      | uses single-processor  |
    +-----------------+--------+------------------------+
    | None	      | None   | creates default        |
    |                 |        | executor with          |
    |                 |        | ``cpu_count`` workers  |
    +-----------------+--------+------------------------+
    | None	      | >1     | creates default        |
    |                 |        | executor with desired  |
    |                 |        | number of workers      |
    +-----------------+--------+------------------------+

    Other ``Executors`` can be found for different computing environments:
    * [1]_ for MPI
    * [2]_ for SLURM or Condor

    .. [1] https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor
    .. [2] https://github.com/sampsyo/clusterfutures
    """
    if isinstance(executor, concurrent.futures.Executor):
        # the executor has already been created
        executor = executor
    elif executor is None and (ncpu is None or ncpu > 1):
        # make the default type of Executor
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    wftr = WrappedFitter(ftr, **fitargs)

    # All other unfrozen parameters will be fitted for at each grid point
    grid = np.meshgrid(*gridvalues)
    chi2 = np.zeros(grid[0].shape)
    out = []
    # convert the gridded values to the actual parameter values
    for j in range(len(parfuncs)):
        out.append(parfuncs[j](*grid))
    extraout = {}
    for extrapar in extraparnames:
        extraout[extrapar] = (
            np.zeros(grid[0].shape, dtype=getattr(ftr.model, extrapar).quantity.dtype)
            * getattr(ftr.model, extrapar).quantity.unit
        )

    # at this point, if the executor is None then run single-processor version
    if executor is not None:
        with executor as e:
            if printprogress and tqdm is not None:
                result = list(
                    tqdm(
                        e.map(
                            wftr.doonefit,
                            # (ftr,) * len(out[0].flatten()),
                            (parnames,) * len(out[0].flatten()),
                            list(zip(*[x.flatten() for x in out])),
                            (extraparnames,) * len(out[0].flatten()),
                            chunksize=chunksize,
                        ),
                        total=len(out[0].flatten()),
                        ascii=True,
                    )
                )
            else:
                result = e.map(
                    wftr.doonefit,
                    # (ftr,) * len(out[0].flatten()),
                    (parnames,) * len(out[0].flatten()),
                    list(zip(*[x.flatten() for x in out])),
                    (extraparnames,) * len(out[0].flatten()),
                    chunksize=chunksize,
                )

        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r[0]
            for extrapar, extraparvalue in zip(extraparnames, r[1]):
                extraout[extrapar][i] = extraparvalue
    else:
        indices = list(np.ndindex(grid[0].shape))
        if printprogress:
            if tqdm is not None:
                indices = tqdm(indices, ascii=True)
            else:
                indices = ProgressBar(indices)

        for i in indices:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[parnum][i]
            ftr.fit_toas(**fitargs)
            chi2[i] = ftr.resids.chi2
            for extrapar in extraparnames:
                extraout[extrapar] = getattr(ftr.model, extrapar).quantity

    # Restore saved model
    ftr.model = savemod
    return chi2, out, extraout


def tuple_chisq(
    ftr,
    parnames,
    parvalues,
    extraparnames=[],
    executor=None,
    ncpu=None,
    chunksize=1,
    printprogress=True,
    **fitargs,
):
    """Compute chisq over a list of parameter tuples

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters to grid over
    parvalues : list
        List of parameter values to grid over (each should be tuple of :class:`astropy.units.Quantity`)
    extraparnames : list, optional
        Names of other parameters to return
    executor : concurrent.futures.Executor or None, optional
        Executor object to run multiple processes in parallel
        If None, will use default :class:`concurrent.futures.ProcessPoolExecutor`, unless overridden by ``ncpu=1``
    ncpu : int, optional
        If an existing Executor is not supplied, one will be created with this number of workers.
        If 1, will run single-processor version
        If None, will use :func:`multiprocessing.cpu_count`
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress (requires :mod:`tqdm` for `ncpu`>1)
    fitargs :
        additional arguments pass to fit_toas()

    Returns
    -------
    np.ndarray : array of chisq values
    extraout : dict of np.ndarray
        Parameter values computed at each point for `extraparnames`

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.config
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model, get_model_and_toas
    # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    # We'll do something like 3-sigma around the best-fit values of  F0
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,25)
    >>> F1 = np.ones(len(F0))*f.model.F1.quantity
    >>> chi2_F0,extra = pint.gridutils.tuple_chisq(f, ("F0","F1",), parvalues, extraparnames=("DM",))

    Notes
    -----
    By default, it will create :class:`~concurrent.futures.ProcessPoolExecutor`
    with ``max_workers`` equal to the desired number of cpus.
    However, if you are running this as a script you may need something like::

        import multiprocessing

        if __name__ == "__main__":
            multiprocessing.freeze_support()
            ...
            tuple_chisq(...)

    If an instantiated :class:`~concurrent.futures.Executor` is passed instead, it will be used as-is.

    The behavior for different combinations of `executor` and `ncpu` is:
    +-----------------+--------+------------------------+
    | `executor`      | `ncpu` | result                 |
    +=================+========+========================+
    | existing object | any    | uses existing executor |
    +-----------------+--------+------------------------+
    | None	      | 1      | uses single-processor  |
    +-----------------+--------+------------------------+
    | None	      | None   | creates default        |
    |                 |        | executor with          |
    |                 |        | ``cpu_count`` workers  |
    +-----------------+--------+------------------------+
    | None	      | >1     | creates default        |
    |                 |        | executor with desired  |
    |                 |        | number of workers      |
    +-----------------+--------+------------------------+

    Other ``Executors`` can be found for different computing environments:
    * [1]_ for MPI
    * [2]_ for SLURM or Condor

    .. [1] https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor
    .. [2] https://github.com/sampsyo/clusterfutures
    """
    if isinstance(executor, concurrent.futures.Executor):
        # the executor has already been created
        executor = executor
    elif executor is None and (ncpu is None or ncpu > 1):
        # make the default type of Executor
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the  params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    wftr = WrappedFitter(ftr, **fitargs)

    # All other unfrozen parameters will be fitted for at each grid point
    chi2 = np.zeros(len(parvalues))

    extraout = {}
    for extrapar in extraparnames:
        extraout[extrapar] = (
            np.zeros(chi2.shape, dtype=getattr(ftr.model, extrapar).quantity.dtype)
            * getattr(ftr.model, extrapar).quantity.unit
        )

    # at this point, if the executor is None then run single-processor version
    if executor is not None:
        with executor as e:
            if printprogress and tqdm is not None:
                result = list(
                    tqdm(
                        e.map(
                            wftr.doonefit,
                            (parnames,) * len(chi2),
                            parvalues,
                            (extraparnames,) * len(chi2),
                            chunksize=chunksize,
                        ),
                        total=len(chi2),
                        ascii=True,
                    )
                )
            else:
                result = e.map(
                    wftr.doonefit,
                    (parnames,) * len(chi2),
                    parvalues,
                    (extraparnames,) * len(chi2),
                    chunksize=chunksize,
                )
        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r[0]
            for extrapar, extraparvalue in zip(extraparnames, r[1]):
                extraout[extrapar][i] = extraparvalue
    else:
        indices = list(np.ndindex(chi2.shape))
        if printprogress:
            if tqdm is not None:
                indices = tqdm(indices, ascii=True)
            else:
                indices = ProgressBar(indices)

        for i in indices:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = parvalues[i[0]][parnum]
            ftr.fit_toas(**fitargs)
            chi2[i[0]] = ftr.resids.chi2
            for extrapar in extraparnames:
                extraout[extrapar][i[0]] = getattr(ftr.model, extrapar).quantity

    # Restore saved model
    ftr.model = savemod
    return chi2, extraout


def tuple_chisq_derived(
    ftr,
    parnames,
    parfuncs,
    parvalues,
    extraparnames=[],
    executor=None,
    ncpu=None,
    chunksize=1,
    printprogress=True,
    **fitargs,
):
    """Compute chisq over a list of derived parameter tuples

    Parameters
    ----------
    ftr : pint.fitter.Fitter
        The base fitter to use.
    parnames : list
        Names of the parameters (available in `ftr`) to grid over
    parfuncs : list
        List of functions to convert `gridvalues` to quantities accessed through `parnames`
    parvalues : list
        List of underlying parameter values to fit (each should be a tuple of astropy.units.Quantity)
    extraparnames : list, optional
        Names of other parameters to return
    executor : concurrent.futures.Executor or None, optional
        Executor object to run multiple processes in parallel
        If None, will use default :class:`concurrent.futures.ProcessPoolExecutor`, unless overridden by ``ncpu=1``
    ncpu : int, optional
        If an existing Executor is not supplied, one will be created with this number of workers.
        If 1, will run single-processor version
        If None, will use :func:`multiprocessing.cpu_count`
    chunksize : int
        Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
        Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
    printprogress : bool, optional
        Print indications of progress (requires :mod:`tqdm` for `ncpu`>1)
    fitargs :
        additional arguments pass to fit_toas()

    Returns
    -------
    np.ndarray : array of chisq values
    outparvalues : list of tuples
        Parameter values computed from `parvalues` and `parfuncs`
    extraout : dict of np.ndarray
        Parameter values computed at each point for `extraparnames`

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.config
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model, get_model_and_toas
    # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    # do a grid for F0 and tau
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,15,)
    # make sure it's the same length
    >>> tau = np.linspace(8.1, 8.3, 15) * 100 * u.Myr
    >>> parvalues = list(zip(F0,tau))
    >>> chi2_tau, params, _ = pint.gridutils.tuple_chisq_derived(f,("F0", "F1"),(lambda x, y: x, lambda x, y: -x / 2 / y),(F0, tau))

    Notes
    -----
    By default, it will create :class:`~concurrent.futures.ProcessPoolExecutor`
    with ``max_workers`` equal to the desired number of cpus.
    However, if you are running this as a script you may need something like::

    import multiprocessing
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        ...
        tuple_chisq_derived(...)

    If an instantiated :class:`~concurrent.futures.Executor` is passed instead, it will be used as-is.

    The behavior for different combinations of `executor` and `ncpu` is:
    +-----------------+--------+------------------------+
    | `executor`      | `ncpu` | result                 |
    +=================+========+========================+
    | existing object | any    | uses existing executor |
    +-----------------+--------+------------------------+
    | None	      | 1      | uses single-processor  |
    +-----------------+--------+------------------------+
    | None	      | None   | creates default        |
    |                 |        | executor with          |
    |                 |        | ``cpu_count`` workers  |
    +-----------------+--------+------------------------+
    | None	      | >1     | creates default        |
    |                 |        | executor with desired  |
    |                 |        | number of workers      |
    +-----------------+--------+------------------------+

    Other ``Executors`` can be found for different computing environments:
    * [1]_ for MPI
    * [2]_ for SLURM or Condor

    .. [1] https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor
    .. [2] https://github.com/sampsyo/clusterfutures
    """
    if isinstance(executor, concurrent.futures.Executor):
        # the executor has already been created
        executor = executor
    elif executor is None and (ncpu is None or ncpu > 1):
        # make the default type of Executor
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    wftr = WrappedFitter(ftr, **fitargs)

    # All other unfrozen parameters will be fitted for at each grid point
    chi2 = np.zeros(len(parvalues))
    out = []
    # convert the tuples of values to the actual parameter values
    for i in range(len(parvalues)):
        out.append([f(*parvalues[i]) for f in parfuncs])

    extraout = {}
    for extrapar in extraparnames:
        extraout[extrapar] = (
            np.zeros(len(chi2), dtype=getattr(ftr.model, extrapar).quantity.dtype)
            * getattr(ftr.model, extrapar).quantity.unit
        )

    # at this point, if the executor is None then run single-processor version
    if executor is not None:
        with executor as e:
            if printprogress and tqdm is not None:
                result = list(
                    tqdm(
                        e.map(
                            wftr.doonefit,
                            (parnames,) * len(chi2),
                            out,
                            (extraparnames,) * len(chi2),
                            chunksize=chunksize,
                        ),
                        total=len(chi2),
                        ascii=True,
                    )
                )
            else:
                result = e.map(
                    wftr.doonefit,
                    (parnames,) * len(chi2),
                    out,
                    (extraparnames,) * len(chi2),
                    chunksize=chunksize,
                )

        it = np.ndindex(chi2.shape)
        for i, r in zip(it, result):
            chi2[i] = r[0]
            for extrapar, extraparvalue in zip(extraparnames, r[1]):
                extraout[extrapar][i] = extraparvalue
    else:
        indices = list(np.ndindex(chi2.shape))
        if printprogress:
            if tqdm is not None:
                indices = tqdm(indices, ascii=True)
            else:
                indices = ProgressBar(indices)

        for i in indices:
            for parnum, parname in enumerate(parnames):
                getattr(ftr.model, parname).quantity = out[i[0]][parnum]
            ftr.fit_toas(**fitargs)
            chi2[i[0]] = ftr.resids.chi2
            for extrapar in extraparnames:
                extraout[extrapar][i[0]] = getattr(ftr.model, extrapar).quantity

    # Restore saved model
    ftr.model = savemod
    return chi2, out, extraout


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
