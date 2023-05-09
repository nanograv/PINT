"""Tools for building chi-squared grids."""
import concurrent.futures
import copy
import multiprocessing
import subprocess
import sys

import numpy as np

from loguru import logger as log

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from astropy.utils.console import ProgressBar

from pint import fitter
from pint.observatory import clock_file

__all__ = ["doonefit", "grid_chisq", "grid_chisq_derived"]


def hostinfo():
    return subprocess.check_output("uname -a", shell=True)


def set_log(logger_):
    global log
    log = logger_


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
        # copy the log to all imported modules
        # this makes them respect the logger settings
        for m in sys.modules:
            if m.startswith("pint") and hasattr(sys.modules[m], "log"):
                setattr(sys.modules[m], "log", log)

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

    Examples
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.config
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model, get_model_and_toas
    >>> # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    >>> # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    >>> # We'll do something like 3-sigma around the best-fit values of  F0
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,25)
    >>> chi2_F0,_ = pint.gridutils.grid_chisq(f, ("F0",), (F0,))


    A 2D example with a plot:

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pint.gridutils
    >>> from pint.fitter import WLSFitter
    >>> from pint.models.model_builder import get_model_and_toas
    >>> import scipy.stats
    >>> import matplotlib.pyplot as plt
    >>> # Load in a basic dataset
    >>> parfile = pint.config.examplefile("NGC6440E.par")
    >>> timfile = pint.config.examplefile("NGC6440E.tim")
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = WLSFitter(t, m)
    >>> # find the best-fit
    >>> f.fit_toas()
    >>> bestfit = f.resids.chi2
    >>> F0 = np.linspace(f.model.F0.quantity - 3 * f.model.F0.uncertainty,f.model.F0.quantity + 3 * f.model.F0.uncertainty,25)
    >>> F1 = np.linspace(f.model.F1.quantity - 3 * f.model.F1.uncertainty,f.model.F1.quantity + 3 * f.model.F1.uncertainty,27)
    >>> chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1))[0]
    >>> # 1, 2, and 3 sigma confidence limits
    >>> nsigma = np.arange(1, 4)
    >>> # these are the CDFs going from -infinity to nsigma.  So subtract away 0.5 and double for the 2-sided values
    >>> CIs = (scipy.stats.norm().cdf(nsigma) - 0.5) * 2
    >>> print(f"Confidence intervals for {nsigma} sigma: {CIs}")
    >>> # chi^2 random variable for 2 parameters
    >>> rv = scipy.stats.chi2(2)
    >>> # the ppf = Percent point function is the inverse of the CDF
    >>> contour_levels = rv.ppf(CIs)
    >>> fig, ax = plt.subplots(figsize=(16, 9))
    >>> # just plot the values offset from the best-fit values
    >>> twod = ax.contour(F0 - f.model.F0.quantity,F1 - f.model.F1.quantity,chi2grid - bestfit,levels=contour_levels,colors="b")
    >>> ax.errorbar(0, 0, xerr=f.model.F0.uncertainty.value, yerr=f.model.F1.uncertainty.value, fmt="ro")
    >>> ax.set_xlabel("$\Delta F_0$ (Hz)", fontsize=24)
    >>> ax.set_ylabel("$\Delta F_1$ (Hz/s)", fontsize=24)
    >>> plt.show()

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

    # Save the current model so we can tweak it for gridding, then restore it at the end
    savemod = ftr.model
    gridmod = copy.deepcopy(ftr.model)
    ftr.model = gridmod

    # Freeze the  params we are going to grid over
    for parname in parnames:
        getattr(ftr.model, parname).frozen = True

    wftr = WrappedFitter(ftr, **fitargs)

    if isinstance(executor, concurrent.futures.Executor):
        # the executor has already been created
        executor = executor
    elif executor is None and (ncpu is None or ncpu > 1):
        # make the default type of Executor
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=ncpu, initializer=set_log, initargs=(log,)
        )

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
