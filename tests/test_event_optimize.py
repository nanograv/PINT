# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO.
# Actually it's not disabled? Unclear what the above is supposed to mean.
import os
import shutil
import sys
from io import StringIO
from pathlib import Path

import emcee.backends
import pytest
import numpy as np
import pickle
import scipy.stats as stats
from pint.scripts import event_optimize
from pinttestdata import datadir


def test_result(tmp_path):
    parfile = datadir / "PSRJ0030+0451_psrcat.par"
    eventfile_orig = (
        datadir
        / "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
    )
    temfile = datadir / "templateJ0030.3gauss"
    eventfile = tmp_path / "event.fits"
    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)

    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
    try:
        os.chdir(tmp_path)
        cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin=10"
        event_optimize.main(cmd.split())
        lines = sys.stdout.getvalue()
        # Need to add some check here.
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout


def test_parallel(tmp_path):
    parfile = datadir / "PSRJ0030+0451_psrcat.par"
    eventfile_orig = (
        datadir
        / "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
    )
    temfile = datadir / "templateJ0030.3gauss"
    eventfile = tmp_path / "event.fits"
    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)

    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))

    np.random.seed(1)
    event_optimize.maxpost = -9e99
    event_optimize.numcalls = 0
    try:
        import pathos

        os.chdir(tmp_path)
        cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10 --clobber"
        event_optimize.main(cmd.split())
        with open("J0030+0451_samples.pickle", "rb") as f:
            samples1 = pickle.load(f)
        f.close
        np.random.seed(1)
        event_optimize.maxpost = -9e99
        event_optimize.numcalls = 0

        cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10 --multicore --clobber"
        event_optimize.main(cmd.split())
        with open("J0030+0451_samples.pickle", "rb") as f:
            samples2 = pickle.load(f)
        f.close()

        for i in range(samples1.shape[1]):
            assert stats.ks_2samp(samples1[:, i], samples2[:, i])[1] == 1.0
    except ImportError:
        pytest.skip(f"Pathos multiprocessing package not found")
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout


def test_backend(tmp_path):
    parfile = datadir / "PSRJ0030+0451_psrcat.par"
    eventfile_orig = (
        datadir
        / "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
    )
    temfile = datadir / "templateJ0030.3gauss"
    eventfile = tmp_path / "event.fits"
    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)

    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
    try:
        import h5py

        samples = None

        # Running with backend
        os.chdir(tmp_path)
        cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin=10 --backend --clobber"
        event_optimize.maxpost = -9e99
        event_optimize.numcalls = 0
        event_optimize.main(cmd.split())
        reader = emcee.backends.HDFBackend("J0030+0451_chains.h5")
        samples = reader.get_chain(discard=10)
        assert samples is not None

        try:
            # Clobber flag test
            timestamp = os.path.getmtime("J0030+0451_chains.h5")
            cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin=10 --backend"
            event_optimize.maxpost = -9e99
            event_optimize.numcalls = 0
            event_optimize.main(cmd.split())
        except Exception:
            assert timestamp == os.path.getmtime("J0030+0451_chains.h5")

    except ImportError:
        pytest.skip(f"h5py package not found")
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout


def test_autocorr(tmp_path):
    # Defining a log posterior function based on the emcee tutorials
    def ln_prob(theta):
        ln_prior = -0.5 * np.sum((theta - 1.0) ** 2 / 100.0)
        ln_prob = -0.5 * np.sum(theta**2) + ln_prior
        return ln_prob

    # Setting a random starting position for 10 walkers with 5 dimenisions
    coords = np.random.randn(10, 5)
    nwalkers, ndim = coords.shape

    # Establishing the Sampler
    nsteps = 500000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob)

    # Running the sampler with the autocorrelation check function from event_optimize
    autocorr = event_optimize.run_sampler_autocorr(sampler, coords, nsteps, burnin=10)

    # Extracting the samples and asserting that the autocorrelation check
    # stopped the sampler once convergence was reached
    samples = np.transpose(sampler.get_chain(discard=10), (1, 0, 2)).reshape((-1, ndim))

    assert len(samples) < nsteps
