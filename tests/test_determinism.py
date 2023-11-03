import random
from os.path import join

import emcee
import numpy as np
import numpy.random
from numpy.testing import assert_array_equal

import pint.fermi_toas as fermi
import pint.models
import pint.toa as toa
from pint.mcmc_fitter import MCMCFitterBinnedTemplate
from pint.sampler import EmceeSampler
from pint.scripts.event_optimize import read_gaussfitfile
from pinttestdata import datadir


def test_sampler():
    r = []
    for _ in range(2):
        random.seed(0)
        numpy.random.seed(0)
        s = numpy.random.mtrand.RandomState(0)

        parfile = join(datadir, "PSRJ0030+0451_psrcat.par")
        eventfile = join(
            datadir,
            "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits",
        )
        gaussianfile = join(datadir, "templateJ0030.3gauss")
        weightcol = "PSRJ0030+0451"

        minWeight = 0.9
        nwalkers = 10
        nsteps = 1
        nbins = 256
        phs = 0.0

        model = pint.models.get_model(parfile)
        ts = fermi.get_Fermi_TOAs(
            eventfile,
            weightcolumn=weightcol,
            minweight=minWeight,
            ephem="DE421",
            planets=False,
        )
        # Introduce a small error so that residuals can be calculated
        ts.table["error"] = 1.0
        ts.filename = eventfile

        weights, _ = ts.get_flag_value("weight", as_type=float)
        weights = np.array(weights)
        template = read_gaussfitfile(gaussianfile, nbins)
        template /= template.mean()

        sampler = EmceeSampler(nwalkers)
        fitter = MCMCFitterBinnedTemplate(
            ts, model, sampler, template=template, weights=weights, phs=phs
        )
        fitter.sampler.random_state = s

        # phases = fitter.get_event_phases()
        # maxbin, like_start = marginalize_over_phase(phases, template,
        #                                            weights=fitter.weights,
        #                                            minimize=True,
        #                                            showplot=True)
        # fitter.fitvals[-1] = 1.0 - maxbin[0] / float(len(template))

        # fitter.set_priors(fitter, 10)
        pos = fitter.sampler.get_initial_pos(
            fitter.fitkeys,
            fitter.fitvals,
            fitter.fiterrs,
            0.1,
            minMJD=fitter.minMJD,
            maxMJD=fitter.maxMJD,
        )
        # pos = fitter.clip_template_params(pos)
        fitter.sampler.initialize_sampler(fitter.lnposterior, fitter.n_fit_params)
        fitter.sampler.run_mcmc(pos, nsteps)
        # fitter.fit_toas(maxiter=nsteps, pos=None)
        # fitter.set_parameters(fitter.maxpost_fitvals)

        # fitter.phaseogram()
        # samples = sampler.sampler.chain[:, 10:, :].reshape((-1, fitter.n_fit_params))
        samples = np.transpose(sampler.get_chain(), (1, 0, 2))

        # r.append(np.random.randn())
        # r.append(sampler.sampler.chain[0])
        r.append(samples[0])
    assert_array_equal(r[0], r[1])


def test_raw_emcee():
    r = []
    for i in range(2):
        random.seed(0)
        numpy.random.seed(0)
        s = numpy.random.mtrand.RandomState(0)

        def log_prob(x, ivar):
            return -0.5 * np.sum(ivar * x**2)

        ndim, nwalkers = 5, 100
        ivar = 1.0 / np.random.rand(ndim)
        # r.append(ivar[0])
        p0 = np.random.randn(nwalkers, ndim)
        # r.append(p0[0, 0])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
        sampler.random_state = s
        sampler.run_mcmc(p0, 100)

        # samples = sampler.chain.reshape((-1, ndim))
        samples = np.transpose(sampler.get_chain(), (1, 0, 2)).reshape((-1, ndim))
        r.append(samples[0, 0])
    assert r[0] == r[1]
