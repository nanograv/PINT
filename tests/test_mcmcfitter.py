from pint.mcmc_fitter import MCMCFitter, lnlikelihood_chi2, set_priors_basic
from pint.sampler import EmceeSampler
from pint.models import get_model_and_toas
from pint.config import examplefile

import pytest


@pytest.fixture()
def data_NGC6440E():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    return get_model_and_toas(parfile, timfile)


def test_mcmc_fitter(data_NGC6440E):
    m, t = data_NGC6440E

    nwalkers = 10
    nsteps = 5

    sampler = EmceeSampler(nwalkers)
    f = MCMCFitter(
        t,
        m,
        sampler,
        resids=True,
        phs=0.50,
        phserr=0.01,
        lnlike=lnlikelihood_chi2,
    )
    set_priors_basic(f)
    f.fit_toas(nsteps)
    chains = sampler.chains_to_dict(f.fitkeys)

    assert set(chains.keys()) == set(m.free_params)
    assert all(chains[par].shape == (nsteps, nwalkers) for par in m.free_params)
