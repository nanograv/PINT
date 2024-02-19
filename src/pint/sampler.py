import emcee
import numpy as np

__all__ = ["MCMCSampler", "EmceeSampler"]


class MCMCSampler:
    """Base class for samplers used in MCMC fitting.

    The sampling method should be implemented in the run_mcmc() method.

    This was created initially to work with the emcee package but can
    be extended for other samplers.

    The use of this class can be somewhat tricky, since MCMCFitter must
    own some subclass of the abstract class MCMCSampler. However, the
    MCMCSampler requires access to functions inside MCMCFitter. So the
    specific MCMCSampler must be instantiated first and passed into
    the Fitter, but the Sampler cannot be initialized until after the Fitter
    has been created.

    The general flow of using the MCMCSampler class with MCMCFitter is::
        #Create the sampler object but don't initialize the internals
        sampler = MCMCSampler()

        #Create the fitter using the new sampler
        fitter = MCMCFitter(...stuff..., sampler=sampler)

        #Initialize the sampler at some point before use using
        sampler.initialize_sampler(fitter.lnposterior)
    """

    def __init__(self):
        self.method = None

    def initialize_sampler(self, lnpostfn, ndim):
        """Initialize the internals of the sampler using the posterior probability function

        This function must be called before run_mcmc()
        """
        raise NotImplementedError

    def get_initial_pos(self, fitkeys, fitvals, fiterrs, errfact, **kwargs):
        """Give the initial position(s) for the fitter based on given values."""
        raise NotImplementedError

    def run_mcmc(self, pos, nsteps):
        """Run the MCMC process from the given initial position

        Parameters
        ----------
        pos
            The initial sampling point
        nstesps : int
            The number of iterations to run for
        """
        raise NotImplementedError


class EmceeSampler(MCMCSampler):
    """Wrapper class around the emcee sampling package.

    This is to let it work within the PINT Fitter framework

    Be warned: emcee can only handle double precision. You will never get a
    longdouble response back.
    """

    def __init__(self, nwalkers):
        super().__init__()
        self.method = "Emcee"
        self.nwalkers = nwalkers
        self.sampler = None

    def is_initalized(self):
        """Simple way to check if the EmceeSampler can run yet."""
        return self.sampler is None

    def initialize_sampler(self, lnpostfn, ndim):
        """Initialize the internal sampler data.

        This is usually done after __init__ because ndim and lnpostfn are properties
        of the Fitter that holds this sampler.

        """
        self.ndim = ndim
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnpostfn)

    def get_initial_pos(self, fitkeys, fitvals, fiterrs, errfact, **kwargs):
        """Get the initial positions for each walker of the sampler.

        fitkeys, fitvals, fiterrs, and errfact all come from the Fitter,
        kwargs might contain min/maxMJD in the event of a glep_1 parameter
        """
        if len(fitkeys) != len(fitvals):
            raise ValueError(
                f"Number of keys does ({len(fitkeys)}) not match number of values ({len(fitvals)})!"
            )
        n_fit_params = len(fitvals)
        pos = [
            fitvals + fiterrs * errfact * np.random.randn(n_fit_params)
            for _ in range(self.nwalkers)
        ]
        # set starting params
        # FIXME: what about other glitch phase parameters? This can't be right!
        for param in [
            "GLPH_1",
            "GLEP_1",
            "SINI",
            "M2",
            "E",
            "ECC",
            "PX",
            "A1",
            "PHASE",
        ]:
            if param in fitkeys:
                idx = fitkeys.index(param)
                if param == "GLPH_1":
                    svals = np.random.uniform(-0.5, 0.5, self.nwalkers)
                elif param == "GLEP_1":
                    if "minMJD" in kwargs and "maxMJD" in kwargs:
                        svals = np.random.uniform(
                            kwargs["minMJD"] + 100,
                            kwargs["maxMJD"] - 100,
                            self.nwalkers,
                        )
                    else:
                        raise ValueError("minMJD or maxMJD is None for glep_1 param")
                elif param == "SINI":
                    svals = np.random.uniform(0.0, 1.0, self.nwalkers)
                elif param == "M2":
                    svals = np.random.uniform(0.1, 0.6, self.nwalkers)
                elif param == "PHASE":
                    svals = np.random.uniform(0.0, 1.0, self.nwalkers)
                elif param in ["E", "ECC", "PX", "A1"]:
                    svals = np.fabs(
                        fitvals[idx] + fiterrs[idx] * np.random.randn(self.nwalkers)
                    )
                    if param in ["E", "ECC"]:
                        svals[svals > 1.0] = 1.0 - (svals[svals > 1.0] - 1.0)
                for ii in range(self.nwalkers):
                    pos[ii][idx] = svals[ii]
        pos[0] = fitvals
        return [pos_i.astype(float) for pos_i in pos]

    def get_chain(self):
        """
        A safe method of getting the sampler chain, if it exists
        """
        if self.sampler is None:
            raise ValueError("MCMCSampler object has not called initialize_sampler()")
        return self.sampler.get_chain()

    def chains_to_dict(self, names):
        """
        Convert the sampler chains to a dictionary
        """
        if self.sampler is None:
            raise ValueError("MCMCSampler object has not called initialize_sampler()")
        samples = np.transpose(self.sampler.get_chain(), (1, 0, 2))
        chains = [samples[:, :, ii].T for ii in range(len(names))]
        return dict(zip(names, chains))

    def run_mcmc(self, pos, nsteps):
        """
        Wraps around emcee.run_mcmc
        """
        if self.sampler is None:
            raise ValueError("MCMCSampler object has not called initialize_sampler()")
        self.sampler.run_mcmc(
            pos, nsteps, progress=True, store=True, skip_initial_state_check=True
        )
