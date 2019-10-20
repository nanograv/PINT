"""
An interface for pint compatible to the interface of libstempo
"""
from __future__ import absolute_import, division, print_function

import collections
import time
from collections import OrderedDict

import astropy.constants as ac
import astropy.coordinates.angles as ang
import astropy.units as u
import numpy as np
from astropy import log

import pint.models as pm
import pint.toa as pt
from pint import fitter
from pint.utils import has_astropy_unit
from pint.residuals import Residuals

__all__ = ["PINTPar", "PINTPulsar"]

# Make sure we have the minuteangle and secondangle units
u.minuteangle = u.def_unit("minuteangle", u.hourangle / 60)
u.secondangle = u.def_unit("secondangle", u.minuteangle / 60)
u.lts = u.def_unit(["lightsecond", "ls", "lts"], ac.c * u.s)

# In the case of a unitless interface, we need to make sure we provide the
# 'correct' tempo2-like units
map_units = {
    "F0": u.Hz,
    "F1": u.Hz / u.s,
    "F2": u.Hz / u.s ** 2,
    "F3": u.Hz / u.s ** 3,
    "F4": u.Hz / u.s ** 4,
    "F5": u.Hz / u.s ** 5,
    "F6": u.Hz / u.s ** 6,
    "F7": u.Hz / u.s ** 7,
    "F8": u.Hz / u.s ** 8,
    "F9": u.Hz / u.s ** 9,
    "F10": u.Hz / u.s ** 10,
    "F11": u.Hz / u.s ** 11,
    "F12": u.Hz / u.s ** 12,
    "RAJ": u.rad,
    "DECJ": u.rad,
    "ELONG": u.deg,
    "ELAT": u.deg,
    "PMRA": u.mas / u.yr,
    "PMDEC": u.mas / u.yr,
    "PMELONG": u.mas / u.yr,
    "PMELAT": u.mas / u.yr,
    "PX": u.mas,
    "PB": u.d,
    "E": u.dimensionless_unscaled,  # == ECC
    "ECC": u.dimensionless_unscaled,
    "A1": u.lts,
    "OM": u.deg,
    "EPS1": u.dimensionless_unscaled,
    "EPS2": u.dimensionless_unscaled,
    # KOM, KIN?
    "SHAPMAX": u.dimensionless_unscaled,
    "OMDOT": u.deg / u.yr,
    # PBDOT?
    "ECCDOT": 1 / u.s,
    "A1DOT": u.lts / u.s,  # == XDOT
    "XDOT": u.lts / u.s,
    "GAMMA": u.s,
    # XPBDOT?
    # EPS1DOT, EPS2DOT?
    "MTOT": u.Msun,
    "M2": u.Msun,
    # DTHETA, XOMDOT
    "SIN1": u.dimensionless_unscaled,
    # DR, A0, B0, BP, BPP, AFAC
    "DM": u.cm ** -3 * u.pc,
    "DM1": u.cm ** -3 * u.pc * u.yr ** -1,  # how many should we do?
    "POSEPOCH": u.day,
    "T0": u.day,
    "TASC": u.day,
}


class PINTPar(object):
    """
    Similar to the parameter class defined in libstempo, this class gives a nice
    interface to the timing model parameters
    """

    def __init__(self, par, parname, *args, **kwargs):
        """Initialize parameter object

        :param par:
            The PINT parameter object

        :param parname:
            The name (key) of the parameter
        """

        self.name = parname
        self._par = par
        self._set = True

    @property
    def val(self):
        return self._par.value

    @val.setter
    def val(self, value):
        self._par.value = value

    @property
    def err(self):
        return self._par.uncertainty

    @err.setter
    def err(self, value):
        self._par.uncertainty = value

    @property
    def fit(self):
        return not self._par.frozen

    @fit.setter
    def fit(self, value):
        # TODO: When we set it, this value seems decoupled from self._par.frozen
        self._par.frozen = not value

    @property
    def frozen(self):
        return self._par.frozen

    @frozen.setter
    def frozen(self, value):
        self._par.frozen = value

    @property
    def set(self):
        return self._set

    @set.setter
    def set(self, value):
        self._set = value


class PINTPulsar(object):
    """
    Pulsar object class with an interface similar to tempopulsar of libstempo
    """

    def __init__(
        self,
        parfile,
        timfile=None,
        warnings=False,
        fixprefiterrors=True,
        dofit=False,
        maxobs=None,
        units=False,
    ):
        """
        The same init function as used in libstempo

        :param parfile:
            Name of the parfile

        :param timfile:
            Name of the timfile, if we want to load it

        :param warnings:
            Whether we are shoing warnings

        :param fixprefiterrors:
            TODO: check what this should do

        :param maxobs:
            PINT has no need for a maxobs parameter. Included here for
            compatibility

        :param units:
            Whether or not we are using the 'units' interface of libstempo
        """
        if warnings:
            log.setLevel("INFO")
        else:
            log.setLevel("ERROR")

        self.loadparfile(parfile)

        if timfile is not None:
            self.loadtimfile(timfile)
        else:
            self.t = None
            self.deleted = None

        if dofit and self.t is not None:
            self.fit()

        self._units = units

    def loadparfile(self, parfile):
        """
        Load a parfile with pint

        :param parfile:
            Name of the parfile
        """

        self.model = pm.get_model(parfile)
        log.info("model.as_parfile():\n%s" % self.model.as_parfile())

        try:
            self.planets = self.model.PLANET_SHAPIRO.value
        except AttributeError:
            self.planets = False

        self._readparams()

    def _readparams(self):
        """Process the timing model parameters, libstempo style"""
        self.pardict = OrderedDict()
        for par in self.model.params:
            self.pardict[par] = PINTPar(getattr(self.model, par), par)

    def loadtimfile(self, timfile):
        """
        Load a pulsar with pint

        :param parfile:
            Name of the parfile

        :param timfile:
            Name of the timfile, if we want to load it
        """

        t0 = time.time()
        self.t = pt.get_TOAs(timfile, planets=self.planets, usepickle=False)
        time_toa = time.time() - t0
        log.info("Read/corrected TOAs in %.3f sec" % time_toa)

        if log.level < 25:
            self.t.print_summary()

        self.deleted = np.zeros(self.t.ntoas, dtype=np.bool)
        self._readflags()

    def _readflags(self):
        """Process the pint flags to be in the same format as in libstempo"""

        self.flagnames_ = []
        self.flags_ = dict()

        for ii, obsflags in enumerate(self.t.get_flags()):
            for jj, flag in enumerate(obsflags):

                if flag not in self.flagnames_:
                    self.flagnames_.append(flag)
                    self.flags_[flag] = [""] * self.t.ntoas

                self.flags_[flag][ii] = obsflags[flag]

        # As is done in libstempo
        # for flag in self.flags_:
        #    self.flags_[flag].flags.writeable = False

    def toas(self, updatebats=False):
        """Return TDB arrival times in MJDs"""

        # TODO: do high-precision as long double

        return np.array(self.t.table["tdbld"])[~self.deleted]

    @property
    def stoas(self):
        """Return site arrival times"""

        return np.array(self.t.get_mjds())[~self.deleted]

    @property
    def toaerrs(self):
        """Return the TOA uncertainty in microseconds"""

        return np.array(self.t.get_errors().to(u.us))[~self.deleted]

    @property
    def freqs(self):
        """Returns observation frequencies in units of MHz as a numpy.double array."""

        return np.array(self.t.get_freqs())

    def ssbfreqs(self):
        """Return SSB frequencies"""

        log.warning("Not using freqSSB just yet.")
        return np.array(self.t.get_freqs())

    def deletedmask(self):
        """tempopulsar.deletedmask()

        Returns a numpy.bool array of the delection station of observations.
        You get a copy of the current values."""

        return self.deleted

    def flags(self):
        """Returns the list of flags defined in this dataset (for at least some observations)."""

        return self.flagnames_

    # TO DO: setting flags
    def flagvals(self, flagname, values=None):
        """Returns (or sets, if `values` are given) a numpy unicode-string array
        containing the values of flag `flagname` for every observation."""

        if values is None:
            return self.flags_[flagname]
        else:
            raise NotImplementedError("Flag-setting capabilities are coming soon.")

    def formresiduals(self):
        """Form the residuals"""

        log.info("Computing residuals...")
        t0 = time.time()
        self.resids_us = Residuals(self.t, self.model).time_resids.to(u.us)
        time_phase = time.time() - t0
        log.info("Computed phases and residuals in %.3f sec" % time_phase)

        # resids in (approximate) us:
        log.info("RMS PINT residuals are %.3f us" % self.resids_us.std().value)

    def residuals(self, updatebats=True, formresiduals=True, removemean=True):
        """Returns residuals"""
        self.formresiduals()

        return np.array(self.resids_us.to(u.s))[~self.deleted]

    @property
    def name(self):
        """Get or set pulsar name."""

        return self.model.PSR.value

    @property
    def binarymodel(self):
        """Return the binary model"""

        # Obtain the binary model from PINT
        """
        def __get__(self):
            return self.psr[0].binaryModel.decode('ascii')

        def __set__(self,value):
            model_bytes = value.encode('ascii')

            if len(model_bytes) < 100:
                stdio.sprintf(self.psr[0].binaryModel,"%s",<char *>model_bytes)
            else:
                raise ValueError
        """
        return None

    def pars(self, which="fit"):
        """Return parameter keys"""

        if which == "fit":
            return [
                par for par in self.model.params if not getattr(self.model, par).frozen
            ]
        elif which == "set":
            return [par for par in self.model.params]
        if which == "frozen":
            return [par for par in self.model.params if getattr(self.model, par).frozen]
        elif which == "all":
            raise NotImplementedError("PINT does not track 'all' parameters")

    @property
    def nobs(self):
        """Number of observations"""

        return self.t.ntoas - np.sum(self.deleted)

    @property
    def ndim(self, incoffset=True):
        """Number of dimensions."""

        return int(incoffset) + len(self.pars(which="fit"))

    # --- dictionary access to parameters
    def __contains__(self, key):
        return key in self.model.params

    def __getitem__(self, key):
        # return getattr(self.model, key).value
        return self.pardict[key]

    # --- bulk access to parameter values
    def vals(self, values=None, which="fit"):
        """tempopulsar.vals(values=None,which='fit')

        Get (if no `values` provided) or set the parameter values, depending on `which`:

        - if `which` is 'fit' (default), fitted parameters;
        - if `which` is 'set', all parameters with a defined value;
        - if `which` is 'all', all parameters;
        - if `which` is a sequence, all parameters listed there.

        Parameter values are returned as a numpy long double array.

        Values to be set can be passed as a numpy array, sequence (in which case they
        are taken to correspond to parameters in the order given by `pars(which=which)`),
        or dict (in which case which will be ignored).

        Notes:

        - Passing values as anything else than numpy long double may result in loss of precision.
        - Not all parameters in the selection need to be set.
        - Setting an unset parameter sets its `set` flag (obviously).
        - Unlike in earlier libstempo versions, setting a parameter does not set its error to zero."""

        if values is None:
            return np.fromiter(
                (self[par].val for par in self.pars(which)), np.longdouble
            )
        elif isinstance(values, collections.Mapping):
            for par in values:
                self[par].val = values[par]
        elif isinstance(values, collections.Iterable):
            for par, val in zip(self.pars(which), values):
                self[par].val = val
        else:
            raise TypeError

    def errs(self, values=None, which="fit"):
        """tempopulsar.errs(values=None,which='fit')

        Same as `vals()`, but for parameter errors."""
        if values is None:
            # return np.fromiter((getattr(self.model, par).uncertainty for par in
            #        self.pars(which)),np.longdouble)
            return np.fromiter(
                (self[par].err for par in self.pars(which)), np.longdouble
            )
        elif isinstance(values, collections.Mapping):
            for par in values:
                self[par].err = values[par]
        elif isinstance(values, collections.Iterable):
            for par, val in zip(self.pars(which), values):
                self[par].err = val
        else:
            raise TypeError

    def designmatrix(
        self, updatebats=True, fixunits=True, fixsigns=True, incoffset=True
    ):
        """tempopulsar.designmatrix(updatebats=True,fixunits=True,incoffset=True)

        Returns the design matrix [nobs x (ndim+1)] as a long double array
        for current fit-parameter values. If fixunits=True, adjust the units
        of the design-matrix columns so that they match the tempo2
        parameter units. If fixsigns=True, adjust the sign of the columns
        corresponding to FX (F0, F1, ...) and JUMP parameters, so that
        they match finite-difference derivatives. If incoffset=False, the
        constant phaseoffset column is not included in the designmatrix."""
        M, params, units = self.model.designmatrix(
            self.t.table, incfrozen=False, incoffset=incoffset
        )
        return M

    def telescope(self):
        """tempopulsar.telescope()

        Returns a numpy character array of the telescope for each observation,
        mapping tempo2 `telID` values to names by way of the tempo2 runtime file
        `observatory/aliases`."""
        pass

    def binarydelay(self):
        """tempopulsar.binarydelay()

        Return a long-double numpy array of the delay introduced by the binary model.
        Does not reform residuals."""
        pass

    def elevation(self):
        """tempopulsar.elevation()

        Return a numpy double array of the elevation of the pulsar
        at the time of the observations."""
        pass

    def phasejumps(self):
        """tempopulsar.phasejumps()

        Return an array of phase-jump tuples: (MJD, phase). These are
        copies.

        NOTE: As in tempo2, we refer the phase-jumps to site arrival times. The
        tempo2 definition of a phasejump is such that it is applied when
        observation_SAT > phasejump_SAT
        """
        pass

    def add_phasejump(self, mjd, phasejump):
        """tempopulsar.add_phasejump(mjd,phasejump)

        Add a phase jump of value `phasejump` at time `mjd`.

        Note: due to the comparison observation_SAT > phasejump_SAT in tempo2,
        the exact MJD itself where the jump was added is not affected.
        """
        pass

    def remove_phasejumps(self):
        """tempopulsar.remove_phasejumps()

        Remove all phase jumps."""
        pass

    @property
    def nphasejumps(self):
        """Return the number of phase jumps."""
        pass

    @property
    def pulse_number(self):
        """Return the pulse number relative to PEPOCH, as detected by tempo2

        WARNING: Will be deprecated in the future. Use `pulsenumbers`.
        """
        return self.model.phase(self.t.table).int.quantity.value

    def pulsenumbers(self, updatebats=True, formresiduals=True, removemean=True):
        """Return the pulse number relative to PEPOCH, as detected by tempo2

        Returns the pulse numbers as a numpy array. Will update the
        TOAs/recompute residuals if `updatebats`/`formresiduals` is True
        (default for both). If that is requested, the residual mean is removed
        `removemean` is True. All this just like in `residuals`.
        """
        return self.model.phase(self.t.table).int.quantity.value

    def fit(self, iters=1):
        """tempopulsar.fit(iters=1)

        Runs `iters` iterations of the tempo2 fit, recomputing
        barycentric TOAs and residuals each time."""
        f = fitter.wls_fitter(toas=self.t, model=self.model)

        for ii in range(iters + 1):
            f.call_minimize()

        fitp = f.get_fitparams()
        # TODO: handle these units correctly
        for p, val in zip(fitp.keys(), fitp.values()):
            modval = getattr(f.model, p).value

            if (not has_astropy_unit(val)) and has_astropy_unit(modval):
                if type(modval) is ang.Angle:
                    val = ang.Angle(val, unit=modval.unit)
                else:
                    val = val * modval.unit

            self[p].val = val

    def chisq(self, removemean="weighted"):
        """tempopulsar.chisq(removemean='weighted')

        Computes the chisq of current residuals vs errors,
        removing the noise-weighted residual, unless
        specified otherwise."""
        res, err = self.residuals(removemean=removemean), self.toaerrs

        return np.sum(res * res / (1e-12 * err * err))

    def rms(self, removemean="weighted"):
        """tempopulsar.rms(removemean='weighted')

        Computes the current residual rms,
        removing the noise-weighted residual, unless
        specified otherwise."""
        err = self.toaerrs
        norm = np.sum(1.0 / (1e-12 * err * err))

        return np.sqrt(self.chisq(removemean=removemean) / norm)

    def savepar(self, parfile):
        """tempopulsar.savepar(parfile)

        Save current par file (calls tempo2's `textOutput(...)`)."""
        pass

    def savetim(self, timfile):
        """tempopulsar.savetim(timfile)

        Save current par file (calls tempo2's `writeTim(...)`)."""
        pass
