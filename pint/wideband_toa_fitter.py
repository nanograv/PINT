"""A wideband TOAs fitter (This fitter name may change or be integrated to the
base fitter class)"""
from __future__ import absolute_import, print_function, division
import copy, numbers
import numpy as np
import astropy.units as u
import abc
import scipy.optimize as opt, scipy.linalg as sl
from .residuals import resids
from .fitter import GLSFitter


class WidebandFitter(GLSFitter):
    """The Wideband Fitter is designed to perform general least fitting with
       noise model and wideband TOAs.

       Parameter
       ---------
       toas : `~pint.toa.TOAs` object
           Input TOAs for fitting.
       model : `~pint.models.TimingModel` object
           Initial timing model.

       Note
       ----
       This fitter class is designed for temporary use of wideband TOAs. Its API
       will change and some functionality may be merged to the base fitter.
    """
    def __init__(self, toas=None, model=None):
        super(WidebandFitter, self).__init__(toas=toas, model=model, )
        self.method = 'Wideband_TOA_fitter'

    def fit_toas(self, maxiter=1, threshold=False, full_cov=False):
        """Run a Generalized least-squared fitting method"""
        chi2 = 0
        if self.model.DMDATA.value:
            dm_data = self.toas.get_flag_value('pp_dm', dtype=float,
                                               unit=u.pc * u.cm**-3)
            dm_model = np.zeros(self.toas.ntoas) * self.model.DM.units
            for dm_f in self.model.get_component_common('dm_value_funcs'):
                dm_model += dm_f(self.toas)
                dm_res = dm_data - dm_model
            dm_error = self.toas.get_flag_value('pp_dme', dtype=float,
                                                unit=u.pc * u.cm**-3)

        for i in range(maxiter):
            fitp = self.get_fitparams()
            fitpv = self.get_fitparams_num()
            fitperrs = self.get_fitparams_uncertainty()

            # Define the linear system
            M, params, units, scale_by_F0 = self.get_designmatrix()

            # Get residuals and TOA uncertainties in seconds
            self.update_resids()
            residuals = self.resids.time_resids.to(u.s).value
            if self.model.DMDATA.value:
                data = np.hstack((residuals, dm_res.quantity.value))
            else:
                data = residuals
            #TODO There should be a better design for handling this over all.
            if self.model.DMDATA.value:
                M_dm, params_dm, units_dm = self.model.dm_designmatrix(self.toas)
                # pad DM designmatrix to M
                new_permutation = []
                for pm_dm in params_dm:
                    pm_idx = params.index(pm_dm)
                    new_permutation.append(pm_idx)
                new_idx = np.argsort(new_permutation)
                M = np.vstack((M, M_dm[:, new_idx]))

            # get any noise design matrices and weight vectors
            if not full_cov:
                Mn = self.model.noise_model_designmatrix(self.toas)
                phi = self.model.noise_model_basis_weight(self.toas)
                phiinv = np.zeros(M.shape[1])
                if Mn is not None and phi is not None:
                    # Match the data length
                    if self.model.DMDATA.value:
                        np.pad(Mn, ((0, M.shape[0] - Mn.shape[0]), (0,0)),
                               'constant')
                        np.pad(phi, (0,  M.shape[0] - Mn.shape[0]), 'constant')
                    phiinv = np.concatenate((phiinv, 1/phi))
                    M = np.hstack((M, Mn))

            # normalize the design matrix
            norm = np.sqrt(np.sum(M**2, axis=0))
            ntmpar = len(fitp)
            if M.shape[1] > ntmpar:
                norm[ntmpar:] = 1
            if np.any(norm == 0):
                print("Warning: one or more of the design-matrix columns is null.")
            M /= norm

            # compute covariance matrices
            if full_cov:
                cov = self.model.covariance_matrix(self.toas)
                cf = sl.cho_factor(cov)
                cm = sl.cho_solve(cf, M)
                mtcm = np.dot(M.T, cm)
                mtcy = np.dot(cm.T, data)

            else:
                Nvec = self.model.scaled_sigma(self.toas).to(u.s).value**2
                if self.model.DMDATA.value:
                    # NOTE how do you handle the dm_error?
                    Nvec = np.hstack((Nvec, dm_error.quantity.value ** 2))
                cinv = 1 / Nvec
                mtcm = np.dot(M.T, cinv[:,None]*M)
                mtcm += np.diag(phiinv)
                mtcy = np.dot(M.T, cinv * data)


            try:
                c = sl.cho_factor(mtcm)
                xhat = sl.cho_solve(c, mtcy)
                xvar = sl.cho_solve(c, np.eye(len(mtcy)))
            except:
                U, s, Vt = sl.svd(mtcm, full_matrices=False)

                if threshold:
                    threshold_val = np.finfo(np.longdouble).eps * max(M.shape) * s[0]
                    s[s<threshold_val] = 0.0

                xvar = np.dot(Vt.T / s, Vt)
                xhat = np.dot(Vt.T, np.dot(U.T, mtcy)/s)


            # compute linearized chisq
            newres = data - np.dot(M, xhat)
            if full_cov:
                chi2 = np.dot(newres, sl.cho_solve(cf, newres))
            else:
                chi2 = np.dot(newres, cinv*newres) + np.dot(xhat,phiinv*xhat)

            # compute absolute estimates, normalized errors, covariance matrix
            dpars = xhat/norm
            errs = np.sqrt(np.diag(xvar)) / norm

            for ii, pn in enumerate(fitp.keys()):
                uind = params.index(pn)             # Index of designmatrix
                un = 1.0 / (units[uind])     # Unit in designmatrix
                if scale_by_F0:
                    un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv+dpv) / fitp[pn].units)
                #NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            _ = self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)
        return chi2
        #return M, phi
