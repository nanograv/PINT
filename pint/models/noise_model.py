# noise_model.py
# Defines the pulsar timing noise model.
from .timing_model import Component,  MissingParameter
from . import parameter as p
from .. import utils
import numpy as np
import astropy.units as u

class NoiseComponent(Component):
    def __init__(self,):
        super(NoiseComponent, self).__init__()
        self.category = 'timing_noise'
        self.covariance_matrix_funcs = []
        self.scaled_sigma_funcs = []
        self.basis_funcs = []

class ScaleToaError(NoiseComponent):
    """This is a class to correct template fitting timing noise.
    Notes
    -----
    Ref: NanoGrav 11 yrs data
    """
    register = True
    def __init__(self,):
        super(ScaleToaError, self).__init__()
        self.add_param(p.maskParameter(name ='EFAC', units="",
                                       aliases=['T2EFAC', 'TNEFAC'],
                                       description="A multiplication factor on" \
                                                   " the measured TOA uncertainties,"))

        self.add_param(p.maskParameter(name='EQUAD', units="us",\
                                       aliases=['T2EQUAD'],
                                       description="An error term added in "
                                                  "quadrature to the scaled (by"
                                                  " EFAC) TOA uncertainty."))

        self.add_param(p.maskParameter(name='TNEQ', \
                                       units=u.LogUnit(physical_unit=u.second),\
                                       description="An error term added in "
                                                  "quadrature to the scaled (by"
                                                  " EFAC) TOA uncertainty in "
                                                  " the unit of log10(second)."))
        self.covariance_matrix_funcs += [self.sigma_scaled_cov_matrix, ]
        self.scaled_sigma_funcs += [self.scale_sigma, ]

    def setup(self):
        super(ScaleToaError, self).setup()
        # Get all the EFAC parameters and EQUAD
        self.EFACs = {}
        self.EQUADs = {}
        self.TNEQs = {}
        for mask_par in self.get_params_of_type('maskParameter'):
            if mask_par.startswith('EFAC'):
                par = getattr(self, mask_par)
                self.EFACs[mask_par] = (par.key, par.key_value)
            elif mask_par.startswith('EQUAD'):
                par = getattr(self, mask_par)
                self.EQUADs[mask_par] = (par.key, par.key_value)
            elif  mask_par.startswith('TNEQ'):
                par = getattr(self, mask_par)
                self.TNEQs[mask_par] = (par.key, par.key_value)
            else:
                continue
        # convert all the TNEQ to EQUAD

        for tneq in list(self.TNEQs.keys()):
            tneq_par = getattr(self, tneq)
            if tneq_par.key is None:
                continue
            if self.TNEQs[tneq] in list(self.EQUADs.values()):
                log.warn("'%s %s %s' is provided by parameter EQUAD, using" \
                         " EQUAD instead. " % (tneq, tneq_par.key, tneq_par.key_value))
            else:
                EQUAD_name = 'EQUAD' + str(tneq_par.index)
                if EQUAD_name in list(self.EQUADs.keys()):
                    EQUAD_par = getattr(self, EQUAD_name)
                    EQUAD_par.key = tneq_par.key
                    EQUAD_par.key_value = tneq_par.key_value
                    EQUAD_par.quantity = tneq_par.quantity.to(u.us)
                else:
                    self.add_param(p.maskParameter(name='EQUAD', units="us",
                                                   index=tneq_par.index,
                                                   aliases=['T2EQUAD'],
                                                   description="An error term "\
                                                   " added in quadrature to the"\
                                                   " scaled (by EFAC) TOA uncertainty."))
                    EQUAD_par = getattr(self, EQUAD_name)
                    EQUAD_par.key = tneq_par.key
                    EQUAD_par.key_value = tneq_par.key_value
                    EQUAD_par.quantity = tneq_par.quantity.to(u.us)
        for pp in self.params:
            if pp.startswith('EQUAD'):
                par = getattr(self, pp)
                self.EQUADs[pp] = (par.key, par.key_value)
        # check duplicate
        for el in ['EFACs', 'EQUADs']:
            l  = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError("'%s' have duplicated keys and key values." % el)

    # pairing up EFAC and EQUAD
    def pair_EFAC_EQUAD(self):
        pairs = []
        for efac, efac_key in list(self.EFACs.items()):
            for equad, equad_key in list(self.EQUADs.items()):
                if efac_key == equad_key:
                    pairs.append((getattr(self, efac), getattr(self, equad)))
        if len(pairs) != len(list(self.EFACs.items())):
            # TODO may be define an parameter error would be helpful
            raise ValueError("Can not pair up EFACs and EQUADs, please "
                             " check the EFAC/EQUAD keys and key values.")
        return pairs

    def scale_sigma(self, toas):
        sigma_old = toas['error'].quantity
        sigma_scaled = np.zeros_like(sigma_old)
        EF_EQ_pairs = self.pair_EFAC_EQUAD()
        for pir in EF_EQ_pairs:
            efac = pir[0]
            equad = pir[1]
            mask = efac.select_toa_mask(toas)
            sigma_scaled[mask] = efac.quantity * np.sqrt(sigma_old[mask] ** 2 + \
                                 (equad.quantity)**2)
        return sigma_scaled

    def sigma_scaled_cov_matrix(self, toas):
        scaled_sigma = self.scale_sigma(toas)
        return np.diag(scaled_sigma) * scaled_sigma.unit


class EcorrNoise(NoiseComponent):
    """This is a class to model ECORR type noise.
    Notes
    -----
    Ref: NanoGrav 11 yrs data
    """
    register = True
    def __init__(self,):
        super(EcorrNoise, self).__init__()
        self.add_param(p.maskParameter(name='ECORR', units="us",\
                                       aliases=['TNECORR'],
                                       description="An error term added that"\
                                                  " correlated all TOAs in an"\
                                                  " observing epoch."))


        self.covariance_matrix_funcs += [self.ecorr_cov_matrix, ]
        self.basis_funcs += [self.ecorr_basis_prior_pair, ]

    def setup(self):
        super(EcorrNoise, self).setup()
        # Get all the EFAC parameters and EQUAD
        self.ECORRs = {}
        for mask_par in self.get_params_of_type('maskParameter'):
            if mask_par.startswith('ECORR'):
                par = getattr(self, mask_par)
                self.ECORRs[mask_par] = (par.key, par.key_value)
            else:
                continue

        # check duplicate
        for el in ['ECORRs']:
            l  = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError("'%s' have duplicated keys and key values." % el)

    def get_ecorrs(self):
        ecorrs = []
        for ecorr, ecorr_key in list(self.ECORRs.items()):
            ecorrs.append(getattr(self, ecorr))
        return ecorrs

    def ecorr_basis_prior_pair(self, toas):
        t = (toas['tdbld'].quantity * u.day).to(u.s).value
        ecorrs = self.get_ecorrs()
        Umats = []
        for ec in ecorrs:
            mask = ec.select_toa_mask(toas)
            Umats.append(utils.create_quantization_matrix(t[mask]))
        nc = np.sum(U.shape[1] for U in Umats)
        Umat = np.zeros((len(t), nc))
        prior = np.zeros(nc)
        nctot = 0
        for ct, ec in enumerate(ecorrs):
            mask = ec.select_toa_mask(toas)
            nn = Umats[ct].shape[1]
            Umat[mask, nctot:nn+nctot] = Umats[ct]
            prior[nctot:nn+nctot] = ec.quantity.to(u.s).value ** 2
            nctot += nn
        return (Umat, prior)

    def ecorr_cov_matrix(self, toas):
        U, Jvec = self.ecorr_basis_prior_pair(toas)
        return np.dot(U * Jvec[None,:], U.T)
