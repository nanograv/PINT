"""Pulsar timing noise models."""

from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log

from pint.models.parameter import floatParameter, maskParameter
from pint.models.timing_model import Component


class NoiseComponent(Component):
    def __init__(self,):
        super(NoiseComponent, self).__init__()
        self.covariance_matrix_funcs = []
        self.scaled_toa_sigma_funcs = []  # Need to move this to a speical place.
        self.scaled_dm_sigma_funcs = []
        # TODO This works right now. But if we want to expend noise model, we
        # need to think about the design now. If we do not define the list
        # here and calling the same name from other component, it will get
        # it from the component that hosts it. It has the risk to dulicate
        # the list elements.
        self.dm_covariance_matrix_funcs_component = []
        self.basis_funcs = []

    def validate(self,):
        super(NoiseComponent, self).validate()


class ScaleToaError(NoiseComponent):
    """Correct reported template fitting uncertainties.

    Note
    ----
    Ref: NanoGrav 11 yrs data

    """

    register = True
    category = "scale_toa_error"

    def __init__(self,):
        super(ScaleToaError, self).__init__()
        self.introduces_correlated_errors = False
        self.add_param(
            maskParameter(
                name="EFAC",
                units="",
                aliases=["T2EFAC", "TNEF"],
                description="A multiplication factor on"
                " the measured TOA uncertainties,",
            )
        )

        self.add_param(
            maskParameter(
                name="EQUAD",
                units="us",
                aliases=["T2EQUAD"],
                description="An error term added in "
                "quadrature to the scaled (by"
                " EFAC) TOA uncertainty.",
            )
        )

        self.add_param(
            maskParameter(
                name="TNEQ",
                units=u.LogUnit(physical_unit=u.second),
                description="An error term added in "
                "quadrature to the scaled (by"
                " EFAC) TOA uncertainty in "
                " the unit of log10(second).",
            )
        )
        self.covariance_matrix_funcs += [self.sigma_scaled_cov_matrix]
        self.scaled_toa_sigma_funcs += [self.scale_toa_sigma]

    def setup(self):
        super(ScaleToaError, self).setup()
        # Get all the EFAC parameters and EQUAD
        self.EFACs = {}
        self.EQUADs = {}
        self.TNEQs = {}
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("EFAC"):
                par = getattr(self, mask_par)
                self.EFACs[mask_par] = (par.key, par.key_value)
            elif mask_par.startswith("EQUAD"):
                par = getattr(self, mask_par)
                self.EQUADs[mask_par] = (par.key, par.key_value)
            elif mask_par.startswith("TNEQ"):
                par = getattr(self, mask_par)
                self.TNEQs[mask_par] = (par.key, par.key_value)
            else:
                continue
        # convert all the TNEQ to EQUAD

        for tneq in self.TNEQs:
            tneq_par = getattr(self, tneq)
            if tneq_par.key is None:
                continue
            if self.TNEQs[tneq] in list(self.EQUADs.values()):
                log.warning(
                    "'%s %s %s' is provided by parameter EQUAD, using"
                    " EQUAD instead. " % (tneq, tneq_par.key, tneq_par.key_value)
                )
            else:
                EQUAD_name = "EQUAD" + str(tneq_par.index)
                if EQUAD_name in list(self.EQUADs.keys()):
                    EQUAD_par = getattr(self, EQUAD_name)
                    EQUAD_par.key = tneq_par.key
                    EQUAD_par.key_value = tneq_par.key_value
                    EQUAD_par.quantity = tneq_par.quantity.to(u.us)
                else:
                    self.add_param(
                        p.maskParameter(
                            name="EQUAD",
                            units="us",
                            index=tneq_par.index,
                            aliases=["T2EQUAD"],
                            description="An error term "
                            " added in quadrature to the"
                            " scaled (by EFAC) TOA uncertainty.",
                        )
                    )
                    EQUAD_par = getattr(self, EQUAD_name)
                    EQUAD_par.key = tneq_par.key
                    EQUAD_par.key_value = tneq_par.key_value
                    EQUAD_par.quantity = tneq_par.quantity.to(u.us)
        for pp in self.params:
            if pp.startswith("EQUAD"):
                par = getattr(self, pp)
                self.EQUADs[pp] = (par.key, par.key_value)

    def validate(self):
        super(ScaleToaError, self).validate()
        # check duplicate
        for el in ["EFACs", "EQUADs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError("'%s' have duplicated keys and key values." % el)

    def scale_toa_sigma(self, toas):
        sigma_scaled = toas.table["error"].quantity.copy()
        for equad_name in self.EQUADs:
            equad = getattr(self, equad_name)
            if equad.quantity is None:
                continue
            mask = equad.select_toa_mask(toas)
            sigma_scaled[mask] = np.hypot(sigma_scaled[mask], equad.quantity)
        for efac_name in self.EFACs:
            efac = getattr(self, efac_name)
            sigma_scaled[efac.select_toa_mask(toas)] *= efac.quantity
        return sigma_scaled

    def sigma_scaled_cov_matrix(self, toas):
        scaled_sigma = self.scale_toa_sigma(toas).to(u.s).value ** 2
        return np.diag(scaled_sigma)


class ScaleDmError(NoiseComponent):
    """Correction for estimated wideband DM measurement uncertainty.

    Note
    ----
    Ref: NanoGrav 12.5 yrs wideband data
    """

    register = True
    category = "scale_dm_error"

    def __init__(self,):
        super(ScaleDmError, self).__init__()
        self.introduces_correlated_errors = False
        self.add_param(
            maskParameter(
                name="DMEFAC",
                units="",
                description="A multiplication factor on"
                " the measured DM uncertainties,",
            )
        )

        self.add_param(
            maskParameter(
                name="DMEQUAD",
                units="pc / cm ^ 3",
                description="An error term added in "
                "quadrature to the scaled (by"
                " EFAC) TOA uncertainty.",
            )
        )

        self.dm_covariance_matrix_funcs_component = [self.dm_sigma_scaled_cov_matrix]
        self.scaled_dm_sigma_funcs += [self.scale_dm_sigma]
        self._paired_DMEFAC_DMEQUAD = None

    def setup(self):
        super(ScaleDmError, self).setup()
        # Get all the EFAC parameters and EQUAD
        self.DMEFACs = {}
        self.DMEQUADs = {}
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("DMEFAC"):
                par = getattr(self, mask_par)
                if par.key is not None:
                    self.DMEFACs[mask_par] = (par.key, tuple(par.key_value))
            elif mask_par.startswith("DMEQUAD"):
                par = getattr(self, mask_par)
                if par.key is not None:
                    self.DMEQUADs[mask_par] = (par.key, tuple(par.key_value))
            else:
                continue

        if len(self.DMEFACs) != len(self.DMEQUADs):
            self._match_DMEFAC_DMEQUAD()
        else:
            self._paired_DMEFAC_DMEQUAD = self.pair_DMEFAC_DMEQUAD()

    def validate(self):
        super(ScaleDmError, self).validate()
        # check duplicate
        for el in ["DMEFACs", "DMEQUADs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError("'%s' have duplicated keys and key values." % el)

    def _match_DMEFAC_DMEQUAD(self, add_param_to_model=False):
        """ Match the DMEFAC and DMEQUAD parameter, if only one parameter of the
        DMEFAC-DMEQUAD pair is given. This match is based on the parameters key
        and key value.

        Parameters
        ----------
        add_param_to_model: bool
            Flags to add the parameters to the timing model instead of use the
            default value temporarily. This is useful, if one wants to fit for
            the match up parameters.
        """
        keys_and_values = {}
        p_map = {0: (self.DMEFAC1, 1), 1: (self.DMEQUAD1, 0)}
        keys_and_values = self.pair_DMEFAC_DMEQUAD()
        # match params.
        for kvs, params in keys_and_values.items():
            if None in params:
                p_type = params.index(None)
                example_add_param = p_map[p_type][0]
                pair_param = params[1 - p_type]
                param_idx = pair_param.index
                param_name = example_add_param.prefix + str(param_idx)
                # search existing param but without any assigned keys
                add_param = example_add_param.new_param(param_idx)
                add_param.value = p_map[p_type][1]
                add_param.key = kvs[0]
                add_param.key_value = kvs[1]
                params[p_type] = add_param
                if add_param_to_model:
                    self.add_param(add_param)
        if add_param_to_model:
            self.setup()
        else:
            self._paired_DMEFAC_DMEQUAD = keys_and_values

    # pairing up EFAC and EQUAD
    def pair_DMEFAC_DMEQUAD(self):
        """ Pair the DMEFAC and DMEQUAD.
        """
        keys_and_values = {}
        # Check the dm efac first
        for dmefac, efac_key in self.DMEFACs.items():
            if efac_key[0] is not None:
                keys_and_values[efac_key] = [getattr(self, dmefac), None]
        # Check the dm equad then
        for dmequad, equad_key in self.DMEQUADs.items():
            if equad_key[0] is not None:
                # Add matches.
                if equad_key in keys_and_values.keys():
                    keys_and_values[equad_key][1] = getattr(self, dmequad)
                else:
                    keys_and_values[equad_key] = [None, getattr(self, dmequard)]

        return keys_and_values

    def scale_dm_sigma(self, toas):
        """
        Scale the DM uncertainty.

        Parameters
        ----------
        toas: `pint.toa.TOAs` object
            Input DM error object. We assume DM error is stored in the TOA
            objects.
        """
        sigma_old = toas.get_dm_errors()
        sigma_scaled = np.zeros_like(sigma_old)
        if self._paired_DMEFAC_DMEQUAD is None:
            self.setup()
        for pir in self._paired_DMEFAC_DMEQUAD.values():
            efac = pir[0]
            equad = pir[1]
            mask = efac.select_toa_mask(toas)
            sigma_scaled[mask] = efac.quantity * np.sqrt(
                sigma_old[mask] ** 2 + (equad.quantity) ** 2
            )
        return sigma_scaled

    def dm_sigma_scaled_cov_matrix(self, toas):
        scaled_sigma = self.scale_dm_sigma(toas).to_value(u.pc / u.cm ** 3) ** 2
        return np.diag(scaled_sigma)


class EcorrNoise(NoiseComponent):
    """Noise correlated between nearby TOAs.

    This can occur, for example, if multiple TOAs were taken at different
    frequencies simultaneously: pulsar intrinsic emission jitters back
    and forth within the average profile, and this effect is the same
    for all frequencies. Thus these TOAs have correlated errors.

    Note
    ----
    Ref: NanoGrav 11 yrs data

    """

    register = True
    category = "ecorr_noise"

    def __init__(self,):
        super(EcorrNoise, self).__init__()
        self.introduces_correlated_errors = True
        self.add_param(
            maskParameter(
                name="ECORR",
                units="us",
                aliases=["TNECORR"],
                description="An error term added that"
                " correlated all TOAs in an"
                " observing epoch.",
            )
        )

        self.covariance_matrix_funcs += [self.ecorr_cov_matrix]
        self.basis_funcs += [self.ecorr_basis_weight_pair]

    def setup(self):
        super(EcorrNoise, self).setup()
        # Get all the EFAC parameters and EQUAD
        self.ECORRs = {}
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("ECORR"):
                par = getattr(self, mask_par)
                self.ECORRs[mask_par] = (par.key, par.key_value)
            else:
                continue

    def validate(self):
        super(EcorrNoise, self).validate()

        # check duplicate
        for el in ["ECORRs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError("'%s' have duplicated keys and key values." % el)

    def get_ecorrs(self):
        ecorrs = []
        for ecorr, ecorr_key in list(self.ECORRs.items()):
            ecorrs.append(getattr(self, ecorr))
        return ecorrs

    def ecorr_basis_weight_pair(self, toas):
        """Return a quantization matrix and ECORR weights.

        A quantization matrix maps TOAs to observing epochs.
        The weights used are the square of the ECORR values.

        """
        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        ecorrs = self.get_ecorrs()
        umats = []
        for ec in ecorrs:
            mask = ec.select_toa_mask(toas)
            umats.append(create_quantization_matrix(t[mask]))
        nc = sum(u.shape[1] for u in umats)
        umat = np.zeros((len(t), nc))
        weight = np.zeros(nc)
        nctot = 0
        for ct, ec in enumerate(ecorrs):
            mask = ec.select_toa_mask(toas)
            nn = umats[ct].shape[1]
            umat[mask, nctot : nn + nctot] = umats[ct]
            weight[nctot : nn + nctot] = ec.quantity.to(u.s).value ** 2
            nctot += nn
        return (umat, weight)

    def ecorr_cov_matrix(self, toas):
        """Full ECORR covariance matrix."""
        U, Jvec = self.ecorr_basis_weight_pair(toas)
        return np.dot(U * Jvec[None, :], U.T)


class PLRedNoise(NoiseComponent):
    """Timing noise with a power-law spectrum.

    Over the long term, pulsars are observed to experience timing noise
    dominated by low frequencies. This can occur, for example, if the
    torque on the pulsar varies randomly. If the torque experiences
    white noise, the phase we observe will experience "red" noise, that
    is noise dominated by the lowest frequency. This results in errors
    that are correlated between TOAs over fairly long time spans.

    Note
    ----
    Ref: NanoGrav 11 yrs data

    """

    register = True
    category = "pl_red_noise"

    def __init__(self,):
        super(PLRedNoise, self).__init__()
        self.introduces_correlated_errors = True
        self.add_param(
            floatParameter(
                name="RNAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw " "red noise.",
            )
        )
        self.add_param(
            floatParameter(
                name="RNIDX",
                units="",
                aliases=[],
                description="Spectral index of " "powerlaw red noise.",
            )
        )

        self.add_param(
            floatParameter(
                name="TNRedAmp",
                units="",
                aliases=[],
                description="Amplitude of powerlaw " "red noise in tempo2 format",
            )
        )
        self.add_param(
            floatParameter(
                name="TNRedGam",
                units="",
                aliases=[],
                description="Spectral index of powerlaw " "red noise in tempo2 format",
            )
        )
        self.add_param(
            floatParameter(
                name="TNRedC",
                units="",
                aliases=[],
                description="Number of red noise frequencies.",
            )
        )

        self.covariance_matrix_funcs += [self.pl_rn_cov_matrix]
        self.basis_funcs += [self.pl_rn_basis_weight_pair]

    def setup(self):
        super(PLRedNoise, self).setup()

    def validate(self):
        super(PLRedNoise, self).validate()

    def get_pl_vals(self):
        nf = int(self.TNRedC.value) if self.TNRedC.value is not None else 30
        if self.TNRedAmp.value is not None and self.TNRedGam.value is not None:
            amp, gam = 10 ** self.TNRedAmp.value, self.TNRedGam.value
        elif self.RNAMP.value is not None and self.RNIDX is not None:
            fac = (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0))
            amp, gam = self.RNAMP.value / fac, -1 * self.RNIDX.value
        return (amp, gam, nf)

    def pl_rn_basis_weight_pair(self, toas):
        """Return a Fourier design matrix and red noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion.
        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNRedC] and T is the total observing duration of
        the dataset.

        """
        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        amp, gam, nf = self.get_pl_vals()
        Fmat, f = create_fourier_design_matrix(t, nf)
        weight = powerlaw(f, amp, gam) * f[0]
        return (Fmat, weight)

    def pl_rn_cov_matrix(self, toas):
        Fmat, phi = self.pl_rn_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


def create_quantization_matrix(toas_table, dt=1, nmin=2):
    """Create quantization matrix mapping TOAs to observing epochs."""
    isort = np.argsort(toas_table)

    bucket_ref = [toas_table[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas_table[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas_table[i])
            bucket_ind.append([i])

    # find only epochs with more than 1 TOA
    bucket_ind2 = [ind for ind in bucket_ind if len(ind) >= nmin]

    U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
    for i, l in enumerate(bucket_ind2):
        U[l, i] = 1

    return U


def create_fourier_design_matrix(t, nmodes, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    f = np.linspace(1 / T, nmodes / T, nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    F[:, ::2] = np.sin(2 * np.pi * t[:, None] * f[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * t[:, None] * f[None, :])

    return F, Ffreqs


def powerlaw(f, A=1e-16, gamma=5):
    """Power-law PSD.

    :param f: Sampling frequencies
    :param A: Amplitude of red noise [GW units]
    :param gamma: Spectral index of red noise process
    """

    fyr = 1 / 3.16e7
    return A ** 2 / 12.0 / np.pi ** 2 * fyr ** (gamma - 3) * f ** (-gamma)
