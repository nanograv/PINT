"""Pulsar timing noise models."""

import copy
from typing import Callable, List, Optional, Tuple
import warnings

import astropy.units as u
import numpy as np
from loguru import logger as log

from pint import DMconst
from pint.models.parameter import Parameter, floatParameter, intParameter, maskParameter
from pint.models.timing_model import Component
from pint.toa import TOAs


class NoiseComponent(Component):

    introduces_dm_errors = False

    def __init__(
        self,
    ):
        super().__init__()
        self.covariance_matrix_funcs = []
        self.scaled_toa_sigma_funcs = []  # Need to move this to a special place.
        self.scaled_dm_sigma_funcs = []
        # TODO This works right now. But if we want to expend noise model, we
        # need to think about the design now. If we do not define the list
        # here and calling the same name from other component, it will get
        # it from the component that hosts it. It has the risk to duplicate
        # the list elements.
        self.dm_covariance_matrix_funcs_component = []
        self.basis_funcs = []


class ScaleToaError(NoiseComponent):
    """Correct reported template fitting uncertainties.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.ScaleToaError

    Note
    ----
    Ref: NANOGrav 11 yrs data

    """

    register = True
    category = "scale_toa_error"

    introduces_correlated_errors = False

    def __init__(
        self,
    ):
        super().__init__()

        self.add_param(
            maskParameter(
                name="EFAC",
                units="",
                aliases=["T2EFAC", "TNEF"],
                description="A multiplication factor on the measured TOA uncertainties,",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            maskParameter(
                name="EQUAD",
                units="us",
                aliases=["T2EQUAD"],
                description="An error term added in quadrature to the scaled (by EFAC) TOA uncertainty.",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            maskParameter(
                name="TNEQ",
                units=u.LogUnit(physical_unit=u.second),
                description="An error term added in quadrature to the scaled (by EFAC) TOA uncertainty in units of log10(second).",
                convert_tcb2tdb=False,
            )
        )
        self.covariance_matrix_funcs += [self.sigma_scaled_cov_matrix]
        self.scaled_toa_sigma_funcs += [self.scale_toa_sigma]
        self.toasigma_deriv_funcs = {}

    def setup(self):
        super().setup()
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

        for tneq, value in self.TNEQs.items():
            tneq_par = getattr(self, tneq)
            if tneq_par.key is None:
                continue
            if value in list(self.EQUADs.values()):
                log.warning(
                    f"'{tneq} {tneq_par.key} {tneq_par.key_value}' is provided by parameter EQUAD, using EQUAD instead. "
                )
            else:
                EQUAD_name = f"EQUAD{str(tneq_par.index)}"
                if EQUAD_name not in list(self.EQUADs.keys()):
                    self.add_param(
                        maskParameter(
                            name="EQUAD",
                            units="us",
                            index=tneq_par.index,
                            aliases=["T2EQUAD"],
                            description="An error term added in quadrature to the scaled (by EFAC) TOA uncertainty.",
                            convert_tcb2tdb=False,
                        )
                    )
                EQUAD_par = getattr(self, EQUAD_name)
                EQUAD_par.quantity = tneq_par.quantity.to(u.us)
                EQUAD_par.key_value = tneq_par.key_value
                EQUAD_par.key = tneq_par.key
        for pp in self.params:
            if pp.startswith("EQUAD"):
                par = getattr(self, pp)
                self.EQUADs[pp] = (par.key, par.key_value)

        for ef in self.EFACs:
            self.register_toasigma_deriv_funcs(self.d_toasigma_d_EFAC, ef)

        for eq in self.EQUADs:
            self.register_toasigma_deriv_funcs(self.d_toasigma_d_EQUAD, eq)

    def register_toasigma_deriv_funcs(self, func: Callable, param: str):
        pn = self.match_param_aliases(param)
        if pn not in list(self.toasigma_deriv_funcs.keys()):
            self.toasigma_deriv_funcs[pn] = [func]
        elif func in self.toasigma_deriv_funcs[pn]:
            return
        else:
            self.toasigma_deriv_funcs[pn] += [func]

    def validate(self):
        super().validate()
        # check duplicate
        for el in ["EFACs", "EQUADs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError(f"'{el}' have duplicated keys and key values.")

    def scale_toa_sigma(self, toas: TOAs, warn: bool = True) -> u.Quantity:
        sigma_scaled = toas.table["error"].quantity.copy()
        for equad_name in self.EQUADs:
            equad = getattr(self, equad_name)
            if equad.quantity is None:
                continue
            mask = equad.select_toa_mask(toas)
            if len(mask) > 0:
                sigma_scaled[mask] = np.hypot(sigma_scaled[mask], equad.quantity)
            elif warn:
                warnings.warn(f"EQUAD {equad} has no TOAs")
        for efac_name in self.EFACs:
            efac = getattr(self, efac_name)
            mask = efac.select_toa_mask(toas)
            if len(mask) > 0:
                sigma_scaled[mask] *= efac.quantity
            elif warn:
                warnings.warn(f"EFAC {efac} has no TOAs")
        return sigma_scaled

    def sigma_scaled_cov_matrix(self, toas: TOAs) -> np.ndarray:
        scaled_sigma = self.scale_toa_sigma(toas).to(u.s).value ** 2
        return np.diag(scaled_sigma)

    def d_toasigma_d_EFAC(self, toas: TOAs, param: str) -> u.Quantity:
        par = getattr(self, param)
        mask = par.select_toa_mask(toas)
        result = np.zeros(len(toas)) << u.s
        result[mask] = self.scale_toa_sigma(toas[mask], warn=False).to(
            u.s
        ) / par.quantity.to(u.dimensionless_unscaled)
        return result

    def d_toasigma_d_EQUAD(self, toas: TOAs, param: str) -> u.Quantity:
        par = getattr(self, param)
        mask = par.select_toa_mask(toas)
        toas_mask = toas[mask]

        result = np.zeros(len(toas)) << u.dimensionless_unscaled

        sigma_mask = self.scale_toa_sigma(toas_mask, warn=False)

        sigma2_mask_noefac = toas_mask.get_errors().to(u.s) ** 2
        for equad_name in self.EQUADs:
            equad = getattr(self, equad_name)
            if equad.quantity is None:
                continue
            eqmask = equad.select_toa_mask(toas_mask)
            if np.any(eqmask):
                sigma2_mask_noefac[eqmask] += equad.quantity**2

        result[mask] = (sigma_mask * par.quantity / sigma2_mask_noefac).to(
            u.dimensionless_unscaled
        )

        return result


class ScaleDmError(NoiseComponent):
    """Correction for estimated wideband DM measurement uncertainty.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.ScaleDmError

    Note
    ----
    Ref: NANOGrav 12.5 yrs wideband data
    """

    register = True
    category = "scale_dm_error"

    introduces_correlated_errors = False
    introduces_dm_errors = True

    def __init__(
        self,
    ):
        super().__init__()
        self.add_param(
            maskParameter(
                name="DMEFAC",
                units="",
                description="A multiplication factor on the measured DM uncertainties,",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            maskParameter(
                name="DMEQUAD",
                units="pc / cm ^ 3",
                description="An error term added in quadrature to the scaled (by EFAC) TOA uncertainty.",
                convert_tcb2tdb=False,
            )
        )

        self.dm_covariance_matrix_funcs_component = [self.dm_sigma_scaled_cov_matrix]
        self.scaled_dm_sigma_funcs += [self.scale_dm_sigma]
        self._paired_DMEFAC_DMEQUAD = None

    def setup(self):
        super().setup()
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

        # if len(self.DMEFACs) != len(self.DMEQUADs):
        #     self._match_DMEFAC_DMEQUAD()
        # else:
        #     self._paired_DMEFAC_DMEQUAD = self.pair_DMEFAC_DMEQUAD()

    def validate(self):
        super().validate()
        # check duplicate
        for el in ["DMEFACs", "DMEQUADs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError(f"'{el}' have duplicated keys and key values.")

    def scale_dm_sigma(self, toas: TOAs) -> u.Quantity:
        """
        Scale the DM uncertainty.

        Parameters
        ----------
        toas: `pint.toa.TOAs` object
            Input DM error object. We assume DM error is stored in the TOA
            objects.
        """
        sigma_scaled = copy.deepcopy(toas.get_dm_errors())
        # Apply DMEQUAD first
        for dmequad_name in self.DMEQUADs:
            dmequad = getattr(self, dmequad_name)
            if dmequad.quantity is None:
                continue
            mask = dmequad.select_toa_mask(toas)
            sigma_scaled[mask] = np.hypot(sigma_scaled[mask], dmequad.quantity)
        # Then apply the DMEFAC
        for dmefac_name in self.DMEFACs:
            dmefac = getattr(self, dmefac_name)
            sigma_scaled[dmefac.select_toa_mask(toas)] *= dmefac.quantity
        return sigma_scaled

    def dm_sigma_scaled_cov_matrix(self, toas: TOAs) -> np.ndarray:
        scaled_sigma = self.scale_dm_sigma(toas).to_value(u.pc / u.cm**3) ** 2
        return np.diag(scaled_sigma)


class EcorrNoise(NoiseComponent):
    """Noise correlated between nearby TOAs.

    This can occur, for example, if multiple TOAs were taken at different
    frequencies simultaneously: pulsar intrinsic emission jitters back
    and forth within the average profile, and this effect is the same
    for all frequencies. Thus these TOAs have correlated errors.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.EcorrNoise

    Note
    ----
    Ref: NANOGrav 11 yrs data

    """

    register = True
    category = "ecorr_noise"

    introduces_correlated_errors = True
    is_time_correlated = False

    def __init__(
        self,
    ):
        super().__init__()
        self.add_param(
            maskParameter(
                name="ECORR",
                units="us",
                aliases=["TNECORR"],
                description="An error term that is correlated among all TOAs in an observing epoch.",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.ecorr_cov_matrix]
        self.basis_funcs += [self.ecorr_basis_weight_pair]

    def setup(self):
        super().setup()
        # Get all the EFAC parameters and EQUAD
        self.ECORRs = {}
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("ECORR"):
                par = getattr(self, mask_par)
                self.ECORRs[mask_par] = (par.key, par.key_value)
            else:
                continue

    def validate(self):
        super().validate()

        # check duplicate
        for el in ["ECORRs"]:
            l = list(getattr(self, el).values())
            if [x for x in l if l.count(x) > 1] != []:
                raise ValueError(f"'{el}' have duplicated keys and key values.")

    def get_ecorrs(self) -> List[Parameter]:
        return [getattr(self, ecorr) for ecorr in self.ECORRs.keys()]

    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return the quantization matrix for ECORR.

        A quantization matrix maps TOAs to observing epochs.
        """
        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        ecorrs = self.get_ecorrs()
        umats = []
        for ec in ecorrs:
            mask = ec.select_toa_mask(toas)
            if np.any(mask):
                umats.append(create_ecorr_quantization_matrix(t[mask]))
            else:
                warnings.warn(f"ECORR {ec} has no TOAs")
                umats.append(np.zeros((0, 0)))
        nc = sum(u.shape[1] for u in umats)
        umat = np.zeros((len(t), nc))
        nctot = 0
        for ct, ec in enumerate(ecorrs):
            mask = ec.select_toa_mask(toas)
            nn = umats[ct].shape[1]
            umat[mask, nctot : nn + nctot] = umats[ct]
            nctot += nn
        return umat

    def get_noise_weights(self, toas: TOAs, nweights: int = None) -> np.ndarray:
        """Return the ECORR weights
        The weights used are the square of the ECORR values.
        """
        ecorrs = self.get_ecorrs()
        if nweights is None:
            ts = (toas.table["tdbld"].quantity * u.day).to(u.s).value
            nweights = [
                get_ecorr_nweights(ts[ec.select_toa_mask(toas)]) for ec in ecorrs
            ]
        nc = sum(nweights)
        weights = np.zeros(nc)
        nctot = 0
        for ec, nn in zip(ecorrs, nweights):
            weights[nctot : nn + nctot] = ec.quantity.to(u.s).value ** 2
            nctot += nn
        return weights

    def ecorr_basis_weight_pair(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return a quantization matrix and ECORR weights.

        A quantization matrix maps TOAs to observing epochs.
        The weights used are the square of the ECORR values.
        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def ecorr_cov_matrix(self, toas: TOAs) -> np.ndarray:
        """Full ECORR covariance matrix."""
        U, Jvec = self.ecorr_basis_weight_pair(toas)
        return np.dot(U * Jvec[None, :], U.T)


class PLDMNoise(NoiseComponent):
    """Model of DM variations as radio frequency-dependent noise with a power-law spectrum.

    Variations in DM over time result from both the proper motion of the
    pulsar and the changing electron number density along the line of sight
    from the solar wind and ISM. In particular, Kolmogorov turbulence in the
    ionized ISM will induce stochastic DM variations with a power law
    spectrum. Timing errors due to unmodelled DM variations can therefore
    appear very similar to intrinsic red noise, however the amplitude of these
    variations will scale with the inverse of the square of the (Earth Doppler
    corrected) radio frequency.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.PLDMNoise

    References
    ----------
    - Lentati et al. 2014, MNRAS 437(3), 3004-3023 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3004L/abstract

    """

    register = True
    category = "pl_DM_noise"

    introduces_correlated_errors = True
    introduces_dm_errors = True
    is_time_correlated = True

    def __init__(
        self,
    ):
        super().__init__()

        self.add_param(
            floatParameter(
                name="TNDMAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw DM noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNDMGAM",
                units="",
                aliases=[],
                description="Spectral index of powerlaw DM noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            intParameter(
                name="TNDMC",
                units="",
                aliases=[],
                description="Number of DM noise frequencies.",
            )
        )
        self.add_param(
            intParameter(
                name="TNDMFLOG",
                units="",
                description="Number of logarithmically spaced DM noise frequencies in the basis.",
            )
        )
        self.add_param(
            floatParameter(
                name="TNDMFLOG_FACTOR",
                units="",
                description="Scaling factor for the log-spaced DM frequencies (2 -> [1/8, 1/4, 1/2, ...]).",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNDMTSPAN",
                units="year",
                description="Time span corresponding to the fundamental frequency of the DM noise Fourier series (data span is used by default).",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.covariance_matrix_funcs += [self.pl_dm_cov_matrix]
        self.basis_funcs += [self.pl_dm_basis_weight_pair]

    def get_plc_vals(self) -> Tuple[float, float, int, int, float]:
        """
        Retrieve power-law parameters and frequency-basis parameters
        from the model, substituting defaults if unspecified.
        """
        n_lin = int(self.TNDMC.value) if self.TNDMC.value is not None else 30
        n_log = int(self.TNDMFLOG.value) if (self.TNDMFLOG.value is not None) else None
        dm_log_factor = (
            self.TNDMFLOG_FACTOR.value
            if (self.TNDMFLOG_FACTOR.value is not None)
            else 2
        )
        amp, gam = 10**self.TNDMAMP.value, self.TNDMGAM.value
        f_min_ratio = 1 / (dm_log_factor**n_log) if n_log is not None else 1

        return amp, gam, n_lin, n_log, f_min_ratio

    def get_time_frequencies(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return the frequencies of the noise model"""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        T = (
            np.max(t) - np.min(t)
            if self.TNDMTSPAN.quantity is None
            else self.TNDMTSPAN.quantity
        )

        (_, _, n_lin, n_log, f_min_ratio) = self.get_plc_vals()
        f_min = f_min_ratio / T

        return t, get_rednoise_freqs(
            t, n_lin, Tspan=T, logmode=0, f_min=f_min, nlog=n_log
        )

    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return a Fourier design matrix for DM noise.

        See the documentation for pl_dm_basis_weight_pair function for details."""

        t, f = self.get_time_frequencies(toas)
        Fmat = create_fourier_design_matrix(t, f)
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        fref = 1400 * u.MHz
        D = (fref.value / freqs.value) ** 2

        return Fmat * D[:, None]

    def get_noise_weights(self, toas: TOAs) -> np.ndarray:
        """Return power law DM noise weights.

        See the documentation for pl_dm_basis_weight_pair for details."""

        (amp, gam, _, _, _) = self.get_plc_vals()
        _, f = self.get_time_frequencies(toas)
        df = np.diff(np.concatenate([[0], f]))

        return powerlaw(f.repeat(2), amp, gam) * df.repeat(2)

    def pl_dm_basis_weight_pair(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return a Fourier design matrix and power law DM noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion. Here we scale the design matrix by
        (fref/f)**2, where fref = 1400 MHz to match the convention used in
        enterprise.

        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNDMC] and T is the total observing duration of
        the dataset.

        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def pl_dm_cov_matrix(self, toas: TOAs) -> np.ndarray:
        Fmat, phi = self.pl_dm_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


class PLSWNoise(NoiseComponent):
    """Model of solar wind DM variations as radio frequency-dependent noise with a
    power-law spectrum.

    Commonly used as perturbations on top of a deterministic solar wind model.


    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.PLSWNoise

    References
    ----------
    - Lentati et al. 2014, MNRAS 437(3), 3004-3023 [1]_
    - van Haasteren & Vallisneri, 2014, MNRAS 446(2), 1170-1174 [2]_
    - Hazboun et al. 2022, APJ, Volume 929, Issue 1, id.39, 11 pp. [3]_
    - Susurla et al. 2024, A&A, Volume 692, id.A18, 18 pp.[4]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3004L/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1170V/abstract
    .. [3] https://ui.adsabs.harvard.edu/abs/2022ApJ...929...39H/abstract
    .. [4] https://ui.adsabs.harvard.edu/abs/2024A%26A...692A..18S/abstract
    """

    register = True
    category = "pl_SW_noise"

    introduces_correlated_errors = True
    is_time_correlated = True

    def __init__(
        self,
    ):
        super().__init__()

        self.add_param(
            floatParameter(
                name="TNSWAMP",
                units="",
                aliases=[],
                description="Amplitude of power-law SW DM noise in tempo2 format",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNSWGAM",
                units="",
                aliases=[],
                description="Spectral index of power-law "
                "SW DM noise in tempo2 format",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNSWC",
                units="",
                aliases=[],
                description="Number of SW DM noise frequencies.",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNSWFLOG",
                units="",
                description="Number of logarithmically solar wind frequencies in the basis.",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNSWFLOG_FACTOR",
                units="",
                description="Scaling factor for the log-spaced solar wind frequencies (2 -> [1/8,1/4,1/2,...])",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.pl_sw_cov_matrix]
        self.basis_funcs += [self.pl_sw_basis_weight_pair]

    def get_plc_vals(self) -> Tuple[float, float, int, int, float]:
        """
        Retrieve power-law parameters and frequency-basis parameters
        from the model, substituting defaults if unspecified.
        """
        n_lin = int(self.TNSWC.value) if self.TNSWC.value is not None else 100
        n_log = int(self.TNSWFLOG.value) if (self.TNSWFLOG.value is not None) else None
        sw_log_factor = (
            self.TNSWFLOG_FACTOR.value
            if (self.TNSWFLOG_FACTOR.value is not None)
            else 2
        )
        amp, gam = 10**self.TNSWAMP.value, self.TNSWGAM.value
        f_min_ratio = 1 / (sw_log_factor**n_log) if n_log is not None else 1

        return amp, gam, n_lin, n_log, f_min_ratio

    def get_time_frequencies(self, toas: TOAs) -> np.ndarray:
        """Return the frequencies of the noise model"""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        T = np.max(t) - np.min(t)

        (_, _, n_lin, n_log, f_min_ratio) = self.get_plc_vals()
        f_min = f_min_ratio / T

        return t, get_rednoise_freqs(
            t, n_lin, Tspan=T, logmode=0, f_min=f_min, nlog=n_log
        )

    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return a Fourier design matrix for SW DM noise.

        See the documentation for pl_sw_basis_weight_pair function for details."""

        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        # get the achromatic Fourier design matrix
        t, f = self.get_time_frequencies(toas)
        Fmat = create_fourier_design_matrix(t, f)
        # get solar wind geometry from pint.models.solar_wind_dispersion.SolarWindDispersion
        solar_wind_geometry = self._parent.solar_wind_geometry(toas)
        # since this is the SW DM value if n_earth = 1 cm^-3. the GP will scale it.
        dt_DM = (solar_wind_geometry * DMconst / (freqs**2)).value

        return Fmat * dt_DM[:, None]

    def get_noise_weights(self, toas: TOAs) -> np.ndarray:
        """Return power law SW noise weights.

        See the documentation for pl_sw_basis_weight_pair for details."""

        (amp, gam, _, _, _) = self.get_plc_vals()
        _, f = self.get_time_frequencies(toas)
        df = np.diff(np.concatenate([[0], f]))

        return powerlaw(f.repeat(2), amp, gam) * df.repeat(2)

    def pl_sw_basis_weight_pair(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return a Fourier design matrix and power law SW noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion. Here we scale the design matrix by
        (fref/f)**2 and a geometric factor where fref = 1400 MHz
        to match the convention used in enterprise.

        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNSWC] and T is the total observing duration of
        the dataset.

        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def pl_sw_cov_matrix(self, toas: TOAs) -> np.ndarray:
        Fmat, phi = self.pl_sw_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


class PLChromNoise(NoiseComponent):
    """Model of a radio frequency-dependent noise with a power-law spectrum and arbitrary chromatic index.

    Such variations are usually attributed to time-variable scattering in the
    ISM. Scattering smears/broadens the shape of the pulse profile by convolving it with
    a transfer function that is determined by the geometry and electron distribution
    in the scattering screen(s). The scattering timescale is typically a decreasing
    function of the observing frequency.

    Scatter broadening causes systematic offsets in the TOA measurements due to the
    pulse shape mismatch. While this offset need not be a simple function of frequency,
    it has been often modeled using a delay that is proportional to f^-alpha where alpha
    is known as the chromatic index.

    This model should be used in combination with the ChromaticCM model.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.PLChromNoise

    References
    ----------
    - Lentati et al. 2014, MNRAS 437(3), 3004-3023 [1]_
    - van Haasteren & Vallisneri, 2014, MNRAS 446(2), 1170-1174 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3004L/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1170V/abstract
    """

    register = True
    category = "pl_chrom_noise"

    introduces_correlated_errors = True
    is_time_correlated = True

    def __init__(
        self,
    ):
        super().__init__()

        self.add_param(
            floatParameter(
                name="TNCHROMAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw chromatic noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMGAM",
                units="",
                aliases=[],
                description="Spectral index of powerlaw chromatic noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            intParameter(
                name="TNCHROMC",
                units="",
                aliases=[],
                description="Number of chromatic noise frequencies.",
            )
        )
        self.add_param(
            intParameter(
                name="TNCHROMFLOG",
                units="",
                description="Number of logarithmically spaced chromatic noise frequencies in the basis.",
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMFLOG_FACTOR",
                units="",
                description="Scaling factor for the log-spaced chromatic frequencies (2 -> [1/8,1/4,1/2,...])",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMTSPAN",
                units="year",
                description="Time span corresponding to the fundamental frequency of the chromatic noise Fourier series (data span is used by default).",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.covariance_matrix_funcs += [self.pl_chrom_cov_matrix]
        self.basis_funcs += [self.pl_chrom_basis_weight_pair]

    def get_plc_vals(self) -> Tuple[float, float, int, int, float]:
        """
        Retrieve power-law parameters and frequency-basis parameters
        from the model, substituting defaults if unspecified.
        """
        n_lin = int(self.TNCHROMC.value) if self.TNCHROMC.value is not None else 30
        n_log = (
            int(self.TNCHROMFLOG.value)
            if (self.TNCHROMFLOG.value is not None)
            else None
        )
        chrom_log_factor = (
            self.TNCHROMFLOG_FACTOR.value
            if (self.TNCHROMFLOG_FACTOR.value is not None)
            else 2
        )
        amp, gam = 10**self.TNCHROMAMP.value, self.TNCHROMGAM.value
        f_min_ratio = 1 / (chrom_log_factor**n_log) if n_log is not None else 1

        return amp, gam, n_lin, n_log, f_min_ratio

    def get_time_frequencies(self, toas: TOAs) -> np.ndarray:
        """Return the frequencies of the noise model"""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        T = (
            np.max(t) - np.min(t)
            if self.TNCHROMTSPAN.quantity is None
            else self.TNCHROMTSPAN.quantity
        )

        (_, _, n_lin, n_log, f_min_ratio) = self.get_plc_vals()
        f_min = f_min_ratio / T

        return t, get_rednoise_freqs(
            t, n_lin, Tspan=T, logmode=0, f_min=f_min, nlog=n_log
        )

    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return a Fourier design matrix for chromatic noise.

        See the documentation for pl_chrom_basis_weight_pair function for details."""

        t, f = self.get_time_frequencies(toas)
        Fmat = create_fourier_design_matrix(t, f)
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        fref = 1400 * u.MHz
        alpha = self._parent.TNCHROMIDX.value
        D = (fref.value / freqs.value) ** alpha

        return Fmat * D[:, None]

    def get_noise_weights(self, toas: TOAs) -> np.ndarray:
        """Return power law chromatic noise weights.

        See the documentation for pl_chrom_basis_weight_pair for details."""

        (amp, gam, _, _, _) = self.get_plc_vals()
        _, f = self.get_time_frequencies(toas)
        df = np.diff(np.concatenate([[0], f]))

        return powerlaw(f.repeat(2), amp, gam) * df.repeat(2)

    def pl_chrom_basis_weight_pair(self, toas: TOAs) -> np.ndarray:
        """Return a Fourier design matrix and power law chromatic noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion. Here we scale the design matrix by
        (fref/f)**2, where fref = 1400 MHz to match the convention used in
        enterprise.

        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNCHROMC] and T is the total observing duration of
        the dataset.

        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def pl_chrom_cov_matrix(self, toas: TOAs) -> np.ndarray:
        Fmat, phi = self.pl_chrom_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


class PLRedNoise(NoiseComponent):
    """Timing noise with a power-law spectrum.

    Over the long term, pulsars are observed to experience timing noise
    dominated by low frequencies. This can occur, for example, if the
    torque on the pulsar varies randomly. If the torque experiences
    white noise, the phase we observe will experience "red" noise, that
    is noise dominated by the lowest frequency. This results in errors
    that are correlated between TOAs over fairly long time spans.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.PLRedNoise

    References
    ----------
    - Lentati et al. 2014, MNRAS 437(3), 3004-3023 [1]_
    - van Haasteren & Vallisneri, 2014, MNRAS 446(2), 1170-1174 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3004L/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1170V/abstract
    """

    register = True
    category = "pl_red_noise"

    introduces_correlated_errors = True
    is_time_correlated = True

    def __init__(
        self,
    ):
        super().__init__()

        self.add_param(
            floatParameter(
                name="RNAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw red noise.",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="RNIDX",
                units="",
                aliases=[],
                description="Spectral index of powerlaw red noise.",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.add_param(
            floatParameter(
                name="TNREDAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw red noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNREDGAM",
                units="",
                aliases=[],
                description="Spectral index of powerlaw red noise in tempo2 format",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            intParameter(
                name="TNREDC",
                units="",
                aliases=[],
                description="Number of red noise frequencies.",
            )
        )
        self.add_param(
            intParameter(
                name="TNREDFLOG",
                units="",
                description="Number of logarithmically spaced red noise frequencies in the basis.",
            )
        )
        self.add_param(
            floatParameter(
                name="TNREDFLOG_FACTOR",
                units="",
                description="Scaling factor for the log-spaced frequencies (2 -> [1/8,1/4,1/2,...])",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNREDTSPAN",
                units="year",
                description="Time span corresponding to the fundamental frequency of the achromatic red noise Fourier series (data span is used by default).",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.covariance_matrix_funcs += [self.pl_rn_cov_matrix]
        self.basis_funcs += [self.pl_rn_basis_weight_pair]

    def get_plc_vals(self) -> Tuple[float, float, int, int, float]:
        """
        Retrieve power-law parameters and frequency-basis parameters
        from the model, substituting defaults if unspecified.
        """
        n_lin = int(self.TNREDC.value) if self.TNREDC.value is not None else 30
        n_log = (
            int(self.TNREDFLOG.value) if (self.TNREDFLOG.value is not None) else None
        )
        red_log_factor = (
            self.TNREDFLOG_FACTOR.value
            if (self.TNREDFLOG_FACTOR.value is not None)
            else 2
        )

        if self.TNREDAMP.value is not None and self.TNREDGAM.value is not None:
            amp, gam = 10**self.TNREDAMP.value, self.TNREDGAM.value
        elif self.RNAMP.value is not None and self.RNIDX is not None:
            fac = (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0))
            amp, gam = self.RNAMP.value / fac, -1 * self.RNIDX.value

        f_min_ratio = 1 / (red_log_factor**n_log) if n_log is not None else 1

        return amp, gam, n_lin, n_log, f_min_ratio

    def get_time_frequencies(self, toas: TOAs) -> np.ndarray:
        """Return the frequencies of the noise model"""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        T = (
            np.max(t) - np.min(t)
            if self.TNREDTSPAN.quantity is None
            else self.TNREDTSPAN.quantity
        )

        (_, _, n_lin, n_log, f_min_ratio) = self.get_plc_vals()
        f_min = f_min_ratio / T

        return t, get_rednoise_freqs(
            t, n_lin, Tspan=T, logmode=0, f_min=f_min, nlog=n_log
        )

    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return a Fourier design matrix for red noise.

        See the documentation for pl_rn_basis_weight_pair function for details."""

        t, f = self.get_time_frequencies(toas)
        Fmat = create_fourier_design_matrix(t, f)

        return Fmat

    def get_noise_weights(self, toas: TOAs) -> np.ndarray:
        """Return power law red noise weights.

        See the documentation for pl_rn_basis_weight_pair for details."""

        (amp, gam, _, _, _) = self.get_plc_vals()
        _, f = self.get_time_frequencies(toas)
        df = np.diff(np.concatenate([[0], f]))

        return powerlaw(f.repeat(2), amp, gam) * df.repeat(2)

    def pl_rn_basis_weight_pair(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return a Fourier design matrix and power law red noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion.
        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNREDC] and T is the total observing duration of
        the dataset.

        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def pl_rn_cov_matrix(self, toas: TOAs) -> np.ndarray:
        Fmat, phi = self.pl_rn_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


def get_ecorr_epochs(toas_table: np.ndarray, dt: float = 1, nmin: int = 2) -> List[int]:
    """Find only epochs with more than 1 TOA for applying ECORR."""
    if len(toas_table) == 0:
        return []

    isort = np.argsort(toas_table)

    bucket_ref = [toas_table[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas_table[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas_table[i])
            bucket_ind.append([i])

    return [ind for ind in bucket_ind if len(ind) >= nmin]


def get_ecorr_nweights(toas_table: np.ndarray, dt: float = 1, nmin: int = 2) -> int:
    """Get the number of epochs associated with each ECORR.
    This is equal to the number of weights of that ECORR."""
    return len(get_ecorr_epochs(toas_table, dt=dt, nmin=nmin))


def create_ecorr_quantization_matrix(
    toas_table: np.ndarray, dt: float = 1, nmin: int = 2
) -> np.ndarray:
    """Create quantization matrix mapping TOAs to observing epochs.
    Only epochs with more than 1 TOA are included."""

    bucket_ind2 = get_ecorr_epochs(toas_table, dt=dt, nmin=nmin)

    U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
    for i, l in enumerate(bucket_ind2):
        U[l, i] = 1

    return U


def get_rednoise_freqs(
    t,
    nmodes: int,
    Tspan: Optional[u.Quantity] = None,
    logmode: Optional[int] = None,
    f_min: Optional[float] = None,
    nlog: Optional[int] = None,
) -> np.ndarray:
    """
    Compute an array of red-noise frequencies, optionally mixing log- and
    linearly spaced frequencies.

    If log-spaced parameters (`logmode`, `f_min`, `nlog`) are provided and valid,
    this function will prepend `nlog` log-spaced frequencies and then
    append `nmodes` linearly spaced frequencies. Otherwise, it uses purely
    linear spacing for `nmodes` frequencies.

    :param nmodes: int
        Number of linear frequency modes (if using purely linear spacing).
        If log-spacing is used, these will be the number of linear modes
        appended after the log-spaced part.
    :param Tspan: float, optional
        Span of the data in seconds. If None, but `t` is provided, it is
        taken as `max(t) - min(t)`.
    :param t: array-like, optional
        Vector of time series (TOAs) in seconds. Only required if `Tspan` is
        None, so we can calculate `Tspan` internally.
    :param logmode: int, optional
        The linear mode index at which to switch to log spacing.
        If < 0 or None, the function reverts to purely linear spacing.
        Must be >= 0 for log modes.
    :param f_min: float, optional
        Minimum frequency for log spacing, expressed as a fraction of 1/Tspan.
        Only used if logmode >= 0.
    :param nlog: int, optional
        Number of log-spaced frequencies. Only used if logmode >= 0.
    :return:
        freqs : ndarray
            Frequencies array of length either `nmodes` (linear-only) or
            `(nlog + nmodes)` (log + linear).
    """
    if Tspan is None:
        if t is None:
            raise ValueError("Must provide either Tspan or t.")
        Tspan = np.max(t) - np.min(t)

    def _get_linear_freqs(n_lin, T):
        """
        Return an array of n_lin linearly spaced frequencies:
            [1/T, 2/T, ..., n_lin/T].
        """
        return np.arange(1, n_lin + 1) / T

    def _get_loglin_freqs(logmode_, f_min_, n_log, n_lin, T):
        """
        Return an array of n_log log-spaced frequencies from f_min_ up to
        (1+logmode_)/T, then append n_lin linearly spaced frequencies
        from (1+logmode_)/T onward.
        """
        if logmode_ < 0:
            raise ValueError("Cannot do log-spacing when logmode < 0.")

        # Linear portion
        df_lin = 1.0 / T
        f_min_lin = (1.0 + logmode_) / T
        f_lin = np.linspace(f_min_lin, f_min_lin + (n_lin - 1) * df_lin, n_lin)

        # Log portion
        f_log = np.logspace(
            np.log10(f_min_), np.log10((1 + logmode_) / T), n_log, endpoint=False
        )

        # Combine log + linear
        return np.concatenate((f_log, f_lin))

    have_logmode = logmode is not None and logmode >= 0
    have_nlog = nlog is not None and nlog > 0
    have_fmin = f_min is not None and f_min > 0

    use_log = all([have_logmode, have_nlog, have_fmin])

    if not use_log and np.any([have_logmode, have_nlog, have_fmin]):
        log.warning(
            "Log-spaced parameters are ignored because "
            "logmode, nlog, and f_min ALL neeed to be set"
            "Use: logmode > 0 and nlog > 0 and f_min > 0."
        )

    if not use_log:
        # Purely linear spacing: nmodes frequencies
        freqs = _get_linear_freqs(nmodes, Tspan)
    else:
        # Log + linear: nlog log-freqs + nmodes linear-freqs
        freqs = _get_loglin_freqs(logmode, f_min, nlog, nmodes, Tspan)

    return freqs


def create_fourier_design_matrix(t, f) -> np.ndarray:
    """
    Construct a Fourier design matrix from a given set of frequencies.
    The matrix alternates sine and cosine columns.

    :param t: array-like
        Vector of time series (TOAs) in seconds.
    :param f: array-like
        Array of frequencies (e.g., from get_rednoise_freqs).
    :return:
        F : ndarray
            Fourier design matrix of shape (len(t), 2 * len(f)).
        freqs : ndarray
            The same frequencies array `f` is returned for convenience.
    """
    t = np.asarray(t)
    f = np.asarray(f)

    N = len(t)
    nfreqs = len(f)

    # Initialize design matrix
    F = np.zeros((N, 2 * nfreqs))

    # Fill sine (even columns) and cosine (odd columns)
    F[:, 0::2] = np.sin(2.0 * np.pi * t[:, None] * f)
    F[:, 1::2] = np.cos(2.0 * np.pi * t[:, None] * f)

    return F


def powerlaw(
    f, A: float = 1e-16, gamma: float = 5.0, f_low_cut: Optional[float] = None
):
    """Power-law PSD.

    :param f: Sampling frequencies
    :param A: Amplitude of red noise [GW units]
    :param gamma: Spectral index of red noise process
    :param f_low_cut: Minimum frequency to include [Hz]
    :return: Power spectral density
    """
    f_low_cut = f_low_cut if f_low_cut is not None else np.min(f)
    above_fl = np.array(f >= f_low_cut, dtype=float)

    fyr = (1 / u.year).to_value(u.Hz)
    return A**2 / 12.0 / np.pi**2 * fyr ** (gamma - 3) * f ** (-gamma) * above_fl
