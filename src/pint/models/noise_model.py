"""Pulsar timing noise models."""

import copy
from typing import Callable, List, Optional, Tuple
import warnings

import astropy.units as u
import numpy as np
from scipy import interpolate
from loguru import logger as log

from pint import DMconst, dmu
from pint.models.parameter import (
    Parameter,
    floatParameter,
    intParameter,
    maskParameter,
    prefixParameter,
    strParameter,
)
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

    @property
    def introduces_correlated_errors(self) -> bool:
        return isinstance(self, CorrelatedNoiseComponent)


class WhiteNoiseComponent(NoiseComponent):
    """Abstract base class for all white noise components."""

    pass


class CorrelatedNoiseComponent(NoiseComponent):
    """Abstract base class for all correlated noise components."""

    is_time_correlated = False

    def get_noise_basis(self, toas):
        raise NotImplementedError

    def get_noise_weights(self, toas):
        raise NotImplementedError

    def get_dm_noise_basis(self, toas):
        """The DM part of the basis matrix for wideband datasets. This is non-zero
        only for DM noise. The output is a numpy array but it has units of dmu/s
        by convention since the noise amplitudes are defined to have dimensions of
        time."""
        toa_noise_basis = self.get_noise_basis(toas)
        if self.introduces_dm_errors:
            freqs = self._parent.barycentric_radio_freq(toas)
            return (toa_noise_basis * (freqs**2 / DMconst)[:, None]).to_value(dmu / u.s)
        else:
            return np.zeros_like(toa_noise_basis)

    def get_wideband_noise_basis(self, toas):
        """The wideband noise basis including both TOA and DM parts. The TOA part
        of the matrix is dimensionless but the DM part of the basis has units of
        dmu/s."""
        M_toa = self.get_noise_basis(toas)
        M_dm = self.get_dm_noise_basis(toas)
        return np.vstack((M_toa, M_dm))


def project_basis_covariance(U: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Project basis-space covariance to data-space covariance."""
    if np.ndim(Phi) == 1:
        return np.dot(U * Phi[None, :], U.T)
    return np.dot(U, np.dot(Phi, U.T))


def get_tdb_seconds(tbl) -> np.ndarray:
    """Return TOA TDB times in seconds as float64."""
    return np.asarray((tbl["tdbld"].quantity * u.day).to(u.s).value, dtype=np.float64)


def _add_tdsw_node_component(model, node, index=None):
    """Add one TDSWNODE_ prefix parameter to a time-domain SW noise component."""
    dct = model.get_prefix_mapping_component("TDSWNODE_")
    if index is None:
        available = [
            idx
            for idx, par_name in dct.items()
            if getattr(model, par_name).value is None
        ]
        if len(available) > 0:
            index = int(np.min(available))
        else:
            index = np.max(list(dct.keys())) + 1
    i = f"{int(index):04d}"

    if isinstance(node, u.quantity.Quantity):
        node = node.to_value(u.day)

    if int(index) in dct:
        par = getattr(model, dct[int(index)])
        if par.value is not None:
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )
        par.value = node
    else:
        model.add_param(
            prefixParameter(
                name=f"TDSWNODE_{i}",
                units="day",
                value=node,
                description="Interpolation node for time-domain SW noise basis (MJD).",
                parameter_type="float",
                convert_tcb2tdb=False,
            )
        )
    model.setup()

    node_map = model.get_prefix_mapping_component("TDSWNODE_")
    nset = sum(
        getattr(model, node_name).value is not None for _, node_name in node_map.items()
    )
    if nset >= 2:
        model.validate()
    return index


class ScaleToaError(WhiteNoiseComponent):
    """Correct the reported TOA uncertainties. The corrections account for
    imperfections in the TOA measurement and pulse jitter.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.ScaleToaError

    Note
    ----
    Ref: NANOGrav 11 yrs data

    """

    register = True
    category = "scale_toa_error"

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


class ScaleDmError(WhiteNoiseComponent):
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


class EcorrNoise(CorrelatedNoiseComponent):
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
        t = get_tdb_seconds(tbl)
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
            tbl = toas.table
            ts = get_tdb_seconds(tbl)
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


class PLDMNoise(CorrelatedNoiseComponent):
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
        t = get_tdb_seconds(tbl)
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
        return project_basis_covariance(Fmat, phi)


class PLSWNoise(CorrelatedNoiseComponent):
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

    introduces_dm_errors = True
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
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
            )
        )
        self.add_param(
            floatParameter(
                name="TNSWGAM",
                units="",
                aliases=[],
                description="Spectral index of power-law "
                "SW DM noise in tempo2 format",
                convert_tcb2tdb=True,
                tcb2tdb_scale_factor=1,
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
        t = get_tdb_seconds(tbl)
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
        return project_basis_covariance(Fmat, phi)


class PLChromNoise(CorrelatedNoiseComponent):
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
        t = get_tdb_seconds(tbl)
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
        return project_basis_covariance(Fmat, phi)


class PLRedNoise(CorrelatedNoiseComponent):
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
        t = get_tdb_seconds(tbl)
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
        return project_basis_covariance(Fmat, phi)


class TimeDomainSWNoise(NoiseComponent):
    """Time-domain solar wind noise model with a selectable GP kernel.

    Solar wind electron number density fluctuations produce dispersive delays
    that vary in time.  This component models those fluctuations as a Gaussian
    Process (GP) in the time domain, using a linear interpolation basis
    (controlled by ``TDSWDT`` or explicit ``TDSWNODE_*`` parameters) and a
    kernel covariance function selected via ``TDSWKERNEL``.

    The basis matrix projects the GP onto each TOA using a solar-wind geometry
    factor so that the effective delay at frequency :math:`f` is

    .. math::

        \\delta t(f) = \\frac{\\mathrm{DM}_{\\odot}(t)}{f^2} \\cdot K_{\\mathrm{DM}}

    where :math:`\\mathrm{DM}_{\\odot}(t)` is the line-of-sight integral of the
    solar wind electron density evaluated at each epoch by the parent
    ``SolarWindDispersion`` component.

    Kernel definitions
    ------------------
    Let :math:`\\tau = |t_i - t_j|` (in seconds at the interpolation nodes),
    :math:`\\sigma = 10^{\\mathtt{TDSWLOGSIG}}`,
    :math:`\\ell = 10^{\\mathtt{TDSWLOGELL}}` days,
    :math:`p = 10^{\\mathtt{TDSWLOGP}}` years.

    .. note::
        ``TDSWLOGELL`` is in **log10(days)** and ``TDSWLOGP`` is in **log10(years)**
        in both PINT and discovery, matching the enterprise convention for the
        quasi-periodic kernel.  The kernel functions internally convert to seconds.

    **ridge** (white-noise / diagonal)

    .. math::

        K(t_i, t_j) = \\sigma^2 \\,\\delta_{ij}

    Only ``TDSWLOGSIG`` is required.

    **sqexp** (squared-exponential / RBF)

    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(-\\frac{\\tau^2}{2\\ell^2}\\right)

    Requires ``TDSWLOGSIG``, ``TDSWLOGELL``.

    **matern** (Matérn with half-integer smoothness :math:`\\nu`)

    For :math:`\\nu = 1/2`:

    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(-\\frac{\\tau}{\\ell}\\right)

    For :math:`\\nu = 3/2`:

    .. math::

        K(\\tau) = \\sigma^2 \\left(1 + \\frac{\\sqrt{3}\\,\\tau}{\\ell}\\right)
                   \\exp\\!\\left(-\\frac{\\sqrt{3}\\,\\tau}{\\ell}\\right)

    For :math:`\\nu = 5/2`:

    .. math::

        K(\\tau) = \\sigma^2 \\left(1 + \\frac{\\sqrt{5}\\,\\tau}{\\ell}
                   + \\frac{5\\tau^2}{3\\ell^2}\\right)
                   \\exp\\!\\left(-\\frac{\\sqrt{5}\\,\\tau}{\\ell}\\right)

    Requires ``TDSWLOGSIG``, ``TDSWLOGELL``.  ``TDSWNU`` (default 1.5) selects
    :math:`\\nu \\in \\{0.5, 1.5, 2.5\\}`.

    **quasi_periodic** (squared-exponential × periodic envelope)

    Let :math:`\\Gamma_p = 10^{\\mathtt{TDSWLOGGAMP}}` and
    :math:`P = 10^{\\mathtt{TDSWLOGP}}` years:

    .. math::

        K(\\tau) = \\sigma^2
                   \\exp\\!\\left(-\\frac{\\tau^2}{2\\ell^2}\\right)
                   \\exp\\!\\left(-\\frac{2\\sin^2\\!\\frac{\\pi\\tau}{P}}{\\Gamma_p^2}\\right)

    Requires ``TDSWLOGSIG``, ``TDSWLOGELL``, ``TDSWLOGGAMP``, ``TDSWLOGP``.

    The kernel is evaluated at the interpolation nodes and the resulting weight
    matrix is projected back onto the TOA residuals via the linear interpolation
    basis, yielding the full :math:`N_{\\mathrm{TOA}} \\times N_{\\mathrm{TOA}}`
    covariance contribution.

    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.TimeDomainSWNoise

    Notes
    -----
    * ``TimeDomainSWNoise`` requires a ``SolarWindDispersion`` component in the
      timing model so that the solar wind geometry factor is available.
    * The interpolation basis is built from either a uniform grid with spacing
      ``TDSWDT`` (days) or an explicit set of ``TDSWNODE_NNNN`` parameters
      (MJD).  The two modes are mutually exclusive.
    * ``register = False`` — attach this component explicitly via
      :meth:`~pint.models.timing_model.TimingModel.add_component`.

    Examples
    --------
    Add a time-domain solar wind GP with a Matérn-3/2 kernel to an existing
    timing model, using a 14-day interpolation grid:

    >>> from pint.models.timing_model import Component
    >>> from pint.models.noise_model import TimeDomainSWNoise
    >>> all_components = Component.component_types
    >>> if "SolarWindDispersion" not in model.components:
    ...     sw_det = all_components["SolarWindDispersion"]()
    ...     model.add_component(sw_det, validate=False)
    ...     model["NE_SW"].quantity = 4.0
    ...     model["SWM"] = 1
    ...     model["SWP"] = 2.0
    >>> sw_comp = TimeDomainSWNoise()
    >>> model.add_component(sw_comp, validate=False)
    >>> model["TDSWKERNEL"].value = "matern"
    >>> model["TDSWLOGSIG"].value = -8.0
    >>> model["TDSWLOGELL"].value = 1.5
    >>> model["TDSWNU"].value = 1.5
    >>> model["TDSWDT"].value = 14.0
    >>> model["TDSWINTERP_KIND"].value = "linear"
    >>> model.validate()

    Notes
    -----
    The above example will appear in the par file as::

        TDSWKERNEL        matern
        TDSWDT            14.0
        TDSWLOGSIG        -8.0
        TDSWLOGELL        1.5
        TDSWNU            1.5
        TDSWINTERP_KIND   linear

    To use explicit interpolation nodes instead of a uniform grid, set
    ``TDSWNODE_NNNN`` parameters (MJD) via :meth:`add_tdsw_node_component`.
    Kernel-specific parameters must be configured **before** adding nodes because
    :meth:`add_tdsw_node_component` calls ``validate()`` internally once two or
    more nodes are present::

    >>> sw_comp = TimeDomainSWNoise()
    >>> model.add_component(sw_comp, validate=False)
    >>> model["TDSWKERNEL"].value = "ridge"
    >>> model["TDSWLOGSIG"].value = -7.0
    >>> for i, mjd in enumerate(node_mjd_array, start=1):
    ...     sw_comp.add_tdsw_node_component(mjd, index=i)
    >>> model.validate()

    References
    ----------
    Stochastic solar wind modeling is introduced in PTA literature by Hazboun et al. 2022.
    Time-domain Gaussian processes are introduced to the PTA literature in Hazboun et al. 2026.
    - Hazboun et al. 2022 [1]_
    - Hazboun et al. 2026 [2]_

    .. [1] https://iopscience.iop.org/article/10.3847/1538-4357/ac5829
    .. [2] https://iopscience.iop.org/article/10.3847/1538-4357/ae4ee0
    """

    register = False
    category = "SW_noise"

    introduces_correlated_errors = True
    introduces_dm_errors = True
    is_time_correlated = True

    #: Mapping from kernel name to required and optional parameter names.
    KERNEL_PARAMS: dict = {
        "ridge": {"required": ["TDSWLOGSIG"], "optional": []},
        "sqexp": {"required": ["TDSWLOGSIG", "TDSWLOGELL"], "optional": []},
        "matern": {
            "required": ["TDSWLOGSIG", "TDSWLOGELL"],
            "optional": ["TDSWNU"],
        },
        "quasi_periodic": {
            "required": ["TDSWLOGSIG", "TDSWLOGELL", "TDSWLOGGAMP", "TDSWLOGP"],
            "optional": [],
        },
    }

    ALLOWED_KERNELS: frozenset = frozenset(KERNEL_PARAMS)

    def __init__(self):
        super().__init__()

        self.add_param(
            strParameter(
                name="TDSWKERNEL",
                value="ridge",
                description=(
                    "Kernel for time-domain SW noise GP. "
                    "Allowed values: 'ridge', 'sqexp', 'matern', 'quasi_periodic'."
                ),
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWDT",
                units="day",
                aliases=[],
                value=30.0,
                description="Linear interpolation time step for time-domain SW noise.",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWLOGSIG",
                units="s",
                aliases=[],
                description="Log10 amplitude of time-domain SW noise kernel.",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWLOGELL",
                units="",
                aliases=[],
                description=(
                    "Log10 characteristic length scale for sqexp / matern / "
                    "quasi_periodic time-domain SW noise (days)."
                ),
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWNU",
                units="",
                aliases=[],
                value=1.5,
                description="Matern smoothness parameter (supported: 0.5, 1.5, 2.5).",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWLOGGAMP",
                units="",
                aliases=[],
                description="Log10 mixing parameter for quasi-periodic time-domain SW noise.",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TDSWLOGP",
                units="",
                aliases=[],
                description="Log10 periodicity of quasi-periodic time-domain SW noise (years).",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            strParameter(
                name="TDSWINTERP_KIND",
                value="linear",
                description="Interpolation kind passed to scipy.interpolate.interp1d.",
            )
        )

        self.add_param(
            prefixParameter(
                name="TDSWNODE_0001",
                units="day",
                value=None,
                description="Interpolation node for time-domain SW noise basis (MJD).",
                parameter_type="float",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.sw_cov_matrix]
        self.basis_funcs += [self.sw_basis_weight_pair]

    def add_tdsw_node_component(self, node, index=None):
        """Add a TDSWNODE component.

        Parameters
        ----------
        node : float or astropy.units.Quantity
            Interpolation node in MJD (days).
        index : int or None
            Integer index label for the node. If None, increments max index by 1.
        """
        return _add_tdsw_node_component(self, node=node, index=index)


    def validate(self):
        super().validate()

        kernel = self.TDSWKERNEL.value
        if kernel not in self.ALLOWED_KERNELS:
            raise ValueError(
                f"TimeDomainSWNoise TDSWKERNEL must be one of "
                f"{sorted(self.ALLOWED_KERNELS)}, got '{kernel}'."
            )

        # Check that all required parameters for this kernel are set.
        required = self.KERNEL_PARAMS[kernel]["required"]
        missing = [p for p in required if getattr(self, p).value is None]
        if missing:
            raise ValueError(
                f"TimeDomainSWNoise with kernel='{kernel}' requires "
                f"parameter(s) {missing} to be set."
            )

        if kernel == "matern" and self.TDSWNU.value not in (0.5, 1.5, 2.5):
            raise ValueError("TimeDomainSWNoise TDSWNU must be one of {0.5, 1.5, 2.5}.")

        allowed_kinds = {
            "linear",
            "nearest",
            "nearest-up",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "previous",
            "next",
        }
        if self.TDSWINTERP_KIND.value not in allowed_kinds:
            raise ValueError(
                f"TimeDomainSWNoise TDSWINTERP_KIND must be one of {sorted(allowed_kinds)}."
            )

        node_map = self.get_prefix_mapping_component("TDSWNODE_")
        node_values = []
        for _, node_name in node_map.items():
            node_val = getattr(self, node_name).value
            if node_val is not None:
                node_values.append(float(node_val))

        dt_val = self.TDSWDT.value
        has_nodes = len(node_values) > 0
        has_nondefault_dt = dt_val is not None and dt_val != 30.0

        if has_nodes and has_nondefault_dt:
            raise ValueError(
                "TimeDomainSWNoise requires exactly one interpolation mode: "
                "set TDSWDT or set TDSWNODE_ parameters, but not both."
            )

        if 0 < len(node_values) < 2:
            raise ValueError(
                "TimeDomainSWNoise requires at least 2 TDSWNODE_ values when using "
                "node-based interpolation. Set >=2 TDSWNODE_ parameters."
            )

        if len(node_values) >= 2:
            nodes = np.asarray(node_values, dtype=float)
            if not np.all(np.isfinite(nodes)):
                raise ValueError("TimeDomainSWNoise TDSWNODE_ values must be finite.")
            if len(np.unique(nodes)) != len(nodes):
                raise ValueError("TimeDomainSWNoise TDSWNODE_ values must be unique.")
        else:
            if dt_val is not None and dt_val <= 0:
                raise ValueError(
                    "TimeDomainSWNoise TDSWDT must be set to a positive value."
                )

    def _has_nodes(self) -> bool:
        """Return True if any TDSWNODE_ parameter is set."""
        node_map = self.get_prefix_mapping_component("TDSWNODE_")
        return any(
            getattr(self, node_name).value is not None
            for _, node_name in node_map.items()
        )

    def _get_nodes(self, toas: TOAs) -> np.ndarray:
        """Return sorted interpolation nodes (MJD) from TDSWNODE_ parameters."""
        node_map = self.get_prefix_mapping_component("TDSWNODE_")
        nodes = []
        for _, node_name in node_map.items():
            node_par = getattr(self, node_name)
            if node_par.value is not None:
                nodes.append(float(node_par.value))

        if len(nodes) >= 2:
            return np.array(sorted(nodes), dtype=float)

        raise ValueError(
            "TimeDomainSWNoise node interpolation requires at least 2 TDSWNODE_ values."
        )

    def _get_basis_and_nodes(self, toas: TOAs):
        """Return ``(Umat, nodes)`` from the linear interpolation basis."""
        tbl = toas.table
        t = get_tdb_seconds(tbl)
        interp_kind = self.TDSWINTERP_KIND.value
        if self._has_nodes():
            nodes_in = self._get_nodes(toas)
            Umat, nodes = make_interpolation_basis(t, nodes=nodes_in, kind=interp_kind)
        else:
            dt = 30.0 if self.TDSWDT.value is None else self.TDSWDT.value
            Umat, nodes = make_interpolation_basis(t, dt=dt, kind=interp_kind)
        return Umat, nodes


    def get_noise_basis(self, toas: TOAs) -> np.ndarray:
        """Return chromatic linear interpolation matrix for time-domain SW noise."""
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        Umat, _ = self._get_basis_and_nodes(toas)
        # Solar wind geometry from pint.models.solar_wind_dispersion.SolarWindDispersion.
        # This is the SW DM contribution if n_earth = 1 cm^-3; the GP scales it.
        solar_wind_geometry = self._parent.solar_wind_geometry(toas)
        dt_DM = (solar_wind_geometry * DMconst / (freqs**2)).value
        return Umat * dt_DM[:, None]

    def get_noise_weights(self, toas: TOAs) -> np.ndarray:
        """Return GP prior weights for the selected kernel.

        The kernel is controlled by ``TDSWKERNEL``:

        * **ridge**
          :math:`K(t_i, t_j) = \\sigma^2 \\delta(t_i - t_j)`
        * **sqexp**
          :math:`K(t_i, t_j) = \\sigma^2 \\exp\\!\\left(-\\frac{(t_i-t_j)^2}{2\\ell^2}\\right)`
        * **matern**
          Matern kernel with smoothness :math:`\\nu \\in \\{0.5, 1.5, 2.5\\}`
        * **quasi_periodic**
          :math:`K_{SE}(t_i,t_j) \\cdot \\exp\\!\\left(-\\frac{2\\sin^2\\!\\frac{\\pi(t_i-t_j)}{p}}{\\Gamma_p^2}\\right)`
        """
        _, nodes = self._get_basis_and_nodes(toas)
        kernel = self.TDSWKERNEL.value
        log10_sigma = self.TDSWLOGSIG.value

        if kernel == "ridge":
            return ridge_kernel(nodes, log10_sigma)
        elif kernel == "sqexp":
            return se_kernel(nodes, log10_sigma, self.TDSWLOGELL.value)
        elif kernel == "matern":
            return matern_kernel(
                nodes, log10_sigma, self.TDSWLOGELL.value, self.TDSWNU.value
            )
        elif kernel == "quasi_periodic":
            return periodic_kernel(
                nodes,
                log10_sigma,
                self.TDSWLOGELL.value,
                self.TDSWLOGGAMP.value,
                self.TDSWLOGP.value,
            )
        else:  # unreachable after validate()
            raise ValueError(f"TimeDomainSWNoise: unknown TDSWKERNEL '{kernel}'")

    def sw_basis_weight_pair(self, toas: TOAs) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(basis, weights)`` for the time-domain SW noise GP."""
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def sw_cov_matrix(self, toas: TOAs) -> np.ndarray:
        """Return the covariance matrix for the time-domain SW noise GP."""
        Umat, phi = self.sw_basis_weight_pair(toas)
        return project_basis_covariance(Umat, phi)


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

    have_logmode = logmode is not None and logmode > 0
    have_nlog = nlog is not None and nlog > 0
    have_fmin = f_min is not None and f_min > 0

    use_log = all([have_logmode, have_nlog, have_fmin])

    if not use_log and (have_logmode or have_nlog):
        log.warning(
            "Log-linear frequency spacing appears to be "
            "incorrectly specified. Got logmode={logmode}, "
            "nlog={nlog}, f_min={f_min}. Proceeding with "
            "linearly-spaced frequencies.",
            logmode=logmode,
            nlog=nlog,
            f_min=f_min,
        )

    if not use_log:
        # Purely linear spacing: nmodes frequencies
        freqs = _get_linear_freqs(nmodes, Tspan)
    else:
        # Log + linear: nlog log-freqs + nmodes linear-freqs
        freqs = _get_loglin_freqs(logmode, f_min, nlog, nmodes, Tspan)

    return np.asarray(freqs, dtype=np.float64)


def make_interpolation_basis(
    toas,
    nodes=None,
    dt=None,
    kind="linear",
):
    """
    Construct an interpolation basis for the given TOAs and interpolation parameters using
    scipy.interpolate.interp1d.

    :param toas: array-like
        Vector of time series (TOAs) in seconds.
    :param nodes: array-like, optional
        Vector of interpolation nodes in MJD. If None, nodes are generated from dt.
    :param dt: float, optional
        Time step in days for generating interpolation nodes if `nodes` is None.
    :param kind: str, optional
        Interpolation kind passed to scipy.interpolate.interp1d. Default is "linear".
        See scipy.interpolate.interp1d documentation for allowed values.

    :return:
        M : the achromatic interpolation design matrix of shape (len(toas), n_nodes).
        nodes : the interpolation nodes in MJD corresponding to the columns of M.
    """
    if nodes is None:
        if dt is None:
            raise ValueError(
                "Must provide either nodes or dt for linear interpolation basis."
            )
        t_min, t_max = np.min(toas) / 86400, np.max(toas) / 86400
        nodes = np.arange(t_min, t_max + dt, dt)
    nodes = nodes * 86400  # MJD to seconds
    basis = np.identity(len(nodes))
    interp = interpolate.interp1d(
        nodes,
        basis,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    M = interp(toas)
    # only return non-zero columns for rank reduction
    idx = M.sum(axis=0) != 0
    if not np.any(idx):
        raise RuntimeError(
            "Interpolation basis has no support in the TOA range. Perhaps check units."
        )

    return M[:, idx], nodes[idx]


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
    f = np.asarray(f, dtype=np.float64)
    f_low_cut = f_low_cut if f_low_cut is not None else np.min(f)
    above_fl = np.array(f >= f_low_cut, dtype=float)

    fyr = (1 / u.year).to_value(u.Hz)
    return A**2 / 12.0 / np.pi**2 * fyr ** (gamma - 3) * f ** (-gamma) * above_fl


def periodic_kernel(
    nodes: np.ndarray,
    log10_sigma: float = -7,
    log10_ell: float = 2,
    log10_gam_p: float = 0,
    log10_p: float = 0,
) -> np.ndarray:
    """Quasi-periodic (SE × periodic) covariance matrix.

    Matches the ``periodic_kernel`` convention in enterprise_extensions.

    Parameters
    ----------
    nodes : np.ndarray
        1-D array of evaluation points in seconds (e.g. average TOA at each epoch).
    log10_sigma : float
        Log10 of the amplitude in the same units as the residuals.
    log10_ell : float
        Log10 of the squared-exponential length scale in **days**.
    log10_gam_p : float
        Log10 of the periodic damping amplitude (larger -> stronger periodic decay).
    log10_p : float
        Log10 of the periodicity in **years**.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(len(nodes), len(nodes))``.

    Notes
    -----
    The kernel is

    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(
            -\\frac{\\tau^2}{2\\ell^2}
            - \\gamma_p \\sin^2\\!\\left(\\frac{\\pi\\tau}{p}\\right)
        \\right) + d\\,\\delta_{ij}

    where :math:`d = (\\sigma / 50000)^2` is a small diagonal regulariser.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])

    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400   # days -> seconds
    p = 10**log10_p * 365.25 * 86400  # years -> seconds
    gam_p = 10**log10_gam_p
    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    K = sigma**2 * np.exp(-(r**2) / 2 / l**2 - gam_p * np.sin(np.pi * r / p) ** 2) + d
    return K


def se_kernel(
    nodes: np.ndarray,
    log10_sigma: float = -7,
    log10_ell: float = 2,
) -> np.ndarray:
    """Squared-exponential (RBF) covariance matrix.

    Parameters
    ----------
    nodes : np.ndarray
        1-D array of evaluation points in seconds.
    log10_sigma : float
        Log10 of the amplitude.
    log10_ell : float
        Log10 of the length scale in **days**.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(len(nodes), len(nodes))``.

    Notes
    -----
    The kernel is

    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(-\\frac{\\tau^2}{2\\ell^2}\\right)
            + d\\,\\delta_{ij}

    where :math:`d = (\\sigma / 50000)^2` is a small diagonal regulariser.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])

    l = 10**log10_ell * 86400   # days -> seconds
    sigma = 10**log10_sigma
    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    K = sigma**2 * np.exp(-(r**2) / 2 / l**2) + d
    return K


def matern_kernel(
    nodes: np.ndarray,
    log10_sigma: float = -7,
    log10_ell: float = 2,
    nu: float = 1.5,
) -> np.ndarray:
    """Matérn covariance matrix.

    Parameters
    ----------
    nodes : np.ndarray
        1-D array of evaluation points in seconds.
    log10_sigma : float
        Log10 of the amplitude.
    log10_ell : float
        Log10 of the length scale in **days**.
    nu : float
        Smoothness parameter; must be one of ``{0.5, 1.5, 2.5}``.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(len(nodes), len(nodes))``.

    Raises
    ------
    ValueError
        If *nu* is not in ``{0.5, 1.5, 2.5}``.

    Notes
    -----
    The Matérn-1/2 (``nu=0.5``), Matérn-3/2 (``nu=1.5``), and Matérn-5/2
    (``nu=2.5``) closed-form kernels are supported.  A small diagonal
    regulariser :math:`d = (\\sigma / 50000)^2` is added for numerical
    stability.
    """
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("matern_kernel currently supports nu in {0.5, 1.5, 2.5}.")

    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])

    l = 10**log10_ell * 86400   # days -> seconds
    sigma = 10**log10_sigma

    rr = r / l
    if nu == 0.5:
        k = np.exp(-rr)
    elif nu == 1.5:
        c = np.sqrt(3.0)
        k = (1.0 + c * rr) * np.exp(-c * rr)
    else:  # nu == 2.5
        c = np.sqrt(5.0)
        k = (1.0 + c * rr + (5.0 / 3.0) * rr**2) * np.exp(-c * rr)

    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    return sigma**2 * k + d


def ridge_kernel(
    nodes: np.ndarray,
    log10_sigma: float = -7,
) -> np.ndarray:
    """Ridge (diagonal) covariance matrix.

    Parameters
    ----------
    nodes : np.ndarray
        1-D array of evaluation points in seconds.  Only ``len(nodes)`` is
        used; the values themselves are ignored.
    log10_sigma : float
        Log10 of the amplitude; the diagonal entries are
        :math:`\\sigma^2 = 10^{2\\,\\texttt{log10\_sigma}}`.

    Returns
    -------
    np.ndarray
        Diagonal covariance matrix of shape ``(len(nodes), len(nodes))``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])

    # Convert to seconds
    sigma = 10 ** (log10_sigma * 2)
    return np.eye(r.shape[0]) * sigma
