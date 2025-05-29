"""Pulsar timing noise models."""

import copy
import warnings

import astropy.units as u
import astropy.constants as const
import numpy as np

from loguru import logger as log

from pint.models.parameter import floatParameter, maskParameter
from pint.models.timing_model import Component

AU_light_sec = const.au.to("lightyear").value * u.year.to("s")  # 1 AU in light seconds
AU_pc = const.au.to("parsec").value  # 1 AU in parsecs (for DM normalization)


class NoiseComponent(Component):
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

    def register_toasigma_deriv_funcs(self, func, param):
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

    def scale_toa_sigma(self, toas, warn=True):
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

    def sigma_scaled_cov_matrix(self, toas):
        scaled_sigma = self.scale_toa_sigma(toas).to(u.s).value ** 2
        return np.diag(scaled_sigma)

    def d_toasigma_d_EFAC(self, toas, param):
        par = getattr(self, param)
        mask = par.select_toa_mask(toas)
        result = np.zeros(len(toas)) << u.s
        result[mask] = self.scale_toa_sigma(toas[mask], warn=False).to(
            u.s
        ) / par.quantity.to(u.dimensionless_unscaled)
        return result

    def d_toasigma_d_EQUAD(self, toas, param):
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

    def scale_dm_sigma(self, toas):
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

    def dm_sigma_scaled_cov_matrix(self, toas):
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

    def get_ecorrs(self):
        return [getattr(self, ecorr) for ecorr, ecorr_key in list(self.ECORRs.items())]

    def get_noise_basis(self, toas):
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

    def get_noise_weights(self, toas, nweights=None):
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

    def ecorr_basis_weight_pair(self, toas):
        """Return a quantization matrix and ECORR weights.

        A quantization matrix maps TOAs to observing epochs.
        The weights used are the square of the ECORR values.
        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def ecorr_cov_matrix(self, toas):
        """Full ECORR covariance matrix."""
        U, Jvec = self.ecorr_basis_weight_pair(toas)
        return np.dot(U * Jvec[None, :], U.T)


class PLDMNoise(NoiseComponent):
    """Model of DM variations as radio frequency-dependent noise with a
    power-law spectrum.

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

    Note
    ----
    Ref: Lentati et al. 2014, MNRAS 437(3), 3004-3023

    """

    register = True
    category = "pl_DM_noise"

    introduces_correlated_errors = True
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
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNDMGAM",
                units="",
                aliases=[],
                description="Spectral index of powerlaw " "DM noise in tempo2 format",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNDMC",
                units="",
                aliases=[],
                description="Number of DM noise frequencies.",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.pl_dm_cov_matrix]
        self.basis_funcs += [self.pl_dm_basis_weight_pair]

    def get_pl_vals(self):
        nf = int(self.TNDMC.value) if self.TNDMC.value is not None else 30
        amp, gam = 10**self.TNDMAMP.value, self.TNDMGAM.value
        return (amp, gam, nf)

    def get_noise_basis(self, toas):
        """Return a Fourier design matrix for DM noise.

        See the documentation for pl_dm_basis_weight_pair function for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        fref = 1400 * u.MHz
        D = (fref.value / freqs.value) ** 2
        nf = self.get_pl_vals()[2]
        Fmat = create_fourier_design_matrix(t, nf)

        return Fmat * D[:, None]

    def get_noise_weights(self, toas):
        """Return power law DM noise weights.

        See the documentation for pl_dm_basis_weight_pair for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        amp, gam, nf = self.get_pl_vals()
        Ffreqs = get_rednoise_freqs(t, nf)
        return powerlaw(Ffreqs, amp, gam) * Ffreqs[0]

    def pl_dm_basis_weight_pair(self, toas):
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

    def pl_dm_cov_matrix(self, toas):
        Fmat, phi = self.pl_dm_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


class PLSWNoise(NoiseComponent):
    """Model of solar wind DM variations as radio frequency-dependent noise with a
    power-law spectrum.

    Commonly used as perturbations on top of a deterministic solar wind model.


    Parameters supported:

    .. paramtable::
        :class: pint.models.noise_model.PLSWNoise

    Note
    ----
    Ref: Lentati et al. 2014, MNRAS 437(3), 3004-3023.
    See also: Hazboun et al. 2019 and Susurla et al. 2024.

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

        self.covariance_matrix_funcs += [self.pl_sw_cov_matrix]
        self.basis_funcs += [self.pl_sw_basis_weight_pair]

    def get_pl_vals(self):
        nf = int(self.TNSWC.value) if self.TNSWC.value is not None else 30
        amp, gam = 10**self.TNSWAMP.value, self.TNSWGAM.value
        return (amp, gam, nf)

    def get_noise_basis(self, toas): # planetssb, sunssb, pos_t):
        """Return a Fourier design matrix for SW DM noise.

        See the documentation for pl_sw_basis_weight_pair function for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        nf = self.get_pl_vals()[2]
        # get the achromatic Fourier design matrix
        Fmat = create_fourier_design_matrix(t, nf)
        # get solar wind geometry from pint.models.solar_wind_dispersion.SolarWindDispersion
        solar_wind_geometry = self._parent.solar_wind_geometry(toas)
        # dispersion constant taken from Handbook of Pulsar Astronomy, eq. 4.5
        # the cm**-3 at the end reflect the units of the GP perturbations which are solar electron number density
        # since this is the SW DM value if n_earth = 1 cm^-3. the GP will scale it.
        dt_DM = (solar_wind_geometry * 4.148808e3*u.MHz**2/u.pc*u.cm**3*u.s / (freqs**2) * u.cm**-3).value

        return Fmat * dt_DM[:, None]

    def get_noise_weights(self, toas):
        """Return power-law SW DM noise weights.

        See the documentation for pl_sw_basis_weight_pair for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        amp, gam, nf = self.get_pl_vals()
        Ffreqs = get_rednoise_freqs(t, nf)
        return powerlaw(Ffreqs, amp, gam) * Ffreqs[0]

    def pl_sw_basis_weight_pair(self, toas):
        """Return a Fourier design matrix and power law SW DM noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion. Here we scale the design matrix by
        (fref/f)**2 and a solar wind geometric factor,
        where fref = 1400 MHz to match the convention used in enterprise.

        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNDMC] and T is the total observing duration of
        the dataset.

        """
        return (
            self.get_noise_basis(toas),
            self.get_noise_weights(toas),
        )

    def pl_sw_cov_matrix(self, toas):
        Fmat, phi = self.pl_sw_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


class PLChromNoise(NoiseComponent):
    """Model of a radio frequency-dependent noise with a power-law spectrum and
    arbitrary chromatic index.

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

    Note
    ----
    Ref: Lentati et al. 2014, MNRAS 437(3), 3004-3023
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
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMGAM",
                units="",
                aliases=[],
                description="Spectral index of powerlaw chromatic noise in tempo2 format",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMC",
                units="",
                aliases=[],
                description="Number of chromatic noise frequencies.",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.pl_chrom_cov_matrix]
        self.basis_funcs += [self.pl_chrom_basis_weight_pair]

    def get_pl_vals(self):
        nf = int(self.TNCHROMC.value) if self.TNCHROMC.value is not None else 30
        amp, gam = 10**self.TNCHROMAMP.value, self.TNCHROMGAM.value
        return (amp, gam, nf)

    def get_noise_basis(self, toas):
        """Return a Fourier design matrix for chromatic noise.

        See the documentation for pl_chrom_basis_weight_pair function for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        freqs = self._parent.barycentric_radio_freq(toas).to(u.MHz)
        fref = 1400 * u.MHz
        alpha = self._parent.TNCHROMIDX.value
        D = (fref.value / freqs.value) ** alpha
        nf = self.get_pl_vals()[2]
        Fmat = create_fourier_design_matrix(t, nf)
        return Fmat * D[:, None]

    def get_noise_weights(self, toas):
        """Return power law chromatic noise weights.

        See the documentation for pl_chrom_basis_weight_pair for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        amp, gam, nf = self.get_pl_vals()
        Ffreqs = get_rednoise_freqs(t, nf)
        return powerlaw(Ffreqs, amp, gam) * Ffreqs[0]

    def pl_chrom_basis_weight_pair(self, toas):
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

    def pl_chrom_cov_matrix(self, toas):
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

    Note
    ----
    Ref: Lentati et al. 2014, MNRAS 437(3), 3004-3023

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
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="RNIDX",
                units="",
                aliases=[],
                description="Spectral index of powerlaw red noise.",
                convert_tcb2tdb=False,
            )
        )

        self.add_param(
            floatParameter(
                name="TNREDAMP",
                units="",
                aliases=[],
                description="Amplitude of powerlaw red noise in tempo2 format",
                convert_tcb2tdb=False,
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
            floatParameter(
                name="TNREDC",
                units="",
                aliases=[],
                description="Number of red noise frequencies.",
                convert_tcb2tdb=False,
            )
        )

        self.covariance_matrix_funcs += [self.pl_rn_cov_matrix]
        self.basis_funcs += [self.pl_rn_basis_weight_pair]

    def get_pl_vals(self):
        nf = int(self.TNREDC.value) if self.TNREDC.value is not None else 30
        if self.TNREDAMP.value is not None and self.TNREDGAM.value is not None:
            amp, gam = 10**self.TNREDAMP.value, self.TNREDGAM.value
        elif self.RNAMP.value is not None and self.RNIDX is not None:
            fac = (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0))
            amp, gam = self.RNAMP.value / fac, -1 * self.RNIDX.value
        return (amp, gam, nf)

    def get_noise_basis(self, toas):
        """Return a Fourier design matrix for red noise.

        See the documentation for pl_rn_basis_weight_pair function for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        nf = self.get_pl_vals()[2]
        return create_fourier_design_matrix(t, nf)

    def get_noise_weights(self, toas):
        """Return power law red noise weights.

        See the documentation for pl_rn_basis_weight_pair for details."""

        tbl = toas.table
        t = (tbl["tdbld"].quantity * u.day).to(u.s).value
        amp, gam, nf = self.get_pl_vals()
        Ffreqs = get_rednoise_freqs(t, nf)
        return powerlaw(Ffreqs, amp, gam) * Ffreqs[0]

    def pl_rn_basis_weight_pair(self, toas):
        """Return a Fourier design matrix and power law red noise weights.

        A Fourier design matrix contains the sine and cosine basis_functions
        in a Fourier series expansion.
        The weights used are the power-law PSD values at frequencies n/T,
        where n is in [1, TNREDC] and T is the total observing duration of
        the dataset.

        """
        return (self.get_noise_basis(toas), self.get_noise_weights(toas))

    def pl_rn_cov_matrix(self, toas):
        Fmat, phi = self.pl_rn_basis_weight_pair(toas)
        return np.dot(Fmat * phi[None, :], Fmat.T)


def get_ecorr_epochs(toas_table, dt=1, nmin=2):
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


def get_ecorr_nweights(toas_table, dt=1, nmin=2):
    """Get the number of epochs associated with each ECORR.
    This is equal to the number of weights of that ECORR."""
    return len(get_ecorr_epochs(toas_table, dt=dt, nmin=nmin))


def create_ecorr_quantization_matrix(toas_table, dt=1, nmin=2):
    """Create quantization matrix mapping TOAs to observing epochs.
    Only epochs with more than 1 TOA are included."""

    bucket_ind2 = get_ecorr_epochs(toas_table, dt=dt, nmin=nmin)

    U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
    for i, l in enumerate(bucket_ind2):
        U[l, i] = 1

    return U


def get_rednoise_freqs(t, nmodes, Tspan=None):
    """Frequency components for creating the red noise basis matrix."""

    T = Tspan if Tspan is not None else t.max() - t.min()

    f = np.linspace(1 / T, nmodes / T, nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[::2] = f
    Ffreqs[1::2] = f

    return Ffreqs


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

    Ffreqs = get_rednoise_freqs(t, nmodes, Tspan=Tspan)

    F[:, ::2] = np.sin(2 * np.pi * t[:, None] * Ffreqs[::2])
    F[:, 1::2] = np.cos(2 * np.pi * t[:, None] * Ffreqs[1::2])

    return F


def powerlaw(f, A=1e-16, gamma=5):
    """Power-law PSD.

    :param f: Sampling frequencies
    :param A: Amplitude of red noise [GW units]
    :param gamma: Spectral index of red noise process
    """

    fyr = 1 / 3.16e7
    return A**2 / 12.0 / np.pi**2 * fyr ** (gamma - 3) * f ** (-gamma)


#### the follow four functions are copied from enterprise_extensions.chromatic.solar_wind.py


def _dm_solar_close(n_earth, r_earth):
    return n_earth * AU_light_sec * AU_pc / r_earth


def _dm_solar(n_earth, theta, r_earth):
    return (np.pi - theta) * (
        n_earth * AU_light_sec * AU_pc / (r_earth * np.sin(theta))
    )


def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 2007 for more details.
    """
    return np.where(
        np.pi - theta >= 1e-5,
        _dm_solar(n_earth, theta, r_earth),
        _dm_solar_close(n_earth, r_earth),
    )


def theta_impact(planetssb, sunssb, pos_t):
    """
    Copied from `enterprise_extensions.chromatic.solar_wind`.
    Calculate the solar impact angle.

    ::param :planetssb Solar system barycenter time series supplied with
    enterprise.Pulsar objects.
    ::param :sunssb Solar system sun-to-barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.

    returns: Solar impact angle (rad), Distance to Earth (R_earth),
            impact distance (b), perpendicular distance (z_earth)
    """
    earth = planetssb[:, 2, :3]
    sun = sunssb[:, :3]
    earthsun = earth - sun
    R_earth = np.sqrt(np.einsum("ij,ij->i", earthsun, earthsun))
    Re_cos_theta_impact = np.einsum("ij,ij->i", earthsun, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact
