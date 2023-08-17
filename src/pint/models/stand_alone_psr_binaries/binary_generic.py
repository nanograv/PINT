"""Parent class for internal binary models."""

import astropy.constants as c
import astropy.units as u
import numpy as np

from loguru import logger as log

from erfa import DAYSEC as SECS_PER_DAY

from pint import Tsun, ls
from pint.models.stand_alone_psr_binaries.binary_orbits import OrbitPB

SECS_PER_JUL_YEAR = SECS_PER_DAY * 365.25


class PSR_BINARY:
    """A base (generic) object for psr binary models.

    In this class, a set of generally used binary parameters and several commonly used
    calculations are defined. For each binary model, the specific parameters and
    calculations are defined in the subclass.

    A binary model takes the solar system barycentric time (ssb time) as input.
    When a binary model is instantiated, the parameters are set to the default
    values and input time is not initialized. The update those values, method
    `.update_input()` should be used.

    Example of build a specific binary model class::

        >>> from pint.models.stand_alone_psr_binaries.pulsar_binary import PSR_BINARY
        >>> import numpy
        >>> class foo(PSR_BINARY):
                def __init__(self):
                    # This is to initialize the mother class attributes.
                    super().__init__()
                    self.binary_name = 'foo'
                    # Add parameter that specific for my_binary, with default value and units
                    self.param_default_value.update({'A0':0*u.second,'B0':0*u.second,
                                           'DR':0*u.Unit(''),'DTH':0*u.Unit(''),
                                           'GAMMA':0*u.second,})
                    self.set_param_values() # This is to set all the parameters to attributes
                    self.binary_delay_funcs += [self.foo_delay]
                    self.d_binarydelay_d_par_funcs += [self.d_foo_delay_d_par]
                    # If you have intermediate value in the calculation
                    self.dd_interVars = ['er','eTheta','beta','alpha','Dre','Drep','Drepp',
                                         'nhat', 'TM2']
                    self.add_inter_vars(self.dd_interVars)

                def foo_delay(self):
                    pass

                def d_foo_delay_d_par(self):
                    pass
        >>> # To build a model instance
        >>> binary_foo = foo()
        >>> # binary_foo class has the default parameter value without toa input.
        >>> # Update the toa input and parameters
        >>> t = numpy.linspace(54200.0,55000.0,800)
        >>> parameters_dict = {'A0':0.5,'ECC':0.01}
        >>> binary_foo.update_input(t, parameters_dict)
        >>> # Now the binary delay and derivatives can be computed.

    To access the binary model class from pint platform, a pint pulsar binary
    wrapper is needed. See docstrings in the source code of pint/models/pulsar_binary
    class `PulsarBinary`.

    Included general parameters:
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param A1DOT:       Time-derivative of A1 (lt-sec/sec)
    @param T0:          Time of periastron passage (barycentric MJD)
    @param OM:          Omega (longitude of periastron) [deg]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param XPBDOT:      Rate of change of orbital period minus GR prediction
    @param OMDOT:       Time-derivative of OMEGA [0.0]

    Intermediate variables calculation method are given here:
    Eccentric Anomaly               E (not parameter ECC)
    Mean Anomaly                    M
    True Anomaly                    nu
    Eccentric                       ecc
    Longitude of periastron         omega
    projected semi-major axis of orbit   a1
    TM2

    """

    def __init__(
        self,
    ):
        # Necessary parameters for all binary model
        self.binary_name = None
        self.param_default_value = {
            "PB": np.longdouble(10.0) * u.day,
            "PBDOT": 0.0 * u.day / u.day,
            "ECC": 0.0 * u.Unit(""),
            "EDOT": 0.0 / u.second,
            "A1": 10.0 * ls,
            "A1DOT": 0.0 * ls / u.second,
            "T0": np.longdouble(54000.0) * u.day,
            "OM": 0.0 * u.deg,
            "OMDOT": 0.0 * u.deg / u.year,
            "XPBDOT": 0.0 * u.day / u.day,
            "M2": 0.0 * u.M_sun,
            "SINI": 0 * u.Unit(""),
            "GAMMA": 0 * u.second,
            "FB0": 1.1574e-6 * u.Unit("") / u.second,
        }
        # For Binary phase calculation
        self.param_default_value.update(
            {
                "P0": 1.0 * u.second,
                "P1": 0.0 * u.second / u.second,
                "PEPOCH": np.longdouble(54000.0) * u.day,
            }
        )

        self.param_aliases = {"ECC": ["E"], "EDOT": ["ECCDOT"], "A1DOT": ["XDOT"]}
        self.binary_params = list(self.param_default_value.keys())
        self.inter_vars = ["E", "M", "nu", "ecc", "omega", "a1", "TM2"]
        self.cache_vars = ["E", "nu"]
        self.binary_delay_funcs = []
        self.d_binarydelay_d_par_funcs = []
        self.orbits_cls = OrbitPB(self, ["PB", "PBDOT", "XPBDOT", "T0"])

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, val):
        self._t = val
        if hasattr(self, "T0"):
            self._tt0 = self.get_tt0(self._t)

    @property
    def T0(self):
        return self._T0

    @T0.setter
    def T0(self, val):
        self._T0 = val
        if hasattr(self, "_t"):
            self._tt0 = self.get_tt0(self._t)

    @property
    def tt0(self):
        return self._tt0

    def update_input(self, **updates):
        """Update the toas and parameters."""
        # Update toas
        if "barycentric_toa" in updates:
            self.t = np.atleast_1d(updates["barycentric_toa"])
        # Update observatory position.
        if "obs_pos" in updates:
            self.obs_pos = np.atleast_1d(updates["obs_pos"])

        if "psr_pos" in updates:
            self.psr_pos = np.atleast_1d(updates["psr_pos"])
        # update parameters
        d_list = ["barycentric_toa", "obs_pos", "psr_pos"]
        parameters = {key: value for key, value in updates.items() if key not in d_list}
        self.set_param_values(parameters)

        # Switch the cache off
        # NOTE Having cache is needs to be very careful.
        for cv in self.cache_vars:
            setattr(self, f"_{cv}", None)

    def set_param_values(self, valDict=None):
        """Set the parameters and assign values,

        If the valDict is not provided, it will set parameter as default value
        """
        if valDict is None:
            for par in self.param_default_value.keys():
                setattr(self, par.upper(), self.param_default_value[par])
        else:
            for par in valDict.keys():
                if par not in self.binary_params:  # search for aliases
                    parname = self.search_alias(par)
                    if parname is None:
                        raise AttributeError(
                            f"Can not find parameter {par} in {self.binary_name} model"
                        )
                else:
                    parname = par
                if valDict[par] is None:
                    setattr(self, parname, self.param_default_value[parname])
                    continue
                if not hasattr(valDict[par], "unit"):
                    bm_par = getattr(self, parname)
                    val = (
                        valDict[par] * getattr(self, parname).unit
                        if hasattr(bm_par, "unit")
                        else valDict[par]
                    )
                else:
                    val = valDict[par]
                setattr(self, parname, val)

    def add_binary_params(self, parameter, defaultValue, unit=False):
        """Add one parameter to the binary class."""
        if parameter not in self.binary_params:
            self.binary_params.append(parameter)
            if not hasattr(defaultValue, "unit") and unit:
                log.warning(
                    f"Binary parameter values generally have units."
                    f" Treating parameter {parameter} as a dimensionless quantity."
                )
                self.param_default_value[parameter] = defaultValue * u.Unit("")
            else:
                self.param_default_value[parameter] = defaultValue
            setattr(self, parameter, self.param_default_value[parameter])

    def add_inter_vars(self, interVars):
        if not isinstance(interVars, list):
            interVars = [interVars]
        for v in interVars:
            if v not in self.inter_vars:
                self.inter_vars.append(v)

    def search_alias(self, parname):
        for pn in self.param_aliases.keys():
            return pn if parname in self.param_aliases[pn] else None

    def binary_delay(self):
        """Returns total pulsar binary delay.

        Returns
        -------
        float
            Pulsar binary delay in the units of second

        """
        bdelay = np.longdouble(np.zeros(len(self.t))) * u.s
        for bdf in self.binary_delay_funcs:
            bdelay += bdf()
        return bdelay

    def d_binarydelay_d_par(self, par):
        """Get the binary delay derivatives respect to parameters.

        Parameters
        ----------
        par : str
            Parameter name.
        """
        # search for aliases
        if par not in self.binary_params and self.search_alias(par) is None:
            raise AttributeError(
                f"Can not find parameter {par} in {self.binary_name} model"
            )

        # Get first derivative in the delay derivative function
        result = self.d_binarydelay_d_par_funcs[0](par)
        if len(self.d_binarydelay_d_par_funcs) > 1:
            for df in self.d_binarydelay_d_par_funcs[1:]:
                result += df(par)

        return result

    def prtl_der(self, y, x):
        """Find the partial derivatives in binary model pdy/pdx

        Parameters
        ----------
        y : str
           Name of variable to be differentiated
        x : str
           Name of variable the derivative respect to

        Returns
        -------
        np.array
           The derivatives pdy/pdx
        """
        if y not in self.binary_params + self.inter_vars:
            errorMesg = f"{y} is not in binary parameter and variables list."
            raise ValueError(errorMesg)

        if x not in self.inter_vars + self.binary_params:
            errorMesg = f"{x} is not in binary parameters and variables list."
            raise ValueError(errorMesg)
        # derivative to itself
        if x == y:
            return np.longdouble(np.ones(len(self.tt0))) * u.Unit("")
        # Get the unit right

        yAttr = getattr(self, y)
        xAttr = getattr(self, x)
        U = [None, None]
        for i, attr in enumerate([yAttr, xAttr]):
            if hasattr(attr, "units"):  # If attr is a PINT Parameter class type
                U[i] = attr.units
            elif hasattr(attr, "unit"):  # If attr is a Quantity type
                U[i] = attr.unit
            elif hasattr(attr, "__call__"):  # If attr is a method
                U[i] = attr().unit
            else:
                raise TypeError(f"{type(attr)}can not get unit")

            # U[i] = 1*U[i]

            # commonU = list(set(U[i].unit.bases).intersection([u.rad,u.deg]))
            # if commonU != []:
            #     strU = U[i].unit.to_string()
            #     for cu in commonU:
            #         scu = cu.to_string()
            #         strU = strU.replace(scu,'1')
            #     U[i] = U[i].to(strU, equivalencies=u.dimensionless_angles()).unit

        yU = U[0]
        xU = U[1]
        # Call derivative functions
        derU = yU / xU
        if hasattr(self, f"d_{y}_d_{x}"):
            dername = f"d_{y}_d_{x}"
            result = getattr(self, dername)()

        elif hasattr(self, f"d_{y}_d_par"):
            dername = f"d_{y}_d_par"
            result = getattr(self, dername)(x)

        else:
            result = np.longdouble(np.zeros(len(self.tt0)))

        if hasattr(result, "unit"):
            return result.to(derU, equivalencies=u.dimensionless_angles())
        else:
            return result * derU

    def compute_eccentric_anomaly(self, eccentricity, mean_anomaly):
        """Solve the Kepler Equation, E - e * sin(E) = M

        Parameters
        ----------
        eccentricity : array_like
            Eccentricity of binary system
        mean_anomaly : array_like
            Mean anomaly of the binary system

        Returns
        -------
        array_like
            The eccentric anomaly in radians, given a set of mean_anomalies
            in radians.
        """
        if hasattr(eccentricity, "unit"):
            # FIXME: isn't this an error?
            e = np.longdouble(eccentricity).value
        else:
            e = eccentricity
        if any(e < 0) or any(e >= 1):
            raise ValueError("Eccentricity should be in the range of [0,1).")

        if hasattr(mean_anomaly, "unit"):
            ma = np.longdouble(mean_anomaly).value
        else:
            ma = mean_anomaly
        k = lambda E: E - e * np.sin(E) - ma  # Kepler Equation
        dk = lambda E: 1 - e * np.cos(E)  # derivative Kepler Equation
        U = ma
        while np.max(abs(k(U))) > 5e-15:  # Newton-Raphson method
            U = U - k(U) / dk(U)
        return U * u.rad

    def get_tt0(self, barycentricTOA):
        """tt0 = barycentricTOA - T0"""
        if barycentricTOA is None or self.T0 is None:
            return None
        T0 = self.T0
        if not hasattr(barycentricTOA, "unit") or barycentricTOA.unit is None:
            barycentricTOA = barycentricTOA * u.day
        return (barycentricTOA - T0).to("second")

    def ecc(self):
        """Calculate eccentricity with EDOT"""
        if hasattr(self, "_tt0"):
            return self.ECC + (self.tt0 * self.EDOT).decompose()
        return self.ECC

    def d_ecc_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.EDOT.value)
        return result * u.Unit(self.EDOT.unit)

    def d_ecc_d_ECC(self):
        return np.longdouble(np.ones(len(self.tt0))) * u.Unit("")

    def d_ecc_d_EDOT(self):
        return self.tt0

    def a1(self):
        return self.A1 + self.tt0 * self.A1DOT if hasattr(self, "_tt0") else self.A1

    def d_a1_d_A1(self):
        return np.longdouble(np.ones(len(self.tt0))) * u.Unit("")

    def d_a1_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.A1DOT.value)
        return result * u.Unit(self.A1DOT.unit)

    def d_a1_d_A1DOT(self):
        return self.tt0

    def d_a1_d_par(self, par):
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        try:
            func = getattr(self, f"d_a1_d_{par}")
        except AttributeError:
            func = lambda: np.zeros(len(self.tt0)) * self.A1.unit / par_obj.unit
        return func()

    def pb(self):
        return self.orbits_cls.pbprime()

    def d_pb_d_par(self, par):
        """derivative for pbprime respect to binary parameter.

        Parameters
        ----------
        par : string
             parameter name

        Returns
        -------
        Derivative of M respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        result = self.orbits_cls.d_pbprime_d_par(par)
        return result.to(self.PB.unit / par_obj.unit)

    def pbdot(self):
        return self.orbits_cls.pbdot_orbit()

    def orbits(self):
        return self.orbits_cls.orbits()

    def M(self):
        """Orbit phase."""
        return self.orbits_cls.orbit_phase()

    def d_M_d_par(self, par):
        """derivative for M respect to binary parameter.

        Parameters
        ----------
        par : string
             parameter name

        Returns
        -------
        Derivative of M respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        result = self.orbits_cls.d_orbits_d_par(par)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            result = result.to(u.Unit("") / par_obj.unit)
        return result

    ###############################################

    def E(self):
        """Eccentric Anomaly"""
        if not hasattr(self, "_E") or self._E is None:
            self._E = self.compute_eccentric_anomaly(self.ecc(), self.M())
        return self._E

    # Analytically calculate derivatives.

    def d_E_d_T0(self):
        """Analytic derivative

        d(E-e*sinE)/dT0 = dM/dT0
        dE/dT0(1-cosE*e)-de/dT0*sinE = dM/dT0
        dE/dT0(1-cosE*e)+eDot*sinE = dM/dT0
        """
        RHS = self.prtl_der("M", "T0")
        E = self.E()
        EDOT = self.EDOT
        ecc = self.ecc()
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (RHS - EDOT * np.sin(E)) / (1.0 - np.cos(E) * ecc)

    def d_E_d_ECC(self):
        E = self.E()
        return np.sin(E) / (1.0 - self.ecc() * np.cos(E))

    def d_E_d_EDOT(self):
        return self.tt0 * self.d_E_d_ECC()

    def d_E_d_par(self, par):
        """derivative for E respect to binary parameter.

        Parameters
        ----------
        par : string
             parameter name

        Returns
        -------
        Derivative of E respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        try:
            return getattr(self, f"d_E_d_{par}")()
        except AttributeError:
            if par in self.orbits_cls.orbit_params:
                d_M_d_par = self.d_M_d_par(par)
                return d_M_d_par / (1.0 - np.cos(self.E()) * self.ecc())
            else:
                E = self.E()
                return np.zeros(len(self.tt0)) * E.unit / par_obj.unit
        return func()

    def nu(self):
        """True anomaly  (Ae)"""
        if not hasattr(self, "_nu") or self._nu is None:
            ecc = self.ecc()
            nu = 2 * np.arctan(
                np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(self.E() / 2.0)
            )
            # Normalize True anomaly to on orbit.
            nu[nu < 0] += 2 * np.pi * u.rad
            nu2 = 2 * np.pi * self.orbits() * u.rad + nu - self.M()
            self._nu = nu2
        return self._nu

    def d_nu_d_E(self):
        nu = self.nu()
        E = self.E()
        ecc = self.ecc()
        brack1 = (1 + ecc * np.cos(nu)) / (1 - ecc * np.cos(E))
        brack2 = np.sin(E) / np.sin(nu)
        return brack1 * brack2

    def d_nu_d_ecc(self):
        ecc = self.ecc()
        E = self.E()
        return np.sin(E) ** 2 / (ecc * np.cos(E) - 1) ** 2 / np.sin(self.nu())

    def d_nu_d_T0(self):
        """Analytic calculation.

        dnu/dT0 = dnu/de*de/dT0+dnu/dE*dE/dT0
        de/dT0 = -EDOT
        """
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return self.d_nu_d_ecc() * (-self.EDOT) + self.d_nu_d_E() * self.d_E_d_T0()

    # def d_nu_d_PB(self):
    #     """dnu(e,E)/dPB = dnu/de*de/dPB+dnu/dE*dE/dPB
    #        de/dPB = 0
    #        dnu/dPB = dnu/dE*dE/dPB
    #     """
    #     return self.d_nu_d_E()*self.d_E_d_PB()
    #
    #
    # def d_nu_d_PBDOT(self):
    #     """dnu(e,E)/dPBDOT = dnu/de*de/dPBDOT+dnu/dE*dE/dPBDOT
    #        de/dPBDOT = 0
    #        dnu/dPBDOT = dnu/dE*dE/dPBDOT
    #     """
    #     return self.d_nu_d_E()*self.d_E_d_PBDOT()
    #
    #
    # def d_nu_d_XPBDOT(self):
    #     """dnu/dPBDOT = dnu/dE*dE/dPBDOT
    #     """
    #     return self.d_nu_d_E()*self.d_E_d_XPBDOT()

    def d_nu_d_ECC(self):
        """Analytic calculation.

        dnu(e,E)/dECC = dnu/de*de/dECC+dnu/dE*dE/dECC
        de/dECC = 1
        dnu/dPBDOT = dnu/dE*dE/dECC+dnu/de
        """
        return self.d_nu_d_ecc() + self.d_nu_d_E() * self.d_E_d_ECC()

    def d_nu_d_EDOT(self):
        return self.tt0 * self.d_nu_d_ECC()

    def d_nu_d_par(self, par):
        """derivative for nu respect to binary parameter.

        Parameters
        ----------
        par : string
             parameter name
        Returns
        -------
        Derivative of nu respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        try:
            return getattr(self, f"d_nu_d_{par}")()
        except AttributeError:
            if par in self.orbits_cls.orbit_params:
                return self.d_nu_d_E() * self.d_E_d_par(par)

            nu = self.nu()
            return np.zeros(len(self.tt0)) * nu.unit / par_obj.unit

    def omega(self):
        PB = self.pb().to("second")
        OMDOT = self.OMDOT
        OM = self.OM
        return OM + OMDOT * self.tt0

    def d_omega_d_par(self, par):
        """derivative for omega respect to user input Parameter.

        Parameters
        ----------
        par : string
             parameter name
        Returns
        -------
        Derivative of omega respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)

        OMDOT = self.OMDOT
        OM = self.OM
        if par not in ["OM", "OMDOT", "T0"]:
            return np.longdouble(np.zeros(len(self.tt0))) * self.OM.unit / par_obj.unit
        dername = f"d_omega_d_{par}"
        return getattr(self, dername)()

    def d_omega_d_OM(self):
        """dOmega/dOM = 1"""
        return np.longdouble(np.ones((len(self.tt0)))) * u.Unit("")

    def d_omega_d_OMDOT(self):
        """dOmega/dOMDOT = tt0"""
        return self.tt0

    def d_omega_d_T0(self):
        """dOmega/dPB = dnu/dPB*k+dk/dPB*nu"""
        result = np.empty(len(self.tt0))
        result.fill(-self.EDOT.value)
        return result * u.Unit(self.EDOT.unit)

    def TM2(self):
        return self.M2.value * Tsun

    def d_TM2_d_M2(self):
        return Tsun / (1.0 * u.Msun)

    def pbprime(self):
        return self.pb() - self.pbdot() * self.tt0

    def P(self):
        return self.P0 + self.P1 * (self.t - self.PEPOCH).to("second")

    def t0(self):
        return self.t - self.PEPOCH

    def Doppler(self):
        a1 = self.a1() / c.c
        return 2 * np.pi * a1 / (self.pbprime() * np.sqrt(1 - self.ecc() ** 2))

    def d_Pobs_d_P0(self):
        geom = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        ds = self.Doppler() * geom
        return 1.0 + ds

    def d_Pobs_d_P1(self):
        geom = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        ds = self.Doppler() * geom
        return self.t0() * (1 + ds)

    def d_Pobs_d_A1(self):
        geom = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        return (
            2
            * np.pi
            * self.P()
            * geom
            / (self.pbprime() * np.sqrt(1 - self.ecc() ** 2))
        )

    def d_Pobs_d_PB(self):
        geom1 = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        geom2 = -np.cos(self.nu()) * np.sin(self.omega()) - np.sin(self.nu()) * np.cos(
            self.omega()
        )
        pref1 = (
            -self.P()
            * 2
            * np.pi
            * self.a1()
            / (self.pbprime() ** 2 * np.sqrt(1 - self.ecc() ** 2))
            * self.SECS_PER_DAY
        )
        pref2 = self.P() * self.Doppler() * self.d_nu_d_PB()
        return pref1 * geom1 + pref2 * geom2

    def d_Pobs_d_PBDOT(self):
        geom1 = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        geom2 = -np.cos(self.nu()) * np.sin(self.omega()) - np.sin(self.nu()) * np.cos(
            self.omega()
        )
        pref1 = (
            self.P()
            * self.tt0
            * 2
            * np.pi
            * self.a1()
            / (self.pbprime() ** 2 * np.sqrt(1 - self.ecc() ** 2))
        )
        pref2 = self.P() * self.Doppler() * self.d_nu_d_PBDOT()
        return pref1 * geom1 + pref2 * geom2

    def d_Pobs_d_OM(self):
        geom = -np.sin(self.nu()) * np.cos(self.omega()) - (
            np.cos(self.nu()) + self.ecc()
        ) * np.sin(self.omega())
        return self.P() * self.Doppler() * geom * self.DEG2RAD

    def d_Pobs_d_ECC(self):
        geom1 = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        geom2 = -np.cos(self.nu()) * np.sin(self.omega()) - np.sin(self.nu()) * np.cos(
            self.omega()
        )
        pref1 = (
            self.P()
            * self.ecc()
            * 2
            * np.pi
            * self.a1()
            / (self.pbprime() * (1 - self.ecc() ** 2) ** (1.5))
        )
        pref2 = self.P() * self.Doppler() * self.d_nu_d_ECC()
        return (
            pref1 * geom1
            + pref2 * geom2
            + self.P0 * self.Doppler() * np.cos(self.omega())
        )

    def d_Pobs_d_T0(self):
        geom1 = -np.sin(self.nu()) * np.sin(self.omega()) + (
            np.cos(self.nu()) + self.ecc()
        ) * np.cos(self.omega())
        geom2 = -np.cos(self.nu()) * np.sin(self.omega()) - np.sin(self.nu()) * np.cos(
            self.omega()
        )
        pref1 = (
            -self.P()
            * self.pbdot()
            * 2
            * np.pi
            * self.a1()
            / (self.pbprime() ** 2 * np.sqrt(1 - self.ecc() ** 2))
            * self.SECS_PER_DAY
        )
        pref2 = self.P() * self.Doppler() * self.d_nu_d_T0()
        return pref1 * geom1 + pref2 * geom2

    def d_Pobs_d_EDOT(self):
        return self.tt0 * self.d_Pobs_d_ECC()

    def d_Pobs_d_OMDOT(self):
        return self.tt0 * self.d_Pobs_d_OM() / self.SECS_PER_YEAR

    def d_Pobs_d_A1DOT(self):
        return self.tt0 * self.d_Pobs_d_A1()

    ############## Calculation for design matrix  ################
    def Pobs_designmatrix(self, params):
        npars = len(params)
        M = np.zeros((len(self.t), npars))

        for ii, par in enumerate(params):
            dername = f"d_Pobs_d_{par}"
            M[:, ii] = getattr(self, dername)()
        return M

    def delay_designmatrix(self, params):
        npars = len(params)
        M = np.zeros((len(self.t), npars))

        for ii, par in enumerate(params):
            dername = f"d_delay_d_{par}"
            M[:, ii] = getattr(self, dername)()

        return M
