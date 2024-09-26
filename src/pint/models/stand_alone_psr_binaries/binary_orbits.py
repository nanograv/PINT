import re

import astropy.units as u
import numpy as np

from pint.utils import taylor_horner, taylor_horner_deriv


class Orbit:
    """Base class for implementing different parameterization of pulsar binary orbits.

    It should be constructed with a ``parent`` class, so that parameter lookups can be
    referred to the parent class by a custom ``__getattr__``.
    """

    def __init__(self, orbit_name, parent, orbit_params=[]):
        self.name = orbit_name
        self._parent = parent
        self.orbit_params = orbit_params

    def orbits(self):
        """Orbital phase (number of orbits since T0)."""
        raise NotImplementedError

    def orbit_phase(self):
        """Orbital phase (between zero and two pi)."""
        orbits = self.orbits()
        norbits = np.array(np.floor(orbits), dtype=int)
        return (orbits - norbits) * 2 * np.pi * u.rad

    def pbprime(self):
        """Instantaneous binary period as a function of time."""
        raise NotImplementedError

    def pbdot_orbit(self):
        """Reported value of PBDOT."""
        raise NotImplementedError

    def d_orbits_d_par(self, par):
        """Derivative of orbital phase with respect to some parameter.

        Note
        ----
        This gives the derivative of ``orbit_phase``, that is, it is scaled by 2 pi
        with respect to the derivative of ``orbits``.
        """
        par_obj = getattr(self, par)
        try:
            func = getattr(self, f"d_orbits_d_{par}")
        except AttributeError:

            def func():
                return np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit

        result = func()
        return result

    def d_pbprime_d_par(self, par):
        """Derivative of binary period with respect to some parameter."""
        par_obj = getattr(self, par)
        try:
            func = getattr(self, f"d_pbprime_d_{par}")
        except AttributeError:

            def func():
                return np.zeros(len(self.tt0)) * u.day / par_obj.unit

        result = func()
        return result

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            p = super().__getattribute__("_parent")
            if p is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'."
                ) from e
            else:
                return self._parent.__getattribute__(name)


class OrbitPB(Orbit):
    """Orbits using PB, PBDOT, XPBDOT.

    PBDOT is just the conventional derivative of the binary period.

    XPBDOT is something else, not completely clear what. It is added to PBDOT
    when computing ``orbits`` and its derivative with respect to PB, but it is
    subtracted from PBDOT when computing the derivative of orbits with respect
    to T0. It is also not included when computing ``pbdot_orbit``.
    """

    def __init__(self, parent, orbit_params=["PB", "PBDOT", "XPBDOT", "T0"]):
        super().__init__("orbitPB", parent, orbit_params)

    def orbits(self):
        """Orbital phase (number of orbits since T0)."""
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return (
            self.tt0 / PB - 0.5 * (PBDOT + XPBDOT) * (self.tt0 / PB) ** 2
        ).decompose()

    def pbprime(self):
        """Instantaneous binary period as a function of time."""
        return self.PB + self.PBDOT * self.tt0

    def pbdot_orbit(self):
        """Reported value of PBDOT."""
        return self.PBDOT

    def d_orbits_d_T0(self):
        """The derivatve of orbits with respect to T0."""
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return ((PBDOT - XPBDOT) * self.tt0 / PB - 1.0) * 2 * np.pi * u.rad / PB

    def d_orbits_d_PB(self):
        """dM/dPB this could be a generic function"""
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return (
            2
            * np.pi
            * u.rad
            * ((PBDOT + XPBDOT) * self.tt0**2 / PB**3 - self.tt0 / PB**2)
        )

    def d_orbits_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function"""
        PB = self.PB.to("second")
        return -np.pi * u.rad * self.tt0**2 / PB**2

    def d_orbits_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function"""
        PB = self.PB.to("second")
        return -np.pi * u.rad * self.tt0**2 / PB**2

    def d_pbprime_d_PB(self):
        return np.ones(len(self.tt0)) * u.Unit("")

    def d_pbprime_d_PBDOT(self):
        return self.tt0

    def d_pbprime_d_T0(self):
        if not np.isscalar(self.PBDOT):
            return -self.PBDOT
        result = np.empty(len(self.tt0))
        result.fill(-self.PBDOT.value)
        return result * u.Unit(self.PBDOT.unit)


class OrbitFBX(Orbit):
    """Orbits expressed in terms of orbital frequency and its derivatives FB0, FB1, FB2..."""

    def __init__(self, parent, orbit_params=["FB0"]):
        super().__init__("orbitFBX", parent, orbit_params)
        # add the rest of FBX parameters.
        indices = set()
        for k in self.binary_params:
            if re.match(r"FB\d+", k) is not None and k not in self.orbit_params:
                self.orbit_params += [k]
                indices.add(int(k[2:]))
        if indices != set(range(len(indices))):
            raise ValueError(
                f"Indices must be 0 up to some number k without gaps "
                f"but are {indices}."
            )

    def _FBXs(self):
        FBXs = [0 * u.Unit("")]
        ii = 0
        while f"FB{ii}" in self.orbit_params:
            FBXs.append(getattr(self, f"FB{ii}"))
            ii += 1
        return FBXs

    def orbits(self):
        """Orbital phase (number of orbits since T0)."""
        orbits = taylor_horner(self.tt0, self._FBXs())
        return orbits.decompose()

    def pbprime(self):
        """Instantaneous binary period as a function of time."""
        orbit_freq = taylor_horner_deriv(self.tt0, self._FBXs(), 1)
        return 1.0 / orbit_freq

    def pbdot_orbit(self):
        """Reported value of PBDOT."""
        orbit_freq_dot = taylor_horner_deriv(self.tt0, self._FBXs(), 2)
        return -(self.pbprime() ** 2) * orbit_freq_dot

    def d_orbits_d_par(self, par):
        return (
            self.d_orbits_d_FBX(par)
            if re.match(r"FB\d+", par) is not None
            else super().d_orbits_d_par(par)
        )

    def d_orbits_d_FBX(self, FBX):
        par = getattr(self, FBX)
        ii = 0
        FBXs = [0 * u.Unit("")]
        while f"FB{ii}" in self.orbit_params:
            if f"FB{ii}" != FBX:
                FBXs.append(0.0 * getattr(self, f"FB{ii}").unit)
            else:
                FBXs.append(1.0 * getattr(self, f"FB{ii}").unit)
                break
            ii += 1
        d_orbits = taylor_horner(self.tt0, FBXs) / par.unit
        return d_orbits.decompose() * 2 * np.pi * u.rad

    def d_pbprime_d_FBX(self, FBX):
        par = getattr(self, FBX)
        ii = 0
        FBXs = [0 * u.Unit("")]
        while f"FB{ii}" in self.orbit_params:
            if f"FB{ii}" != FBX:
                FBXs.append(0.0 * getattr(self, f"FB{ii}").unit)
            else:
                FBXs.append(1.0 * getattr(self, f"FB{ii}").unit)
                break
            ii += 1
        d_FB = taylor_horner_deriv(self.tt0, FBXs, 1) / par.unit
        return -(self.pbprime() ** 2) * d_FB

    def d_pbprime_d_par(self, par):
        par_obj = getattr(self, par)
        return (
            self.d_pbprime_d_FBX(par)
            if re.match(r"FB\d+", par) is not None
            else np.zeros(len(self.tt0)) * u.second / par_obj.unit
        )


class OrbitWaves(Orbit):
    """Orbit with orbital phase variations described by a Fourier series"""

    def __init__(
        self,
        parent,
        orbit_params=[
            "PB",
            "TASC",
            "ORBWAVE_OM",
            "ORBWAVE_EPOCH",
            "ORBWAVEC0",
            "ORBWAVES0",
        ],
    ):
        # Generalize this to allow for instantiation with OrbitWavesFBX
        label = self.__class__.__name__
        label = label[0].lower() + label[1:]
        super().__init__(label, parent, orbit_params)

        Cindices = set()
        Sindices = set()

        nc = 0
        ns = 0

        for k in self.binary_params:
            if re.match(r"ORBWAVEC\d+", k) is not None:
                if k not in self.orbit_params:
                    self.orbit_params += [k]
                Cindices.add(int(k[8:]))
                nc += 1
            if re.match(r"ORBWAVES\d+", k) is not None:
                if k not in self.orbit_params:
                    self.orbit_params += [k]
                Sindices.add(int(k[8:]))
                ns += 1

        if Cindices != set(range(len(Cindices))) or Sindices != set(
            range(len(Sindices))
        ):
            raise ValueError(
                f"Orbwave indices must be 0 up to some number k without gaps "
                f"but are {Cindices} and {Sindices}."
            )

        if nc != ns:
            raise ValueError(
                f"Must have equal number of sine/cosine ORBWAVE pairs "
                f"but have {ns} and {nc}."
            )

        if self.ORBWAVEC0 is None or self.ORBWAVES0 is None:
            raise ValueError("The ORBWAVEC0 or ORBWAVES0 parameter is unset")

        self.nwaves = ns

    def _ORBWAVEs(self):
        ORBWAVEs = np.zeros((self.nwaves, 2)) * u.Unit("")

        for ii in range(self.nwaves):
            ORBWAVEs[ii, 0] = getattr(self, f"ORBWAVEC{ii}")
            ORBWAVEs[ii, 1] = getattr(self, f"ORBWAVES{ii}")

        return ORBWAVEs

    def _tw(self):
        tw = self.t - self.ORBWAVE_EPOCH.value * u.d
        return tw

    def _deltaPhi(self):
        tw = self._tw()
        waveamps = self._ORBWAVEs()
        OM = self.ORBWAVE_OM.to("radian/second")

        nh = np.arange(self.nwaves) + 1
        Cwaves = waveamps[:, 0, None] * np.cos(OM * nh[:, None] * tw[None, :])
        Swaves = waveamps[:, 1, None] * np.sin(OM * nh[:, None] * tw[None, :])

        delta_Phi = np.sum(Cwaves + Swaves, axis=0)

        return delta_Phi

    def _d_deltaPhi_dt(self):
        tw = self._tw()
        waveamps = self._ORBWAVEs()
        OM = self.ORBWAVE_OM.to("radian/second")

        nh = np.arange(self.nwaves) + 1
        Cwaves = (
            -OM
            * nh[:, None]
            * waveamps[:, 0, None]
            * np.sin(OM * nh[:, None] * tw[None, :])
        )
        Swaves = (
            OM
            * nh[:, None]
            * waveamps[:, 1, None]
            * np.cos(OM * nh[:, None] * tw[None, :])
        )

        d_deltaPhi_dt = np.sum(Cwaves + Swaves, axis=0)

        return d_deltaPhi_dt.to(u.Unit("1/s"), equivalencies=u.dimensionless_angles())

    def _d2_deltaPhi_dt2(self):
        tw = self._tw()
        waveamps = self._ORBWAVEs()
        OM = self.ORBWAVE_OM.to("radian/second")

        nh = np.arange(self.nwaves) + 1
        Cwaves = (
            -((OM * nh[:, None]) ** 2)
            * waveamps[:, 0, None]
            * np.cos(OM * nh[:, None] * tw[None, :])
        )
        Swaves = (
            -((OM * nh[:, None]) ** 2)
            * waveamps[:, 1, None]
            * np.sin(OM * nh[:, None] * tw[None, :])
        )

        d2_deltaPhi_dt2 = np.sum(Cwaves + Swaves, axis=0)

        return d2_deltaPhi_dt2.to(
            u.Unit("1/s^2"), equivalencies=u.dimensionless_angles()
        )

    def orbits(self):
        """Orbital phase (number of orbits since TASC)."""
        PB = self.PB.to("second")
        orbits = self.tt0 / PB

        dphi = self._deltaPhi()

        orbits += dphi
        return orbits.decompose()

    def pbprime(self):
        """Instantaneous binary period as a function of time."""
        PB = self.PB.to("second")
        FB0 = 1.0 / PB

        FB0_shift = self._d_deltaPhi_dt()

        return 1.0 / (FB0 + FB0_shift).decompose()

    def pbdot_orbit(self):
        """Reported value of PBDOT."""
        FB1 = self._d2_deltaPhi_dt2()

        return -(self.pbprime() ** 2) * FB1

    def d_orbits_d_TASC(self):
        PB = self.PB.to("second")
        return -(1 / PB) * 2 * np.pi * u.rad

    def d_orbits_d_PB(self):
        PB = self.PB.to("second")
        return -(self.tt0 / PB**2).decompose() * 2 * np.pi * u.rad

    def d_orbits_d_orbwave(self, par):
        tw = self._tw()
        WOM = self.ORBWAVE_OM.to("radian/second")
        nh = int(par[8:]) + 1

        return (
            (np.cos(WOM * nh * tw) if par[7] == "C" else np.sin(WOM * nh * tw))
            * 2
            * np.pi
            * u.rad
        )

    def d_pbprime_d_PB(self):
        PB = self.PB.to("second")
        FB0 = 1.0 / PB

        FB0_shift = self._d_deltaPhi_dt()

        FB0_prime = FB0 + FB0_shift

        return (1.0 / ((FB0_prime * PB) ** 2)).decompose()

    def d_pbprime_d_orbwave(self, par):
        tw = self._tw()
        WOM = self.ORBWAVE_OM.to("radian/second")

        nh = int(par[8:]) + 1
        if par[7] == "C":
            d_deltaFB0_d_orbwave = -WOM * nh * np.sin(WOM * nh * tw)
        else:
            d_deltaFB0_d_orbwave = WOM * nh * np.cos(WOM * nh * tw)

        # use of pbprime should account for both Taylor and Fourier terms
        return (-self.pbprime() ** 2 * d_deltaFB0_d_orbwave).to(
            "d", equivalencies=u.dimensionless_angles()
        )

    def d_orbits_d_par(self, par):
        return (
            self.d_orbits_d_orbwave(par)
            if re.match(r"ORBWAVE[CS]\d+", par) is not None
            else super().d_orbits_d_par(par)
        )

    def d_pbprime_d_par(self, par):
        par_obj = getattr(self, par)

        if re.match(r"ORBWAVE[CS]\d+", par) is not None:
            return self.d_pbprime_d_orbwave(par)

        return super().d_pbprime_d_par(par)


class OrbitWavesFBX(OrbitWaves):
    """Orbit with orbital phase variations described both by FBX terms and
    by a Fourier series"""

    def __init__(
        self,
        parent,
        orbit_params=[
            "FB0",
            "TASC",
            "ORBWAVE_OM",
            "ORBWAVE_EPOCH",
            "ORBWAVEC0",
            "ORBWAVES0",
        ],
    ):
        if not hasattr(parent, "FB1"):
            parent.add_param(
                floatParameter(name="FB1", units=u.Hz**2, long_double=True, value=0)
            )
            parent.binary_instance.add_binary_params("FB1", parent.FB1)
            orbit_params += ["FB1"]
        super().__init__(parent, orbit_params)

    def orbits(self):
        """Orbital phase (number of orbits since TASC)."""
        tt = self.tt0
        orbits = self.FB0 * tt * (1 + (0.5 * self.FB1 / self.FB0) * tt)
        orbits += self._deltaPhi()
        return orbits.decompose()

    def pbprime(self):
        """Instantaneous binary period as a function of time."""

        orbit_freq = self.FB0 + self.FB1 * self.tt0
        orbit_freq += self._d_deltaPhi_dt()
        return 1.0 / orbit_freq.decompose()

    def pbdot_orbit(self):
        """Reported value of PBDOT."""
        orbit_freq_dot = self._d2_deltaPhi_dt2() + self.FB1
        return -(self.pbprime() ** 2) * orbit_freq_dot

    def d_orbits_d_TASC(self):
        return -self.FB0.to("Hz") * 2 * np.pi * u.rad

    def d_orbits_d_FB0(self):
        return self.tt0.decompose() * (2 * np.pi * u.rad)

    def d_orbits_d_FB1(self):
        return (0.5 * self.tt0**2).decompose() * (2 * np.pi * u.rad)

    def d_pbprime_d_FB0(self):
        return -1 * self.pbprime() ** 2

    def d_pbprime_d_FB1(self):
        return -self.tt0 * self.pbprime() ** 2

    def d_pbprime_d_TASC(self):
        return self.FB1 * self.pbprime() ** 2
