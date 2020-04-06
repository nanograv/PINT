import re

import astropy.units as u
import numpy as np

import pint.utils as ut
from pint.utils import taylor_horner, taylor_horner_deriv


class orbits(object):
    """This is a base class for implementing different parameterization
    of pulsar binary orbits
    """

    def __init__(self, orbit_name, parent, orbit_params=[]):
        self.name = orbit_name
        self._parent = parent
        self.orbit_params = orbit_params

    def orbits(self):
        raise NotImplementedError

    def orbit_phase(self):
        orbits = self.orbits()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits) * 2 * np.pi * u.rad
        return phase

    def pbprime(self):
        raise NotImplementedError

    def pbdot_orbit(self):
        raise NotImplementedError

    def d_orbits_d_par(self, par):
        par_obj = getattr(self, par)
        try:
            func = getattr(self, "d_orbits_d_" + par)
        except:
            func = lambda: np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit
        result = func()
        return result

    def d_pbprime_d_par(self, par):
        par_obj = getattr(self, par)
        try:
            func = getattr(self, "d_pbprime_d_" + par)
        except:
            func = lambda: np.zeros(len(self.tt0)) * u.day / par_obj.unit
        result = func()
        return result

    def __getattr__(self, name):
        try:
            return super(orbits, self).__getattribute__(name)
        except AttributeError:
            try:
                p = super(orbits, self).__getattribute__("_parent")
                if p is None:
                    raise AttributeError(
                        "'%s' object has no attribute '%s'."
                        % (self.__class__.__name__, name)
                    )
                else:
                    return self._parent.__getattribute__(name)
            except:
                raise AttributeError(
                    "'%s' object has no attribute '%s'."
                    % (self.__class__.__name__, name)
                )


class OrbitPB(orbits):
    def __init__(self, parent, orbit_params=["PB", "PBDOT", "XPBDOT", "T0"]):
        super(OrbitPB, self).__init__("orbitPB", parent, orbit_params)

    def orbits(self,):
        """Orbits using PB, PBDOT, XPBDOT
        """
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        orbits = (
            self.tt0 / PB - 0.5 * (PBDOT + XPBDOT) * (self.tt0 / PB) ** 2
        ).decompose()
        return orbits

    def pbprime(self):
        return self.PB + self.PBDOT * self.tt0

    def pbdot_orbit(self):
        return self.PBDOT

    def d_orbits_d_T0(self):
        """The derivitve of orbits respect to T0
        """
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return ((PBDOT - XPBDOT) * self.tt0 / PB - 1.0) * 2 * np.pi * u.rad / PB

    def d_orbits_d_PB(self):
        """dM/dPB this could be a generic function
        """
        PB = self.PB.to("second")
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return (
            2
            * np.pi
            * u.rad
            * ((PBDOT + XPBDOT) * self.tt0 ** 2 / PB ** 3 - self.tt0 / PB ** 2)
        )

    def d_orbits_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.to("second")
        return -np.pi * u.rad * self.tt0 ** 2 / PB ** 2

    def d_orbits_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.to("second")
        return -np.pi * u.rad * self.tt0 ** 2 / PB ** 2

    def d_pbprime_d_PB(self):
        return np.ones(len(self.tt0)) * u.Unit("")

    def d_pbprime_d_PBDOT(self):
        return self.tt0

    def d_pbprime_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.PBDOT.value)
        return result * u.Unit(self.PBDOT.unit)


class OrbitFBX(orbits):
    def __init__(self, parent, orbit_params=["FB0"]):
        super(OrbitFBX, self).__init__("orbitFBX", parent, orbit_params)
        # add the rest of FBX parameters.
        for k in self.binary_params:
            if re.match(r"FB\d", k) is not None:
                if k not in self.orbit_params:
                    self.orbit_params += [k]

    def orbits(self):
        FBXs = [0 * u.Unit("")]
        ii = 0
        while "FB" + str(ii) in self.orbit_params:
            FBXs.append(getattr(self, "FB" + str(ii)))
            ii += 1
        orbits = taylor_horner(self.tt0, FBXs)
        return orbits.decompose()

    def pbprime(self):
        FBXs = [0 * u.Unit("")]
        ii = 0
        while "FB" + str(ii) in self.orbit_params:
            FBXs.append(getattr(self, "FB" + str(ii)))
            ii += 1
        orbit_freq = taylor_horner_deriv(self.tt0, FBXs, 1)
        return 1.0 / orbit_freq

    def pbdot_orbit(self):
        FBXs = [0 * u.Unit("")]
        ii = 0
        while "FB" + str(ii) in self.orbit_params:
            FBXs.append(getattr(self, "FB" + str(ii)))
            ii += 1
        orbit_freq_dot = taylor_horner_deriv(self.tt0, FBXs, 2)
        return -(self.pbprime() ** 2) * orbit_freq_dot

    def d_orbits_d_par(self, par):
        if re.match(r"FB\d", par) is not None:
            result = self.d_orbits_d_FBX(par)
        else:
            result = super(OrbitFBX, self).d_orbits_d_par(par)
        return result

    def d_orbits_d_FBX(self, FBX):
        par = getattr(self, FBX)
        ii = 0
        FBXs = [0 * u.Unit("")]
        while "FB" + str(ii) in self.orbit_params:
            if "FB" + str(ii) != FBX:
                FBXs.append(0.0 * getattr(self, "FB" + str(ii)).unit)
            else:
                FBXs.append(1.0 * getattr(self, "FB" + str(ii)).unit)
            ii += 1
        d_orbits = taylor_horner(self.tt0, FBXs) / par.unit
        return d_orbits.decompose() * 2 * np.pi * u.rad

    def d_pbprime_d_FBX(self, FBX):
        par = getattr(self, FBX)
        ii = 0
        FBXs = [0 * u.Unit("")]
        while "FB" + str(ii) in self.orbit_params:
            if "FB" + str(ii) != FBX:
                FBXs.append(0.0 * getattr(self, "FB" + str(ii)).unit)
            else:
                FBXs.append(1.0 * getattr(self, "FB" + str(ii)).unit)
            ii += 1
        d_FB = taylor_horner_deriv(self.tt0, FBXs, 1) / par.unit
        return -(self.pbprime() ** 2) * d_FB

    def d_pbprime_d_par(self, par):
        par_obj = getattr(self, par)
        if re.match(r"FB\d", par) is not None:
            result = self.d_pbprime_d_FBX(par)
        else:
            result = np.zeros(len(self.tt0)) * u.second / par_obj.unit
        return result
