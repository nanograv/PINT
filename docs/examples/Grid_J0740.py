# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grid PSR J0740

# %% [markdown]
# Try to reproduce Figure 6 from [Fonseca et al. 2021, ApJ, 915, L12](https://ui.adsabs.harvard.edu/abs/2021ApJ...915L..12F/abstract)
#
# We have downloaded wideband files from [Zenodo](https://zenodo.org/record/4773599#.YWdRDC-cbUI):
# * [J0740+6620.FCP+21.wb.tim](https://zenodo.org/record/4773599/files/J0740%2B6620.FCP%2B21.wb.tim?download=1)
# * [J0740+6620.FCP+21.wb.DMX3.0.par](https://zenodo.org/record/4773599/files/J0740%2B6620.FCP%2B21.wb.DMX3.0.par?download=1)

# %%
# imports
from astropy import units as u, constants as c
import numpy as np

import pint.config
import pint.gridutils
import pint.models.parameter as param
import pint.residuals
import pint.fitter
import pint.derived_quantities
from pint.models.model_builder import get_model, get_model_and_toas
from pint.toa import get_TOAs

import concurrent.futures
import datetime
import multiprocessing


# %% [markdown]
# We want to set up a grid to cover:
# * distance $d$
# * $\cos i$
# * companion mass $M_c$
#
# However, the intrinsic parameters are:
# * parallax $\varpi$
# * $\sin i$
# * companion mass $M2$
# * orbital decay rate $\dot P_B$
#
# These depend on the grid parameters in various ways, some easy, some complicated.  So:
# * $d=1/\varpi$
# * $M_c = M2$
# * $\sin i = \sqrt{1-\cos^2 i}$
#
# What about $\dot P_B$?  That is harder.  We need to compute the expected $\dot P_B$ from GR, and compare it with the measured value corrected for kinematic effects (Shklovskii effect, Galactic acceleration).  And we also know that $\dot P_B(GR)$ depends on both $M_c$ and pulsar mass $M_p$.  But we can connect $M_c$ and $\cos i$ to get $M_p$, and then $M_p$ and $M_c$ to get $\dot P_B(GR)$.  We compute the various kinematic effects using the same analytic versions in Fonseca et al.
#
# We will make an object to bundle all of these computations together to run in each iteration.

# %%
class constrained_gridder:
    def __init__(
        self,
        fitter,
        R0=8.178 * u.kpc,
        Theta0=236.9 * u.km / u.s,
        rho0=1e-2 * u.Msun / u.pc ** 3,
        Sigma0=46 * u.Msun / u.pc ** 2,
        z0=0.18 * u.kpc,
    ):
        """Compute the chi^2 given various input parameters that are then compared with the model
        
        Parameters
        ----------
        fitter : pint.fitter.Fitter
        R0 : astropy.units.Quantity, optional
            Galactic radius of the Sun
        Theta0 : astropy.units.Quantity, optional
            Galactic rotation speed for the Sun      
        rho0 : astropy.units.Quantity, optional
            local mass density of the Galactic disk
        Sigma0 : astropy.units.Quantity, optional
            local mass surface density of the Galactic disk
        z0 : astropy.units.Quantity, optional
            local scale-height of the Galactic disk
        """
        self.fitter = fitter
        self.e = np.sqrt(
            fitter.model.EPS1.quantity ** 2 + fitter.model.EPS2.quantity ** 2
        )
        self.coords_galactic = fitter.model.coords_as_GAL()
        self.mu = np.sqrt(
            self.coords_galactic.pm_l_cosb ** 2 + self.coords_galactic.pm_b ** 2
        )
        self.R0 = R0
        self.Theta0 = Theta0
        self.rho0 = rho0
        self.Sigma0 = Sigma0
        self.z0 = z0

    def Mp(self, Mc, cosi, d):
        """Pulsar mass as a function of companion mass and inclination
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance (not used, just for consistency among function calls)        
            
        Returns
        -------
        astropy.units.Quantity
        """
        i = np.arccos(cosi) * u.rad
        return pint.derived_quantities.pulsar_mass(
            self.fitter.model.PB.quantity, self.fitter.model.A1.quantity, Mc, i
        )

    def PBDOT_GR(self, Mc, cosi, d):
        """Orbital decay due to GR, as a function of companion mass and inclination
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance (not used, just for consistency among function calls)
        
        Returns
        -------
        astropy.units.Quantity
        """
        Mp = self.Mp(Mc, cosi, d)
        return pint.derived_quantities.pbdot(
            Mp, Mc, self.fitter.model.PB.quantity, self.e
        ).to(u.d / u.s)

    def PBDOT_mu(self, Mc, cosi, d):
        """Apparent orbital decay due to proper motion (Shklovskii)
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance
            
        Returns
        -------
        astropy.units.Quantity
        
        Notes
        -----
        Proper motion is stored in the ``fitter`` object
        """
        return ((self.mu ** 2 * d / c.c) * self.fitter.model.PB.quantity).to(
            u.d / u.s, equivalencies=u.dimensionless_angles()
        )

    def PBDOT_DR(self, Mc, cosi, d):
        """Apparent orbital decay due to Galactic acceleration in the plane
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance
            
        Returns
        -------
        astropy.units.Quantity
        """
        # this is beta in Nice & Taylor (1995), after Eqn. 5
        kappa = (d / self.R0) * np.cos(self.coords_galactic.b) - np.cos(
            self.coords_galactic.l
        )
        # and Nice & Talor (1995), Eqn. 5
        return (
            -self.fitter.model.PB.quantity
            * (self.Theta0 ** 2 / c.c / self.R0)
            * (
                np.cos(self.coords_galactic.l)
                + kappa ** 2 / (np.sin(self.coords_galactic.l) ** 2 + kappa ** 2)
            )
        ).to(u.d / u.s)

    def PBDOT_z(self, Mc, cosi, d):
        """Apparent orbital decay due to Galactic acceleration in the vertical direction
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance
            
        Returns
        -------
        astropy.units.Quantity
        """
        z = d * np.sin(self.coords_galactic.b)
        # This was:
        # Nice & Taylor (1995, ApJ, 441, 429), Eqn. 4
        # which used values from:
        # Kuijken and Gilmore (1989, MNRAS, 239, 605)
        # we will recast it in terms of the original measured quantities
        # so that they can be re-evaluated if needed
        # (they had be written in funny units)
        # it differs from the Nice & Taylor value by a few percent: maybe some
        # difference in constant conversions
        az = (
            -2
            * np.pi
            * c.G
            * (self.Sigma0 * z / np.sqrt(z ** 2 + self.z0 ** 2) + 2 * self.rho0 * z)
        )
        return (
            (az * self.fitter.model.PB.quantity / c.c) * np.sin(self.coords_galactic.b)
        ).to(u.s / u.s)

    def PBDOT(self, Mc, cosi, d):
        """Total orbital decay due to GR, Shklovskii, Galactic acceleration
            
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass
        cosi : float
            cosine of inclination
        d : astropy.units.Quantity
            distance
            
        Returns
        -------
        astropy.units.Quantity
        """
        return (
            self.PBDOT_GR(Mc, cosi, d)
            + self.PBDOT_DR(Mc, cosi, d)
            + self.PBDOT_mu(Mc, cosi, d)
            + self.PBDOT_z(Mc, cosi, d)
        )

    def parallax(self, Mc, cosi, d):
        """parallax from distance
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass (not used, just for consistency among function calls)
        cosi : float
            cosine of inclination (not used, just for consistency among function calls)
        d : astropy.units.Quantity
            distance
            
        Returns
        -------
        astropy.units.Quantity
        """
        return d.to(u.mas, equivalencies=u.parallax())

    def sini(self, Mc, cosi, d):
        """sine of inclination from cos of inclination
        
        Parameters
        ----------
        Mc : astropy.units.Quantity
            Companion mass (not used, just for consistency among function calls)
        cosi : float
            cosine of inclination 
        d : astropy.units.Quantity
            distance (not used, just for consistency among function calls)
            
        Returns
        -------
        astropy.units.Quantity
        """
        return np.sqrt(1 - cosi ** 2)


# %%
print(f"Starting at {datetime.datetime.now().isoformat()}")

# %%
# Load in a basic dataset, construct a fitter, do an initial fit
parfile = pint.config.examplefile("J0740+6620.FCP+21.wb.DMX3.0.par")
timfile = pint.config.examplefile("J0740+6620.FCP+21.wb.tim")
m, t = get_model_and_toas(parfile, timfile)

f = pint.fitter.WidebandDownhillFitter(t, m)

f.fit_toas()
print(
    f"Computed best-fit chi2={f.resids.chi2} at  {datetime.datetime.now().isoformat()}"
)

# %%
# make the constrained gridder object out of the fitter
cg = constrained_gridder(f)

# %%
# set up the grid.  The number of grid points will depend on your available CPU/time
distance = np.linspace(0.5, 2, 1) * u.kpc
cosi = np.linspace(0.03, 0.06, 1)
Mc = np.linspace(0.23, 0.28, 1) * u.Msun
outfile = "J0740_PINT_grid.npz"

# %%
# may be needed for some executables
# multiprocessing.freeze_support()

# %%
# use this for SLURM
# with cfut.SlurmExecutor(debug=True,keep_logs=False,additional_setup_lines=slurm_options) as executor:

# use this for MPI
# with MPIPoolExecutor() as executor:

# default (could also just not supply any executor)
with concurrent.futures.ProcessPoolExecutor() as executor:

    # the input grid is (Mc, cosi, distance)
    # that gets transformed to the intrinsic measured quantities (M2, SINI, PX, PBDOT)
    # via
    # M2 = Mc
    # sini = cg.sini(Mc, cosi, d)
    # parallax = cg.parallax(Mc, cosi, d)
    # PBDOT = cg.PBDOT(Mc, cosi, d)
    # note that all of those have the same input parameters,
    # even if they aren't needed for a specific calculation
    chi2, params = pint.gridutils.grid_chisq_derived(
        f,
        ("M2", "SINI", "PX", "PBDOT"),
        (lambda Mc, cosi, d: Mc, cg.sini, cg.parallax, cg.PBDOT),
        (Mc, cosi, distance),
        executor=executor,
        printprogress=False,
    )
    # save this if desired
    # np.savez(
    #    outfile,
    #    distance=distance,
    #    cosi=cosi,
    #    Mc=Mc,
    #    chi2=chi2,
    #    params=params,
    # )
    # print(f"Wrote {outfile}")
    print(
        f"Finished {str(chi2.shape)}={len(chi2.flatten())} points at {datetime.datetime.now().isoformat()}"
    )


# %%
