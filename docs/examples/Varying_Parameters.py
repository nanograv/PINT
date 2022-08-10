# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # An Introduction to Pulsar Timing and Orbital Variations.

# %% [markdown]
# This notebook can be condsidered an introduction to pulsar timing fits. This notebook uses a set of pulse arrival times, called TOAs (contained in a .tim file); and a compiled list parameters describing the pulsar and the system in which it resides, called a model (contained in a .par file). The model and TOAs file are both included in the PINT test directory, though a modified version of the model will be used for demonstrative purposes.
#
# The focus of this notebook will be to highlight the effects of over/under estimating model parameters on the timing resiuals. The reason is to highlight the pattern of each parameter misestimation.

# %% [markdown]
# ## Modules and Functions

# %% [markdown]
# This first cell imports all the dependencies required to run the notebook.

# %%
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
import pint.toa as toa
from pint.models import get_model
import pint.residuals
import matplotlib.colors
import pint.fitter
from pylab import *
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from copy import deepcopy
import matplotlib.patches as mpatches
from pint.models.timing_model import (
    TimingModel,
    Component,
)


# %% [markdown]
# These next two cells declare the functions used to plot the residuals as a function of decimal year or orbital phase. The decision of which axis to use will be important in spotting these patterns. This is due to certain patterns are only apparent over a yearly timescale while some are only apparent on the timescale of an orbital period.

# %% [markdown]
# We allow the functions to accept a label, the only current purpose of the label argument is to colour code the data based on frequency. We will use this later in the notebook to better highlight frequency dependant effects. We could choose the frequency cut to be anything but the choice of 1000 MHz is an informed choice given the data set we will be using.
#
# This functions can be expanded on to highlight finer structure within the range. In addition if we want to colour code based on observatory, date range, etc. we can modify the "get_freqs" command to "get_obss" or "get_mjds" and passing it a sensible discriminator.

# %%
def plot_resids(t, m, color=None, label=None):
    rs = pint.residuals.Residuals(t, m, track_mode="use_pulse_numbers")
    rorb = rs.toas.get_mjds()
    ba = Time(rorb, format="mjd")
    ba = ba.decimalyear
    ntoas = t.ntoas
    if label == "split_by_freqs":
        color = []
        i = 0
        while i < ntoas:
            if t.get_freqs()[i] >= 1000 * u.MHz:
                color.append("blue")
            elif t.get_freqs()[i] < 1000 * u.MHz:
                color.append("red")
            i = i + 1

    if label == None:
        plt.errorbar(
            ba,
            rs.time_resids.to(u.ms),
            c=color,
            ms=0.75,
            yerr=rs.toas.get_errors().to(u.ms),
            fmt="x",
        )
        plt.title("%s Timing Residuals" % m.PSR.value)
        plt.xlabel("Year")
        plt.ylabel("Residual (ms)")
        plt.grid()

    elif label == "split_by_freqs":
        plt.scatter(
            ba,
            rs.time_resids.to(u.ms),
            c=color,
            s=0.75,
        )
        plt.title("%s Timing Residuals" % m.PSR.value)
        plt.xlabel("Year")
        plt.ylabel("Residual (ms)")
        plt.grid()


# %%
def plot_resids_orbit(t, m, color=None, label=None):
    rs = pint.residuals.Residuals(t, m, track_mode="use_pulse_numbers")
    rorb = rs.model.orbital_phase(rs.toas.get_mjds()) / (
        2 * np.pi * u.rad
    )  # need to be barycentered!!!
    ntoas = t.ntoas
    if label == "split_by_freqs":
        color = []
        i = 0
        while i < ntoas:
            if t.get_freqs()[i] >= 1000 * u.MHz:
                color.append("blue")
            elif t.get_freqs()[i] < 1000 * u.MHz:
                color.append("red")
            i = i + 1

    if label == None:
        plt.errorbar(
            rorb,
            rs.time_resids.to(u.ms),
            c=color,
            ms=0.75,
            yerr=rs.toas.get_errors().to(u.ms),
            fmt="x",
        )
        plt.title("%s Timing Residuals" % m.PSR.value)
        plt.xlabel("Orbital Phase")
        plt.ylabel("Residual (ms)")
        plt.grid()

    if label == "split_by_freqs":
        plt.scatter(
            rorb,
            rs.time_resids.to(u.ms),
            c=color,
            s=0.75,
        )
        plt.title("%s Timing Residuals" % m.PSR.value)
        plt.xlabel("Orbital Phase")
        plt.ylabel("Residual (ms)")
        plt.grid()


# %% [markdown]
# This cell combines the two above functions and plots the residual as a function of decimal year **and** orbital phase. The hexbin plot splits the axis into hexagonal bins with the colour indicating the average residual of the bin. I find it easiest to read these plots from bottom to top going left to right. The patterns visible on an orbital timescale will be seen in the colour of bins running up the y-axis. Similarly patterns visible on a yearly timescale will be apparent running along the x-axis.

# %%
def plot_hexbin(t, m, color=None):
    rs = pint.residuals.Residuals(t, m, track_mode="use_pulse_numbers")
    rorb = rs.toas.get_mjds()
    ba = Time(rorb, format="mjd")
    ba = ba.decimalyear
    plt.hexbin(
        ba,
        rs.model.orbital_phase(rs.toas.get_mjds()) / (2 * np.pi * u.rad),
        rs.time_resids.to_value(u.ms),
        cmap="viridis",
        gridsize=(12, 4),
        mincnt=0,
        color="grey",
        edgecolor="white",
    )

    plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
    plt.xlabel("Year")
    plt.ylabel("Orbital Phase")
    plt.gca().set_facecolor("grey")
    cbar = plt.colorbar()
    cbar.set_label("Residual (ms)", rotation=90)
    plt.grid()


# %% [markdown]
# Here is the function that calls and formats the plotting functions listed above.

# %%
def plot_all(t, m, color=None, label=None):
    fig = plt.figure()
    fig.tight_layout()
    fig.set_figheight(3)
    fig.set_figwidth(20)
    subplot(1, 3, 1)
    plot_resids(t, m, color, label)
    if label == "split_by_freqs":
        red_patch = mpatches.Patch(color="red", label="~450 MHz")
        blue_patch = mpatches.Patch(color="blue", label="~1400 MHz")
        plt.legend(handles=[red_patch, blue_patch], loc="right")
    subplot(1, 3, 2)
    plot_resids_orbit(t, m, color, label)
    if label == "split_by_freqs":
        red_patch = mpatches.Patch(color="red", label="~450 MHz")
        blue_patch = mpatches.Patch(color="blue", label="~1400 MHz")
        plt.legend(handles=[red_patch, blue_patch], loc="right")
    subplot(1, 3, 3)
    plot_hexbin(t, m, color)

    fig.subplots_adjust(wspace=0.24)


# %% [markdown]
# ## Loading in & Fitting Data

# %% [markdown]
# Now all the functions are declared, let's load some data. The model (m) and TOAs (t) files are loaded in. From the .tim files the time of arrivals are stored in object t. From the .par file, parameters describing the pulsar's intrinsic/orbital properties are stored in object m.
#
# A point to note, the .par file contains a flag on each parameter. This flag determines whether a parameter is free or frozen. If a parameter is free, the parameter will be allowed to be modified by a fitting function. The opposite is true for the frozen parameters. The TOAs object stores the observed time of pulse arrival whereas a model describes when the pulse is expected to arrive. The residual is the time difference between the two.
#
# A plot of the residuals is also performed to see how well the model describes the TOAs.  Ideally, if the model perfectly describes the system, the residuals would be zero.
#
# We also use the argument "usepickle=True" when loading in the TOAS. This pickles the loaded TOAs object which acts as a transparent cache of TOAs. Meaning if we need to reload the TOAs from the file, firstly the code will check if there are any changes in the .tim file. If there are none it will load the TOAs object from the pickle file and remove the need to recompute the TOAs object, saving computation time.

# %%
m = pint.models.get_model(
    "/data/c0048827/pulsar/pars/B1855+09_NANOGrav_dfg+12_DMX-Copy1.par"
)
t = toa.get_TOAs(
    "/home/c0048827/pulsars/PINT/tests/datafile/B1855+09_NANOGrav_dfg+12.tim"
)  # ,usepickle=True,ephem = 'de421')
t.compute_pulse_numbers(m)
plot_resids(t, m)

# %% [markdown]
# Unfortunately as can be seen above, we are not in the ideal world where the residuals nicely lie around 0. Luckily there is a parameter in the model that will help us account for this. To probe the model to see which parameters are defined, we can open up the contents of the model with the following command.
#
# The model is broken into separate columns:
# -Column 1: Parameter Name,
# -Column 2: Parameter Value,
# -Column 3: Free or Frozen Parameter (0 representing a frozen parameter, 1 representing a free parameter),
# -Column 4: Parameter Uncertainty.
#
# The definition of all parameter names can be found in the Tempo2 user manual.

# %%
print(m.as_parfile())

# %% [markdown]
# The parameter we're looking to adjust first is the "JUMP" parameter. This parameter accounts for a constant offset between TOAs. If we look at the .tim file and check the flags, optional additional information of the measurement, we can see the observations were taken with two different front ends (-fe). This relates to the observing frequency but we will touch on the effect of observing frequency later. For now a fitted "JUMP" parameter will soak up this offset and should produce a cleaner set of residuals.
#
# (Other optional flags in this example: 'be' 'B' 'bw' 'tobs' 'pta' 'proc' 'chanid'. These can be inserted into the cell below in place of the string used in "flag").

# %%
flag = "tobs"
set_flag = set(t.get_flag_value(flag)[0])
print(
    "Unique flag values associated with -%s used in observations: %s" % (flag, set_flag)
)

# %% [markdown]
# So now we have justified to ourselves why we need this "JUMP" parameter, let us allow this variable to be free, hence the model can tweak this parameter to find the best fit. We'll do this by, making a copy of the .par file such that any changes made post-fit will be stored in a separate object. Therefore we will have an original copy if it is required later. And then looping through all the parameters in the file setting the free/frozen states appropriately (i.e. only allowing JUMP to be free). For reference, the jump parameter requires a flag and a value to make the selection. The flag for this JUMP parameter has been set to '-fe' within the model, hence residuals will be adjusted based on the value of the -fe flag.
#
# If you want to add your own components to the model you can (see PINT tutorial on "Building a Model"). This can mean you can implement multiple jumps acting on a single flag if there are more than two unique values. For example if the data comes from n different telescopes, (n-1) jumps can account for systematics of the telescopes.

# %%
m2 = pint.models.get_model(
    "/data/c0048827/pulsar/pars/B1855+09_NANOGrav_dfg+12_DMX-Copy1.par"
)
m2.free_params = [
    "JUMP1"
]  # currently not functional line but is the new way of setting free/frozen parameters

# %% [markdown]
# Now using a Weighted Least Square fitting method we pass the .tim files and the new copy of the model with a single flexible parameter. This post-fit model is stored as object f. The syntax of this process is: pass the TOA and model objects to the fitter, then perfom the least squares fit (using the 'f.fit_toas()' line).

# %%
f = pint.fitter.WLSFitter(t, m2)
f.fit_toas()

# %% [markdown]
# We can print the summary of the fitted model. It provides information regarding the fit, a comparison between the pre- and post-fit parameters and weighted root mean square values are provided.

# %%
f.print_summary()

# %% [markdown]
# Now we can take a look at the residuals of this newly fitted model. Plotting the residuals in the same way as before, but using the new model denoted f.model. We expect the residuals to be situated around 0 a lot more closely.

# %%
plot_all(t, f.model)

# %% [markdown]
# The residuals look much better now. Introducing a constant offset in JUMP seems to account for the the excess residuals. This is the general form we aim to achieve. For this examples sake, at a glance, the majority of residuals lie within uncertainty of 0.

# %% [markdown]
# # Epochs of Determination: POSEPOCH, T0,PEPOCH

# %% [markdown]
# ## Time of Position Determination: POSEPOCH

# %% [markdown]
# The epoch of position determination, POSEPOCH, is an MJD value at which the right ascension and delcination are measured. This is important as we need to be able to track the pulsar's motion to be able to accurately determine the residual of each TOA measurement. From the model, POSEPOCH is defined outside of our data range. This difference can be calculated in the following cell:

# %%
diff_value = f.model.START.quantity - f.model.POSEPOCH.quantity
print("Time difference between POSEPOCH and START: %s days" % diff_value)

# %% [markdown]
# If we want to move POSEPOCH to fit within our data range, we can use the "change_posepoch" method. This operation translates POSEPOCH and scales the position parameters associated. This is simply a reparameterisation of POSEPOCH so will have no effect on the observed residuals. If we run the following cell, we should reobtain the plot obtained when using the fitted model.

# %%
m3 = deepcopy(f.model)
m3.change_posepoch(
    f.model.START.value + (f.model.FINISH.value - f.model.START.value) / 2
)
plot_all(t, m3)

# %% [markdown]
# However if the initial positions are improperly handled it gives rise to excess residuals, with a sinusoidal pattern visible on a yearly timescale. This effect can be simulated by taking the fitted model and simply shifting the POSEPOCH without updating the position.

# %%
m3 = deepcopy(f.model)
m3.POSEPOCH.value = m3.POSEPOCH.value + 0.5 * diff_value.value
plot_all(t, m3)

# %% [markdown]
# If we do not give the model a POSEPOCH value, the model automatically assigns POSEPOCH the value of PEPOCH which is discussed later in this section.

# %% [markdown]
# ## Time of Periastron Advance: T0

# %% [markdown]
# The epoch of periastron, T0, is an MJD value at which the pulsar is determined to be closest to us along our line of sight. Looking at the model, T0 is determined outside the data range. Unfortunately there is no current functionality to change T0 without introducing a position error but there may be a way to simulate this by taking into account the orbital period, PB. Realistically there may be some orbital shirnkage, PBdot, but has been excluded from the model so is not considered here.

# %%
diff_value = f.model.START.value - f.model.T0.value
print("Time difference between T0 and START: %s days" % diff_value)


# %% [markdown]
# Now we have established the time difference between T0 and the start of our data range we should be able to write an equation that accounts for the orbital frequency. To do this, take the integer number of orbits occuring between the T0 and START, then multiply the integer number of orbits by the orbital period. This determines the day closest to START at which the orbital phase is the same as when T0 was originally defined. Hence we should reobtain the residuals plotted in the fitted model.

# %%
m3 = deepcopy(f.model)
scale_orbit = m3.PB.value
scale_orbit = diff_value // scale_orbit
x = m3.T0.value + m3.PB.value * scale_orbit
m3.T0.value = x
plot_all(t, m3)

# %% [markdown]
# If we do not correclty scale T0 we can introduce a position error as we will be misestimating where the closest approach lies. This will give rise to a sinusoidal pattern in the residuals on the timescale of an orbit. To demonstrate this we will set the T0 to be at the scaled time of periastron, then add value that is not a multiple of the orbital period to ensure an incorrect shift has been performed.

# %%
m3 = deepcopy(f.model)
m3.T0.value = x + 100
plot_all(t, m3)

# %% [markdown]
# ## Time of Period Determination: PEPOCH

# %% [markdown]
# The epoch of period determination, PEPOCH, is a MJD value that tells the model when the spin-frequency of the pulsar is determined. From here the spin-frequency, amongst other time varying parameters are scaled to the correct values for the data range. From the model we can see PEPOCH is determined far outside our data range. If we look at the time difference between when PEPOCH is evaluated and when we have data for:

# %%
diff_value = f.model.START.quantity - f.model.PEPOCH.quantity
print("Time difference between PEPOCH and START: %s days" % diff_value)

# %% [markdown]
# If we want to move PEPOCH to lie within our data range, we can use the "change_pepoch" method. Using this method we should reobtain the residuals found in the fitted model plot.

# %%
m3 = deepcopy(f.model)
m3.change_pepoch(f.model.START.value + (f.model.FINISH.value - f.model.START.value) / 2)
plot_all(t, m3)

# %% [markdown]
# Now that PEPOCH is determined within the data range, a good exercise is to check that frequency is being correctly adjusted. Taking the time difference between the old and new PEPOCH's, then taking the cumulative spin-down effects over the timespan. We should see the change_pepoch() method is scaling the spin-frequency by roughly the same amount.

# %%
print("New PEPOCH value: %s" % m3.PEPOCH.value)

scale_by = (m3.PEPOCH.value - f.model.PEPOCH.value) * (u.d)
spin_down = f.model.F1.quantity
spin_down = spin_down.to(u.Hz / u.d)
scale_fac = spin_down * scale_by
diff = (m3.F0.value - f.model.F0.value) * u.Hz
print("Frequency should approximately change by: %s" % scale_fac)
print("Actual frequency difference between original and modified model: %s" % diff)

# %% [markdown]
# We can see these two values are similar but not exactly the same. This is most likely due to rounding errors being propagated forward in the calculation of the cumulative spin-frequency change. If we want, we can perform another fit to further constrain the value of the spin-frequency.
#
# A good question to ask is "How would we know if a second fit is required?". The next section looks at how a misestimation in the spin and spin-down frequency affects the observed residuals.

# %% [markdown]
# # Effects of changing intrinsic pulsar parameters: F0 & F1

# %% [markdown]
# ## Spin-Frequency: F0

# %% [markdown]
# Let's start with by copying the post-fit model but lets change the spin-frequency (F0) of the pulsar. Since the estimation of the spin-down (F1) and other orbital parameters are correctly fitted the spin-frequency will evolve correctly but from an incorrect starting value.
#
# A way of framing this problem is thinking of a wheel turning. In the ideal case where we know the rotational speed and its associated time derivative we can accurately describe when we think a full rotation has occured. If the initial rotational speed is independantly poorly estimated our model that describes when one full rotation occurs will not be coherent with the wheel's actual motion. The time difference between when the model predicts a rotation and when the rotation actually occurs can be seen as an excess residual.

# %% [markdown]
# In this example notebook we will add these arbitrary errors to the fitted parameters using the attached uncertainty. How much of an offset we add is exaggerated for purposes of demonstration, so in real data these effects can be a lot more subtle.

# %%
m3 = deepcopy(f.model)
m3.F0.value = m2.F0.value + 400 * m2.F0.uncertainty_value
plot_all(t, m3)

# %% [markdown]
# With this linear increase we see the residuals now lie beyond the rotaional period of the pulsar. If we did not compute and lock in the pulse numbers when loading the TOAs, the model have no idea of which pulse it was looking at as the pulsar could have turned multiple times before the next pulse arrival. Failure to lock in the pulse numbers would result in a phase wrapping effect as the model tries to minimise the residual, irrespective of the pulse number. If we know the next pulse arrival is associated with the next pulse number but the residual is greater than what the rotation frequency predicts, we run into the problem of attributing incorrect pulse numbers.

# %% [markdown]
# ## Spin-Down Frequency: F1

# %% [markdown]
# So what if we correctly estimate the spin-frequency but not the spin-down? Going back to the wheel rotating analogy, if we correctly predict the rotational speed but not its time derivative, we expect a non-linear trend in the time between a predicted rotation and the actual rotation. The non-linearity comes from the the spin-frequency being incorrectly estimated at the end of each rotation due to a poor constraint on the spin-down.

# %% [markdown]
# Hence we take a copy of the fitted model, add the offset in the spin-down measurement, and then allow the model to fit for the spin-frequency. We fit for F0 such that our spin-frequency measurement is not contaminating the observed residuals with a poor estimation.

# %%
model = deepcopy(f.model)
model.F1.value = model.F1.value + 400 * model.F1.uncertainty_value
model.free_params = ["F0"]
fit_mod = pint.fitter.WLSFitter(t, model)
fit_mod.fit_toas()
plot_all(t, fit_mod.model)

# %% [markdown]
# # Changing Orbital Parameters: PB, A1, ECC

# %% [markdown]
# ## Orbital Period: PB

# %% [markdown]
# Now we will trying adjusting parameters relating to the orbit of the pulsar. We will start by adding an offset to the orbital period. Since the position of the pulsar is determined by the orbital period, and the model predicts the pulse arrival time from the pulsar position; we expect there to be a pattern in the residuals on the timescale of an orbital period.
#
# Over the course of an orbit there are times the model predicts the pulsar is further or closer than what is actually observed. This gives reason to believe the pattern that arises will be a sinusoid.

# %% [markdown]
# Using a simalar approach as before, we take a copy of the fitted model and then add our offset. For reference we also plot the unchanged fitted model in red to more clearly demonstrate the offset seen in blue.

# %%
m3 = deepcopy(f.model)
m3.PB.value = m2.PB.value + 400 * m2.PB.uncertainty_value
plot_all(t, m3)

# %% [markdown]
# ## Orbital Amplitude: A1

# %% [markdown]
# Moving onto orbital amplitude. Changing this value means the model calculates the position of the pulsar to be much different to the actual position of the pulsar. This means the position when a pulse is emitted will be incorrect. This should, once again, be observed as a sinusoidal pattern in the residuals when plotted against orbital phase. The original copy of the model is also plotted for comparison's sake and will generally be shown in red.

# %%
m3 = deepcopy(f.model)
m3.A1.value = m2.A1.value + 400 * m2.A1.uncertainty_value
plot_all(t, m3)

# %% [markdown]
# ## Orbital Eccentricity: ECC

# %% [markdown]
# As you can probably guess now, adding an offset into the orbital parameters will produce a position error in the residual calculation. For completeness sake we can adjust the eccentricity of the orbit, meaning the pulsar will follow a different path to what is actually observed. Ergo this will give rise to a sinusoidal pattern in the residual plot on the timescale of an orbit.

# %%
m3 = deepcopy(f.model)
m3.ECC.value = m2.ECC.value + 400 * m2.ECC.uncertainty_value
plot_all(t, m3)

# %% [markdown]
# ## Another Interesting Parameter: DM

# %% [markdown]
# As we saw in the initial plot, there exists a divide in the data. Since we know the data comes from two different receivers on the same telescope there is reason to believe the observing frequency is different.
#
# We can scroll through the .tim file and physically look at the observing frequency column. This is understandably an inefficient method of checking especially when we can be working with thousands of TOAs. Instead we can fetch the frequency from each TOA and plot it in a histogram with the following cell:

# %%
plt.hist(t.get_freqs().to_value(u.MHz))
plt.show()

# %% [markdown]
# We can clearly see the observations are split across two discrete frequency bands. Hence we need to account for frequency dependent effects. The frequency dependant effect we can parameterise is the dispersion measure. The interstellar medium acts as a translucent screen, this acts on all observed pulses but to different extents depending on frequency.

# %% [markdown]
# So let's proceed by, once again, taking a copy of the original model and offsetting the dispersion measure. We cannot use the "uncertainty_value" since the model does not have an uncertainty attached to the dispersion value. Hence a relatively large arbitrary offset has been used instead.

# %% [markdown]
# The next block invokes the all the plotting functions with the label "split_by_freqs". The output of this cell we expect to reobtain the bands we saw previously in the unfitted data.

# %%
m3 = deepcopy(f.model)
m3.DM.value = m2.DM.value + 0.00001 * 400
plot_all(t, m3, label="split_by_freqs")

# %% [markdown]
# ________________________________________________________________
