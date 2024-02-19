# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to compose a timing model component
#
# PINT's design makes it easy to add a new, custom timing model component to meet specific needs. This notebook demonstrates how to write your own timing model component with the minimal requirements so that PINT can recognize and use it in fits. Here, we implement a new spindown class, `PeriodSpindown`, that is parameterized by `P0`, `P1`, instead of the built-in `Spindown` model component, which uses `F0`, `F1`.
#
# ## Building the timing model component from scratch
#
# This example notebook includes the following contents
# * Defining a timing model component class
#   * Necessary parts
#   * Conventions
# * Use it with the `TimingModel` class
#   * Add the new component to the `TimingModel` class
#   * Use the functions in the `TimingModel` class to interact with the new component.
#
# We will build a simple model component, pulsar spindown model with spin period as parameters, instead of spin frequency.

# %% [markdown]
# ## Import the necessary modules

# %%
# PINT uses astropy units in the internal calculation and is highly recommended for a new component
import astropy.units as u

# Import the component classes.
from pint.models.spindown import SpindownBase
import pint.models.parameter as p
import pint.config
import pint.logging

# setup the logging
pint.logging.setup(level="INFO")


# %% [markdown]
# ## Define the timing model class
#
# A timing model component should be an inheritance/subclass of `pint.models.timing_model.Component`. PINT also pre-defines three component subclasses for the most used type of components and they have different attribute and functions (see: https://nanograv-pint.readthedocs.io/en/latest/api/pint.models.timing_model.html):
# * DelayComponent for delay type of models.
# * PhaseComponent for phase type of models.
# * NoiseComponent for noise type of models.
#
# Here since we are making a spin-down model, we will use the `PhaseComponent`.

# %% [markdown]
# ### Required parts
# * Model parameters, generally defined as `PINT.models.parameter.Parameter` class or its subclasses. (see https://nanograv-pint.readthedocs.io/en/latest/api/pint.models.parameter.html)
# * Model functions, defined as methods in the component, including:
#     * .setup(), for setting up the component(e.g., registering the derivatives).
#     * .validate(), for checking if the parameters have the correct inputs.
#     * Modeled quantity functions.
#     * The derivative of modeled quantities.
#     * Other support functions.

# %% [markdown]
# ### Conventions
#
# To make a component work as a part of a timing model, it has to follow the following rules to interface the `TimingModel` class. Using the analog of a circuit board, the `TimingModel` object is the mother board, and the `Component` objects are the electronic components(e.g., resistors and transistors); and the following rules are the pins of a component.
#
# * Set the class attribute `.register` to be True so that the component is in the searching space of model builder
# * Add the method of final result in the designated list, so the `TimingModel`'s collecting function(e.g., total delay or total phase) can collect the result. Here are the designated list for the most common component type:
#   * DelayComponent: .delay_funcs_component
#   * PhaseComponent: .phase_funcs_component
#   * NoiseComponent: .
#     * `.basis_funcs`
#     * `.covariance_matrix_funcs`
#     * `.scaled_toa_sigma_funcs`
#     * `.scaled_dm_sigma_funcs`
#     * `.dm_covariance_matrix_funcs_component`
#
# * Register the analytical derivative functions using the `.register_deriv_funcs(derivative function, parameter name)` if any.
# * If one wants to access the attribute in the parent `TimingModel` class or from other components, please use `._parent` attribute which is a linker to the `TimingModel` class and other components.


# %%
class PeriodSpindown(SpindownBase):
    """This is an example model component of pulsar spindown but parametrized as period."""

    register = True  # Flags for the model builder to find this component.

    # define the init function.
    # Most components do not have a parameter for input.
    def __init__(self):
        # Get the attributes that initialized in the parent class
        super().__init__()
        # Add parameters using the add_params in the TimingModel
        # Add spin period as parameter
        self.add_param(
            p.floatParameter(
                name="P0",
                value=None,
                units=u.s,
                description="Spin period",
                longdouble=True,
            )
        )
        # Add spin period derivative P1. Since it is not all rquired, we are setting the
        # default value to 0.0
        self.add_param(
            p.floatParameter(
                name="P1",
                value=0.0,
                units=u.s / u.s,
                description="Spin period derivative",
                longdouble=True,
            )
        )
        # Add reference epoch time.
        self.add_param(
            p.MJDParameter(
                name="PEPOCH_P0",
                description="Reference epoch for spin-down",
                time_scale="tdb",
            )
        )
        # Add spindown phase model function to phase functions
        self.phase_funcs_component += [self.spindown_phase_period]
        # Add the d_phase_d_delay derivative to the list
        self.phase_derivs_wrt_delay += [self.d_spindown_phase_period_d_delay]

    def setup(self):
        """Setup the model. Register the derivative functions"""
        super().setup()  # This will run the setup in the Component class.
        # The following lines are resgistering the derivative functions to the timingmodel.
        self.register_deriv_funcs(self.d_phase_d_P0, "P0")
        self.register_deriv_funcs(self.d_phase_d_P1, "P1")

    def validate(self):
        """Check the parameter value."""
        super().validate()  # This will run the .validate() in the component class
        # Check required parameters, since P1 is not required, we are not checking it here
        for param in ["P0"]:
            if getattr(self, param) is None:
                raise ValueError(f"Spindown period model needs {param}")

    # One can always setup properties for updating attributes automatically.
    @property
    def F0(self):
        # We return F0 as parameter here since the other place of PINT code use F0
        # in the format of PINT parameter.
        return p.floatParameter(
            name="F0",
            value=1.0 / self.P0.quantity,
            units="Hz",
            description="Spin-frequency",
            long_double=True,
        )

    # Defining the derivatives. In the PINT, a common format of derivative naming is
    # d_xxx_d_xxxx
    @property
    def d_F0_d_P0(self):
        return -1.0 / self.P0.quantity**2

    @property
    def F1(self):
        return p.floatParameter(
            name="F1",
            value=self.d_F0_d_P0 * self.P1.quantity,
            units=u.Hz / u.s,
            description="Spin down frequency",
            long_double=True,
        )

    @property
    def d_F1_d_P0(self):
        return self.P1.quantity * 2.0 / self.P0.quantity**3

    @property
    def d_F1_d_P1(self):
        return self.d_F0_d_P0

    def get_dt(self, toas, delay):
        """dt from the toas to the reference time."""
        # toas.table['tdbld'] stores the tdb time in longdouble.
        return (toas.table["tdbld"] - self.PEPOCH_P0.value) * u.day - delay

    # Defining the phase function, which is added to the self.phase_funcs_component
    def spindown_phase_period(self, toas, delay):
        """Spindown phase using P0 and P1"""
        dt = self.get_dt(toas, delay)
        return self.F0.quantity * dt + 0.5 * self.F1.quantity * dt**2

    def d_spindown_phase_period_d_delay(self, toas, delay):
        """This is part of the derivative chain for the parameters in the delay term."""
        dt = self.get_dt(toas, delay)
        return -(self.F0.quantity + dt * self.F1.quantity)

    def d_phase_d_P0(self, toas, param, delay):
        dt = self.get_dt(toas, delay)
        return self.d_F0_d_P0 * dt + 0.5 * self.d_F1_d_P0 * dt**2

    def d_phase_d_P1(self, toas, param, delay):
        dt = self.get_dt(toas, delay)
        return 0.5 * self.d_F1_d_P1 * dt**2


# %% [markdown]
# ## Apply the new component to the `TimingModel`
#
# Let us use this new model component in our example pulsar "NGC6440E", which has `F0` and `F1`. Instead, we will use the model component above. The following `.par` file string if converted from the `NGC6440E.par` with `P0` and `P1` instead of `F0`, `F1`.

# %%
par_string = """
             PSR              1748-2021E
             RAJ       17:48:52.75  1 0.05
             DECJ      -20:21:29.0  1 0.4
             P0        0.016264003404474613 1 0
             P1        3.123955D-19 1 0
             PEPOCH_P0     53750.000000
             POSEPOCH      53750.000000
             DM              223.9  1 0.3
             SOLARN0               0.00
             EPHEM               DE421
             UNITS               TDB
             TIMEEPH             FB90
             CORRECT_TROPOSPHERE N
             PLANET_SHAPIRO      N
             DILATEFREQ          N
             TZRMJD  53801.38605120074849
             TZRFRQ            1949.609
             TZRSITE                  1
             """

# %%
import io
from pint.models import get_model

# %% [markdown]
# ### Load the timing model with new parameterization.

# %%
model = get_model(
    io.StringIO(par_string)
)  # PINT can take a string IO for inputing the par file

# %% [markdown]
# #### Check if the component is loaded into the timing model and make sure there is no built-in spindown model.

# %%
print(model.components["PeriodSpindown"])
print(
    "Is the built-in spin-down model in the timing model: ",
    "Spindown" in model.components.keys(),
)
print("Is 'P0' in the timing model: ", "P0" in model.params)
print("Is 'P1' in the timing model: ", "P1" in model.params)
print("Is 'F0' in the timing model: ", "F0" in model.params)
print("Is 'F1' in the timing model: ", "F1" in model.params)

# %% [markdown]
# ### Load TOAs and prepare for fitting

# %%
from pint.fitter import WLSFitter
from pint.toa import get_TOAs

# %%
toas = get_TOAs(pint.config.examplefile("NGC6440E.tim"), ephem="DE421")
f = WLSFitter(toas, model)

# %% [markdown]
# ### Plot the residuals

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ### Plot the prefit residuals.

# %%
plt.errorbar(
    toas.get_mjds().value,
    f.resids_init.time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt=".",
)
plt.title(f"{model.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()

# %% [markdown]
# ### Fit the TOAs using `P0` and `P1`

# %%
f.fit_toas()

# %% [markdown]
# ### Plot the post-fit residuals

# %%
plt.errorbar(
    toas.get_mjds().value,
    f.resids.time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt=".",
)
plt.title(f"{model.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()

# %% [markdown]
# ### Print out the summary

# %% tags=[]
f.print_summary()

# %% [markdown]
# ### Write out a par file for the result

# %%
f.model.write_parfile("/tmp/output.par")
print(f.model.as_parfile())
