---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Examples for building the timing model from scratch

## This example includes:
 * Build a timing model from scratch
 * Add and delete the components
 * Assign parameter value
 * Add prefix-able parameter

```python
import astropy.units as u  # Astropy units is a very useful module.
from pint.models import parameter as p # We would like to add parameters to the model, so we need parameter module.
from pint.models.timing_model import TimingModel, Component # Interface for timing model
```

## Building a timing model from scratch

Timing model can be put together from the model components and then fill up the with parameter values.

We are going to build the model for "NGC6440E.par" from scratch

### First let us see what components do we have. 

All built-in component classes can be viewed from `Component` class, which uses the meta-class to collect the built-in component class. For how to make a component class, see example "make_component_class"(in preperation) 

```python
# list all the existing component
# all_components below is a dictionary, with the component name as the key and component class as the value.
all_components = Component.component_types 
# Print the component class names. 
_ = [print(x) for x in all_components] # '_' means the out put is not very important. 
```

### Choose your components

Let's start from a relative simple model, with 
`AbsPhase`: The absolute phase of the pulsar, typical parameters, `TZRMJD`, `TZRFREQ`...
`AstrometryEquatorial`: The ICRS equatorial coordinate, parameters, `RAJ`, `DECJ`, `PMRA`, `PMDEC`...  
`Spindown`: The pulsar spin-down model, parameters, `F0`, `F1`...

We will add dispersion model as demo.

```python
selected_components = ["AbsPhase", "AstrometryEquatorial", "Spindown"]
component_instances = []

# Initiate the component instances
for cp_name in selected_components:
    component_class = all_components[cp_name]  # Get the component class
    component_instance = component_class()  # Instantiate component instance
    component_instances.append(component_instance)  # Add component instances into the component instances list



```

### Make timing model (i.e., `TimingModel` instance)

`TimingModel` class provides the storage and interface for the components. It also manages the components internally. 

```python
# Make timing model instance. 
tm = TimingModel("NGC6400E", component_instances)
```

### View the components in the timing model instance.

To view all the components in `TimingModel` instance, we can use the property `.components`, which returns a dictionary (name as the key, component instance as the value).  

Internally, the components are stored in a list(ordered list, you will see why this is important below) according to their types. All the delay type of components (inheritances of `DelayComponent` class) are stored in the `DelayComponent_list`, and the phase type of components(inheritances of `PhaseComponent` class) in the `PhaseComponent_list`. 

```python
# print the components in the timing model
for cp_name, cp_instance in tm.components.items(): # loop over the items in the dictionary. 
    print(cp_name, cp_instance)
```

### Useful functions in `TimingModel`

* `TimingModel.components()`           : List all the components in the timing model.
* `TimingModel.add_component()`        : Add component into the timing model.
* `TimingModel.remove_component()`     : Remove a component from the timing model. 
* `TimingModel.params()`               : List all the parameters in the timing model from all components.
* `TimingModel.setup()`                : Setup the components(e.g., register the derivatives).
* `TimingModel.validate()`             : Validate the components see if the parameter setup properly.
* `TimingModel.delay()`                : Compute the total delay.
* `TimingModel.phase()`                : Compute the total phase.
* `TimingModel.delay_funcs()`          : List all the delay functions from all the delay components.
* `TimingModel.phase_funcs()`          : List all the phase functions from all the phase components.
* `TimingModel.get_component_type()`   : Get all the components from one category
* `TimingModel.map_component()`        : Get the component location. It returns the component's instance, order in                                          the list, host list and its type. 
* `TimingModel.get_params_mapping()`   : Report which component each parameter comes from.
* `TimingModel.get_perfix_mapping()`   : Get the index mapping for one prefix parameters.
* `TimingModel.param_help()`           : Print the help line for all available parameters. 



### Component order

The ordering of the component can affect the timing result significantly. For instance, the binary model needs barycentric time as input, the components before binary model will contribute to the barycentric time. 

PINT provides a default ordering for the components. So in most case, you should not worry about it, unless if you are an expert.

Here is the default order.

```python
from pint.models.timing_model import DEFAULT_ORDER

_ = [print(order) for order in DEFAULT_ORDER]  
```

### Add parameter values

Right now the parameters have no values or the default values. We are going to add the values
to the model.

Please note, PINT's convention for fitting flag is defined in the `Parameter.frozen` attribute. `Parameter.frozen = True` means "do **not** fit this parameter". This is the opposite of TEMPO/TEMPO2 .par file convention. 

```python
# The format of parameter input here is :
#  {'pulsar name': (parameter value, TEMPO Fit flag, uncertainty)}, like the .par file format.
params = {
    "PSR": ("1748-2021E",), # Some parameters can not be fitted and don't have uncertainties. 
    "RAJ": ("17:48:52.75", 1, 0.05),
    "DECJ": ("-20:21:29.0", 1, 0.4),
    "F0": (61.485476554, 1, 5e-10),
    "F1": ("-1.181D-15", 1, 1e-18),
    "PEPOCH": (53750.000000,),
    "POSEPOCH": (53750.000000,),
    "TZRMJD": (53801.38605120074849,),
    "TZRFRQ": (1949.609,),
    "TZRSITE": (1,),
}

# Assign the parameters
for name, info in params.items():
    par = getattr(tm, name)  # Get parameter object from name
    par.value = info[0]  # add parameter value
    if len(info) > 1:
        if info[1] == 1:
            par.frozen = False   # Frozen means not fit.
            par.uncertainty = info[2]
```
### After assign the parameters, it is important to validate the model

Validating model checks if there is any important parameter value missing, and if the
parameters assigned right. If there is anything not assigned right, it will raise expection.

```python
tm.validate()
# You should see all the assigned parameters.
print(tm)
```
Validate function are also integrated to the add_component() function. When adding a component it will validate the timing model as default however it can be switched off by set the flag validate=False. We will use this flag in the next section.    


### Add a component to the timing model

We will add the dispersion component to the timing model. The steps are:
1. Instantiate the Dispersion class 
2. Add dispersion instance into the timing model, with validation as False. 
    <p> Since the dispersion model's parameter have not set yet, validation will fail. We will validate it after the parameters filled in.</p>
3. Add parameters
4. Validate the timing model.

```python
dispersion_class = all_components["DispersionDM"]
dispersion = dispersion_class() # Make the dispersion instance.

# Using validate=False here allows a component being added first and validate later.
tm.add_component(dispersion, validate=False)
```
Let us exam the components in the timing model.

```python
# print the components out, DispersionDM should be there.
print("All components in timing model:")
display(tm.components)

print('\n')
print("Delay components in the DelayComponent_list (Order matters):")

# print the delay component order, dispersion should be after the astrometry
display(tm.DelayComponent_list)
```

The DM value can be set as setting the parameters above.

```python
tm.DM.value = 223.9
tm.DM.frozen = not bool(1)  # Frozen means not fit.
tm.DM.uncertainty = 0.3
```

Run validate again and just make sure everything is setup good. 

```python
tm.validate() # If this fails, that means the DM model was not setup correctly.
```

Now the dispersion model component is added and you are now set for your analysis. 


### Delete a component

Deleting a component will remove the component from component list. 

```python
# Remove by name
tm.remove_component("DispersionDM")
```
```python
display(tm.components)
```

Dispersion model should disappear from the timing model. 


### Add prefix-style parameters

Prefix style parameters are used in certain models (e.g., DMX model). They often don't have a number limit. 

Let us use the `DisperisonDMX` model to demonstrate how it works. 

```python
tm.add_component(all_components["DispersionDMX"]())
```
```python
_  = [print(cp) for cp in tm.components]  # "DispersionDMX" should be there.
```

### Display the existing DMX parameters

What do we have in DMX model. 

Note, `Component` class also has the attribute `params`, which is only for the parameters in the component. 

```python
print(tm.components['DispersionDMX'].params)
```

### Add DMX parameters

Since we already have DMX_0001, we will add DMX_0003, just want to show that for DMX model, DMX index('0001' part) does not have to follow the consecutive order. 

The prefix type of parameters have to use `prefixParameter` class from `pint.models.parameter` module. 
```python
# Add prefix parameters
dmx_0003 = p.prefixParameter(
    parameter_type="float", name="DMX_0003", value=None, unit=u.pc / u.cm ** 3
)

tm.components["DispersionDMX"].add_param(dmx_0003, setup=True)
# tm.add_param_from_top(dmx_0003, "DispersionDMX", setup=True)
# # Component should given by its name string. use setup=True make sure new parameter get registered.
```

### Check if the parameter and component setup correctly. 

```python
display(tm.params)
display(tm.delay_deriv_funcs.keys())  # the derivative function should be added.
```

However only adding DMX_0003 is not enough, since one DMX parameter also need a DMX range, `DMXR1_0003`, `DMXR2_0003` in this case. Without them, the validation will fail. So let us add them as well. 

```python
dmxr1_0003 = p.prefixParameter(
    parameter_type="mjd", name="DMXR1_0003", value=None, unit=u.day
)  # DMXR1 is a type of MJD parameter internally.
dmxr2_0003 = p.prefixParameter(
    parameter_type="mjd", name="DMXR2_0003", value=None, unit=u.day
)  # DMXR1 is a type of MJD parameter internally.

tm.components["DispersionDMX"].add_param(dmxr1_0003, setup=True)
tm.components["DispersionDMX"].add_param(dmxr2_0003, setup=True)

```

```python
tm.params
```

Then validate it. 

```python
tm.validate()
```

### Remove a parameter

Remove a parameter is just use the `remove_param()` function.

```python
tm.remove_param("DMX_0003")
tm.remove_param("DMXR1_0003")
tm.remove_param("DMXR2_0003")
display(tm.params)
```
### Add higher order derivatives of spin frequency to timing model

Adding higher order derivatives of spin frequency(e.g., F2, F2, F4) is a common case for a lot of pulsar. For instance, `F2`/`Fx` is a prefixParameter, but it is not like the `DMX_` parameters, it has to follow the consecutive order. Since it is the coefficient of a Taylor expansion.

Let us list the current spindown model parameters.

```python
display(tm.components['Spindown'].params)
```

Let us add `F2` to the model. `F2` needs a very high precision, we use longdouble=True flag to specify the `F2` value to be a longdouble type.

Note, if we add `F3` directly with out `F2`, the validation will fail. 

```python
f2 = p.prefixParameter(
    parameter_type="float", name="F2", value=0.0, units=u.Hz/(u.s)**2, longdouble=True,
)  
```

```python
tm.components['Spindown'].add_param(f2, setup=True)
```

```python
tm.validate()
```

```python
display(tm.params)
```

Now `F2` can be used in the timing model. 

```python
tm.F2.value = 2e-10
display(tm.F2)
```
