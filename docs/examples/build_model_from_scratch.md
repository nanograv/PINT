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
import astropy.units as u
from pint.models import parameter as p
from pint.models.timing_model import TimingModel, Component
```

### Building a timing model from scrath

Timing model can be put together from the components and then fill up the with parameter values.

We are going to build the model for "NGC6440E.par" from scratch


```python
# list all the existing component
all_components = Component.component_types
print(all_components)
```

### Choose your components
We are not adding dispersion model here, for the demo later.

```python
selected_components = ["AbsPhase", "AstrometryEquatorial", "Spindown"]
components = []
# Initiate the components
for cp_name in selected_components:
    cp = all_components[cp_name]()
    components.append(cp)

tm = TimingModel("NGC6400E", components)
# print the components in the timing model
print(tm.components)
```

### Add parameter values

Right now the parameters have no values or the default values. We are going to add the values
to the model.

```python
params = {
    "PSR": ("1748-2021E",),
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
        par.frozen = not bool(info[1])  # Frozen means not fit.
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
### Add component to the timing model
Here we are adding dispersion model to the model.

```python
dispersion = all_components["DispersionDM"]()
# Using validate False here allows a component being added first and validate later.
# If validate is True, component will be validated automatically.
tm.add_component(dispersion, validate=False)
# print the components out, DispersionDM should be there.
print(tm.components)
# print the delay component order, dispersion should be after the astrometry
print(tm.DelayComponent_list)
```
The DM value can be set as setting the parameters above.

```python
tm.DM.value = 223.9
tm.DM.frozen = not bool(1)  # Frozen means not fit.
tm.DM.uncertainty = 0.3
```

### To delete a component

Deleting a component will remove the component from component list

```python
# Remove by name
tm.remove_component("DispersionDM")
print(tm.components)
```
or remove by object

```python
# Remove by object
tm.add_component(dispersion, validate=False)
print("Before\n", tm.components)
tm.remove_component(dispersion)
print("After\n", tm.components)
```

You can find where does the component stroed using name string via map_component() method

```python
component, order, from_list, comp_type = tm.map_component("Spindown")
# This function return the component instance, the order it in the list, the list it is
# stored and the type of the component class.
print(component)
print(from_list)
print(comp_type)
```
Let us switch the 'DispersionDM' model to 'DisperisonDMX' model.

```python
tm.add_component(all_components["DispersionDMX"]())
print(tm.components)  # "DispersionDMX" should be there.
```
Add DMX parameters
```python
dmx_0002 = p.prefixParameter(
    parameter_type="float", name="DMX_0002", value=None, unit=u.pc / u.cm ** 3
)
tm.add_param_from_top(dmx_0002, "DispersionDMX", setup=True)
# Component should given by its name string. use setup=True make sure new parameter get registered.
assert hasattr(tm, "DMX_0002")
print(tm.delay_deriv_funcs["DMX_0002"])  # the derivative function should be added.
```

### Remove the parameter

```python
tm.remove_param("DMX_0002")
assert not hasattr(tm, "DMX_0002")
```


