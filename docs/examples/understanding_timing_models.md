---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Understanding Timing Models


## Build a timing model starting from a par file

```python
from pint.models import get_model
from pint.models.timing_model import Component
```

One can build a timing model via `get_model()` method. This will read the par file and instantiate all the delay and phase components, using the default ordering.

```python
par = "B1855+09_NANOGrav_dfg+12_TAI.par"
m = get_model(par)
```


Each of the parameters in the model can be accessed as an attribute of the `TimingModel` object.
Behind the scenes PINT figures out which component the parameter is stored in.

Each parameter has attributes like the quantity (which includes units), and a description (see the Understanding Parameters notebook for more detail)

```python
print(m.F0.quantity)
print(m.F0.description)
```

We can now explore the structure of the model

```python
# This gives a list of all of the component types (so far there are only delay and phase components)
m.component_types
```

```python
dir(m)
```

The TimingModel class stores lists of the delay model components and phase components that make up the model

```python
# When this list gets printed, it shows the parameters that are associated with each component as well.
m.DelayComponent_list
```

```python
# Now let's look at the phase components. These include the absolute phase, the spindown model, and phase jumps
m.PhaseComponent_list
```

We can add a component to an existing model

```python
from pint.models.astrometry import AstrometryEcliptic
```

```python
a = AstrometryEcliptic()  # init the AstrometryEcliptic instance
```

```python
# Add the component to the model
# It will be put in the default order
# We set validate=False since we have not set the parameter values yet, which would cause validate to fail
m.add_component(a, validate=False)
```

```python
m.DelayComponent_list  # The new instance is added to delay component list
```

There are two ways to remove a component from a model. This simplest is to use the `remove_component` method to remove it by name.

```python
# We will not do this here, since we'll demonstrate a different method below.
# m.remove_component("AstrometryEcliptic")
```

Alternatively, you can have more control using the `map_component()` method, which takes either a string with component name,
or a Component instance and returns a tuple containing the Component instance, its order in the relevant component list,
the list of components of this type in the model, and the component type (as a string)

```python
component, order, from_list, comp_type = m.map_component("AstrometryEcliptic")
print("Component : ", component)
print("Type      : ", comp_type)
print("Order     : ", order)
print("List      : ")
_ = [print(c) for c in from_list]
```


```python
# Now we can remove the component by directly manipulating the list
from_list.remove(component)
```

```python
m.DelayComponent_list  # AstrometryEcliptic has been removed from delay list.
```

To switch the order of a component, just change the order of the component list

**NB: that this should almost never be done!  In most cases the default order of the delay components is correct. Experts only!**

```python
# Let's look at the order of the components in the delay list first
_ = [print(dc.__class__) for dc in m.DelayComponent_list]
```

```python
# Now let's swap the order of DispersionDMX and Dispersion
component, order, from_list, comp_type = m.map_component("DispersionDMX")
new_order = 3
from_list[order], from_list[new_order] = from_list[new_order], from_list[order]
```

```python
# Print the classes to see the order switch
_ = [print(dc.__class__) for dc in m.DelayComponent_list]
```

Delays are always computed in the order of the DelayComponent_list

```python
# First get the toas
from pint.toa import get_TOAs

t = get_TOAs("B1855+09_NANOGrav_dfg+12.tim")
```

```python
# compute the total delay
total_delay = m.delay(t)
total_delay
```

One can get the delay up to some component. For example, if you want the delay computation stop after the Solar System Shapiro delay.

By default the delay of the specified component *is* included. This can be changed by the keyword parameter `include_last=False`.

```python
to_jump_delay = m.delay(t, cutoff_component="SolarSystemShapiro")
to_jump_delay
```

Here is a list of all the Component types that PINT knows about

```python
Component.component_types
```

When PINT builds a model from a par file, it has to infer what components to include in the model.
This is done by the `component_special_params` of each `Component`. A component will be instantiated
when one of its special parameters is present in the par file.

```python
from collections import defaultdict

special = defaultdict(list)
for comp, tp in Component.component_types.items():
    for p in tp().component_special_params:
        special[p].append(comp)


special
```
