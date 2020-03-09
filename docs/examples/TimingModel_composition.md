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

# Timing Model usage example


## Build a timing model

```python
from pint.models import get_model
from pint.models.timing_model import Component
```

One can build a timing model via get_model() method. get_model() will make the model according to the .par file. All the model components, delays or phases, will be stored in order.

```python
par = "B1855+09_NANOGrav_dfg+12_TAI.par"
m = get_model(par)
```


```python
print(m.F0.quantity)
print(m.F0.units)
print(m.F0.description)
```

```python
# m.dispersion_delay
```

To take a look what type of model component are in the model

```python
m.component_types  # We have delay component and phase component
```

TimingModel class stores the delay model components and phase components in the lists

```python
m.DelayComponent_list  # Delays are stored in the DelayComponent_list
```

```python
m.PhaseComponent_list  # phases are stored in the PhaseComponent_list
```

To add a component:

```python
from pint.models.astrometry import AstrometryEcliptic
```

```python
a = AstrometryEcliptic()  # init the AstrometryEcliptic instance
```

Add the new component instance into time model with order 3

```python
m.add_component(a, validate=False)
```

```python
m.DelayComponent_list  # The new instance is added to delay component list
# index 3
```

To remove a component is simple. Just use remove it from the list. You can map the component instance using name string via map_component() method

```python
component, order, from_list, comp_type = m.map_component("AstrometryEcliptic")
from_list.remove(component)
```

```python
m.DelayComponent_list  # AstrometryEcliptic is removed from delay list.
```

To switch the order of a component, just change the index in the component list.

```python
# First map the component instance
component, order, from_list, comp_type = m.map_component("PhaseJump")
# If one wants to move this component to a new order without swapping
from_list.remove(component)
from_list.insert(5, component)
```

```python
m.DelayComponent_list
```

```python
# If one wants to swap with other component
component, order, from_list, comp_type = m.map_component("DispersionDMX")
new_order = 3
from_list[order], from_list[new_order] = from_list[new_order], from_list[order]
```

```python
m.DelayComponent_list
```

Delays will be computed in order.

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

One can get the delay upto some component. For example, I want to the delay computation stop at jump delay.

```python
to_jump_delay = m.delay(t, cutoff_component="SolarSystemShapiro")
to_jump_delay
```

```python
m.F1
```

```python
Component.component_types
```

```python
t, *_ = Component.component_types.values()
ti = t()
ti.component_special_params
```

```python
from collections import defaultdict

special = defaultdict(list)
for n, t in Component.component_types.items():
    for p in t().component_special_params:
        special[p].append(n)


special
```

```python
G = Component.component_types["Glitch"]
g = G()
g.component_special_params
```
