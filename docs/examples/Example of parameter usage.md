---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Example of parameter usage

```python jupyter={"outputs_hidden": false}
import pint.models.model_builder as mb
import pint.models.parameter as pp
import astropy.units as u
from astropy.coordinates.angles import Angle
import pytest
```

```python jupyter={"outputs_hidden": false}
model = mb.get_model("B1855+09_NANOGrav_dfg+12_TAI.par")
```

```python jupyter={"outputs_hidden": false}
print(model.params)
```

## Attributions in Parameters

```python jupyter={"outputs_hidden": false}
printed = []
for p in model.params:
    par = getattr(model, p)
    if type(par) in printed:
        continue
    print("Name           ", par.name)
    print("Type           ", type(par))
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Value          ", par.value)
    print("units          ", par.units)
    print("Uncertainty    ", par.uncertainty)
    print("Uncertainty_value", par.uncertainty_value)
    print("Summary        ", par)
    print("Parfile Style  ", par.as_parfile_line())
    print()
    printed.append(type(par))
# Note JUMP and DMX is different.
```

## Making a parameter

```python jupyter={"outputs_hidden": false}
t = pp.floatParameter(name="TEST", value=100, units="Hz", uncertainty=0.03)
print(t)
```

```python jupyter={"outputs_hidden": false}
t2 = pp.floatParameter(name="TEST", value="200", units="Hz", uncertainty=".04")
print(t2)
```

```python jupyter={"outputs_hidden": false}
t3 = pp.floatParameter(
    name="TEST", value=0.3 * u.kHz, units="Hz", uncertainty=4e-5 * u.kHz
)
print(t3)
print(t3.quantity)
print(t3.value)
print(t3.uncertainty)
print(t3.uncertainty_value)
```

## Change Parameter quantity of value

```python jupyter={"outputs_hidden": false}
par = model.F0
print(par)
par.quantity = 200
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python jupyter={"outputs_hidden": false}
# Test F0
print(par)
par.value = 150
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python jupyter={"outputs_hidden": false}
# Example for F0
print(par)
par.value = "100"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python jupyter={"outputs_hidden": false}
# Example for F0
print(par)
par.quantity = "300"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python jupyter={"outputs_hidden": false}
# Examle  F0
par.quantity = 0.3 * u.kHz
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python jupyter={"outputs_hidden": false}
try:
    # Examle  F0
    print(par)
    par.value = 100 * u.second  # SET F0 to seconds as time.
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Value          ", par.value)
    print(par)
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")
```

### For MJD parameters

```python jupyter={"outputs_hidden": false}
par = model.TZRMJD
print(par)
par.quantity = 54000
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for TZRMJD
par.quantity = "54001"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for TZRMJD
par.value = 54002
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for TZRMJD
par.value = "54003"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

### For AngleParameters

```python jupyter={"outputs_hidden": false}
# Example for RAJ
par = model.RAJ
print(par)
par.quantity = 50
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
import astropy
```

```python jupyter={"outputs_hidden": false}
astropy.__version__
```

```python jupyter={"outputs_hidden": false}
Angle(50.0 * u.hourangle)
```

```python jupyter={"outputs_hidden": false}
# Example for RAJ
print(par)
par.quantity = 30.5
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for RAJ
print(par)
par.quantity = "20:30:00"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for RAJ
print(par)
par.value = "20:05:0"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for RAJ
print(par)
par.quantity = 30 * u.deg
print("Quantity       ", par.quantity, type(par.quantity))
print("Quantity in deg", par.quantity.to(u.deg))
print("Value          ", par.value)
print(par)
par.quantity
```

```python jupyter={"outputs_hidden": false}
# Example for RAJ
print(par)
par.value = 40 * u.rad
print("Quantity       ", par.quantity, type(par.quantity))
print("Quantity in rad", par.quantity.to(u.rad))
print("Value          ", par.value)
print(par)
par.quantity
```

Test for wrong unit

```python jupyter={"outputs_hidden": false}
# Example for RAJ
try:
    print(par)
    par.value = 40 * u.second  #  Here second is in the unit of time, not hourangle
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Quantity in rad", par.quantity.to(u.rad))
    print("Value          ", par.value)
    print(par)
    par.quantity
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")
```

```python jupyter={"outputs_hidden": false}
try:
    # Example for RAJ
    print(par)
    par.quantity = 30 * u.hour  # Here hour is in the unit of time, not hourangle
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Quantity in deg", par.quantity.to(u.deg))
    print("Value          ", par.value)
    print(par)
    par.quantity
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")
```

## Example for uncertainty

```python jupyter={"outputs_hidden": false}
par = model.F0
```


```python jupyter={"outputs_hidden": false}
# Example for F0
print(par.uncertainty)
print(par.uncertainty_value)
par.uncertainty = par.uncertainty_value / 1000.0 * u.kHz
print(par)
print(par.uncertainty)
```


```python jupyter={"outputs_hidden": false}
# Example for F0
par.uncertainty_value = 6e-13
print(par)
print(par.uncertainty)
```

```python jupyter={"outputs_hidden": false}
# Example for F0
par.uncertainty_value = 7e-16 * u.kHz
print(par)
print(par.uncertainty)
```

<!-- #region jupyter={"outputs_hidden": false} -->
## How do "prefix parameters" and "mask parameters" work?
<!-- #endregion -->

```python
cat = pp.prefixParameter(
    parameter_type="float", name="CAT0", units=u.ml, long_double=True
)
```

```python
dir(cat)
```

```python
cat.is_prefix
```

```python
cat.index
```

```python

```
