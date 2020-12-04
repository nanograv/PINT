---
jupyter:
  jupytext:
    cell_metadata_json: true
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

# Understanding Parameters

```python execution={"iopub.execute_input": "2020-09-10T16:29:05.962203Z", "iopub.status.busy": "2020-09-10T16:29:05.961161Z", "iopub.status.idle": "2020-09-10T16:29:09.110676Z", "shell.execute_reply": "2020-09-10T16:29:09.110045Z"} jupyter={"outputs_hidden": false}
import pint.models
import pint.models.parameter as pp
import astropy.units as u
from astropy.coordinates.angles import Angle
from astropy.time import Time
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.114066Z", "iopub.status.busy": "2020-09-10T16:29:09.113524Z", "iopub.status.idle": "2020-09-10T16:29:09.336889Z", "shell.execute_reply": "2020-09-10T16:29:09.336282Z"} jupyter={"outputs_hidden": false}
# Load a model to play with
model = pint.models.get_model("B1855+09_NANOGrav_dfg+12_TAI.par")
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.348075Z", "iopub.status.busy": "2020-09-10T16:29:09.347518Z", "iopub.status.idle": "2020-09-10T16:29:09.350913Z", "shell.execute_reply": "2020-09-10T16:29:09.350441Z"} jupyter={"outputs_hidden": false}
# This model has a large number of parameters of various types
model.params
```

## Attributes of Parameters

Each parameter has attributes that specify the name and type of the parameter, its units, and the uncertainty.
The `par.quantity` and `par.uncertainty` are both astropy quantities with units. If you need the bare values,
access `par.value` and `par.uncertainty_value`, which will be numerical values in the units of `par.units`

Let's look at those for each of the types of parameters in this model.

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.360328Z", "iopub.status.busy": "2020-09-10T16:29:09.359604Z", "iopub.status.idle": "2020-09-10T16:29:09.377048Z", "shell.execute_reply": "2020-09-10T16:29:09.377476Z"} jupyter={"outputs_hidden": false}
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
```

Note that DMX_nnnn is an example of a `prefixParameter`. These are parameters that are indexed by a numerical value and a componenent can have an arbitrary number of them.
In some cases, like `Fn` they are coefficients of a Taylor expansion and so all indices up to the maximum must be present. For others, like `DMX_nnnn` some indices can be missing without a problem.

`prefixParameter`s can be used to hold indexed parameters of various types ( float, bool, str, MJD, angle ). Each one will instantiate a parameter of that type as `par.param_comp`.
When you print the parameter it looks like the `param_comp` type.

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.381843Z", "iopub.status.busy": "2020-09-10T16:29:09.381295Z", "iopub.status.idle": "2020-09-10T16:29:09.384568Z", "shell.execute_reply": "2020-09-10T16:29:09.384053Z"}
# Note that for each instance of a prefix parameter is of type `prefixParameter`
print("Type = ", type(model.DMX_0016))
print("param_comp type = ", type(model.DMX_0016.param_comp))
print("Printing gives : ", model.DMX_0016)
```

## Constructing a parameter

You can make a Parameter instance by calling its constructor

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.388561Z", "iopub.status.busy": "2020-09-10T16:29:09.388009Z", "iopub.status.idle": "2020-09-10T16:29:09.390425Z", "shell.execute_reply": "2020-09-10T16:29:09.390875Z"} jupyter={"outputs_hidden": false}
# You can specify the vaue as a number
t = pp.floatParameter(name="TEST", value=100, units="Hz", uncertainty=0.03)
print(t)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.394886Z", "iopub.status.busy": "2020-09-10T16:29:09.394261Z", "iopub.status.idle": "2020-09-10T16:29:09.397410Z", "shell.execute_reply": "2020-09-10T16:29:09.396920Z"} jupyter={"outputs_hidden": false}
# Or as a string that will be parsed
t2 = pp.floatParameter(name="TEST", value="200", units="Hz", uncertainty=".04")
print(t2)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.402455Z", "iopub.status.busy": "2020-09-10T16:29:09.401844Z", "iopub.status.idle": "2020-09-10T16:29:09.404895Z", "shell.execute_reply": "2020-09-10T16:29:09.404422Z"} jupyter={"outputs_hidden": false}
# Or as an astropy Quantity with units (this is the preferred method!)
t3 = pp.floatParameter(
    name="TEST", value=0.3 * u.kHz, units="Hz", uncertainty=4e-5 * u.kHz
)
print(t3)
print(t3.quantity)
print(t3.value)
print(t3.uncertainty)
print(t3.uncertainty_value)
```

## Setting Parameters

The value of a parameter can be set in multiple ways. As usual, the preferred method is to set it using an astropy Quantity, so units will be checked and respected

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.409561Z", "iopub.status.busy": "2020-09-10T16:29:09.409014Z", "iopub.status.idle": "2020-09-10T16:29:09.411819Z", "shell.execute_reply": "2020-09-10T16:29:09.412262Z"} jupyter={"outputs_hidden": false}
par = model.F0
# Here we set it using a Quantity in kHz. Because astropy Quantities are used, it does the right thing!
par.quantity = 0.3 * u.kHz
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.416589Z", "iopub.status.busy": "2020-09-10T16:29:09.416031Z", "iopub.status.idle": "2020-09-10T16:29:09.419490Z", "shell.execute_reply": "2020-09-10T16:29:09.418920Z"} jupyter={"outputs_hidden": false}
# Here we set it with a bare number, which is interpreted as being in the units `par.units`
print(par)
par.quantity = 200
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.424527Z", "iopub.status.busy": "2020-09-10T16:29:09.423970Z", "iopub.status.idle": "2020-09-10T16:29:09.426589Z", "shell.execute_reply": "2020-09-10T16:29:09.427036Z"} jupyter={"outputs_hidden": false}
# If you try to set the parameter to a quantity that isn't compatible with the units, it raises an exception
try:
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

### MJD parameters

These parameters hold a date as an astropy `Time` object. Numbers will be interpreted as MJDs in the default time scale of the parameter (which is UTC for the TZRMJD parameter)

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.433972Z", "iopub.status.busy": "2020-09-10T16:29:09.433400Z", "iopub.status.idle": "2020-09-10T16:29:09.437110Z", "shell.execute_reply": "2020-09-10T16:29:09.437563Z"} jupyter={"outputs_hidden": false}
par = model.TZRMJD
print(par)
par.quantity = 54000
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.443202Z", "iopub.status.busy": "2020-09-10T16:29:09.442648Z", "iopub.status.idle": "2020-09-10T16:29:09.446875Z", "shell.execute_reply": "2020-09-10T16:29:09.446314Z"} jupyter={"outputs_hidden": false}
# And of course, you can set them with a `Time` object
par.quantity = Time.now()
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.452453Z", "iopub.status.busy": "2020-09-10T16:29:09.451797Z", "iopub.status.idle": "2020-09-10T16:29:09.455744Z", "shell.execute_reply": "2020-09-10T16:29:09.455178Z"}
# I wonder if this should get converted to UTC?
par.quantity = Time(58000.0, format="mjd", scale="tdb")
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

### AngleParameters

These store quanities as angles using astropy coordinates

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.461206Z", "iopub.status.busy": "2020-09-10T16:29:09.460628Z", "iopub.status.idle": "2020-09-10T16:29:09.463885Z", "shell.execute_reply": "2020-09-10T16:29:09.463434Z"} jupyter={"outputs_hidden": false}
# The unit for RAJ is hourangle
par = model.RAJ
print(par)
par.quantity = 12
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.470200Z", "iopub.status.busy": "2020-09-10T16:29:09.469585Z", "iopub.status.idle": "2020-09-10T16:29:09.473559Z", "shell.execute_reply": "2020-09-10T16:29:09.473085Z"} jupyter={"outputs_hidden": false}
# Best practice is to set using a quantity with units
print(par)
par.quantity = 30.5 * u.hourangle
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.479653Z", "iopub.status.busy": "2020-09-10T16:29:09.479085Z", "iopub.status.idle": "2020-09-10T16:29:09.483177Z", "shell.execute_reply": "2020-09-10T16:29:09.482702Z"} jupyter={"outputs_hidden": false}
# But a string will work
par.quantity = "20:30:00"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.490228Z", "iopub.status.busy": "2020-09-10T16:29:09.489668Z", "iopub.status.idle": "2020-09-10T16:29:09.493522Z", "shell.execute_reply": "2020-09-10T16:29:09.494042Z"} jupyter={"outputs_hidden": false}
# And the units can be anything that is convertable to hourangle
print(par)
par.quantity = 30 * u.deg
print("Quantity       ", par.quantity, type(par.quantity))
print("Quantity in deg", par.quantity.to(u.deg))
print("Value          ", par.value)
print(par)
par.quantity
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:09.499142Z", "iopub.status.busy": "2020-09-10T16:29:09.498578Z", "iopub.status.idle": "2020-09-10T16:29:09.501075Z", "shell.execute_reply": "2020-09-10T16:29:09.501562Z"} jupyter={"outputs_hidden": false}
# Here, setting RAJ to an incompatible unit will raise an exception
try:
    # Example for RAJ
    print(par)
    par.quantity = 30 * u.hour  # Here hour is in the unit of time, not hourangle
    print("Quantity       ", par.quantity, type(par.quantity))
    print(par)
    par.quantity
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")
```
