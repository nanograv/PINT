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

```python jupyter={"outputs_hidden": false}
import pint.models.model_builder as mb
```

# To get all the model componets

```python jupyter={"outputs_hidden": false}
# mb.get_components()
```

# Update on timing model

```python jupyter={"outputs_hidden": false}
model = mb.get_model("NGC6440E.par")
```

```python jupyter={"outputs_hidden": false}
model.get_params_of_type("str")
```

```python jupyter={"outputs_hidden": false}
model.get_params_of_type("float")
```

```python jupyter={"outputs_hidden": false}
model.get_params_of_type("prefix")
```

```python jupyter={"outputs_hidden": false}

```
