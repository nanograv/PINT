__all__ = [
    "DegeneracyWarning",
    "ConvergenceFailure",
    "MaxiterReached",
    "StepProblem",
    "CorrelatedErrors",
    "MissingTOAs",
    "PropertyAttributeError",
    "TimingModelError",
    "MissingParameter",
    "AliasConflict",
    "UnknownParameter",
    "UnknownBinaryModel",
    "MissingBinaryError",
    "PINTPrecisionError",
    "PrefixError",
    "InvalidModelParameters",
    "ComponentConflict",
    "ClockCorrectionError",
    "NoClockCorrections",
    "ClockCorrectionOutOfRange",
]


# originally from fitter.py
class DegeneracyWarning(UserWarning):
    pass


class ConvergenceFailure(ValueError):
    pass


class MaxiterReached(ConvergenceFailure):
    pass


class StepProblem(ConvergenceFailure):
    pass


class CorrelatedErrors(ValueError):
    def __init__(self, model):
        trouble_components = [
            c.__class__.__name__
            for c in model.NoiseComponent_list
            if c.introduces_correlated_errors
        ]
        super().__init__(
            f"Model has correlated errors and requires a GLS-based fitter; "
            f"remove {trouble_components} if you want to use WLS"
        )
        self.trouble_components = trouble_components


# from timing_model.py
class MissingTOAs(ValueError):
    """Some parameter does not describe any TOAs."""

    def __init__(self, parameter_names):
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        if len(parameter_names) == 1:
            msg = f"Parameter {parameter_names[0]} does not correspond to any TOAs: you might need to run `model.find_empty_masks(toas, freeze=True)`"
        elif len(parameter_names) > 1:
            msg = f"Parameters {' '.join(parameter_names)} do not correspond to any TOAs: you might need to run `model.find_empty_masks(toas, freeze=True)`"
        else:
            raise ValueError("Incorrect attempt to construct MissingTOAs")
        super().__init__(msg)
        self.parameter_names = parameter_names


class PropertyAttributeError(ValueError):
    pass


class TimingModelError(ValueError):
    """Generic base class for timing model errors."""

    pass


class MissingParameter(TimingModelError):
    """A required model parameter was not included.

    Parameters
    ----------
    module
        name of the model class that raised the error
    param
        name of the missing parameter
    msg
        additional message

    """

    def __init__(self, module, param, msg=None):
        super().__init__(msg)
        self.module = module
        self.param = param
        self.msg = msg

    def __str__(self):
        result = f"{self.module}.{self.param}"
        if self.msg is not None:
            result += "\n  " + self.msg
        return result


class AliasConflict(TimingModelError):
    """If the same alias is used for different parameters."""

    pass


class UnknownParameter(TimingModelError):
    """Signal that a parameter name does not match any PINT parameters and their aliases."""

    pass


class UnknownBinaryModel(TimingModelError):
    """Signal that the par file requested a binary model not in PINT."""

    def __init__(self, message, suggestion=None):
        super().__init__(message)
        self.suggestion = suggestion

    def __str__(self):
        base_message = super().__str__()
        if self.suggestion:
            return f"{base_message} Perhaps use {self.suggestion}?"
        return base_message


class MissingBinaryError(TimingModelError):
    """Error for missing BINARY parameter."""

    pass


# from utils.py
class PINTPrecisionError(RuntimeError):
    pass


class PrefixError(ValueError):
    pass


# from models.parameter.py
class InvalidModelParameters(ValueError):
    pass


# models.model_builder.py
class ComponentConflict(ValueError):
    """Error for multiple components can be select but no other indications."""


# observatories.__init__.py
class ClockCorrectionError(RuntimeError):
    """Unspecified error doing clock correction."""

    pass


class NoClockCorrections(ClockCorrectionError):
    """Clock corrections are expected but none are available."""

    pass


class ClockCorrectionOutOfRange(ClockCorrectionError):
    """Clock corrections are available but the requested time is not covered."""

    pass
