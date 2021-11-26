"""Functions for helping write .parfile out in different format. Maybe a better
way is to move this to the parameter class.
"""
import yaml
import numpy as np

class ParfileFormat:
    """Class for different parfile format.

    Parameters
    ----------
    format_config: str
        Configuration file for the parfile format. It is a yaml file with format
        name as the top level keys and the value is a dictionary map from PINT
        parameter name to other format name. If the format has a head, one can
        add a head entry, where 'head' is the key and the head string is the
        value.
    format: str
        The name of the request format.
    """
    _operations = ('to_name', 'to_value', 'to_type', 'to_unit')

    def __init__(self, format_config, format):
        with open(format_config, "r") as cf:
            config = yaml.safe_load(cf)
        try:
            self.entry = config[format]
        except KeyError:
            raise KeyError(f"Parfile format config file {format_config} does "
                "not have format {format}")
        self.format_name = format

    def to_format(self, pint_par):
        """Translate the PINT parameter to .parfile formate that accepted by other package.

        Parameter
        ---------
        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        """
        param_entry = self.entry.get(pint_par.name, None)
        # Make a new pint parameter and apply the translations on the new object
        translated_par = copy.deepcopy(pint_par)
        if param_entry:
            for op in self._operations:
                func = getattr(self, op)
                translated_par = func(translated_par, param_entry)
        return translated_par

    def to_name(self, pint_par, translate_entry):
        """Replace the parameter name"""
        pint_par.name = translate_entry['to_name']
        return pint_par

    def to_value(self, pint_par, translate_entry):
        """Hard Replace the parameter value"""
        pint_par.value = translate_entry['to_value']
        return pint_par

    def to_unit(self, pint_par, translate_entry):
        """Change the parameter quantity's unit"""
        pint_par.units = u.Unit(translate_entry['to_unit'])
        return pint_par

    def to_type(self, pint_par, translate_entry):
        """Change the parameter quantity to a new type"""
        pint_par.quantity = pint_par.quantity.astype(translate_entry['to_type'])
        return pint_par
