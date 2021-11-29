"""Functions for helping write .parfile out in different format. Maybe a better
way is to move this to the parameter class.
"""
import yaml
import numpy as np
import copy

import astropy.units as u

class ParfileTranslator:
    """Class for translating PINT-style parfile to other formats.

    Parameters
    ----------
    format_config: str, dict
        Configuration file for the parfile format. It is a yaml file or a
        dictionary with format name as the top level keys and the value is a
        dictionary map from PINT parameter name to other format name. If the
        format has a head, one can add a head entry, where 'head' is the key
        and the head string is the value.
    format: str
        The name of the request format.
    """
    _operations = ('to_name', 'to_value', 'to_type', 'to_unit')

    def __init__(self, format_config, format):
        if isinstance(format_config, str):
            self.format_config_file = format_config
            with open(format_config, "r") as cf:
                config = yaml.safe_load(cf)
        elif isinstance(format_config, dict):
            config = format_config
            self.format_config_file = None
        else:
            raise ValueError(f"Unacceptable format of config: {type(format_config)}.")

        try:
            self.format_entry = config[format]
        except KeyError:
            raise KeyError(f"Parfile format config file {format_config} does "
                f"not have the format {format}")
        self.format_name = format

    def _get_param_entry(self, param_name):
        """Get the parameter translation entry.

        Parameter
        ---------
        param_name: str
            Name of the parameter that need to be translated.

        Raises
        ------
        KeyError
            If the parameter name not in the .format_entry dictionary.
        """
        try:
            param_entry = self.format_entry[param_name]
        except KeyError:
            raise KeyError(f"Parameter {param_name} is not in the"
                f" '{self.format_name}' translator. Please check your format"
                f" config file {self.format_config_file}.")
        return param_entry

    def to_format(self, pint_par):
        """Translate the PINT parameter to .parfile formate that accepted by other package.

        Parameter
        ---------
        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        """
        # Make a new pint parameter and apply the translations on the new object
        translated_par = copy.deepcopy(pint_par)
        param_entry = self._get_param_entry(translated_par.name)
        for op in param_entry.keys():
            if op in self._operations:
                func = getattr(self, op)
                translated_par = func(translated_par, param_entry)
        return translated_par

    def to_name(self, pint_par, param_entry):
        """Replace the parameter name to the format required name.

        Parameters
        ----------
        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        param_entry: dict
            The parameter translate entry that includes 'to_name'.
        """
        pint_par.name = param_entry['to_name']
        return pint_par

    def to_value(self, pint_par, param_entry):
        """Hard Replace the parameter value.

        Parameters
        ----------
        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        param_entry: dict
            The parameter translate entry that includes 'to_value'.
        """
        pint_par.value = param_entry['to_value']
        return pint_par

    def to_unit(self, pint_par, param_entry):
        """Change the parameter quantity's unit

        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        param_entry: dict
            The parameter translate entry that includes 'to_unit'.
        """
        pint_par.units = u.Unit(param_entry['to_unit'])
        return pint_par

    def to_type(self, pint_par, param_entry):
        """Change the parameter quantity to a new type

        pint_par: `pint.models.parameter.Parameter` object
            The parameter object that needs to be translated.
        param_entry: dict
            The parameter translate entry that includes 'to_type'.
        """
        pint_par.quantity = pint_par.quantity.astype(param_entry['to_type'])
        return pint_par
