"""Functions for helping write .parfile out in different format. Maybe a better
way is to move this to the parameter class.
"""
import yaml


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
    def __init__(self, format_config, format):
        with open(config_file, "r") as cf:
            config = yaml.safe_load(cf)
        try:
            self.entry = config[format]
        except KeyError:
            raise KeyError(f"Parfile format config file {format_config} does "
                "not have format {format}")
        self.format_name = format

    def translate_format(self, pint_par):
        param_entry = self.entry.get(pint_par, None)
