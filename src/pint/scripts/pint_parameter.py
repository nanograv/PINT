""" A script that quarry the PINT built-in parameters.
"""

from pint.models.timing_model import TimingModel
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Output the all parameters in"
                                                 "the built-in models."
                                                 "The current output is"
                                                 "['name', 'description',"
                                                 " 'unit', 'value', 'aliases']")
    parser.add_argument('-p', '--parameter', nargs='*', default=['all'],
                        help="Quarried parameter names.")
    parser.add_argument('-c', '--component', nargs='*', default=['all'],
                        help="Quarried model component name.")
    parser.add_argument('-o', '--output', type=str, default="stdout",
                        help="Output format:\n"
                             "'rst': Sphinx .rst file.\n"
                             "'json': JSON dictionary file.\n "
                             "'stdout': Stand out.")

    args = parser.parse_args()
    # Get the all the components
    all_components = Component.component_types
    component_name_map = {}
    for c in all_components.keys():
        component_name_map[c.lower()] = c
    result = {}
    required_info = ['name', 'description', 'unit', 'value', 'aliases']

    # Check the component input.
    quarry_components = args.c
    if 'all' in quarry_components:
        quarry_components = list(all_components.keys())
    else:
        built_in_cps = list(component_name_map.keys())
        for quarry_com in quarry_components:
            if quarry_com.lower() in built_in_cps:
                cp_name = component_name_map[quarry_com]
                result[cp_name] = {'exists': True}
            else:
                result[quarry_com] = {'exists': False}

    # Check the parameter input.
    quarry_params = args.p
    if 'all' in quarry_params:
        all_param = True
    else:
        all_param = False

    # Get parameter mapping.
    aliases = {} 
    for cp_name, cp in all_components.items():
        for param in cp.params:
            par = getattr(param)
            for ali in par.aliases:
                aliases[ali] = (param, cp)

    # Start quarrying the parameter information.
    parameters = {}
    for cp_name in result.keys():
        cp_obj = all_components[cp_name]()
        if all_param:
            quarry_params = cp_obj.params
        

        for ri in required_info:



    # Identify the format
    if args.format == 'rst':
