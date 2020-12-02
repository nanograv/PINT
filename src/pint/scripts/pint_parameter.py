""" A script that quarry the PINT built-in parameters.
"""

from pint.models.timing_model import TimingModel, Component
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
    # Get the all the components map from all lower case name
    #  to real component name.
    all_components = Component.component_types
    component_name_map = {}
    for c in all_components.keys():
        component_name_map[c.lower()] = c

    # Set up the parameter name space including the aliases
    all_params = []
    param_name_map = {}
    for cp_name, cp_cls in all_components.items():
        cp = cp_cls()
        for param in cp.params:
            all_params.append(param)
            par = getattr(cp, param)
            # If one parameter does not have aliases,
            # Take the parameter name as key.
            param_name_map[param] = (param, cp_name, par)
            for ali in par.aliases:
                param_name_map[ali] = (param, cp_name, par)

    # set up the required info.
    all_info = {}
    required_info = ['name', 'description', 'units', 'value', 'aliases']

    # Get information for all buiting parameters. The result will be a subset
    # of the this result.
    for param in all_params:
        param_entry = param_name_map[param]
        pint_param = param_entry[0]
        host_cp = param_entry[1]
        info_entry = {}
        for ri in required_info:
            info_entry[ri] = getattr(param_entry[2], ri)
        if host_cp in all_info.keys():
            all_info[host_cp].update({pint_param: info_entry})
        else:
            all_info[host_cp] = {pint_param: info_entry}

    print(all_info)

    # Check the component input.
    if 'all' in args.component:
        quarry_components = list(all_components.keys())
    else:
        quarry_components = args.component

    # Check the parameter input.
    if 'all' in args.parameter:
        quarr_params = all_params
    else:
        quarry_params = args.parameter

    # Construct the output dictionary
    #

    # Get quarry parameters.
    for q_param in quarry_params:
        # First check if the quarry parameter is in the
        # PINT parameter name space
        if q_param in param_names.keys():
            param_entry = param_names[q_param]
            pint_param = param_entry[0]
            host_cp = param_entry[1]
            info_entry = {'quarry_name': q_param}
            for ri in required_info:
                info_entry[ri] = getattr(param_entry[2], ri)

            if host_cp in result.keys():
               result[host_cp].update({pint_param: info_entry})
            else:
               result[host_cp] = {'exists': True, pint_param: info_entry}


    print(result)

    # TODO Fix the out put of prefix and mask parameter.
    # TODO add parameter type to info
    # TODO get all parameters first??

    # Identify thi e format
    if args.format == 'rst':
        pass
