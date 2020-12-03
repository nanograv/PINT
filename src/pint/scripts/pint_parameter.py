""" A script that quarry the PINT built-in parameters.
"""

import argparse
import copy
from pint.models.timing_model import TimingModel, Component
from pint.utils import split_prefixed_name, PrefixError


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
    prefixed_param = {}
    for cp_name, cp_cls in all_components.items():
        cp = cp_cls()
        for param in cp.params:
            all_params.append(param)
            par = getattr(cp, param)
            if par.is_prefix:
                prefixed_param[par.prefix] = param
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
        info_entry = {'type': str(type(param_entry[2]))}
        for ri in required_info:
            info_entry[ri] = getattr(param_entry[2], ri)
        if host_cp in all_info.keys():
            all_info[host_cp].update({pint_param: info_entry})
        else:
            all_info[host_cp] = {pint_param: info_entry}

    print(all_info)
    print(prefixed_param)

    # Check the component input.
    if 'all' in args.component:
        quarry_components = list(all_components.keys())
    else:
        quarry_components = args.component

    # Check the parameter input.
    if 'all' in args.parameter:
        quarry_params = all_params
    else:
        quarry_params = args.parameter

    # Construct the output dictionary
    out_put = {}

    # check quarry parameters.
    for q_param in quarry_params:
        # First check if the quarry parameter is in the
        # PINT parameter name space
        p_builtin = False
        if q_param in param_name_map.keys(): # The alises should be included
            param_entry = param_name_map[q_param]
            pint_param = param_entry[0]
            cp_name = param_entry[1]
            # Use copy to aviod over write the original.
            out_entry = copy.deepcopy(all_info[cp_name][pint_param])
            p_builtin = True
        else:
            # check prefix
            # First check if the quarried parameter a prefix already
            if q_param in prefixed_param.keys():
                example_param = prefixed_param[q_param]
                index = 1
            else:
                try:
                    prefix, index_str, index = split_prefixed_name(q_param)
                    example_param = prefixed_param[prefix]
                    p_builtin = True
                except PrefixError:
                    continue
                # Get the data from the example parameter
                param_entry = param_name_map[example_param]
                cp_name = param_entry[1]
                # First get the example param info entry
                out_entry = copy.deepcopy(all_info[cp_name][example_param])
                if index > 1:
                    prefix_par = param_entry[2].new_param(index)
                    # rewrite the info entry using the new prefix data.
                    for ri in required_info:
                        out_entry[ri] = getattr(prefix_par, ri)


        if p_builtin:
            out_put[q_param] = out_entry
        else:
            out_put[q_param] = {}





    print(out_put)

    # TODO Fix the out put of prefix and mask parameter.
    # TODO Fix the parameter input and componenet input conflict
    # parameter has 'all', component do not have it.
    # If the parameter is give, component will not display
    # If the compoent is give, it will display all the parameters from that comp
    # Identify thi e format
    if args.format == 'rst':
        pass
