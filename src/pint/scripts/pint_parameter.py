""" A script that queries the PINT built-in parameters. Currently this script
only has a limited functionality (Print all parameters in .rst format).
"""

import argparse
import copy
from collections import OrderedDict
from pint.models.timing_model import TimingModel, Component
from pint.utils import get_param_name_map, split_prefixed_name, PrefixError


def get_request_info(par, info):
    """ Get the requested information from the parameter object.

    Parameter
    ---------
    par: `pint.models.Parameter` object
        The parameter to get information from
    info: list of strings
        The request list of information. They should be the parameter attribute
        name.
    """
    result = {"name": par.name, "type": type(par).__name__}
    for key in request_info:
        result[key] = getattr(par, key)
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Output all the parameters in"
        "the built-in models."
        "The current output is"
        "['name', 'description',"
        " 'unit', 'value', 'aliases', 'host components']"
    )
    parser.add_argument(
        "-p",
        "--parameter",
        nargs="*",
        default=["all"],
        help="Quarried parameter names.",
    )
    parser.add_argument(
        "-c",
        "--component",
        nargs="*",
        default=None,
        help="Print parameters from components.",
    )
    parser.add_argument(
        "-i",
        "--info",
        nargs="*",
        default=["value", "description", "units", "aliases"],
        help="The requested information from parameter.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="rst",
        help="Output format:\n" "'rst': Sphinx .rst file.\n",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path."
    )

    args = parser.parse_args()

    # Get the all the components map from all lower case name
    #  to real component name.
    all_components = Component.component_types
    component_name_map = {}
    for c in all_components.keys():
        component_name_map[c.lower()] = c

    # Set up the parameter name space including the aliases
    param_name_map, all_params, prefixed_param = get_param_name_map(all_components)

    # Check the parameter input.
    if "all" in args.parameter:
        is_all_param = True
    else:
        is_all_param = False

    # Check the component input.
    if args.component is None:
        parse_comp = False
    else:
        if not is_all_param:
            raise ValueError(
                "Currently only support getting all parameters"
                " from requested components."
            )
        else:
            parse_comp = True
            if "all" in args.component:
                query_components = list(all_components.keys())
            else:
                query_components = args.component

    # Get information for all buiting parameters. The result will be a subset
    # of the this result.

    # When only parse the parameters.
    if not parse_comp:
        if is_all_param:
            query_params = all_params
        else:
            query_params = args.parameter
    else:
        query_params = []
        for qc in query_components:
            if qc.lower() not in component_name_map.keys():
                raise ValueError(
                    "Component `{}` is not recognised by" " PINT.".format(qc)
                )
            else:
                cp = component_name_map[qc]
                cp_obj = all_components[cp]()
                query_params += cp_obj.params

    request_info = args.info
    result = {}
    # check query parameters.
    if not parse_comp:
        for q_param in query_params:
            # First check if the query parameter is in the
            # PINT parameter name space
            q_param = q_param.upper()
            p_builtin = False
            if q_param in param_name_map.keys():  # The alises should be included
                param_entry = param_name_map[q_param]
                pint_param = param_entry[0]
                host_cp = param_entry[1]
                request_par = param_entry[2]
                p_builtin = True
            else:
                # check prefix
                # First check if the queried parameter a prefix already
                if q_param in prefixed_param.keys():
                    example_param = prefixed_param[q_param][0]
                    index = 1
                    p_builtin = True
                else:
                    try:
                        prefix, index_str, index = split_prefixed_name(q_param)
                        example_param = prefixed_param[prefix][0]
                        p_builtin = True
                    except PrefixError:
                        continue
                # Get the data from the example parameter
                param_entry = param_name_map[example_param]

                host_cp = param_entry[1]
                # First get the example param info entry
                if index > 1:
                    request_par = param_entry[2].new_param(index)
                else:
                    request_par = param_entry[2]

            if p_builtin:
                out_entry = get_request_info(request_par, request_info)
                out_entry.update({"host": host_cp, "status": True})
            else:
                out_entry = {"status": False}
            result[q_param] = out_entry
    else:
        pass

    # Out put the result
    if args.format == "rst":
        title_map = OrderedDict()
        title_map["Name"] = "name"
        title_map["Type"] = "type"
        title_map["Unit"] = "units"
        title_map["Description"] = "description"
        title_map["Default Value"] = "value"
        title_map["Aliases"] = "aliases"
        title_map["Host Components"] = "host"
        # title entry
        title_entry = "   * "
        for ii, title in enumerate(title_map.keys()):
            if ii == 0:
                pad = "- "
            else:
                pad = "     - "
            title_entry += pad + title + "\n"
        # parameter entries
        param_entries = ""
        result_list = list(result.items())
        result_list.sort()
        for pp in result_list:
            entries = "   * "
            for ii, info in enumerate(title_map.values()):
                if ii == 0:
                    pad = "- "
                else:
                    pad = "     - "
                if isinstance(pp[1][info], list):
                    out_info = ", ".join(pp[1][info])
                else:
                    out_info = str(pp[1][info])
                entries += pad + out_info + "\n"
            param_entries += entries
        table = title_entry + param_entries

        header = ".. list-table::\n" "   :header-rows: 1\n\n\n"
        out_str = header + table

    if args.output is None:
        print(out_str)
    else:
        f = open(args.output, "w")
        f.write(out_str)
        f.close()
