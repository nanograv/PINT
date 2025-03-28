""" Parses profiler output for function times. Designed specifically for 
cProfile output files. Prints functions and times in neat, user-friendly layout.
Requires pandas package: https://pandas.pydata.org/
Install with: pip install pandas
"""

import re
import pandas as pd

dictionary = {
    "tottime": re.compile(r"(?P<tottime>seconds)"),
    "time": re.compile(r"(?P<time>cumtime)"),
}


def parse_line(line):
    for key, attr in dictionary.items():
        if match := attr.search(line):
            return key, attr

    return None, None


def parse_file(file):
    data = []
    with open(file, "r") as filename:
        line = filename.readline()
        # while there are lines to be read...
        while line:
            # look for total program runtime
            key, match = parse_line(line)
            if key == "tottime":
                # extract values separated by spaces in the line, store in vals
                vals = line.split()
                total = vals[7]
                print("Total Time: " + total + " s")
                # exit loop since only need total time once, appears multiple times in input
                break
            # go to next line in file
            line = filename.readline()
        # while there are lines to be read...
        while line:
            # look for keyword 'cumtime' to add desired data to array
            key, match = parse_line(line)
            if key == "time":
                # if match, read next line to get time
                line = filename.readline()
                n = 1
                # while there's a non-blank line under the keyword line...
                while line.strip():
                    # extract values separated by spaces in the line, store in vals
                    vals = line.split()
                    runtime = vals[3]
                    func = vals[5]
                    if "__init__" in func and n == 1:
                        func = "Construct TOA Object"
                    elif "__init__" in func and n == 2:
                        func = "Construct TOAs Object"
                    elif "apply_clock_corrections" in func:
                        func = "Apply Clock Corrections"
                    elif "compute_TDBs" in func:
                        func = "Compute TDBs"
                    elif "compute_posvels" in func:
                        func = "Compute Posvels"
                    elif "get_designmatrix" in func:
                        func = "Get Designmatrix"
                    elif "update_resids" in func:
                        func = "Update Resids"
                    elif "cho_factor" in func:
                        func = "Cho Factor"
                    elif "cho_solve" in func:
                        func = "Cho Solve"
                    elif "svd" in func:
                        func = "svd"
                    elif "select_toa_mask" in func:
                        func = "Select TOA Mask"
                    row = {"Function": func, "Time(s)": runtime}
                    # add row to data array and go to next line in file
                    data.append(row)
                    line = filename.readline()
                    n = n + 1
            # go to next line in file
            line = filename.readline()
        # put data into pandas' DataFrame for automatic formatting
        df = pd.DataFrame(data)
    if not df.empty:
        print(df.to_string(index=False))
