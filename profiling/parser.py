
""" Parses inputed profiler output for function times. Designed specifically for 
cProfile output files. Prints functions and times in neat, user-friendly layout.
Requires pandas package: https://pandas.pydata.org/
Install with: pip install pandas
"""

import re
import pandas as pd

dictionary = {
    'tottime': re.compile(r'(?P<tottime>seconds)'),
    'time': re.compile(r'(?P<time>cumtime)')
}

def parse_line(line):
    for key, attr in dictionary.items():
        match = attr.search(line)
        if match:
            return key, attr
    
    return None, None

def parse_file(file):
    data = []
    with open(file, 'r') as filename:
        line = filename.readline()
        # while there are lines to be read...
        while line:
            # look for total program runtime
            key, match = parse_line(line)
            if key == 'tottime':
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
            if key == 'time':
                # if match, read next line to get time
                line = filename.readline()
                # while there's a nonblank line under the keywork line...
                while line.strip():
                    # extract values separated by spaces in the line, store in vals
                    vals = line.split()
                    runtime = vals[3]
                    func = vals[5]
                    if func == 'toa.py:575(__init__)':
                        func = 'Construct TOA Object'
                    elif func == 'toa.py:721(__init__)':
                        func = 'Construct TOAs Object'
                    elif func == 'toa.py:1154(apply_clock_corrections)':
                        func = 'Apply Clock Corrections'
                    elif func == 'toa.py:1228(compute_TDBs)':
                        func = 'Compute TDBs'
                    elif func == 'toa.py:1303(compute_posvels)':
                        func = 'Compute Posvels'
                    elif func == 'fitter.py:143(get_designmatrix)':
                        func = 'Get Designmatrix'
                    elif func == 'fitter.py:70(update_resids)':
                        func = 'Update Resids'
                    elif func == 'decomp_cholesky.py:95(cho_factor)':
                        func = 'Cho Factor'
                    elif func == 'decomp_cholesky.py:159(cho_solve)':
                        func = 'Cho Solve'
                    elif func == 'decomp_svd.py:16(svd)':
                        func = 'svd'
                    row = {
                        'Function': func,
                        'Time(s)': runtime
                    }
                    # add row to data array and go to next line in file
                    data.append(row)
                    line = filename.readline()
            # go to next line in file
            line = filename.readline()
        # put data into pandas' DataFrame for automatic formatting
        df = pd.DataFrame(data)
    if not df.empty:
        print(df.to_string(index=False)) 
                            
                                          
