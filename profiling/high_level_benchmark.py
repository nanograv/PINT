"""
This script provides a top-level, brief outline of the profile of
various benchmarks.
Uses script cpuinfo.py: https://pypi.org/project/py-cpuinfo/
Install with: pip install py-cpuinfo
"""

import argparse
import cpuinfo
import scipy
import astropy
import numpy
import subprocess
import cProfile
import pstats
import pint
import sys
import os
import platform
from parser import parse_file


def bench_file(script):
    outfile = script.replace(".py", "_prof_summary")
    cline = "python -m cProfile -o " + outfile + " " + script
    print(cline)
    # use DENULL to suppress logging output
    subprocess.call(
        cline, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return outfile


def get_results(script, outfile):
    print("*******************************************************************")
    print("OUTPUT FOR " + script.upper() + ":")
    # put output in file for parsing
    f = open("bench.out", "w")
    old_stdout = sys.stdout
    sys.stdout = f
    # Check stats
    p = pstats.Stats(outfile)
    p.strip_dirs()
    # choose the functions to display
    if script == "bench_load_TOAs.py":
        p.print_stats("\(__init__", "toa")
        p.print_stats("\(apply_clock")
        p.print_stats("\(compute_TDBs")
        p.print_stats("\(compute_posvels")
    elif script == "bench_chisq_grid.py" or script == "bench_chisq_grid_WLSFitter.py":
        p.print_stats("\(get_designmatrix")
        p.print_stats("\(update_resid")
        p.print_stats("\(cho_factor")
        p.print_stats("\(cho_solve")
        p.print_stats("\(svd")
        p.print_stats("\(select_toa_mask")
    else:
        p.print_stats("only print total time")  # for MCMC, only display total runtime
    f.close()
    # return output to terminal
    sys.stdout = old_stdout
    # parse file for desired info and format user-friendly output
    parse_file("bench.out")
    os.remove("bench.out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-level summary of python file timing."
    )
    # scripts to be evaluated
    script1 = "bench_load_TOAs.py"
    script2 = "bench_chisq_grid.py"
    script3 = "bench_chisq_grid_WLSFitter.py"
    script4 = "bench_MCMC.py"

    # time scripts
    output1 = bench_file(script1)
    output2 = bench_file(script2)
    output3 = bench_file(script3)
    output4 = bench_file(script4)
    print()

    # display computer & software info
    compID = cpuinfo.get_cpu_info()["brand"]
    print("Processor running this script: " + compID)
    pyversion = platform.python_version()
    spversion = scipy.__version__
    apversion = astropy.__version__
    npversion = numpy.__version__
    pintversion = pint.__version__
    print("Python version: " + pyversion)
    print(
        "SciPy version: "
        + spversion
        + ", AstroPy version: "
        + apversion
        + ", NumPy version: "
        + npversion
    )
    print("PINT version: " + pintversion)

    # output results
    print()
    get_results(script1, output1)
    print()
    get_results(script2, output2)
    print()
    get_results(script3, output3)
    print()
    get_results(script4, output4)
    print()
