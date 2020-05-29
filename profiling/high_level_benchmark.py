
""" 
This script provides a top-level, brief outline of the profile of 
various benchmarks.
Uses script cpuinfo.py -- 'pip install py-cpuinfo' 
Accessed here: https://pypi.org/project/py-cpuinfo/ 
"""

from __future__ import print_function
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

def bench_file(script, args):
      outfile = script.replace(".py", "_prof_summary")
      if args.sort is None:
            cline = "python -m cProfile -o " + outfile + " " + script
      else:
            cline = (
                  "python -m cProfile -o " + outfile + " -s " + args.sort + " " + script
            )
      print(cline)
      subprocess.call(cline, shell=True)
      return outfile

def display_results(script, outfile):
      # Check stats
      p = pstats.Stats(outfile)
      p.strip_dirs()
      print("*******************************************************************")
      print("OUTPUT FOR " + script.upper() + ":")
      # choose the functions to display
      if script == "bench_load_TOAs.py":
            p.sort_stats(args.sort).print_stats('\(__init__', 'toa')
            p.sort_stats(args.sort).print_stats('\(apply_clock')
            p.sort_stats(args.sort).print_stats('\(compute_TDBs')
            p.sort_stats(args.sort).print_stats('\(compute_posvels')
      elif script == "bench_chisq_grid.py" or script == "bench_chisq_grid_WLSFitter.py":
            p.sort_stats(args.sort).print_stats('\(get_designmatrix')
            p.sort_stats(args.sort).print_stats('\(update_resid')
            p.sort_stats(args.sort).print_stats('\(cho_factor')
            p.sort_stats(args.sort).print_stats('\(cho_solve')
            p.sort_stats(args.sort).print_stats('svd')
      else:
            p.sort_stats(args.sort).print_stats('only print total time')  # for MCMC, only display total runtime
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-level summary of python file timing.")
    parser.add_argument(
        "--sort",
        help="The key for sort result ['cumtime','time']."
        " See https://docs.python.org/2/library/profile.html",
        type=str,
        default="time",
    )
    args = parser.parse_args()
    # scripts to be evaluated
    script1 = "bench_load_TOAs.py"
    script2 = "bench_chisq_grid.py"
    script3 = "bench_chisq_grid_WLSFitter.py"
    script4 = "bench_MCMC.py"

    # display computer & software info
    compID = cpuinfo.get_cpu_info()['brand']
    print("Processor running this script: " + compID)
    spversion = scipy.__version__
    apversion = astropy.__version__
    npversion = numpy.__version__
    pintversion = pint.__version__
    print("SciPy version: " + spversion + ", AstroPy version: " + apversion 
          + ", NumPy version: " + npversion)
    print("PINT version: " + pintversion)

    # time scripts
    output1 = bench_file(script1, args)
    output2 = bench_file(script2, args)
    output3 = bench_file(script3, args)
    output4 = bench_file(script4, args)
    # output results  
    display_results(script1, output1)
    display_results(script2, output2)
    display_results(script3, output3)
    display_results(script4, output4)


