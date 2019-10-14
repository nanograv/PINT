""" This is a script for profiling a python script.
NOTE:
this script uses a program gprof2dot which can be downloaded at
https://github.com/jrfonseca/gprof2dot

The 100 most time consumed runs will be listed in the screen. Depending the
selected output sort key.
A .png image file with the name <script_name> + <git_branch_name>.png will be
generated for listing all the calls.
"""
from __future__ import print_function
import cProfile
import argparse
import subprocess
import pstats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINT tool for run profiling.")
    parser.add_argument("-f", help="The script for profiling.")
    parser.add_argument(
        "--s",
        help="The key for sort result ['cumtime','time']."
        " See https://docs.python.org/2/library/profile.html",
        type=str,
        default="time",
    )
    args = parser.parse_args()
    try:
        branch_name = subprocess.check_output(
            "basename $(git symbolic-ref HEAD)", shell=True
        )
    except:
        branch_name = "profile"
    outfile = args.f.replace(".py", "_") + branch_name.strip()
    if args.s is None:
        cline = "python -m cProfile -o " + outfile + " " + args.f
    else:
        cline = "python -m cProfile -o " + outfile + " -s " + args.s + " " + args.f
    print(cline)
    subprocess.call(cline, shell=True)
    call_tree_line = (
        "gprof2dot -f pstats " + outfile + " | dot -Tpng -o " + outfile + ".png"
    )
    subprocess.call(call_tree_line, shell=True)
    # Check stats
    p = pstats.Stats(outfile)
    p.strip_dirs()
    p.sort_stats(args.s).print_stats(100)
