#!/usr/bin/env python
""" This is a script for profiling a python script.
NOTE:
this script uses a program gprof2dot which can be downloaded at
https://github.com/jrfonseca/gprof2dot

The 100 most time consumed runs will be listed in the screen. Depending the
selected output sort key.
A .pdf file with the name <script_name> + <git_branch_name>.pdf will be
generated for listing all the calls.
"""


import argparse
import subprocess
import pstats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINT tool for run profiling.")
    parser.add_argument("script", help="The script for profiling.")
    parser.add_argument(
        "--sort",
        help="The key for sort result ['cumtime','time']."
        " See https://docs.python.org/2/library/profile.html",
        type=str,
        default="time",
    )
    args = parser.parse_args()
    outfile = args.script.replace(".py", "_profile")
    if args.sort is None:
        cline = f"python -m cProfile -o {outfile} {args.script}"
    else:
        cline = f"python -m cProfile -o {outfile} -s {args.sort} {args.script}"
    print(cline)
    subprocess.call(cline, shell=True)
    call_tree_line = f"gprof2dot -f pstats {outfile} | dot -Tpdf -o {outfile}.pdf"
    subprocess.call(call_tree_line, shell=True)
    # Check stats
    p = pstats.Stats(outfile)
    p.strip_dirs()
    p.sort_stats(args.sort).print_stats(100)

    # if you would like to view the outfile with an interactive html viewer,
    # use cprofilev, https://github.com/ymichael/cprofilev (pip install cprofilev):
    # cline = "cprofilev -f " + outfile
    # print(cline)
    # subprocess.call(cline, shell=True)
