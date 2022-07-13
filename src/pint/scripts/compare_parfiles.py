import argparse
import os
import sys

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

from pint.models import get_model

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool to compare parfiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input1", help="First input parfile", type=str)
    parser.add_argument("input2", help="Second input parfile", type=str)
    parser.add_argument(
        "--nodmx", type=bool, default=True, help="Do not print DMX parameters"
    )
    parser.add_argument(
        "--sigma",
        default=3,
        type=float,
        help="Pulsar parameters for which diff_sigma > threshold will be printed with an exclamation point at the end of the line",
    )
    parser.add_argument(
        "--uncertainty_ratio",
        default=1.05,
        type=float,
        help="Pulsar parameters for which the uncertainty has increased by a factor of unc_rat_threshold will be printed with an asterisk at the end of the line",
    )
    parser.add_argument(
        "--nocolor",
        default=False,
        action="store_true",
        help="Turn off colorized output",
    )
    parser.add_argument(
        "--comparison",
        choices=["max", "med", "min", "check"],
        default="max",
        help=""""max"     - print all lines from both models whether they are fit or not (note that nodmx will override this); DEFAULT
                "med"     - only print lines for parameters that are fit
                "min"     - only print lines for fit parameters for which diff_sigma > threshold
                "check"   - only print significant changes with logging.warning, not as string (note that all other modes will still print this))""",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help="Logging level",
        dest="loglevel",
    )
    args = parser.parse_args(argv)
    pint.logging.setup(level=args.loglevel)

    m1 = get_model(args.input1)
    m2 = get_model(args.input2)
    print(
        "\n".join(
            m1.compare(
                m2,
                nodmx=args.nodmx,
                threshold_sigma=args.sigma,
                unc_rat_threshold=args.uncertainty_ratio,
                verbosity=args.comparison,
                usecolor=not args.nocolor,
            )
        )
    )
