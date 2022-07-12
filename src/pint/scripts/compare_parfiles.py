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
            )
        )
    )
