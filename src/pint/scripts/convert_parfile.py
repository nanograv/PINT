import argparse
import os

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

from pint.models import get_model
from pint.models.parameter import _parfile_formats

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for command-line parfile format conversions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input parfile", type=str)

    parser.add_argument(
        "-f",
        "--format",
        help=("Format for output"),
        choices=_parfile_formats,
        default="pint",
    )
    parser.add_argument(
        "-o",
        "--out",
        help=("Output filename [default=stdout]"),
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=pint.logging.levels,
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", default=0, action="count", help="Decrease output verbosity"
    )

    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )
    if not os.path.exists(args.input):
        log.error(f"Cannot open '{args.input}' for reading")
        return

    log.info(f"Reading '{args.input}'")
    model = get_model(args.input)
    output = model.as_parfile(format=args.format)
    if args.out is None:
        # just output to STDOUT
        print(output)
    else:
        with open(args.out, "w") as outfile:
            outfile.write(output)
        log.info(f"Wrote to '{args.out}'")

    return
