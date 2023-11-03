import argparse
import os

from astropy import units as u

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

from pint.models import get_model
from pint.models.parameter import _parfile_formats
import pint.binaryconvert

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
        "-b",
        "--binary",
        help="Binary model for output",
        choices=pint.binaryconvert.binary_types,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--out",
        help=("Output filename [default=stdout]"),
        default=None,
    )
    parser.add_argument(
        "--nharms",
        default=7,
        type=int,
        help="Number of harmonics (convert to ELL1H only); note that default value for tempo2 is 4",
    )
    parser.add_argument(
        "--usestigma",
        action="store_true",
        help="Use STIGMA instead of H4? (convert to ELL1H only)",
    )
    parser.add_argument(
        "--kom",
        type=float,
        default=0,
        help="KOM (longitude of ascending node) in deg (convert to DDK only)",
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
    if hasattr(model, "BINARY") and args.binary is not None:
        log.info(f"Converting from {model.BINARY.value} to {args.binary}")
        if args.binary == "ELL1H":
            model = pint.binaryconvert.convert_binary(
                model, args.binary, NHARMS=args.nharms, useSTIGMA=args.usestigma
            )
        elif args.binary == "DDK":
            model = pint.binaryconvert.convert_binary(
                model, args.binary, KOM=args.kom * u.deg
            )
        else:
            model = pint.binaryconvert.convert_binary(model, args.binary)
    output = model.as_parfile(format=args.format)
    if args.out is None:
        # just output to STDOUT
        print(output)
    else:
        with open(args.out, "w") as outfile:
            outfile.write(output)
        log.info(f"Wrote to '{args.out}'")

    return
