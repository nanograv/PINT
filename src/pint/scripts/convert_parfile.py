#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import argparse
import logging
import os

from pint.models import get_model
from pint.models.parameter import _parfile_formats

log = logging.getLogger(__name__)

__all__ = ["main"]

# log.setLevel("INFO")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for command-line parfile format conversions"
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
        "-o", "--out", help=("Output filename [default=stdout]"), default=None,
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        log.error(f'Cannot open "{args.input}" for reading')

    log.info(f'Reading "{args.input}"')
    model = get_model(args.input)
    output = model.as_parfile(format=args.format)
    if args.out is None:
        # just output to STDOUT
        print(output)
    else:
        with open(args.out, "w") as outfile:
            outfile.write(output)
        log.info(f'Wrote to "{args.out}"')

    return
