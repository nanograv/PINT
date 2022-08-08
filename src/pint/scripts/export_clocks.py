import argparse
import os
import sys

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint.observatory
import pint.observatory.topo_obs

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for export clock files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "observatories", help="Observatories to export", type=str, nargs="+"
    )
    parser.add_argument(
        "-o",
        "--out",
        default=os.path.abspath(os.curdir),
        type=str,
        dest="directory",
        help="Destination directory",
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
    if not os.path.isdir(args.directory):
        log.error(f"Destination {args.directory} is not a directory")
        sys.exit(1)

    for obsname in args.observatories:
        obs = pint.observatory.get_observatory(obsname)
        obs._load_clock_corrections()

    pint.observatory.topo_obs.export_all_clock_files(os.path.abspath(os.curdir))
