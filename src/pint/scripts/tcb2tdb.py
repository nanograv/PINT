"""PINT-based tool for converting TCB par files to TDB."""

import argparse

from loguru import logger as log

import pint.logging
from pint.models.model_builder import ModelBuilder

pint.logging.setup(level="INFO")

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="""`tcb2tdb` converts TCB par files to TDB.
        Please note that this conversion is not exact and the timing model 
        should be re-fit to the TOAs. 
       
        The following parameters are NOT converted although they are 
        in fact affected by the TCB to TDB conversion:
            1. TZRMJD and TZRFRQ
            2. DM Jumps (the wideband kind)
            3. FD parameters and FD jumps
            4. EQUADs and ECORRs
            5. GP Red noise parameters and GP DM noise parameters
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_par", help="Input par file name (TCB)")
    parser.add_argument("output_par", help="Output par file name (TDB)")
    parser.add_argument(
        "--allow_T2",
        action="store_true",
        help="Guess the underlying binary model when T2 is given",
    )

    args = parser.parse_args(argv)

    mb = ModelBuilder()
    model = mb(args.input_par, allow_tcb=True, allow_T2=args.allow_T2)
    model.write_parfile(args.output_par)

    log.info(f"Output written to {args.output_par}.")
