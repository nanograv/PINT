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

        The following parameters are converted to TDB:
            1. Spin frequency, its derivatives and spin epoch
            2. Sky coordinates, proper motion and the position epoch
            3. DM, DM derivatives and DM epoch
            4. Keplerian binary parameters and FB1
        
        The following parameters are NOT converted although they are 
        in fact affected by the TCB to TDB conversion:
            1. Parallax
            2. TZRMJD and TZRFRQ
            3. DMX parameters
            4. Solar wind parameters
            5. Binary post-Keplerian parameters including Shapiro delay 
               parameters (except FB1)
            6. Jumps and DM Jumps
            7. FD parameters
            8. EQUADs
            9. Red noise parameters including FITWAVES, powerlaw red noise and 
            powerlaw DM noise parameters
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_par", help="Input par file name (TCB)")
    parser.add_argument("output_par", help="Output par file name (TDB)")

    args = parser.parse_args(argv)

    mb = ModelBuilder()
    model = mb(args.input_par, allow_tcb=True)
    model.write_parfile(args.output_par)

    log.info(f"Output written to {args.output_par}.")
