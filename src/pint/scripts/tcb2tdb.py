from loguru import logger as log

import pint.logging
from pint.models.tcb_conversion import convert_tcb_to_tdb
from pint.models import get_model

import argparse

pint.logging.setup(level=pint.logging.script_level)

def main(argv):
    parser = argparse.ArgumentParser(
        description="PINT tool for converting TCB par files to TBD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_par", help="Input par file name (TCB)")
    parser.add_argument("output_par", help="Output par file name (TDB)")

    args = parser.parse_args(argv)

    welcome_message = """This script converts TCB par files to TDB.
    Please note that this conversion is exact and the timing model 
    should be re-fit to the TOAs. 

    The following parameters are converted to TCB:
        1. Spin frequency, its derivatives and spin epoch
        2. Sky coordinates, proper motion and the position epoch
        3. Keplerian binary parameters
    
    The following parameters are NOT converted although they are 
    in fact affected by the TCB to TDB conversion:
        1. Parallax
        2. TZRMJD and TZRFRQ
        2. DM, DM derivatives, DM epoch, DMX parameters
        3. Solar wind parameters
        4. Binary post-Keplerian parameters including Shapiro delay 
           parameters
        5. Jumps and DM Jumps
        6. FD parameters
        7. EQUADs
        8. Red noise parameters including FITWAVES, powerlaw red noise and 
           powerlaw DM noise parameters
    """
    log.info(welcome_message)

    model = get_model(args.input_par, allow_tcb=True)
    convert_tcb_to_tdb(model)
    model.write_parfile(args.output_par)

    log.info(f"Output written to {args.output_par}.")