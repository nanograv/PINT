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

    model = get_model(args.input_par, allow_tcb=True)
    convert_tcb_to_tdb(model)
    model.write_parfile(args.output_par)

    log.info(f"Output written to {args.output_par}")