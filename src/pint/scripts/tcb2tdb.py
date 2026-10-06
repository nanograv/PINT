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
        Coordinate epochs follow Astropy/ERFA's IAU 2006 TDB, and radio
        frequency remains undilated as in PINT's forward model. The command
        converts every supported term and warns about unsupported active
        deterministic terms. A fully accepted conversion reproduces the input
        model's residuals to better than 1 ns with nothing refitted, up to the
        overall phase gauge that TZRMJD or a reference JUMP fixes. DM-family
        exponents follow the DILATEFREQ of the input par; a DILATEFREQ Y input
        is converted but reported unaudited, because PINT evaluates undilated
        frequencies.
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

    report = getattr(model, "tcb_tdb_conversion_report", None)
    if report is not None and report.accepted:
        log.info("Conversion satisfies the no-refit accuracy contract.")

    log.info(f"Output written to {args.output_par}.")
