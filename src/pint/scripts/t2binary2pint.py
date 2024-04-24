"""PINT-based tool for converting T2 par files to PINT."""

import argparse

from loguru import logger as log

import pint.logging
from pint.models.model_builder import ModelBuilder

pint.logging.setup(level="INFO")

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="""`t2binary2pint` converts par with binary models for
        Tempo2 to par files with binary models compatible with PINT.

        To guess the binary model, the model parameters in the par file are
        compared to all the binary models that PINT knows about. Then, the
        simplest binary model that contains all these parameters is chosen.

        The following parameters are optionally converted:
            1. KOM (AIU <--> DT96 convention)
            2. KIN (AIU <--> DT96 convention)

        The following parameters are optionally ignored:
            1. SINI (if the binary model is DDK)

        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_par", help="Input par file name (TCB)")
    parser.add_argument("output_par", help="Output par file name (TDB)")
    parser.add_argument(
        "--convert_komkin",
        type=bool,
        default=True,
        help="Whether to convert KOM/KIN parameters (True)",
    )
    parser.add_argument(
        "--drop_ddk_sini",
        type=bool,
        default=True,
        help="Whether to drop SINI if the model is DDK (True)",
    )

    args = parser.parse_args(argv)

    mb = ModelBuilder()

    model = mb(args.input_par, allow_T2=True, allow_tcb=True)
    model.write_parfile(args.output_par)
    print(f"Output written to {args.output_par}")

    log.info(f"Output written to {args.output_par}.")
