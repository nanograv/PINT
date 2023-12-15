"""Generate LaTeX summary of a timing model and TOAs."""
from pint.models import get_model_and_toas
from pint.output.publish import publish
from pint.logging import setup as setup_log
import argparse


def main(argv=None):
    setup_log(level="WARNING")

    parser = argparse.ArgumentParser(
        description="Publication output for PINT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="TOA file name")
    parser.add_argument("--outfile", help="Output file", default=None)
    parser.add_argument(
        "--include_dmx",
        help="Include DMX parameters",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_noise",
        help="Include noise parameters",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_jumps", help="Include jumps", action="store_true", default=False
    )
    parser.add_argument(
        "--include_zeros",
        help="Include parameters equal to 0",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_fd", help="Include FD parameters", action="store_true", default=False
    )
    parser.add_argument(
        "--include_glitches",
        help="Include glitches",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_swx",
        help="Include SWX parameters",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_tzr",
        help="Include TZR parameters",
        action="store_true",
        default=False,
    )

    args = parser.parse_args(argv)

    model, toas = get_model_and_toas(args.parfile, args.timfile)

    output = publish(
        model,
        toas,
        include_dmx=args.include_dmx,
        include_noise=args.include_noise,
        include_jumps=args.include_jumps,
        include_zeros=args.include_zeros,
        include_fd=args.include_fd,
        include_glitches=args.include_glitches,
        include_swx=args.include_swx,
        include_tzr=args.include_tzr,
    )

    if args.outfile is None:
        print(output)
    else:
        with open(args.outfile, "w") as f:
            f.write(output)
