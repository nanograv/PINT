import tempo_utils as t1u
from pint.pulsar_mjd import longdouble2str
import argparse


def get_tempo_result(parfile, timfile):
    """This function is to get the results from tempo and write them to a file.

    Parameter
    ---------
    parfile : str
        The file to read parameters.
    timfile : str
        The file to read toas.

    Return
    ----------
    A file named as parfile name ends with '.tempo_test' stored residuals in the
    first column
    """
    t1_toas = t1u.read_toa_file(timfile)
    t1u.run_tempo(t1_toas, parfile)
    t1_resids = t1_toas.get_resids(units="phase")

    outfile = parfile + ".tempo_test"
    f = open(outfile, "w")
    outstr = "residuals_phase "
    outstr += "\n"
    f.write(outstr)
    for res in t1_resids:
        outstr = longdouble2str(res) + "\n"
        f.write(outstr)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to get tempo results.")
    parser.add_argument("parfile", help="par file to read model from.")
    parser.add_argument("timfile", help="tim file to read toas from.")

    args = parser.parse_args()
    get_tempo_result(args.parfile, args.timfile)
