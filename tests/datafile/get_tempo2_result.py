"""This is a script for getting tempo/tempo2/libstempo result for the propose
of testing PINT
"""
# from pint.utils import longdouble2str

try:
    import tempo2_utils as t2u

    has_tempo2_utils = True
except:
    has_tempo2_utils = False
import argparse


def get_tempo2_result(parfile, timfile, general2=None):
    """This function is to get the results from tempo and write them to a file.
    Parameter
    ---------
    parfile : str
        The file to read parameters.
    timfile : str
        The file to read toas.
    general2 : list/ None
        The values required from tempo2 general2 plugin.
    Return
    ----------
    A file named as parfile name ends with '.tempo2_test' stored residuals in the
    first column, binary delay in the second column and general2 results if
    general2 are provided.
    """
    if not has_tempo2_utils:
        raise ImportError(
            "To get tempo2 general2 results, tempo2_utils are"
            " required. See page"
            " https://github.com/demorest/tempo_utils"
        )
    residuals = t2u.general2(parfile, timfile, ["pre"])["pre"]
    outfile = parfile + ".tempo2_test"
    f = open(outfile, "w")
    outstr = "residuals "

    if general2 is not None and general2 != []:
        tempo2_vals = t2u.general2(parfile, timfile, general2)
        for keys in general2:
            outstr += keys + " "

    outstr += "\n"
    f.write(outstr)
    for ii in range(len(residuals)):
        outstr = str(residuals[ii]) + " "
        if general2 is not None:
            for keys in general2:
                outstr += str(tempo2_vals[keys][ii]) + " "
        outstr += "\n"
        f.write(outstr)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to get tempo results.")
    parser.add_argument("parfile", help="par file to read model from.")
    parser.add_argument("timfile", help="tim file to read toas from.")
    parser.add_argument(
        "--general2", help="Use tempo2 general2 plugin", nargs="+", type=str
    )
    args = parser.parse_args()
    get_tempo2_result(args.parfile, args.timfile, general2=args.general2)
