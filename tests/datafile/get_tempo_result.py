"""This is a script for getting tempo/tempo2/libstempo result for the propose
of testing PINT
"""
import libstempo as lt
from pint.utils import longdouble2string
import argparse

def get_tempo_result(parfile, timfile):
    psr = lt.tempopulsar(parfile, timfile)
    residuals = psr.residuals()
    binary_delay = psr.binarydelay()
    outfile = parfile + '.tempo_test'
    f = open(outfile, 'w')
    outstr = '# This is a tempo result.\n'
    outstr += '# Parfile : ' + parfile + '\n'
    outstr += '# Timfile : ' + timfile + '\n'
    outstr += '# Column keys:\n'
    outstr += '# residuals  BinaryDelay\n'
    f.write(outstr)
    for res, bdelay in zip(residuals, binary_delay):
        outstr = longdouble2string(res) + ' ' +longdouble2string(bdelay) +'\n'
        f.write(outstr)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile",help="par file to read model from")
    parser.add_argument("timfile",help="Output TOA file name")
    args = parser.parse_args()
    get_tempo_result(args.parfile, args.timfile)
