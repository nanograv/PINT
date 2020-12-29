"""
A script for comparing the timing of PINT and TEMPO/Tempo2 for standard fitting 
cases.
Requires: 
    - TEMPO
    - TEMPO2 
"""
from pint import toa
import os
import sys
import subprocess
import timeit
import datetime
from astropy.table import Table
from astropy.io import ascii

MAXIT = 5  # number of iterations to time and average


def pintrun(parfile, timfile, ptime_arr, pickle, fitter):
    gls = ""
    if fitter == "gls":
        gls = " --gls"
    usepickle = ""
    if pickle:
        usepickle = " --usepickle"
    for i in range(MAXIT):
        subprocess.call(
            "time -o pinttimes.txt -a pintempo"
            + usepickle
            + gls
            + " "
            + parfile
            + " "
            + timfile,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    pinttime = datetime.timedelta()  # defaults to 0
    with open("pinttimes.txt") as f:
        for line in f:
            if "user" in line:
                vals = line.split()
                timestr = vals[2][:-7]
                t = datetime.datetime.strptime(
                    timestr, "%M:%S.%f"
                ).time()  # format string into datetime obj
                pinttime = pinttime + datetime.timedelta(
                    minutes=t.minute, seconds=t.second, microseconds=t.microsecond
                )
    os.remove("pinttimes.txt")
    ptime_arr.append(pinttime.total_seconds() / 5)


def temporun(parfile, timfile, ttime_arr, fitter):
    fit = ""
    if fitter == "gls":
        fit = " -G"
    for i in range(MAXIT):
        subprocess.call(
            "time -o tempotimes.txt -a tempo"
            + fit
            + " -f "
            + parfile
            + " "
            + timfile
            + "",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    tempotime = datetime.timedelta()  # defaults to 0
    with open("tempotimes.txt") as f:
        for line in f:
            if "user" in line:
                vals = line.split()
                timestr = vals[2][:-7]
                t = datetime.datetime.strptime(
                    timestr, "%M:%S.%f"
                ).time()  # format string into datetime obj
                tempotime = tempotime + datetime.timedelta(
                    minutes=t.minute, seconds=t.second, microseconds=t.microsecond
                )
    os.remove("tempotimes.txt")
    ttime_arr.append(tempotime.total_seconds() / 5)


def tempo2run(parfile, timfile, t2time_arr):
    for i in range(MAXIT):
        subprocess.call(
            "time -o tempo2times.txt -a tempo2 -nobs 100003 -f "
            + parfile
            + " "
            + timfile
            + "",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    tempo2time = datetime.timedelta()  # defaults to 0
    with open("tempo2times.txt") as f:
        for line in f:
            if "user" in line:
                vals = line.split()
                timestr = vals[2][:-7]
                t = datetime.datetime.strptime(
                    timestr, "%M:%S.%f"
                ).time()  # format string into time obj
                tempo2time = tempo2time + datetime.timedelta(
                    minutes=t.minute, seconds=t.second, microseconds=t.microsecond
                )
    os.remove("tempo2times.txt")
    t2time_arr.append(tempo2time.total_seconds() / 5)


if __name__ == "__main__":
    # Generate simple, fake TOAs for the timing runs
    make_fake_TOA1 = "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa 100 NGC6440E.par NGC6440E_fake100.tim"
    make_fake_TOA2 = "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa 1000 NGC6440E.par NGC6440E_fake1k.tim"
    make_fake_TOA3 = "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa 10000 NGC6440E.par NGC6440E_fake10k.tim"
    make_fake_TOA4 = "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa 100000 NGC6440E.par NGC6440E_fake100k.tim"
    # call operations on command line
    print("Making fake TOAs...")
    # subprocess.call(
    #    make_fake_TOA1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # )
    # subprocess.call(make_fake_TOA2, shell=True)
    # subprocess.call(make_fake_TOA3, shell=True)
    # subprocess.call(make_fake_TOA4, shell=True)

    print("Running PINT fitting w/o pickling (5 runs for each file pair)...")
    # run PINT w/o pickling and average time over 5 runs
    ptimes_nopickle = []
    pintrun(
        "NGC6440E.par",
        "NGC6440E_fake100.tim",
        ptimes_nopickle,
        pickle=False,
        fitter="wls",
    )

    # run PINT with pickling and average time over 5 runs
    subprocess.call(
        "pintempo --usepickle NGC6440E.par NGC6440E_fake100.tim",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # create pickle file
    ptimes_pickle = []
    pintrun(
        "NGC6440E.par", "NGC6440E_fake100.tim", ptimes_pickle, pickle=True, fitter="wls"
    )

    ttimes = []
    temporun("NGC6440E.par", "NGC6440E_fake100.tim", ttimes, fitter="wls")

    t2times = []
    tempo2run("NGC6440E.par", "NGC6440E_fake100.tim", t2times)

    # create table 7 in PINT paper
    ntoas = [100]
    simple_comparison = Table(
        (ntoas, ttimes, t2times, ptimes_nopickle, ptimes_pickle),
        names=(
            "Number of TOAs",
            "TEMPO (sec)",
            "Tempo2 (sec)",
            "PINT - No Pickle (sec)",
            "PINT - Pickle (sec)",
        ),
    )
    ascii.write(
        simple_comparison,
        "simple_tables.pdf",
        Writer=ascii.Latex,
        latexdict={"tabletype": "table*"},
        overwrite=True,
    )

    # time the individual PINT functions

    # explore more complex model
    # use J1910+1256 with the following parameter additions to the par file (to ensure GLS fit with Tempo2):
    # - TNRedAmp -14.227505410948254
    # - TNRedGam 4.91353
    # - TNRedC 45
    print("Running PINT fitting w/o pickling (5 runs for each file pair)...")
    # run PINT w/o pickling and average time over 5 runs
    ptimes_nopickle2 = []
    pintrun(
        "J1910+1256_NANOGrav_12yv4.gls.par",
        "J1910+1256_NANOGrav_12yv4.tim",
        ptimes_nopickle2,
        pickle=False,
        fitter="gls",
    )

    # run PINT with pickling and average time over 5 runs
    subprocess.call(
        "pintempo --usepickle J1910+1256_NANOGrav_12yv4.gls.par J1910+1256_NANOGrav_12yv4.tim",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # create pickle file

    ptimes_pickle2 = []
    pintrun(
        "J1910+1256_NANOGrav_12yv4.gls.par",
        "J1910+1256_NANOGrav_12yv4.tim",
        ptimes_pickle2,
        pickle=True,
        fitter="gls",
    )

    ttimes2 = []
    temporun(
        "J1910+1256_NANOGrav_12yv4.gls.par",
        "J1910+1256_NANOGrav_12yv4.tim",
        ttimes2,
        fitter="gls",
    )

    t2times2 = []
    tempo2run(
        "J1910+1256_NANOGrav_12yv4.gls.par", "J1910+1256_NANOGrav_12yv4.tim", t2times2
    )

    # create table 7 in PINT paper
    ntoas = [5012]

    complex_comparison = Table(
        (ntoas, ttimes2, t2times2, ptimes_nopickle2, ptimes_pickle2),
        names=(
            "Number of TOAs",
            "TEMPO (sec)",
            "Tempo2 (sec)",
            "PINT - No Pickle (sec)",
            "PINT - Pickle (sec)",
        ),
    )
    ascii.write(
        complex_comparison,
        "complex_tables.pdf",
        Writer=ascii.Latex,
        latexdict={"tabletype": "table*"},
        overwrite=True,
    )
