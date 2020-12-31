"""
A script for comparing the timing of PINT and TEMPO/Tempo2 for standard fitting 
cases. Recreates Tables 7, 8, and 9 from PINT paper.
Requires: 
    - TEMPO
    - TEMPO2 
*** THIS IS A PLACEHOLDER SCRIPT, final script will be in a notebook and cleaner/easier to work with. ***
"""
from pint import toa
from pint import models
from pint import fitter
import os
import sys
import subprocess
import timeit
import datetime
from astropy.table import Table
from astropy.io import ascii

MAXIT = 1  # number of iterations to time and average
# number of TOAs run for each test
ntoas_simple = [100, 1000, 10000, 100000]
ntoas_complex = [5012, 10024, 25060]
timfiles = [
    "NGC6440E_fake100.tim",
    "NGC6440E_fake1000.tim",
    "NGC6440E_fake10000.tim",
    "NGC6440E_fake100000.tim",
]  # timfiles for simple model and individual functions
timfiles2 = [
    "J1910+1256_NANOGrav_12yv4.tim",
    "J1910+1256_NANOGrav_12yv4_10k.tim",
    "J1910+1256_NANOGrav_12yv4_25k.tim",
]  # timfiles for complex model


def pintrun(parfile, timfile, ptime_arr, pickle, fitter):
    """ Runs and times pintempo 5 times and averages times, appending to a list. """
    gls = ""
    if fitter == "gls":
        gls = " --gls"
    usepickle = ""
    if pickle:
        usepickle = " --usepickle"
    total = 0
    for i in range(MAXIT):
        start = timeit.default_timer()
        subprocess.call(
            "pintempo" + usepickle + gls + " " + parfile + " " + timfile,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = timeit.default_timer()
        total = total + (end - start)
    ptime_arr.append(total / MAXIT)  # averages time


def temporun(parfile, timfile, ttime_arr, fitter):
    """ Runs and times TEMPO 5 times and averages times, appending to a list. """
    fit = ""
    if fitter == "gls":
        fit = " -G"
    total = 0
    for i in range(MAXIT):
        start = timeit.default_timer()
        subprocess.call(
            "tempo" + fit + " -f " + parfile + " " + timfile + "",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = timeit.default_timer()
        total = total + (end - start)
    ttime_arr.append(total / MAXIT)  # average time


def tempo2run(parfile, timfile, t2time_arr):
    """ Runs and times Tempo2 5 times and averages times, appending to a list. """
    total = 0
    for i in range(MAXIT):
        start = timeit.default_timer()
        subprocess.call(
            "tempo2 -nobs 100003 -f " + parfile + " " + timfile + "",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = timeit.default_timer()
        total = total + (end - start)
    t2time_arr.append(total / MAXIT)  # average time


if __name__ == "__main__":
    """
    # Generate simple, fake TOAs for the timing runs
    print("Making fake TOAs...")
    for num in ntoas_simple:
        call = (
            "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa "
            + str(num)
            + " NGC6440E.par NGC6440E_fake"
            + str(num)
            + ".tim"
        )
        if not os.path.exists("NGC6440E_fake" + str(num) + ".tim"):
            subprocess.call(
                call, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    ptimes_nopickle = []
    ptimes_pickle = []
    ttimes = []
    t2times = []

    for tim in timfiles:
        print("Running PINT fitting w/o pickling...")
        # run PINT w/o pickling and average time over 5 runs
        pintrun("NGC6440E.par", tim, ptimes_nopickle, pickle=False, fitter="wls")

        print("Running PINT w/ pickling...")
        # run PINT with pickling and average time over 5 runs
        subprocess.call(
            "pintempo --usepickle NGC6440E.par " + tim,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # create pickle file
        pintrun("NGC6440E.par", tim, ptimes_pickle, pickle=True, fitter="wls")
        print("Running TEMPO...")
        temporun("NGC6440E.par", tim, ttimes, fitter="wls")
        print("Running Tempo2...")
        tempo2run("NGC6440E.par", tim, t2times)

    # create table 7 in PINT paper
    simple_comparison = Table(
        (ntoas_simple, ttimes, t2times, ptimes_nopickle, ptimes_pickle),
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
    importtimes = []
    getTOAs_nopickle = []
    getTOAs_pickle = []
    fittimes = []

    # time import statements
    total = 0
    for i in range(MAXIT):
        start = timeit.default_timer()
        subprocess.call(
            "python3 import_statements.py",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = timeit.default_timer()
        total = total + (end - start)
    for i in range(len(ntoas_simple)):
        importtimes.append(total / MAXIT)

    # setup
    m = models.get_model("NGC6440E.par")
    use_planets = False
    if m.PLANET_SHAPIRO.value:
        use_planets = True
    model_ephem = "DE421"
    if m.EPHEM is not None:
        model_ephem = m.EPHEM.value

    for tim in timfiles:
        # no pickle time of get_TOAs
        print("timing get_TOAs w/o pickling...")
        total = 0
        for i in range(MAXIT):
            start = timeit.default_timer()
            toa.get_TOAs(tim, planets=use_planets, ephem=model_ephem, usepickle=False)
            end = timeit.default_timer()
            total = total + (end - start)
        getTOAs_nopickle.append(total / MAXIT)

        t = toa.get_TOAs(
            tim, planets=use_planets, ephem=model_ephem, usepickle=True
        )  # to use in timing fitter

        f = fitter.WLSFitter(t, m)

        print("timing fitter...")
        total = 0
        for i in range(MAXIT):
            start = timeit.default_timer()
            f.fit_toas()
            end = timeit.default_timer()
            total = total + (end - start)
        fittimes.append(total / MAXIT)

        # pickle time of get_TOAs
        print("timing get_TOAs w/ pickling...")
        total = 0
        for i in range(MAXIT):
            start = timeit.default_timer()
            toa.get_TOAs(tim, planets=use_planets, ephem=model_ephem, usepickle=True)
            end = timeit.default_timer()
            total = total + (end - start)
        getTOAs_pickle.append(total / MAXIT)

    # create table 8 in PINT paper
    function_comparison = Table(
        (ntoas_simple, importtimes, getTOAs_nopickle, getTOAs_pickle, fittimes),
        names=(
            "Number of TOAs",
            "Import Statements (sec)",
            "Load TOAs - No Pickle (sec)",
            "Load TOAs - Pickle (sec)",
            "WLS Fitting - No Pickle (sec)",
        ),
    )
    ascii.write(
        function_comparison,
        "function_tables.pdf",
        Writer=ascii.Latex,
        latexdict={"tabletype": "table*"},
        overwrite=True,
    )
    """
    # explore more complex model
    # use J1910+1256 with the following parameter additions to the par file (to ensure GLS fit with Tempo2):
    # - TNRedAmp -14.227505410948254
    # - TNRedGam 4.91353
    # - TNRedC 45

    # add needed params for GLS fitting
    subprocess.call(
        'echo "TNRedAmp -14.227505410948254" >> J1910+1256_NANOGrav_12yv4.gls.par',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.call(
        'echo "TNRedGam 4.91353" >> J1910+1256_NANOGrav_12yv4.gls.par',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.call(
        'echo "TNRedC 45" >> J1910+1256_NANOGrav_12yv4.gls.par',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # copy TOAs to create 2x the number of TOAs
    if "J1910+1256_NANOGrav_12yv4_10k.tim" in timfiles2:
        subprocess.call(
            "cat J1910+1256_NANOGrav_12yv4.tim > J1910+1256_NANOGrav_12yv4_10k.tim",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            "sed -n '6,6761p' J1910+1256_NANOGrav_12yv4.tim >> J1910+1256_NANOGrav_12yv4_10k.tim",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # copy TOAs to create 5x the number of TOAs
    if "J1910+1256_NANOGrav_12yv4_25k.tim" in timfiles2:
        subprocess.call(
            "cat J1910+1256_NANOGrav_12yv4.tim > J1910+1256_NANOGrav_12yv4_25k.tim",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for i in range(4):
            subprocess.call(
                "sed -n '6,6761p' J1910+1256_NANOGrav_12yv4.tim >> J1910+1256_NANOGrav_12yv4_25k.tim",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    ptimes_nopickle2 = []
    ptimes_pickle2 = []
    ttimes2 = []
    t2times2 = []

    for tim in timfiles2:
        # run PINT w/o pickling and average time over 5 runs
        print("Running PINT w/o pickling...")
        pintrun(
            "J1910+1256_NANOGrav_12yv4.gls.par",
            tim,
            ptimes_nopickle2,
            pickle=False,
            fitter="gls",
        )

        # run PINT with pickling and average time over 5 runs
        print("Running PINT w/ pickling...")
        subprocess.call(
            "pintempo --usepickle J1910+1256_NANOGrav_12yv4.gls.par " + tim,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # create pickle file
        pintrun(
            "J1910+1256_NANOGrav_12yv4.gls.par",
            tim,
            ptimes_pickle2,
            pickle=True,
            fitter="gls",
        )

        print("running TEMPO...")
        temporun("J1910+1256_NANOGrav_12yv4.gls.par", tim, ttimes2, fitter="gls")

        print("running Tempo2...")
        tempo2run("J1910+1256_NANOGrav_12yv4.gls.par", tim, t2times2)

    # create table 9 in PINT paper
    complex_comparison = Table(
        (ntoas_complex, ttimes2, t2times2, ptimes_nopickle2, ptimes_pickle2),
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

    # remove added params from par file for future use
    subprocess.call("sed -i '$d' J1910+1256_NANOGrav_12yv4.gls.par", shell=True)
    subprocess.call("sed -i '$d' J1910+1256_NANOGrav_12yv4.gls.par", shell=True)
    subprocess.call("sed -i '$d' J1910+1256_NANOGrav_12yv4.gls.par", shell=True)
    os.remove("J1910+1256_NANOGrav_12yv4_10k.tim")
    os.remove("J1910+1256_NANOGrav_12yv4_25k.tim")
