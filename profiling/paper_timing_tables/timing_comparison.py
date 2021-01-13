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
import subprocess
import timeit
from tempfile import TemporaryDirectory
from shutil import copy
from astropy.table import Table
from astropy.io import ascii

MAXIT = 1  # number of iterations to time and average
# number of TOAs run for each test
ntoas_simple = [100, 1000, 10000, 100000]
ntoas_complex = [5012, 10024, 25060]
par_simple = "NGC6440E.par"
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
        subprocess.check_call(
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
        subprocess.check_call(
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
        subprocess.check_call(
            "tempo2 -nobs 100003 -f " + parfile + " " + timfile + "",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = timeit.default_timer()
        total = total + (end - start)
    t2time_arr.append(total / MAXIT)  # average time


if __name__ == "__main__":
    with TemporaryDirectory() as tempdir:
        # Generate simple, fake TOAs for the timing runs
        print("Making fake TOAs...")
        for num in ntoas_simple:
            call = (
                "zima --startMJD 53478 --duration 700 --freq 1400 2000 --ntoa "
                + str(num)
                + " "
                + par_simple
                + " "
                + tempdir
                + "/NGC6440E_fake"
                + str(num)
                + ".tim"
            )

            subprocess.check_call(
                call, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        ptimes_nopickle = []
        ptimes_pickle = []
        ttimes = []
        t2times = []

        for tim in timfiles:
            tim = tempdir + "/" + tim  # add temporary directory path to file
            print("Running PINT fitting w/o pickling...")
            # run PINT w/o pickling and average time over 5 runs
            pintrun(par_simple, tim, ptimes_nopickle, pickle=False, fitter="wls")

            print("Running PINT w/ pickling...")
            # run PINT with pickling and average time over 5 runs
            subprocess.check_call(
                "pintempo --usepickle " + par_simple + " " + tim,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )  # create pickle file
            pintrun(par_simple, tim, ptimes_pickle, pickle=True, fitter="wls")
            print("Running TEMPO...")
            temporun(par_simple, tim, ttimes, fitter="wls")
            print("Running Tempo2...")
            tempo2run(par_simple, tim, t2times)

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
            subprocess.check_call(
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
        m = models.get_model(par_simple)
        use_planets = False
        if m.PLANET_SHAPIRO.value:
            use_planets = True
        model_ephem = "DE421"
        if m.EPHEM is not None:
            model_ephem = m.EPHEM.value

        for tim in timfiles:
            tim = tempdir + "/" + tim  # add temporary directory path to file
            # no pickle time of get_TOAs
            print("timing get_TOAs w/o pickling...")
            total = 0
            for i in range(MAXIT):
                start = timeit.default_timer()
                toa.get_TOAs(
                    tim, planets=use_planets, ephem=model_ephem, usepickle=False
                )
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
                toa.get_TOAs(
                    tim, planets=use_planets, ephem=model_ephem, usepickle=True
                )
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

        # explore more complex model
        # use J1910+1256 with the following parameter additions to the par file (to ensure GLS fit with Tempo2):
        # - TNRedAmp -14.227505410948254
        # - TNRedGam 4.91353
        # - TNRedC 45

        # add needed params for GLS fitting
        """
        copy(
            "J1910+1256_NANOGrav_12yv4.gls.par", tempdir
        )  # copy file into temporary directory for modification
        with open(tempdir + "/J1910+1256_NANOGrav_12yv4.gls.par", "a") as f:
            print("TNRedAmp -14.227505410948254", file=f)
            print("TNRedGam 4.91353", file=f)
            print("TNRedC 45", file=f)
        """
        par = tempdir + "/" + "J1910+1256_NANOGrav_12yv4.gls.par"

        copy(
            "J1910+1256_NANOGrav_12yv4.tim", tempdir
        )  # copy file into temporary directory
        # copy TOAs to create 2x the number of TOAs
        with open("J1910+1256_NANOGrav_12yv4.tim") as original:
            with open(tempdir + "/J1910+1256_NANOGrav_12yv4_10k.tim", "w") as new:
                # copy entire file
                for line in original:
                    new.write(line)
                    if "MODE" in line or "FORMAT" in line:
                        pass
                    else:
                        new.write(line)  # copy TOAs to create 2x number of TOAs

        with open("J1910+1256_NANOGrav_12yv4.tim") as original:
            with open(tempdir + "/J1910+1256_NANOGrav_12yv4_25k.tim", "w") as new:
                # copy entire file
                for line in original:
                    new.write(line)
                    if "MODE" in line or "FORMAT" in line:
                        pass
                    else:
                        new.write(line)  # copy TOAs to create 5x number of TOAs
                        new.write(line)
                        new.write(line)
                        new.write(line)

        ptimes_nopickle2 = []
        ptimes_pickle2 = []
        ttimes2 = []
        t2times2 = []

        for tim in timfiles2:
            tim = tempdir + "/" + tim  # add temporary directory path to file
            # run PINT w/o pickling and average time over 5 runs
            print("Running PINT w/o pickling...")
            pintrun(par, tim, ptimes_nopickle2, pickle=False, fitter="gls")

            # run PINT with pickling and average time over 5 runs
            print("Running PINT w/ pickling...")
            subprocess.check_call(
                "pintempo --usepickle " + par + " " + tim,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )  # create pickle file
            pintrun(par, tim, ptimes_pickle2, pickle=True, fitter="gls")

            print("running TEMPO...")
            temporun(par, tim, ttimes2, fitter="gls")

            print("running Tempo2...")
            tempo2run(par, tim, t2times2)

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
