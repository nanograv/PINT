This directory contains a suite of benchmark codes for PINT. It only lives here and is not installed as part of
any PINT packages.

To run the codes, there are a couple of prerequisites, besides just a working PINT install.

# Get .tim file needed from here (it is large so is not made part of PINT):
curl -O https://data.nanograv.org/static/data/J0740+6620.cfr+19.tim

# Also, a couple of other packages are required:
# First, if you want to run `run_profile.py` you need the "dot" program,
# which is part of the graphviz package (try your package manager to install this)
# It also uses the python packages gprof2dot and py-cpuinfo
# Those should be installed if you did `pip install -r requirements_dev.txt`

# Then to run to the top level benchmark suite, just run
python high_level_benchmark.py

# To get useful output on an individual benchmarking script, do this to get
# a list of the top 100 calls by execution time, as well as a PDF showing a tree of all the execution times.
run_profile.py bench_MCMC.py

# The available benchmarks are (though more can be added!)
# bench_load_TOAs.py: Load a large number of TOAs
# bench_chisq_grid.py: Run a 3x3 grid of fits over M2, SINI for J0740+6620, using the GLS fitter
# bench_chisq_grid_WLSFitter.py: Run a 3x3 grid of fits over M2, SINI for J0740+6620, using the WLS fitter
# bench_MCMC.py: Run an MCMC fit of NGC6440E

# And example run of high_level_benchmark.py is below:

 % python high_level_benchmark.py 
python -m cProfile -o bench_load_TOAs_prof_summary bench_load_TOAs.py
python -m cProfile -o bench_chisq_grid_prof_summary bench_chisq_grid.py
python -m cProfile -o bench_chisq_grid_WLSFitter_prof_summary bench_chisq_grid_WLSFitter.py
python -m cProfile -o bench_MCMC_prof_summary bench_MCMC.py

Processor running this script: Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
Python version: 3.9.7
SciPy version: 1.7.1, AstroPy version: 4.3.1, NumPy version: 1.21.2
PINT version: 0.8.4+166.g0d23b53c

*******************************************************************
OUTPUT FOR BENCH_LOAD_TOAS.PY:
Total Time: 15.973 s
               Function Time(s)
   Construct TOA Object   0.030
  Construct TOAs Object   2.766
  toa.py:1248(__init__)   5.381
Apply Clock Corrections   5.350
           Compute TDBs   2.006
        Compute Posvels   1.083

*******************************************************************
OUTPUT FOR BENCH_CHISQ_GRID.PY:
Total Time: 181.281 s
        Function Time(s)
Get Designmatrix 123.870
   Update Resids  21.085
      Cho Factor   0.011
       Cho Solve   0.018
 Select TOA Mask  10.802

*******************************************************************
OUTPUT FOR BENCH_CHISQ_GRID_WLSFITTER.PY:
Total Time: 176.437 s
        Function Time(s)
Get Designmatrix 121.546
   Update Resids  21.527
      Cho Factor   0.004
       Cho Solve   0.005
             svd   0.145
 Select TOA Mask   8.598

*******************************************************************
OUTPUT FOR BENCH_MCMC.PY:
Total Time: 12.974 s
