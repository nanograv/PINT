This directory contains a suite of benchmark codes for PINT. It only lives here and is not installed as part of
any PINT packages.

To run the codes, there are a couple of prerequisites, besides just a working PINT install.

# Get .tim file needed from here (it is large so is not made part of PINT):
curl -O https://data.nanograv.org/static/data/J0740+6620.cfr+19.tim

# Also, a couple of other packages are required:
# First, you need the "dot" program, which is part of the graphviz package (try your package manager to install this)
# Then use pip to install the following:
pip install gprof2dot py-cpuinfo

# Then to run to the top level benchmark suite, just run
python high_level_benchmark.py

# To get useful output on an individual benchmarking script, do this to get
# a list of the top 100 calls by execution time, as well as a PDF shwoing a tree of all the execution times.
run_profile.py bench_MCMC.py

# The available benchmarks are (though more can be added!)
# bench_load_TOAs.py: Load a large number of TOAs
# bench_chisq_grid.py: Run a 3x3 grid of fits over M2, SINI for J0740+6620, using the GLS fitter
# bench_chisq_grid_WLSFitter.py: Run a 3x3 grid of fits over M2, SINI for J0740+6620, using the WLS fitter
# bench_MCMC.py: Run an MCMC fit of NGC6440E

# And example run of high_level_benchmark.py is below:

