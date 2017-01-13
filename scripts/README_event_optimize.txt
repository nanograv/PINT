The code event_optimize.py reads in Fermi photon events, along with 
a par file and a pulse profile template, and optimizes the timing model
using an MCMC sampling process.  The parameters to fit and their
priors are determined by reading the par file.  It can use photon weights,
if available, or compute them based on a simple heuristic computation, if 
desired.  There are many options to control the behavior.

An example run is shown below, using sample files that are included in
the examples subdirectory of the PINT distro.

% python event_optimize.py J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits PSRJ0030+0451_psrcat.par templateJ0030.3gauss --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=100 --nsteps=500

