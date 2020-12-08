.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Unit
     - Description
     - Default Value
     - Aliases
     - Host Components
   * - A0
     - floatParameter
     - s
     - DD model aberration parameter A0
     - 0.0
     -
     - BinaryDD, BinaryDDK
   * - A1
     - floatParameter
     - ls
     - Projected semi-major axis, a*sin(i)
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - A1DOT
     - floatParameter
     - ls / s
     - Derivative of projected semi-major axis, da*sin(i)/dt
     - None
     - XDOT
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - B0
     - floatParameter
     - s
     - DD model aberration parameter B0
     - 0.0
     -
     - BinaryDD, BinaryDDK
   * - CORRECT_TROPOSPHERE
     - boolParameter
     - None
     - Enable Troposphere Delay Model
     - True
     -
     - TroposphereDelay
   * - DECJ
     - AngleParameter
     - deg
     - Declination (J2000)
     - None
     - DEC
     - AstrometryEquatorial
   * - DM
     - floatParameter
     - pc / cm3
     - Dispersion measure
     - 0.0
     -
     - DispersionDM
   * - DM1
     - prefixParameter
     - pc / (cm3 yr)
     - 1'th time derivative of the dispersion measure
     - 0.0
     -
     - DispersionDM
   * - DMEFAC1
     - maskParameter
     -
     - A multiplication factor on the measured DM uncertainties,
     - None
     - DMEFAC
     - ScaleDmError
   * - DMEPOCH
     - MJDParameter
     - d
     - Epoch of DM measurement
     - None
     -
     - DispersionDM
   * - DMEQUAD1
     - maskParameter
     - pc / cm3
     - An error term added in quadrature to the scaled (by EFAC) TOA uncertainty.
     - None
     - DMEQUAD
     - ScaleDmError
   * - DMJUMP1
     - maskParameter
     - pc / cm3
     - DM value offset.
     - None
     - DMJUMP
     - DispersionJump
   * - DMX
     - floatParameter
     - pc / cm3
     - Dispersion measure
     - 0.0
     -
     - DispersionDMX
   * - DMXR1_0001
     - prefixParameter
     - d
     - Beginning of DMX interval
     - None
     -
     - DispersionDMX
   * - DMXR2_0001
     - prefixParameter
     - d
     - End of DMX interval
     - None
     -
     - DispersionDMX
   * - DMX_0001
     - prefixParameter
     - pc / cm3
     - Dispersion measure
     - 0.0
     -
     - DispersionDMX
   * - DR
     - floatParameter
     -
     - Relativistic deformation of the orbit
     - 0.0
     -
     - BinaryDD, BinaryDDK
   * - DTH
     - floatParameter
     -
     - Relativistic deformation of the orbit
     - 0.0
     -
     - BinaryDD, BinaryDDK
   * - ECC
     - floatParameter
     -
     - Eccentricity
     - None
     - E
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - ECL
     - strParameter
     - None
     - None
     - IERS2010
     -
     - AstrometryEcliptic
   * - ECORR1
     - maskParameter
     - us
     - An error term added that correlated all TOAs in an observing epoch.
     - None
     - TNECORR1, ECORR
     - EcorrNoise
   * - EDOT
     - floatParameter
     - 1 / s
     - Eccentricity derivitve respect to time
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - EFAC1
     - maskParameter
     -
     - A multiplication factor on the measured TOA uncertainties,
     - None
     - T2EFAC1, TNEF1, EFAC
     - ScaleToaError
   * - ELAT
     - AngleParameter
     - deg
     - Ecliptic latitude
     - None
     - BETA
     - AstrometryEcliptic
   * - ELONG
     - AngleParameter
     - deg
     - Ecliptic longitude
     - None
     - LAMBDA
     - AstrometryEcliptic
   * - EPS1
     - floatParameter
     -
     - First Laplace-Lagrange parameter, ECC x sin(OM) for ELL1 model
     - None
     -
     - BinaryELL1, BinaryELL1H
   * - EPS1DOT
     - floatParameter
     - 1e-12 / s
     - First derivative of first Laplace-Lagrange parameter
     - None
     -
     - BinaryELL1, BinaryELL1H
   * - EPS2
     - floatParameter
     -
     - Second Laplace-Lagrange parameter, ECC x cos(OM) for ELL1 model
     - None
     -
     - BinaryELL1, BinaryELL1H
   * - EPS2DOT
     - floatParameter
     - 1e-12 / s
     - Second derivative of first Laplace-Lagrange parameter
     - None
     -
     - BinaryELL1, BinaryELL1H
   * - EQUAD1
     - maskParameter
     - us
     - An error term added in quadrature to the scaled (by EFAC) TOA uncertainty.
     - None
     - T2EQUAD1, EQUAD
     - ScaleToaError
   * - F0
     - floatParameter
     - Hz
     - Spin-frequency
     - 0.0
     -
     - Spindown
   * - F1
     - prefixParameter
     - Hz / s
     - Spin-frequency 1 derivative
     - 0.0
     -
     - Spindown
   * - FB0
     - prefixParameter
     - 1 / s
     - 0th time derivative of frequency of orbit
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - FD1
     - prefixParameter
     - s
     - None
     - 0.0
     -
     - FD
   * - GAMMA
     - floatParameter
     - s
     - Time dilation & gravitational redshift
     - 0.0
     -
     - BinaryBT, BinaryDD, BinaryDDK
   * - GLEP_1
     - prefixParameter
     - d
     - Epoch of glitch 1
     - None
     -
     - Glitch
   * - GLF0D_1
     - prefixParameter
     - Hz
     - Decaying frequency change for glitch 1
     - 0.0
     -
     - Glitch
   * - GLF0_1
     - prefixParameter
     - Hz
     - Permanent frequency change for glitch 1
     - 0.0
     -
     - Glitch
   * - GLF1_1
     - prefixParameter
     - Hz / s
     - Permanent frequency-derivative change for glitch 1
     - 0.0
     -
     - Glitch
   * - GLF2_1
     - prefixParameter
     - Hz / s2
     - Permanent second frequency-derivative change for glitch 1
     - 0.0
     -
     - Glitch
   * - GLPH_1
     - prefixParameter
     -
     - Phase change for glitch 1
     - 0.0
     -
     - Glitch
   * - GLTD_1
     - prefixParameter
     - d
     - Decay time constant for glitch 1
     - 0.0
     -
     - Glitch
   * - H3
     - floatParameter
     - s
     - Shapiro delay parameter H3 as in Freire and Wex 2010 Eq(20)
     - None
     -
     - BinaryELL1H
   * - H4
     - floatParameter
     - s
     - Shapiro delay parameter H4 as in Freire and Wex 2010 Eq(21)
     - None
     -
     - BinaryELL1H
   * - IFUNC1
     - prefixParameter
     - s
     - Interpolation Components (MJD+delay)
     - None
     -
     - IFunc
   * - JUMP1
     - maskParameter
     - s
     - None
     - None
     - JUMP
     - PhaseJump
   * - K96
     - boolParameter
     - None
     - Flag for Kopeikin binary model proper motion
     - None
     -
     - BinaryDDK
   * - KIN
     - floatParameter
     - deg
     - Inclination angle
     - 0.0
     -
     - BinaryDDK
   * - KOM
     - floatParameter
     - deg
     - The longitude of the ascending node
     - 0.0
     -
     - BinaryDDK
   * - M2
     - floatParameter
     - solMass
     - Mass of companian in the unit Sun mass
     - None
     -
     - BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - NE_SW
     - floatParameter
     - 1 / cm3
     - Solar Wind Parameter
     - 0.0
     - NE1AU, SOLARN0
     - SolarWindDispersion
   * - NHARMS
     - floatParameter
     -
     - Number of harmonics for ELL1H shapiro delay.
     - 3.0
     -
     - BinaryELL1H
   * - OM
     - floatParameter
     - deg
     - Longitude of periastron
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - OMDOT
     - floatParameter
     - deg / yr
     - Longitude of periastron
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - PB
     - floatParameter
     - d
     - Orbital period
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - PBDOT
     - floatParameter
     -
     - Orbital period derivitve respect to time
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - PEPOCH
     - MJDParameter
     - d
     - Reference epoch for spin-down
     - None
     -
     - Spindown
   * - PLANET_SHAPIRO
     - boolParameter
     - None
     - None
     - False
     -
     - SolarSystemShapiro
   * - PMDEC
     - floatParameter
     - mas / yr
     - Proper motion in DEC
     - 0.0
     -
     - AstrometryEquatorial
   * - PMELAT
     - floatParameter
     - mas / yr
     - Proper motion in ecliptic latitude
     - 0.0
     - PMBETA
     - AstrometryEcliptic
   * - PMELONG
     - floatParameter
     - mas / yr
     - Proper motion in ecliptic longitude
     - 0.0
     - PMLAMBDA
     - AstrometryEcliptic
   * - PMRA
     - floatParameter
     - mas / yr
     - Proper motion in RA
     - 0.0
     -
     - AstrometryEquatorial
   * - POSEPOCH
     - MJDParameter
     - d
     - Reference epoch for position
     - None
     -
     - AstrometryEquatorial, AstrometryEcliptic
   * - PX
     - floatParameter
     - mas
     - Parallax
     - 0.0
     -
     - AstrometryEquatorial, AstrometryEcliptic
   * - RAJ
     - AngleParameter
     - hourangle
     - Right ascension (J2000)
     - None
     - RA
     - AstrometryEquatorial
   * - RNAMP
     - floatParameter
     -
     - Amplitude of powerlaw red noise.
     - None
     -
     - PLRedNoise
   * - RNIDX
     - floatParameter
     -
     - Spectral index of powerlaw red noise.
     - None
     -
     - PLRedNoise
   * - SIFUNC
     - floatParameter
     -
     - Type of interpolation
     - None
     -
     - IFunc
   * - SINI
     - floatParameter
     -
     - Sine of inclination angle
     - None
     -
     - BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - STIGMA
     - floatParameter
     -
     - Shapiro delay parameter STIGMA as in Freire and Wex 2010 Eq(12)
     - None
     -
     - BinaryELL1H
   * - SWM
     - floatParameter
     -
     - Solar Wind Model
     - 0.0
     -
     - SolarWindDispersion
   * - T0
     - MJDParameter
     - d
     - Epoch of periastron passage
     - None
     -
     - BinaryBT, BinaryDD, BinaryDDK, BinaryELL1, BinaryELL1H
   * - TASC
     - MJDParameter
     - d
     - Epoch of ascending node
     - None
     -
     - BinaryELL1, BinaryELL1H
   * - TNEQ1
     - maskParameter
     - dex(s)
     - An error term added in quadrature to the scaled (by EFAC) TOA uncertainty in  the unit of log10(second).
     - None
     - TNEQ
     - ScaleToaError
   * - TZRFRQ
     - floatParameter
     - MHz
     - The frequency of the zero phase mearsured.
     - None
     -
     - AbsPhase
   * - TZRMJD
     - MJDParameter
     - d
     - Epoch of the zero phase.
     - None
     -
     - AbsPhase
   * - TZRSITE
     - strParameter
     - None
     - None
     - None
     -
     - AbsPhase
   * - WAVE1
     - prefixParameter
     - s
     - Wave components
     - None
     -
     - Wave
   * - WAVEEPOCH
     - MJDParameter
     - d
     - Reference epoch for wave solution
     - None
     -
     - Wave
   * - WAVE_OM
     - floatParameter
     - 1 / d
     - Base frequency of wave solution
     - None
     -
     - Wave
