# Wideband TOA fitting


```python
import os

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
import matplotlib.pyplot as plt
import astropy.units as u
```

    WARNING: Using astropy version 3.2.3. To get most recent IERS data, upgrade to astropy >= 4.0 [pint]
    WARNING: Using astropy version 3.2.3. To get most recent IERS data, upgrade to astropy >= 4.0 [pint.erfautils]


## Setup your inputs


```python
model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim", ephem='de436')
```

    INFO: Parameter PBDOT's value will be scaled by 1e-12 [pint.models.parameter]
    INFO: Parameter PBDOT's value will be scaled by 1e-12 [pint.models.parameter]
    INFO: Applying clock corrections (include_GPS = True, include_BIPM = True) [pint.toa]
    INFO: Observatory gbt, loading clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/time.dat [pint.observatory.topo_obs]


    WARNING: More than one component made use of par file line 'DMEFAC -f Rcvr1_2_GASP   1.17100': [('ScaleDmError', 'DMEFAC1'), ('ScaleDmError', 'DMEFAC2'), ('ScaleDmError', 'DMEFAC3'), ('ScaleDmError', 'DMEFAC4')] [pint.models.timing_model]
    WARNING: More than one component made use of par file line 'DMJUMP -fe Rcvr1_2  -0.00022': [('DispersionJump', 'DMJUMP1'), ('DispersionJump', 'DMJUMP2')] [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'DMDATA                   1' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EFAC -f Rcvr1_2_GASP   0.913' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EFAC2 -f Rcvr1_2_GUPPI 1.063' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EFAC3 -f Rcvr_800_GASP 0.819' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EFAC4 -f Rcvr_800_GUPPI 0.816' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EQUAD -f Rcvr1_2_GASP   0.01860' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EQUAD2 -f Rcvr1_2_GUPPI 0.03258' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EQUAD3 -f Rcvr_800_GASP 0.00715' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'T2EQUAD4 -f Rcvr_800_GUPPI 0.26405' [pint.models.timing_model]


    INFO: Applying observatory clock corrections. [pint.observatory.topo_obs]
    INFO: Applying GPS to UTC clock correction (~few nanoseconds) [pint.observatory.topo_obs]
    INFO: Observatory gbt, loading GPS clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/gps2utc.clk [pint.observatory.topo_obs]
    INFO: Applying TT(TAI) to TT(BIPM) clock correction (~27 us) [pint.observatory.topo_obs]
    INFO: Observatory gbt, loading BIPM clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/tai2tt_bipm2015.clk [pint.observatory.topo_obs]
    INFO: Computing TDB columns. [pint.toa]
    INFO: Doing astropy mode TDB conversion [pint.observatory]
    INFO: Computing PosVels of observatories and Earth, using de436 [pint.toa]
    INFO: Set solar system ephemeris to link:
    	https://data.nanograv.org/static/data/ephem/de436.bsp [pint.solar_system_ephemerides]


## Setup the fitter like old time


```python
fitter = WidebandTOAFitter(toas, model)
```

## Run your fits like old time


```python
fitter.fit_toas()
```

    <class 'pint.residuals.CombinedResiduals'>





    1059.0350708947311679



## What are the difference?

### Concept of fitting different types of data together
#### Residuals are combined with TOA/time residuals and dm residuals


```python
type(fitter.resids)
```




    pint.residuals.CombinedResiduals



#### If we look into the resids attribute, it has two independent Residual objects.


```python
fitter.resids.residual_objs
```




    [<pint.residuals.Residuals at 0x7f67004cce48>,
     <pint.residuals.WidebandDMResiduals at 0x7f6700245f28>]



#### Each of them can be used independently

* Time residual


```python
time_resids = fitter.resids.residual_objs[0].time_resids
plt.errorbar(toas.get_mjds().value, time_resids.to_value(u.us), yerr=toas.get_errors().to_value(u.us), fmt='x')
plt.ylabel('us')
plt.xlabel('MJD')
```




    Text(0.5, 0, 'MJD')




![png](Wideband_TOA_walkthrough_files/Wideband_TOA_walkthrough_14_1.png)



```python
# Time RMS
print(fitter.resids.residual_objs[0].rms_weighted())
print(fitter.resids.residual_objs[0].chi2)
```

    0.19817223711663554 us
    252.6162534714601


* DM residual


```python
dm_resids = fitter.resids.residual_objs[1].resids
dm_error = fitter.resids.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt='x')
plt.ylabel('pc/cm^3')
plt.xlabel('MJD')
```




    Text(0.5, 0, 'MJD')




![png](Wideband_TOA_walkthrough_files/Wideband_TOA_walkthrough_17_1.png)



```python
# DM RMS
print(fitter.resids.residual_objs[1].rms_weighted())
print(fitter.resids.residual_objs[1].chi2)
```

    0.0006650610360444273 pc / cm3
    1037.2814700992433


#### However, in the combined residuals, one can access rms and chi2 as well


```python
print(fitter.resids.rms_weighted())
print(fitter.resids.chi2)
```

    0.0006650601239650586014
    1289.8977235707034


#### The initial residuals is also a combined residual object


```python
time_resids = fitter.resids_init.residual_objs[0].time_resids
plt.errorbar(toas.get_mjds().value, time_resids.to_value(u.us), yerr=toas.get_errors().to_value(u.us), fmt='x')
plt.ylabel('us')
plt.xlabel('MJD')
```




    Text(0.5, 0, 'MJD')




![png](Wideband_TOA_walkthrough_files/Wideband_TOA_walkthrough_22_1.png)



```python
dm_resids = fitter.resids_init.residual_objs[1].resids
dm_error = fitter.resids_init.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt='x')
plt.ylabel('pc/cm^3')
plt.xlabel('MJD')
```




    Text(0.5, 0, 'MJD')




![png](Wideband_TOA_walkthrough_files/Wideband_TOA_walkthrough_23_1.png)


#### Design Matrix are combined


```python
d_matrix = fitter.get_designmatrix()
```


```python
print("Number of TOAs:", toas.ntoas)
print("Number of DM measurments:", len(fitter.resids.residual_objs[1].dm_data))
print("Number of fit params:", len(fitter.get_fitparams()))
print("Shape of design matrix:", d_matrix.shape)
```

    Number of TOAs: 275
    Number of DM measurments: 275
    Number of fit params: 130
    Shape of design matrix: (550, 131)


#### Covariance Matrix are combined


```python
c_matrix = fitter.make_noise_covariancematrix()
```


```python
print("Shape of covariance matrix:", c_matrix.shape)
```

    Shape of covariance matrix: (550, 550)


### NOTE the matrix are PINTMatrix object right now, here are the difference

If you want to access the matrix data


```python
print(d_matrix.matrix)
```

    [[ 1.00000000e+00  1.10478428e-06 -8.32110225e+00 ...  4.35057784e+05
      -3.00359933e+13 -1.00000000e+00]
     [ 1.00000000e+00  1.10470023e-06 -8.32078558e+00 ...  4.35055879e+05
      -3.00357303e+13 -1.00000000e+00]
     [ 1.00000000e+00  1.10457498e-06 -8.32031362e+00 ...  4.35053039e+05
      -3.00353382e+13 -1.00000000e+00]
     ...
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
       0.00000000e+00  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
       0.00000000e+00  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
       0.00000000e+00  0.00000000e+00]]


PINT matrix has labels that marks all the element in the matrix. It has the label name, index of range of the matrix, and the unit.


```python
print("labels for dimension 0:", d_matrix.labels[0])
```

    labels for dimension 0: [('toa', (0, 275, Unit("s2"))), ('dm', (275, 550, Unit("pc / cm3")))]



```python

```
