# Validation Example for PINT paper

A comparison between PINT result and Tempo/Tempo2 result. This example is presented in the PINT paper. But it can be used for other datasets. 

* Requirement
  * Data set: NANOGrav 11-year data J1600-3053
  * TEMPO and its python utils tempo_utils. Download from https://github.com/demorest/tempo_utils
  * TEMPO2 and its python utils tempo2_utils. Download from https://github.com/demorest/tempo_utils
  * TEMPO2 general2 plugins. 



```python
import pint
import sys
from pint import toa
from pint import models
from pint.fitter import GLSFitter
import os 
import matplotlib.pyplot as plt
import astropy.units as u
import tempo2_utils as t2u
import tempo_utils
import tempo2_utils
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import subprocess
import tempfile
from pint import ls
import astropy.constants as ct
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from astropy.time import Time
```

    WARNING: Using astropy version 3.2.3. To get most recent IERS data, upgrade to astropy >= 4.0 [pint]
    WARNING: Using astropy version 3.2.3. To get most recent IERS data, upgrade to astropy >= 4.0 [pint.erfautils]


### Print the PINT and TEMPO/TEMPO2 version


```python
print("PINT version: ", pint.__version__)
tempo_v = subprocess.check_output(["tempo", "-v"])
print("TEMPO version: ", tempo_v.decode("utf-8"))
#Not sure why tempo2_v = subprocess.check_output(["tempo2", "-v"]) does not work.
process = subprocess.Popen(['tempo2', '-v'], stdout=subprocess.PIPE)
tempo2_v = process.communicate()[0]
print("TEMPO2 version: ", tempo2_v.decode("utf-8"))
```

    PINT version:  0.7+432.g6854a9ec.dirty
    TEMPO version:   Tempo v 13.101 (2020-11-04 c5fbddf)
    
    TEMPO2 version:  2019.01.1
    


### Redefine the Tempo2_util function for larger number of observations


```python
_nobs = 30000
def newpar2(parfile,timfile):
    """
    Run tempo2, return new parfile (as list of lines).  input parfile
    can be either lines or a filename.
    """
    orig_dir = os.getcwd()
    try:
        temp_dir = tempfile.mkdtemp(prefix="tempo2")
        try:
            lines = open(parfile,'r').readlines()
        except:
            lines = parfile
        open("%s/pulsar.par" % temp_dir, 'w').writelines(lines)
        timpath = os.path.abspath(timfile)
        os.chdir(temp_dir)
        cmd = "tempo2 -nobs %d -newpar -f pulsar.par %s -norescale" % (_nobs, timpath)
        os.system(cmd + " > /dev/null")
        outparlines = open('new.par').readlines()
    finally:
        os.chdir(orig_dir)
    os.system("rm -rf %s" % temp_dir)
    for l in outparlines:
        if l.startswith('TRES'): rms = float(l.split()[1])
        elif l.startswith('CHI2R'): (foo, chi2r, ndof) = l.split()
    return float(chi2r)*float(ndof), int(ndof), rms, outparlines
```

### Get the data file for PSR J1600-3053. 

* Note
  * For other data set, one can change the cell below. 


```python
psr = "J1600-3053"
par_file = os.path.join('.', psr + "_NANOGrav_11yv1.gls.par")
tim_file = os.path.join('.', psr + "_NANOGrav_11yv1.tim")
```

## PINT run

### Load TOAs to PINT


```python
t = toa.get_TOAs(tim_file, ephem="DE436", bipm_version="BIPM2015")
```

    INFO: Applying clock corrections (include_GPS = True, include_BIPM = True) [pint.toa]
    INFO: Observatory gbt, loading clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/time.dat [pint.observatory.topo_obs]
    INFO: Applying observatory clock corrections. [pint.observatory.topo_obs]
    INFO: Applying GPS to UTC clock correction (~few nanoseconds) [pint.observatory.topo_obs]
    INFO: Observatory gbt, loading GPS clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/gps2utc.clk [pint.observatory.topo_obs]
    INFO: Applying TT(TAI) to TT(BIPM) clock correction (~27 us) [pint.observatory.topo_obs]
    INFO: Observatory gbt, loading BIPM clock file 
    	/home/luo/.local/lib/python3.6/site-packages/pint/datafiles/tai2tt_bipm2015.clk [pint.observatory.topo_obs]
    INFO: Computing TDB columns. [pint.toa]
    INFO: Doing astropy mode TDB conversion [pint.observatory]
    INFO: Computing PosVels of observatories and Earth, using DE436 [pint.toa]
    INFO: Set solar system ephemeris to link:
    	https://data.nanograv.org/static/data/ephem/de436.bsp [pint.solar_system_ephemerides]



```python
print("There are {} TOAs in the dataset.".format(t.ntoas))
```

    There are 12433 TOAs in the dataset.


### Load timing model from .par file


```python
m = models.get_model(par_file)
```

    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]
    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]


### Make the General Least Square fitter


```python
f = GLSFitter(model=m, toas=t)
```

### Fit TOAs for 9 iterations.


```python
chi2 = f.fit_toas(9)
print("Postfit Chi2: ", chi2)
print("Degree of freedom: ", f.resids.dof)
```

    Postfit Chi2:  12368.09539037636076
    Degree of freedom:  12307



### The weighted RMS value for pre-fit and post-fit residuals


```python
print(f.resids_init.rms_weighted())
print(f.resids.rms_weighted())
```

    0.9441707008147506 us
    0.9441138158055049 us


### Plot the pre-fit and post-fit residuals


```python
pint_prefit = f.resids_init.time_resids.to_value(u.us)
pint_postfit = f.resids.time_resids.to_value(u.us)

plt.figure(figsize=(8,5), dpi=150)
plt.subplot(2, 1, 1)
plt.errorbar(t.get_mjds().to_value(u.day), f.resids_init.time_resids.to_value(u.us), 
             yerr=t.get_errors().to_value(u.us), fmt='x')

plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (us)')
plt.title('PINT pre-fit residuals for PSR J1600-3053 NANOGrav 11-year data')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.errorbar(t.get_mjds().to_value(u.day), f.resids.time_resids.to_value(u.us), 
             yerr=t.get_errors().to_value(u.us), fmt='x')
plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (us)')
plt.title('PINT post-fit residuals for PSR J1600-3053 NANOGrav 11-year data')
plt.grid(True)
plt.tight_layout()
plt.savefig("J1600_PINT")
```


![png](paper_validation_example_files/paper_validation_example_20_0.png)


## TEMPO run

### Use tempo_utils to analysis the same data set.


```python
tempo_toa = tempo_utils.read_toa_file(tim_file)
tempo_chi2, ndof, rms_t, tempo_par = tempo_utils.run_tempo(tempo_toa ,par_file, get_output_par=True, 
                                                   gls=True)
```


```python
print("TEMPO postfit chi2: ", tempo_chi2)
print("TEMPO postfit weighted rms: ", rms_t)
```

    TEMPO postfit chi2:  12368.46
    TEMPO postfit weighted rms:  0.944


### Write the TEMPO postfit residuals to a new .par file, for comparison later


```python
# Write out the post fit tempo parfile.
tempo_parfile = open(psr + '_tempo.par', 'w')
for line in tempo_par:
    tempo_parfile.write(line)
tempo_parfile.close()
```

### Get the TEMPO residuals


```python
tempo_prefit = tempo_toa.get_prefit()
tempo_postfit = tempo_toa.get_resids()
mjds = tempo_toa.get_mjd()
freqs = tempo_toa.get_freq()
errs = tempo_toa.get_resid_err()
```

### Plot the PINT - TEMPO residual difference.


```python
tp_diff_pre = (pint_prefit - tempo_prefit) * u.us 
tp_diff_post = (pint_postfit - tempo_postfit) * u.us
```


```python
plt.figure(figsize=(8,5), dpi=150)
plt.subplot(2, 1, 1)
plt.plot(mjds, (tp_diff_pre - tp_diff_pre.mean()).to_value(u.ns), '+')
plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (ns)')
plt.title('PSR J1600-3053 prefit residual differences between PINT and TEMPO')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(mjds, (tp_diff_post - tp_diff_post.mean()).to_value(u.ns), '+')
plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (ns)')
plt.title('PSR J1600-3053 postfit residual differences between PINT and TEMPO')
plt.grid(True)
plt.tight_layout()
plt.savefig("J1600_PINT_tempo.eps")
```


![png](paper_validation_example_files/paper_validation_example_30_0.png)


### Compare the parameter between TEMPO and PINT

* Reported quantities
  * TEMPO value
  * TEMPO uncertainty 
  * Parameter units
  * TEMPO parameter value - PINT parameter value
  * TEMPO/PINT parameter absolute difference divided by TEMPO uncertainty 
  * PINT uncertainty divided by TEMPO uncertainty
  * If TEMPO provides the uncertainty value


```python
# Create the parameter compare table
tv = []
tu = []
tv_pv = []
tv_pv_tc = []
tc_pc = []
units = []
names = []
no_t_unc = []
tempo_new_model = models.get_model(psr + '_tempo.par')
for param in tempo_new_model.params:
    t_par = getattr(tempo_new_model, param)
    pint_par = getattr(f.model, param)
    tempoq = t_par.quantity 
    pintq = pint_par.quantity
    try:
        diffq =  tempoq - pintq
        if t_par.uncertainty_value != 0.0:
            diff_tcq = np.abs(diffq) / t_par.uncertainty
            uvsu = pint_par.uncertainty / t_par.uncertainty
            no_t_unc.append(False)
        else:
            diff_tcq = np.abs(diffq) / pint_par.uncertainty
            uvsu = t_par.uncertainty
            no_t_unc.append(True)
    except TypeError:
        continue
    uvsu = pint_par.uncertainty / t_par.uncertainty
    tv.append(tempoq.value)
    tu.append(t_par.uncertainty.value)
    tv_pv.append(diffq.value)
    tv_pv_tc.append(diff_tcq.value)
    tc_pc.append(uvsu)
    units.append(t_par.units)
    names.append(param)
    
compare_table = Table((names, tv, tu, units, tv_pv, tv_pv_tc, tc_pc, no_t_unc), names = ('name', 'Tempo Value', 'Tempo uncertainty', 'units', 
                                                                                      'Tempo_V-PINT_V', 
                                                                                      'Tempo_PINT_diff/unct', 
                                                                                      'PINT_unct/Tempo_unct', 
                                                                                      'no_t_unc')) 
compare_table.sort('Tempo_PINT_diff/unct')
compare_table = compare_table[::-1]
compare_table.write('parameter_compare.t.html', format='html', overwrite=True)
```

    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]
    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]


### Print the parameter difference in a table.

The table is sorted by relative difference in descending order. 


```python
compare_table
```




<i>Table length=125</i>
<table id="table140411937689552" class="table-striped table-bordered table-condensed">
<thead><tr><th>name</th><th>Tempo Value</th><th>Tempo uncertainty</th><th>units</th><th>Tempo_V-PINT_V</th><th>Tempo_PINT_diff/unct</th><th>PINT_unct/Tempo_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>ELONG</td><td>244.347677844079</td><td>5.9573e-09</td><td>deg</td><td>-5.921065165948036224e-10</td><td>0.09939175743957894053</td><td>0.9999766504295133643</td><td>False</td></tr>
<tr><td>ELAT</td><td>-10.0718390253651</td><td>3.36103e-08</td><td>deg</td><td>-3.1913434074201663115e-09</td><td>0.094951351443461269657</td><td>1.000072183713741608</td><td>False</td></tr>
<tr><td>PMELONG</td><td>0.4626</td><td>0.010399999999999999523</td><td>mas / yr</td><td>0.00071187905827979625073</td><td>0.068449909449980417264</td><td>1.0031591779004100928</td><td>False</td></tr>
<tr><td>F0</td><td>277.9377112429746148</td><td>5.186e-13</td><td>Hz</td><td>-1.471045507628332416e-14</td><td>0.028365705893334601157</td><td>1.0000736554074983135</td><td>False</td></tr>
<tr><td>PX</td><td>0.504</td><td>0.07349999999999999589</td><td>mas</td><td>-0.0020703029707025422113</td><td>0.028167387356497174122</td><td>0.99982582356450722116</td><td>False</td></tr>
<tr><td>ECC</td><td>0.0001737294</td><td>8.9000000000000002855e-09</td><td></td><td>-2.384406823461443503e-10</td><td>0.02679108790406116089</td><td>1.0022775207693099819</td><td>False</td></tr>
<tr><td>DMX_0010</td><td>0.00066927561</td><td>0.00020051850499999999489</td><td>pc / cm3</td><td>-5.0847948039543485257e-06</td><td>0.025358232168918019844</td><td>0.99999786016599179206</td><td>False</td></tr>
<tr><td>DMX_0001</td><td>0.0016432056</td><td>0.00022434462499999998828</td><td>pc / cm3</td><td>-5.328772561611662406e-06</td><td>0.023752619710018293281</td><td>1.0000068575953371397</td><td>False</td></tr>
<tr><td>DMX_0002</td><td>0.00136024872</td><td>0.00020941304000000001188</td><td>pc / cm3</td><td>-4.905837050632510035e-06</td><td>0.023426607295479354859</td><td>1.000010656028552436</td><td>False</td></tr>
<tr><td>OM</td><td>181.84956816578</td><td>0.01296546975</td><td>deg</td><td>-0.00026376195878985431165</td><td>0.020343417082119551561</td><td>0.9909562505170320566</td><td>False</td></tr>
<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>DMX_0045</td><td>3.64190777e-05</td><td>0.00020164094999999999935</td><td>pc / cm3</td><td>-1.0880461652431374893e-07</td><td>0.00053959583370497780225</td><td>1.0000006821705129667</td><td>False</td></tr>
<tr><td>DMX_0071</td><td>-0.000176912603</td><td>0.00019118353399999999634</td><td>pc / cm3</td><td>-9.349008259535601141e-08</td><td>0.0004890069800433546844</td><td>1.0000046190658971046</td><td>False</td></tr>
<tr><td>DMX_0075</td><td>2.00017094e-06</td><td>0.00019663653799999999744</td><td>pc / cm3</td><td>-9.082493292551379154e-08</td><td>0.00046189245319968859436</td><td>0.9999419961581833549</td><td>False</td></tr>
<tr><td>DMX_0094</td><td>0.000929849121</td><td>0.00019402737299999999105</td><td>pc / cm3</td><td>5.00029611804828078e-08</td><td>0.00025771086010880955765</td><td>0.9999940905421160764</td><td>False</td></tr>
<tr><td>DMX_0073</td><td>-0.000156953835</td><td>0.00019724444300000000259</td><td>pc / cm3</td><td>4.9749872529263657744e-08</td><td>0.00025222445698641892406</td><td>1.0000039385363455047</td><td>False</td></tr>
<tr><td>DMX_0017</td><td>0.000178762757</td><td>0.00021197504699999999088</td><td>pc / cm3</td><td>-2.7382927343715330118e-08</td><td>0.00012917995646777864055</td><td>0.9999889282854504957</td><td>False</td></tr>
<tr><td>DMX_0043</td><td>-0.000494848648</td><td>0.0001997188189999999947</td><td>pc / cm3</td><td>2.5596058013453541757e-08</td><td>0.00012816047151497297354</td><td>0.9999848366739377825</td><td>False</td></tr>
<tr><td>DMX_0083</td><td>8.70047706e-06</td><td>0.00020486178099999999887</td><td>pc / cm3</td><td>2.416989696640591326e-08</td><td>0.000117981484142256444004</td><td>1.0000060508320367525</td><td>False</td></tr>
<tr><td>DMX_0069</td><td>-0.000251368356</td><td>0.00019942850700000000919</td><td>pc / cm3</td><td>2.1310941707771303283e-08</td><td>0.000106860057412811609596</td><td>1.0000028555921218754</td><td>False</td></tr>
<tr><td>DMX_0067</td><td>-0.000377967984</td><td>0.00019749766400000001308</td><td>pc / cm3</td><td>-1.27852614923047724904e-08</td><td>6.473626691759135337e-05</td><td>0.999976952183460277</td><td>False</td></tr>
</table>



### If one wants the Latex output please use the cell below. 


```python
#ascii.write(compare_table, sys.stdout, Writer = ascii.Latex,
#            latexdict = {'tabletype': 'table*'})
```

### Check out the maximum DMX difference


```python
max_dmx = 0
max_dmx_index = 0
for ii, row in enumerate(compare_table):
    if row['name'].startswith('DMX_'):
        if row['Tempo_PINT_diff/unct'] > max_dmx:
            max_dmx = row['Tempo_PINT_diff/unct']
            max_dmx_index = ii

dmx_max = compare_table[max_dmx_index]['name']

compare_table[max_dmx_index]

```




<i>Row index=6</i>
<table id="table140411937689552">
<thead><tr><th>name</th><th>Tempo Value</th><th>Tempo uncertainty</th><th>units</th><th>Tempo_V-PINT_V</th><th>Tempo_PINT_diff/unct</th><th>PINT_unct/Tempo_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>DMX_0010</td><td>0.00066927561</td><td>0.00020051850499999999489</td><td>pc / cm3</td><td>-5.0847948039543485257e-06</td><td>0.025358232168918019844</td><td>0.99999786016599179206</td><td>False</td></tr>
</table>



### Output the table in the paper


```python
paper_params = ['F0', 'F1', 'FD1', 'FD2', 'JUMP1', 'PX', 
                'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'PB', 
                'A1', 'A1DOT', 'ECC', 'T0', 'OM', 'OMDOT', 'M2',
                'SINI', dmx_max]
# Get the table index of the parameters above
paper_param_index = []
for pp in paper_params:
    # We assume the parameter name are unique in the table
    idx = np.where(compare_table['name'] == pp)[0][0]
    paper_param_index.append(idx)
paper_param_index = np.array(paper_param_index)
compare_table[paper_param_index]
```




<i>Table length=20</i>
<table id="table140411939385416" class="table-striped table-bordered table-condensed">
<thead><tr><th>name</th><th>Tempo Value</th><th>Tempo uncertainty</th><th>units</th><th>Tempo_V-PINT_V</th><th>Tempo_PINT_diff/unct</th><th>PINT_unct/Tempo_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>F0</td><td>277.9377112429746148</td><td>5.186e-13</td><td>Hz</td><td>-1.471045507628332416e-14</td><td>0.028365705893334601157</td><td>1.0000736554074983135</td><td>False</td></tr>
<tr><td>F1</td><td>-7.338737472765e-16</td><td>4.619148184227e-21</td><td>Hz / s</td><td>6.3513794158537125015e-23</td><td>0.013750109679403142984</td><td>1.0001125509037762049</td><td>False</td></tr>
<tr><td>FD1</td><td>3.98314325e-05</td><td>1.6566479199999999207e-06</td><td>s</td><td>-2.5078110490474835037e-09</td><td>0.0015137863747461100493</td><td>0.99999722077930985886</td><td>False</td></tr>
<tr><td>FD2</td><td>-1.47296057e-05</td><td>1.1922595999999999884e-06</td><td>s</td><td>1.3481392263201306481e-09</td><td>0.0011307430246903700001</td><td>0.9999985886156963488</td><td>False</td></tr>
<tr><td>JUMP1</td><td>-8.789e-06</td><td>1.2999999999999999941e-07</td><td>s</td><td>-4.662519179964242028e-10</td><td>0.0035865532153571094524</td><td>1.0037094614491930411</td><td>False</td></tr>
<tr><td>PX</td><td>0.504</td><td>0.07349999999999999589</td><td>mas</td><td>-0.0020703029707025422113</td><td>0.028167387356497174122</td><td>0.99982582356450722116</td><td>False</td></tr>
<tr><td>ELONG</td><td>244.347677844079</td><td>5.9573e-09</td><td>deg</td><td>-5.921065165948036224e-10</td><td>0.09939175743957894053</td><td>0.9999766504295133643</td><td>False</td></tr>
<tr><td>ELAT</td><td>-10.0718390253651</td><td>3.36103e-08</td><td>deg</td><td>-3.1913434074201663115e-09</td><td>0.094951351443461269657</td><td>1.000072183713741608</td><td>False</td></tr>
<tr><td>PMELONG</td><td>0.4626</td><td>0.010399999999999999523</td><td>mas / yr</td><td>0.00071187905827979625073</td><td>0.068449909449980417264</td><td>1.0031591779004100928</td><td>False</td></tr>
<tr><td>PMELAT</td><td>-7.1555</td><td>0.058200000000000001732</td><td>mas / yr</td><td>-0.00050443897788454705733</td><td>0.008667336389768850319</td><td>0.9992173055526453185</td><td>False</td></tr>
<tr><td>PB</td><td>14.34846572550302</td><td>2.1222661e-06</td><td>d</td><td>-3.4563402588790037573e-08</td><td>0.016286083346847993084</td><td>1.0000724303266271833</td><td>False</td></tr>
<tr><td>A1</td><td>8.801653122</td><td>8.1100000000000004906e-07</td><td>ls</td><td>1.4914297352675021102e-08</td><td>0.018390009066183748282</td><td>0.98441828361417216264</td><td>False</td></tr>
<tr><td>A1DOT</td><td>-4e-15</td><td>6.260000000000000155e-16</td><td>ls / s</td><td>8.913933368296104875e-18</td><td>0.0142395101729969730114</td><td>0.99986819306556440345</td><td>False</td></tr>
<tr><td>ECC</td><td>0.0001737294</td><td>8.9000000000000002855e-09</td><td></td><td>-2.384406823461443503e-10</td><td>0.02679108790406116089</td><td>1.0022775207693099819</td><td>False</td></tr>
<tr><td>T0</td><td>55878.2618980451000000</td><td>0.0005167676</td><td>d</td><td>-1.0512816950025705154e-05</td><td>0.020343413460955572977</td><td>0.9909421696656756166</td><td>False</td></tr>
<tr><td>OM</td><td>181.84956816578</td><td>0.01296546975</td><td>deg</td><td>-0.00026376195878985431165</td><td>0.020343417082119551561</td><td>0.9909562505170320566</td><td>False</td></tr>
<tr><td>OMDOT</td><td>0.0052209</td><td>0.0013554</td><td>deg / yr</td><td>-2.2104443267939444245e-05</td><td>0.016308427968082812635</td><td>1.0000991630450569873</td><td>False</td></tr>
<tr><td>M2</td><td>0.271894</td><td>0.089418999999999998485</td><td>solMass</td><td>-0.0016414501218400268101</td><td>0.01835683827642924787</td><td>0.978663730015661093</td><td>False</td></tr>
<tr><td>SINI</td><td>0.906285</td><td>0.03399300000000000238</td><td></td><td>0.00054383117444134487783</td><td>0.015998328315869291688</td><td>0.9838897673347273276</td><td>False</td></tr>
<tr><td>DMX_0010</td><td>0.00066927561</td><td>0.00020051850499999999489</td><td>pc / cm3</td><td>-5.0847948039543485257e-06</td><td>0.025358232168918019844</td><td>0.99999786016599179206</td><td>False</td></tr>
</table>



## TEMPO2 run

Before TEMPO2 run, the .par file has to be modified for a more accurate TEMPO2 vs PINT comparison. We save the modified .par file in a file named "[PSR name]_tempo2.par". In this case, "J1600-3053_tempo2.par"

* Modified parameters 
  * ECL IERS2010   ----> ECL IERS 2003   (TEMPO2 use IERS 2003 Obliquity angle as default)
  * T2CMETHOD TEMPO  ----> # T2CMETHOD TEMPO (Make TEMPO2 ues the new precession and nutation model IAU 2000)


```python
tempo2_par = "J1600-3053_tempo2.par"
```


```python
less J1600-3053_tempo2.par
```

### PINT refit using the modified tempo2-style parfile


```python
m_t2 = models.get_model(tempo2_par)
```

    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]
    INFO: Parameter A1DOT's value will be scaled by 1e-12 [pint.models.parameter]



```python
f_t2 = GLSFitter(toas=t, model=m_t2)
f_t2.fit_toas()
```




    12368.094237187552177



### Tempo2 fit


```python
tempo2_chi2, ndof, rms_t2, tempo2_new_par = newpar2(tempo2_par, tim_file)
print("TEMPO2 chi2: ", tempo2_chi2)
print("TEMPO2 rms: ", rms_t2)
```

    TEMPO2 chi2:  12265.156200000001
    TEMPO2 rms:  0.944


### Get TEMPO2 residuals, toa value, observing frequencies, and data error


```python
tempo2_result = t2u.general2(tempo2_par, tim_file, ['sat', 'pre', 'post', 'freq', 'err'])
# TEMPO2's residual unit is second
tp2_diff_pre = f_t2.resids_init.time_resids - tempo2_result['pre'] * u.s
tp2_diff_post = f_t2.resids.time_resids - tempo2_result['post'] * u.s
```

### Plot the TEMPO2 - PINT residual difference


```python
plt.figure(figsize=(8,5), dpi=150)
plt.subplot(2, 1, 1)
plt.plot(mjds, (tp2_diff_pre - tp2_diff_pre.mean()).to_value(u.ns), '+')
plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (ns)')
plt.title('PSR J1600-3053 prefit residual differences between PINT and TEMPO2')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(mjds, (tp2_diff_post - tp2_diff_post.mean()).to_value(u.ns), '+')
plt.xlabel('MJD (day)')
plt.ylabel('Time Residuals (ns)')
plt.title('PSR J1600-3053 postfit residual differences between PINT and TEMPO2')
plt.grid(True)
plt.tight_layout()
plt.savefig("J1600_PINT_tempo2")
```


![png](paper_validation_example_files/paper_validation_example_52_0.png)


### Write out the TEMPO2 postfit parameter to a new file

* Note, since the ECL parameter is hard coded in tempo2, we will have to add it manually 


```python
# Write out the post fit tempo parfile.
tempo2_parfile = open(psr + '_new_tempo2.2.par', 'w')
for line in tempo2_new_par:
    tempo2_parfile.write(line)
tempo2_parfile.write("ECL IERS2003")
tempo2_parfile.close()
```

### Compare the parameter between TEMPO2 and PINT

* Reported quantities
  * TEMPO2 value
  * TEMPO2 uncertainty 
  * Parameter units
  * TEMPO2 parameter value - PINT parameter value
  * TEMPO2/PINT parameter absolute difference divided by TEMPO2 uncertainty 
  * PINT uncertainty divided by TEMPO2 uncertainty
  * If TEMPO2 provides the uncertainty value


```python
# Create the parameter compare table
tv = []
t2_unc = []
tv_pv = []
tv_pv_tc = []
tc_pc = []
units = []
names = []
no_t2_unc = []
tempo2_new_model = models.get_model(psr + '_new_tempo2.2.par')
for param in tempo2_new_model.params:
    t2_par = getattr(tempo2_new_model, param)
    pint2_par = getattr(f_t2.model, param)
    tempo2q = t2_par.quantity 
    pint2q = pint2_par.quantity
    try:
        diff2q =  tempo2q - pint2q
        if t2_par.uncertainty_value != 0.0:
            diff_tcq = np.abs(diff2q) / t2_par.uncertainty
            uvsu = pint2_par.uncertainty / t2_par.uncertainty
            no_t2_unc.append(False)
        else:
            diff_tcq = np.abs(diff2q) / pint2_par.uncertainty
            uvsu = t2_par.uncertainty
            no_t2_unc.append(True)
    except TypeError:
        continue
    uvsu = pint2_par.uncertainty / t2_par.uncertainty
    tv.append(tempo2q.value)
    t2_unc.append(t2_par.uncertainty.value)
    tv_pv.append(diff2q.value)
    tv_pv_tc.append(diff_tcq.value)
    tc_pc.append(uvsu)
    units.append(t2_par.units)
    names.append(param)
    
compare_table2 = Table((names, tv, t2_unc,units, tv_pv, tv_pv_tc, tc_pc, no_t2_unc), names = ('name', 'Tempo2 Value', 'T2 unc','units', 
                                                                                      'Tempo2_V-PINT_V', 
                                                                                      'Tempo2_PINT_diff/unct', 
                                                                                      'PINT_unct/Tempo2_unct', 
                                                                                      'no_t_unc')) 
compare_table2.sort('Tempo2_PINT_diff/unct')
compare_table2 = compare_table2[::-1]
compare_table2.write('parameter_compare.t2.html', format='html', overwrite=True)
```

    WARNING: EPHVER 5 does nothing in PINT [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'NE_SW          0' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'NE_SW2 0.000' [pint.models.timing_model]
    /home/luo/.local/lib/python3.6/site-packages/astropy/units/quantity.py:464: RuntimeWarning: divide by zero encountered in true_divide
      result = super().__array_ufunc__(function, method, *arrays, **kwargs)


### Print the parameter difference in a table.
The table is sorted by relative difference in descending order. 


```python
compare_table2
```




<i>Table length=125</i>
<table id="table140412177358184" class="table-striped table-bordered table-condensed">
<thead><tr><th>name</th><th>Tempo2 Value</th><th>T2 unc</th><th>units</th><th>Tempo2_V-PINT_V</th><th>Tempo2_PINT_diff/unct</th><th>PINT_unct/Tempo2_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>ECC</td><td>0.00017372966157521168</td><td>8.922286680669999241e-09</td><td></td><td>4.168033894912624715e-11</td><td>0.0046714861829564476026</td><td>1.0000400789683185909</td><td>False</td></tr>
<tr><td>DMX_0098</td><td>0.0013394613122489417</td><td>0.00019579968831114546654</td><td>pc / cm3</td><td>-5.162393215032545085e-07</td><td>0.0026365686582855950293</td><td>0.99999926235860314705</td><td>False</td></tr>
<tr><td>DMX_0070</td><td>-0.00023747963906517973</td><td>0.00019767137320477682749</td><td>pc / cm3</td><td>-4.6318680163804021657e-07</td><td>0.0023432163905605278911</td><td>1.0000006066661308868</td><td>False</td></tr>
<tr><td>DMX_0097</td><td>0.0013928330661987446</td><td>0.00019620100461426303326</td><td>pc / cm3</td><td>-4.3591375636898264945e-07</td><td>0.0022217712759728318155</td><td>0.99999985479541497746</td><td>False</td></tr>
<tr><td>DMX_0055</td><td>-0.0005307704904403621</td><td>0.00019675128861832102923</td><td>pc / cm3</td><td>-3.936735762570617997e-07</td><td>0.0020008691125817808752</td><td>1.0000000155376389532</td><td>False</td></tr>
<tr><td>DMX_0063</td><td>-0.00048410571072825574</td><td>0.00019894769104906708185</td><td>pc / cm3</td><td>-3.8388090987831295642e-07</td><td>0.0019295569999032320674</td><td>1.0000001737671666557</td><td>False</td></tr>
<tr><td>DMX_0079</td><td>0.00018976795294000216</td><td>0.00019490725481464179483</td><td>pc / cm3</td><td>-3.6413058978400727507e-07</td><td>0.0018682249161545991592</td><td>1.0000001869460202197</td><td>False</td></tr>
<tr><td>DMX_0010</td><td>0.00067403356955979</td><td>0.00020051850482404336064</td><td>pc / cm3</td><td>-3.734867366063333513e-07</td><td>0.001862604835070314456</td><td>0.9999998791435175116</td><td>False</td></tr>
<tr><td>F1</td><td>-7.3387383041227678664e-16</td><td>4.619148404392432094e-21</td><td>Hz / s</td><td>-8.212906306513322778e-24</td><td>0.001778013085421442844</td><td>0.999998198306420979</td><td>False</td></tr>
<tr><td>DMX_0086</td><td>0.00029525346690830644</td><td>0.0001961188165133768578</td><td>pc / cm3</td><td>-3.4086588203348670498e-07</td><td>0.0017380580206093430157</td><td>1.0000003760250906204</td><td>False</td></tr>
<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>DMX_0024</td><td>-6.464357906175583e-05</td><td>0.00019594538945981657802</td><td>pc / cm3</td><td>-3.0733074786039684713e-08</td><td>0.00015684510296856083583</td><td>1.000000040794174927</td><td>False</td></tr>
<tr><td>DMX_0092</td><td>0.0013207295138539894</td><td>0.00019585216459019454951</td><td>pc / cm3</td><td>-2.987272381023420298e-08</td><td>0.00015252690146540150845</td><td>1.0000000598805378615</td><td>False</td></tr>
<tr><td>DMX_0058</td><td>-0.0005377581468744793</td><td>0.00019927530538964258904</td><td>pc / cm3</td><td>-2.7240830902737663e-08</td><td>0.00013669948140073714471</td><td>1.0000003494124634074</td><td>False</td></tr>
<tr><td>DMX_0089</td><td>0.0007495614446295846</td><td>0.00021586616414944812654</td><td>pc / cm3</td><td>2.664672352724588994e-08</td><td>0.00012344094607063047083</td><td>1.0000000318049040438</td><td>False</td></tr>
<tr><td>DMX_0032</td><td>-6.265675469663684e-05</td><td>0.00019561483985536690729</td><td>pc / cm3</td><td>2.0624768732425491705e-08</td><td>0.00010543560369793503542</td><td>1.0000001069330806125</td><td>False</td></tr>
<tr><td>DMX_0040</td><td>-0.0005242449385393532</td><td>0.00020212647115737782458</td><td>pc / cm3</td><td>-1.3087848262614831807e-08</td><td>6.4750787898654266965e-05</td><td>0.99999999750656676234</td><td>False</td></tr>
<tr><td>DMX_0001</td><td>0.0016484372168232325</td><td>0.00022434462780433157077</td><td>pc / cm3</td><td>-1.1670286121810355406e-08</td><td>5.2019458794390748143e-05</td><td>1.0000004809807354622</td><td>False</td></tr>
<tr><td>DMX_0027</td><td>-0.00018288082535181414</td><td>0.00019391445756469536201</td><td>pc / cm3</td><td>-9.576702027013850663e-09</td><td>4.938621981715206116e-05</td><td>1.0000001643637170812</td><td>False</td></tr>
<tr><td>DMX_0083</td><td>8.544780315309648e-06</td><td>0.00020486177918444288125</td><td>pc / cm3</td><td>-3.8797037898631337458e-09</td><td>1.8938153350558018293e-05</td><td>1.0000001513452474455</td><td>False</td></tr>
<tr><td>DMX_0044</td><td>-0.0003390023662491028</td><td>0.00021062295971768858391</td><td>pc / cm3</td><td>1.0720102164903794195e-09</td><td>5.0897120519399367266e-06</td><td>1.0000001883055240626</td><td>False</td></tr>
</table>



### If one wants to get the latex version, please use the line below.


```python
#ascii.write(compare_table2, sys.stdout, Writer = ascii.Latex,
#            latexdict = {'tabletype': 'table*'})
```

### Check out the maximum DMX difference


```python
max_dmx = 0
max_dmx_index = 0
for ii, row in enumerate(compare_table2):
    if row['name'].startswith('DMX_'):
        if row['Tempo2_PINT_diff/unct'] > max_dmx:
            max_dmx = row['Tempo2_PINT_diff/unct']
            max_dmx_index = ii

dmx_max2 = compare_table2[max_dmx_index]['name']

compare_table2[max_dmx_index]
```




<i>Row index=1</i>
<table id="table140412177358184">
<thead><tr><th>name</th><th>Tempo2 Value</th><th>T2 unc</th><th>units</th><th>Tempo2_V-PINT_V</th><th>Tempo2_PINT_diff/unct</th><th>PINT_unct/Tempo2_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>DMX_0098</td><td>0.0013394613122489417</td><td>0.00019579968831114546654</td><td>pc / cm3</td><td>-5.162393215032545085e-07</td><td>0.0026365686582855950293</td><td>0.99999926235860314705</td><td>False</td></tr>
</table>



### Output the table in the paper


```python
paper_params = ['F0', 'F1', 'FD1', 'FD2', 'JUMP1', 'PX', 
                'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'PB', 
                'A1', 'A1DOT', 'ECC', 'T0', 'OM', 'OMDOT', 'M2',
                'SINI', dmx_max]
# Get the table index of the parameters above
paper_param_index = []
for pp in paper_params:
    # We assume the parameter name are unique in the table
    idx = np.where(compare_table2['name'] == pp)[0][0]
    paper_param_index.append(idx)
paper_param_index = np.array(paper_param_index)
compare_table2[paper_param_index]
```




<i>Table length=20</i>
<table id="table140412181099184" class="table-striped table-bordered table-condensed">
<thead><tr><th>name</th><th>Tempo2 Value</th><th>T2 unc</th><th>units</th><th>Tempo2_V-PINT_V</th><th>Tempo2_PINT_diff/unct</th><th>PINT_unct/Tempo2_unct</th><th>no_t_unc</th></tr></thead>
<thead><tr><th>str8</th><th>str32</th><th>float128</th><th>object</th><th>float128</th><th>float128</th><th>float128</th><th>bool</th></tr></thead>
<tr><td>F0</td><td>277.93771124297462788</td><td>5.1859268946902080184e-13</td><td>Hz</td><td>-6.6613381477509392425e-16</td><td>0.0012845029023782387781</td><td>1.0000082417695045875</td><td>False</td></tr>
<tr><td>F1</td><td>-7.3387383041227678664e-16</td><td>4.619148404392432094e-21</td><td>Hz / s</td><td>-8.212906306513322778e-24</td><td>0.001778013085421442844</td><td>0.999998198306420979</td><td>False</td></tr>
<tr><td>FD1</td><td>3.983282287426775e-05</td><td>1.6566478062738200598e-06</td><td>s</td><td>-1.6031694728325007922e-09</td><td>0.0009677189483251698093</td><td>1.0000000032094340519</td><td>False</td></tr>
<tr><td>FD2</td><td>-1.4729805752137882e-05</td><td>1.1922596055992699934e-06</td><td>s</td><td>1.4162333940147552079e-09</td><td>0.001187856560235392954</td><td>1.0000000136320319477</td><td>False</td></tr>
<tr><td>JUMP1</td><td>-8.7887456483184e-06</td><td>0.0</td><td>s</td><td>-5.0316350075314342574e-11</td><td>0.0003856129036950526755</td><td>inf</td><td>True</td></tr>
<tr><td>PX</td><td>0.5061242012322064</td><td>0.07348886965486496614</td><td>mas</td><td>1.9718059246720542887e-05</td><td>0.00026831354651833601603</td><td>1.0000000164253699531</td><td>False</td></tr>
<tr><td>ELONG</td><td>244.34767784255382</td><td>5.95727548431e-09</td><td>deg</td><td>9.322320693172514439e-12</td><td>0.0015648631186731613756</td><td>1.0000013810109530107</td><td>False</td></tr>
<tr><td>ELAT</td><td>-10.071839047043065</td><td>3.361025894297e-08</td><td>deg</td><td>-1.5125678487493132707e-11</td><td>0.00045003159639913618783</td><td>0.9999926235884047247</td><td>False</td></tr>
<tr><td>PMELONG</td><td>0.4619096015625491</td><td>0.010433361011620021289</td><td>mas / yr</td><td>7.3610413870994761965e-06</td><td>0.00070552925168612602887</td><td>1.0000025739354863052</td><td>False</td></tr>
<tr><td>PMELAT</td><td>-7.155145674275822</td><td>0.058156247552489513664</td><td>mas / yr</td><td>-7.1059018702079868035e-05</td><td>0.0012218638872452151304</td><td>0.99999263990213360653</td><td>False</td></tr>
<tr><td>PB</td><td>14.348465754661366786</td><td>2.12226632065849e-06</td><td>d</td><td>-1.9218603731011030256e-09</td><td>0.0009055698403133468854</td><td>0.9999977436618808823</td><td>False</td></tr>
<tr><td>A1</td><td>8.80165312286463</td><td>8.114047416773300209e-07</td><td>ls</td><td>-8.203180357213568641e-10</td><td>0.001010985015968234816</td><td>0.9999811897467071331</td><td>False</td></tr>
<tr><td>A1DOT</td><td>-4.008979189463729e-15</td><td>6.2586911221949290846e-16</td><td>ls / s</td><td>-1.0370977784914106416e-18</td><td>0.0016570521827057340097</td><td>1.0000003677433377813</td><td>False</td></tr>
<tr><td>ECC</td><td>0.00017372966157521168</td><td>8.922286680669999241e-09</td><td></td><td>4.168033894912624715e-11</td><td>0.0046714861829564476026</td><td>1.0000400789683185909</td><td>False</td></tr>
<tr><td>T0</td><td>55878.2618994738495070</td><td>0.00051676746764245482</td><td>d</td><td>4.890245969835227413e-07</td><td>0.00094631459525597103496</td><td>1.000008278658846269</td><td>False</td></tr>
<tr><td>OM</td><td>181.84960401549451478</td><td>0.01296564244572522874</td><td>deg</td><td>1.2275557313340401677e-05</td><td>0.00094677586280251403394</td><td>1.0000088591412276617</td><td>False</td></tr>
<tr><td>OMDOT</td><td>0.0052395528517645540778</td><td>0.00135543635075636363</td><td>deg / yr</td><td>-1.2270890653582061121e-06</td><td>0.0009053092494335592534</td><td>0.99999775911651708066</td><td>False</td></tr>
<tr><td>M2</td><td>0.2717633814383356</td><td>0.08941866471282471085</td><td>solMass</td><td>0.000104187428365654088935</td><td>0.0011651642159974146158</td><td>1.0000236859702187342</td><td>False</td></tr>
<tr><td>SINI</td><td>0.9064200568225846</td><td>0.03399283139781983376</td><td></td><td>-4.2613468352326044908e-05</td><td>0.0012536016153999781763</td><td>1.0000254830135359985</td><td>False</td></tr>
<tr><td>DMX_0010</td><td>0.00067403356955979</td><td>0.00020051850482404336064</td><td>pc / cm3</td><td>-3.734867366063333513e-07</td><td>0.001862604835070314456</td><td>0.9999998791435175116</td><td>False</td></tr>
</table>



### The residual difference between PINT and TEMPO2 is at the level of ~1ns. 

* We believe the discrepancy is mainly from the solar system geometric delay. 
* We will use the tempo2 postfit parameters, which are wrote out to `J1600-3053_new_tempo2.2.par`


```python
tempo2_result2 = t2u.general2('J1600-3053_new_tempo2.2.par', tim_file, ['sat', 'pre', 'post', 'freq', 'err'])
m_t22 = models.get_model('J1600-3053_new_tempo2.2.par')
f_t22 = GLSFitter(toas=t, model=m_t22)
f_t22.fit_toas()
tp2_diff_pre2 = f_t22.resids_init.time_resids - tempo2_result2['pre'] * u.s
tp2_diff_post2 = f_t22.resids.time_resids - tempo2_result2['post'] * u.s
```

    WARNING: EPHVER 5 does nothing in PINT [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'NE_SW          0' [pint.models.timing_model]
    WARNING: Unrecognized parfile line 'NE_SW2 0.000' [pint.models.timing_model]



```python
PINT_solar = m_t22.solar_system_geometric_delay(t)
tempo2_solar = t2u.general2('J1600-3053_new_tempo2.2.par', tim_file, ['roemer'])
```


```python
diff_solar = PINT_solar + tempo2_solar['roemer'] * u.s
plt.figure(figsize=(8,2.5), dpi=150)
plt.plot(mjds, (tp2_diff_post2 - tp2_diff_post2.mean()).to_value(u.ns), '+')
plt.plot(mjds, (diff_solar - diff_solar.mean()).to_value(u.ns, equivalencies=[(ls, u.s)]), 'x')

plt.xlabel('MJD (day)')
plt.ylabel('Discrepancies (ns)')
#plt.title('PSR J1600-3053 postfit residual differences between PINT and TEMPO2')
plt.grid(True)
plt.legend(['Postfit Residual Differences', 'Solar System Geometric Delay Difference'],
           loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)
plt.tight_layout()
plt.savefig("solar_geo")
```


![png](paper_validation_example_files/paper_validation_example_68_0.png)



```python

```
