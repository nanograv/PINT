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


    INFO: IERS B Table appears to be old. Attempting to re-download. [pint.erfautils]


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

### Plot the PINT - TEMPO residual difference.

#### Get the TEMPO residuals


```python
tempo_prefit = tempo_toa.get_prefit()
tempo_postfit = tempo_toa.get_resids()
mjds = tempo_toa.get_mjd()
freqs = tempo_toa.get_freq()
errs = tempo_toa.get_resid_err()
```


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


![png](paper_validation_example_files/paper_validation_example_29_0.png)


### Compare the parameter between TEMPO and PINT

####  Reported quantities
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


```python
compare_table.show_in_notebook()
```




<i>Table length=125</i>
<table id="table140671879831112-30846" class="table-striped table-bordered table-condensed">
<thead><tr><th>idx</th><th>name</th><th>Tempo Value</th><th>Tempo uncertainty</th><th>units</th><th>Tempo_V-PINT_V</th><th>Tempo_PINT_diff/unct</th><th>PINT_unct/Tempo_unct</th><th>no_t_unc</th></tr></thead>
<tr><td>0</td><td>ELONG</td><td>244.347677844079</td><td>5.9573e-09</td><td>deg</td><td>-5.921065165948036224e-10</td><td>0.09939175743957894053</td><td>0.9999766504295133643</td><td>False</td></tr>
<tr><td>1</td><td>ELAT</td><td>-10.0718390253651</td><td>3.36103e-08</td><td>deg</td><td>-3.1913434074201663115e-09</td><td>0.094951351443461269657</td><td>1.000072183713741608</td><td>False</td></tr>
<tr><td>2</td><td>PMELONG</td><td>0.4626</td><td>0.010399999999999999523</td><td>mas / yr</td><td>0.00071187905827979625073</td><td>0.068449909449980417264</td><td>1.0031591779004100928</td><td>False</td></tr>
<tr><td>3</td><td>F0</td><td>277.9377112429746148</td><td>5.186e-13</td><td>Hz</td><td>-1.471045507628332416e-14</td><td>0.028365705893334601157</td><td>1.0000736554074983135</td><td>False</td></tr>
<tr><td>4</td><td>PX</td><td>0.504</td><td>0.07349999999999999589</td><td>mas</td><td>-0.0020703029707025422113</td><td>0.028167387356497174122</td><td>0.99982582356450722116</td><td>False</td></tr>
<tr><td>5</td><td>ECC</td><td>0.0001737294</td><td>8.9000000000000002855e-09</td><td></td><td>-2.384406823461443503e-10</td><td>0.02679108790406116089</td><td>1.0022775207693099819</td><td>False</td></tr>
<tr><td>6</td><td>DMX_0010</td><td>0.00066927561</td><td>0.00020051850499999999489</td><td>pc / cm3</td><td>-5.0847948039543485257e-06</td><td>0.025358232168918019844</td><td>0.99999786016599179206</td><td>False</td></tr>
<tr><td>7</td><td>DMX_0001</td><td>0.0016432056</td><td>0.00022434462499999998828</td><td>pc / cm3</td><td>-5.328772561611662406e-06</td><td>0.023752619710018293281</td><td>1.0000068575953371397</td><td>False</td></tr>
<tr><td>8</td><td>DMX_0002</td><td>0.00136024872</td><td>0.00020941304000000001188</td><td>pc / cm3</td><td>-4.905837050632510035e-06</td><td>0.023426607295479354859</td><td>1.000010656028552436</td><td>False</td></tr>
<tr><td>9</td><td>OM</td><td>181.84956816578</td><td>0.01296546975</td><td>deg</td><td>-0.00026376195878985431165</td><td>0.020343417082119551561</td><td>0.9909562505170320566</td><td>False</td></tr>
<tr><td>10</td><td>T0</td><td>55878.2618980451000000</td><td>0.0005167676</td><td>d</td><td>-1.0512816950025705154e-05</td><td>0.020343413460955572977</td><td>0.9909421696656756166</td><td>False</td></tr>
<tr><td>11</td><td>DMX_0033</td><td>-0.000478304275</td><td>0.00020741596399999998892</td><td>pc / cm3</td><td>3.9118573704016680256e-06</td><td>0.018859962825241688433</td><td>1.0000479309056897748</td><td>False</td></tr>
<tr><td>12</td><td>A1</td><td>8.801653122</td><td>8.1100000000000004906e-07</td><td>ls</td><td>1.4914297352675021102e-08</td><td>0.018390009066183748282</td><td>0.98441828361417216264</td><td>False</td></tr>
<tr><td>13</td><td>M2</td><td>0.271894</td><td>0.089418999999999998485</td><td>solMass</td><td>-0.0016414501218400268101</td><td>0.01835683827642924787</td><td>0.978663730015661093</td><td>False</td></tr>
<tr><td>14</td><td>DMX_0055</td><td>-0.000533923668</td><td>0.00019675128900000001127</td><td>pc / cm3</td><td>-3.5169591964483550364e-06</td><td>0.01787515199683573433</td><td>1.0000020080096945208</td><td>False</td></tr>
<tr><td>15</td><td>DMX_0053</td><td>-0.000430238906</td><td>0.00020538452700000000574</td><td>pc / cm3</td><td>-3.5324598467549594982e-06</td><td>0.017199250101031026555</td><td>0.99997843698213984</td><td>False</td></tr>
<tr><td>16</td><td>DMX_0035</td><td>-0.000513755891</td><td>0.00019491340499999998989</td><td>pc / cm3</td><td>-3.341545378133300298e-06</td><td>0.017143743285041378871</td><td>1.0000012469213486188</td><td>False</td></tr>
<tr><td>17</td><td>OMDOT</td><td>0.0052209</td><td>0.0013554</td><td>deg / yr</td><td>-2.2104443267939444245e-05</td><td>0.016308427968082812635</td><td>1.0000991630450569873</td><td>False</td></tr>
<tr><td>18</td><td>PB</td><td>14.34846572550302</td><td>2.1222661e-06</td><td>d</td><td>-3.4563402588790037573e-08</td><td>0.016286083346847993084</td><td>1.0000724303266271833</td><td>False</td></tr>
<tr><td>19</td><td>DMX_0011</td><td>0.000704265134</td><td>0.00020910491599999999463</td><td>pc / cm3</td><td>-3.4012905693230413573e-06</td><td>0.016265952204218104421</td><td>1.0000021681421706887</td><td>False</td></tr>
<tr><td>20</td><td>SINI</td><td>0.906285</td><td>0.03399300000000000238</td><td></td><td>0.00054383117444134487783</td><td>0.015998328315869291688</td><td>0.9838897673347273276</td><td>False</td></tr>
<tr><td>21</td><td>DMX_0031</td><td>-0.000179354961</td><td>0.00019425115300000000604</td><td>pc / cm3</td><td>2.9236507108392417124e-06</td><td>0.01505087957361695393</td><td>1.0000002511076313549</td><td>False</td></tr>
<tr><td>22</td><td>A1DOT</td><td>-4e-15</td><td>6.260000000000000155e-16</td><td>ls / s</td><td>8.913933368296104875e-18</td><td>0.0142395101729969730114</td><td>0.99986819306556440345</td><td>False</td></tr>
<tr><td>23</td><td>F1</td><td>-7.338737472765e-16</td><td>4.619148184227e-21</td><td>Hz / s</td><td>6.3513794158537125015e-23</td><td>0.013750109679403142984</td><td>1.0001125509037762049</td><td>False</td></tr>
<tr><td>24</td><td>DMX_0056</td><td>-0.000421347758</td><td>0.00019678664899999999123</td><td>pc / cm3</td><td>-2.6695933518053631059e-06</td><td>0.013565927187496155948</td><td>0.99998689604544521714</td><td>False</td></tr>
<tr><td>25</td><td>DMX_0105</td><td>0.00262684657</td><td>0.00020525335700000000476</td><td>pc / cm3</td><td>-2.7729101312362787468e-06</td><td>0.0135096944175011890454</td><td>0.99998273896043621267</td><td>False</td></tr>
<tr><td>26</td><td>DMX_0051</td><td>-0.000733277186</td><td>0.00019519430499999998918</td><td>pc / cm3</td><td>-2.6027462090958027041e-06</td><td>0.013334129851256688523</td><td>0.99998203174591293596</td><td>False</td></tr>
<tr><td>27</td><td>DMX_0065</td><td>-0.000433771476</td><td>0.00019653452700000000747</td><td>pc / cm3</td><td>2.55244988362391019e-06</td><td>0.012987284842952353733</td><td>0.9999926429088619173</td><td>False</td></tr>
<tr><td>28</td><td>DMX_0041</td><td>-0.000465702114</td><td>0.00019991872200000000768</td><td>pc / cm3</td><td>2.5630846810842161307e-06</td><td>0.012820633582702755521</td><td>1.000002561739135265</td><td>False</td></tr>
<tr><td>29</td><td>DMX_0102</td><td>0.00206506621</td><td>0.00021999474200000001135</td><td>pc / cm3</td><td>-2.7990664467889871114e-06</td><td>0.012723333391254355168</td><td>0.9999938661690124242</td><td>False</td></tr>
<tr><td>30</td><td>DMX_0018</td><td>0.000148437633</td><td>0.00021436072500000000413</td><td>pc / cm3</td><td>2.6106353508762163585e-06</td><td>0.0121787018161849201064</td><td>1.0000016374151876608</td><td>False</td></tr>
<tr><td>31</td><td>DMX_0005</td><td>0.00127540597</td><td>0.00020848229500000000085</td><td>pc / cm3</td><td>2.369920663773195449e-06</td><td>0.011367491247989166062</td><td>0.9999924235250563509</td><td>False</td></tr>
<tr><td>32</td><td>DMX_0100</td><td>0.00162779064</td><td>0.00019986402499999999095</td><td>pc / cm3</td><td>2.2350040057427401213e-06</td><td>0.011182622814399641176</td><td>0.99999524330118094717</td><td>False</td></tr>
<tr><td>33</td><td>DMX_0052</td><td>-0.00060519003</td><td>0.0001963108310000000092</td><td>pc / cm3</td><td>-2.1681529670681404243e-06</td><td>0.011044489781962872274</td><td>0.99998002829372600875</td><td>False</td></tr>
<tr><td>34</td><td>DMX_0027</td><td>-0.000180903388</td><td>0.0001939144580000000135</td><td>pc / cm3</td><td>1.9385153421868869285e-06</td><td>0.00999675507530690087</td><td>0.9999997520244241489</td><td>False</td></tr>
<tr><td>35</td><td>DMX_0038</td><td>-0.000549879373</td><td>0.00019406231999999999497</td><td>pc / cm3</td><td>-1.8433444778984250587e-06</td><td>0.009498724316489801436</td><td>0.99999733194742501663</td><td>False</td></tr>
<tr><td>36</td><td>DMX_0082</td><td>-5.73415891e-05</td><td>0.00019599958599999999453</td><td>pc / cm3</td><td>1.8060890422916904205e-06</td><td>0.00921475947552098662</td><td>0.99999223779884427454</td><td>False</td></tr>
<tr><td>37</td><td>DMX_0064</td><td>-0.000332436758</td><td>0.00019756578099999999283</td><td>pc / cm3</td><td>1.7632373177434764594e-06</td><td>0.008924811315090423747</td><td>0.99999751295342376167</td><td>False</td></tr>
<tr><td>38</td><td>PMELAT</td><td>-7.1555</td><td>0.058200000000000001732</td><td>mas / yr</td><td>-0.00050443897788454705733</td><td>0.008667336389768850319</td><td>0.9992173055526453185</td><td>False</td></tr>
<tr><td>39</td><td>DMX_0103</td><td>0.00213551486</td><td>0.00019960339100000000262</td><td>pc / cm3</td><td>-1.6629830203091812424e-06</td><td>0.00833143671546733057</td><td>1.0000111082350762182</td><td>False</td></tr>
<tr><td>40</td><td>DMX_0068</td><td>-0.000460660442</td><td>0.000197973982999999999</td><td>pc / cm3</td><td>-1.6116407039094100481e-06</td><td>0.008140669190402710642</td><td>0.9999948346324893933</td><td>False</td></tr>
<tr><td>41</td><td>DMX_0096</td><td>0.00126516428</td><td>0.00019871720499999999572</td><td>pc / cm3</td><td>-1.5946752274548399442e-06</td><td>0.008024847307281923747</td><td>0.9999919061492179573</td><td>False</td></tr>
<tr><td>42</td><td>DMX_0054</td><td>-0.000434562424</td><td>0.00019725068499999999744</td><td>pc / cm3</td><td>-1.5787960878066963077e-06</td><td>0.008004008137192002506</td><td>0.9999977517462443899</td><td>False</td></tr>
<tr><td>43</td><td>DMX_0099</td><td>0.00166579784</td><td>0.00019833680699999999388</td><td>pc / cm3</td><td>1.5577460130062649457e-06</td><td>0.0078540440202118661644</td><td>1.0000058843876995507</td><td>False</td></tr>
<tr><td>44</td><td>DMX_0063</td><td>-0.00048241158</td><td>0.00019894769199999999066</td><td>pc / cm3</td><td>1.545569698093427792e-06</td><td>0.007768723942238183258</td><td>0.9999906240169186278</td><td>False</td></tr>
<tr><td>45</td><td>DMX_0077</td><td>0.000144509434</td><td>0.00020177111600000000057</td><td>pc / cm3</td><td>1.558508202363330852e-06</td><td>0.0077241392785047137404</td><td>1.0000024582436515264</td><td>False</td></tr>
<tr><td>46</td><td>DMX_0020</td><td>5.47353961e-05</td><td>0.0002081347350000000004</td><td>pc / cm3</td><td>1.5674105002988128867e-06</td><td>0.0075307492538370058785</td><td>0.9999922837607039261</td><td>False</td></tr>
<tr><td>47</td><td>DMX_0003</td><td>0.00125364077</td><td>0.00021051686999999999483</td><td>pc / cm3</td><td>-1.5307773452424497385e-06</td><td>0.007271518644764430467</td><td>1.0000078703768562338</td><td>False</td></tr>
<tr><td>48</td><td>DMX_0026</td><td>-0.000194499954</td><td>0.00019515168700000000836</td><td>pc / cm3</td><td>1.4055339118381122206e-06</td><td>0.0072022637028913418997</td><td>0.99999803308654033884</td><td>False</td></tr>
<tr><td>49</td><td>DMX_0072</td><td>-0.000200242101</td><td>0.00020093379300000001157</td><td>pc / cm3</td><td>-1.4307313121312322923e-06</td><td>0.0071204116080724770207</td><td>1.0000310288683174065</td><td>False</td></tr>
<tr><td>50</td><td>DMX_0014</td><td>0.000644805855</td><td>0.00021712707199999999345</td><td>pc / cm3</td><td>-1.5017921551000856101e-06</td><td>0.0069166508868138089003</td><td>1.0000021093639188674</td><td>False</td></tr>
<tr><td>51</td><td>DMX_0104</td><td>0.00280156853</td><td>0.00021134331599999999678</td><td>pc / cm3</td><td>-1.4502770774933899933e-06</td><td>0.006862185684137699429</td><td>0.99999369938154392123</td><td>False</td></tr>
<tr><td>52</td><td>DMX_0037</td><td>-0.000369999865</td><td>0.00019414035900000001017</td><td>pc / cm3</td><td>-1.2912550273230967435e-06</td><td>0.006651141648105722906</td><td>1.0000052717265923707</td><td>False</td></tr>
<tr><td>53</td><td>DMX_0089</td><td>0.000750810427</td><td>0.00021586616399999999146</td><td>pc / cm3</td><td>1.4269928461587040591e-06</td><td>0.006610544328562321763</td><td>1.0000027627266114827</td><td>False</td></tr>
<tr><td>54</td><td>DMX_0057</td><td>-0.000510024727</td><td>0.00020307030200000000504</td><td>pc / cm3</td><td>-1.3364295328476601815e-06</td><td>0.0065811175720202558584</td><td>0.99998317470625519565</td><td>False</td></tr>
<tr><td>55</td><td>DMX_0015</td><td>0.000269426488</td><td>0.00019949274600000000768</td><td>pc / cm3</td><td>-1.3113440716208462841e-06</td><td>0.00657339225568054613</td><td>0.99999899866494790235</td><td>False</td></tr>
<tr><td>56</td><td>DMX_0022</td><td>-0.000337579853</td><td>0.00019909290500000001315</td><td>pc / cm3</td><td>-1.2900624385661272307e-06</td><td>0.0064797007134238518086</td><td>1.0000060392126834952</td><td>False</td></tr>
<tr><td>57</td><td>DMX_0019</td><td>-5.05393994e-05</td><td>0.00020931933899999999337</td><td>pc / cm3</td><td>1.3550845642282494065e-06</td><td>0.0064737666892223915543</td><td>0.99999317265147769085</td><td>False</td></tr>
<tr><td>58</td><td>DMX_0036</td><td>-0.000380662206</td><td>0.00019450115199999998901</td><td>pc / cm3</td><td>-1.2485303719284219538e-06</td><td>0.006419141270322254768</td><td>0.999991726707249895</td><td>False</td></tr>
<tr><td>59</td><td>DMX_0004</td><td>0.00140149462</td><td>0.00020875585300000000966</td><td>pc / cm3</td><td>1.2005408850416458139e-06</td><td>0.0057509328135659301684</td><td>1.0000053613469559455</td><td>False</td></tr>
<tr><td>60</td><td>DMX_0006</td><td>0.00142904809</td><td>0.0002169403669999999903</td><td>pc / cm3</td><td>1.226544483889620632e-06</td><td>0.005653832437232028489</td><td>0.99999946641651471513</td><td>False</td></tr>
<tr><td>61</td><td>DMX_0047</td><td>-0.000560422813</td><td>0.00019698719399999998678</td><td>pc / cm3</td><td>-1.1068601458961990769e-06</td><td>0.0056189446807196974723</td><td>0.9999981570180671575</td><td>False</td></tr>
<tr><td>62</td><td>DMX_0023</td><td>-0.00029698608</td><td>0.00020075357699999999209</td><td>pc / cm3</td><td>-1.1001870305885259362e-06</td><td>0.005480286065281546673</td><td>1.0000031772970874311</td><td>False</td></tr>
<tr><td>63</td><td>DMX_0030</td><td>-0.000276292507</td><td>0.00019338047300000000238</td><td>pc / cm3</td><td>1.033258670074511075e-06</td><td>0.00534313860156144637</td><td>0.99999654362101131344</td><td>False</td></tr>
<tr><td>64</td><td>DMX_0032</td><td>-6.17411226e-05</td><td>0.00019561483700000000408</td><td>pc / cm3</td><td>1.0211425011750064999e-06</td><td>0.0052201689648674574895</td><td>0.99999143136936163856</td><td>False</td></tr>
<tr><td>65</td><td>DMX_0101</td><td>0.00164729443</td><td>0.0002156763620000000034</td><td>pc / cm3</td><td>1.1139591568685817957e-06</td><td>0.005164957098398116446</td><td>0.9999918001706793458</td><td>False</td></tr>
<tr><td>66</td><td>DMX_0028</td><td>7.30420287e-05</td><td>0.00019595359900000000435</td><td>pc / cm3</td><td>1.000818026096483588e-06</td><td>0.0051074235492683328977</td><td>0.99999899122330893064</td><td>False</td></tr>
<tr><td>67</td><td>DMX_0097</td><td>0.00139217947</td><td>0.00019620100300000000229</td><td>pc / cm3</td><td>-9.5059856186326466276e-07</td><td>0.00484502396689207894</td><td>0.999990178371573335</td><td>False</td></tr>
<tr><td>68</td><td>DMX_0085</td><td>0.000111528747</td><td>0.00019378922699999999836</td><td>pc / cm3</td><td>9.362368037293142351e-07</td><td>0.0048312118182364917687</td><td>1.0000007033138935686</td><td>False</td></tr>
<tr><td>69</td><td>DMX_0039</td><td>-0.00047461992</td><td>0.00019388960500000000826</td><td>pc / cm3</td><td>-9.213850378656050083e-07</td><td>0.0047521115836282453934</td><td>0.9999908369302923372</td><td>False</td></tr>
<tr><td>70</td><td>DMX_0042</td><td>-0.000492796209</td><td>0.0001964148330000000124</td><td>pc / cm3</td><td>8.8733857247786773054e-07</td><td>0.004517675976528043834</td><td>0.9999990730953730589</td><td>False</td></tr>
<tr><td>71</td><td>DMX_0095</td><td>0.00101732017</td><td>0.00019350028399999999072</td><td>pc / cm3</td><td>-8.6762346572207341144e-07</td><td>0.004483835619187377837</td><td>0.99999228888331448406</td><td>False</td></tr>
<tr><td>72</td><td>DMX_0013</td><td>0.000465466978</td><td>0.00021433810700000001295</td><td>pc / cm3</td><td>-9.5279928624632031706e-07</td><td>0.004445309793868433383</td><td>1.0000033208756189396</td><td>False</td></tr>
<tr><td>73</td><td>DMX_0088</td><td>0.000613814704</td><td>0.00020883815999999999274</td><td>pc / cm3</td><td>8.9567450821761464275e-07</td><td>0.0042888450473688079917</td><td>1.0000005789029351444</td><td>False</td></tr>
<tr><td>74</td><td>DMX_0081</td><td>1.64437918e-05</td><td>0.00019802516499999999345</td><td>pc / cm3</td><td>-8.450709838690101188e-07</td><td>0.0042674928909616624204</td><td>0.99999002061633068816</td><td>False</td></tr>
<tr><td>75</td><td>DMX_0084</td><td>7.53027171e-05</td><td>0.00019698282299999999678</td><td>pc / cm3</td><td>8.3440913356853528733e-07</td><td>0.00423594870283961443</td><td>0.99999837158372184565</td><td>False</td></tr>
<tr><td>76</td><td>DMX_0008</td><td>0.00117011674</td><td>0.00021557487400000001072</td><td>pc / cm3</td><td>7.983803255271360727e-07</td><td>0.0037034943391739431548</td><td>0.9999974349769962245</td><td>False</td></tr>
<tr><td>77</td><td>DMX_0029</td><td>-0.000149081739</td><td>0.0001953084259999999996</td><td>pc / cm3</td><td>7.0209507923327208953e-07</td><td>0.003594801789213498272</td><td>0.9999987027148998786</td><td>False</td></tr>
<tr><td>78</td><td>JUMP1</td><td>-8.789e-06</td><td>1.2999999999999999941e-07</td><td>s</td><td>-4.662519179964242028e-10</td><td>0.0035865532153571094524</td><td>1.0037094614491930411</td><td>False</td></tr>
<tr><td>79</td><td>DMX_0076</td><td>6.32076295e-05</td><td>0.0001970266330000000122</td><td>pc / cm3</td><td>-7.0508419791288424235e-07</td><td>0.0035786237991129055819</td><td>1.0000003480325394545</td><td>False</td></tr>
<tr><td>80</td><td>DMX_0090</td><td>0.000837190511</td><td>0.0001964115959999999906</td><td>pc / cm3</td><td>7.0256038385021167547e-07</td><td>0.0035769801689825467088</td><td>0.99998720562907705833</td><td>False</td></tr>
<tr><td>81</td><td>DMX_0034</td><td>-0.000219517582</td><td>0.00020007094000000001199</td><td>pc / cm3</td><td>7.0717686197075270253e-07</td><td>0.003534630576388318561</td><td>1.000001131565585899</td><td>False</td></tr>
<tr><td>82</td><td>DMX_0021</td><td>-4.23695167e-07</td><td>0.00020397483800000000814</td><td>pc / cm3</td><td>-7.0835792723677347136e-07</td><td>0.0034727711230577050158</td><td>0.9999987049068678191</td><td>False</td></tr>
<tr><td>83</td><td>DMX_0060</td><td>-0.00027606191</td><td>0.00019877439499999999847</td><td>pc / cm3</td><td>-6.7258933200914708395e-07</td><td>0.0033836819476127551198</td><td>1.0000033223481654687</td><td>False</td></tr>
<tr><td>84</td><td>DMX_0106</td><td>0.00250246721</td><td>0.000213191204999999993</td><td>pc / cm3</td><td>-6.973124112570443234e-07</td><td>0.0032708310422892182198</td><td>1.0000056111684749727</td><td>False</td></tr>
<tr><td>85</td><td>DMX_0086</td><td>0.000295028478</td><td>0.00019611881300000001249</td><td>pc / cm3</td><td>-6.397050327271646894e-07</td><td>0.0032618239063437765718</td><td>1.0000008584806781009</td><td>False</td></tr>
<tr><td>86</td><td>DMX_0087</td><td>0.000471334939</td><td>0.00019621266900000000795</td><td>pc / cm3</td><td>6.3167910168993599354e-07</td><td>0.0032193594068583614339</td><td>1.000000465515636261</td><td>False</td></tr>
<tr><td>87</td><td>DMX_0007</td><td>0.00112283379</td><td>0.00021651593300000000872</td><td>pc / cm3</td><td>6.8753117456231763183e-07</td><td>0.0031754299327353316916</td><td>1.0000009585708271587</td><td>False</td></tr>
<tr><td>88</td><td>DMX_0074</td><td>-8.53245743e-05</td><td>0.00019436179500000001297</td><td>pc / cm3</td><td>-5.18145797933985291e-07</td><td>0.002665882962924813731</td><td>1.0000157692717448477</td><td>False</td></tr>
<tr><td>89</td><td>DMX_0061</td><td>-0.000647584593</td><td>0.00021842225200000001313</td><td>pc / cm3</td><td>-5.569400976012999979e-07</td><td>0.0025498322286380417992</td><td>1.0000126836295164523</td><td>False</td></tr>
<tr><td>90</td><td>DMX_0058</td><td>-0.000538023723</td><td>0.00019927530600000000253</td><td>pc / cm3</td><td>-4.9610693264210640324e-07</td><td>0.0024895554928522170036</td><td>1.0000188498949289517</td><td>False</td></tr>
<tr><td>91</td><td>DMX_0040</td><td>-0.000523768585</td><td>0.00020212647099999999744</td><td>pc / cm3</td><td>4.978606436452370776e-07</td><td>0.0024631145103465299818</td><td>0.9999993457408867803</td><td>False</td></tr>
<tr><td>92</td><td>DMX_0066</td><td>-0.000463097236</td><td>0.00019734125199999998914</td><td>pc / cm3</td><td>4.7977929438933414677e-07</td><td>0.002431216431064976615</td><td>1.0000028852782862909</td><td>False</td></tr>
<tr><td>93</td><td>DMX_0091</td><td>0.0010543979</td><td>0.00019625349900000000464</td><td>pc / cm3</td><td>4.6727622962303150267e-07</td><td>0.002380982922618013909</td><td>0.9999740924369630024</td><td>False</td></tr>
<tr><td>94</td><td>DMX_0012</td><td>0.000529275001</td><td>0.00021944345800000000573</td><td>pc / cm3</td><td>-5.2019396662688743155e-07</td><td>0.0023705148076316197824</td><td>1.0000004306309258073</td><td>False</td></tr>
<tr><td>95</td><td>DMX_0059</td><td>1.13301417e-05</td><td>0.00019859404400000000342</td><td>pc / cm3</td><td>-4.294857863349665672e-07</td><td>0.002162631757148600689</td><td>1.0000066065967603279</td><td>False</td></tr>
<tr><td>96</td><td>DMX_0098</td><td>0.00134025305</td><td>0.00019579969600000001091</td><td>pc / cm3</td><td>4.001641074554990879e-07</td><td>0.0020437422306084636163</td><td>0.9999965325297406338</td><td>False</td></tr>
<tr><td>97</td><td>DMX_0016</td><td>0.000273862849</td><td>0.00021262252799999998903</td><td>pc / cm3</td><td>-4.227590594932427795e-07</td><td>0.0019883079345818085656</td><td>1.0000014589470898052</td><td>False</td></tr>
<tr><td>98</td><td>DMX_0078</td><td>0.000355456415</td><td>0.0002078381100000000056</td><td>pc / cm3</td><td>4.0912298299579201444e-07</td><td>0.0019684695121399632851</td><td>1.0000007925966629685</td><td>False</td></tr>
<tr><td>99</td><td>DMX_0092</td><td>0.00132019531</td><td>0.00019585215799999998649</td><td>pc / cm3</td><td>-3.6651712927724192093e-07</td><td>0.001871396940529202477</td><td>1.0000086079475452028</td><td>False</td></tr>
<tr><td>100</td><td>DMX_0009</td><td>0.000689985252</td><td>0.00020937671899999999586</td><td>pc / cm3</td><td>-3.7588711465261995942e-07</td><td>0.0017952670022144151963</td><td>0.9999891731736462175</td><td>False</td></tr>
<tr><td>101</td><td>DMX_0080</td><td>0.000173972859</td><td>0.00019499145700000000735</td><td>pc / cm3</td><td>3.4848555984336778563e-07</td><td>0.001787183732071747971</td><td>1.0000043702457701578</td><td>False</td></tr>
<tr><td>102</td><td>DMX_0070</td><td>-0.000237615254</td><td>0.00019767136999999999729</td><td>pc / cm3</td><td>-3.3088528276304990036e-07</td><td>0.0016739160697022026792</td><td>0.9999750056012474131</td><td>False</td></tr>
<tr><td>103</td><td>FD1</td><td>3.98314325e-05</td><td>1.6566479199999999207e-06</td><td>s</td><td>-2.5078110490474835037e-09</td><td>0.0015137863747461100493</td><td>0.99999722077930985886</td><td>False</td></tr>
<tr><td>104</td><td>DMX_0024</td><td>-6.43052299e-05</td><td>0.00019594539100000000478</td><td>pc / cm3</td><td>2.839220007323913983e-07</td><td>0.0014489853488433999864</td><td>1.0000000060466058827</td><td>False</td></tr>
<tr><td>105</td><td>DMX_0050</td><td>-0.000587701875</td><td>0.00019801309199999998778</td><td>pc / cm3</td><td>2.2998644031921311459e-07</td><td>0.0011614708805174009127</td><td>1.0000012673255922468</td><td>False</td></tr>
<tr><td>106</td><td>DMX_0093</td><td>0.0011628873</td><td>0.00019334416100000000461</td><td>pc / cm3</td><td>2.2370304937234172793e-07</td><td>0.0011570199390316302478</td><td>1.0000052990898598004</td><td>False</td></tr>
<tr><td>107</td><td>DMX_0062</td><td>-0.000466362595</td><td>0.00020673830600000000883</td><td>pc / cm3</td><td>-2.3600561441591708448e-07</td><td>0.0011415669354276176354</td><td>0.9999965650357631741</td><td>False</td></tr>
<tr><td>108</td><td>FD2</td><td>-1.47296057e-05</td><td>1.1922595999999999884e-06</td><td>s</td><td>1.3481392263201306481e-09</td><td>0.0011307430246903700001</td><td>0.9999985886156963488</td><td>False</td></tr>
<tr><td>109</td><td>DMX_0025</td><td>-0.000119213129</td><td>0.0001993283730000000001</td><td>pc / cm3</td><td>-2.1632912679417717634e-07</td><td>0.0010852901849260424005</td><td>0.999999959538387162</td><td>False</td></tr>
<tr><td>110</td><td>DMX_0044</td><td>-0.000339256292</td><td>0.00021062295900000000716</td><td>pc / cm3</td><td>-2.2697240848818444822e-07</td><td>0.0010776242512488131038</td><td>0.9999991557839980061</td><td>False</td></tr>
<tr><td>111</td><td>DMX_0049</td><td>-0.000527616723</td><td>0.00019778945300000000569</td><td>pc / cm3</td><td>-2.1233907142629519088e-07</td><td>0.0010735611439619846975</td><td>0.9999933987894804588</td><td>False</td></tr>
<tr><td>112</td><td>DMX_0046</td><td>-0.000485115169</td><td>0.00019873250400000001294</td><td>pc / cm3</td><td>1.6429452485267450126e-07</td><td>0.00082671189435963880407</td><td>1.0000075847076443925</td><td>False</td></tr>
<tr><td>113</td><td>DMX_0048</td><td>-0.000655235255</td><td>0.00021281515900000000287</td><td>pc / cm3</td><td>-1.5656777639736635388e-07</td><td>0.0007356984207941989805</td><td>0.99999458552246589527</td><td>False</td></tr>
<tr><td>114</td><td>DMX_0079</td><td>0.000190066396</td><td>0.00019490725099999999578</td><td>pc / cm3</td><td>1.15761614804529367265e-07</td><td>0.00059393180197554252316</td><td>0.99996601531162898624</td><td>False</td></tr>
<tr><td>115</td><td>DMX_0045</td><td>3.64190777e-05</td><td>0.00020164094999999999935</td><td>pc / cm3</td><td>-1.0880461652431374893e-07</td><td>0.00053959583370497780225</td><td>1.0000006821705129667</td><td>False</td></tr>
<tr><td>116</td><td>DMX_0071</td><td>-0.000176912603</td><td>0.00019118353399999999634</td><td>pc / cm3</td><td>-9.349008259535601141e-08</td><td>0.0004890069800433546844</td><td>1.0000046190658971046</td><td>False</td></tr>
<tr><td>117</td><td>DMX_0075</td><td>2.00017094e-06</td><td>0.00019663653799999999744</td><td>pc / cm3</td><td>-9.082493292551379154e-08</td><td>0.00046189245319968859436</td><td>0.9999419961581833549</td><td>False</td></tr>
<tr><td>118</td><td>DMX_0094</td><td>0.000929849121</td><td>0.00019402737299999999105</td><td>pc / cm3</td><td>5.00029611804828078e-08</td><td>0.00025771086010880955765</td><td>0.9999940905421160764</td><td>False</td></tr>
<tr><td>119</td><td>DMX_0073</td><td>-0.000156953835</td><td>0.00019724444300000000259</td><td>pc / cm3</td><td>4.9749872529263657744e-08</td><td>0.00025222445698641892406</td><td>1.0000039385363455047</td><td>False</td></tr>
<tr><td>120</td><td>DMX_0017</td><td>0.000178762757</td><td>0.00021197504699999999088</td><td>pc / cm3</td><td>-2.7382927343715330118e-08</td><td>0.00012917995646777864055</td><td>0.9999889282854504957</td><td>False</td></tr>
<tr><td>121</td><td>DMX_0043</td><td>-0.000494848648</td><td>0.0001997188189999999947</td><td>pc / cm3</td><td>2.5596058013453541757e-08</td><td>0.00012816047151497297354</td><td>0.9999848366739377825</td><td>False</td></tr>
<tr><td>122</td><td>DMX_0083</td><td>8.70047706e-06</td><td>0.00020486178099999999887</td><td>pc / cm3</td><td>2.416989696640591326e-08</td><td>0.000117981484142256444004</td><td>1.0000060508320367525</td><td>False</td></tr>
<tr><td>123</td><td>DMX_0069</td><td>-0.000251368356</td><td>0.00019942850700000000919</td><td>pc / cm3</td><td>2.1310941707771303283e-08</td><td>0.000106860057412811609596</td><td>1.0000028555921218754</td><td>False</td></tr>
<tr><td>124</td><td>DMX_0067</td><td>-0.000377967984</td><td>0.00019749766400000001308</td><td>pc / cm3</td><td>-1.27852614923047724904e-08</td><td>6.473626691759135337e-05</td><td>0.999976952183460277</td><td>False</td></tr>
</table><style>table.dataTable {clear: both; width: auto !important; margin: 0 !important;}
.dataTables_info, .dataTables_length, .dataTables_filter, .dataTables_paginate{
display: inline-block; margin-right: 1em; }
.paginate_button { margin-right: 5px; }
</style>
<script>

var astropy_sort_num = function(a, b) {
    var a_num = parseFloat(a);
    var b_num = parseFloat(b);

    if (isNaN(a_num) && isNaN(b_num))
        return ((a < b) ? -1 : ((a > b) ? 1 : 0));
    else if (!isNaN(a_num) && !isNaN(b_num))
        return ((a_num < b_num) ? -1 : ((a_num > b_num) ? 1 : 0));
    else
        return isNaN(a_num) ? -1 : 1;
}

require.config({paths: {
    datatables: 'https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min'
}});
require(["datatables"], function(){
    console.log("$('#table140671879831112-30846').dataTable()");

jQuery.extend( jQuery.fn.dataTableExt.oSort, {
    "optionalnum-asc": astropy_sort_num,
    "optionalnum-desc": function (a,b) { return -astropy_sort_num(a, b); }
});

    $('#table140671879831112-30846').dataTable({
        order: [],
        pageLength: 50,
        lengthMenu: [[10, 25, 50, 100, 500, 1000, -1], [10, 25, 50, 100, 500, 1000, 'All']],
        pagingType: "full_numbers",
        columnDefs: [{targets: [0, 3, 5, 6, 7], type: "optionalnum"}]
    });
});
</script>




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
<table id="table140671879831112">
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
<table id="table140671881631448" class="table-striped table-bordered table-condensed">
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

Before TEMPO2 run, the .par file has to be modified for a more accurate TEMPO2 vs PINT comparison.
We save the modified .par file in a file named "[PSR name]_tempo2.par". In this case, "J1600-3053_tempo2.par"
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


![png](paper_validation_example_files/paper_validation_example_51_0.png)


### Write out the TEMPO2 postfit parameter to a new file

Note, since the ECL parameter is hard coded in tempo2, we will have to add it manually 


```python
# Write out the post fit tempo parfile.
tempo2_parfile = open(psr + '_new_tempo2.2.par', 'w')
for line in tempo2_new_par:
    tempo2_parfile.write(line)
tempo2_parfile.write("ECL IERS2003")
tempo2_parfile.close()
```

## Compare the parameter between TEMPO2 and PINT

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


## Print the parameter difference in a table.


```python
compare_table2.show_in_notebook()
```




<i>Table length=125</i>
<table id="table140672074185640-254392" class="table-striped table-bordered table-condensed">
<thead><tr><th>idx</th><th>name</th><th>Tempo2 Value</th><th>T2 unc</th><th>units</th><th>Tempo2_V-PINT_V</th><th>Tempo2_PINT_diff/unct</th><th>PINT_unct/Tempo2_unct</th><th>no_t_unc</th></tr></thead>
<tr><td>0</td><td>ECC</td><td>0.00017372966157521168</td><td>8.922286680669999241e-09</td><td></td><td>4.168033894912624715e-11</td><td>0.0046714861829564476026</td><td>1.0000400789683185909</td><td>False</td></tr>
<tr><td>1</td><td>DMX_0098</td><td>0.0013394613122489417</td><td>0.00019579968831114546654</td><td>pc / cm3</td><td>-5.162393215032545085e-07</td><td>0.0026365686582855950293</td><td>0.99999926235860314705</td><td>False</td></tr>
<tr><td>2</td><td>DMX_0070</td><td>-0.00023747963906517973</td><td>0.00019767137320477682749</td><td>pc / cm3</td><td>-4.6318680163804021657e-07</td><td>0.0023432163905605278911</td><td>1.0000006066661308868</td><td>False</td></tr>
<tr><td>3</td><td>DMX_0097</td><td>0.0013928330661987446</td><td>0.00019620100461426303326</td><td>pc / cm3</td><td>-4.3591375636898264945e-07</td><td>0.0022217712759728318155</td><td>0.99999985479541497746</td><td>False</td></tr>
<tr><td>4</td><td>DMX_0055</td><td>-0.0005307704904403621</td><td>0.00019675128861832102923</td><td>pc / cm3</td><td>-3.936735762570617997e-07</td><td>0.0020008691125817808752</td><td>1.0000000155376389532</td><td>False</td></tr>
<tr><td>5</td><td>DMX_0063</td><td>-0.00048410571072825574</td><td>0.00019894769104906708185</td><td>pc / cm3</td><td>-3.8388090987831295642e-07</td><td>0.0019295569999032320674</td><td>1.0000001737671666557</td><td>False</td></tr>
<tr><td>6</td><td>DMX_0079</td><td>0.00018976795294000216</td><td>0.00019490725481464179483</td><td>pc / cm3</td><td>-3.6413058978400727507e-07</td><td>0.0018682249161545991592</td><td>1.0000001869460202197</td><td>False</td></tr>
<tr><td>7</td><td>DMX_0010</td><td>0.00067403356955979</td><td>0.00020051850482404336064</td><td>pc / cm3</td><td>-3.734867366063333513e-07</td><td>0.001862604835070314456</td><td>0.9999998791435175116</td><td>False</td></tr>
<tr><td>8</td><td>F1</td><td>-7.3387383041227678664e-16</td><td>4.619148404392432094e-21</td><td>Hz / s</td><td>-8.212906306513322778e-24</td><td>0.001778013085421442844</td><td>0.999998198306420979</td><td>False</td></tr>
<tr><td>9</td><td>DMX_0086</td><td>0.00029525346690830644</td><td>0.0001961188165133768578</td><td>pc / cm3</td><td>-3.4086588203348670498e-07</td><td>0.0017380580206093430157</td><td>1.0000003760250906204</td><td>False</td></tr>
<tr><td>10</td><td>A1DOT</td><td>-4.008979189463729e-15</td><td>6.2586911221949290846e-16</td><td>ls / s</td><td>-1.0370977784914106416e-18</td><td>0.0016570521827057340097</td><td>1.0000003677433377813</td><td>False</td></tr>
<tr><td>11</td><td>DMX_0038</td><td>-0.0005482957712926262</td><td>0.00019406232394920700456</td><td>pc / cm3</td><td>-3.0585239190448771512e-07</td><td>0.0015760524025495033749</td><td>0.9999999990249419657</td><td>False</td></tr>
<tr><td>12</td><td>ELONG</td><td>244.34767784255382</td><td>5.95727548431e-09</td><td>deg</td><td>9.322320693172514439e-12</td><td>0.0015648631186731613756</td><td>1.0000013810109530107</td><td>False</td></tr>
<tr><td>13</td><td>DMX_0042</td><td>-0.0004940039798916835</td><td>0.00019641482962940920508</td><td>pc / cm3</td><td>-2.9825226018748672574e-07</td><td>0.0015184813730726033198</td><td>1.0000000809520381839</td><td>False</td></tr>
<tr><td>14</td><td>DMX_0057</td><td>-0.0005087481263132066</td><td>0.00020307030290145331483</td><td>pc / cm3</td><td>-2.9374758932318879162e-07</td><td>0.0014465314973491701263</td><td>0.9999987529607743397</td><td>False</td></tr>
<tr><td>15</td><td>DMX_0088</td><td>0.0006125794192867647</td><td>0.00020883815823101754415</td><td>pc / cm3</td><td>-3.0114430235475640046e-07</td><td>0.0014419984590250478976</td><td>0.9999998886061222114</td><td>False</td></tr>
<tr><td>16</td><td>DMX_0036</td><td>-0.0003795487044144885</td><td>0.00019450115140772264964</td><td>pc / cm3</td><td>-2.8000681475889854788e-07</td><td>0.0014396152039837276874</td><td>1.0000000823893255841</td><td>False</td></tr>
<tr><td>17</td><td>DMX_0033</td><td>-0.0004819567646363058</td><td>0.00020741594616667348504</td><td>pc / cm3</td><td>-2.9590167292299499394e-07</td><td>0.0014266100480298507438</td><td>1.0000015596686879782</td><td>False</td></tr>
<tr><td>18</td><td>DMX_0022</td><td>-0.0003366382462135058</td><td>0.00019909290455080911708</td><td>pc / cm3</td><td>-2.8140481767187624368e-07</td><td>0.0014134346892310312768</td><td>1.000000276890799622</td><td>False</td></tr>
<tr><td>19</td><td>DMX_0043</td><td>-0.0004949544096424111</td><td>0.00019971881972794227797</td><td>pc / cm3</td><td>-2.8205023024422109373e-07</td><td>0.0014122366166014347147</td><td>1.0000002509314975807</td><td>False</td></tr>
<tr><td>20</td><td>DMX_0039</td><td>-0.0004738786560061594</td><td>0.00019388960405229325144</td><td>pc / cm3</td><td>-2.7357393523903122956e-07</td><td>0.0014109778426554867286</td><td>0.9999997687736489338</td><td>False</td></tr>
<tr><td>21</td><td>DMX_0076</td><td>6.340643734163504e-05</td><td>0.00019702663031172622785</td><td>pc / cm3</td><td>-2.7554955794128848084e-07</td><td>0.0013985396669745962881</td><td>0.9999996696554116493</td><td>False</td></tr>
<tr><td>22</td><td>DMX_0078</td><td>0.00035474733258982833</td><td>0.00020783811179429391284</td><td>pc / cm3</td><td>-2.837475355594513797e-07</td><td>0.0013652334170562913963</td><td>0.99999986789745354265</td><td>False</td></tr>
<tr><td>23</td><td>DMX_0019</td><td>-5.209241737200372e-05</td><td>0.0002093193378019121947</td><td>pc / cm3</td><td>-2.830275994888738927e-07</td><td>0.0013521330731359132785</td><td>0.9999999446794847202</td><td>False</td></tr>
<tr><td>24</td><td>DMX_0101</td><td>0.0016461143495036883</td><td>0.00021567636847723566204</td><td>pc / cm3</td><td>-2.913164346490196982e-07</td><td>0.0013507109596931465411</td><td>0.99999933975707633316</td><td>False</td></tr>
<tr><td>25</td><td>DMX_0029</td><td>-0.00015000366900843785</td><td>0.00019530842588202432914</td><td>pc / cm3</td><td>-2.6200724745402566816e-07</td><td>0.0013415050900686211789</td><td>1.0000000312777432843</td><td>False</td></tr>
<tr><td>26</td><td>DMX_0099</td><td>0.0016640840837521922</td><td>0.0001983368094349508351</td><td>pc / cm3</td><td>-2.630347183019306201e-07</td><td>0.0013262022266633212084</td><td>1.0000000223941387656</td><td>False</td></tr>
<tr><td>27</td><td>DMX_0077</td><td>0.0001427234160398769</td><td>0.00020177112518042920387</td><td>pc / cm3</td><td>-2.6728043183498286818e-07</td><td>0.0013246713651221078169</td><td>0.9999996630552079324</td><td>False</td></tr>
<tr><td>28</td><td>DMX_0020</td><td>5.3117036722839084e-05</td><td>0.00020813473513131200811</td><td>pc / cm3</td><td>-2.7179329591115039197e-07</td><td>0.00130585265232002843</td><td>1.0000001267467690802</td><td>False</td></tr>
<tr><td>29</td><td>F0</td><td>277.93771124297462788</td><td>5.1859268946902080184e-13</td><td>Hz</td><td>-6.6613381477509392425e-16</td><td>0.0012845029023782387781</td><td>1.0000082417695045875</td><td>False</td></tr>
<tr><td>30</td><td>DMX_0081</td><td>1.710704563416846e-05</td><td>0.0001980251643661329298</td><td>pc / cm3</td><td>-2.5393696526508527204e-07</td><td>0.0012823469485705150678</td><td>0.9999999494488003293</td><td>False</td></tr>
<tr><td>31</td><td>SINI</td><td>0.9064200568225846</td><td>0.03399283139781983376</td><td></td><td>-4.2613468352326044908e-05</td><td>0.0012536016153999781763</td><td>1.0000254830135359985</td><td>False</td></tr>
<tr><td>32</td><td>DMX_0005</td><td>0.0012728885012668423</td><td>0.00020848229506442136948</td><td>pc / cm3</td><td>-2.5825493498539062176e-07</td><td>0.0012387379700784155543</td><td>1.0000000118434695384</td><td>False</td></tr>
<tr><td>33</td><td>DMX_0021</td><td>5.3866716427319967e-08</td><td>0.00020397483853785099476</td><td>pc / cm3</td><td>-2.513757553378762266e-07</td><td>0.0012323860979119205701</td><td>1.0000000403495790113</td><td>False</td></tr>
<tr><td>34</td><td>PMELAT</td><td>-7.155145674275822</td><td>0.058156247552489513664</td><td>mas / yr</td><td>-7.1059018702079868035e-05</td><td>0.0012218638872452151304</td><td>0.99999263990213360653</td><td>False</td></tr>
<tr><td>35</td><td>DMX_0012</td><td>0.0005296290484440793</td><td>0.00021944345545559323604</td><td>pc / cm3</td><td>-2.6638303703336620176e-07</td><td>0.0012139028547482552407</td><td>1.0000000918307434539</td><td>False</td></tr>
<tr><td>36</td><td>DMX_0011</td><td>0.0007073930149308514</td><td>0.00020910491932369515487</td><td>pc / cm3</td><td>-2.522924834023627244e-07</td><td>0.0012065353805082560962</td><td>1.0000000559788380095</td><td>False</td></tr>
<tr><td>37</td><td>FD2</td><td>-1.4729805752137882e-05</td><td>1.1922596055992699934e-06</td><td>s</td><td>1.4162333940147552079e-09</td><td>0.001187856560235392954</td><td>1.0000000136320319477</td><td>False</td></tr>
<tr><td>38</td><td>DMX_0091</td><td>0.0010538805339301686</td><td>0.00019625349834566545369</td><td>pc / cm3</td><td>-2.3256715643332932786e-07</td><td>0.0011850344497997373442</td><td>1.0000000161657975895</td><td>False</td></tr>
<tr><td>39</td><td>DMX_0007</td><td>0.0011221066247962318</td><td>0.00021651592968159987505</td><td>pc / cm3</td><td>-2.5255446510540416338e-07</td><td>0.0011664475010074370695</td><td>0.99999977325635780456</td><td>False</td></tr>
<tr><td>40</td><td>M2</td><td>0.2717633814383356</td><td>0.08941866471282471085</td><td>solMass</td><td>0.000104187428365654088935</td><td>0.0011651642159974146158</td><td>1.0000236859702187342</td><td>False</td></tr>
<tr><td>41</td><td>DMX_0095</td><td>0.0010180671480248778</td><td>0.00019350028172601836923</td><td>pc / cm3</td><td>-2.1857074997307102127e-07</td><td>0.0011295629547586423427</td><td>1.0000000254528691457</td><td>False</td></tr>
<tr><td>42</td><td>DMX_0104</td><td>0.0028028581372787262</td><td>0.00021134331743690571272</td><td>pc / cm3</td><td>-2.3282431142529133594e-07</td><td>0.0011016402801323423206</td><td>1.000000042833738334</td><td>False</td></tr>
<tr><td>43</td><td>DMX_0080</td><td>0.00017334953727913454</td><td>0.00019499146298943662871</td><td>pc / cm3</td><td>-2.1245123294977232183e-07</td><td>0.001089541201920627431</td><td>0.99999961139674531374</td><td>False</td></tr>
<tr><td>44</td><td>DMX_0009</td><td>0.0006902777710259759</td><td>0.00020937671927892699672</td><td>pc / cm3</td><td>-2.2516669811859825467e-07</td><td>0.0010754142050465325016</td><td>1.0000005715070843237</td><td>False</td></tr>
<tr><td>45</td><td>DMX_0103</td><td>0.0021367782515491268</td><td>0.00019960339391631736383</td><td>pc / cm3</td><td>-2.1339631551924032049e-07</td><td>0.0010691016386660517207</td><td>1.0000002959512961365</td><td>False</td></tr>
<tr><td>46</td><td>DMX_0037</td><td>-0.0003687313513484703</td><td>0.00019414036258329412624</td><td>pc / cm3</td><td>-2.0521050286766194873e-07</td><td>0.0010570213228051341878</td><td>0.9999982285943487259</td><td>False</td></tr>
<tr><td>47</td><td>DMX_0018</td><td>0.00014555033283061047</td><td>0.00021436072428795556704</td><td>pc / cm3</td><td>-2.2571554225304523052e-07</td><td>0.0010529706083183246104</td><td>0.9999998868591526424</td><td>False</td></tr>
<tr><td>48</td><td>DMX_0067</td><td>-0.00037750338215465976</td><td>0.0001974976632319764018</td><td>pc / cm3</td><td>2.0746903121160844077e-07</td><td>0.0010504885365044541734</td><td>0.9999996351143883855</td><td>False</td></tr>
<tr><td>49</td><td>A1</td><td>8.80165312286463</td><td>8.114047416773300209e-07</td><td>ls</td><td>-8.203180357213568641e-10</td><td>0.001010985015968234816</td><td>0.9999811897467071331</td><td>False</td></tr>
<tr><td>50</td><td>DMX_0068</td><td>-0.0004591808995712415</td><td>0.00019797398343078884001</td><td>pc / cm3</td><td>-1.9855041930385858989e-07</td><td>0.0010029116748730333188</td><td>1.0000000821768730841</td><td>False</td></tr>
<tr><td>51</td><td>DMX_0026</td><td>-0.00019604126707036584</td><td>0.00019515168673082307501</td><td>pc / cm3</td><td>-1.9537535440915590468e-07</td><td>0.0010011461221887431317</td><td>0.99999980480436989616</td><td>False</td></tr>
<tr><td>52</td><td>FD1</td><td>3.983282287426775e-05</td><td>1.6566478062738200598e-06</td><td>s</td><td>-1.6031694728325007922e-09</td><td>0.0009677189483251698093</td><td>1.0000000032094340519</td><td>False</td></tr>
<tr><td>53</td><td>DMX_0085</td><td>0.00011041779738876759</td><td>0.00019378922563451398634</td><td>pc / cm3</td><td>-1.8622419345447891511e-07</td><td>0.000960962575936483176</td><td>1.0000000251302354481</td><td>False</td></tr>
<tr><td>54</td><td>OM</td><td>181.84960401549451478</td><td>0.01296564244572522874</td><td>deg</td><td>1.2275557313340401677e-05</td><td>0.00094677586280251403394</td><td>1.0000088591412276617</td><td>False</td></tr>
<tr><td>55</td><td>T0</td><td>55878.2618994738495070</td><td>0.00051676746764245482</td><td>d</td><td>4.890245969835227413e-07</td><td>0.00094631459525597103496</td><td>1.000008278658846269</td><td>False</td></tr>
<tr><td>56</td><td>PB</td><td>14.348465754661366786</td><td>2.12226632065849e-06</td><td>d</td><td>-1.9218603731011030256e-09</td><td>0.0009055698403133468854</td><td>0.9999977436618808823</td><td>False</td></tr>
<tr><td>57</td><td>OMDOT</td><td>0.0052395528517645540778</td><td>0.00135543635075636363</td><td>deg / yr</td><td>-1.2270890653582061121e-06</td><td>0.0009053092494335592534</td><td>0.99999775911651708066</td><td>False</td></tr>
<tr><td>58</td><td>DMX_0090</td><td>0.0008365968722806777</td><td>0.00019641159660997595466</td><td>pc / cm3</td><td>-1.6729950636656682611e-07</td><td>0.0008517801863745427587</td><td>0.99999991349767602955</td><td>False</td></tr>
<tr><td>59</td><td>DMX_0060</td><td>-0.0002757321013267392</td><td>0.00019877439845465119574</td><td>pc / cm3</td><td>-1.6765339167792124575e-07</td><td>0.00084343553788276232844</td><td>0.9999996776389862285</td><td>False</td></tr>
<tr><td>60</td><td>DMX_0064</td><td>-0.0003339126001174257</td><td>0.0001975657807158899906</td><td>pc / cm3</td><td>1.6554261815759615042e-07</td><td>0.0008379113911211940203</td><td>0.9999999587441709137</td><td>False</td></tr>
<tr><td>61</td><td>DMX_0071</td><td>-0.0001770123194933147</td><td>0.00019118353589100777314</td><td>pc / cm3</td><td>-1.5878878125786047293e-07</td><td>0.0008305567763344679378</td><td>0.99999999496721447834</td><td>False</td></tr>
<tr><td>62</td><td>DMX_0049</td><td>-0.0005272400689970337</td><td>0.00019778945222269806326</td><td>pc / cm3</td><td>-1.6396018772031357297e-07</td><td>0.00082896325298330395415</td><td>1.0000003844385727536</td><td>False</td></tr>
<tr><td>63</td><td>DMX_0074</td><td>-8.508105788347532e-05</td><td>0.00019436179297763081595</td><td>pc / cm3</td><td>-1.570019060636505771e-07</td><td>0.000807781733530931129</td><td>0.99999998883081020473</td><td>False</td></tr>
<tr><td>64</td><td>DMX_0102</td><td>0.0020677615785958396</td><td>0.00021999474018152901907</td><td>pc / cm3</td><td>-1.733685127580826546e-07</td><td>0.0007880575354439262785</td><td>1.000001328113558241</td><td>False</td></tr>
<tr><td>65</td><td>DMX_0047</td><td>-0.0005596362035486417</td><td>0.00019698719335984341211</td><td>pc / cm3</td><td>-1.4886066453708834273e-07</td><td>0.00075568701699891399886</td><td>1.0000000997648352818</td><td>False</td></tr>
<tr><td>66</td><td>DMX_0073</td><td>-0.00015722695908922736</td><td>0.00019724444535403976302</td><td>pc / cm3</td><td>-1.4577911090177687187e-07</td><td>0.00073907840923030166443</td><td>0.99999986690420972213</td><td>False</td></tr>
<tr><td>67</td><td>DMX_0015</td><td>0.00027061984568647364</td><td>0.00019949274463237635555</td><td>pc / cm3</td><td>-1.4117327001503254813e-07</td><td>0.0007076611747218453357</td><td>0.99999991672614918503</td><td>False</td></tr>
<tr><td>68</td><td>PMELONG</td><td>0.4619096015625491</td><td>0.010433361011620021289</td><td>mas / yr</td><td>7.3610413870994761965e-06</td><td>0.00070552925168612602887</td><td>1.0000025739354863052</td><td>False</td></tr>
<tr><td>69</td><td>DMX_0096</td><td>0.0012663968227548262</td><td>0.00019871719250690825665</td><td>pc / cm3</td><td>-1.3636655514254886201e-07</td><td>0.0006862343082761104525</td><td>0.99999987626368358473</td><td>False</td></tr>
<tr><td>70</td><td>DMX_0013</td><td>0.0004664122959219204</td><td>0.00021433810656524713905</td><td>pc / cm3</td><td>-1.4582604301421582885e-07</td><td>0.00068035518905652271615</td><td>1.0000000938979594078</td><td>False</td></tr>
<tr><td>71</td><td>DMX_0087</td><td>0.00047056558302748696</td><td>0.00019621266949622890685</td><td>pc / cm3</td><td>-1.3277487230083659733e-07</td><td>0.0006766885779686537816</td><td>0.99999998742245421735</td><td>False</td></tr>
<tr><td>72</td><td>DMX_0093</td><td>0.00116270185664641</td><td>0.00019334415612049150488</td><td>pc / cm3</td><td>1.263370579239215391e-07</td><td>0.00065343096196395335576</td><td>1.0000004120427543608</td><td>False</td></tr>
<tr><td>73</td><td>DMX_0030</td><td>-0.0002773766328802759</td><td>0.00019338047264200034815</td><td>pc / cm3</td><td>-1.2288480924835384553e-07</td><td>0.0006354561428539215722</td><td>1.0000000468745264826</td><td>False</td></tr>
<tr><td>74</td><td>DMX_0046</td><td>-0.00048511094784289003</td><td>0.0001987325016671356154</td><td>pc / cm3</td><td>-1.2254420611959061793e-07</td><td>0.0006166289111825524029</td><td>0.99999921792490664707</td><td>False</td></tr>
<tr><td>75</td><td>DMX_0004</td><td>0.0014002738188092162</td><td>0.00020875585305964335352</td><td>pc / cm3</td><td>-1.2687744618583747525e-07</td><td>0.0006077791081124201897</td><td>1.0000001062703025578</td><td>False</td></tr>
<tr><td>76</td><td>DMX_0084</td><td>7.437906648633243e-05</td><td>0.00019698282264668675223</td><td>pc / cm3</td><td>-1.18720425420452014437e-07</td><td>0.0006026943051445246384</td><td>0.99999990222525569905</td><td>False</td></tr>
<tr><td>77</td><td>DMX_0028</td><td>7.200316651012961e-05</td><td>0.00019595360139601760634</td><td>pc / cm3</td><td>-1.1597165213507497697e-07</td><td>0.00059183220573067699555</td><td>1.0000006390666473788</td><td>False</td></tr>
<tr><td>78</td><td>DMX_0075</td><td>2.2169833444943287e-06</td><td>0.00019663653810633987466</td><td>pc / cm3</td><td>-1.01211204042285119885e-07</td><td>0.00051471209276248905667</td><td>0.9999998269259767758</td><td>False</td></tr>
<tr><td>79</td><td>DMX_0072</td><td>-0.00019857366878237687</td><td>0.00020093379379496133057</td><td>pc / cm3</td><td>-1.0295980771161451252e-07</td><td>0.000512406627909876085</td><td>0.9999998085033193762</td><td>False</td></tr>
<tr><td>80</td><td>DMX_0023</td><td>-0.0002960500315749317</td><td>0.00020075357758775206132</td><td>pc / cm3</td><td>-1.0260043520441157125e-07</td><td>0.0005110764970530279252</td><td>1.0000001619091680727</td><td>False</td></tr>
<tr><td>81</td><td>DMX_0054</td><td>-0.00043272493252836963</td><td>0.0001972506851651888778</td><td>pc / cm3</td><td>1.0004210194579279264e-07</td><td>0.0005071825320252342081</td><td>1.000000959531155198</td><td>False</td></tr>
<tr><td>82</td><td>DMX_0014</td><td>0.0006461452563964197</td><td>0.00021712707189445966112</td><td>pc / cm3</td><td>-9.9545063116969478845e-08</td><td>0.0004584645399047981764</td><td>0.9999999947625476393</td><td>False</td></tr>
<tr><td>83</td><td>DMX_0069</td><td>-0.0002515381271240074</td><td>0.00019942850555529088364</td><td>pc / cm3</td><td>-8.9973889279542501596e-08</td><td>0.00045115861962169467089</td><td>1.0000000132706672318</td><td>False</td></tr>
<tr><td>84</td><td>ELAT</td><td>-10.071839047043065</td><td>3.361025894297e-08</td><td>deg</td><td>-1.5125678487493132707e-11</td><td>0.00045003159639913618783</td><td>0.9999926235884047247</td><td>False</td></tr>
<tr><td>85</td><td>DMX_0052</td><td>-0.0006028536128560329</td><td>0.00019631083308107704624</td><td>pc / cm3</td><td>-8.7023479245309544317e-08</td><td>0.0004432943301166092343</td><td>0.99999961140991278086</td><td>False</td></tr>
<tr><td>86</td><td>DMX_0056</td><td>-0.0004185770457364223</td><td>0.00019678664833816403518</td><td>pc / cm3</td><td>-8.545672174065099824e-08</td><td>0.00043426077156311754766</td><td>1.0000006094061713036</td><td>False</td></tr>
<tr><td>87</td><td>DMX_0003</td><td>0.0012549647637102447</td><td>0.00021051686920745200145</td><td>pc / cm3</td><td>-9.0115474821444302433e-08</td><td>0.00042806771334149380445</td><td>0.99999886989619368727</td><td>False</td></tr>
<tr><td>88</td><td>DMX_0031</td><td>-0.00018236055629661875</td><td>0.00019425115308989885179</td><td>pc / cm3</td><td>-8.30396642321096657e-08</td><td>0.00042748608134994780768</td><td>1.0000002836672390316</td><td>False</td></tr>
<tr><td>89</td><td>DMX_0035</td><td>-0.0005105159392847188</td><td>0.00019491340730905452167</td><td>pc / cm3</td><td>-8.319573408985779517e-08</td><td>0.00042683433242713119142</td><td>1.0000000969479057034</td><td>False</td></tr>
<tr><td>90</td><td>DMX_0051</td><td>-0.0007305208972559884</td><td>0.00019519430772376580928</td><td>pc / cm3</td><td>-7.9817719641508098893e-08</td><td>0.00040891417671085045606</td><td>0.999999545543776569</td><td>False</td></tr>
<tr><td>91</td><td>DMX_0065</td><td>-0.00043608356683680225</td><td>0.00019653452661633818229</td><td>pc / cm3</td><td>7.636458198428952571e-08</td><td>0.00038855555458386939688</td><td>0.9999994717140098244</td><td>False</td></tr>
<tr><td>92</td><td>JUMP1</td><td>-8.7887456483184e-06</td><td>0.0</td><td>s</td><td>-5.0316350075314342574e-11</td><td>0.0003856129036950526755</td><td>inf</td><td>True</td></tr>
<tr><td>93</td><td>DMX_0066</td><td>-0.00046369883002376897</td><td>0.00019734125214827190039</td><td>pc / cm3</td><td>-7.528661332767763725e-08</td><td>0.00038150469052011088365</td><td>1.000000020624521424</td><td>False</td></tr>
<tr><td>94</td><td>DMX_0062</td><td>-0.00046614007461461915</td><td>0.00020673830580582332997</td><td>pc / cm3</td><td>-7.57934810510893682e-08</td><td>0.00036661556626219797897</td><td>0.9999997767104176205</td><td>False</td></tr>
<tr><td>95</td><td>DMX_0016</td><td>0.0002741631091917851</td><td>0.00021262252906742716866</td><td>pc / cm3</td><td>-7.727592105219362878e-08</td><td>0.0003634418299468527712</td><td>0.9999998435828147958</td><td>False</td></tr>
<tr><td>96</td><td>DMX_0094</td><td>0.0009298023212023978</td><td>0.00019402737193812574258</td><td>pc / cm3</td><td>-6.9384752823570435e-08</td><td>0.00035760290999404376446</td><td>0.99999981199345389093</td><td>False</td></tr>
<tr><td>97</td><td>DMX_0053</td><td>-0.00042632501713326163</td><td>0.00020538452684497413838</td><td>pc / cm3</td><td>7.1548196181841876296e-08</td><td>0.0003483621540577251912</td><td>0.9999999772807683929</td><td>False</td></tr>
<tr><td>98</td><td>DMX_0041</td><td>-0.0004681066306378019</td><td>0.00019991872189383875277</td><td>pc / cm3</td><td>6.882739005093603188e-08</td><td>0.00034427686111101134898</td><td>0.9999999323838031362</td><td>False</td></tr>
<tr><td>99</td><td>DMX_0105</td><td>0.002629910506495805</td><td>0.00020525337023226432874</td><td>pc / cm3</td><td>6.717837640638976704e-08</td><td>0.0003272948762320971347</td><td>0.99999953009258713</td><td>False</td></tr>
<tr><td>100</td><td>DMX_0106</td><td>0.002503115471290146</td><td>0.00021319120384599940543</td><td>pc / cm3</td><td>6.708653808515374628e-08</td><td>0.00031467779568248189365</td><td>1.0000000242611839507</td><td>False</td></tr>
<tr><td>101</td><td>DMX_0045</td><td>3.657857770771979e-05</td><td>0.00020164095225180838538</td><td>pc / cm3</td><td>5.7593869168755837222e-08</td><td>0.00028562585390309430102</td><td>1.0000003409635465079</td><td>False</td></tr>
<tr><td>102</td><td>DMX_0061</td><td>-0.0006473483456711322</td><td>0.00021842225155475336325</td><td>pc / cm3</td><td>-6.0508353508887877115e-08</td><td>0.000277024676186527841</td><td>1.0000006056580448277</td><td>False</td></tr>
<tr><td>103</td><td>PX</td><td>0.5061242012322064</td><td>0.07348886965486496614</td><td>mas</td><td>1.9718059246720542887e-05</td><td>0.00026831354651833601603</td><td>1.0000000164253699531</td><td>False</td></tr>
<tr><td>104</td><td>DMX_0002</td><td>0.0013650356467340933</td><td>0.00020941304486956529348</td><td>pc / cm3</td><td>5.3791013572210277793e-08</td><td>0.00025686562938671977192</td><td>0.99999969937323018865</td><td>False</td></tr>
<tr><td>105</td><td>DMX_0025</td><td>-0.00011903870034693864</td><td>0.00019932837253076407611</td><td>pc / cm3</td><td>-4.843600742236969358e-08</td><td>0.00024299605122644616956</td><td>1.0000000997897962041</td><td>False</td></tr>
<tr><td>106</td><td>DMX_0006</td><td>0.0014277787660864258</td><td>0.00021694036547929127401</td><td>pc / cm3</td><td>-5.1603611456218015374e-08</td><td>0.0002378700309746827773</td><td>1.0000000142724037033</td><td>False</td></tr>
<tr><td>107</td><td>DMX_0048</td><td>-0.0006550448374455454</td><td>0.00021281515843724028449</td><td>pc / cm3</td><td>-4.7584949043936140833e-08</td><td>0.0002235975547670823369</td><td>0.99999996092908194356</td><td>False</td></tr>
<tr><td>108</td><td>DMX_0034</td><td>-0.00022038937988193576</td><td>0.00020007094225672108069</td><td>pc / cm3</td><td>-4.4135434691281935637e-08</td><td>0.00022059892452872811516</td><td>1.0000011213245350028</td><td>False</td></tr>
<tr><td>109</td><td>DMX_0059</td><td>1.1538650772853197e-05</td><td>0.00019859404338518557034</td><td>pc / cm3</td><td>-4.2320538730526282363e-08</td><td>0.00021310074566759763066</td><td>0.999999778968374442</td><td>False</td></tr>
<tr><td>110</td><td>DMX_0100</td><td>0.0016256552921773938</td><td>0.00019986402391381952897</td><td>pc / cm3</td><td>4.151053035745914943e-08</td><td>0.00020769385877749717219</td><td>1.0000000249611562531</td><td>False</td></tr>
<tr><td>111</td><td>DMX_0008</td><td>0.001169344990461584</td><td>0.0002155748755993486753</td><td>pc / cm3</td><td>-3.9906891861375692887e-08</td><td>0.00018511847333983229421</td><td>1.0000004356732830058</td><td>False</td></tr>
<tr><td>112</td><td>DMX_0017</td><td>0.0001789867729265184</td><td>0.00021197504981685315979</td><td>pc / cm3</td><td>3.7416408757044239755e-08</td><td>0.00017651326790285972249</td><td>0.99999951834728328937</td><td>False</td></tr>
<tr><td>113</td><td>DMX_0082</td><td>-5.930100454954811e-05</td><td>0.00019599958056300959127</td><td>pc / cm3</td><td>-3.3634067057408505164e-08</td><td>0.00017160275017321215216</td><td>1.0000003149303002825</td><td>False</td></tr>
<tr><td>114</td><td>DMX_0050</td><td>-0.0005879680644854433</td><td>0.0001980130914430486928</td><td>pc / cm3</td><td>-3.207174417772622188e-08</td><td>0.00016196779689665367582</td><td>1.0000004990057795862</td><td>False</td></tr>
<tr><td>115</td><td>DMX_0024</td><td>-6.464357906175583e-05</td><td>0.00019594538945981657802</td><td>pc / cm3</td><td>-3.0733074786039684713e-08</td><td>0.00015684510296856083583</td><td>1.000000040794174927</td><td>False</td></tr>
<tr><td>116</td><td>DMX_0092</td><td>0.0013207295138539894</td><td>0.00019585216459019454951</td><td>pc / cm3</td><td>-2.987272381023420298e-08</td><td>0.00015252690146540150845</td><td>1.0000000598805378615</td><td>False</td></tr>
<tr><td>117</td><td>DMX_0058</td><td>-0.0005377581468744793</td><td>0.00019927530538964258904</td><td>pc / cm3</td><td>-2.7240830902737663e-08</td><td>0.00013669948140073714471</td><td>1.0000003494124634074</td><td>False</td></tr>
<tr><td>118</td><td>DMX_0089</td><td>0.0007495614446295846</td><td>0.00021586616414944812654</td><td>pc / cm3</td><td>2.664672352724588994e-08</td><td>0.00012344094607063047083</td><td>1.0000000318049040438</td><td>False</td></tr>
<tr><td>119</td><td>DMX_0032</td><td>-6.265675469663684e-05</td><td>0.00019561483985536690729</td><td>pc / cm3</td><td>2.0624768732425491705e-08</td><td>0.00010543560369793503542</td><td>1.0000001069330806125</td><td>False</td></tr>
<tr><td>120</td><td>DMX_0040</td><td>-0.0005242449385393532</td><td>0.00020212647115737782458</td><td>pc / cm3</td><td>-1.3087848262614831807e-08</td><td>6.4750787898654266965e-05</td><td>0.99999999750656676234</td><td>False</td></tr>
<tr><td>121</td><td>DMX_0001</td><td>0.0016484372168232325</td><td>0.00022434462780433157077</td><td>pc / cm3</td><td>-1.1670286121810355406e-08</td><td>5.2019458794390748143e-05</td><td>1.0000004809807354622</td><td>False</td></tr>
<tr><td>122</td><td>DMX_0027</td><td>-0.00018288082535181414</td><td>0.00019391445756469536201</td><td>pc / cm3</td><td>-9.576702027013850663e-09</td><td>4.938621981715206116e-05</td><td>1.0000001643637170812</td><td>False</td></tr>
<tr><td>123</td><td>DMX_0083</td><td>8.544780315309648e-06</td><td>0.00020486177918444288125</td><td>pc / cm3</td><td>-3.8797037898631337458e-09</td><td>1.8938153350558018293e-05</td><td>1.0000001513452474455</td><td>False</td></tr>
<tr><td>124</td><td>DMX_0044</td><td>-0.0003390023662491028</td><td>0.00021062295971768858391</td><td>pc / cm3</td><td>1.0720102164903794195e-09</td><td>5.0897120519399367266e-06</td><td>1.0000001883055240626</td><td>False</td></tr>
</table><style>table.dataTable {clear: both; width: auto !important; margin: 0 !important;}
.dataTables_info, .dataTables_length, .dataTables_filter, .dataTables_paginate{
display: inline-block; margin-right: 1em; }
.paginate_button { margin-right: 5px; }
</style>
<script>

var astropy_sort_num = function(a, b) {
    var a_num = parseFloat(a);
    var b_num = parseFloat(b);

    if (isNaN(a_num) && isNaN(b_num))
        return ((a < b) ? -1 : ((a > b) ? 1 : 0));
    else if (!isNaN(a_num) && !isNaN(b_num))
        return ((a_num < b_num) ? -1 : ((a_num > b_num) ? 1 : 0));
    else
        return isNaN(a_num) ? -1 : 1;
}

require.config({paths: {
    datatables: 'https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min'
}});
require(["datatables"], function(){
    console.log("$('#table140672074185640-254392').dataTable()");

jQuery.extend( jQuery.fn.dataTableExt.oSort, {
    "optionalnum-asc": astropy_sort_num,
    "optionalnum-desc": function (a,b) { return -astropy_sort_num(a, b); }
});

    $('#table140672074185640-254392').dataTable({
        order: [],
        pageLength: 50,
        lengthMenu: [[10, 25, 50, 100, 500, 1000, -1], [10, 25, 50, 100, 500, 1000, 'All']],
        pagingType: "full_numbers",
        columnDefs: [{targets: [0, 3, 5, 6, 7], type: "optionalnum"}]
    });
});
</script>




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
<table id="table140672074185640">
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
<table id="table140672075763048" class="table-striped table-bordered table-condensed">
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



### The residual difference between PINT and TEMPO2 is at the level of ~1ns. Let us exam the remaining residual difference more.

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


![png](paper_validation_example_files/paper_validation_example_67_0.png)



```python

```
