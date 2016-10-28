# This program is designed to predict the pulsar's phase and pulse-period over a
# given interval using polynomial expansion. The return will be some necessary
# information and the polynomial coefficients

import functools
from ..phase import Phase
import numpy as np
import pint.toa as toa
import pint.utils as utils
import astropy.units as u
import astropy.constants as const
import astropy.time as at
from .parameter import Parameter
from .timing_model import TimingModel, MissingParameter, Cache
import astropy.table as table
from astropy.io import registry
MIN_PER_DAY = 60.0*24.0


class polycoEntry:
    """
    Polyco Entry class:
    A Class for one Polyco entry.
    Referenced from polyco.py authored by
        Paul S. Ray <paul.ray@nrl.navy.mil>
        Matthew Kerr <matthew.kerr@gmail.com>
    Parameters
    ---------
    tmid : float
        Middle point of the time span in mjd
    mjdspan : float
        Time span in mjd
    rphase : float
        Reference phase
    f0 : float
        Reference spin frequency
    ncoeff : int
        Number of coefficients
    obs : str
        Observatory code
    """
    def __init__(self,tmid,mjdspan,rphaseInt,rphaseFrac,f0,ncoeff,coeffs,obs):
        self.tmid = tmid*u.day
        self.mjdspan = mjdspan*u.day
        self.tstart = np.longdouble(self.tmid) - np.longdouble(self.mjdspan)/2.0
        self.tstop = np.longdouble(self.tmid) + np.longdouble(self.mjdspan)/2.0
        self.rphase = Phase(rphaseInt,rphaseFrac)
        self.f0 = np.longdouble(f0)
        self.ncoeff = ncoeff
        self.coeffs = np.longdouble(coeffs)
        self.obs = obs

    def __str__(self):
        return("Middle Point mjd : "+repr(self.tmid)+"\n"+
               "Time Span in mjd : "+repr(self.mjdspan)+"\n"+
               "Reference Phase : "+repr(self.rphase)+"\n"+
               "Number of Coefficients : "+repr(self.ncoeff)+"\n"+
               "Coefficients : "+repr(self.coeffs))

    def valid(self,t):
        '''Return True if this polyco entry is valid for the time given (MJD)'''
        return t>=(self.tmid-self.mjdspan/2.0) and t<(self.tmid+self.mjdspan/2.0)

    def evalabsphase(self,t):
        '''Return the phase at time t, computed with this polyco entry'''
        dt = (np.longdouble(t)-self.tmid.value)*np.longdouble(1440.0)
        # Compute polynomial by factoring out the dt's
        phase = Phase(self.coeffs[self.ncoeff-1]) # Compute phase using two long double
        for i in range(self.ncoeff-2,-1,-1):
            pI = Phase(dt*phase.int)
            pF = Phase(dt*phase.frac)
            c = Phase(self.coeffs[i])
            phase = pI+pF+c

        # Add DC term
        phase += self.rphase +Phase(dt*60.0*self.f0)
        return(phase)

    def evalphase(self,t):
        '''Return the phase at time t, computed with this polyco entry'''
        return(self.evalabsphase(t).frac)

    def evalfreq(self,t):
        '''Return the freq at time t, computed with this polyco entry'''
        dt = (np.longdouble(t)-self.tmid.value)*np.longdouble(1440.0)
        s = np.longdouble(0.0)
        for i in range(1,self.ncoeff):
            s += np.longdouble(i) * self.coeffs[i] * dt**(i-1)
        freq = self.f0 + s/60.0
        return(freq)

    def evalfreqderiv(self,t):
        """ Return the frequency derivative at time t."""
        dt = (np.longdouble(t)-self.tmid.value)*np.longdouble(1440.0)
        s = np.longdouble(0.0)
        for i in range(2,self.ncoeff):
            # Change to long double
            s += np.longdouble(i) * np.longdouble(i-1) * self.coeffs[i] * dt**(i-2)
        freqd = s/(60.0*60.0)
        return(freqd)

# Read polycos file data to table
def tempo_polyco_table_reader(filename):
    """
    Read tempo style polyco file to an astropy table

    Parameters
    ---------
    filename : str
        Name of the input poloco file.

    Tempo style:
    The polynomial ephemerides are written to file 'polyco.dat'.  Entries
    are listed sequentially within the file.  The file format is:

    Line  Columns     Item
    ----  -------   -----------------------------------
     1       1-10   Pulsar Name
            11-19   Date (dd-mmm-yy)
            20-31   UTC (hhmmss.ss)
            32-51   TMID (MJD)
            52-72   DM
            74-79   Doppler shift due to earth motion (10^-4)
            80-86   Log_10 of fit rms residual in periods
     2       1-20   Reference Phase (RPHASE)
            21-38   Reference rotation frequency (F0)
            39-43   Observatory number
            44-49   Data span (minutes)
            50-54   Number of coefficients
            55-75   Observing frequency (MHz)
            76-80   Binary phase
     3*      1-25   Coefficient 1 (COEFF(1))
            26-50   Coefficient 2 (COEFF(2))
            51-75   Coefficient 3 (COEFF(3))

    * Subsequent lines have three coefficients each, up to NCOEFF

    One polyco file could include more then one entrie

    The pulse phase and frequency at time T are then calculated as:
    DT = (T-TMID)*1440
    PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
    FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

    Reference:
        http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
    """
    f = open(filename, "r")
    # Read entries to the end of file
    entries = []
    while True:
        # Read first line
        line1 = f.readline()
        if len(line1) == 0:
            break

        fields = line1.split()
        psrname = fields[0].strip()
        date = fields[1].strip()
        utc = fields[2]
        tmid = utils.MJD_string2longdouble(fields[3])
        dm = float(fields[4])
        doppler = float(fields[5])
        logrms = float(fields[6])
        # Read second line
        line2 = f.readline()
        fields = line2.split()
        refPhaseInt,refPhaseFrac = fields[0].split('.')
        refPhaseInt = np.longdouble(refPhaseInt)
        refPhaseFrac = np.longdouble('.'+refPhaseFrac)
        if refPhaseInt<0:
            refPhaseFrac = -refPhaseFrac

        refF0 = np.longdouble(fields[1])
        obs = fields[2]
        mjdSpan = np.longdouble(fields[3])/MIN_PER_DAY   # Here change to constant
        nCoeff = int(fields[4])
        obsfreq = float(fields[5].strip())

        try:
            binaryPhase = np.longdouble(fields[6])
        except:
            binaryPhase = np.longdouble(0.0)

        # Read coefficients
        nCoeffLines = nCoeff/3

        if nCoeff%3>0:
            nCoeffLines += 1
        coeffs = []

        for i in range(nCoeffLines):
            line = f.readline()
            for c in line.split():
                coeffs.append(np.longdouble(c))
        coeffs = np.array(coeffs)


        tmid = tmid*u.day
        mjdspan = mjdSpan*u.day
        tstart = np.longdouble(tmid) - np.longdouble(mjdspan)/2.0
        tstop = np.longdouble(tmid) + np.longdouble(mjdspan)/2.0
        rphase = Phase(refPhaseInt, refPhaseFrac)
        refF0 = np.longdouble(refF0)
        coeffs = np.longdouble(coeffs)

        entries.append((psrname, date, utc, tmid.value, dm, doppler, logrms,
                        binaryPhase, obs, obsfreq, mjdSpan, tstart, tstop,
                        rphase, refF0, nCoeff, coeffs))
    entry_list  = []
    for ii in range(len(entries[0])):
         entry_list.append([t[ii] for t in entries])

    #Construct the polyco data table
    pTable = table.Table(entry_list,
                         names = ( 'psr','date','utc','tmid','dm',
                         'dopper','logrms','binary_phase', 'obs', 'obsfreq',
                         'mjd_span', 't_start', 't_stop','ref_phase','ref_freq',
                         'num_coeffs', 'coeffs'),
                         meta={'name': 'Ployco Data Table'})

    pTable['index'] = np.arange(len(entries))
    return pTable

def tempo_polyco_table_writer(polycoTable, filename = 'polyco.dat'):
    """
    Write tempo style polyco file from an astropy table

    Parameters
    ---------
    polycoTalbe: astropy table
        Polycos style table
    filename : str
        Name of the output poloco file.


    Tempo style polyco file:
    The polynomial ephemerides are written to file 'polyco.dat'.  Entries
    are listed sequentially within the file.  The file format is:

    Line  Columns     Item
    ----  -------   -----------------------------------
     1       1-10   Pulsar Name
            11-19   Date (dd-mmm-yy)
            20-31   UTC (hhmmss.ss)
            32-51   TMID (MJD)
            52-72   DM
            74-79   Doppler shift due to earth motion (10^-4)
            80-86   Log_10 of fit rms residual in periods
     2       1-20   Reference Phase (RPHASE)
            21-38   Reference rotation frequency (F0)
            39-43   Observatory number
            44-49   Data span (minutes)
            50-54   Number of coefficients
            55-75   Observing frequency (MHz)
            76-80   Binary phase
     3*      1-25   Coefficient 1 (COEFF(1))
            26-50   Coefficient 2 (COEFF(2))
            51-75   Coefficient 3 (COEFF(3))

    * Subsequent lines have three coefficients each, up to NCOEFF

    One polyco file could include more then one entrie

    The pulse phase and frequency at time T are then calculated as:
    DT = (T-TMID)*1440
    PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
    FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

    Reference:
        http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
    """
    f = open(filename,'w')
    try:
        lenTable = len(polycoTable)
        if lenTable == 0:
            errorMssg = ("No sufficent polyco data."+
                         " Plese make sure polycoTable has data.")
            raise AttributeError(errorMssg)

    except:
        errorMssg = "No sufficent polycoTable. "
        raise AttributeError(errorMssg)

    for i in range(lenTable):
        entry = polycoTable['entry'][i]
        psrname = polycoTable['psr'][i].ljust(10)
        dateDMY = polycoTable['date'][i].ljust(10)
        utcHMS = polycoTable['utc'][i][0:9].ljust(10)
        tmidMjd = utils.longdouble2string(entry.tmid.value)+' '
        dm = str(polycoTable['dm'][i]).ljust(72-52+1)
        dshift = str(polycoTable['dopper'][i]).ljust(79-74+1)
        logrms = str(polycoTable['logrms'][i]).ljust(80-86+1)

        line1 = psrname+dateDMY+utcHMS+tmidMjd+dm+dshift+logrms+'\n'
        # Get the reference phase
        rph = (entry.rphase.int+entry.rphase.frac).data[0]
        rphase  = utils.longdouble2string(rph)[0:19].ljust(20)
        f0 = ('%.12lf' % entry.f0).ljust(38-21+1)
        obs = entry.obs.ljust(43-39+1)
        tspan = (str(entry.mjdspan.to('min')).split())[0].ljust(49-44+1)
        ncoeff = str(entry.ncoeff).ljust(54-50+1)
        obsfreq = str(polycoTable['obsfreq'][i]).ljust(75-55+1)
        binPhase = str(polycoTable['binary_phase'][i]).ljust(80-76+1)
        line2 = rphase+f0+obs+tspan+ncoeff+obsfreq+binPhase+'\n'

        coeffBlock = ""
        for j,coeff in enumerate(entry.coeffs):
            coeffBlock += ('%.17e' % coeff).ljust(25)
            if (j+1)%3==0:
                coeffBlock += '\n'

        f.write(line1+line2+coeffBlock)
    f.close()


class Polycos(object):
    """
    A class for polycos model. Ployco is a fast phase calculator. It fits a set
    of data using polynomials.


    """
    def __init__(self):
        self.mjdMid = None
        self.mjdSpan = None
        self.tStart = None
        self.tStop = None
        self.ncoeff = None
        self.coeffs = None
        self.obs = None
        self.fileName = None
        self.fileFormat = None
        self.newFileName = None
        self.polycoTable = None
        self.polycoFormat = [{'format': 'tempo',
                            'read_method' : tempo_polyco_table_reader,
                            'write_method' : tempo_polyco_table_writer},]

        # Register the table built-in reading and writing format
        for fmt in self.polycoFormat:
            if fmt['format'] not in registry.get_formats()['Format']:
                if fmt['read_method'] != None:
                   registry.register_reader(fmt['format'], table.Table,
                                            fmt['read_method'])

                if fmt['write_method'] != None:
                    registry.register_writer(fmt['format'], table.Table,
                                            fmt['write_method'])

    def add_polyco_file_format(self, formatName, methodMood, readMethod = None,
                                writeMethod = None):
        """
        Add a polyco file format and its reading/writting method to the class.
        Then register it to the table reading.
        Parameters
        ---------
        formatName : str
            The name for the format.
        methodMood : str
            ['r','w','rw']. 'r'  represent as reading
                            'w'  represent as writting
                            'rw' represent as reading and writting
        readMethod : method
            The method for reading the file format.
        writeMethod : method
            The method for writting the file to disk.
        """
        # Check if the format already exist.
        if (formatName in [f['format'] for f in self.polycoFormat]
            or formatName in registry.get_formats()['Format']):
            errorMssg = 'Format name \''+formatName+ '\' is already exist. '
            raise Exception(errorMssg)

        pFormat = {'format' : formatName}

        if methodMood == 'r':
            if readMethod == None:
                raise BaseException('Argument readMethod should not be \'None\'.')

            pFormat['read_method'] = readMethod
            pFormat['write_method'] = writeMethod
            registry.register_reader(pFormat['format'], table.Table,
                                    pFormat['read_method'])
        elif methodMood == 'w':
            if writeMethod == None:
                raise BaseException('Argument writeMethod should not be \'None\'.')

            pFormat['read_method'] = readMethod
            pFormat['write_method'] = writeMethod
            registry.register_writer(pFormat['format'], table.Table,
                                    pFormat['write_method'])
        elif methodMood == 'rw':
            if readMethod == None or writeMethod == None:
                raise BaseException('Argument readMethod and writeMethod'
                                    'should not be \'None\'.')

            pFormat['read_method'] = readMethod
            pFormat['write_method'] = writeMethod

            registry.register_reader(pFormat['format'], table.Table,
                                    pFormat['read_method'])
            registry.register_writer(pFormat['format'], table.Table,
                                    pFormat['write_method'])

        self.polycoFormat.append(pFormat)


    def generate_polycos(self, model, mjdStart, mjdEnd, obs,
                         segLength, ncoeff, obsFreq, maxha,
                         method = "TEMPO",numNodes = 20):
        """
        Generate the polyco file data file.

        Parameters
        ---------
        model : TimingModel
            TimingModel for generate the Polycos with parameters
            setup.

        mjdStart : float / nump longdouble
            Start time of polycos in mjd

        mjdEnd : float / nump longdouble
            Ending time of polycos in mjd

        obs : str
            Observatory code

        segLength :
            Length of polyco segement [unit: minutes]

        ncoeff :
            number of coefficents

        obsFreq :
            observing frequency

        maxha :
            Maximum hour angle

        method : sting optional ['TEMPO','TEMPO2',...] Default TEMPO
            Method to generate polycos. Now it is only support the TEMPO method.

        numNodes : int optional. Default 20
            Number of nodes for fitting. It can not be less then the number of
            coefficents.
        Return
        ---------

        A polyco table.


        """
        mjdStart = np.longdouble(mjdStart)*u.day
        mjdEnd = np.longdouble(mjdEnd)*u.day
        timeLength = mjdEnd-mjdStart
        segLength = np.longdouble(segLength)*u.min
        obsFreq = float(obsFreq)
        month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
                 'Sep','Oct','Nov','Dec']
        # Alocate memery
        coeffs = np.longdouble(np.zeros(ncoeff))
        entryList = []
        entryIntvl = np.arange(mjdStart.value,mjdEnd.value,
                                segLength.to('day').value)
        if entryIntvl[-1] < mjdEnd.value:
            entryIntvl = np.append(entryIntvl, mjdEnd.value)

        # Make sure the number of nodes is bigger then number of coeffs.
        if numNodes < ncoeff:
            numNodes = ncoeff+1
    	# generate the ploynomial coefficents
    	if method == "TEMPO":
    	    # Using tempo1 method to create polycos
            for i in range(len(entryIntvl)-1):
                tStart = entryIntvl[i]
                tStop = entryIntvl[i+1]
                nodes = np.linspace(tStart,tStop,numNodes)
                tmid = ((tStart+tStop)/2.0)*u.day
                toaMid = toa.get_TOAs_list([toa.TOA((np.modf(tmid.value)[1],
                                    np.modf(tmid.value)[0]),obs = obs,
                                    freq = obsFreq),])
                refPhase = model.phase(toaMid.table)
                mjdSpan = ((tStop-tStart)*u.day).to('min')
                # Create node toas(Time sample using TOA class)
                toaList = [toa.TOA((np.modf(toaNode)[1],
                                    np.modf(toaNode)[0]),obs = obs,
                                    freq = obsFreq) for toaNode in nodes]

                toas = toa.get_TOAs_list(toaList)

                ph = model.phase(toas.table)
                dt = (nodes*u.day - tmid).to('min') # Use constant
                rdcPhase = ph-refPhase
                rdcPhase = rdcPhase.int-dt.value*model.F0.value*60.0+rdcPhase.frac
                dtd = dt.value.astype(float)  # Trancate to double
                rdcPhased = rdcPhase.astype(float)
                coeffs = np.polyfit(dtd,rdcPhased,ncoeff-1)
                coeffs = coeffs[::-1]
                midTime = at.Time(int(tmid.value),np.modf(tmid.value)[0],
                                  format = 'mjd',scale = 'utc')
                date,hms = midTime.iso.split()
                yy,mm,dd = date.split('-')
                date = dd+'-'+month[int(mm)-1]+'-'+yy[2:4]
                hms = hms.replace(':',"")
                entry = polycoEntry(tmid.value,mjdSpan.to('day').value,
                                refPhase.int,refPhase.frac, model.F0.value, ncoeff,
                                coeffs,obs)
                entryList.append((model.PSR.value, date, hms, tmid.value,
                                  model.DM.value,0.0,0.0,0.0,obsFreq,entry))

            pTable = table.Table(rows = entryList, names = ('psr','date','utc',
                                  'tmid','dm','dopper','logrms','binary_phase',
                                  'obsfreq','entry'),
                                   meta={'name': 'Ployco Data Table'})
            self.polycoTable = pTable

    	else:
    		#  Reading from an old polycofile
    		pass


    def read_polyco_file(self,filename,format):
        """
        Read polyco file from one type of format to a table.

        Parameters
        ---------
        filename : str
            The name of the polyco file.
        format : str
            The format of the file.

        Return
        ---------
        Polycos Table with read_in data.
        """
        self.fileName = filename

        if format not in [f['format'] for f in self.polycoFormat]:
            raise Exception('Unknown polyco file format \''+ format +'\'\n'
                            'Plese use function \'self.add_polyco_file_format()\''
                            ' to register the format\n')
        else:
            self.fileFormat = format

        self.polycoTable = table.Table.read(filename, format = format)

    def write_polyco_file(self,format,filename=None):
        """ Write Polyco table to a file.
        """

        if format not in [f['format'] for f in self.polycoFormat]:
             raise Exception('Unknown polyco file format \''+ format +'\'\n'
                            'Plese use function \'self.add_polyco_file_format()\''
                            ' to register the format\n')
        if filename is not None:
            self.polycoTable.write(filename,format = format)
        else:
            self.polycoTable.write(format = format)

    def find_entry(self,t):
        """Find the right entry for the input time.
        """
        if not isinstance(t, (np.ndarray, list)):
            t = np.array([t,])
        # Check if polyco table exist
        try:
            lenEntry = len(self.polycoTable)
            if lenEntry == 0:
                errorMssg = ("No sufficent polyco data."
                             "Plese read or generate polyco data correctlly.")
                raise AttributeError(errorMssg)

        except:
            errorMssg = "No sufficent polyco data. Plese read or generate polyco data correctlly."
            raise AttributeError(errorMssg)

        startIndex = np.searchsorted(self.polycoTable['t_start'], t)
        entryIndex = startIndex-1
        overFlow = np.where(t > self.polycoTable['t_stop'][entryIndex])[0]
        if overFlow.size!=0:
            errorMssg = "Input time "
            for i in overFlow:
                errorMssg += str(t[i]) + " "
            errorMssg += " may be not coverd by entries."
            raise ValueError(errorMssg)

        return entryIndex

    def eval_phase(self,t):
        if not isinstance(t, np.ndarray) and not isinstance(t,list):
            t = np.array([t,])
        return self.eval_abs_phase(t).frac

    def eval_abs_phase(self,t):
        '''
        Polyco evalate absolute phase for a time array.
        Parameters
        ---------
        t: numpy.ndarray or a single number.
           An time array in MJD. Time sample should be in order
        Returns
        ---------
        out: PINT Phase class
             Polyco evaluated absolute phase for t.
        '''
        if not isinstance(t, (np.ndarray, list)):
            t = np.array([t,])

        entryIndex = self.find_entry(t)
        phaseInt = np.array([])
        phaseFrac= np.array([])
        # Compute phase for time in each entry
        for i in range(len(self.polycoTable)):
            mask = np.where(entryIndex==i) # Build mask for time in each entry
            t_in_entry = t[mask]
            if len(t_in_entry) == 0:
                continue
            # Calculate the phase as an array
            absp = self.polycoTable['entry'][i].evalabsphase(t_in_entry)
            phaseInt = np.hstack((phaseInt,absp.int))
            phaseFrac = np.hstack((phaseFrac,absp.frac))
            # Maybe add sort function here, since the time has been masked.
        absPhase = Phase(phaseInt,phaseFrac)

        return absPhase

    def eval_spin_freq(self,t):
        """FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)
        """
        if not isinstance(t, np.ndarray) and not isinstance(t,list):
            t = np.array([t,])

        entryIndex = self.find_entry(t)
        poly_result = np.longdouble(np.zeros(len(t)))

        dt = (np.longdouble(t) - self.polycoTable[entryIndex]['tmid']) * np.longdouble(1440.0)
        s = np.longdouble(0.0)
        for ii, (tt, eidx) in enumerate(zip(dt, entryIndex)):
            coeffs = self.polycoTable[eidx]['coeffs']
            coeffs = np.longdouble(range(len(coeffs))) * coeffs
            coeffs = coeffs[::-1][:-1]
            poly_result[ii] = np.polyval(coeffs, tt)
        spinFreq = self.polycoTable[entryIndex]['ref_freq'] + poly_result / np.longdouble(60.0)

        return spinFreq
