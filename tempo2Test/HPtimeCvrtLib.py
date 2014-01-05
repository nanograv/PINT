import mpmath as mp
import sys
import math as ma
import struct
import spice 
##### Global setting
mp.mp.dps = 30

class glb(object):
	def __init__(self):
		self.JDpara = None
		self.ncon = None
		self.ipt1 = None
		self.ipt2 = None
		self.reclen = None
		self.L_c = None
		self.f = None
global JDpara
#################### Functions ##################################
######## Load spice files #############
#leap = '/home/jing/Research/SPICE_kernals/lsk/naif0009.tls'

#spk = '/home/jing/Research/SPICE_kernals/spk/de405.bsp'

#pcktpc = '/home/jing/Research/SPICE_kernals/pck/pck00010.tpc'

#pckbpc = '/home/jing/Research/SPICE_kernals/pck/earth_latest_high_prec.bpc'


#spice.furnsh(leap)
#spice.furnsh(spk)
#spice.furnsh(pcktpc)
#spice.furnsh(pckbpc)

######## IF time ephemeris method ############# 
def initIFParas(filename,glb):
	glb.f = open(filename,'r')
	byte_prt = 0               
# Set the byte pointor in the begining of the file
	ephmdata_bin = glb.f.read()
	ephmhead1 = struct.unpack('252c',ephmdata_bin[byte_prt:byte_prt+1*252])
# Read the first part of header description
#
	byte_prt=byte_prt+1*252   # Move the byte pointor
	glb.f.seek(byte_prt)          # Locate the byte pointor 
	ephmdata_bin = glb.f.read()
	ephmhead2 = struct.unpack('12c',ephmdata_bin[0:1*12])
# Read the second part of header description
#
	byte_prt=byte_prt+1*12       # Move the byte pointor
	glb.f.seek(byte_prt)         # Locate the byte pointor
        ephmdata_bin = glb.f.read()
	glb.JDpara = struct.unpack('>3d',ephmdata_bin[0:24])

# Get the start time in JD
#
        byte_prt=byte_prt+3*8    # Move the byte pointor
	glb.f.seek(byte_prt)         # Locate the byte pointor 
	ephmdata_bin = glb.f.read()
	glb.ncon = struct.unpack('>i',ephmdata_bin[0:4])
# Get the number of coefficients
#
	byte_prt=byte_prt+1*4    # Move the byte pointor
	glb.f.seek(byte_prt)         # Locate the byte pointor
	ephmdata_bin = glb.f.read()
	glb.ipt0 = struct.unpack('>3i',ephmdata_bin[0:4*3])
# Get the ipt0
#
	byte_prt=byte_prt+4*3   # Move the byte pointor
	glb.f.seek(byte_prt)         # Locate the byte pointor
	ephmdata_bin = glb.f.read()	
	glb.ipt1 = struct.unpack('>3i',ephmdata_bin[0:4*3])
# Get the ipt1
#
	glb.reclen = 4*2*(glb.ipt1[0]-1+3*glb.ipt1[1]*glb.ipt1[2])
	
	glb.f.seek(glb.reclen)
	ephmdata_bin = glb.f.read()
	double_in,glb.L_c =  struct.unpack('>2d',ephmdata_bin[0:16])

def readCoeff(JDeph0,JDeph1,glb):
	
	ncoeff = glb.reclen/8
	irec = int(mp.floor((JDeph0-glb.JDpara[0])/glb.JDpara[2]))+2
	if (JDeph0==glb.JDpara[1]):
		irec=irec-1

	t=mp.matrix(2,1)
	t[0]= (JDeph0-(glb.JDpara[0]+glb.JDpara[2]*(irec-2))+JDeph1)/glb.JDpara[2];
	t[1]=glb.JDpara[2]

	glb.f.seek(glb.reclen*irec)
	ephmdata_bin=glb.f.read()
	ncoeffs = '>'+str(ncoeff)+'d'
	
	buff = struct.unpack(ncoeffs,ephmdata_bin[0:ncoeff*8])
		
	lengthbuff = len(buff)
	return buff,t
#####
def interp(coef,t,ncf,ncm,na,ifl,posvel):
	
	pc=mp.matrix(18,1)
	vc=mp.matrix(18,1)
	np=2
	nv=3
	first=1
	twot=0.0
	

	
	if(first):
		pc[0]=1.0
                pc[1]=0.0
             	vc[1]=1.0
             	first=0
	
	dna = mp.mpf(na)
	dt1 = int(t[0])
	temp = dna*t[0]
	l = int(temp-dt1)
# tc is the normalized chebyshev time (-1 <= tc <= 1)
	
	tc=2.0*(mp.fmod(temp,1)+dt1)-1.0
 	if(tc !=pc[1]):
		np=2
		nv=3
		pc[1]=tc
		twot=tc+tc
	
	if(np<ncf):
		for i in range(np,ncf):
			pc[i] = twot*pc[i-1]-pc[i-2]
			np = ncf

	for i2 in range(ncm):
		posvel[i2]=0.0
		for j in range(ncf,-1,-1):			
			posvel[i2]=posvel[i2]+pc[j]*coef[j+i2*ncf+l*ncf*ncm+2]

#			print '%23.17f  %23.17f  %23.17f\n' %(posvel[i2],pc[j],coef[j+i2*ncf+l*ncf*ncm+2])		

		if(ifl<= 1): 
			return
	return posvel[0]
#####

######## mjdutc to tdt  #############
def mjd2tdt(mjd):	
    dt = 10.0
    if (mjd >= 41499.0): 
	dt = 11.0	#/* 1972 Jul 1 */

    if (mjd >= 41683.0):
	dt = 12.0;	#/* 1973 Jan 1 */

    if (mjd >= 42048.0):
	dt = 13.0;	#/* 1974 Jan 1 */

    if (mjd >= 42413.0):
	dt = 14.0;	#/* 1975 Jan 1 */

    if (mjd >= 42778.0):
	dt = 15.0;	#/* 1976 Jan 1 */

    if (mjd >= 43144.0):
	dt = 16.0;	#/* 1977 Jan 1 */

    if (mjd >= 43509.0):
	 dt = 17.0;	#/* 1978 Jan 1 */

    if (mjd >= 43874.0):
	 dt = 18.0;	#/* 1979 Jan 1 */

    if (mjd >= 44239.0):
	 dt = 19.0;	#/* 1980 Jan 1 */

    if (mjd >= 44786.0):
	 dt = 20.0;	#/* 1981 Jul 1 */

    if (mjd >= 45151.0):
	 dt = 21.0;	#/* 1982 Jul 1 */

    if (mjd >= 45516.0):
	 dt = 22.0;	#/* 1983 Jul 1 */

    if (mjd >= 46247.0):
	 dt = 23.0;	#/* 1985 Jul 1 */

    if (mjd >= 47161.0):
	 dt = 24.0;	#/* 1988 Jan 1 */

    if (mjd >= 47892.0):
	 dt = 25.0;	#/* 1990 Jan 1 */

    if (mjd >= 48257.0):
	 dt = 26.0;	#/* 1991 Jan 1 */

    if (mjd >= 48804.0):
	 dt = 27.0;	#/* 1992 July 1 */

    if (mjd >= 49169.0):
	 dt = 28.0;	#/* 1993 July 1 */

    if (mjd >= 49534.0):
	 dt = 29.0;	#/* 1994 July 1 */

    if (mjd >= 50083.0):
	 dt = 30.0;	#/* 1996 Jan 1 */

    if (mjd >= 50630.0):
	 dt = 31.0;	#/* 1997 Jul 1 */

    if (mjd >= 51179.0):
	 dt = 32.0;	#/* 1999 Jan 1 */

    if (mjd >= 53736.0):
	 dt = 33.0;	#/* 2006 Jan 1 */

    if (mjd >= 54832.0):
	 dt = 34.0;	#/* 2009 Jan 1 */

    if (mjd >= 56109.0):
         dt = 35.0;	#/* 2012 July 1 */
		
    delta_TT = mp.mpf(dt)+32.184 
    delta_TT_DAY = mp.mpf(delta_TT)/mp.mpf(86400.0)
    delta_TT_DAY = mp.mpf(delta_TT_DAY)
    mjd_tt = mp.mpf(mjd)+delta_TT_DAY		
    return mjd_tt

######## jdutc to et  #############
def jdutc2et(jdutc):
	filename = 'TIMEEPH_short.te405'
	const = glb()
	initIFParas(filename,const)
	ncf=7
	ncm=1
	na=8
	ifl=2 
	posvel=mp.matrix(6,1)
	jdutc = mp.mpf(jdutc)
	mjdutc = jdutc-mp.mpf(2400000.5)
	mjd_tt = mjd2tdt(mjdutc)
        print mjd_tt
	mjd_tt = mp.mpf(mjd_tt)
	JDeph0,JDeph1=FracMjdTT(mjd_tt)
	coef,t = readCoeff(JDeph0,JDeph1,const)
	deltaT = interp(coef,t,ncf,ncm,na,ifl,posvel)
#	print deltaT
	jdutc = mp.mpf(jdutc)	
	L_C = mp.mpf(const.L_c)
	tdb_correction = mp.mpf('-0.00006556451800000')+(deltaT*mp.mpf(86400.0))/mp.mpf(1.0-L_C)
#	print tdb_correction                 
	et = (mjd_tt-mp.mpf(51544.5))*mp.mpf(86400.0)+mp.mpf(tdb_correction)
#	print mjd_tt 
        const.f.close()
	return et

######## Take the integer and fractional part of 
def FracMjdTT(mjd_tt):

	JDeph0=2400000.0+int(mjd_tt)
	JDeph1=0.5+(mjd_tt-int(mjd_tt))
	
	whole0 = mp.floor(JDeph0-0.5);
	frac0 = JDeph0-0.5-whole0

	whole1 = mp.floor(JDeph1)
	frac1 = JDeph1-whole1;
	whole0 =whole0+whole1 + 0.5;
	frac0 =frac0+frac1;
	whole1 = mp.floor(frac0);
	frac1 = frac0-whole1;
	whole0 =whole0+whole1;
	JDeph0 = whole0;
	JDeph1 = frac1;
	return JDeph0, JDeph1

######## Et 2 JDUTC ################
def et2mjdutc(et):

	e = mp.mpf('1e-12')	
	et = mp.mpf(et)
	utc = spice.et2utc(et,'J',10)
	utcjd = mp.mpf(utc[3:])
	utcjdexc = mp.findroot(lambda x: jdutc2et(x)-et,utcjd)
	
	return utcjdexc-mp.mpf(2400000.5)

######## 

 
#################### Main function  ################################
def main():
	f = 'TIMEEPH_short.te405'
	consts = glb()
	initIFParas(f,consts)
	JDeph0= 2452752.5;
	JDeph1= 0.50074332816197753
	coeff,t = readCoeff(JDeph0,JDeph1,consts)
	ncf=7
	ncm=1
	na=8
	ifl=2 
	posvel = mp.matrix(6,1)
#	dt = interp(coeff,t,)
	jd = mp.mpf('2451545.0')	
	frac,intger = FracMjdTT(jd)
	et = jdutc2et(jd)
#	utc = et2mjdutc(et)
#	print utc
if __name__ == "__main__":
	main()




	
