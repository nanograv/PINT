import numpy as np
import functools
import collections
from astropy import log
from .timing_model import Cache
from scipy.optimize import newton
import astropy.units as u 
from pint import ls
class DDmdoel(object):
	"""A frist test of DD model. 
	   See T. Damour and N. Deruelle Annales de l' I.H.P.(1986)
	   Parameters
	   @n
	   @PB
	   @ECC
	   @T0
	   @eT
	   @ar 
	   @sini
	   @omg0
	   @k
	   @deltaR
	   @deltaTheta
       @gamma
       @mpr
       @A
       @B
	"""
	def __init__(self,):
		self.binaryName = 'DD'
		self.PB = 0.0*u.day
		self.ECC = 0.0
		self.T0 = 54000.0*u.day
		self.EDOT = 0.0/u.second
		self.A1 = 10.0*ls        # Light-Sec
        self.A1DOT = 0.0*ls/u.second        # Light-Sec / Sec
		self.A0 = 0.0
		self.B0 = 0.0
 

	def compute_eccentric_anomalies(self,t):
        self.t = t
        Eest = 2*np.pi*(self.t-self.T0)/self.PB
        E = netwon.(self.ecc_anmls,Eest, maxiter=500)

    @Cache.use_cache
    def binary_romoer_delay(self,t):
    	tt0 = self.tt0()

    	pass

	def binary_shapiro_delay(self):
		pass
	def binary_E_delay(self):
		pass 

    @Cache.use_cache
    def tt0(self,t):
    	if not hasattr(t,'unit'):
    		t = t*u.day
        return (t-self.T0).to('second')

    def ecct(self,t):
        return self.ECC + self.tt0(t)*self.EDOT

    def a1(self):
        return self.A1 + self.tt0()*self.A1DOT


