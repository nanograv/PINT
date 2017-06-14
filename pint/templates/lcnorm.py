"""
A module for handling normalization of light curves with an arbitrary
number of primitive components.

This is done by treating each primitives' normalization parameter as
the square of a cartesian variable lying within or on an
n-dimensional ball of unit radius.

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/pulsar/lcnorm.py,v 1.9 2017/04/07 19:31:37 kerrm Exp $

author: M. Kerr <matthew.kerr@gmail.com>
"""

import numpy as np
from math import sin,cos,asin,acos,pi

# can some of the code be reduced with inheritance here?
# TODO -- error propagation to norms

class NormAngles(object):
    """ Keep track of N angles (0 to pi/2) representing the coordinates
        inside a unit radius N-ball.
        
        Generally, the apportionment of the amplitudes of components is
        indicated by the position of the vector, while the overall
        normalization is given by the an additional angle, the sine of
        which provides the (squared) normalization."""

    def is_energy_dependent(self):
        return False

    def init(self):
        self.free = np.asarray([True]*self.dim,dtype=bool)
        self.errors = np.zeros(self.dim)
        self.pnames = ['Ang%d'%(i+1) for i in xrange(self.dim)]
        self.name = 'NormAngles'
        self.shortname = 'None'

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def _asarrays(self):
        for key in ['p','free','bounds','errors','slope','slope_free']:
            if hasattr(self,key):
                v = self.__dict__[key]
                if v is not None:
                    self.__dict__[key] = np.asarray(v,dtype=bool if 'free' in key else float)

    def __init__(self,norms,**kwargs):
        """ norms -- a tuple or array with the amplitudes of a set of
            components; their sum must be <= 1."""
        self.dim = len(norms)
        self.init()
        if not self._check_norms(norms):
            raise ValueError('Provided norms ... \n%s\n ... do not satisfy constraints.'%(str(norms)))
        self.p = self._get_angles(norms)
        self.__dict__.update(**kwargs)
        self._asarrays()

    def __str__(self):
        # IN PROGRESS
        norms = self()
        errs = self.get_errors(free=False,propagate=True)
        dcderiv = 2*sin(self.p[0])*cos(self.p[0])
        dcerr = self.errors[0]*abs(dcderiv)
        def norm_string(i):
            fstring = '' if self.free[i] else ' [FIXED]'
            return 'P%d : %.4f +\- %.4f%s'%(i+1,norms[i],errs[i],fstring)
        s0 = '\nMixture Amplitudes\n------------------\n'+\
             '\n'.join([norm_string(i) for i in xrange(self.dim)])+\
             '\nDC : %.4f +\- %.4f'%(1-self.get_total(),dcerr)
        return s0

    def __len__(self): return self.dim

    def _check_norms(self,norms,eps=1e-15):
        ok = True
        for n in norms:
            ok = ok and (n <= (1+eps))
        return ok and (sum(norms)<=(1+eps))

    def _get_angles(self,norms):
        """ Determine the n-sphere angles from a set of normalizations."""
        sines = sum(norms)**0.5
        if (sines > 1):
            if (abs(sines-1)<1e-12):
                sines = 1
            else:
                raise ValueError('Invalid norm specification')
        angles = [asin(sines)]
        norms = np.asarray(norms)**0.5
        for i in xrange(self.dim-1):
            t = norms[i]/sines
            if (t > 1):
                if (abs(t-1)<1e-12):
                    t = 1
                else:
                    raise ValueError('Invalid norm specification')
            phi = acos(t)
            sines *= sin(phi)            
            angles.append(phi)
        return np.asarray(angles)

    def set_parameters(self,p,free=True):
        if free:
            self.p[self.free] = p
        else:
            self.p[:] = p

    def get_parameters(self,free=True):
        if free:
            return self.p[self.free]
        return self.p

    def get_parameter_names(self,free=True):
        return [p for (p,b) in zip(self.pnames,self.free) if b]

    def set_errors(self,errs):
        """ errs an array with the 1-sigma error estimates with shape
            equal to the number of free parameters."""
        self.errors[:] = 0.
        self.errors[self.free] = errs

    def get_errors(self,free=True,propagate=True):
        """ Get errors on components.  If specified, propagate errors from
            the internal angle parameters to the external normalizations.
        """
        # TODO -- consider using finite difference instead
        if not propagate:
            return self.errors[self.free] if free else self.errors
        g = self.gradient()**2
        g *= self.errors**2
        errors = g.sum(axis=1)**0.5
        return errors[self.free] if free else errors

    def get_bounds(self):
        """ Angles are always [0,pi/2). """
        return np.asarray([ [0,pi/2] for i in xrange(self.dim) ])[self.free]

    def sanity_checks(self,eps=1e-6):
        t1 = abs(self().sum() - sin(self.p[0])**2) < eps
        return t1

    def __call__(self,log10_ens=3):
        """ Return the squared value of the Cartesian coordinates.

            E.g., for a 3-sphere, return
            z^2 = sin^2(a)*cos^2(b)
            x^2 = sin^2(a)*sin^2(b)*cos^2(c)
            y^2 = sin^2(a)*sin^2(b)*sin^2(c)

            Recall that the normalization is *also* given as an angle,
            s.t. the vector lies within the unit sphere.

            These values are guaranteed to satisfy the constraint of
            a sum <= unity and so are suitable for normalizations of
            a light curve.
        """
        p = self.p
        m = sin(p[0]) # normalization
        norms = np.empty(self.dim)
        for i in xrange(1,self.dim):
            norms[i-1] = m * cos(p[i])
            m *= sin(p[i])
        norms[self.dim-1] = m
        return norms**2

    def gradient(self,log10_ens=3,free=False):
        """ Return a matrix giving the value of the partial derivative
            of the ith normalization with respect to the jth angle, i.e.

            M_ij = dn_i/dphi_j

            Because of the way the normalizations are defined, the ith
            normalization only depends on the (i+1)th first angles, so
            the upper half of M_ij is zero (see break statement below).

            Likewise, as can either be seen by logs or trigonometry, the
            general form of the derivative is M_ij = n_i*cot(phi_j) if
            n_i depends on phi_j through sin^2, else M_ij = n_i/cot(phi_j).
            The cos^2 dependence only occurs on the diagonal terms (i==j
            below) except for the final normalization (i==dim-1) which again
            depends on phi_j through sin^2.
        """
        m = np.zeros([self.dim,self.dim],dtype=float)
        n = self()
        p = self.p
        cots = 1./np.tan(p)
        for i in xrange(self.dim):
            for j in xrange(self.dim):
                if j > (i+1): break
                if j <= i:
                    m[i,j] = n[i]*cots[j]
                else:
                    if i==(self.dim-1):
                        m[i,j] = n[i]*cots[j]
                    else:
                        m[i,j] = -n[i]/cots[j] #-cotangent
        if free:
            return (2*m)[:,self.free]
        return 2*m

    def get_total(self):
        """ Return the amplitude of all norms."""
        return sin(self.p[0])**2

    def set_total(self,val):
        """ Set overall normalization of the represented components."""
        norms = self()
        self.p = self._get_angles(norms*(val/norms.sum()))

    def set_single_norm(self,index,val):
        norms = self()
        norms[index] = val
        if not self._check_norms(norms):
            raise ValueError('Provided norms ... \n%s\n ... do not satisfy constraints.'%(str(norms)))
        self.p = self._get_angles(norms)

    def eval_string(self):
        """ Return a string that can be evaled to instantiate a nearly-
            identical object."""
        t = self()
        if len(t.shape)>1:
            t = t[:,0] # handle e-dep
        return '%s(%s,free=%s,slope=%s,slope_free=%s)'%(
        self.__class__.__name__,str(list(t)),str(list(self.free)),
        str(list(self.slope)) if hasattr(self,'slope') else None,
        str(list(self.slope_free)) if hasattr(self,'slope_free') else None)

    def dict_string(self):
        """ Round down to avoid input errors w/ normalization."""
        t = self()
        if len(t.shape)>1:
            t = t[:,0] # handle e-dep
        def pretty_list(l,places=6,round_down=True):
            if round_down:
                r = np.round(l,decimals=places)
                r[r>np.asarray(l)] -= 10**-places
            else:
                r = l
            fmt = '%.'+'%d'%places+'f'
            s = ', '.join([fmt%x for x in r])
            return '['+s+']'
        return [
            'name = %s'%self.__class__.__name__,
            'norms = %s'%(pretty_list(t)),
            'free = %s'%(str(list(self.free))),
            'slope = %s'%(pretty_list(self.slope,round_down=False) if hasattr(self,'slope') else None),
            'slope_free = %s'%(str(list(self.slope_free)) if hasattr(self,'slope_free') else None)
        ]

