import numpy as np
cimport numpy as np
cimport libc.stdlib
from libc.stdlib cimport free, malloc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, long n)
cdef extern from "SpiceUsr.h":
    void spkezr_c ( char *targ,
                    double et,
                    char *ref,
                    char *abcorr,
                    char *obs,
                    double starg[6],
                    double *lt)
    void furnsh_c ( char  * file )
    void unload_c ( char  * file )
cdef inline np.ndarray c2npy_double(double **a, int m, int n):
    cdef np.ndarray[DTYPE_t,ndim=2] result = np.zeros((m,n),dtype=DTYPE)
    cdef double *dest
    cdef int i
    dest = <double *> PyMem_Malloc(m*n*sizeof(double*))
    for i in range(m):
        memcpy(dest + i*n,a[i],m*sizeof(double*))
        PyMem_Free(a[i])
    memcpy(result.data,dest,m*n*sizeof(double*))
   
    PyMem_Free(dest)
    PyMem_Free(a)
    return result
    

def furnsh_py(filename):
    furnsh_c(filename)

def unload_py(filename):
    unload_c(filename)     

def spkezr_array(target,observer,np.ndarray[DTYPE_t,ndim=1] et_array,et_length):
    cdef double state[6]
    cdef double **posvel
    cdef double lt
    cdef int i,j 
   
    #posvel = <double **> malloc(et_length*sizeof(double*)) 
    posvel = <double **>PyMem_Malloc(et_length*sizeof(double*))
    if not posvel:
        raise MemoryError()
        
    for i in range(et_length):
        posvel[i] = <double *> PyMem_Malloc(6*sizeof(double))
        if not posvel[i]:
            print "In posvel second demision"
            raise MemoryError()
    
           
    for i in range(et_length):
        spkezr_c(target, et_array[i], "J2000","None",observer,state,&lt)
        for j in range(6):
            posvel[i][j] = state[j]
    #posvel_np = c2npy_double(posvel,et_length,6)
     
    #for i in range(et_length):
    #    PyMem_Del(posvel[i]) 
    #PyMem_Free(posvel)
    
    posvel_np = c2npy_double(posvel,et_length,6) 
     
        
    #return    
    return posvel_np 

def spkezr_array_np(target,observer,np.ndarray[DTYPE_t,ndim=1] et_array,et_length):
    cdef np.ndarray[DTYPE_t,ndim=2] posvel = np.zeros((et_length,6),dtype=DTYPE)
    cdef double state[6]
    cdef double lt
    cdef int i,j
    
    for i in range(et_length):
        spkezr_c(target, et_array[i], "J2000","None",observer,state,&lt)
        for j in range(6):
            posvel[i,j] = state[j]


    return posvel




    """
    
    for i in range(et_length):
        free(posvel[i])
            
    free(posvel)
    free(state)
    print "I am here in the end"
    return 
    """

"""
def spkezr_np_array(np.ndarray[double, ndim=1, mode="c"] et not None,
                    target,observer,
                    np.ndarray[double, ndim=2, mode="c"] posvel not None,
                    etLength):
    cdef bytes target_py
    cdef bytes observer_py
    cdef char* target_c = target_py
    cdef char* observer_c = observer_py 

"""    
