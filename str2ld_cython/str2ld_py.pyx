import numpy as np
cimport numpy as np
cimport libc.stdlib
from libc.stdlib cimport free, malloc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "stdlib.h":
    long double strtold(const char *nptr, char **endptr)
    void *memcpy(void *dst, void *src, long n)

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

def str2ldarr1(char *number):
    # This returns a length-1 long-double array
    cdef np.ndarray[np.longdouble_t, ndim=1] output = np.empty(shape=1, dtype=np.longfloat)
    output[0] = strtold(number, NULL)
    return output
    
