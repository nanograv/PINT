#!/bin/bash
cython spice_util_py.pyx
gcc -fno-strict-aliasing -I/Users/jingluo/anaconda/include -arch x86_64 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/jingluo/anaconda/lib/python2.7/site-packages/numpy/core/include -I/Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/include /Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/lib/cspice.a -I/Users/jingluo/anaconda/include/python2.7 -c spice_util_py.c -o spice_util_py.o

gcc  -bundle -undefined dynamic_lookup -I/Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/include /Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/lib/cspice.a -g spice_util_py.o -o spice_util_py.so

