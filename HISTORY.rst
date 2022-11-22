History
=======

TEMPO was originally written in the 1980s, aiming for microsecond-level accuracy. 
TEMPO2 was written more recently, with an attention to nanosecond-level effects. 
Both are still in current use. But TEMPO is in FORTRAN, TEMPO2 is in C++, and neither 
is easy to extend for use in tasks different from plain pulsar timing. Most of TEMPO2 
is also a direct conversion of TEMPO, so many bugs from TEMPO were carried over to 
TEMPO2. PINT was created to be, as far as possible, an independent re-implementation 
based on existing libraries - notably astropy - and to be a flexible toolkit for 
working with pulsar timing models and data.
