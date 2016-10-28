from .observatory import Observatory

# Include any files that define observatories here.  This will start
# with the standard distribution files, then will read any system- or
# user-defined files.  These can override the default settings by
# redefining an Observatory with the same name.
# TODO read the files from the other locations, if they exist
from . import special_locations
from . import observatories
