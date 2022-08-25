Working offline
===============

PINT takes advantage of the Internet to ensure that you are always working with
up-to-date clock correction and other files. Normally this is very convenient,
but sometimes one doesn't have access to the Internet. This could be because
you were on an airplane or a mountaintop or inside a secure facility and can't
wait to time your pulsar, or it could be because you are running PINT on the
nodes of a cluster that don't have connections to the general Internet. PINT
has some tools to support these situations.

Here are some general guidelines:

- You will need to ensure that you pre-download an appropriate collection of files before you disconnect from the Internet.
- You can find yourself using out-of-date clock files; PINT will by default emit warnings if this happens.
- You can find yourself using out-of-date IERS (Earth-rotation or leap-second) files. Newer versions of Astropy will warn you if this occurs, but older ones may not.

The files that PINT and Astropy go looking for on the Internet get downloaded
and stored in the "Astropy cache". This is a location, usually in your home
directory, that depends on your operating system, version of Astropy, and
possibly environment variable settings. If you need to know where this is, you
can run ``...``. In general this directory will work and be accessible when you
need it to; the Astropy documentation has some more details on this.

Pre-loading the Astropy cache should be easy: simply call
:func:`pint.utils.preload_cache` and it will download every file it thinks you
might need in the cache. You can often get away with less; if you have been
running your scripts fine, your cache probably already contains all the files
you need.

When you want to ensure that PINT and Astropy do not reach out to the Internet,
you can call :func:`pint.utils.set_no_internet`; this will set several Astropy
config options for the duration of the current Python session, and together
they will ensure that your script does not reach out to the Internet and
out-of-date files result in warnings.

If you want a longer-lasting setup to disable Internet access, you can set up
an Astropy config file. See ... for details on how to do this. Once the file
exists, you can set some options to achieve the same effect as
:func:`pint.utils.set_no_internet`. Unfortunately, which options are available
differs between Astropy 4.0, 4.3, 5.0, and 5.1; support for disconnected
operation has been gradually improving.
