How to Control PINT Logging Output
==================================

If you have run PINT, you have probably noticed that PINT can emit a generous
amount of information in the form of log messages. These come in two forms:
warnings emitted through the python :mod:`warnings` module, and messages
sent through the python :mod:`logging` mechanism (some of which are also
warnings). The amount of information emitted can result in the messages of
interest being lost among routine messages, or one could wish for further
detail (for example for debugging PINT code). There are tools for managing this
information flow; although PINT is a library and thus simply emits messages, a
user with a notebook, script, or GUI application can use these tools to
manage this output.

Controlling log messages
------------------------

Python's :mod:`logging` is somewhat complicated and confusing. In PINT's case we use the 
:mod:`loguru` to reconfigure it and make it easier to use, with some additional code in 
:mod:`pint.logging` to adapt it to our purposes (things like changing the format, adding colors, 
capturing warnings, and preventing duplicate messages from overwhelming users).  It is worth explaining a design
principle: libraries simply emit messages, while applications, notebooks, and
scripts configure what to do with those messages.  You can (re)configure the logging output::

    import pint.logging
    pint.logging.setup(level="DEBUG")

You can optionally pass other options to  the :func:`~pint.logging.setup` function, such as 
a destination, level, formats, custom filters, colors, etc.  See documentation for :func:`pint.logging.setup`.

``level`` can be any of the existing ``loguru`` levels: ``TRACE``, ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or you can define new ones.

The format can be something new, or you can use :py:data:`pint.logging.format`.  A full format that might be useful as a reference is::
    
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

while the default for :mod:`pint.logging` is::

    format = "<level>{level: <8}</level> ({name: <30}): <level>{message}</level>"

If you want to use command-line arguments in a script to set the level you can do that like::

    parser.add_argument("--log-level",type=str,choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),default=pint.logging.script_level,help="Logging level",dest="loglevel")
    args = parser.parse_args(argv)
    pint.logging.setup(level=args.loglevel)

Note that ``loguru`` does not allow you to change the properties of an existing logger.
Instead it's better to remove it and make another (e.g., if you want to change the level).  This is done by default, but 
if instead you want to add another logger (say to a file) you can run :func:`~pint.logging.setup` with ``removeprior=False``.

Defaults can be changed with environment variables like:
``$LOGURU_LEVEL``, ``$LOGURU_FORMAT``, ``$LOGURU_DEBUG_COLOR``.

See `loguru documentation <https://loguru.readthedocs.io/en/stable/>`_ for full set of options.


Warnings versus logging
-----------------------

The logging HOWTO_ describes the difference between ``warnings.warn`` and ``logging.warning`` thus:

    - ``warnings.warn()`` in library code if the issue is avoidable and the client application should be modified to eliminate the warning
    - ``logging.warning()`` if there is nothing the client application can do about the situation, but the event should still be noted

Although PINT does not follow these rules perfectly, it does emit both kinds of
warning, and users may quite reasonably want to handle them in various ways.  By default :func:`~pint.logging.setup`
will capture warnings and emit them through the logging module, but this can be turned off by setting ``capturewarnings=False``.

Users can control the handling of warnings with "warning filters"; in the simplest arrangements, users can just use ``warnings.simplefilter("ignore")`` or similar to arrange for all warnings to be ignored or treated as exceptions; users can use the more sophisticated :func:`warnings.filterwarnings` to control warnings based on their module of origin and/or the class supplied to the ``warnings.warn()`` call. Of particular note is the confusingly named :func:`warnings.catch_warnings`, which is a context manager that supports temporary changes in how warnings are handled::

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter.fit_toas()

For further details on the management of warnings, see the documentation of the module :mod:`warnings`.

.. _HOWTO: https://docs.python.org/3/howto/logging.html

