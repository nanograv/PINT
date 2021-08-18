How to Control PINT Logging Output
==================================

If you have run PINT, you have probably noticed that PINT emits a generous
amount of information in the form of log messages. These come in two forms:
warnings emitted through the python :module:`warnings` module, and messages
sent through the python :module:`logging` mechanism (some of which are also
warnings). The amount of information emitted can result in the messages of
interest being lost among routine messages, or one could wish for further
detail (for example for debugging PINT code). There are tools for managing this
information flow; although PINT is a library and thus simply emits messages, a
user with a notebook, script, or GUI application can use these tools to
manage this output.

Warnings versus logging
-----------------------

The logging HOWTO_ describes the difference between ``warnings.warn`` and ``logging.warning`` thus:

    - ``warnings.warn()`` in library code if the issue is avoidable and the client application should be modified to eliminate the warning
    - ``logging.warning()`` if there is nothing the client application can do about the situation, but the event should still be noted

Although PINT does not follow these rules perfectly, it does emit both kinds of
warning, and users may quite reasonably want to handle them in various ways.
Notably, users can call :func:`logging.captureWarnings` to arrange for
``warnings.warn()`` to be logged as ``logging.warning()``, although they are not tagged with
their module of origin.

Users can control the handling of warnings with "warning filters"; in the simplest arrangements, users can just use ``warnings.simplefilter("ignore")`` or similar to arrange for all warnings to be ignored or treated as exceptions; users can use the more sophisticated :func:`warnings.filterwarnings` to control warnings based on their module of origin and/or the class supplied to the ``warnings.warn()`` call. Of particular note is the confusingly named :func:`warnings.catch_warnings`, which is a context manager that supports temporary changes in how warnings are handled::

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter.fit_toas()

For further details on the management of warnings, see the documentation of the module :module:`warnings`.

.. _HOWTO: https://docs.python.org/3/howto/logging.html

Controlling log messages
------------------------

Python's :module:`logging` is somewhat complicated and confusing but it does
support some important features. In PINT's case it is worth explaining a design
principle: libraries simply emit messages, while applications, notebooks, and
scripts configure what to do with those messages. If PINT tried to decide what
to do with the messages, it would inevitably conflict with the needs of one of
those users (we have seen this in NANOGrav notebooks, where conflicting
configurations resulted in every message appearing twice). So if you are using
PINT and the log messages don't look very nice, that's python's default log
reporting. If you want to. most simply, set the log level that is reported
globally, you can::

    import logging
    logging.getLogger().setLevel("DEBUG")

What PINT does do is ensure that every log message is handled based on the
module that originated it. This uses the fact that python's loggers are
hierarchical, passing messages up towards a global "root" logger. So to obtain
debugging information from only :module:`pint.fitter`, you can::

    import logging
    logging.getLogger("pint.fitter").setLevel("DEBUG")


