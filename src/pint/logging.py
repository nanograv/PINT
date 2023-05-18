"""Custom logging filter for PINT using ``loguru``.

To use this do::

    import pint.logging
    pint.logging.setup()

If you want to emit messages later in your code, set things up with::

    from loguru import logger as log
    log.info("INFO test")

This can happen before or after the :func:`~pint.logging.setup`.

You can optionally pass the desired logging level to the :func:`~pint.logging.setup` function, formats, custom filters, colors, etc.  
See documentation for :func:`pint.logging.setup`.

`level` can be any of the existing ``loguru`` levels: ``TRACE``, ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or you can define new ones.

The format can be something new, or you can use :py:data:`pint.logging.format`.  A full format that might be useful as a reference is::
    
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

while the default for this module is::

    format = "<level>{level: <8}</level> ({name: <30}): <level>{message}</level>"

If you want to use command-line arguments in a script to set the level you can do that like::

    parser.add_argument("--log-level",type=str,choices=pint.logging.levels,default=pint.logging.script_level,help="Logging level",dest="loglevel")
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", default=0, action="count", help="Decrease output verbosity"
    )
    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )

Note that ``loguru`` does not allow you to change the properties of an existing logger.
Instead it's better to remove it and make another (e.g., if you want to change the level).

Defaults can be changed with environment variables like:
``$LOGURU_LEVEL``, ``$LOGURU_FORMAT``, ``$LOGURU_DEBUG_COLOR``.

See `loguru documentation <https://loguru.readthedocs.io/en/stable/>`_ for full set of options.

"""

import os
import re
import sys
import warnings
from loguru import logger as log

from erfa import ErfaWarning

__all__ = ["LogFilter", "setup", "format", "levels", "get_level"]

# defaults can be overridden using $LOGURU_LEVEL and $LOGURU_FORMAT
# default for an individual level can be overridden by $LOGURU_DEBUG_COLOR etc
# or just make a new logger
# a full format that might be useful as a reference:
# format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
format = "<level>{level: <8}</level> ({name: <30}): <level>{message}</level>"
debug_color = "<fg #b790d4><bold>"
# default level to be used in scripts
script_level = "WARNING"
# Other formatting:
# https://loguru.readthedocs.io/en/stable/api/logger.html#color

showwarning_ = warnings.showwarning
warning_onceregistry = {}

# basic loguru level definitions from:
# https://loguru.readthedocs.io/en/stable/api/logger.html
# this is so we can map from increase/decrease of verbosity to level
levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Function to allow ``loguru`` to capture warnings emitted by :func:`warnings.warn`.

    Also look at the existing :data:`warnings.filters` to see if warnings should be ignored or only seen once (most messages are already filtered, but some aren't, especially in older python)

    See https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
    """
    if isinstance(message, Warning):
        message_text = str(message)
        category = message.__class__
    else:
        message_text = message
    # take action according to standard warning filters
    # this isn't the exact same implementation as the standard filter because we don't get all of the relevant pieces
    # but it works for ignoring
    for filter in warnings.filters:
        action, msg, cat, mod, ln = filter
        if (msg is not None) and (msg.match(message_text)) and action == "ignore":
            # if it matches an "ignore" message
            return
        # matching based on category
        if (
            (cat is not None)
            and ((msg is None or msg.match(message_text)) and issubclass(category, cat))
            and action == "ignore"
        ):
            return
        # only issue once, just based on text matching.  construct a key for future matching
        # mostly follows python warnings module
        if action == "once":
            oncekey = (message_text, category)
            if warning_onceregistry.get(oncekey):
                return
            warning_onceregistry[oncekey] = 1
    # turn the warning into text.
    arg_string = f"{filename}:{lineno} {category.__name__}"
    log.warning(f"{arg_string}: {message_text}")
    # I don't think we need to re-emit the message
    # showwarning_(message, category, filename, lineno, file=file, line=line)


class LogFilter:
    """Custom logging filter for ``loguru``.
    Define some messages that are never seen (e.g., Deprecation Warnings).
    Others that will only be seen once.  Filtering of those is done on the basis of regular expressions.
    """

    def __init__(self, onlyonce=None, never=None, onlyonce_level="INFO"):
        """
        Define regexs for messages that will only be seen once.  Use ``\S+`` for a variable that might change.
        If a message comes through with a new value for that variable, it will be seen.

        Make sure to escape other regex commands like ``()``.

        Each message starts with ``state = False``.
        Once it has been emitted, that changes to a list of the messages so that it can keep track.
        These are only suppressed when issued at level `onlyonce_level` or lower (e.g., if `onlyonce_level` is ``INFO``, then ``WARNING`` will always come through)

        They should be defined as:

            r"Error message": False

        where the ``False`` tracks whether or not the message has been issued at all.

        Parameters
        ----------
        onlyonce : list, optional
            list of messages that should only be issued once if at ``INFO`` or below.  Checked using ``re.match``, so must match from beginning of message.
        never : list, optional
            list of messages that should never be seen.  Checked using ``re.search``, so can match anywhere in message.
        onlyonce_level : str, optional
            level below which messages will only be shown once
        """
        self.onlyonce = {
            r"Using EPHEM = \S+ for \S+ calculation": False,
            r"Using CLOCK = \S+ from the given model": False,
            r"Using PLANET_SHAPIRO = \S+ from the given model": False,
            r"Applying clock corrections \(include_gps = \S+, include_bipm = \S+\)": False,
            r"Applying observatory clock corrections.": False,
            r"Applying GPS to UTC clock correction \(\~few nanoseconds\)": False,
            r"Computing \S+ columns.": False,
            r"Using EPHEM = \S+ for \S+ calculation.": False,
            r"Planet PosVels will be calculated.": False,
            r"Computing PosVels of observatories, Earth and planets, using \S+": False,
            r"Computing PosVels of observatories and Earth, using \S+": False,
            r"Set solar system ephemeris to \S+": False,
            r"Adding column \S+": False,
            r"Adding columns .*": False,
            r"Applying TT\(\S+\) to TT\(\S+\) clock correction \(\~27 us\)": False,
            r"No pulse number flags found in the TOAs": False,
            r"SSB obs pos \[\S+\s+\S+\s+\S+\] m": False,
            r"Column \S+ already exists. Removing...": False,
            r"Skipping Shapiro delay for Barycentric TOAs": False,
            r"Special observatory location. No clock corrections applied.": False,
            r"DDK model uses KIN as inclination angle. SINI will not be used. This happens every time a DDK model is constructed.": False,
            r"Glitch phase for glitch (\S+):": False,
        }
        # add in any more defined on init
        if onlyonce is not None:
            for m in onlyonce:
                self.onlyonce[m] = False
        # List of partial matching strings for messages never to be displayed
        self.never = [
            "MatplotlibDeprecationWarning",
            "DeprecationWarning",
            "ProvisionalCompleterWarning",
            "deprecated in Matplotlib",
        ]
        # add in any more defined on init
        if never is not None:
            self.never += never

        self.onlyonce_level = onlyonce_level

    def filter(self, record):
        """Filter the record based on ``record["message"]`` and ``record["level"]``
        If this returns s,``False``, the message is not seen

        Parameters
        ----------
        record : dict
            should contain ``record["message"]`` and ``record["level"]``

        Returns
        -------
        bool
            If ``True``, message is seen.  If ``False``, message is not seen
        """
        for m in self.never:
            if re.search(m, record["message"]):
                return False
        # display all warnings and above
        if record["level"].no > log.level(self.onlyonce_level).no:
            return True
        for m in self.onlyonce:
            if re.match(m, record["message"]):
                # save not the whole message, but only the portion that matches the regex
                # this allows filtering on things like glitch number
                match = re.match(m, record["message"])
                message_to_save = record["message"][slice(*match.span())]
                if not self.onlyonce[m]:
                    self.onlyonce[m] = [message_to_save]
                    return True
                elif message_to_save not in self.onlyonce[m]:
                    self.onlyonce[m].append(message_to_save)
                    return True
                return False
        return True

    def __call__(self, record):
        return self.filter(record)


def setup(
    level="INFO",
    sink=sys.stderr,
    format=format,
    filter=LogFilter(),
    usecolors=True,
    colors={"DEBUG": debug_color},
    capturewarnings=True,
    removeprior=True,
):
    """
    Setup the PINT logging using ``loguru``

    This involves removing previous loggers and adding a new one at the requested level

    Parameters
    ----------
    level : str, optional
        Logging level, unless overridden by ``$LOGURU_LEVEL``
    sink : file-like object, str, or other object accepted by :py:meth:`loguru.Logger.add`, optional
        Destination for the logging messages
    format : str, optional
        Format string for the logging messages, unless overridden by ``$LOGURU_FORMAT``.  See `loguru documentation <https://loguru.readthedocs.io/en/stable/>`_ for full set of options
    filter : callable, optional
        Should be a ``LogFilter`` or similar which returns ``True`` if a message will be seen and ``False`` otherwise.
        The default instance can be modified to change the messages that are never seen/only seen once
    usecolors : bool, optional
        Should it use colors at all
    colors : dict, optional
        Dict of ``level``, ``format`` pairs to override the color/format settings for any level
    capturewarnings : bool, optional
        Whether or not messages emitted by :func:`warnings.warn` should be included in the logging output
    removeprior : bool, optional
        Whether or not to remove prior loggers

    Returns
    -------
    int
        An identifier associated with the added sink and which should be used to
        remove it.

    Example
    -------

        >>> import pint.logging
        >>> import sys
        >>> format = "<level>{level: <8}</level> ({name: <30}): <level>{message}</level>"
        # turn off colors if your terminal does not play well with them
        >>> pint.logging.setup(level="DEBUG", sink=sys.stderr, format=format, filter=pint.logging.LogFilter(), usecolors=False)

    """

    # if this is not used, then the default warning mechanism is not overridden. There may be times when that is desired
    if capturewarnings:
        warnings.showwarning = showwarning

    # remove the default logger so we can put in one with a custom filter
    # this can be done elsewhere if more/different customization is needed
    if removeprior:
        log.remove()
    # Keep these here to see what is set at the enrivonment level
    # again, this isn't needed by default but if you are setting these explicitly
    # then it can be good to check
    if "LOGURU_LEVEL" in os.environ:
        level = os.environ["LOGURU_LEVEL"]
    if "LOGURU_FORMAT" in os.environ:
        format = os.environ["LOGURU_FORMAT"]

    # use colorize=True to force colors
    # otherwise the default selection turns them off e.g., for a Jupyter notebook
    # since it isn't a tty
    loghandler = log.add(
        sink,
        level=level,
        filter=filter,
        format=format,
        colorize=usecolors,
        enqueue=True,
    )
    # change default colors
    for level in colors:
        log.level(level, color=debug_color)

    return loghandler


def get_level(starting_level_name, verbosity, quietness):
    """Get appropriate logging level given command-line input

    Parameters
    ----------
    starting_level_name : str
        Name of level to start with (e.g., "WARNING")
    verbosity : int
        Number of verbose levels requested
    quietness : int
        Number of quiet levels requested

    Returns
    -------
    str
        Name of level

    Examples
    --------

        >>> parser.add_argument("--log-level",type=str, choices=pint.logging.levels,
        default=pint.logging.script_level, help="Logging level",dest="loglevel")
        >>> parser.add_argument("-v", "--verbosity", default=0, action="count", help="Increase output verbosity")
        >>> parser.add_argument("-q", "--quiet", default=0, action="count", help="Decrease output verbosity")
        >>> args = parser.parse_args(argv)
        >>> pint.logging.setup(level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet))

    """
    starting_level = [i for i in range(len(levels)) if levels[i] == starting_level_name]
    level = min(
        max(starting_level[0] - verbosity + quietness, 0),
        len(levels) - 1,
    )
    return levels[level]
