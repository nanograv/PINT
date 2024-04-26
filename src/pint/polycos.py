"""Polynomial coefficients for phase prediction

Polycos designed to predict the pulsar's phase and pulse-period over a
given interval using polynomial expansions.   

The pulse phase and frequency at time T are then calculated as:

.. math::

    \\Delta T = 1440(T-T_{\\rm mid})

    \\phi = \\phi_0 + 60 \Delta T f_0 + COEFF[1] + COEFF[2] \\Delta T + COEFF[3] \\Delta T^2 + \\ldots

    f({\\rm Hz}) = f_0 + \\frac{1}{60}\\left( COEFF[2] + 2 COEFF[3] \Delta T + 3 COEFF[4] \Delta T^2  + \\ldots \\right)

Examples
--------
Read in polycos, predict phases:

    >>> from pint.polycos import Polycos
    >>> p = Polycos.read(filename)
    >>> p.eval_abs_phase(mjds)

Or, to generate polycos from a timing model:

    >>> from pint.models import get_model
    >>> from pint.polycos import Polycos
    >>> model = get_model(filename)
    >>> p = Polycos.generate_polycos(model, 50000, 50001, "AO", 144, 12, 1400)
  
References
----------
http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
"""

import astropy.table as table
import astropy.units as u
import numpy as np
from astropy.io import registry
from astropy.time import Time
from collections import OrderedDict

from loguru import logger as log

try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError) as e:

    def tqdm(*args, **kwargs):
        return args[0] if args else kwargs.get("iterable", None)


import pint.toa as toa
from pint.phase import Phase
from pint.pulsar_mjd import data2longdouble
from pint.utils import open_or_use

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


__all__ = [
    "PolycoEntry",
    "tempo_polyco_table_reader",
    "tempo_polyco_table_writer",
    "Polycos",
]

MIN_PER_DAY = (1 * u.day).to_value(u.min)


class PolycoEntry:
    """One Polyco entry.

    Referenced from polyco.py authored by
        - Paul S. Ray <paul.s.ray3.civ@us.navy.mil>
        - Matthew Kerr <matthew.kerr@gmail.com>

    Parameters
    ---------
    tmid : float
        Middle point of the time span in mjd
    mjdspan : int
        Time span in minutes
    rphase : float
        Reference phase
    f0 : float
        Reference spin frequency
    ncoeff : int
        Number of coefficients
    coeff : numpy.ndarray
        Polynomial coefficents
    """

    def __init__(self, tmid, mjdspan, rph_int, rph_frac, f0, ncoeff, coeffs):
        self.tmid = data2longdouble(tmid) * u.day
        self.mjdspan = data2longdouble(mjdspan / MIN_PER_DAY) * u.day
        self.tstart = self.tmid - (self.mjdspan / 2)
        self.tstop = self.tmid + (self.mjdspan / 2)
        self.f0 = data2longdouble(f0)
        self.ncoeff = ncoeff
        self.rphase = Phase(rph_int, rph_frac)
        self.coeffs = data2longdouble(coeffs)

    def __str__(self):
        return (
            "Middle Point mjd : "
            + repr(self.tmid)
            + "\n"
            + "Time Span in mjd : "
            + repr(self.mjdspan)
            + "\n"
            + "Time Start in mjd : "
            + repr(self.tstart)
            + "\n"
            + "Time Stop in mjd : "
            + repr(self.tstop)
            + "\n"
            + "Reference Phase : "
            + repr(self.rphase)
            + "\n"
            + "Reference Freq in Hz : "
            + repr(self.f0)
            + "\n"
            + "Number of Coefficients : "
            + repr(self.ncoeff)
            + "\n"
            + "Coefficients : "
            + repr(self.coeffs)
        )

    def evalabsphase(self, t):
        """Return the phase at time t, computed with this polyco entry

        Parameters
        ----------
        t : float or numpy.ndarray
            Input times

        Returns
        -------
        pint.phase.Phase
        """
        dt = (data2longdouble(t) - self.tmid.value) * MIN_PER_DAY
        # Compute polynomial by factoring out the dt's
        phase = Phase(
            self.coeffs[self.ncoeff - 1]
        )  # Compute phase using two long double
        for i in range(self.ncoeff - 2, -1, -1):
            pI = Phase(dt * phase.int)
            pF = Phase(dt * phase.frac)
            c = Phase(self.coeffs[i])
            phase = pI + pF + c

        # Add DC term
        phase += self.rphase + Phase(dt * 60.0 * self.f0)
        return phase

    def evalphase(self, t):
        """Return the phase at time t, computed with this polyco entry

        Parameters
        ----------
        t : float or numpy.ndarray
            Input times

        Returns
        -------
        pint.phase.Phase
        """
        return self.evalabsphase(t).frac

    def evalfreq(self, t):
        """Return the freq at time t, computed with this polyco entry

        Parameters
        ----------
        t : float or numpy.ndarray
            Input times

        Returns
        -------
        numpy.ndarray
            Frequency (in Hz)
        """
        dt = (data2longdouble(t) - self.tmid.value) * MIN_PER_DAY
        s = data2longdouble(0.0)
        for i in range(1, self.ncoeff):
            s += data2longdouble(i) * self.coeffs[i] * dt ** (i - 1)
        return self.f0 + s / 60.0

    def evalfreqderiv(self, t):
        """Return the frequency derivative at time t.

        Parameters
        ----------
        t : float or numpy.ndarray
            Input times

        Returns
        -------
        numpy.ndarray
            Frequency derivative (in Hz/s)
        """
        dt = (data2longdouble(t) - self.tmid.value) * MIN_PER_DAY
        s = data2longdouble(0.0)
        for i in range(2, self.ncoeff):
            # Change to long double
            s += (
                data2longdouble(i)
                * data2longdouble(i - 1)
                * self.coeffs[i]
                * dt ** (i - 2)
            )
        return s / (60.0 * 60.0)


# Read polycos file data to table
def tempo_polyco_table_reader(filename):
    """Read tempo style polyco file to an astropy table.

    Tempo style: The polynomial ephemerides are written to file 'polyco.dat'.
    Entries are listed sequentially within the file.  The file format is::

        ====  =======   ============================================
        Line  Columns     Item
        ====  =======   ============================================
         1       1-10   Pulsar Name
                11-19   Date (dd-mmm-yy)
                20-31   UTC (hhmmss.ss)
                32-51   TMID (MJD)
                52-72   DM
                74-79   Doppler shift due to earth motion (10^-4)
                80-86   Log_10 of fit rms residual in periods
         2       1-20   Reference Phase (RPHASE)
                21-38   Reference rotation frequency (F0)
                39-43   Observatory number
                44-49   Data span (minutes)
                50-54   Number of coefficients
                55-75   Observing frequency (MHz)
                76-80   Binary phase
         3*      1-25   Coefficient 1 (COEFF(1))
                26-50   Coefficient 2 (COEFF(2))
                51-75   Coefficient 3 (COEFF(3))
        ====  =======   ============================================
        * Subsequent lines have three coefficients each, up to NCOEFF

    One polyco file could include more then one entry.

    The pulse phase and frequency at time T are then calculated as:

    .. math::

        \\Delta T = 1440(T-T_{\\rm mid})

        \\phi = \\phi_0 + 60 \Delta T f_0 + COEFF[1] + COEFF[2] \\Delta T + COEFF[3] \\Delta T^2 + \\ldots

        f({\\rm Hz}) = f_0 + \\frac{1}{60}\\left( COEFF[2] + 2 COEFF[3] \Delta T + 3 COEFF[4] \Delta T^2  + \\ldots \\right)

    Parameters
    ----------
    filename : str
        Name of the input poloco file.

    Returns
    -------
    astropy.table.Table

    References
    ----------
    http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
    """
    entries = []

    with open_or_use(filename, "r") as f:
        line = f.readline()

        while line != "":
            # First line
            fields = line.split()
            psrname = fields[0]
            date = fields[1]
            utc = float(fields[2])
            tmid = np.longdouble(fields[3])
            dm = float(fields[4])
            doppler = float(fields[5])
            logrms = float(fields[6])

            # Second line
            fields = f.readline().split()
            refPhaseInt, refPhaseFrac = fields[0].split(".")
            refPhaseInt = np.longdouble(refPhaseInt)
            refPhaseFrac = np.longdouble(f".{refPhaseFrac}")
            if refPhaseInt < 0:
                refPhaseFrac = -refPhaseFrac

            refF0 = np.longdouble(fields[1])
            obs = fields[2]
            mjdspan = int(fields[3])
            nCoeff = int(fields[4])
            obsfreq = float(fields[5])

            try:
                binary_phase = float(fields[6])
                f_orbit = float(fields[7])
                is_binary = True
            except IndexError:
                is_binary = False

            # Read coefficients
            coeffs = []
            for _ in range(-(nCoeff // -3)):
                line = f.readline()
                coeffs.extend(data2longdouble(c) for c in line.split())
            coeffs = np.array(coeffs)

            entry = PolycoEntry(
                tmid, mjdspan, refPhaseInt, refPhaseFrac, refF0, nCoeff, coeffs
            )

            entry_dict = OrderedDict()
            entry_dict["psr"] = psrname
            entry_dict["date"] = date
            entry_dict["utc"] = utc
            entry_dict["tmid"] = tmid
            entry_dict["dm"] = dm
            entry_dict["doppler"] = doppler
            entry_dict["logrms"] = logrms
            entry_dict["mjd_span"] = mjdspan
            entry_dict["t_start"] = entry.tstart
            entry_dict["t_stop"] = entry.tstop
            entry_dict["obs"] = obs
            entry_dict["obsfreq"] = obsfreq

            if is_binary:
                entry_dict["binary_phase"] = binary_phase
                entry_dict["f_orbit"] = f_orbit

            entry_dict["entry"] = entry
            entries.append(entry_dict)

            line = f.readline()

    return table.Table(entries, meta={"name": "Polyco Data Table"})


def tempo_polyco_table_writer(polycoTable, filename="polyco.dat"):
    """Write tempo style polyco file from an astropy table.

    Tempo style polyco file:
    The polynomial ephemerides are written to file 'polyco.dat'.  Entries
    are listed sequentially within the file.  The file format is::

        Line  Columns     Item
        ----  -------   -----------------------------------
         1       1-10   Pulsar Name
                11-19   Date (dd-mmm-yy)
                20-31   UTC (hhmmss.ss)
                32-51   TMID (MJD)
                52-72   DM
                74-79   Doppler shift due to earth motion (10^-4)
                80-86   Log_10 of fit rms residual in periods
         2       1-20   Reference Phase (RPHASE)
                21-38   Reference rotation frequency (F0)
                39-43   Observatory number
                44-49   Data span (minutes)
                50-54   Number of coefficients
                55-75   Observing frequency (MHz)
                76-80   Binary phase
         3*      1-25   Coefficient 1 (COEFF(1))
                26-50   Coefficient 2 (COEFF(2))
                51-75   Coefficient 3 (COEFF(3))
        * Subsequent lines have three coefficients each, up to NCOEFF

    One polyco file could include more then one entry.

    The pulse phase and frequency at time T are then calculated as:

    .. math::

        \\Delta T = 1440(T-T_{\\rm mid})

        \\phi = \\phi_0 + 60 \Delta T f_0 + COEFF[1] + COEFF[2] \\Delta T + COEFF[3] \\Delta T^2 + \\ldots

        f({\\rm Hz}) = f_0 + \\frac{1}{60}\\left( COEFF[2] + 2 COEFF[3] \Delta T + 3 COEFF[4] \Delta T^2  + \\ldots \\right)

    Parameters
    ----------
    polycoTable: astropy.table.Table
        Polycos style table
    filename : str or ~pathlib.Path or file-like
        Destination for the output polyco file. Default is 'polyco.dat'.

    References
    ----------
    http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
    """
    if len(polycoTable) == 0:
        err = "Empty polyco table! Please make sure polycoTable has data."
        raise ValueError(err)

    with open_or_use(filename, "w") as f:
        for table_entry in polycoTable:
            psr_name = table_entry["psr"]
            if psr_name[0] == "J":
                psr_name = psr_name[1:]

            spec1 = "{:10.10s} {:>9.9s}{:11.2f}{:20.11f}{:21.6f} {:6.3f}{:7.3f}\n"
            line1 = spec1.format(
                psr_name,
                table_entry["date"],
                table_entry["utc"],
                table_entry["tmid"],
                table_entry["dm"],
                table_entry["doppler"],
                table_entry["logrms"],
            )

            entry = table_entry["entry"]
            ph = entry.rphase
            ph_frac = ph.frac[0].value + (ph.frac[0] < 0)
            excess = ph_frac - np.round(ph_frac, 6)

            rphstr_frac = "{:.6f}".format(np.round(ph_frac, 6))[1:]
            rphstr_int = "{:13d}".format(int(ph.int[0]) - (ph.frac[0] < 0))

            try:
                bin_phase = "{:7.4f}{:9.4f}".format(
                    table_entry["binary_phase"], table_entry["f_orbit"]
                )
            except KeyError:
                bin_phase = ""

            spec2 = "{:20s} {:17.12f}{:>5s}{:5d}{:5d}{:10.3f}{:16s}\n"
            line2 = spec2.format(
                rphstr_int + rphstr_frac,
                entry.f0,
                table_entry["obs"],
                table_entry["mjd_span"],
                entry.ncoeff,
                table_entry["obsfreq"],
                bin_phase,
            )

            coeffs = entry.coeffs
            coeffs[0] += excess

            coeff_block = ""
            for i, coeff in enumerate(coeffs):
                coeff_block += "{:25.17e}".format(coeff)
                if (i + 1) % 3 == 0:
                    coeff_block += "\n"

            f.write(line1 + line2 + coeff_block)


# default formats
# need to define them here rather than at the top so that the reader/writer functions are defined
_polycoFormats = [
    {
        "format": "tempo",
        "read_method": tempo_polyco_table_reader,
        "write_method": tempo_polyco_table_writer,
    }
]


class Polycos:
    """A class for polycos model.

    Polyco is a fast phase calculator. It fits a set of data using polynomials and can predict phases at points in between.
    """

    # loaded formats
    polycoFormats = []

    @classmethod
    def _register(cls, formatlist=_polycoFormats):
        """
        Register the table built-in reading and writing format

        Formats get added to the ``astropy.io`` registry

        Parameters
        ----------
        formatlist : list, optional
            Each entry should be a ``dict`` with entries ``format``, ``read_method``, ``write_method``

        """
        for fmt in formatlist:
            if fmt["read_method"] is not None and fmt["write_method"] is None:
                cls.add_polyco_file_format(fmt["format"], "r", fmt["read_method"], None)
            elif fmt["read_method"] is None and fmt["write_method"] is not None:
                cls.add_polyco_file_format(
                    fmt["format"], "w", None, fmt["write_method"]
                )
            elif fmt["read_method"] is not None:
                cls.add_polyco_file_format(
                    fmt["format"], "rw", fmt["read_method"], fmt["write_method"]
                )

    def __init__(self, filename=None, format="tempo"):
        """Initialize polycos from a file

        Parameters
        ---------
        filename : str or ~pathlib.Path or file-like
            The name of the input polyco file.
        format : str, optional
            The format of the file. Default is 'tempo'.

        """
        self.fileName = filename

        if filename is not None:
            log.info(f"Reading polycos from '{filename}'")
            if format not in [f["format"] for f in self.polycoFormats]:
                raise ValueError(
                    (
                        f"Unknown polyco file format '{format}" + "'\n"
                        "Please use function 'Polyco.add_polyco_file_format()'"
                        " to register the format\n"
                    )
                )
            else:
                self.fileFormat = format

            self.polycoTable = table.Table.read(filename, format=format)
            if len(self.polycoTable) == 0:
                raise ValueError("Zero polycos found for table")

    @classmethod
    def read(cls, filename, format="tempo"):
        """Read polyco file to a table.

        Parameters
        ---------
        filename : str or ~pathlib.Path or file-like
            The name of the input polyco file.
        format : str, optional
            The format of the file. Default is 'tempo'.

        Return
        ---------
        Polycos

        """
        return Polycos(filename, format=format)

    @classmethod
    def add_polyco_file_format(
        cls, formatName, methodMood, readMethod=None, writeMethod=None
    ):
        """
        Add a polyco file format and its reading/writing method to the class.
        Then register it to the table reading.

        Parameters
        ---------
        formatName : str
            The name for the format.
        methodMood : str
            One of ['r', 'w', 'rw'].
        readMethod : function
            The method for reading the file format.
        writeMethod : function
            The method for writting the file to disk.

        Notes
        -----
        The read/write methods should correspond to what are needed by :meth:`astropy.io.registry.register_reader` and
        :meth:`astropy.io.registry.register_writer`, respectively

        """
        assert methodMood.lower() in ["r", "w", "rw"]
        # Check if the format already exist.
        if (
            formatName in [f["format"] for f in cls.polycoFormats]
            or formatName in registry.get_formats()["Format"]
        ):
            return

        pFormat = {"format": formatName}

        if methodMood == "r":
            if readMethod is None:
                raise ValueError("Argument readMethod should not be 'None'.")

            pFormat["read_method"] = readMethod
            pFormat["write_method"] = writeMethod
            log.debug(f"Registering Polyco format '{formatName}' with mode='r'")
            registry.register_reader(
                pFormat["format"], table.Table, pFormat["read_method"]
            )
        elif methodMood == "w":
            if writeMethod is None:
                raise ValueError("Argument writeMethod should not be 'None'.")

            pFormat["read_method"] = readMethod
            pFormat["write_method"] = writeMethod
            log.debug(f"Registering Polyco format '{formatName}' with mode='w'")
            registry.register_writer(
                pFormat["format"], table.Table, pFormat["write_method"]
            )
        elif methodMood == "rw":
            if readMethod is None or writeMethod is None:
                raise ValueError(
                    "Argument readMethod and writeMethod " "should not be 'None'."
                )

            pFormat["read_method"] = readMethod
            pFormat["write_method"] = writeMethod
            log.debug(f"Registering Polyco format '{formatName}' with mode='rw'")
            registry.register_reader(
                pFormat["format"], table.Table, pFormat["read_method"]
            )
            registry.register_writer(
                pFormat["format"], table.Table, pFormat["write_method"]
            )

        cls.polycoFormats.append(pFormat)

    def read_polyco_file(self, filename, format="tempo"):
        """Read polyco file to a table.

        Included for backward compatibility.  It is better to use :meth:`pint.polycos.Polycos.read`.

        Parameters
        ---------
        filename : str or ~pathlib.Path or file-like
            The name of the input polyco file.
        format : str, optional
            The format of the file. Default is 'tempo'.

        Return
        ---------
        Polycos

        """
        raise DeprecationWarning(
            "Use `p=pint.polycos.Polycos.read()` rather than `p.read_polyco_file()`"
        )

        self.fileName = filename

        if format not in [f["format"] for f in self.polycoFormats]:
            raise ValueError(
                "Unknown polyco file format '" + format + "'\n"
                "Please use function 'Polyco.add_polyco_file_format()'"
                " to register the format\n"
            )
        else:
            self.fileFormat = format

        self.polycoTable = table.Table.read(filename, format=format)
        if len(self.polycoTable) == 0:
            raise ValueError("Zero polycos found for table")

    def __len__(self):
        return len(self.polycoTable)

    def __getitem__(self, item):
        return self.polycoTable[item]

    @classmethod
    def generate_polycos(
        cls,
        model,
        mjdStart,
        mjdEnd,
        obs,
        segLength,
        ncoeff,
        obsFreq,
        maxha=12.0,
        method="TEMPO",
        numNodes=20,
        progress=True,
    ):
        """
        Generate the polyco data.

        Parameters
        ----------
        model : TimingModel
            TimingModel to generate the Polycos
        mjdStart : float, np.longdouble
            Start time of polycos in mjd
        mjdEnd : float, np.longdouble
            Ending time of polycos in mjd
        obs : str
            Observatory code
        segLength : int
            Length of polyco segement [minutes]
        ncoeff : int
            Number of coefficents
        obsFreq : float
            Observing frequency [MHz]
        maxha : float, optional.
            Maximum hour angle. Default 12.0. Only 12.0 is supported for now.
        method : str, optional
            Method to generate polycos. Only the ``TEMPO`` method is supported for now.
        numNodes : int, optional
            Number of nodes for fitting. It cannot be less then the number of
            coefficents. Default: 20
        progress : bool, optional
            Whether or not to show the progress bar during calculation

        Return
        ---------
        Polycos
        """
        mjdStart = data2longdouble(mjdStart)
        mjdEnd = data2longdouble(mjdEnd)
        segLength = int(segLength)
        # Use the planetary ephemeris specified in the model, if available.
        if model.EPHEM.value is not None:
            ephem = model.EPHEM.value
        else:
            log.info("No ephemeris specified in model, using DE421 to generate polycos")
            ephem = "DE421"

        if maxha != 12.0:
            raise ValueError("Maximum hour angle != 12.0 is not supported.")

        # Make sure the number of nodes is bigger than number of coeffs.
        if numNodes < ncoeff:
            numNodes = ncoeff + 1

        mjdSpan = data2longdouble(segLength / MIN_PER_DAY)
        # Generate "nice" MJDs for consistency with what tempo2 does
        tmids = np.arange(
            int(mjdStart * 24) * 60, int(mjdEnd * 24) * 60 + segLength, segLength
        )
        tmids = data2longdouble(tmids) / MIN_PER_DAY

        if method != "TEMPO":
            raise NotImplementedError("Only TEMPO method has been implemented.")
        entryList = []
        obsFreq = float(obsFreq)

        # Using tempo1 method to create polycos
        # If you want to disable the progress bar, add disable=True to the tqdm() call.
        for tmid in tqdm(tmids, disable=not progress):
            tStart = tmid - mjdSpan / 2
            tStop = tmid + mjdSpan / 2
            nodes = np.linspace(tStart, tStop, numNodes)

            toaMid = toa.get_TOAs_array(
                (np.modf(tmid)[1], np.modf(tmid)[0]),
                obs=obs,
                freqs=obsFreq,
                ephem=ephem,
            )
            # toaMid = toa.get_TOAs_list(
            #     [toa.TOA()],
            # )

            refPhase = model.phase(toaMid, abs_phase=True)

            # Create node toas(Time sample using TOA class)
            # toaList = [
            #     toa.TOA(
            #         (np.modf(toaNode)[1], np.modf(toaNode)[0]),
            #         obs=obs,
            #         freq=obsFreq,
            #     )
            #     for toaNode in nodes
            # ]

            # toas = toa.get_TOAs_list(toaList, ephem=ephem)
            toas = toa.get_TOAs_array(
                (np.modf(nodes)[0], np.modf(nodes)[1]),
                obs=obs,
                freqs=obsFreq,
                ephem=ephem,
            )

            ph = model.phase(toas, abs_phase=True)
            dt = (nodes - tmid) * MIN_PER_DAY
            rdcPhase = ph - refPhase
            rdcPhase = rdcPhase.int - (dt * model.F0.value * 60.0) + rdcPhase.frac
            dtd = dt.astype(float)  # Truncate to double
            rdcPhased = rdcPhase.astype(float)
            coeffs = np.polyfit(dtd, rdcPhased, ncoeff - 1)[::-1]

            date, hms = Time(tmid, format="mjd", scale="utc").iso.split()
            yy, mm, dd = date.split("-")
            date = f"{dd}-{MONTHS[int(mm) - 1]}-{yy[-2:]}"
            hms = float(hms.replace(":", ""))

            entry = PolycoEntry(
                tmid,
                segLength,
                refPhase.int,
                refPhase.frac,
                model.F0.value,
                ncoeff,
                coeffs,
            )

            entry_dict = OrderedDict()
            entry_dict["psr"] = model.PSR.value
            entry_dict["date"] = date
            entry_dict["utc"] = hms
            entry_dict["tmid"] = tmid
            entry_dict["dm"] = model.DM.value
            entry_dict["doppler"] = 0.0
            entry_dict["logrms"] = 0.0
            entry_dict["mjd_span"] = segLength
            entry_dict["t_start"] = entry.tstart
            entry_dict["t_stop"] = entry.tstop
            entry_dict["obs"] = obs
            entry_dict["obsfreq"] = obsFreq

            if model.is_binary:
                binphase = model.orbital_phase(toaMid, radians=False)[0]
                entry_dict["binary_phase"] = binphase
                b = model.get_components_by_category()["pulsar_system"][0]
                entry_dict["f_orbit"] = 1 / b.PB.value

            entry_dict["entry"] = entry
            entryList.append(entry_dict)

        pTable = table.Table(entryList, meta={"name": "Polyco Data Table"})
        out = cls()
        out.polycoTable = pTable
        if len(out.polycoTable) == 0:
            raise ValueError("Zero polycos found for table")
        return out

    def write_polyco_file(self, filename="polyco.dat", format="tempo"):
        """Write Polyco table to a file.

        Parameters
        ---------
        filename : str
            The name of the polyco file. Default is 'polyco.dat'.
        format : str
            The format of the file. Default is 'tempo'.
        """

        if format not in [f["format"] for f in self.polycoFormats]:
            raise ValueError(
                (
                    f"Unknown polyco file format '{format}" + "'\n"
                    "Please use function 'self.add_polyco_file_format()'"
                    " to register the format\n"
                )
            )

        self.polycoTable.write(filename, format=format)

    def find_entry(self, t):
        """Find the right entry for the input time.

        Parameters
        ---------
        t: numpy.ndarray or float
           A time array in MJD.

        Returns
        -------
        numpy.ndarray
            Indices corresponding to the desired time(s)

        Raises
        ------
        ValueError
            If the input time(s) are not covered by the polycos
        """
        if not isinstance(t, (np.ndarray, list)):
            t = np.array([t])

        # Check if polyco table exists
        if self.polycoTable is None:
            raise ValueError("polycoTable not set!")
        t_start = self.polycoTable["t_start"]
        t_stop = self.polycoTable["t_stop"]

        start_idx = np.searchsorted(t_start, t) - 1
        stop_idx = np.searchsorted(t_stop, t)

        if not np.allclose(start_idx, stop_idx):
            raise ValueError("Some input times not covered by Polyco entries.")
        return start_idx

    def eval_phase(self, t):
        """Polyco evaluation of fractional phase

        Parameters
        ---------
        t: numpy.ndarray or float
           An time array in MJD. Time sample should be in order

        Returns
        ---------
        numpy.ndarray
             Fractional phase

        Notes
        -----
        Returns fractional part of :meth:`pint.polycos.Polycos.eval_abs_phase`
        """
        if not isinstance(t, np.ndarray) and not isinstance(t, list):
            t = np.array([t])
        return self.eval_abs_phase(t).frac

    def eval_abs_phase(self, t):
        """
        Polyco evaluate absolute phase for a time array.

        Parameters
        ---------
        t: numpy.ndarray or float
           An time array in MJD. Time sample should be in order

        Returns
        ---------
        pint.phase.Phase
             Polyco evaluated absolute phase for t.

        Notes
        -----
        Calculates phase according to:

        .. math::

            \\phi = \\phi_0 + 60 \\Delta T f_0 + COEFF[1] + COEFF[2] \Delta T + COEFF[3] \Delta T^2 + \\ldots

        Calculation done using :meth:`pint.polycos.PolycoEntry.evalabsphase`
        """
        if not isinstance(t, (np.ndarray, list)):
            t = np.array([t])

        entryIndex = self.find_entry(t)
        phaseInt = ()
        phaseFrac = ()
        # Compute phase for time in each entry
        for i in range(len(self.polycoTable)):
            mask = np.where(entryIndex == i)  # Build mask for time in each entry
            t_in_entry = t[mask]
            if len(t_in_entry) == 0:
                continue
            # Calculate the phase as an array
            absp = self.polycoTable["entry"][i].evalabsphase(t_in_entry)
            phaseInt += (absp.int,)
            phaseFrac += (absp.frac,)
            # Maybe add sort function here, since the time has been masked.
        phaseInt = np.hstack(phaseInt).value
        phaseFrac = np.hstack(phaseFrac).value
        return Phase(phaseInt, phaseFrac)

    def eval_spin_freq(self, t):
        """
        Polyco evaluate spin frequency for a time array.

        Parameters
        ---------
        t: numpy.ndarray or float
           An time array in MJD. Time samples should be in order

        Returns
        ---------
        numpy.ndarray
             Polyco evaluated spin frequency [Hz] at time t.

        Notes
        -----
        Calculates frequency according to:

        .. math::

            f({\\rm Hz}) = f_0 + \\frac{1}{60}\\left(COEFF[2] + 2 \\Delta T COEFF[3] + 3 \\Delta T^2 COEFF[4] + \\ldots\\right)

        Calculation done using :meth:`pint.polycos.PolycoEntry.evalfreq`
        """
        if not isinstance(t, np.ndarray) and not isinstance(t, list):
            t = np.array([t])

        entryIndex = self.find_entry(t)
        spinFreq = np.zeros(len(t), dtype=np.longdouble)

        for i, idx in enumerate(entryIndex):
            spinFreq[i] = self[idx]["entry"].evalfreq(t[i])

        return spinFreq

    def eval_spin_freq_derivative(self, t):
        """
        Polyco evaluate spin frequency derivative for a time array.

        Parameters
        ---------
        t: numpy.ndarray or float
           An time array in MJD. Time samples should be in order

        Returns
        ---------
        numpy.ndarray
             Polyco evaluated spin frequency derivative [Hz/s] at time t.

        Notes
        -----

        Calculation done using :meth:`pint.polycos.PolycoEntry.evalfreqderiv`
        """
        if not isinstance(t, np.ndarray) and not isinstance(t, list):
            t = np.array([t])

        entryIndex = self.find_entry(t)
        spinFreqDeriv = np.zeros(len(t), dtype=np.longdouble)

        for i, idx in enumerate(entryIndex):
            spinFreqDeriv[i] = self[idx]["entry"].evalfreqderiv(t[i])

        return spinFreqDeriv


# load the default registry on import
Polycos._register()
