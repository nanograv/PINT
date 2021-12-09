"""Utility Fuctions to translate parfile format. This funciton is based on
Nihan Pol's format translator function at
https://gitlab.nanograv.org/nano-time/timing_analysis/-/blob/15yr/src/timing_analysis/lite_utils.py
"""


__all__ = ["convert_pint_to_tempo_parfile"]


def convert_pint_to_tempo_parfile(parfile_str, format):
    """Function to convert PINT produced parfile to tempo-compatible parfile.

    Removes CHI2 and SWM from the parfile. Changes EFAC/EQUAD to T2EFAC/T2EQUAD.
    If converting to tempo2 parfile, make sure ECL IERS2003 is set.

    Parameters
    ----------
    parfile_str: str
        PINT parfile string.
    format : str
        Output format ['tempo', 'tempo2']; if tempo2, sets ECL IERS2003.

    Returns
    -------
    new_par : str
        The new parfile format string.
    Notes
    -----
    The parfile line changes from PINT to Tempo and Tempo2 format.
    +-----------+----------------+----------------+
    | PINT line | TEMPO line     | TEMPO2 line    |
    +===========+================+================+
    | Head      | Empty          | MODE 1         |
    +-----------+----------------+----------------+
    | CHI2      | No Change      | No Change      |
    +-----------+----------------+----------------+
    | SWM       | No Change      | No Change      |
    +-----------+----------------+----------------+
    | A1DOT     | name: XDOT     | name: XDOT     |
    +-----------+----------------+----------------+
    | STIGMA    | name: VARSIGMA | name: VARSIGMA |
    +-----------+----------------+----------------+
    | NHARMS    | type: int      | type: int      |
    +-----------+----------------+----------------+
    | ECL       | No Change      | value: IERS2003|
    +-----------+----------------+----------------+
    | EFAC      | name: T2EFAC   | name: T2EFAC   |
    +-----------+----------------+----------------+
    | EQUAD     | name: T2EQUAD  | name: T2EQUAD  |
    +-----------+----------------+----------------+
    | T2CMETHOD | No Change      | Comment out    |
    +-----------+----------------+----------------+
    """
    par_lines = parfile_str.split("\n")
    new_par = []
    for ii in range(len(par_lines)):
        entries = par_lines[ii].split(" ")
        if ii == 0:
            if format == "tempo2":
                new_par.append("MODE 1")
        if "CHI2" in entries:
            continue
        elif "SWM" in entries:
            continue
        elif "A1DOT" in entries:
            entries[0] = "XDOT"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif "STIGMA" in entries:
            entries[0] = "VARSIGMA"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif "NHARMS" in entries:
            entries[-1] = str(int(float(entries[-1])))
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif ("ECL" in entries) and (format == "tempo2"):
            entries[-1] = "IERS2003"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif "EFAC" in entries:
            entries[0] = "T2EFAC"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif "EQUAD" in entries:
            entries[0] = "T2EQUAD"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        elif ("T2CMETHOD" in entries) and (format == "tempo2"):
            entries[0] = "#T2CMETHOD"
            new_entry = " ".join(entries)
            new_par.append(new_entry)
        else:
            new_par.append(par_lines[ii])
    return "\n".join(new_par)
