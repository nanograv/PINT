"""Tools for working with clock corrections obtained from a global location.

The goal is for PINT (and other programs) to be able to download up-to-date
observatory clock corrections from a central location, which observatories
or third parties will update as new clock correction data becomes available.

The global repository is currently hosted on github. Available clock correction
files and their updating requirements are listed in a file there called index.txt.
This too is checked occasionally for updates.

The downloaded files are stored in the Astropy cache,
to clear out old files you will want to do
``astropy.utils.data.clear_download_cache()``.
"""
import collections
import time
from pathlib import Path
from warnings import warn

from astropy.utils.data import download_file
from loguru import logger as log

from pint.pulsar_mjd import Time

global_clock_correction_url_base = (
    "https://raw.githubusercontent.com/ipta/pulsar-clock-corrections/main/"
)

# These are mirrors that have (presumed) identical data but might be available when
# the base URL is not. If the base URL is not included it will not actually be
# checked.
global_clock_correction_url_mirrors = [global_clock_correction_url_base]

# PINT will check the index if it is more than this old
index_name = "index.txt"
index_update_interval_days = 1


def get_file(
    name,
    update_interval_days=7,
    download_policy="if_expired",
    url_base=None,
    url_mirrors=None,
    invalid_if_older_than=None,
):
    """Obtain a local file pointing to a current version of name.

    The mtime of the returned file will record when the data was last obtained
    from the internet.

    Parameters
    ----------
    name : str
        The name of the file within the repository.
    update_interval_days : float
        How old the cached version can be before needing to be updated. Can be infinity.
    download_policy : str
        When to try downloading from the Net. Options are: "always", "never",
        "if_expired" (if the cached version is older than update_interval_days),
        or "if_missing" (only if nothing is currently available).
    url_base : str or None
        If provided, override the repository location stored in the source code.
        Useful mostly for testing.
    url_mirrors : list of str or None
        If provided, override the repository mirrors stored in the source code.
        Useful mostly for testing.
    invalid_if_older_than : astropy.time.Time or None
        Re-download the file if the cached version is older than this.

    Returns
    -------
    pathlib.Path
        The location of the file.
    """
    log.trace(f"File {name} requested")
    if url_base is None:
        url_base = global_clock_correction_url_base
        if url_mirrors is None:
            url_mirrors = global_clock_correction_url_mirrors
    elif url_mirrors is None:
        url_mirrors = [url_base]
    local_file = None
    remote_url = url_base + name
    mirror_urls = [u + name for u in url_mirrors]

    if download_policy != "always":
        try:
            local_file = Path(download_file(remote_url, cache=True, sources=[]))
            log.trace(f"file {remote_url} found in cache at path: {local_file}")
        except KeyError as e:
            log.trace(f"file {remote_url} not found in cache")
            if download_policy == "never":
                raise FileNotFoundError(name) from e

    if download_policy == "if_missing" and local_file is not None:
        log.trace(
            f"File {name} found and returned due to download policy {download_policy}"
        )
        return local_file

    if local_file is not None:
        file_time = Path(local_file).stat().st_mtime
        if (
            invalid_if_older_than is not None
            and Time(file_time, format="unix") < invalid_if_older_than
        ):
            log.trace(
                f"File {name} found but re-downloaded because "
                f"it is older than {invalid_if_older_than}"
            )
            local_file = None

    if download_policy == "if_expired" and local_file is not None:
        # FIXME: will update_interval_days=np.inf work with unit conversion?
        file_time = Path(local_file).stat().st_mtime
        now = time.time()
        if now - file_time < update_interval_days * 86400:
            # Not expired
            log.trace(
                f"File {name} found and returned due to "
                f"download policy {download_policy} and recentness"
            )
            return local_file

    # By this point we know we need a new file but we want it to wind up in
    # the cache
    log.info(
        f"File {name} to be downloaded due to download policy "
        f"{download_policy}: {remote_url}"
    )
    try:
        return Path(download_file(remote_url, cache="update", sources=mirror_urls))
    except IOError as e:
        if download_policy != "if_expired" or local_file is None:
            raise
        warn(
            f"File {name} should be downloaded but {local_file} is being used "
            f"because an error occurred: {e}"
        )
        return local_file


IndexEntry = collections.namedtuple(
    "IndexEntry", ["file", "update_interval_days", "invalid_if_older_than", "extra"]
)


class Index:
    """Index of files available from the global repository.

    The list is obtained by downloading (via the cache) the file ``index.txt``
    from the repository. The result is stored in a dictionary ``index.files`` that
    maps filenames (like ``gps2utc.clk`` to IndexEntry objects describing those
    files. These entries contain information about expiry and validity of the file.

    For parameter meanings see :func:`pint.observatory.global_clock_corrections.get_file`.
    """

    def __init__(self, download_policy="if_expired", url_base=None, url_mirrors=None):
        index_file = get_file(
            index_name,
            index_update_interval_days,
            download_policy=download_policy,
            url_base=url_base,
            url_mirrors=url_mirrors,
        )
        self.files = {}
        for line in open(index_file):
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                continue
            e = line.split(maxsplit=3)
            date = None if e[2] == "---" else Time(e[2], format="iso")
            t = IndexEntry(
                file=e[0],
                update_interval_days=float(e[1]),
                invalid_if_older_than=date,
                extra=e[3] if len(e) > 3 else "",
            )
            file = Path(t.file).name
            self.files[file] = t


def get_clock_correction_file(
    filename, download_policy="if_expired", url_base=None, url_mirrors=None
):
    """Obtain a current version of the clock correction file.

    The clock correction file is looked up in the index downloaded from the
    repository; unknown clock correction files trigger a KeyError. Known
    ones use the index's information about when they expire.

    Parameters
    ----------
    name : str
        The name of the file within the repository.
    download_policy : str
        When to try downloading from the Net. Options are: "always", "never",
        "if_expired" (if the cached version is older than update_interval_days),
        or "if_missing" (only if nothing is currently available).
    url_base : str or None
        If provided, override the repository location stored in the source code.
        Useful mostly for testing.
    url_mirrors : list of str or None
        If provided, override the repository mirrors stored in the source code.
        Useful mostly for testing.
    """

    # FIXME: cache/share the index object?
    index = Index(
        download_policy=download_policy, url_base=url_base, url_mirrors=url_mirrors
    )

    details = index.files[filename]
    return get_file(
        details.file,
        update_interval_days=details.update_interval_days,
        download_policy=download_policy,
        url_base=url_base,
        url_mirrors=url_mirrors,
        invalid_if_older_than=details.invalid_if_older_than,
    )


def update_all(
    export_to=None, download_policy="if_expired", url_base=None, url_mirrors=None
):
    """Download and update all clock corrections in the index.

    You can also export them all to a directory.

    This includes all the files in the repository, regardless of what
    PINT knows about them. (For example, the repository probably
    includes the file `leap.sec` but PINT does not use it.) If you want to
    download only the clock correction files that PINT uses,
    see :func:`pint.observatory.update_clock_files`.


    Parameters
    ----------
    export_to : str or pathlib.Path, optional
        If provided, write all files to this directory.
    download_policy : str
        Under what conditions to download a new file.
    url_base : str, optional
        The location of the global repository. Useful for debugging.
    url_mirrors : list of str, optional
        A list of places to look for the content. Useful for debugging.
    """
    index = Index(
        download_policy=download_policy, url_base=url_base, url_mirrors=url_mirrors
    )
    for filename, details in index.files.items():
        f = get_file(
            details.file,
            update_interval_days=details.update_interval_days,
            download_policy=download_policy,
            url_base=url_base,
            url_mirrors=url_mirrors,
            invalid_if_older_than=details.invalid_if_older_than,
        )
        if export_to is not None:
            (Path(export_to) / filename).write_text(Path(f).read_text())
