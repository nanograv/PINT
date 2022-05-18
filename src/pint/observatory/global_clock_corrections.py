"""Tools for working with clock corrections obtained from a global location.

The goal is for PINT (and other programs) to be able to download up-to-date
observatory clock corrections from a central location, which observatories
or third parties will update as new clock correction data becomes available.
"""
import collections
import time
import warnings
from pathlib import Path

from astropy.utils.data import download_file
from loguru import logger as log

from pint.pulsar_mjd import Time

global_clock_correction_url_base = (
    "https://raw.githubusercontent.com/aarchiba/pulsar-clock-corrections/main/"
)

# These are mirrors that have (presumed) identical data but might be available when
# the base URL is not. If the base URL is not included it will not actually be
# checked.
global_clock_correction_url_mirrors = [global_clock_correction_url_base]

index_name = "index.txt"
index_update_interval_days = 0.01


def get_file(
    name,
    update_interval_days=7,
    download_policy="if_expired",
    url_base=None,
    url_mirrors=None,
):
    """Obtain a local file pointing to a current version of name.

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
    """

    log.trace(f"File {name} requested")
    if url_base is None:
        url_base = global_clock_correction_url_base
    if url_mirrors is None:
        url_mirrors = global_clock_correction_url_mirrors
    local_file = None
    remote_url = url_base + name
    mirror_urls = [u + name for u in url_mirrors]

    if download_policy != "always":
        try:
            local_file = download_file(
                remote_url, cache=True, sources=[], pkgname="PINT"
            )
        except KeyError:
            if download_policy == "never":
                raise FileNotFoundError(name)

    if download_policy == "if_missing" and local_file is not None:
        log.trace(
            f"File {name} found and returned due to download policy {download_policy}"
        )

    if download_policy == "if_expired" and local_file is not None:
        # FIXME: will update_interval_days=np.inf work with unit conversion?
        file_time = Path(local_file).stat().st_mtime
        now = time.time()
        if now - file_time < update_interval_days * 86400:
            # Not expired
            log.trace(
                f"File {name} found and returned due to download policy {download_policy} and recentness"
            )
            return local_file

    # By this point we know we need a new file but we want it to wind up in
    # the cache
    log.info(f"File {name} to be downloaded due to download policy {download_policy}")
    return download_file(
        remote_url, cache="update", sources=mirror_urls, pkgname="PINT"
    )


IndexEntry = collections.namedtuple(
    "IndexEntry", ["file", "format", "update_interval_days", "extra"]
)


class Index:
    def __init__(self, download_policy="if_expired", url_base=None, url_mirrors=None):
        if url_base is None:
            url_base = global_clock_correction_url_base
        if url_mirrors is None:
            url_mirrors = global_clock_correction_url_mirrors

        index_file = get_file(
            index_name,
            index_update_interval_days,
            download_policy=download_policy,
            url_base=url_base,
            url_mirrors=url_mirrors,
        )
        self.files = {}
        for l in open(index_file).readlines():
            l = l.strip()
            if l.startswith("#"):
                continue
            e = l.split(maxsplit=3)
            t = IndexEntry(
                file=e[0],
                format=e[1],
                update_interval_days=float(e[2]),
                extra=e[3] if len(e) > 3 else "",
            )
            file = Path(t.file).name
            self.files[file] = t


_the_index = None


def get_clock_correction_file(
    filename, download_policy="if_expired", url_base=None, url_mirrors=None
):
    """Obtain a current version of the clock correction file."""

    if url_base is None:
        url_base = global_clock_correction_url_base
    if url_mirrors is None:
        url_mirrors = global_clock_correction_url_mirrors

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
    )
