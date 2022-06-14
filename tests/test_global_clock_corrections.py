import os
import time

import pytest
from astropy.utils.data import clear_download_cache

from pint.observatory.global_clock_corrections import get_file, update_all

# FIXME: this should be less painful with fixtures somehow


def test_not_existing(tmp_path):
    url_base = tmp_path.as_uri() + "/"
    url_mirrors = [url_base]

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / test_file_name
    contents = "some text"
    fn.write_text(contents)

    df = get_file(test_file_name, url_base=url_base, url_mirrors=url_mirrors)
    assert open(df).read() == contents


def test_existing(tmp_path):
    url_base = tmp_path.as_uri() + "/"
    url_mirrors = [url_base]

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / test_file_name
    contents = "some text"
    fn.write_text(contents)

    # Insert into cache
    get_file(test_file_name, url_base=url_base, url_mirrors=url_mirrors)

    # Look it up without any sources to see if it's there
    df = get_file(test_file_name, url_base=url_base, url_mirrors=[])
    assert open(df).read() == contents


def test_update_needed(tmp_path):
    url_base = tmp_path.as_uri() + "/"
    url_mirrors = [url_base]

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / "test_file"
    contents = "some text"
    fn.write_text(contents)

    # Insert into cache
    df_old = get_file(test_file_name, url_base=url_base, url_mirrors=url_mirrors)
    # Set the mtime to old
    long_ago = time.time() - 1_000_000
    os.utime(df_old, (long_ago, long_ago))

    new_contents = "some new text"
    fn.write_text(new_contents)
    df = get_file(test_file_name, url_base=url_base, url_mirrors=url_mirrors)
    assert open(df).read() == new_contents


def test_update_all_runs(tmp_path):
    update_all(tmp_path)
