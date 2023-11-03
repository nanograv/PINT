import os
import time
from pathlib import Path

import pytest
from astropy.time import Time
from astropy.utils.data import clear_download_cache

from pint.observatory.global_clock_corrections import get_file, update_all

# FIXME: this should be less painful with fixtures somehow


def test_not_existing(tmp_path, temp_cache):
    url_base = f"{tmp_path.as_uri()}/"

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / test_file_name
    contents = "some text"
    fn.write_text(contents)

    df = get_file(test_file_name, url_base=url_base)
    assert open(df).read() == contents


def test_existing(tmp_path, temp_cache):
    url_base = f"{tmp_path.as_uri()}/"

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / test_file_name
    contents = "some text"
    fn.write_text(contents)

    # Insert into cache
    get_file(test_file_name, url_base=url_base)

    # Look it up without any sources to see if it's there
    df = get_file(test_file_name, url_base=url_base, url_mirrors=[])
    assert open(df).read() == contents


def test_update_needed(tmp_path, temp_cache):
    url_base = f"{tmp_path.as_uri()}/"

    test_file_name = "test_file"
    url = url_base + test_file_name
    clear_download_cache(url)

    fn = tmp_path / "test_file"
    contents = "some text"
    fn.write_text(contents)

    # Insert into cache
    df_old = get_file(test_file_name, url_base=url_base)
    # Set the mtime to old
    long_ago = time.time() - 1_000_000
    os.utime(df_old, (long_ago, long_ago))

    new_contents = "some new text"
    fn.write_text(new_contents)
    df = get_file(test_file_name, url_base=url_base)
    assert open(df).read() == new_contents


# Note we don't use temp_cache here because it forces us to redownload everything
def test_update_all_runs(tmp_path):
    update_all(tmp_path)
    assert (tmp_path / "gps2utc.clk").exists()
    assert (tmp_path / "time_gbt.dat").exists()
    assert (tmp_path / "tai2tt_bipm2019.clk").exists()


@pytest.fixture
def sandbox(tmp_path):
    class Sandbox:
        pass

    sandbox = Sandbox()

    sandbox.repo_path = tmp_path / "repo"
    sandbox.repo_path.mkdir()
    sandbox.filename = "file.txt"
    sandbox.repo_file = sandbox.repo_path / sandbox.filename
    sandbox.repo_file.write_text("Sample file contents")
    sandbox.index_file = sandbox.repo_path / "index.txt"
    sandbox.index_file.write_text(
        f"""
        # File                                   Update (days)   Invalid if older than
        {sandbox.filename}                                  7.0   --- """
    )
    sandbox.repo_url = f"{sandbox.repo_path.as_uri()}/"

    return sandbox


@pytest.mark.parametrize(
    "download_policy", [None, "if_missing", "always", "if_expired"]
)
def test_can_download(sandbox, temp_cache, download_policy):
    f = get_file(
        sandbox.filename, url_base=sandbox.repo_url, download_policy=download_policy
    )
    assert Path(f).read_text() == sandbox.repo_file.read_text()


def test_never_policy_nonexistent_raises(sandbox, temp_cache):
    with pytest.raises(FileNotFoundError):
        get_file(sandbox.filename, url_base=sandbox.repo_url, download_policy="never")


def test_never_policy_existent_succeeds(sandbox, temp_cache):
    get_file(sandbox.filename, url_base=sandbox.repo_url)
    get_file(sandbox.filename, url_base=sandbox.repo_url, download_policy="never")


def test_if_missing_policy_ancient_keeps(sandbox, temp_cache):
    f = get_file(sandbox.filename, url_base=sandbox.repo_url)
    long_ago = time.time() - 1e8  # More than 7 days
    os.utime(f, (long_ago, long_ago))

    # With url_mirrors=[], attempting to download raises an exception
    get_file(
        sandbox.filename,
        url_base=sandbox.repo_url,
        download_policy="if_missing",
        url_mirrors=[],
    )


def test_if_expired_policy_ancient_downloads(sandbox, temp_cache):
    f = get_file(sandbox.filename, url_base=sandbox.repo_url)
    long_ago = time.time() - 1e8  # More than 7 days
    os.utime(f, (long_ago, long_ago))

    # With url_mirrors=[], attempting to download raises an exception
    with pytest.raises(KeyError):
        get_file(
            sandbox.filename,
            url_base=sandbox.repo_url,
            download_policy="if_expired",
            url_mirrors=[],
        )


def test_if_expired_policy_recent_keeps(sandbox, temp_cache):
    get_file(sandbox.filename, url_base=sandbox.repo_url)

    # With url_mirrors=[], attempting to download raises an exception
    get_file(
        sandbox.filename,
        url_base=sandbox.repo_url,
        download_policy="if_expired",
        url_mirrors=[],
    )


def test_always_policy_ancient_downloads(sandbox, temp_cache):
    f = get_file(sandbox.filename, url_base=sandbox.repo_url)
    long_ago = time.time() - 1e8  # More than 7 days
    os.utime(f, (long_ago, long_ago))

    # With url_mirrors=[], attempting to download raises an exception
    with pytest.raises(KeyError):
        get_file(
            sandbox.filename,
            url_base=sandbox.repo_url,
            download_policy="always",
            url_mirrors=[],
        )


def test_invalidation_works(sandbox, temp_cache):
    f = get_file(sandbox.filename, url_base=sandbox.repo_url)
    long_ago = time.time() - 1e5  # less than 7 days
    os.utime(f, (long_ago, long_ago))

    # With url_mirrors=[], attempting to download raises an exception
    with pytest.raises(KeyError):
        get_file(
            sandbox.filename,
            url_base=sandbox.repo_url,
            invalid_if_older_than=Time.now(),
            url_mirrors=[],
        )


def test_invalidation_not_too_aggressive(sandbox, temp_cache):
    get_file(sandbox.filename, url_base=sandbox.repo_url)
    get_file(
        sandbox.filename,
        url_base=sandbox.repo_url,
        invalid_if_older_than=Time("2001-01-01", format="iso"),
        url_mirrors=[],
    )
