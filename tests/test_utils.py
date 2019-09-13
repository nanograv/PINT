"""Test basic functionality of the :module:`pint.utils`."""

from __future__ import absolute_import, division, print_function

from tempfile import NamedTemporaryFile

import pytest

from pint.utils import open_or_use, taylor_horner, lines_of, interesting_lines


def test_taylor_horner_basic():
    """Check basic calculation against schoolbook formula."""
    assert taylor_horner(2.0, [10]) == 10
    assert taylor_horner(2.0, [10, 3]) == 10 + 3 * 2.0
    assert taylor_horner(2.0, [10, 3, 4]) == 10 + 3 * 2.0 + 4 * 2.0 ** 2 / 2.0
    assert taylor_horner(
        2.0, [10, 3, 4, 12]
    ) == 10 + 3 * 2.0 + 4 * 2.0 ** 2 / 2.0 + 12 * 2.0 ** 3 / (3.0 * 2.0)


contents = """Random text file

with some stuff

"""


def test_open_or_use_string_write():
    """Check writing to filename."""
    with NamedTemporaryFile("w") as w:
        w.write(contents)
        w.flush()
        with open_or_use(w.name) as r:
            assert r.read() == contents


def test_open_or_use_string_read():
    """Check reading from filename."""
    with NamedTemporaryFile("r") as r:
        with open_or_use(r.name, "w") as w:
            w.write(contents)
        assert r.read() == contents


def test_open_or_use_file_write():
    """Check writing to file."""
    with NamedTemporaryFile("w") as wo:
        with open_or_use(wo) as w:
            w.write(contents)
        wo.flush()
        assert open(wo.name).read() == contents


def test_open_or_use_file_read():
    """Check reading from file."""
    with NamedTemporaryFile("r") as ro:
        with open(ro.name, "w") as w:
            w.write(contents)
        with open_or_use(ro) as r:
            assert r.read() == contents


@pytest.mark.parametrize("contents", ["", " ", "\n", "aa", "a\na", contents])
def test_lines_of(contents):
    """Check we get the same lines back through various means."""
    lines = contents.splitlines(True)
    assert list(lines_of(lines)) == lines
    with NamedTemporaryFile("w") as w:
        w.write(contents)
        w.flush()
        assert list(lines_of(w.name)) == lines
        with open(w.name) as r:
            assert list(lines_of(r)) == lines


@pytest.mark.parametrize(
    "lines, goodlines, comments",
    [
        ([" text stuff \n"], ["text stuff"], None),
        ([" text stuff \n\n"], ["text stuff"], None),
        ([" text stuff "], ["text stuff"], None),
        ([" text stuff \n"] * 7, ["text stuff"] * 7, None),
        (["\ttext stuff \n"], ["text stuff"], None),
        (["#\ttext stuff \n"], [], "#"),
        (["  #\ttext stuff \n"], [], "#"),
        (["C \ttext stuff \n"], [], "C "),
        (["  C \ttext stuff \n"], [], "C "),
        (["C\ttext stuff \n"], ["C\ttext stuff"], "C "),
        (["#\ttext stuff \n"], [], ("#", "C ")),
        (["C \ttext stuff \n"], [], ("#", "C ")),
        (["C\ttext stuff \n"], ["C\ttext stuff"], ("#", "C ")),
    ],
)
def test_interesting_lines(lines, goodlines, comments):
    """Check various patterns of text and comments."""
    assert list(interesting_lines(lines, comments=comments)) == goodlines


def test_interesting_lines_input_validation():
    """Check it lets the user know about invalid comment markers."""
    with pytest.raises(ValueError):
        for l in interesting_lines([""], comments=" C "):
            pass
