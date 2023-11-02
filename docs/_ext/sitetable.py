from docutils import nodes
from docutils.parsers.rst.directives.tables import Table
from docutils.parsers.rst.directives import unchanged_required
from docutils.statemachine import ViewList
import urllib.parse
import pint.observatory
import numpy as np

_iptaclock_baseurl = "https://ipta.github.io/pulsar-clock-corrections"
_googlesearch_baseurl = "https://www.google.com/maps/search/?"


class SiteTable(Table):
    option_spec = {"class": unchanged_required}
    has_content = False

    def run(self):
        columns = [
            ("Name / Aliases", "name", 10),
            ("Origin", "origin", 50),
            ("Location", "location", 20),
            ("Clock File(s)", "clock", 20),
        ]

        class_ = None

        table = nodes.table()
        tgroup = nodes.tgroup(len(columns))
        table += tgroup

        thead = nodes.thead()
        row = nodes.row()
        for label, _, w in columns:
            tgroup += nodes.colspec(colwidth=w)
            entry = nodes.entry()
            row += entry
            entry += nodes.paragraph(text=label)
        thead += row
        tgroup += thead

        tbody = nodes.tbody()
        for name in sorted(pint.observatory.Observatory.names()):
            o = pint.observatory.get_observatory(name)
            row = nodes.row()
            for _, c, _ in columns:
                entry = nodes.entry()
                row += entry
                if c == "name":
                    entry += nodes.strong(text=name)
                    if len(o.aliases) > 0:
                        entry += nodes.paragraph(text=" (" + ", ".join(o.aliases) + ")")
                elif c == "origin":
                    if o.fullname != o.name:
                        entry += nodes.strong(text=o.fullname + ".\n")
                    entry += nodes.paragraph(text=o.origin)
                elif c == "location":
                    loc = o.earth_location_itrf()
                    if loc is not None:
                        lat = loc.lat.value
                        lon = loc.lon.value
                        text = f"{np.abs(lat):.4f}{'N' if lat >=0 else 'S'}, {np.abs(lon):.4f}{'E' if lon >= 0 else 'W'}"
                        # https://developers.google.com/maps/documentation/urls/get-started
                        url = _googlesearch_baseurl + urllib.parse.urlencode(
                            {"api": "1", "query": f"{lat},{lon}"}
                        )
                        para = nodes.paragraph()
                        refnode = nodes.reference("", "", internal=False, refuri=url)
                        innernode = nodes.emphasis(text, text)
                        refnode.append(innernode)
                        para += refnode
                        entry += para
                elif c == "clock":
                    if hasattr(o, "clock_files"):
                        for clockfile in o.clock_files:
                            clockfilename = (
                                clockfile
                                if isinstance(clockfile, str)
                                else clockfile["name"]
                            )
                            para = nodes.paragraph()
                            dirname = "tempo" if o.clock_fmt == "tempo" else "T2runtime"
                            url = f"{_iptaclock_baseurl}/{dirname}/clock/{clockfilename}.html"
                            refnode = nodes.reference(
                                "", "", internal=False, refuri=url
                            )
                            innernode = nodes.emphasis(clockfilename, clockfilename)
                            refnode.append(innernode)
                            para += refnode
                            entry += para
            tbody += row
        tgroup += tbody
        return [table]


def setup(app):
    app.add_directive("sitetable", SiteTable)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
