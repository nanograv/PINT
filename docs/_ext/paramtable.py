from docutils import nodes
from docutils.parsers.rst.directives.tables import Table
from docutils.parsers.rst.directives import unchanged_required
from docutils.statemachine import ViewList
import pint.utils


class ParamTable(Table):
    option_spec = {"class": unchanged_required}
    has_content = False

    def run(self):
        columns = [
            ("Name / Aliases", "name", 10),
            ("Description", "description", 30),
            ("Kind", "kind", 10),
        ]
        if "class" in self.options:
            class_ = eval(self.options["class"])
        else:
            class_ = None
            columns.append(("Components", "classes", 30))

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
        for d in pint.utils.list_parameters(class_):
            row = nodes.row()
            for _, c, _ in columns:
                entry = nodes.entry()
                row += entry
                if c not in d:
                    continue
                if c == "classes":
                    para = nodes.paragraph()
                    for cl in d[c]:
                        self.state.nested_parse(
                            ViewList([f":class:`~{cl}` "], "bogus.rst"),
                            self.content_offset,
                            para,
                        )
                    entry += para
                elif c == "name":
                    text = d[c]
                    if alias_list := d.get("aliases", []):
                        text += " / " + ", ".join(d["aliases"])
                    entry += nodes.paragraph(text=text)
                elif isinstance(d[c], str):
                    entry += nodes.paragraph(text=d[c])
                elif isinstance(d[c], list):
                    entry += nodes.paragraph(text=", ".join(d[c]))
                elif d[c] is not None:
                    entry += nodes.paragraph(text=str(d[c]))
            tbody += row
        tgroup += tbody

        return [table]


def setup(app):
    app.add_directive("paramtable", ParamTable)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
