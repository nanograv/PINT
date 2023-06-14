from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList


class ComponentList(Directive):
    has_content = False

    def run(self):
        content = ViewList()
        source = "bogus.rst"
        content.append("Components supported by PINT:", source)
        content.append("", source)

        import pint.models.timing_model

        d = pint.models.timing_model.Component.component_types.copy()
        for k in sorted(d.keys()):
            class_ = d[k]
            full_name = f"{class_.__module__}.{class_.__name__}"
            if hasattr(class_, "__doc__") and class_.__doc__ is not None:
                doc = class_.__doc__.split("\n")[0].strip()
            else:
                doc = ""
            msg = f"* :class:`~{full_name}` - {doc}"
            inst = class_()
            if hasattr(inst, "binary_model_name"):
                msg += f" (``BINARY {inst.binary_model_name}``)"
            content.append(msg, source)
        para = nodes.paragraph()
        self.state.nested_parse(
            content,
            self.content_offset,
            para,
        )

        return [para]


def setup(app):
    app.add_directive("componentlist", ComponentList)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
