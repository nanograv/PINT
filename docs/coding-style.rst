.. _CodingStyle:

PINT coding style
-----------------

   - Code should follow PEP8_. This constrains whitespace,
     formatting, and naming.
   - Code should follow the black_ style.
   - Use the standard formats for imports.

      - The only abbreviated imports are ``import numpy as np``,
        ``import astropy.units as u``, and
        ``import astropy.constants as c``; always use these modules
        in this form.
      - If you want to import a deeply nested module, use
        ``from pint.models import parameter``.
      - If you are using a function frequently or the module name
        is long, use ``from pint.utils import interesting_lines``
      - Sort imports with isort_.
      - Remove all unused imports.

   - Raise an exception if the code cannot continue and produce
     correct results.
   - Use :mod:`~astropy.log` to signal conditions the user should
     know about but that do not prevent the code from producing
     correct results.
   - Use keyword argument names when calling a function with more
     than two or three arguments.
   - Every public function or class should have a docstring. It
     should be in the correct format (numpy guidelines_).
   - Use :class:`~astropy.units.Quantity` for things with physical units.
   - PINT should work with python 2.7, 3.5, 3.6, 3.7, and later. There
     is no need to maintain compatibility with older versions, so use
     approprate modern constructs like sets and iterators.
   - Use six_ to manage python 2/3 compatibility problems.

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pypi.org/project/isort/
.. _guidelines: https://numpy.org/devdocs/docs/howto_document.html
.. _six: https://six.readthedocs.io/

