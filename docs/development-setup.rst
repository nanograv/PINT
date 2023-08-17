.. _`Developing PINT`:

How to Set Up Your Environment For PINT Development
===================================================

See also :ref:`Contributing`

Working on PINT code requires a few more tools than simply running PINT, and
there are a few settings that can make it much easier. Some of what follows
will depend on your editor, but most of it is possible in any modern
programmer's editor, including Sublime Text, Atom, Visual Studio Code, PyCharm,
vim/neovim, and emacs. (Okay those last two are only arguably modern but they
are extensible enough that they can be made to do most of the things described
here.) Some of these tools may also be available in more basic editing
environments like the Jupyter notebook, the JupyterLab text editor, and Spyder.

What you should expect your editor to do
----------------------------------------

It may take some configuration, but once set up any modern editor should be
able to do the following:

   - Highlight python syntax.
   - Flag syntax or style errors (line length, unused/undefined variables,
     dubious exception handling) visually as you edit.
   - Offer completions for any identifier (keyword, function, variable, et
     cetera) you start typing.
   - Reformat text into the ``black`` code style with a keypress.
   - Sort your imports into the standard arrangement with a keypress.
   - Jump to the definition of a function, class, or method with a keypress.
   - Obey ``.editorconfig`` settings.

A little Googling should reveal how to get all this working in your favorite
editor, but if you have some helpful links for a particular editor feel free to
add them to the documentation right here.

Command-line tools and automation
---------------------------------

PINT is developed with ``git`` and on GitHub_. Some operations are presented
graphically in the web interface, but in many cases you will want to do
something direct on your local machine. Having the right tools available and
configured should make this easy.


In your development virtualenv, install the development requirements::

   pip install -Ur requirements_dev.txt

Set up a few tools to make the git repository behave better. ``pre-commit``
runs various things, like check the text formatting, making sure you didn't
accidentally include some huge binary file, and so on, when you go to commit
something to git::

   pre-commit install

Configure git so ``git blame`` ignores large-scale reformatting commits in
favour of changes to the actual contents::

   git config blame.ignoreRevsFile .git-blame-ignore-revs

How To Build and Test From the Command Line
===========================================

To run the whole test suite (on all the cores of your machine)::

   pytest -n auto

To run tests on just one file::

   pytest tests/test_my_new_thing.py

To run just one test::

   pytest tests/test_my_new_thing.py::test_specific

To test everything but start with tests that failed last time, stopping when
something goes wrong (this is great when you're trying to fix that one bug; if
you haven't you'll get new error messages, if you have, it'll continue on to
run all the tests)::

   pytest --ff -x

To drop into the python debugger at the point where a test fails so you can
investigate_, for example go up and down the call history and inspect local
variables::

   pytest --pdb -x

The `python debugger`_ also allows you to step through your code, put in
breakpoints, and many other things. It can save a ton of time compared to
putting print statements in and rerunning your code, especially if the code
takes a while and you don't know exactly what you want to inspect.

To run the whole test suite in fresh installs on several python versions, and
also rebuilt the notebooks and documentation as well as compute combined code
coverage for all the versions::

   tox

To run tests on multiple python versions and build
the documentation in parallel::

   tox --parallel=auto

If this finds a problem in just one python environment that doesn't appear in
your development environment, you can run just the problem environment::

   tox -e py27

You can also run other things in the environments ``tox`` uses, including
interactive python sessions (though these will include only PINT's installation
requirements, so no IPython)::

   tox -e py27 -- pytest --ff --pdb -x
   tox -e py27 -- pytest tests/test_my_new_thing.py
   tox -e py27 -- python

To automatically run black on all of PINT's code::

   black src/ tests/

Under ``examples/`` there are a few Jupyter notebooks. These actually get
incorporated into the online documentation (you may have seen them). To avoid
headaches, we don't store these as notebooks on github but as special markdown
files. If you are using ``jupyter`` or ``jupyter-lab``, it should be smart
enough to synchronize these between the storage format and normal notebooks,
but if there is any confusion, try ``make notebooks``, which synchronizes the
two formats and runs all the notebooks to fill in the outputs. If something
goes wrong, try ``jupytext --sync``, which synchronizes the code between the
notebooks and the storage format but doesn't run the notebooks.

Coping with ``git``
-------------------

To import any changes that have been made to the PINT distribution::

   git fetch --all
   git checkout master
   git merge upstream/master
   git push

To switch between branches::

   git checkout a-branch
   git checkout another-branch
   git checkout master

These are very fast but they do change all the source code files to reflect
what they look like in the branch you're switching to. If you have them open in
editor windows your editor may give you surprised messages as the files change
under it.

To start a new branch for a thing::

   git checkout master
   git checkout -b a-thing

To send your changes to the current branch to your fork of the PINT
repository::

   git push

If this is the first time you've done this with a new branch ``git`` will
refuse because it doesn't exist in your fork on GitHub. It will print out a
command to create the branch on your GitHub. Just paste that. It will look
like::

   git push --set-upstream origin a-thing

If you now go to GitHub and poke around a bit, say on the Issues or Pull
Requests page, GitHub will have a button that says essentially "you just pushed
a new branch, do you want to make it into a pull request?" If your branch was
meant to go into PINT, this is what you want to do, so click that button.
GitHub will allow you to enter a more detailed description and then create a
Pull Request that can be seen on the main PINT pages. People can then comment
on the pull request ("PR") in general or specific lines of code you have
changed in particular.

If you are working on a pull request and the main PINT development has changed
in a way that conflicts with it (itHub will tell you on the pull request page),
you want to rebase_ your pull request. There are more details you can look up,
but in short, update master as above, then::

   git checkout a-thing
   git rebase master

This will attempt to take your branch, ``a-thing``, look at how it differs from
where you created it from, and then apply those same changes to the new
``master``. This will sometimes run into trouble, which you have to resolve
before you can continue normal work. Once you have finished the rebase, you
will need to push it to your GitHub. This is a little more complicated than
usual because you are changing not just the current state of the code but the
*history* that led to the current state of the code in your branch. This may
mess up comments that people have attached to particular lines of your pull
request, so pick a quiet moment to do this. You will need to tell ``git`` that
yes, you really mean to change the public history::

   git push -f

If you are digging through the source code and see something strange in a file,
and if you think "who thought *that* was a good idea?", you can ask ``git`` who
last modified each line in a file, and when::

   git blame src/pint/utils.py

To track and checkout another user's branch (pull request)::

   git remote add other-user-username https://github.com/other-user-username/pint.git
   git fetch other-user-username
   git checkout --track -b branch-name other-user-username/branch-name

If you make a mistake and get ``git`` into a strange or awkward state. Don't
panic, and try Googling the specific error message. ``git`` is quite thorough
about keeping history around, so you can probably undo whatever has happened,
especially if you have been pushing your changes to GitHub. If it helps, there
is `Dang it, git!`_ (there is a ruder version which may feel more appropriate
in the moment), or the `git choose-your-own-adventure` (which is extremely
useful as well as amusing).

Tagging and Releasing versions
------------------------------

This portion is only for developers with permission to modify the master NANOGrav repository!

Tagging
'''''''

The current version string is available as ``pint.__version__``

PINT uses MAJOR.MINOR.PATCH versioning inspired by, but not strictly following, Semantic Versioning.
PINT uses versioneer.py to make sure that ``pint.__version__`` is available in the code for version checking.
This constructs the version string from git using tags and commit hashes.

To create a new tagged version of PINT (assuming you are going from 0.5.0 to 0.5.1):

You can see what tags already exist like this::

   git tag --list

First make sure you are on the PINT master branch in the ``nanograv/PINT`` repository and your working copy is clean (``git status``), then::

   git push origin

Now wait 15 minutes and check that travis-ci says that the build is OK, before tagging! 
If needed, push any bug fixes.

Next, check the unreleased CHANGELOG (`CHANGELOG-unreleased.md`) and make sure all the
significant changes from PRs since the last release have been documented. Move these entries
to the released CHANGELOG (`CHANGELOG.md`), and change title of the newly moved entries 
from "Unreleased" to the version number you are about to tag and commit. **But don't yet push**.

When tagging, always use "annotated tags" by specifying ``-a``, so do these commands to tag and push::

   git tag -a 0.5.1 -m "PINT version 0.5.1"
   git push origin --tags

Releasing
'''''''''

To release, you need to have your PyPI API token in ``~/.pypirc``.
You must be on a clean, tagged, version of the nanograv/master branch. Then you can just::

   make release

This will build the distribution source and wheel packages and use ``twine`` to upload to PyPI.

Doing this will also trigger conda-forge_ to create a new PR for this release. Once this passes tests, 
it will need to be merged.


As a last step, go to the Releases_ tab on github and Draft a new release.


.. _GitHub: https://github.com/nanograv/PINT
.. _investigate: https://realpython.com/python-debugging-pdb/
.. _`python debugger`: https://docs.python.org/3/library/pdb.html
.. _rebase: https://git-scm.com/book/en/v2/Git-Branching-Rebasing
.. _`Dang it, git!`: https://dangitgit.com/
.. _`git choose-your-own-adventure`: http://sethrobertson.github.io/GitFixUm/fixup.html
.. _conda-forge: https://github.com/conda-forge/pint-pulsar-feedstock
.. _Releases: https://github.com/nanograv/PINT/releases
