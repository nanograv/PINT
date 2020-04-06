from setuptools import setup

import versioneer

# If you are surprised by how empty this file is, that is because almost all
# of the information that was here has been moved to the file setup.cfg

setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
