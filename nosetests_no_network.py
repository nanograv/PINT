import astropy.utils.data
def needs_network(*args, **kwargs):
    raise ValueError("Needs network!")
astropy.utils.data.download_file = needs_network
import nose
nose.main()
