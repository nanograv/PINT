from six.moves.urllib.parse import urlsplit
import astropy.utils.data
_download_file = astropy.utils.data.download_file
def needs_network(url, *args, **kwargs):
    if urlsplit(url).netloc in [
            "data.nanograv.org",
            "ssd.jpl.nasa.gov",
            #"naif.jpl.nasa.gov", # This one hasn't blocked us
        ]:
        raise ValueError("Needs network!")
    else:
        return _download_file(url, *args, **kwargs)
astropy.utils.data.download_file = needs_network
import nose
nose.main()
