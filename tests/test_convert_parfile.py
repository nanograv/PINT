import pytest
from pint.scripts import convert_parfile
from pint.config import examplefile
from pint.models import get_model, TimingModel
import os


@pytest.mark.parametrize("format", ["pint", "tempo", "tempo2"])
def test_convert_parfile(format):
    input_par = examplefile("NGC6440E.par.good")
    output_par = "NGC6440E.converted.par"

    argv = f"-f {format} -o {output_par} {input_par}".split()

    convert_parfile.main(argv=argv)

    assert os.path.isfile(output_par)
    assert isinstance(get_model(output_par), TimingModel)
