from __future__ import absolute_import, print_function, division
import numpy as np
import astropy.units as u
from .timing_model import DelayComponent, MissingParameter
from . import parameter as p


class Wave(DelayComponent):
    """This class provides glitches."""
    register = True
    def __init__(self):
        super(Wave, self).__init__()

        self.add_param(p.floatParameter(name="WAVE_OM",
                       description="Base frequency of wave solution",
                       units='1/d'))
        self.add_param(p.prefixParameter(name="WAVE1", units='s',
                                         description="Wave components",
                                         type_match='pair', long_double=True,
                                         parameter_type='pair'))
        self.add_param(p.MJDParameter(name="WAVEEPOCH",
                       description="Reference epoch for wave solution",
                       time_scale='tdb'))
        self.delay_funcs_component += [self.wave_delay,]
        self.category = 'wave'

    def setup(self):
        super(Wave, self).setup()
        if self.WAVEEPOCH.quantity is None:
            if self.PEPOCH.quantity is None:
                raise MissingParameter("Wave", "WAVEEPOCH",
                                       "WAVEEPOCH or PEPOCH are required if "
                                       "WAVE_OM is set.")
            else:
                self.WAVEEPOCH = self.PEPOCH

        wave_terms = list(self.get_prefix_mapping_component('WAVE').keys())
        wave_terms.sort()
        wave_in_order = list(range(1, max(wave_terms)+1))
        if not wave_terms == wave_in_order:
            diff = list(set(wave_in_order) - set(wave_terms))
            raise MissingParameter("Wave", "WAVE%d"%diff[0])

        self.num_wave_terms = len(wave_terms)

    def print_par(self, ):
        result = ''
        wave_terms = ["WAVE%d" % ii for ii in
                      range(1, self.num_wave_terms + 1)]

        result += self.WAVEEPOCH.as_parfile_line()
        result += self.WAVE_OM.as_parfile_line()
        for ft in wave_terms:
            par = getattr(self, ft)
            result += par.as_parfile_line()

        return result

    def wave_delay(self, toas, acc_delay=None):
        delays = 0
        wave_names = ["WAVE%d" % ii for ii in
                      range(1, self.num_wave_terms + 1)]
        wave_terms = [getattr(self, name) for name in wave_names]
        wave_om = self.WAVE_OM.quantity
        time = self.barycentric_time = toas['tdbld'] * u.day
        for k, wave_term in enumerate(wave_terms):
            wave_a, wave_b = wave_term.quantity
            k = k + 1
            delay_term = wave_a * np.sin(k * wave_om * time) + \
                         wave_b * np.cos(k * wave_om * time)
            delays += delay_term

        return delays
