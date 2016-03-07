import numpy as np

class TOASelect(object):
    def __init__(self, key, key_value):
        self.key = key
        self.key_value = key_value
        self.key_section = key + '_section'
        self.range_select = False
        if len(key_value) > 1:
            self.range_select = True

    def check_table_keys(self, toas):
        table_keys = toas.keys()
        if 'flags' in table_keys:
            flag_names = toas['flags'][0].keys()
        return table_keys, flag_names

    def get_key_section(self, toas):
        table_keys, flags = self.check_table_keys(toas)
        if self.key in flags:
            flag_value = []
            for ii, flags_dict in enumerate(toas['flags']):
                try:
                    flag_value.append(flags_dict[self.key])
                except: # TODO allow flags has empty element.
                    raise RuntimeError('TOA %d does not have flag %s.' % (ii, self.key))
            self.key_section = self.key + '_section'
            toas[self.key_section] = flag_value
        elif self.key.lower() in table_keys:
            self.key_section = self.key.lower()
            return
        else:
            raise ValueError("Key %s is not a flag or toas table key." % self.key)

    def get_toa_key_mask(self, toas):
        if self.key_section not in toas.keys():
            self.get_key_section(toas)
        if not self.range_select:
            group = toas.group_by(self.key_section)
            mask = group.groups.keys[self.key_section] == self.key_value[0]
            return group.groups[mask]['index']
        else:
            r1 = self.key_value[0]
            r2 = self.key_value[1]
            mask = np.logical_and(toas[self.key_section] >= r1,
                                 toas[self.key_section] <= r2)
            return toas[mask]['index']
