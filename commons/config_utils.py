# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.


from collections import OrderedDict

class Config(OrderedDict):
    def set(self, name, default=None):
        OrderedDict.__setitem__(self, name, default)

    def __getitem__(self, key):
        if key not in self:
            print("WARN: Unspecified config value '%s', using None as default" % key)

        return OrderedDict.get(self, key)

    def __repr__(self):
        import pprint
        out = u""
        for k,v in self.items():
            vstr = pprint.pformat(v)
            if (len(k) + len(vstr)) > 80:
                vstr = " \\\n" + vstr
            out += u"%r: %s\n" % (k, vstr)
        return out
