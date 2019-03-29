# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.


from UserDict import UserDict


class ModelDict(object):
    """
    Creates a dict that one can access using dot notation

    usage:

        # create new model
        net = model()

        # use net.last to chain together tensor operations
        net.tensor_a = tf.constant(10)
        net.tensor_b = net.last * 100
        net.tensor_c = net.last * 1000

        # list all ops
        net.keys()

    """
    def __init__(self):
        super(ModelDict, self).__init__()
        self.data = {}
        self._order = []

    def __getattr__(self, attr):
        if attr in ("data", "last", "_order"):
            return self.__getattribute__(attr)

        return self.data[attr]

    def __setattr__(self, attr, value):
        if attr in ("data", "last", "_order"):
            super(ModelDict, self).__setattr__(attr, value)
            return

        self._order.append(attr)
        self.data[attr] = value

        # we also set self.last equal to
        # the most recent value for
        self.last = value

    def __setitem__(self, attr, value):
        self._order.append(attr)
        self.data[attr] = value
        self.last = value

    def __getitem__(self, attr):
        return self.data[attr]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return [(key, self.data[key]) for key in self._order if key in self.data]

    def iteritems(self):
        return ((key, self.data[key]) for key in self._order if key in self.data)

    def __repr__(self):
        import pprint
        out = u""
        for k,v in self.items():
            vstr = pprint.pformat(v)
            if (len(k) + len(vstr)) > 80:
                vstr = " \\\n" + vstr
            out += u"%r: %s\n" % (k, vstr)
        return out
