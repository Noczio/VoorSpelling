class Switch(object):

    @classmethod
    def case(cls, attr):
        if hasattr(cls, attr):
            out_attr = getattr(cls, str(attr))
            return out_attr()
        raise AttributeError
