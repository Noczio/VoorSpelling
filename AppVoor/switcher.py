from typing import Any


class Switch(object):

    @classmethod
    def case(cls, attr: str) -> Any:
        if hasattr(cls, attr):
            out_attr = getattr(cls, str(attr))
            return out_attr()
        raise AttributeError
