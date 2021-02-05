from abc import ABC
from typing import Any


class Switch(ABC):

    @classmethod
    def case(cls, attr: str) -> Any:
        if hasattr(cls, attr):
            out_attr = getattr(cls, str(attr))
            return out_attr()
        raise AttributeError
