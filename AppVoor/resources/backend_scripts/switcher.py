from abc import ABC
from typing import Any


class Switch(ABC):

    @classmethod
    def case(cls, attr: str) -> Any:
        if hasattr(cls, attr):
            method = getattr(cls, str(attr))
            return method()
        raise AttributeError
