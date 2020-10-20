from abc import ABC, abstractmethod


class ABCJson(ABC):

    def __init__(self) -> None:
        # initialize _data as None
        self._data = None

    def __len__(self) -> int:
        return len(self._data)

    def get_file_data(self) -> tuple:
        return tuple(self._data)

    @abstractmethod
    def _load_file(self) -> None:
        pass

    @abstractmethod
    def get_data_by_index(self, index: int) -> tuple:
        pass
