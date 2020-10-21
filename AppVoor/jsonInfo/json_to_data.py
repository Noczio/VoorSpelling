from abc import ABC, abstractmethod


class ABCJson(ABC):

    def __init__(self, file_path: str) -> None:
        # initialize _data as None
        self._data = None
        self._file_path = file_path

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
