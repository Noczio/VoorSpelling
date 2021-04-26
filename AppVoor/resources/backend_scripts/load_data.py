import json
from abc import abstractmethod, ABC
from typing import Union, TypeVar, Generic

import pandas as pd

from resources.backend_scripts.is_data import DataEnsurer
from resources.backend_scripts.switcher import Switch

T = TypeVar('T')
DataFrame = pd.DataFrame


class DataLoader(ABC, Generic[T]):
    _data: T = None

    def __init__(self, file_path: str) -> None:
        # initialization when obj is created.
        self._file_path: str = file_path

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, value: T) -> None:
        self._data = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    @abstractmethod
    def get_file_transformed(self) -> T:
        pass


class JSONDataLoader(DataLoader[Union[list, dict]]):

    # abstract class method implementation
    def get_file_transformed(self) -> Union[list, dict]:
        # try to load file and set data, if error raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = json.load(f)
                self.data = temp
                return self.data
        except FileNotFoundError:
            raise FileNotFoundError("Path to JSON file was not found")
        except ValueError:
            raise ValueError("Data does not meet requirements to be considered a json file")
        except OSError:
            raise OSError("Invalid file. It needs a text extension")
        except Exception as e:
            raise Exception(str(e))


class CSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as ","
        separator = ","
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except FileNotFoundError:
            raise FileNotFoundError("Path to CSV file was not found")
        except TypeError:
            raise TypeError("Data does not meet sample or column requirements to train a model")
        except ValueError:
            raise ValueError("Data does not meet requirements to be considered a csv file")
        except OSError:
            raise OSError("Invalid file. It needs a text extension")
        except Exception as e:
            raise Exception(str(e))


class SCSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as ","
        separator = ";"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except FileNotFoundError:
            raise FileNotFoundError("Path to SCSV file was not found")
        except TypeError:
            raise TypeError("Data does not meet sample or column requirements to train a model")
        except ValueError:
            raise ValueError("Data does not meet requirements to be considered a scsv file")
        except OSError:
            raise OSError("Invalid file. It needs a text extension")
        except Exception as e:
            raise Exception(str(e))


class TSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as "\t"
        separator = "\t"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except FileNotFoundError:
            raise FileNotFoundError("Path to TSV file was not found")
        except TypeError:
            raise TypeError("Data does not meet sample or column requirements to train a model")
        except ValueError:
            raise ValueError("Data does not meet requirements to be considered a tsv file")
        except OSError:
            raise OSError("Invalid file. It needs a text extension")
        except Exception as e:
            raise Exception(str(e))


class LoaderPossibilities(Switch):

    @staticmethod
    def CSV() -> DataLoader:
        return CSVDataLoader("")

    @staticmethod
    def TSV() -> DataLoader:
        return TSVDataLoader("")

    @staticmethod
    def JSON() -> DataLoader:
        return JSONDataLoader("")

    @staticmethod
    def SCSV() -> DataLoader:
        return SCSVDataLoader("")


class LoaderCreator:
    __instance = None

    @staticmethod
    def get_instance() -> "LoaderCreator":
        """Static access method."""
        if LoaderCreator.__instance is None:
            LoaderCreator()
        return LoaderCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if LoaderCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LoaderCreator.__instance = self

    def create_loader(self, file_path: str, loader_type: str) -> DataLoader:
        # transform param to capital letters and then replace white spaces
        try:
            loader_name = loader_type.upper().replace(" ", "")
            loader = LoaderPossibilities.case(loader_name)
            loader.file_path = file_path
            return loader
        except():
            available_types = self.get_available_types()
            types_as_string = ", ".join(available_types)
            raise AttributeError(f"Parameter loader type value is wrong. "
                                 f"It should be any of the following: {types_as_string}")

    def get_available_types(self) -> tuple:
        available_types = [func for func in dir(LoaderPossibilities)
                           if callable(getattr(LoaderPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)
