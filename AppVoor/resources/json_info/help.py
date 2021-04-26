from typing import Any

from resources.backend_scripts.is_data import DataEnsurer
from resources.json_info.json_to_data import JSONMessage


class HelpMessage(JSONMessage):

    def __init__(self, file_path: str, data_type: Any = dict) -> None:
        super().__init__(file_path, data_type)

    # abstract class method implementation
    def __getitem__(self, key: str) -> tuple:
        if DataEnsurer.validate_py_data(key, str):
            if key in self.data.keys():
                actual_data = self.data[key]
                title = actual_data["title"]
                body = actual_data["body"]
                example = actual_data["example"]
                url = actual_data["url"]
                return title, body, example, url
            # key does not exist. Raise KeyError
            raise KeyError("Key does not exist")
        # key is not string. Raise TypeError
        raise TypeError("Key is not string")
