from typing import Any

from is_data import DataEnsurer
from jsonInfo.json_to_data import JSONMessage


class CVMetrics(JSONMessage):

    # abstract class method implementation
    def __getitem__(self, key: int) -> str:
        if DataEnsurer.validate_py_data(key, int):
            # make sure index is not out of boundaries
            if (key < len(self.data)) and (key >= -len(self.data)):
                # initialize local var author and quote
                metric = self.data[key]
                return metric
            # index is out of boundaries. Raise IndexError
            raise IndexError
        raise TypeError
