import pandas as pd
from interface import implements, Interface


class IParse_ensurer(Interface):
    """[summary]

    Args:
        Interface ([type]): [description] 
    """

    def is_data_correctly_parsed(self, data) -> bool:
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            bool: [description]
        """
        pass


class DF_parse_ensurer(implements(IParse_ensurer)):
    """[summary]

    Args:
        implements ([type]): [description]
    """

    def __init__(self):
        self._min_data = 100  # this value should be changed if requirements say so

    def is_data_correctly_parsed(self, data) -> bool:
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            bool: [description]
        """
        if (isinstance(data, pd.DataFrame)):
            x_shape, y_shape = data.shape
            if x_shape <= self._min_data or y_shape == 1:
                return False
            return True
        return False
