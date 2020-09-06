import pandas as pd
from abc import ABC,abstractmethod 

class Abs_data_load(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]
    """
    def __init__(self,full_path:str):
        self.full_path =  full_path
    def __str__(self):
        return self.full_path
    def file_as_dataset(self,separator:str):
        pass
    
class CVS_data_type(Abs_data_load):
    def file_as_dataset(self,separator=","):
        """[summary]

        Args:
            separator (str, optional): [description]. Defaults to ",".

        Returns:
            [type]: [description]
        """
        return pd.read_csv(self.full_path,sep=separator)

class TSV_data_type(Abs_data_load):
    def file_as_dataset(self,separator="\t"):
        """[summary]

        Args:
            separator (str, optional): [description]. Defaults to "\t".

        Returns:
            [type]: [description]
        """
        return pd.read_csv(self.full_path,sep=separator)

