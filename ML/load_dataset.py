import pandas as pd
import pathlib
from abc import ABC
from parse_file import DF_parse_ensurer

class ABC_data_loader(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]
    """
    def __init__(self,full_path:str):
        """[summary]

        Args:
            full_path (str): [description]
        """
        self._full_path =  full_path
        self._df_parser = DF_parse_ensurer() # uses Iparse_ensurer

    def __str__(self):
        return self._full_path
    def get_file_as_dataframe(self,separator:str):
        pass
    
class CVS_data_type_loader(ABC_data_loader):
    """[summary]

    Args:
        ABC_data_loader ([type]): [description]
    """
    def get_file_as_dataframe(self,separator=","):
        """[summary]

        Args:
            separator (str, optional): [description]. Defaults to ",".

        Raises:
            TypeError: [description]
            FileNotFoundError: [description]

        Returns:
            [type]: [description]
        """
        file = pathlib.Path(self._full_path)
        if file.exists():
            df = pd.read_csv(self._full_path,sep=separator)
            if (self._df_parser.is_data_correctly_parsed(df)):
                return df
            raise TypeError            
        raise FileNotFoundError
        
class TSV_data_type_loader(ABC_data_loader):
    """[summary]

    Args:
        ABC_data_loader ([type]): [description]
    """
    def get_file_as_dataframe(self,separator="\t"):
        """[summary]

        Args:
            separator (str, optional): [description]. Defaults to "\t".

        Raises:
            TypeError: [description]
            FileNotFoundError: [description]

        Returns:
            [type]: [description]
        """
        file = pathlib.Path(self._full_path)
        if file.exists ():
            df = pd.read_csv(self._full_path,sep=separator)
            if (self._df_parser.is_data_correctly_parsed(df)):
                return df
            raise TypeError
        raise FileNotFoundError
        

