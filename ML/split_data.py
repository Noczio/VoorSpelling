from abc import ABC,abstractmethod 
import pandas as pd

class ABC_data_splitter(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]
    """
    def __init__(self,df_train=None,df_test=None):
        """[summary]

        Args:
            df_train ([type], optional): [description]. Defaults to None.
            df_test ([type], optional): [description]. Defaults to None.
        """
        self.df_train = df_train
        self.df_test = df_test
    def split_data(self):
        pass

class Multi_data_splitter(ABC_data_splitter):
    def split_data(self):
        if (isinstance(self.df_train, pd.DataFrame) and isinstance(self.df_test, pd.DataFrame)):
            y_train=self.df_train[self.df_train.columns[-1]]
            x_train=self.df_train.drop([self.df_train.columns[-1]],axis=1)
            y_test=self.df_test[self.df_test.columns[-1]]
            x_test=self.df_test.drop([self.df_test.columns[-1]],axis=1)
            return x_train,y_train,x_test,y_test
        else: raise TypeError

class Simple_data_splitter(ABC_data_splitter):
    def split_data(self):
        if isinstance(self.df_train, pd.DataFrame):
            y_train=self.df_train[self.df_train.columns[-1]]
            x_train=self.df_train.drop([self.df_train.columns[-1]],axis=1)
            return x_train,y_train
        else: raise TypeError



          


