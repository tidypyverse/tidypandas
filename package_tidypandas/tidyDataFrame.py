import copy
import numpy as np

class tidyDataFrame:
    def __init__(self, x):
        self.__data = copy.deepcopy(x)
    
    def __repr__(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape))
        print('First few rows:')
        print(self.__data.head(10))
        return ''
    
    def to_pandas(self):
        return self.__data
    
    def get_info(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape))
        print('\n')
        return self.__data.info()
        
    def get_nrow(self):
        return self.__data.shape[0]
    
    def get_ncol(self):
        return self.__data.shape[1]
        
    def get_colnames(self):
        return list(self.__data.columns)
        
    def select(self, column_names, include = True):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        if not include:
            column_names = self.__data.columns.difference(column_names)
            if len(column_names) == 0:
                raise "Removing all columns is not allowed"
        
        res = self.__data.loc[:, column_names]
        
        return tidyDataFrame(res)
    
    def group_by(self, column_names):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        res = self.__data.groupby(column_names)
        return tidyGroupedDataFrame(res)
    
    def slice(self, row_numbers):
        
        minval = np.min(row_numbers)
        maxval = np.max(row_numbers)
        assert minval >= 0 and maxval <= self.get_nrow()
        
        res = self.__data.take(row_numbers).reset_index(drop = True)
        
        return tidyDataFrame(res)
