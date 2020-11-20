import copy
import numpy as np
import pandas as pd

class tidyDataFrame:
    def __init__(self, x, check = True):
        
        if check:
            assert isinstance(x, pd.DataFrame)
            row_flag = not isinstance(x.index, pd.MultiIndex)
            columns  = list(x.columns)
            col_flag = not isinstance(columns, pd.MultiIndex)
            # check if row index is rangeIndex
            flag_no_index = isinstance(x.index, (pd.RangeIndex, pd.Int64Index))
            # check if all column names are strings
            str_flag = all([isinstance(y, str) for y in columns])
            # check if column names are unique
            if len(set(columns)) == len(columns):
                unique_flag = True
            else:
                unique_flag = False
            
            flag = all([row_flag, col_flag, flag_no_index, str_flag, unique_flag])
            assert flag # this requires to be more descriptive
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
        
        return tidyDataFrame(res, check = False)
    
    def group_by(self, column_names):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        res = self.__data.groupby(column_names)
        return tidyGroupedDataFrame(res, check = False)
    
    def slice(self, row_numbers):
        
        minval = np.min(row_numbers)
        maxval = np.max(row_numbers)
        assert minval >= 0 and maxval <= self.get_nrow()
        
        res = self.__data.take(row_numbers).reset_index(drop = True)
        
        return tidyDataFrame(res, check = False)
        
    def arrange(self, column_names, ascending = False, na_position = 'last'):
        
        if not isinstance(column_names, list):
            column_names = [column_names]
        assert len(column_names) > 0
        cn = self.get_colnames()
        assert all([x in cn for x in column_names])
        if not isinstance(ascending, list):
            isinstance(ascending, bool)
        else:
            assert all([isinstance(x, bool) for x in ascending])
            assert len(ascending) == len(column_names)
        
        res = self.__data.sort_values(by = column_names
                                        , axis         = 0
                                        , ascending    = ascending
                                        , inplace      = False
                                        , kind         = 'quicksort'
                                        , na_position  = na_position
                                        , ignore_index = True
                                        )
        return tidyDataFrame(res, check = False)
    
    def mutate(self, dictionary):
        '''
        {"hp": [lambda x, y: x - y.mean(), ['a', 'b']]
           , "new" : lambda x: x.hp - x.mp.mean() + x.shape[1]
           , "existing" : [lambda x: x + 1]
           }
        TODO:
            1. assign multiple columns at once case
            2. grouped version
        '''
        mutated = copy.deepcopy(self.__data)
        
        for akey in dictionary:
            
            # lambda function case
            if callable(dictionary[akey]):
                # assigning to single column
                if isinstance(akey, str):
                    mutated[akey] = dictionary[akey](mutated)
            
            # simple function case
            if isinstance(dictionary[akey], (list, tuple)):
                cn = set(mutated.columns)
                
                # case 1: only simple function
                if len(dictionary[akey]) == 1:
                    assert callable(dictionary[akey][0])
                    
                    # assign to a single column
                    # column should pre-exist
                    assert set([akey]).issubset(cn)
                    mutated[akey] = dictionary[akey][0](mutated[akey])
                
                # case2: function with required columns
                elif len(dictionary[akey]) == 2:
                    assert callable(dictionary[akey][0])
                    assert isinstance(dictionary[akey][1], (list, tuple, str))
                    if not isinstance(dictionary[akey][1], (list, tuple)):
                        colnames_to_use = [dictionary[akey][1]]
                    else:
                        colnames_to_use = dictionary[akey][1]
                    assert set(colnames_to_use).issubset(cn)
                    
                    input_list = [mutated[colname] for colname in colnames_to_use]
                    mutated[akey] = dictionary[akey][0](*input_list)
            
        return tidyDataFrame(mutated, check = False)
