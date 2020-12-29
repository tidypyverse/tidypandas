import copy
import numpy as np
import pandas as pd

#from package_tidypandas.tidyGroupedDataFrame import tidyGroupedDataFrame
#from package_tidypandas.tidypredicates import *

def is_string_or_string_list(x):
    
    res = False
    if isinstance(x, str):
        res = True
    elif isinstance(x, list):
        if all([isinstance(i, str) for i in x]):
            res = True
    else:
        res = False
    
    return res
    
def enlist(x):
    if not isinstance(x, list):
        x = [x]
    
    return x


class tidyDataFrame:
    # init method
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
            assert flag # TODO: this requires to be more descriptive
        self.__data = copy.deepcopy(x)
    
    # print method
    def __repr__(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape))
        print('First few rows:')
        print(self.__data.head(10))
        return ''
    
    # pandas copy method
    def to_pandas(self):
        return copy.copy(self.__data)
    
    # get methods
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
    
    # groupby method
    def group_by(self, column_names):
        
        if isinstance(column_names, str):
            column_names = [column_names]
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        res = self.__data.groupby(column_names)
        return tidyGroupedDataFrame(res, check = False)
    
    # basic verbs  
    def select(self, column_names, include = True):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        if not include:
            column_names = self.__data.columns.difference(column_names)
            if len(column_names) == 0:
                raise Exception("Removing all columns is not allowed")
        
        res = self.__data.loc[:, column_names]
        
        return tidyDataFrame(res, check = False)
    
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
        
    def filter(self, query_string = None, mask = None):
   
        if query_string is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query_string is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        if query_string is not None and mask is None:
            res = self.__data.query(query_string)
        if query_string is None and mask is not None:
            res = self.__data.iloc[mask, :]
            
        return tidyDataFrame(res, check = False)
        
    def distinct(self, column_names = None, keep = 'first', retain_all_columns = False):
        if isinstance(column_names, str):
            column_names = [column_names]
        assert (column_names is None) or (isinstance(column_names, list))
        if column_names is not None:
            assert all(isinstance(x, str) for x in column_names)
            cols = self.get_colnames()
            assert all([x in cols for x in column_names])
        assert isinstance(retain_all_columns, bool)
        
        if column_names is None:
            res = self.__data.drop_duplicates(keep = keep, ignore_index = True)
        else:
            if retain_all_columns:
                res = self.__data.drop_duplicates(subset = column_names
                                                  , keep = keep
                                                  , ignore_index = True
                                                  )
            else:
                res = (self.__data
                           .loc[:, column_names]
                           .drop_duplicates(keep = keep, ignore_index = True)
                           )
        
        return tidyDataFrame(res, check = False)
    
    # basic extensions    
    def mutate_across(self, func, column_names = None, predicate = None, prefix = ""):

        assert callable(func)
        assert isinstance(prefix, str)
        
        if (column_names is not None) and (predicate is not None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")
        
        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")
        
        # use column_names
        if column_names is not None:
            assert isinstance(column_names, list)
            assert all([isinstance(acol, str) for acol in column_names])
        # use predicate to assign appropriate column_names
        else:
            mask = list(self.__data.apply(predicate, axis = 0))
            assert all([isinstance(x, bool) for x in mask])
            column_names = self.__data.columns[mask]
        
        # make a copy of the dataframe and apply mutate in order
        for acol in column_names:
            self.__data[prefix + acol] = fun(self.__data[acol])
            
        return tidyDataFrame(self, check = False)
        
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None):

        # assertions
        assert isinstance(y, (tidyDataFrame, tidyGroupedDataFrame))
        assert how in ['inner', 'outer', 'left', 'right', 'anti']
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        is_y_grouped = False
        if isinstance(y, tidyGroupedDataFrame):
            is_y_grouped = True
            assert on is not None # only on is supported when y is grouped
            groupvars = y.get_groupvars()
            y = y.ungroup().to_pandas()
        else:
            y = y.to_pandas()
            
        if on is None:
            assert on_x is not None and on_y is not None
            assert len(on_x) == len(on_y)
            assert is_string_or_string_list(on_x)
            assert is_string_or_string_list(on_y)
            on_x = enlist(on_x)
            on_y = enlist(on_y)
        else:
            assert on_x is None and on_y is None
            assert isinstance(on, (str, list))
            if isinstance(on, str):
                assert on in cn_x
                assert on in cn_y
                on = enlist(on)
            else:
                assert all([on in x for x in cn_x])
                assert all([on in x for x in cn_x])
                
        # merge call
        if how == 'anti':
            res = pd.merge(self.__data
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           , indicator = True
                           )
            res = res.loc[res._merge == 'left_only', :].drop(columns='_merge')
        else:    
            res = pd.merge(self.__data
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           )
                       
        # handle grouping
        if is_y_grouped:
            if all(i in list(res.columns) for i in groupvars):
                res = tidyGroupedDataFrame(res.groupby(groupvars), check = False)
            else:
                raise Exception('Merged output should have groupby columns names. Most likely they have got appended by _x or _y')
        else:
            res = tidyDataFrame(res, check = False)
            
        return res
        
    def join_inner(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'inner', on, on_x, on_y)
        
    def join_outer(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'outer', on, on_x, on_y)
        
    def join_left(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'left', on, on_x, on_y)
        
    def join_right(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'right', on, on_x, on_y)
        
    def join_anti(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'anti', on, on_x, on_y)    

