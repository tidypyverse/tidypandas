import copy
import numpy as np
import pandas as pd
import warnings

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
            if not flag:
                if not row_flag:
                    warnings.warn("Row index should not be a MultiIndex object.")
                if not col_flag:
                    warnings.warn("Column index should not be a MultiIndex object.")
                if not flag_no_index:
                    warnings.warn("Row index should not either RangeIndex or Int64Index. Perhaps, you might want to reset index with df.reset_index(drop = False)?")
                if not str_flag:
                    warnings.warn("Column index should be string column names.")
                if not unique_flag:
                    warnings.warn("Column names(index) should be unique.")
                
                # raise the error After informative warnings
                raise Exception("Cannot use the pandas dataframe due to above warnings.")
            
            print("Success! Created a tidy dataframe.")       
                    
        self.__data = copy.copy(x)
        return None
    
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
        
    def get_shape(self):
        return self.__data.shape
        
    def get_dim(self):
        return self.__data.shape
        
    def get_colnames(self):
        return list(self.__data.columns)
    
    # groupby method
    def group_by(self, column_names):
        
        assert is_string_or_string_list(column_names)
        column_names = enlist(column_names)
        assert len(column_names) > 0
        cols = self.get_colnames()
        assert all([x in cols for x in column_names])
        
        res = self.__data.groupby(column_names)
        return tidyGroupedDataFrame(res, check = False)
        
    # ungroup method
    # just a placeholder
    def ungroup(self):
        warnings.warn("Cannot ungroup an ungrouped tidy dataframe. Ungrouped dataframe as returned asis.")
        return self
    
    # basic verbs  
    def select(self, column_names = None, predicate = None, include = True):
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        
        if column_names is None:
            assert callable(predicate)
            col_bool_dict = dict(self.__data.apply(predicate))
            for akey in col_bool_dict:
                if not col_bool_dict[akey]:
                    del col_bool_dict[akey]
            column_names = list(col_bool_dict.keys())
            assert len(column_names) > 0
        else:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            assert len(column_names) > 0
            cols = self.get_colnames()
            assert all([x in cols for x in column_names])
        
        if not include:
            column_names = set(cols).difference(set(column_names))
            column_names = list(column_names)
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
        
        column_names = enlist(column_names)
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
        mutated = copy.copy(self.__data)
        
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
            self.__data[prefix + acol] = func(self.__data[acol])
            
        return tidyDataFrame(self, check = False)
        
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (tidyDataFrame, tidyGroupedDataFrame))
        assert how in ['inner', 'outer', 'left', 'right', 'anti']
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        if isinstance(y, tidyGroupedDataFrame):
            y = y.ungroup().to_pandas()
        else:
            y = y.to_pandas()
            
        if on is None:
            assert on_x is not None and on_y is not None
            assert is_string_or_string_list(on_x)
            assert is_string_or_string_list(on_y)
            on_x = enlist(on_x)
            on_y = enlist(on_y)
            assert len(on_x) == len(on_y)
            assert all([e in cn_x for e in on_x])
            assert all([e in cn_y for e in on_y])
        else: # on is provided
            assert on_x is None and on_y is None
            assert is_string_or_string_list(on)
            on = enlist(on)
            assert all([e in cn_x for e in on])
            assert all([e in cn_y for e in on])
                
        # merge call
        if how == 'anti':
            res = pd.merge(self.__data
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           , indicator = True
                           , suffixes = (None, suffix_y)
                           )
            res = res.loc[res._merge == 'left_only', :].drop(columns = '_merge')
        else:    
            res = pd.merge(self.__data
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           , suffixes = (None, suffix_y)
                           )
                           
        # remove the new 'on_y' columns
        if on is None:
            def appender(x):
                if x in cn_x:
                    res = x + suffix_y
                else:
                    res = x
                return res
            
            new_on_y = map(appender, on_y)
            res = res.drop(columns = new_on_y)
        
        # check for unique column names
        res_columns = list(res.columns)
        if len(set(res_columns)) != len(res_columns):
            raise Exception('Join should not result in ambiguous column names. Consider changing the value of "suffix_y" argument')
                
        return tidyDataFrame(res, check = False)
        
    def join_inner(self, y, on = None, on_x = None, on_y = None, suffix_y = '_y'):
        return self.join(y, 'inner', on, on_x, on_y, suffix_y)
        
    def join_outer(self, y, on = None, on_x = None, on_y = None, suffix_y = '_y'):
        return self.join(y, 'outer', on, on_x, on_y, suffix_y)
        
    def join_left(self, y, on = None, on_x = None, on_y = None, suffix_y = '_y'):
        return self.join(y, 'left', on, on_x, on_y, suffix_y)
        
    def join_right(self, y, on = None, on_x = None, on_y = None, suffix_y = '_y'):
        return self.join(y, 'right', on, on_x, on_y, suffix_y)
        
    def join_anti(self, y, on = None, on_x = None, on_y = None, suffix_y = '_y'):
        return self.join(y, 'anti', on, on_x, on_y, suffix_y)    
        
    # count
    def count(self, column_names = None, count_column_name = 'n', sort = 'descending'):
        
        assert (column_names is None) or is_string_or_string_list(column_names)
        if column_names is not None:
            column_names = enlist(column_names)
        assert isinstance(count_column_name, str)
        assert count_column_name not in self.get_colnames()
        assert isinstance(sort, str)
        assert sort in ['asending', 'descending', 'natural']
        
        if column_names is not None:
            res = (self.__data
                       .groupby(column_names)
                       .size()
                       .reset_index()
                       .rename(columns = {0: count_column_name})
                       )
            asc = True
            if sort == 'descending':
                asc = False
            
            if sort != 'natural':
                #print('went here')
                res = res.sort_values(by = count_column_name
                                      , axis         = 0
                                      , ascending    = asc
                                      , inplace      = False
                                      , kind         = 'quicksort'
                                      , na_position  = 'first'
                                      , ignore_index = True
                                      )
        else:
            res = pd.DataFrame({count_column_name : self.get_nrow()}, index = [0])
            
        return tidyDataFrame(res, check = False)
