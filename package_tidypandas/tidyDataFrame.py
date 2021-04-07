import copy
import numpy as np
import pandas as pd
import warnings
import re
from functools import reduce
from collections_extended import setlist

# from package_tidypandas.TidyGroupedDataFrame import TidyGroupedDataFrame
# from package_tidypandas.tidypredicates import *

  
class TidyDataFrame:
    # init method
    def __init__(self, x, check = True):
        
        if check:
            assert isinstance(x, pd.DataFrame)
            row_flag = not isinstance(x.index, pd.MultiIndex)
            columns  = list(x.columns)
            col_flag = not isinstance(columns, pd.MultiIndex)
            # check if row index is rangeIndex
            flag_no_index = False
            if isinstance(x.index, pd.RangeIndex):
                if x.index.start == 0 and x.index.step == 1:
                    flag_no_index = True           
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
                    warnings.warn("Row index is not a RangeIndex with start = 0 and step = 1.")
                if not str_flag:
                    warnings.warn("Column index should be string column names.")
                if not unique_flag:
                    warnings.warn("Column names(index) should be unique.")
                
                # raise the error After informative warnings
                raise Exception(("Input pandas dataframe is not tidy."
                                 " See to above warnings."
                                 " Try the 'tidy' function."
                                 " ex: tidy(untidy pandas dataframe) --> tidy pandas dataframe."
                                ))
                               
        self.__data = copy.copy(x)
        return None
    
    # repr method
    def __repr__(self):
        header_line   = '-- Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape)
        few_rows_line = '-- First few rows:'
        pandas_str    = self.__data.head(10).__str__()
    
        tidy_string = (header_line + 
                       '\n' +
                       few_rows_line + 
                       '\n' +
                       pandas_str
                       )
        
        return tidy_string
    
    # pandas copy method
    def to_pandas(self):
        return copy.copy(self.__data)
    
    # series copy method
    def to_series(self, column_name):
        
        assert isinstance(column_name, str)
        assert column_name in self.get_colnames()
        
        return self.select(column_name).to_pandas().loc[:, column_name]
    
    # pipe method
    def pipe(self, func):
        return func(self)
    
    # pipe pandas method
    def pipe_pandas(self, func, as_tidy = True):
        res = func(self.__data)
        if isinstance(res, (pd.DataFrame
                            , pd.core.groupby.DataFrameGroupBy
                            )
                      ):
            res = tidy(res)
            if as_tidy:
                if isinstance(res, pd.DataFrame):
                    res = TidyDataFrame(res, check = False)
                else:
                    res = TidyGroupedDataFrame(res, check = False)
        return res
    
    # alias for pipe_pandas
    pipe2 = pipe_pandas    
    
    # get methods
    def get_info(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape))
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
        return TidyGroupedDataFrame(res, check = False)
    
    # alias for group_by
    groupby = group_by
    
    # ungroup method
    # just a placeholder
    def ungroup(self):
        return self
    
    # basic verbs  
    def select(self, column_names = None, predicate = None, include = True):
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        
        if column_names is None:
            assert callable(predicate)
            col_bool_list = list(self.__data.apply(predicate, axis = 0))
            column_names = list(np.array(self.get_colnames())[col_bool_list])
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
        
        return TidyDataFrame(res, check = False)
    
    def relocate(self, column_names, before = None, after = None):
        
        assert is_string_or_string_list(column_names)
        column_names = enlist(column_names)
        assert len(set(column_names)) == len(column_names) # assert if column_names are unique
        current_colnames = self.get_colnames()
         
        assert (before is None) or (after is None) # at least one of them is None
        if before is None:
            if after is not None:
                assert isinstance(after, str)
                assert after in current_colnames
                assert not (after in column_names)
        
        if after is None:
            if before is not None:
                assert isinstance(before, str)
                assert before in current_colnames
                assert not (before in column_names)
        
        cc_setlist       = setlist(current_colnames)
        cc_trunc_setlist = cc_setlist.difference(column_names)
            
        # case 1: relocate to start when both before and after are None
        if (before is None) and (after is None):
            new_colnames = copy.copy(column_names)
            new_colnames.extend(list(cc_trunc_setlist))
        elif (before is not None):
            # case 2: before is not None
            pos_before   = int(np.where([x == before for x in cc_trunc_setlist])[0])
            cc_left      = list(cc_trunc_setlist[ :pos_before])
            cc_right     = list(cc_trunc_setlist[pos_before: ])
            new_colnames = copy.copy(cc_left)
            new_colnames.extend(column_names)
            new_colnames.extend(cc_right)
        else:
            # case 3: after is not None
            pos_after    = int(np.where([x == after for x in cc_trunc_setlist])[0])      
            cc_left      = list(cc_trunc_setlist[ :(pos_after + 1)])
            cc_right     = list(cc_trunc_setlist[(pos_after + 1): ])
            new_colnames = copy.copy(cc_left)
            new_colnames.extend(column_names)
            new_colnames.extend(cc_right)
      
        res = self.__data.loc[:, new_colnames]
        return TidyDataFrame(res, check = False)
    

    def rename(self, old_new_dict):
        col_names = self.get_colnames()
        assert isinstance(old_new_dict, dict)
        assert set(col_names).issuperset(old_new_dict.keys()) # old names should be there
        assert is_unique_list(list(old_new_dict.values())) # new names should be unique
        
        # new names should not intersect with 'remaining' names
        remaining = set(col_names).difference(old_new_dict.keys())
        assert len(remaining.intersection(old_new_dict.values())) == 0
        
        res = self.__data.rename(columns = old_new_dict)
        return TidyDataFrame(res, check = False)
    
    def slice(self, row_numbers):
        
        minval = np.min(row_numbers)
        maxval = np.max(row_numbers)
        assert minval >= 0 and maxval <= self.get_nrow()
        
        res = self.__data.take(row_numbers).reset_index(drop = True)
        
        return TidyDataFrame(res, check = False)
        
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
        return TidyDataFrame(res, check = False)
        
    def filter(self, query_string = None, mask = None):
   
        if query_string is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query_string is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        if query_string is not None and mask is None:
            res = self.__data.query(query_string)
        if query_string is None and mask is not None:
            res = self.__data.iloc[mask, :]
            
        return TidyDataFrame(res, check = False)
        
    def distinct(self, column_names = None, keep = 'first', retain_all_columns = False):
        
        if column_names is not None:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            cols = self.get_colnames()
            assert set(column_names).issubset(cols)
        else:
            column_names = self.get_colnames()
        assert isinstance(retain_all_columns, bool)
        
        res = (self.__data
                   .drop_duplicates(subset = column_names
                                    , keep = keep
                                    , ignore_index = True
                                    )
                   )
        
        if not retain_all_columns:
            res = res.loc[:, column_names]
        
        return TidyDataFrame(res, check = False)

    def mutate(self, dictionary=None, func=None, column_names = None, predicate = None, prefix = ""):
        if dictionary is None and func is None:
            raise Exception("Either dictionary or func with predicate/column_names should be provided.")

        if dictionary is not None:
            return self._mutate(dictionary)
        else:
            return self._mutate_across(func, column_names=column_names, predicate=predicate, prefix=prefix)

    def _mutate(self, dictionary):
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

        return TidyDataFrame(mutated, check=False)

    # basic extensions    
    def _mutate_across(self, func, column_names = None, predicate = None, prefix = ""):

        assert callable(func)
        assert isinstance(prefix, str)

        mutated = copy.deepcopy(self.__data)
        
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
            mutated[prefix + acol] = func(mutated[acol])
            
        return TidyDataFrame(mutated, check = False)
    
    def summarise(self, dictionary=None, func=None, column_names=None, predicate=None, prefix = ""):
        """

        Parameters
        ----------
        dictionary: dictionary
            map of summarised column names to summarised functions
        func: callable
            aggregate function
        column_names: list
            list of column names(string)
        predicate: callable
            function to select columns, exactly one among column_names and predicate must be specified
        prefix: string
            prefix to add to column names

        Returns
        -------
        a TidyDataFrame

        Note
        -------
        Either dictionary or func with predicate/column names should be provided.

        Examples
        -------
        type1:
        dictionary = {"cross_entropy": [lambda p, q: sum(p*log(q)), ['a', 'b']]}
        dictionary = {"sepal_length": [lambda x: mean(x)]}
        type2:
        dictionary = {"cross_entropy": lambda x: (x.a*log(x.b)).sum()}
        type3:
        dictionary = {"sepal_length": "mean"}

        """
        if dictionary is None and func is None:
            raise Exception("Either dictionary or func with predicate/column_names should be provided.")

        if dictionary is not None:
            return self._summarise(dictionary)
        else:
            return self._summarise_across(func, column_names=column_names, predicate=predicate, prefix=prefix)

    def _summarise(self, dictionary):
        """

        Parameters
        ----------
        dictionary: dictionary
            map of summarised column names to summarised functions

        Returns
        -------
        a TidyDataFrame

        Examples
        -------
        type1:
        dictionary = {"cross_entropy": [lambda p, q: sum(p*log(q)), ['a', 'b']]}
        dictionary = {"sepal_length": [lambda x: mean(x)]}
        type2:
        dictionary = {"cross_entropy": lambda x: (x.a*log(x.b)).sum()}
        type3:
        dictionary = {"sepal_length": "mean"}

        """
        ## 1. will summarised columns earlier in the dictionary be available for later aggregations? NO

        # Type checks
        assert isinstance(dictionary, dict)

        cn = self.get_colnames()

        keys = dictionary.keys()
        values = dictionary.values()

        assert all([isinstance(akey, str) for akey in keys])
        assert all([isinstance(avalue, (list, tuple)) or callable(avalue) for avalue in values])

        # dict: akey -> summarised series
        res = dict()

        for akey in keys:
            avalue = dictionary[akey]
            if isinstance(avalue, (list, tuple)):
                if len(avalue) == 1:
                    func = avalue[0]
                    func_args = [akey]
                elif len(avalue) == 2:
                    func = avalue[0]
                    func_args = avalue[1]
                    assert isinstance(func_args, (list, tuple, str))
                    if isinstance(func_args, str):
                        func_args = [func_args]
                    else:
                        func_args = list(func_args) ## explicitly converts tuples to list
                        assert all(isinstance(acol_name, str) for acol_name in func_args)
                else:
                    raise ValueError("values of type list in the dictionary should be of len 1 or 2")

                assert callable(func)
                assert set(func_args).issubset(cn)

                ## summarise for type 1
                input_cols = map(lambda x: self.__data[x], func_args)
                res.update({akey: pd.Series(func(*input_cols))})

            if callable(avalue):
                ## summarise for type 2
                res.update({akey: pd.Series(self.__data.pipe(avalue))})

            ## TODO: to support avalue to be a string name for popular aggregate functions.

        list_summarised = list(res.values())

        assert all([a_summarised.shape == list_summarised[0].shape for a_summarised in list_summarised[1:]]), \
            "all summarised series don't have same shape"

        return TidyDataFrame(pd.DataFrame(res), check=False)

    def _summarise_across(self, func, column_names=None, predicate=None, prefix = ""):
        """

        Parameters
        ----------
        func: callable
            aggregate function
        column_names: list
            list of column names(string)
        predicate: callable
            function to select columns, exactly one among column_names and predicate must be specified
        prefix: string
            prefix to add to column names

        Returns
        -------
        a TidyDataFrame
        """

        assert callable(func)
        assert isinstance(prefix, str)

        if (column_names is not None) and (predicate is not None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")

        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")

        cn = self.get_colnames()

        # use column_names
        if column_names is not None:
            assert isinstance(column_names, list)
            assert all([isinstance(acol, str) for acol in column_names])
        # use predicate to assign appropriate column_names
        else:
            mask = list(self.__data.apply(predicate, axis=0))
            column_names = list(np.array(cn)[mask])

        # dict: akey -> summarised series
        res = dict()

        for acol in column_names:
            res.update({prefix+acol: pd.Series(func(self.__data[acol]))})

        list_summarised = list(res.values())

        assert all([a_summarised.shape == list_summarised[0].shape for a_summarised in list_summarised[1:]]), \
            "all summarised series don't have same shape"

        return TidyDataFrame(pd.DataFrame(res), check=False)

    
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (TidyDataFrame, TidyGroupedDataFrame))
        assert isinstance(how, str)
        assert how in ['inner', 'outer', 'left', 'right', 'anti']
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        y = y.ungroup().to_pandas()
            
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
                           , how = "left"
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
                
        return TidyDataFrame(res, check = False)
        
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
    
    # binding functions
    def cbind(self, y):
        # number of rows should match
        assert self.get_nrow() == y.get_nrow()
        # column names should differ
        assert len(set(self.get_colnames()).intersection(y.get_colnames())) == 0
        
        res = pd.concat([self.__data, y.ungroup().to_pandas()]
                        , axis = 1
                        , ignore_index = False # not to loose column names
                        )
        return TidyDataFrame(res, check = False)
    
    def rbind(self, y):
        res = pd.concat([self.__data, y.ungroup().to_pandas()]
                        , axis = 0
                        , ignore_index = True # loose row indexes
                        )
        return TidyDataFrame(res, check = False)
    
    # count
    def count(self, column_names = None, count_column_name = 'n', sort_order = 'descending'):
        
        assert (column_names is None) or is_string_or_string_list(column_names)
        if column_names is not None:
            column_names = enlist(column_names)
        assert isinstance(count_column_name, str)
        assert count_column_name not in self.get_colnames()
        assert isinstance(sort_order, str)
        assert sort_order in ['ascending', 'descending', 'natural']
        
        if column_names is not None:
            res = (self.__data
                       .groupby(column_names)
                       .size()
                       .reset_index()
                       .rename(columns = {0: count_column_name})
                       )
            asc = True
            if sort_order == 'descending':
                asc = False
            
            if sort_order != 'natural':
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
            
        return TidyDataFrame(res, check = False)

    def add_count(self
                  , column_names = None
                  , count_column_name = 'n'
                  , sort_order = 'natural'
                  ):
        count_frame = self.count(column_names, count_column_name, sort_order)
        if column_names is None:
            res = self.mutate({count_column_name : lambda x: count_frame.to_pandas().iloc[0,0]})
        else:
            res = self.join_inner(count_frame, on = column_names)
        
        return res
    
    # pivot methods
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = "mean"
                    , id_cols = None
                    , drop_na = True
                    , retain_levels = False
                    , sep = "__"
                    ):
        
        cn = self.get_colnames()
        
        assert is_string_or_string_list(names_from)
        names_from = enlist(names_from)
        assert set(names_from).issubset(cn)
        
        assert is_string_or_string_list(values_from)
        values_from = enlist(values_from)
        assert set(values_from).issubset(cn)
        assert len(set(values_from).intersection(names_from)) == 0
        set_union_names_values_from = set(values_from).union(names_from)
        
        if id_cols is None:
            id_cols = list(set(cn).difference(set_union_names_values_from))
            if len(id_cols) == 0:
                raise Exception("'id_cols' is turning out to be empty. Choose the 'names_from' and 'values_from' appropriately or specify 'id_cols' explicitly.")
            else:
                print("'id_cols' chosen: " + str(id_cols))
        else:
            assert is_string_or_string_list(id_cols)
            id_cols = enlist(id_cols)
            assert len(set(id_cols).intersection(set_union_names_values_from)) == 0
        
        if values_fill is not None:
            if isinstance(values_fill, dict):
                keys_fill = set(values_fill.keys())
                assert set(values_from) == keys_fill
            else:
                assert not isinstance(values_fill, list)          
        if values_fn != "mean":
            if isinstance(values_fn, dict):
                keys_fn = set(values_fn.keys())
                assert set(values_from) == keys_fn
                assert all([callable(x) for x in values_fn.values()])
            else:
                assert not isinstance(values_fn, list)
        
        assert isinstance(drop_na, bool)
        assert isinstance(retain_levels, bool)
        assert isinstance(sep, str)
        
        # make values_from a scalar if it is
        if len(values_from) == 1:
            values_from = values_from[0]
        
        res = pd.pivot_table(data         = self.__data
                             , index      = id_cols
                             , columns    = names_from
                             , values     = values_from
                             , fill_value = values_fill
                             , aggfunc    = values_fn
                             , margins    = False
                             , dropna     = drop_na
                             , observed   = retain_levels
                             )
                
        res = TidyDataFrame(tidy(res, sep), check = False)
            
        return res
    
    def pivot_longer(self
                     , cols
                     , names_to = "key"
                     , values_to = "value"
                     ):
        
        # assertions
        cn = self.get_colnames()
        assert is_string_or_string_list(cols)
        assert set(cols).issubset(cn)
        
        id_vars = set(cn).difference(cols)
        assert isinstance(names_to, str)
        assert isinstance(values_to, str)
        assert names_to not in id_vars
        assert values_to not in id_vars
        
        
        # core operation
        res = (self.__data
                   .melt(id_vars         = id_vars
                         , value_vars    = cols
                         , var_name      = names_to
                         , value_name    = values_to
                         , ignore_index  = True
                         )
                   )
        
        res = TidyDataFrame(res, check = False)
        return res
    
    # slice extensions
    def slice_head(self, n = None, prop = None):
        
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            n = int(np.floor(prop * nr))
        
        return TidyDataFrame(self.__data.head(n).reset_index(drop = True)
                             , check = False
                             )
    
    def slice_tail(self, n = None, prop = None):
        
        nr = self.get_nrow()
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            n = int(np.floor(prop * nr))
        
        return TidyDataFrame(self.__data.tail(n).reset_index(drop = True)
                             , check = False
                             )
    
    def slice_sample(self, n = None, prop = None, random_state = None):
        
        nr = self.get_nrow()
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            n = int(np.floor(prop * nr))
        
        return TidyDataFrame((self.__data
                                  .sample(n
                                          , replace = False
                                          , random_state = random_state
                                          , axis = "index"
                                          )
                                  .reset_index(drop = True)
                                  )
                              
                             , check = False
                             )
    
    def slice_bootstrap(self, n = None, prop = None, random_state = None):
        
        nr = self.get_nrow()
        
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            n = int(np.floor(prop * nr))
        
        return TidyDataFrame((self.__data
                                  .sample(n
                                          , replace = True
                                          , random_state = random_state
                                          , axis = "index"
                                          )
                                  .reset_index(drop = True)
                                  )
                              
                             , check = False
                             )
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  ):
        
        nr = self.get_nrow()
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            n = int(np.floor(prop * nr))
        
        if order_by is None:
            raise Exception("argument 'order_by' should not be None")
        
        if ties_method is None:
            ties_method = "all"
            
        res = (self.__data
                   .nsmallest(n, columns = order_by, keep = ties_method)
                   .reset_index(drop = True)
                   )
        return TidyDataFrame(res, check = False)
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  ):
        
        nr = self.get_nrow()
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            n = int(np.floor(prop * nr))
        
        if order_by is None:
            raise Exception("argument 'order_by' should not be None")
        
        if ties_method is None:
            ties_method = "all"
            
        res = (self.__data
                   .nlargest(n, columns = order_by, keep = ties_method)
                   .reset_index(drop = True)
                   )
        return TidyDataFrame(res, check = False)
    
    # expand and complete utilities
    def expand(self, l):
        # TODO
        return None
            
    def complete(self, l):
        expanded = df.expand(l)
        return expanded.join_left(df, on = expanded.get_colnames())
    
    # set like methods
    def union(self, y):
        assert set(self.get_colnames()) == set(y.get_colnames())
        return self.join_outer(y, on = self.get_colnames())
    
    def intersection(self, y):
        assert set(self.get_colnames()) == set(y.get_colnames())
        return self.join_inner(y, on = self.get_colnames())
        
    def setdiff(self, y):
        assert set(self.get_colnames()) == set(y.get_colnames())
        return self.join_anti(y, on = self.get_colnames())
    
    # na handling methods
    def replace_na(self, column_replace_dict):
        
        assert isinstance(column_replace_dict, dict)
        cns = self.get_colnames()
        for akey in column_replace_dict:
            if akey in cns:
                self = self.mutate({akey : (lambda x: np.where(x.isna()
                                                               , column_replace_dict[akey]
                                                               , x)
                                            ,
                                            )
                                    }
                                   )
        return self
    
    def drop_na(self, column_names = None):
        
        if column_names is not None:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            assert set(column_names).issubset(self.get_colnames())
        else:
            column_names = self.get_colnames()
        
        res = (self.__data
                   .dropna(axis = "index"
                           , how = "any"
                           , subset = column_names
                           , inplace = False
                           )
                   .reset_index(drop = True)
                   )
        
        return TidyDataFrame(res, check = False)
        
    def fill_na(self, column_direction_dict):
        
        assert isinstance(column_direction_dict, dict)
        assert set(column_direction_dict.keys()).issubset(self.get_colnames())
        valid_methods = ["up", "down", "updown", "downup"]
        assert set(column_direction_dict.values()).issubset(valid_methods)
        
        data = self.__data
        
        for akey in column_direction_dict:
            method = column_direction_dict[akey]
            if method == "up":
                data[akey] = data[akey].bfill()
            elif method == "down":
                data[akey] = data[akey].ffill()
            elif method == "updown":
                data[akey] = data[akey].bfill()
                data[akey] = data[akey].ffill()
            else:
                data[akey] = data[akey].ffill()
                data[akey] = data[akey].bfill()
            
            data[akey] = np.where(data[akey].isna(), pd.NA, data[akey])
            
        return TidyDataFrame(data, check = False)
    
    # string utilities
    def separate(self
                 , column_name
                 , into
                 , sep = '[^[:alnum:]]+'
                 , strict = True
                 , keep = False
                 ):
        
        split_df = pd.DataFrame(
            [re.split(sep, i) for i in self.to_series(column_name)]
            ).fillna(pd.NA)
        
        if len(into) == split_df.shape[1]:
            split_df.columns = into
        elif len(into) < split_df.shape[1]:
            if strict:
                raise Exception("Column is split into more number of columns than the length of 'into'")
            else:
                split_df = split_df.iloc[:, 0:len(into)]
                split_df = split_df.set_axis(into, axis = 'columns')
        else:
            if strict:
                raise Exception("Column is split into less number of columns than the length of 'into'")
            else:
                split_df.columns = into[0:split_df.shape[1]]     
            
        if keep:
            res = self.cbind(TidyDataFrame(split_df, check = False))
        else:
            res = (self.select(column_name, include = False)
                       .cbind(TidyDataFrame(split_df, check = False))
                       )
        return res
    
    def unite(self, column_names, into, sep = "_", keep = False):
        
        def reduce_join(df, columns, sep):
            assert len(columns) > 1
            slist = [df[x].astype(str) for x in columns]
            red_series = reduce(lambda x, y: x + sep + y, slist[1:], slist[0])
            return red_series.to_frame(name = into)
                
        joined = reduce_join(self.__data, column_names, sep)
        
        if not keep:
           res = (self.cbind(TidyDataFrame(joined, check = False))
                      .select(column_names, include = False)    
                      )
        else:
           res = self.cbind(TidyDataFrame(joined, check = False)) 
         
        return res
    
    def separate_rows(self, column_name, sep = ";"):
        
        def splitter(str_col):
            return [re.split(sep, x) for x in str_col]
            
        res = self.__data
        res[column_name] = splitter(res[column_name])
        res = res.explode(column_name, ignore_index = True)
        
        return TidyDataFrame(res, check = False)