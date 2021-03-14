import copy
import numpy as np
import pandas as pd
import warnings
from tabulate import tabulate

# from package_tidypandas.tidyGroupedDataFrame import tidyGroupedDataFrame
# from package_tidypandas.tidypredicates import *

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

def get_unique_names(strings):
    theset = set([])
    thelist = []
    for astring in strings:
        while(astring in theset):
            astring = astring + "_1"
        else:
            theset.add(astring)
            thelist.append(astring)
    
    return(thelist)

def tidy(pdf, sep = "__", verbose = False):
    
    assert isinstance(pdf, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy))
    assert isinstance(sep, str)
    
    is_grouped = False
    if isinstance(pdf, pd.core.groupby.DataFrameGroupBy):
        is_grouped = True
        gvs = pdf.grouper.names
        pdf = pdf.obj
    
    try:
        # handle column multiindex
        if isinstance(pdf.columns, pd.MultiIndex):
            # paste vertically with 'sep' and get unique names
            lol = list(map(list, list(pdf.columns)))
            cns = list(map(lambda x: sep.join(map(str, x)).rstrip(sep), lol))
            pdf.columns = get_unique_names(cns)
        else:
            # when not an multiindex, avoid column index from having some name
            pdf.columns.name = None
            pdf.columns = get_unique_names(list(pdf.columns))
    except:
        if verbose:
            raise Exception("Unable to tidy: Problem handling column index or multiindex")
    try:    
        # handle row multiindex    
        if isinstance(pdf.index, pd.MultiIndex):
            n_levels = len(pdf.index[0])
            pdf = pdf.droplevel(level = n_levels - 1).reset_index(drop = False)
        else:
            # convert non-multiindex to a column only if not rangeindex
            if not isinstance(pdf.index, (pd.RangeIndex, pd.Int64Index)):
                pdf = pdf.reset_index(drop = False)
    except:
        if verbose:
            raise Exception("Unable to tidy: Problem handling row index or multiindex")
        
    if is_grouped:
        pdf = pdf.groupby(gvs)
        
    if verbose:
        print("successfully tidied!")
    return pdf


def bind_rows(x, rowid_column_name = "rowid"):
    
    if isinstance(x, dict):
        str_keys = [str(x) for x in x.keys()]
        if len(set(str_keys)) < len(str_keys):
            raise Exception("keys of the dictionary should form unique strings")
        
        def add_rowid_column(val):
            pdf = val[1].ungroup().to_pandas()
            pdf[rowid_column_name] = str(val[0])
            return pdf
        
        pdfs = map(add_rowid_column, x.items())
    else:
        pdfs = map(lambda y: y.to_pandas(), x)
        
    res = pd.concat(pdfs, axis = 'index', ignore_index = True)
    res = tidyDataFrame(res, check = False)
    return res

def bind_cols(x):
    
    pdfs = list(map(lambda y: y.ungroup().to_pandas(), x))
    rls  = set(map(lambda y: y.shape[0], pdfs))
    if len(rls) > 1:
        raise Exception("Cannot bind columns as tidy dataframes have different number of rows")
    res = pd.concat(pdfs, axis = "columns", ignore_index = False)
    res.columns = get_unique_names(list(res.columns))
    return tidyDataFrame(res, check = False)
    
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
                raise Exception("Cannot use the pandas dataframe due to above warnings. Try the 'tidy' function.")
            
            print("Success! Created a tidy dataframe.")       
                    
        self.__data = copy.copy(x)
        return None
    
    # print method
    def __repr__(self):
        print('-- Tidy dataframe with shape: {shape}'\
              .format(shape = self.__data.shape))
        print('-- First few rows:')
        print(self.__data.head(10))
        #print(tabulate(self.__data.head(6),headers = 'keys',tablefmt = 'psql'))
        return ''
    
    # pandas copy method
    def to_pandas(self):
        return copy.copy(self.__data)
    
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
        
        if isinstance(res, pd.DataFrame):
            res = tidyDataFrame(res, check = False)
        else:
            res = tidyGroupedDataFrame(res, check = False)
        return res
    
    # alias for pipe_pandas
    pipe2 = pipe_pandas    
    
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
        
        return tidyDataFrame(res, check = False)

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

        return tidyDataFrame(mutated, check=False)

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
            
        return tidyDataFrame(mutated, check = False)
    
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
        a tidyDataFrame

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
        a tidyDataFrame

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

        return tidyDataFrame(pd.DataFrame(res), check=False)

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
        a tidyDataFrame
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

        return tidyDataFrame(pd.DataFrame(res), check=False)

    
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (tidyDataFrame, tidyGroupedDataFrame))
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
        return tidyDataFrame(res, check = False)
    
    def rbind(self, y):
        res = pd.concat([self.__data, y.ungroup().to_pandas()]
                        , axis = 0
                        , ignore_index = True # loose row indexes
                        )
        return tidyDataFrame(res, check = False)
    
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
            
        return tidyDataFrame(res, check = False)

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
                    , groupby_id_cols = False
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
        
        assert isinstance(groupby_id_cols, bool)
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
                
        res = tidyDataFrame(tidy(res, sep), check = False)
        if groupby_id_cols:
            res = res.group_by(id_cols)
            
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
        
        res = tidyDataFrame(res, check = False)
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
        
        return tidyDataFrame(self.__data.head(n).reset_index(drop = True)
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
        
        return tidyDataFrame(self.__data.tail(n).reset_index(drop = True)
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
        
        return tidyDataFrame((self.__data
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
        
        return tidyDataFrame((self.__data
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
        return tidyDataFrame(res, check = False)
    
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
        return tidyDataFrame(res, check = False)
    
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
        
        return tidyDataFrame(res, check = False)
        
        
        