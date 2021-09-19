# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import pandas as pd

def is_string_or_string_list(x):
    '''
    is_string_or_string_list(x)
    
    Check whether the input is a string or a list of strings

    Parameters
    ----------
    x : object
        Any python object

    Returns
    -------
    bool
    True if input is a string or a list of strings
    
    Examples
    --------
    is_string_or_string_list("bar")      # True
    is_string_or_string_list(["bar"])    # True
    is_string_or_string_list(("bar",))   # False
    is_string_or_string_list(["bar", 1]) # False
    '''
    res = False
    if isinstance(x, str):
        res = True
    elif isinstance(x, list) and len(x) >= 1:
        if all([isinstance(i, str) for i in x]):
            res = True
    else:
        res = False
    
    return res
    
def enlist(x):
    '''
    enlist(x)
    
    Returns the input in a list (as first element of the list) unless input itself is a list

    Parameters
    ----------
    x : object
        Any python object

    Returns
    -------
    list
    Returns the input in a list (as first element of the list) unless input itself is a list
    
    Examples
    --------
    enlist(["a"]) # ["a"]
    enlist("a")   # ["a"]
    enlist((1, )) # [(1, )]
    '''
    if not isinstance(x, list):
        x = [x]
    
    return x

def get_unique_names(strings):
    '''
    get_unique_names(strings)
    
    Returns a list of same length as the input such that elements are unique. This is done by adding '_1'. The resulting list does not alter nth element if the nth element occurs for the first time in the input list starting from left.
    
    Parameters
    ----------
    strings : list
        A list of strings

    Returns
    -------
    list of strings
    
    Examples
    --------
    get_unique_names(['a', 'b'])               # ['a', 'b']
    get_unique_names(['a', 'a'])               # ['a', 'a_1']
    get_unique_names(['a', 'a', 'a_1'])        # ['a', 'a_1_1', 'a_1']
    '''
    assert is_string_or_string_list(strings)
    strings = enlist(strings)

    new_list = []
    old_set = set(strings)
    
    for astring in strings:
        counter = 0 
        while True:
            if astring in new_list:
                counter = 1
                astring = astring + "_1" 
            elif astring in old_set:
                if counter > 0:
                    astring = astring + "_1"
                else:
                    new_list.append(astring)
                    try:
                        old_set.remove(astring)
                    except:
                        pass
                    break
            else:
                new_list.append(astring)
                try:
                    old_set.remove(astring)
                except:
                    pass
                break
        
    return new_list

def get_unique_sublist(x):
    '''
    get_unique_sublist
    Get unique elements of the list in their order of appearance
    
    Parameters
    ----------
    x : list

    Returns
    -------
    list
    
    Examples
    --------
    get_unique_sublist([2,1,2,0,2,3])
    '''
    ell = []
    S = set([])
    
    for ele in x:
        if ele not in S:
           ell.append(ele)
           S.add(ele)
    
    return ell

def reorder_set_by_list(set_obj, list_obj):
    '''
    reorder_set_by_list
    Reorder elements of a set according to their order(first appearance) in a list

    Parameters
    ----------
    set_obj : set
    list_obj : list

    Returns
    -------
    list
    
    Notes
    -----
    1. If the set contains an element not present in the list, then the output list will not contain the element.
    2. If the list contains an element not present in the set, then the output list will not contain the element.
    
    Examples
    --------
    reorder_set_by_list(set([1, 3, 2]), [1,2,3,1])
    '''
    
    list_obj = get_unique_sublist(list_obj)
    new_list = []
    for ele in list_obj:
        if ele in set_obj:
            new_list.append(ele)
    return new_list

def is_pdf(x):
    '''
    is_pdf(x)
    
    Returns True if the input is a grouped or ungrouped pandas dataframe

    Parameters
    ----------
    x : any python object

    Returns
    -------
    bool
    '''
    return isinstance(x, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy))

def is_ungrouped_pdf(x):
    '''
    is_ungrouped_pdf(x)
    
    Returns True if the input is an ungrouped pandas dataframe

    Parameters
    ----------
    x : any python object

    Returns
    -------
    bool
    '''
    return isinstance(x, pd.DataFrame)

def is_grouped_pdf(x):
    '''
    is_grouped_pdf(x)
    
    Returns True if the input is a grouped pandas dataframe

    Parameters
    ----------
    x : any python object

    Returns
    -------
    bool
    '''
    return isinstance(x, pd.core.groupby.DataFrameGroupBy)

def is_unique_list(x):
    '''
    is_unique_list(x)
    
    Returns True if input list does not have duplicates

    Parameters
    ----------
    x : list

    Returns
    -------
    bool
    '''
    assert isinstance(x, list)
    return len(set(x)) == len(x)

def simplify(pdf
             , drop_range_index = True
             , sep = "__"
             , verbose = False
             ):
    '''
    simplify(pdf)
    
    Returns a pandas dataframe with simplified index structure. This might be helpful before creating a TidyPandasDataFrame object.
    
    Parameters
    ----------
    pdf : Pandas dataframe
    drop_range_index: bool (default: True)
        Whether RangeIndex is to be dropped (takes effect only when the row index inherits pd.RangeIndex)
    sep: str (default: "__")
        String separator to be used while concatenating column multiindex
    verbose: bool (default: False)
        Whether to print the progress of simpliying process
    
    Returns
    -------
    A pandas dataframe with simplified index structure.
    
    Notes
    -----
    Returns a `simple` pandas dataframe. A pandas dataframe is 'simple' if:
    1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
    
    This is done by collapsing the column MultiIndex by concatenating names using separator 'sep' and ensuring that the resulting names are unique. The row Index or MultiIndex are added to the dataframe as columns if their names do not clash with the existing column names of the dataframe.
    
    Examples
    --------
    from nycflights13 import flights
    ex1 = flights.groupby('dest').apply(lambda x: x.head(2))
    ex1
    simplify(ex1)
    
    ex2 = pd.crosstab(flights['origin'], flights['dest'])
    ex2
    simplify(ex2)
    
    ex3 = (flights.value_counts(['origin', 'dest', 'month', 'hour'])
                  .reset_index()
                  .rename(columns = {0 : 'n'})
                  .pipe(lambda  x: pd.pivot_table(x
                                                  , index = ['origin', 'month']
                                                  , columns = ['dest', 'hour']
                                                  , values = 'n'
                                                  , fill_value = 0
                                                  )
                        )
                  )
    ex3
    simplify(ex3)
    '''
    
    assert isinstance(pdf, pd.DataFrame)
    assert isinstance(sep, str)
    
    # handle column multiindex
    try:
        if isinstance(pdf.columns, pd.MultiIndex):
            if verbose:
                print("Detected a column multiindex")
            # paste vertically with 'sep' and get unique names
            # a   d 
            # b c e
            # becomes
            # a__b,a__c, d__e
            lol = list(map(list, list(pdf.columns)))
            cns = list(map(lambda x: sep.join(map(str, x)).rstrip(sep), lol))
            pdf.columns = get_unique_names(cns)
        else:
            if verbose:
                print("Detected a simple column index")
            # avoid column index from having some name
            pdf.columns.name = None
            pdf.columns = get_unique_names(list(pdf.columns))
    except:
        if verbose:
            raise Exception("Unable to simplify: column index or multiindex")
    
    # handle row index 
    try:
        # if multiindex is present at row level
        # first attempt to convert them to columns
        # if that leads to an exception, drop it and warn the user
        if isinstance(pdf.index, pd.MultiIndex):
            if verbose:
                print("Detected a row multiindex")
            try:
                pdf = pdf.reset_index(drop = False)
            except:
                pdf = pdf.reset_index(drop = True)
                if verbose:
                    warnings.warn("Dropped the row index as inserting them creates duplicate column names.")
        else:
            # handle simple row index
            if verbose:
                print("Detected a simple row index")
            if isinstance(pdf.index, (pd.RangeIndex, pd.Int64Index)):
                if drop_range_index:
                    pdf = pdf.reset_index(drop = True)
                else:
                    try:
                        pdf = pdf.reset_index(drop = False)
                    except:
                        pdf = pdf.reset_index(drop = True)
                        if verbose:
                            warnings.warn("Dropped the row index as inserting them creates duplicate column names.")
            else:
                # handle simple non range index
                try:
                    pdf = pdf.reset_index(drop = False)
                except:
                    pdf = pdf.reset_index(drop = True)
                    if verbose:
                        warnings.warn("Dropped the row index as inserting them creates duplicate column names.")
    except:
        if verbose:
            raise Exception("Unable to simplify: row index or multiindex")
    
    # ensure column namemes and strigs and unique
    pdf.columns = get_unique_names(list(map(str, pdf.columns)))
            
    if verbose:
        print("Successfully simplified!")
    
    return pdf

def is_simple(pdf, verbose = False):
    '''
    is_simple
    Whether the input pandas dataframe is 'simple' or not

    Parameters
    ----------
    pdf : pandas dataframe
    verbose : bool, (default: False)
        When True, functon might warning(s) specifying the reason why the input is not 'simple'. The default is False.
    
    Notes
    -----
    A pandas (ungrouped or grouped) dataframe is 'simple' if:
    1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
    
    Returns
    -------
    bool
        True if the input is 'simple' and False if the input is not 'simple'.

    '''
    assert isinstance(pdf, pd.DataFrame)
    
    row_flag = not isinstance(pdf.index, pd.MultiIndex)
    col_flag = not isinstance(pdf.columns, pd.MultiIndex)
    # check if row index is rangeIndex
    flag_no_index = False
    if isinstance(pdf.index, (pd.RangeIndex, pd.Int64Index)):
        if pdf.index.start == 0 and pdf.index.step == 1:
            flag_no_index = True 
            
    # check if all column names are strings
    columns = list(pdf.columns)
    str_flag = all([isinstance(y, str) for y in columns])
    # check if column names are unique
    
    if is_unique_list(columns):
        unique_flag = True
    else:
        unique_flag = False
    
    flag = all([row_flag, col_flag, flag_no_index, str_flag, unique_flag])
    
    if verbose and not flag:
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
            
    return flag

# shorthand to convert a non-simple pandas dataframe to a TidyPandasDataFrame
def tidy(pdf
         , drop_range_index = True
         , sep = "__"
         , verbose = False
         , check = True
         ):
    
    res = TidyPandasDataFrame(simplify(pdf, drop_range_index, sep, verbose), check)
    return res

# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import copy
import numpy as np
import pandas as pd
import warnings
import re
from functools import reduce
from collections_extended import setlist
from skimpy import skim

class TidyPandasDataFrame:
    '''
    TidyPandasDataFrame class
    A tidy pandas dataframe is a wrapper over 'simple' ungrouped pandas DataFrame object.
    
    Notes
    -----
    
    A pandas dataframe is said to be 'simple' if:
    
    1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
    
    * Methods constitute a grammar of data manipulation mostly returning a 'simple' dataframe as a result. 
    * When a method returns a tidy dataframe, it always returns a copy and not a view. 
    * Methods 'to_pandas' and 'to_series' convert into pandas dataframe or series.
    * The only attribute of the class is the underlying pandas dataframe. This cannot be accessed by the user. Please use 'to_pandas' method to obtain a copy of the underlying pandas dataframe.
    
    Methods
    -------
    init
        Creates a tidy dataframe from a 'simple' pandas dataframe
        
    'to' methods:
        to_pandas
            Returns the underlying pandas dataframe
        to_series
            Returns selected column as pandas series
    
    'pipe' methods:
        pipe    
            A pipe method for tidy dataframe
        pipe_tee
            A pipe method called for side-effect and returns the input
    
    'get' methods:
        get_nrow
            Returns number of rows of the dataframe
        get_ncol
            Returns number of rows of the dataframe
        get_shape (alias: get_dim)
            Returns the shape or dimension of the dataframe
        get_colnames
            Returns column names of the dataframe

    
    'basic' methods:
        select
            Returns a dataframe with selected columns
        arrange
            Returns a dataframe after sorting rows
        slice
            Returns a dataframe to subset by row numbers
        distinct
            Returns a distinct rows defined by selected columns
        filter
            Returns a dataframe with rows selected by some criteria
        mutate
            Returns a dataframe by adding or changing a few columns
        summarise (alias: summarize)
            Returns a dataframe after aggregating over selected columns
        
    'count' methods:
        count
            Returns a dataframe after counting over selected columns
        add_count
            Returns a dataframe by adding a new count column to input
            
    'pivot' methods:
        pivot_wider
            Returns a dataframe by converting from long to wide format
        pivot_longer
            Returns a dataframe by converting from wide to long format
            
    'missing value' methods:
        drop_na
            Returns a dataframe by dropping rows which have mssing values in selected columns
        relapce_na
            Returns a dataframe by replacing missing values in selected columns
        fill_na
            Returns a dataframe by filling missing values from up, down or both directions for selected columns
    'string' methods:
        separate
            Returns a dataframe by splitting a string column into multiple columns
        unite
            Returns a dataframe by combining multiple string columns 
        separate_rows
            Returns a dataframe by exploding a string column
    'completion' methods:
        expand
            Returns a dataframe with combinations of columns
        complete
            Returns a dataframe by creating additional rows with some comninations of columns 
        slice extensions:
    
    'slice' extensions:
        slice_head
            Returns a dataframe with top few rows of the input
        slice_tail
            Returns a dataframe with last few rows of the input
        slice_max
            Returns a dataframe with few rows corrsponding to maximum values of some columns
        slice_min
            Returns a dataframe with few rows corrsponding to maximum values of some columns
        slice_sample
            Returns a dataframe with a sample of rows
    
    'join' methods:
        join, inner_join, outer_join, left_join, right_join, anti_join:
            Returns a joined dataframe of a pair of dataframes
    'set operations' methods:
        union, intersection, setdiff:
            Returns a dataframe after set like operations over a pair of dataframes
    'bind' methods:
        rbind, cbind:
            Returns a dataframe by rowwise or column wise binding of a pair of dataframes
    
    'misc' methods:
        add_rowid:
            Returns a dataframe with rowids added
    
    '''
    # init method
    def __init__(self, x, check = True):
        '''
        init
        Create tidy dataframe from a 'simple' ungrouped pandas dataframe

        Parameters
        ----------
        x : 'simple' pandas dataframe
        check : bool, optional, Default is True
            Whether to check if the input pandas dataframe is 'simple'. It is advised to set this to True in user level code.

        Raises
        ------
        Exception if the input pandas dataframe is not simple and warning messages pin point to the precise issue so that user can make necessary changes to the input pandas dataframe. 
        
        Notes
        -----
        A pandas dataframe is said to be 'simple' if:
        1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
        2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
        
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(flights)
        flights_tidy
        '''
        assert isinstance(check, bool)
        if check:
            flag_simple = is_simple(x, verbose = True)
            if not flag_simple:    
            # raise the error After informative warnings
                raise Exception(("Input pandas dataframe is not 'simple'."
                                 " See to above warnings."
                                 " Try the 'simplify' function."
                                 " ex: simplify(not simple pandas dataframe) --> simple pandas dataframe."
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
    
    ##########################################################################
    # to pandas methods
    ##########################################################################
    
    # pandas copy method
    def to_pandas(self):
        '''
        to_pandas
        Return (a copy) underlying pandas dataframe
        
        Returns
        -------
        pandas dataframe
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(flights)
        flights_tidy.to_pandas()
        '''
        return copy.copy(self.__data)
    
    # series copy method
    def to_series(self, column_name):
        '''
        to_series
        Returns (a copy) a column as pandas series
        
        Parameters
        ----------
        column_name : str
            Name of the column to be returned as pandas series

        Returns
        -------
        pandas series
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(flights)
        flights_tidy.to_series("origin")
        '''
        
        assert isinstance(column_name, str),\
            "Input column names should be a string"
        assert column_name in list(self.__data.columns), \
            "column_name is not an existing column name"
        
        res = self.__data[column_name]
        return res
    
    ##########################################################################
    # pipe methods
    ##########################################################################
    
    # pipe method
    def pipe(self, func):
        '''
        pipe
        Returns func(self)

        Parameters
        ----------
        func : callable

        Returns
        -------
        Depends on the return type of `func`
        '''
        return func(self)
    
    # pipe_tee
    def pipe_tee(self, func):
        '''
        pipe_tee
        pipe for side-effect

        Parameters
        ----------
        func : callable

        Returns
        -------
        TidyPandasDataFrame
        '''
        func(self)
        return self
    
    # TODO, implement pipe_tee for side-effects
    
    ##########################################################################
    # get methods
    ##########################################################################
            
    def get_nrow(self):
        '''
        get_nrow
        Get the number of rows
        
        Returns
        -------
        int
        '''
        return self.__data.shape[0]
    
    def get_ncol(self):
        '''
        get_ncol
        Get the number of columns
        
        Returns
        -------
        int
        '''
        return self.__data.shape[1]
        
    def get_shape(self):
        '''
        get_shape (alias get_dim)
        Get the number of rows and columns
        
        Returns
        -------
        tuple
            Number of rows and Number of columns
        '''
        return self.__data.shape
        
    get_dim = get_shape
        
    def get_colnames(self):
        '''
        get_colnames
        Get the column names of the dataframe

        Returns
        -------
        list
            List of unique strings that form the column index of the underlying pandas dataframe

        '''
        return list(self.__data.columns)
    
    def skim(self):
        return skim(self.to_pandas())
    
    def add_rowid(self, name = 'rowid', by = None):
        '''
        add_rowid
        Add a rowid column to TidyPandasDataFrame
        
        Parameters
        ----------
        name : str
            Name for the rowid column
        by : str or list of strings
            Columns to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from nycflights13 import planes

        tidy(planes).select(['tailnum', 'seats']).add_rowid()
        
        # add rowid column grouped by 'seats' column
        # row order is preserved
        tidy(planes).select(['tailnum', 'seats']).add_rowid(by = 'seats')
        '''
        assert isinstance(name, str)
        if name[-2:] == "__":
            raise Exception("'name' should not be ending with '__'.")
        
        if name in self.get_colnames():
            raise Expection("'name' should not be an existing column name.")
            
        nr = self.get_nrow()
        
        if by is None:
            res = self.__data.assign(**{name : np.arange(nr)})
        else:
            by = enlist(by)
            assert is_string_or_string_list(by)
            assert set(self.get_colnames()).issuperset(by),\
                "'by' should be a list of valid column names"
            res = (self.__data
                       .assign(**{"rn__" : np.arange(nr)})
                       .groupby(by, sort = False, dropna = False)
                       .apply(lambda x: x.assign(**{name : np.arange(x.shape[0])}))
                       .reset_index(drop = True)
                       .sort_values("rn__", ignore_index = True)
                       .drop(columns = "rn__")
                       )
        
        return TidyPandasDataFrame(res, check = False)
    
    def _validate_by(self, by):
        by = enlist(by)
        assert is_string_or_string_list(by),\
            "arg 'by' should be a string or a list of strings"
        assert set(self.get_colnames()).issuperset(by),\
            "arg 'by' should be a string or list of strings of valid column names"
        
        return None
    
    def apply_over_groups(self
                          , func
                          , by
                          , preserve_row_order = False
                          , row_order_column_name = "rowid_temp"
                          , is_pandas_udf = False
                          , **kwargs
                          ):
        '''
        apply_over_groups
        Split by some columns, apply a function per chunk which returns a dataframe 
        and then combine it back into a single dataframe
        
        Parameters
        ----------
        func: callable
            Type 1. A function: TidyDataFrame --> TidyDataFrame or
            Type 2. A function: simple pandas dataframe --> simple pandas dataframe
            In latter case, set 'is_pandas_udf' to True.
        
        by: string or list of strings
            Column names to group by
        
        preserve_row_order: bool (default is False)
            Whether to preserve the row order of the input dataframe
            
        is_pandas_udf: bool (default is False)
            Whether the 'func' argument is of type 2
            
        **kwargs: arguments to 'func'
        
        Returns
        -------
        TidyPandasDataFrame
        
        '''
        assert callable(func)
        
        by = enlist(by)
        self._validate_by(by)
        
        assert isinstance(preserve_row_order, bool)
        assert isinstance(row_order_column_name, str)
        assert row_order_column_name not in self.get_colnames()
        
        if is_pandas_udf:
            def wrapper_func(chunk, **kwargs):
                res     = func(chunk, **kwargs)
                by_left = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res = res.drop(columns = by_left)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                return res
        else:
            def wrapper_func(chunk, **kwargs):
                # i/o are pdfs
                chunk_tidy = TidyPandasDataFrame(chunk, check = False)
                res        = func(chunk_tidy, **kwargs).to_pandas()
                by_left    = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res = res.drop(columns = by_left)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                return res
        
        nr = self.get_nrow()
        if preserve_row_order:
            res = (self.__data
                        .assign(**{row_order_column_name: np.arange(nr)})
                        .groupby(by, sort = False, dropna = False)
                        .apply(wrapper_func)
                        .reset_index(drop = True)
                        )

            if row_order_column_name in list(res.columns):
                res = (res.sort_values(row_order_column_name, ignore_index = True)
                          .drop(columns = row_order_column_name)
                          )
            else:
                raise Exception("'row_order_column_name' in each chunk should be retained, when preserve_row_order is True")
        else:
            res = (self.__data
                        .groupby(by, sort = False, dropna = False)
                        .apply(wrapper_func)
                        .reset_index(drop = True)
                        )
        
        if (not is_simple(res)) and is_pandas_udf:
            raise Exception("Resulting dataframe after apply should be 'simple', examine the pandas UDF")
            
        return TidyPandasDataFrame(res, check = False)
    
    ##########################################################################
    # basic verbs
    ##########################################################################
    
    
    ##########################################################################
    # select
    ##########################################################################
    
    def select(self, column_names = None, predicate = None, include = True):
        '''
        select
        Select a subset of columns by name or predicate

        Parameters
        ----------
        column_names : list, optional
            list of column names(strings) to be selected. The default is None.
        predicate : callable, optional
            function which returns a bool. The default is None.
        include : bool, optional
            Whether the columns are to be selected or not. The default is True.

        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Select works by either specifying column names or a predicate, not both.
        2. When predicate is used, predicate should accept a pandas series and return a bool. Each column is passed to the predicate and the result indicates whether the column should be selected or not.
        3. When include is False, we select the remaining columns.
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(flights)
        flights_tidy
        
        # select with names
        flights_tidy.select(['origin', 'dest'])
        
        # select using a predicate: only non-numeric columns
        flights_tidy.select(predicate = lambda x: x.dtype == "object")
        
        # select columns ending with 'time'
        flights_tidy.select(predicate = lambda x: bool(re.match(".*time$", x.name)))
        
        # invert the selection
        flights_tidy.select(['origin', 'dest'], include = False)
        '''
        
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among `column_names` and `predicate` should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among `column_names` and `predicate` should not be None')
        
        if column_names is None:
            assert callable(predicate), "`predicate` should be a function"
            col_bool_list = np.array(list(self.__data.apply(predicate, axis = "index")))
            column_names = list(np.array(self.get_colnames())[col_bool_list])
        else:
            assert is_string_or_string_list(column_names),\
                "column names to select should be a list of strings"
            column_names = list(setlist(enlist(column_names)))
            cols = self.get_colnames()
            assert set(cols).issuperset(column_names),\
                "Atleast one string in `column_names` is not an existing column name"
        
        if not include:
            column_names = list(setlist(cols).difference(set(column_names)))
        
        if len(column_names) == 0:
            warnings.warn("None of the columns are selected")
            
        res = self.__data.loc[:, column_names]
        return TidyPandasDataFrame(res, check = False)
    
    def relocate(self, column_names, before = None, after = None):
        '''
        relocate
        relocate the columns of the tidy pandas dataframe

        Parameters
        ----------
        column_names : string or a list of strings
            column names to be moved
        before : string, optional
            column before which the column are to be moved. The default is None.
        after : TYPE, optional
            column after which the column are to be moved. The default is None.
        
        Notes
        -----
        Only one among 'before' and 'after' can be not None. When both are None, the columns are added to the begining of the dataframe (leftmost)
        
        Returns
        -------
        tidy pandas dataframe
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(flights)
        flights_tidy
        
        # move "distance" and "month" columns to the left of the dataframe
        flights_tidy.relocate(["distance", "month"])
        
        # move "distance" and "month" columns to the left of "day" column
        flights_tidy.relocate(["distance", "month"], before = "day")
        
        # move "distance" and "month" columns to the right of "day" column
        flights_tidy.relocate(["distance", "month"], after = "day")
        '''
        
        assert is_string_or_string_list(column_names)
        column_names = enlist(column_names)
        assert is_unique_list(column_names) # assert if column_names are unique
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
        return TidyPandasDataFrame(res, check = False)
    
    def rename(self, old_new_dict):
        '''
        rename
        Rename columns of the tidy pandas dataframe
        
        Parameters
        ----------
        old_new_dict: A dictionary with old names as keys and new names as values
        
        Returns
        -------
        tidy pandas dataframe
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyPandasDataFrame(simplify(flights))
        flights_tidy

        flights_tidy.rename({'year': "Year", 'month': "montH"})
        '''
        col_names = self.get_colnames()
        assert isinstance(old_new_dict, dict)
        assert set(col_names).issuperset(old_new_dict.keys()) # old names should be there
        assert is_unique_list(list(old_new_dict.values())) # new names should be unique
        # TODO some more checks on column names are required
        
        # new names should not intersect with 'remaining' names
        remaining = set(col_names).difference(old_new_dict.keys())
        assert len(remaining.intersection(old_new_dict.values())) == 0
        
        res = self.__data.rename(columns = old_new_dict)
        return TidyPandasDataFrame(res, check = False)
    
    def slice(self, row_numbers, by = None):
        '''
        slice
        Subset rows of a TidyPandasDataFrame
        
        Parameters
        ----------
        row_numbers : list or 1-D numpy array 
            list/array of row numbers.
        by : list of strings, optional
            Column names to groupby. The default is None.

        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from nycflights13 import flights

        flights_tidy = tidy(flights)
        flights_tidy
        
        # pick first three rows of the dataframe
        flights_tidy.slice(np.arange(3))
        
        # pick these row numbers: [0, 3, 8]
        flights_tidy.slice([0, 3, 8])
        
        # pick first three rows for each month
        flights_tidy.slice([0,1,2], by = "month")
        
        # pick first three rows for each month and day
        flights_tidy.slice(np.arange(3), by = ["month", "day"])
        '''
        if by is None:
            minval = np.min(row_numbers)
            maxval = np.max(row_numbers)
            assert minval >= 0 and maxval <= self.get_nrow()
            
            res = self.__data.take(row_numbers).reset_index(drop = True)
        else:
            by = enlist(by)
            assert is_string_or_string_list(by)
            assert set(self.get_colnames()).issuperset(by),\
                "'by' should be a list of valid column names"
            res = (self.__data
                       .groupby(by, sort = False, dropna = False)
                       .apply(lambda chunk: (chunk.take(row_numbers)
                                                  .reset_index(drop = True)
                                                  )
                              )
                       .reset_index(drop = True)
                       )
        
        return TidyPandasDataFrame(res, check = False)
        
    def arrange(self
                , column_names
                , by = None
                , ascending = False
                , na_position = 'last'
                ):
        '''
        arrange
        Orders the rows of a TidyPandasDataFrame by the values of selected
        columns
        
        Parameters
        ----------
        column_names : list of strings
            column names to order by.
        by: str or a list of strings
            column names to group by
        ascending : bool or a list of booleans, optional
            DESCRIPTION. The default is False.
        na_position : str, optional
            One among: 'first', 'last'. The default is 'last'.

        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        # arrange by descending order of column 'distance'
        flights_tidy.arrange('distance')
        
        # arrange by ascending order of column 'distance'
        flights_tidy.arrange('distance', ascending = True)
        
        # arrange by ascending order of column 'distance' and break ties with
        # ascending order to 'dep_time'
        flights_tidy.arrange(['distance', 'dep_time'], ascending = True)
        
        # arrange by ascending order of column 'distance' and break ties with
        # descending order to 'dep_time'
        flights_tidy.arrange(['distance', 'dep_time'], ascending = [True, False])
        
        # determine where NA has to appear
        flights_tidy.arrange(['distance', 'dep_time']
                             , ascending = True
                             , na_position = 'first'
                             )
        
        # TODO: add grouped arrange examples
        
        Notes
        -----
        1. Grouped arrange rearranges the rows within a group without
        changing the relative position of groups in the dataframe.
        
        2. When multiple columns are provided to arrange, second column is
        used to break the ties after sorting by first column and so on.
        
        3. If the column provided in arrange are not sufficint to order 
        the rows of the dataframe, then row number is implicitly used as 
        the last column to deterministicly break ties.
        
        '''
        column_names = enlist(column_names)
        assert len(column_names) > 0
        cn = self.get_colnames()
        assert all([x in cn for x in column_names])
        if not isinstance(ascending, list):
            isinstance(ascending, bool)
        else:
            assert all([isinstance(x, bool) for x in ascending])
            assert len(ascending) == len(column_names)
        
        if by is None:
            res = self.__data.sort_values(by = column_names
                                          , axis         = 0
                                          , ascending    = ascending
                                          , inplace      = False
                                          , kind         = 'quicksort'
                                          , na_position  = na_position
                                          , ignore_index = True
                                          )
        else:
            by = enlist(by)
            assert is_string_or_string_list(by)
            assert set(self.get_colnames()).issuperset(by),\
                "'by' should be a list of valid column names"
            assert len(set(by).intersection(column_names)) != 0, \
                "'column_names' and 'by' should not have common names."
            # TODO implement grouped arrange
            raise Exception('grouped arrange is yet to be implemented')
            res = None
            
        return TidyPandasDataFrame(res, check = False)
        
    def filter(self, query = None, mask = None, by = None):
        
        if query is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        # validate query
        if query is not None:
            assert callable(query) or isinstance(query, str)
            
        # validate mask
        if mask is not None:
            assert isinstance(mask, (pd.Series, np.array, list))
            
        # validate 'by'
        if by is not None:
            by = enlist(by)
            assert is_string_or_string_list(by)
            assert set(self.get_colnames()).issuperset(by),\
                "'by' should be a list of valid column names"
        
        if query is not None and mask is None:
            if by is None:
                if isinstance(query, str):
                    res = self.__data.query(query)
                else:
                    res = (self.__data
                               .assign(**{"mask__": query})
                               .query("mask__")
                               .drop(columns = "mask__")
                               )
            else: # grouped case
                if isinstance(query, str):
                    res = (self.__data
                               .groupby(by, sort = False, dropna = False)
                               .apply(lambda chunk: chunk.query(query))
                               .reset_index(drop = True)
                               )
                else:
                    res = (self.__data
                               .groupby(by, sort = False, dropna = False)
                               .apply(lambda chunk: (chunk.assign(**{"mask__": query})
                                                          .query("mask__")
                                                          .drop(columns = "mask__")
                                                          )
                                     )
                               .reset_index(drop = True)
                               ) 
        
        if query is None and mask is not None:
            res = self.__data.iloc[mask, :]
        
        res = res.reset_index(drop = True)     
        return TidyPandasDataFrame(res, check = False)
        
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
        
        return TidyPandasDataFrame(res, check = False)

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

        return TidyPandasDataFrame(mutated, check=False)

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
            
        return TidyPandasDataFrame(mutated, check = False)
    
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
        a TidyPandasDataFrame

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
        a TidyPandasDataFrame

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

        return TidyPandasDataFrame(pd.DataFrame(res), check=False)

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
        a TidyPandasDataFrame
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

        return TidyPandasDataFrame(pd.DataFrame(res), check=False)
    
    summarize = summarise
    
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (TidyPandasDataFrame, TidyGroupedDataFrame))
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
                
        return TidyPandasDataFrame(res, check = False)
        
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
    
    inner_join = join_inner
    outer_join = join_outer
    left_join = join_left
    right_join = join_right
    anti_join = join_anti
    
    # binding methods
    def cbind(self, y):
        # number of rows should match
        assert self.get_nrow() == y.get_nrow()
        # column names should differ
        assert len(set(self.get_colnames()).intersection(y.get_colnames())) == 0
        
        res = pd.concat([self.__data, y.ungroup().to_pandas()]
                        , axis = 1
                        , ignore_index = False # not to loose column names
                        )
        return TidyPandasDataFrame(res, check = False)
    
    def rbind(self, y):
        res = pd.concat([self.__data, y.ungroup().to_pandas()]
                        , axis = 0
                        , ignore_index = True # loose row indexes
                        )
        return TidyPandasDataFrame(res, check = False)
    
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
            
        return TidyPandasDataFrame(res, check = False)

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
                
        res = TidyPandasDataFrame(tidy(res, sep), check = False)
            
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
        
        res = TidyPandasDataFrame(res, check = False)
        return res
    
    # slice extensions
    def slice_head(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        '''
        slice_head
        Subset first few rows
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            proportion of rows to subset, should be between 0(exclusive) and 1(inclusive)
        rounding_type: string
            When prop is provided, rounding method to be used.
            Options: 'round' (default), 'floor', 'ceiling'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is preserved by the method.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy.slice_head(n = 3)
        penguins_tidy.slice_head(prop = 0.01)
        
        penguins_tidy.slice_head(n = 1, by = 'species')
        penguins_tidy.slice_head(prop = 0.01, by = 'species')
        '''
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
            
        if by is None:
            assert n <= self.__data.shape[0],\
                "arg 'n' should not exceed the number of rows of the dataframe"
            res = TidyPandasDataFrame(self.__data.head(n), check = False)
        else:
            self._validate_by(by)
            by = enlist(by)
            if not case_prop:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                 
                res = (self.apply_over_groups(lambda x: x.head(n)
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
                           )
            else:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    "arg 'ties_method' should be one among: 'round' (default), 'ceiling', 'floor'"
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                    
                res = (self.apply_over_groups(
                    lambda x: x.head(int(roundf(x.shape[0] * prop)))
                    , by = by
                    , is_pandas_udf      = True
                    , preserve_row_order = True
                    ))
            
        return res
    
    def slice_tail(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        '''
        slice_tail
        Subset last few rows
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            proportion of rows to subset, should be between 0(exclusive) and 1(inclusive)
        rounding_type: string
            When prop is provided, rounding method to be used. 
            Options: 'round' (default), 'floor', 'ceiling'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is preserved by the method.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy.slice_tail(n = 3)
        penguins_tidy.slice_tail(prop = 0.01)
        
        penguins_tidy.slice_tail(n = 1, by = 'species')
        penguins_tidy.slice_tail(prop = 0.01, by = 'species')
        '''
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
            
        if by is None:
            assert n <= self.__data.shape[0],\
                "arg 'n' should not exceed the number of rows of the dataframe"
            res = TidyPandasDataFrame(self.__data.tail(n).reset_index(drop = True)
                                      , check = False
                                      )
        else:
            self._validate_by(by)
            by = enlist(by)
            if not case_prop:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                 
                res = (self.apply_over_groups(lambda x: x.tail(n).reset_index(drop = True)
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
                           )
            else:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    "arg 'ties_method' should be one among: 'round' (default), 'ceiling', 'floor'"
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                    
                res = (self.apply_over_groups(
                    lambda x: (x.tail(int(roundf(x.shape[0] * prop)))
                                .reset_index(drop = True)
                                )
                    , by = by
                    , is_pandas_udf      = True
                    , preserve_row_order = True
                    ))
            
        return res
    
    def slice_sample(self
                     , n            = None
                     , prop         = None
                     , replace      = False
                     , weights      = None
                     , random_state = None
                     , by           = None
                     ):
        '''
        slice_sample
        Sample a few rows
        
        Parameters
        ----------
        n: int
            Number of rows to sample, should be atleast 1
        prop: float
            proportion of rows to subset, should be non-negative
        replace: bool, default is False
            Whether to sampling should be done with replacement
        weights: string, pandas series, numpy array
            When a string, it should be an column of numeric dtype
            Default is None
        random_state: positive integer
            seed to keep the sampling reproducible
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is not preserved by the method.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        penguins_tidy = tidy(penguins)
        
        # sample by specifiying count
        penguins_tidy.slice_sample(n = 5)                    # sample without replacement
        penguins_tidy.slice_sample(n = 5, replace = True)    # sample with replacement, smaller than input
        penguins_tidy.slice_sample(n = 1000, replace = True) # sample with replacement, larger than input
        
        # sample by specifiying proportion of number of rows of the input
        penguins_tidy.slice_sample(prop = 0.3) # sample without replacement
        penguins_tidy.slice_sample(prop = 0.3) # sample with replacement, smaller than input
        penguins_tidy.slice_sample(prop = 1.1) # sample with replacement, larger than input
        
        # sample with weights
        penguins_tidy.slice_sample(prop = 0.3, weights = 'year')
        
        # sampling is reproducible by setting a random state
        penguins_tidy.slice_sample(n = 3, random_state = 42)
        
        # per group sampling
        penguins_tidy.slice_sample(n = 5, by = 'species')
        '''
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int"
            assert prop > 0,\
                "arg 'prop' should be a positive float or int"
        
        assert isinstance(replace, bool),\
            "Arg 'replace' should be True or False"
        if prop is not None:
            if prop > 1 and replace is False:
                raise Exception("When arg 'prop' > 1, arg 'replace' should be True")
            
        if n is not None:
            if n > nr and replace is False:
                raise Exception("When arg 'n' is greater than the number of rows of the input, arg 'replace' should be True")
            
        # Arg weights is not validated
        # Arg random_state is not validated
        
        if by is None:
            if n is not None:
                res = (self.__data
                           .sample(n              = n
                                   , axis         = "index"
                                   , replace      = replace
                                   , weights      = weights
                                   , random_state = random_state
                                   )
                           .reset_index(drop = True)
                           )
            else:
                res = (self.__data
                           .sample(frac           = prop
                                   , axis         = "index"
                                   , replace      = replace
                                   , weights      = weights
                                   , random_state = random_state
                                   )
                           .reset_index(drop = True)
                           )
        else:
            by = enlist(by)
            self._validate_by(by)
            n_groups = (self.__data
                            .groupby(by, sort = False, dropna = False)
                            .ngroups
                            )
            
            if random_state is not None:
                np.random.seed(random_state)
            seeds = np.random.choice(n_groups * 100, n_groups, replace = False)
            
            groups_frame = self.__data.loc[:, by].drop_duplicates()
            groups_frame["seed__"] = seeds
            
            if n is not None:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                def sample_chunk_n(x):
                    rs = x['seed__'].iloc[0]
                    res = (x.sample(n              = n
                                    , axis         = "index"
                                    , replace      = replace
                                    , weights      = weights
                                    , random_state = rs
                                    )
                            .reset_index(drop = True)
                            )
                    return res
                
                res = (self.__data
                           .merge(groups_frame, on = by, how = 'left')
                           .groupby(by, sort = False, dropna = False)
                           .apply(sample_chunk_n)
                           .reset_index(drop = True)
                           .drop(columns = "seed__")
                           )
            else:
                def sample_chunk_prop(x):
                    rs = x['seed__'].iloc[0]
                    res = (x.sample(frac           = prop
                                    , axis         = "index"
                                    , replace      = replace
                                    , weights      = weights
                                    , random_state = rs
                                    )
                            .reset_index(drop = True)
                            )
                    return res
                
                res = (self.__data
                           .merge(groups_frame, on = by, how = 'left')
                           .groupby(by, sort = False, dropna = False)
                           .apply(sample_chunk_prop)
                           .reset_index(drop = True)
                           .drop(columns = "seed__")
                           )
            
            res = (res.sample(frac = 1, random_state = random_state)
                      .reset_index(drop = True)
                      )
            
        return TidyPandasDataFrame(res, check = False)
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  , rounding_type = "round"
                  , by = None
                  ):
        
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
        
        if order_by is None:
            raise Exception("arg 'order_by' should not be None")
        
        assert isinstance(ties_method, str),\
            "arg 'ties_method' should be a string"
        assert ties_method in ['all', 'first', 'last'],\
            "arg 'ties_method' should be one among: 'all' (default), 'first', 'last'"
        
        if by is None:  
            res = (self.__data
                       .nsmallest(n, columns = order_by, keep = ties_method)
                       .reset_index(drop = True)
                       .pipe(lambda x: TidyPandasDataFrame(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = enlist(by)
            if case_prop:
                
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    "arg 'ties_method' should be one among: 'round' (default), 'ceiling', 'floor'"
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                res = self.apply_over_groups(lambda x: x.nsmallest(int(roundf(x.shape[0] * prop))
                                                                   , columns = order_by
                                                                   , keep = ties_method
                                                                   )
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
            else:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                res = self.apply_over_groups(lambda x: x.nsmallest(n
                                                                   , columns = order_by
                                                                   , keep = ties_method
                                                                   )
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
            
        return res
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  , rounding_type = "round"
                  , by = None
                  ):
        
        nr = self.get_nrow()

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
        
        if order_by is None:
            raise Exception("arg 'order_by' should not be None")
        
        assert isinstance(ties_method, str),\
            "arg 'ties_method' should be a string"
        assert ties_method in ['all', 'first', 'last'],\
            "arg 'ties_method' should be one among: 'all' (default), 'first', 'last'"
        
        if by is None:  
            res = (self.__data
                       .nlargest(n, columns = order_by, keep = ties_method)
                       .reset_index(drop = True)
                       .pipe(lambda x: TidyPandasDataFrame(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = enlist(by)
            if case_prop:
                
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    "arg 'ties_method' should be one among: 'round' (default), 'ceiling', 'floor'"
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                res = self.apply_over_groups(lambda x: x.nlargest(int(roundf(x.shape[0] * prop))
                                                                   , columns = order_by
                                                                   , keep = ties_method
                                                                   )
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
            else:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                res = self.apply_over_groups(lambda x: x.nlargest(n
                                                                   , columns = order_by
                                                                   , keep = ties_method
                                                                   )
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             )
            
        return res
    
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
        
        return TidyPandasDataFrame(res, check = False)
        
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
            
        return TidyPandasDataFrame(data, check = False)
    
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
            res = self.cbind(TidyPandasDataFrame(split_df, check = False))
        else:
            res = (self.select(column_name, include = False)
                       .cbind(TidyPandasDataFrame(split_df, check = False))
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
           res = (self.cbind(TidyPandasDataFrame(joined, check = False))
                      .select(column_names, include = False)    
                      )
        else:
           res = self.cbind(TidyPandasDataFrame(joined, check = False)) 
         
        return res
    
    def separate_rows(self, column_name, sep = ";"):
        
        def splitter(str_col):
            return [re.split(sep, x) for x in str_col]
            
        res = self.__data
        res[column_name] = splitter(res[column_name])
        res = res.explode(column_name, ignore_index = True)
        
        return TidyPandasDataFrame(res, check = False)
    
    # nest and unnest ----------------------------------------------------------------------
    
    def nest(self, by, column_name = 'data'):
        
        self._validate_by(by)
        by = enlist(by)
        
        cn = self.get_colnames()
        assert column_name not in cn,\
            "arg 'column_name' should not be an exisiting column name"
        
        # make a df from the distinct values
        go = self.__data.groupby(by, sort = False, dropna = False)
        res = pd.DataFrame(list(go.groups.keys()), columns = by)
        # add data into column_name column
        res[column_name] = pd.Series(map(lambda x: TidyPandasDataFrame(x.drop(columns = by)
                                                                       , check = False
                                                                       )
                                         , dict(tuple(go)).values()
                                         )
                                    )
        
        # non-exisiting groups result in NaN, make them empty dataframe                            
        cols   = setlist(cn).difference(by)
        na_pos = np.where(res[column_name].isna().to_numpy())[0]
        for x in na_pos:
            res[column_name][x] = TidyPandasDataFrame(pd.DataFrame(columns = cols)
                                                      , check = False
                                                      )
        
        return TidyPandasDataFrame(res, check = False)
    
    def unnest(self, column_name = 'data'):
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        assert column_name in cn,\
            "arg 'column_name' should be a exisiting column name"
         
        assert all(map(lambda x: isinstance(x, TidyPandasDataFrame)
                       , list(self.__data[column_name])
                       )
                   ),\
            "arg 'column_name' column is not a column of TidyPandasDataFrames"
            
        cols = setlist(cn).difference([column_name])
        # create a filled up dataframe per row via cross join and rbind them
        res = pd.concat(map(lambda x: pd.merge(self.__data.loc[[x], cols]
                                               , (self.__data[column_name][x]).to_pandas()
                                               , how = 'cross'
                                               )
                            , range(nr)
                            )
                        , axis = 'index'
                        , ignore_index = True
                        )
        return TidyPandasDataFrame(res, check = False)
