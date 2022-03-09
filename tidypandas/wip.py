# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import pandas as pd
import inspect

def _is_kwargable(func):
  
    res = False
    spec = inspect.getfullargspec(func)
    if spec.varkw is not None:
        res = True
    return res

def _is_valid_colname(string):
    return isinstance(string, str) and len(string) != 0 and string[0] != "_"
  
def _is_string_or_string_list(x):
    '''
    _is_string_or_string_list(x)
    
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
    _is_string_or_string_list("bar")      # True
    _is_string_or_string_list(["bar"])    # True
    _is_string_or_string_list(("bar",))   # False
    _is_string_or_string_list(["bar", 1]) # False
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
    
def _enlist(x):
    '''
    _enlist(x)
    
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
    _enlist(["a"]) # ["a"]
    _enlist("a")   # ["a"]
    _enlist((1, )) # [(1, )]
    '''
    if not isinstance(x, list):
        x = [x]
    
    return x

def _get_unique_names(strings):
    '''
    _get_unique_names(strings)
    
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
    _get_unique_names(['a', 'b'])               # ['a', 'b']
    _get_unique_names(['a', 'a'])               # ['a', 'a_1']
    _get_unique_names(['a', 'a', 'a_1'])        # ['a', 'a_1_1', 'a_1']
    '''
    assert _is_string_or_string_list(strings)
    strings = _enlist(strings)

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

def _get_unique_sublist(x):
    '''
    _get_unique_sublist
    Get unique elements of the list in their order of appearance
    
    Parameters
    ----------
    x : list

    Returns
    -------
    list
    
    Examples
    --------
    _get_unique_sublist([2,1,2,0,2,3])
    '''
    ell = []
    S = set([])
    
    for ele in x:
        if ele not in S:
           ell.append(ele)
           S.add(ele)
    
    return ell

def _reorder_set_by_list(set_obj, list_obj):
    '''
    _reorder_set_by_list
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
    1. If the set contains an element not present in the list, 
    then the output list will not contain the element.
    2. If the list contains an element not present in the set, 
    then the output list will not contain the element.
    
    Examples
    --------
    _reorder_set_by_list(set([1, 3, 2]), [1,2,3,1])
    '''
    
    list_obj = _get_unique_sublist(list_obj)
    new_list = []
    for ele in list_obj:
        if ele in set_obj:
            new_list.append(ele)
    return new_list

def _is_unique_list(x):
    '''
    _is_unique_list(x)
    
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

def _get_dtype_dict(pdf):
    assert isinstance(pdf, pd.DataFrame)
    
    dtf = (pd.DataFrame(pdf.dtypes)
             .reset_index(drop = False)
             .rename(columns = {'index': "column_name", 0: "dtype"})
             .assign(dtype = lambda x: x['dtype'].astype('string'))
             )
    
    return dict(zip(dtf['column_name'], dtf['dtype']))

def _generate_new_string(strings):
    
    assert isinstance(strings, list)
    assert all([isinstance(x, str) for x in strings])
    
    while True:
        random_string = "".join(np.random.choice(list(string.ascii_letters), 20))
        if random_string not in strings:
            break
    
    return random_string


def _expand_flat(pdf, column_names):
    # tuple <-> nesting
    if isinstance(column_names, tuple):
        res = (pdf.loc[:, column_names]
                  .drop_duplicates(ignore_index = True)
                  .copy()
                  )
    # list <-> crossing
    else:
        res = functools.reduce(lambda x, y: pd.merge(x, y, how = "cross")
                               , [pdf.loc[:, (x)].drop_duplicates() for x in column_names]
                               )
    return res                           
        
def simplify(pdf
             , sep = "__"
             , verbose = False
             ):
    '''
    simplify(pdf)
    
    Returns a pandas dataframe with simplified index structure.
    This might be helpful before creating a TidyPandasDataFrame object.
    
    Parameters
    ----------
    pdf : Pandas dataframe
    sep: str (default: "__")
        String separator to be used while concatenating column multiindex
    verbose: bool (default: False)
        Whether to print the progress of simpliying process
    
    Returns
    -------
    A pandas dataframe with simplified index structure
    
    Notes
    -----
    Returns a `simple` pandas dataframe. A pandas dataframe is 'simple' if:
        
        1. Column names (x.columns) are an unnamed pd.Index object of unique 
           strings. Column names do not start from "_".
        2. Row index is absent (pandas rangeindex starting from 1)
    
    This is done by collapsing the column MultiIndex by concatenating names 
    using separator 'sep' and ensuring that the resulting names are unique. 
    The row Index or MultiIndex are added to the dataframe as columns if their 
    names do not clash with the existing column names of the dataframe. Row
    indexes without a name are dropped.
    
    Additionally, 
        1. string columns stored as object are converted to string dtype
        via 'convert_dtypes' method.
        2. All missing values are replaced by pandas NA type.
    
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
    
    assert isinstance(pdf, pd.DataFrame),\
        "input should be a pandas dataframe"
    assert isinstance(sep, str),\
        "arg 'sep' should be a string"
    pdf = pdf.copy()
    
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
            pdf.columns = _get_unique_names(cns)
        else:
            if verbose:
                print("Detected a simple column index")
            # avoid column index from having some name
            pdf.columns.name = None
            pdf.columns = _get_unique_names(list(pdf.columns))
    except:
        if verbose:
            raise Exception("Unable to simplify: column index or multiindex")
    
    # handle row index 
    try:
        # if multiindex is present at row level
        # remove indexes without a name
        # write only non-existing columns
        if isinstance(pdf.index, pd.MultiIndex):
            if verbose:
                print("Detected a row multiindex")
            
            # numbers in place of None
            index_frame = pdf.index.to_frame().reset_index(drop = True)
            f_cn = list(index_frame.columns)
            is_str = [isinstance(x, str) for x in f_cn]
            index_frame = index_frame.loc[:, list(np.array(f_cn)[is_str])]
            left = list(set(index_frame.columns).difference(list(pdf.columns)))
            if len(left) > 0:
                pdf = pd.concat([index_frame.loc[:, left]
                                 , pdf.reset_index(drop = True)
                                 ]
                                , axis = "columns"
                                )
            else:
                pdf = pdf.reset_index(drop = True)
            if verbose and len(left) < len(index_frame.columns):
                print("Dropped some row index")
        else:
            # handle simple row index
            if verbose:
                print("Detected a simple row index")
            if pdf.index.name is None:
                pdf = pdf.reset_index(drop = True)
                if verbose:
                    print("Dropped some row index")
            else:
                if pdf.index.name in list(pdf.columns):
                    pdf = pdf.reset_index(drop = True)
                    print("Dropped some row index")
                else:
                    pdf = pdf.reset_index(drop = False)
    except:
        if verbose:
            raise Exception("Unable to simplify: row index or multiindex")
    
    # ensure column names are strings and unique
    pdf.columns = _get_unique_names(list(map(str, pdf.columns)))
    
    # column names should not start from an underscore
    if any([acol[0] == "_" for acol in list(pdf.columns)]):
        raise Exception("Unable to simplify as one of the columns starts from " 
                        "an underscore"
                        )
            
    if verbose:
        print("Successfully simplified!")
    
    # simplify dtypes and maintain standard NAs
    pdf = pdf.convert_dtypes().fillna(pd.NA)
    return pdf

def is_simple(pdf, verbose = False):
    '''
    is_simple
    Whether the input pandas dataframe is 'simple' or not

    Parameters
    ----------
    pdf : pandas dataframe
    verbose : bool, (default: False)
        When True, prints the reason(s) why the input is not 'simple'.
    
    Notes
    -----
    A pandas (ungrouped or grouped) dataframe is 'simple' if:
    
        1. Column names (x.columns) are an unnamed pd.Index object of unique 
           strings. Column names do not start from "_".
        2. Row index is absent (pandas rangeindex starting from 0).
        
    Returns
    -------
    bool
        True if the input is 'simple' and False if the input is not 'simple'.
    
    Examples
    --------
    from palmerpenguins import load_penguins
    penguins = load_penguins().convert_dtypes()
    ex1 = penguins.groupby('species').apply(lambda x: x.head(2))
    ex1
    
    is_simple(ex1, verbose = True)
    '''
    assert isinstance(pdf, pd.DataFrame),\
        "input should be a pandas dataframe"
    
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
    
    if _is_unique_list(columns):
        unique_flag = True
    else:
        unique_flag = False
    
    underscore_flag = False    
    if not any([acol[0] == "_" for acol in columns]):
        underscore_flag = True
    
    flag = all([row_flag, col_flag, flag_no_index, str_flag
                , unique_flag, underscore_flag])
    
    if verbose and not flag:
        if not row_flag:
            print("Row index should not be a MultiIndex object.")
        if not col_flag:
            print("Column index should not be a MultiIndex object.")
        if not flag_no_index:
            print("Row index should be a RangeIndex (0-n)")
        if not str_flag:
            print("Column index should be string column names.")
        if not unique_flag:
            print("Column names should be unique.")
        if not unique_flag:
            print("Column names should not start with an underscore.")
            
    return flag

# shorthand to convert a non-simple pandas dataframe to a TidyPandasDataFrame
def tidy(pdf
         , sep = "__"
         , verbose = False
         , copy = True
         , **kwargs
         ):
    '''
    tidy
    Utility function to convert a not simple pandas dataframe to a 
    TidyPandasDataFrame
    
    Parameters
    ----------
    pdf : Pandas dataframe
    sep: str (default: "__")
        String separator to be used while concatenating column multiindex
    verbose: bool (default: False)
        Whether to print the progress of simpliying process
    copy: bool, Default is True
            Whether the TidyPandasDataFrame object to be created should refer 
            to a copy of the input pandas dataframe or the input itself
    kwargs: Optinal arguments for TidyPandasDataFrame init method
    
    Notes
    -----
    Input is attempted to be simplified using 'simplify'. Some notorious 
    pandas dataframes may not be simplified, in which case an exception is
    raised.
    
    Examples
    --------
    from palmerpenguins import load_penguins
    penguins = load_penguins().convert_dtypes()
    ex1 = penguins.groupby('species').apply(lambda x: x.head(2))
    ex1
    
    tidy(ex1)
    '''
    res = TidyPandasDataFrame(simplify(pdf
                                       , sep = sep
                                       , verbose = verbose
                                       )
                              , copy = copy
                              , check = False
                              , **kwargs
                              )
    return res

# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import copy as util_copy
import numpy as np
import pandas as pd
import warnings
import re
import functools
from collections_extended import setlist
from skimpy import skim
import string as string
from collections import namedtuple

class TidyPandasDataFrame:
    '''
    TidyPandasDataFrame class
    A tidy pandas dataframe is a wrapper over 'simple' ungrouped pandas 
    DataFrame object.
    
    Notes
    -----
    
    A pandas dataframe is said to be 'simple' if:
    
    1. Column names (x.columns) are an unnamed pd.Index object of unique 
       strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0
       and step = 1.
    
    * Methods constitute a grammar of data manipulation mostly returning a 
      tidy dataframe as a result. 
    * When a method returns a tidy dataframe, it returns a copy and not 
      a view, unless copy is set to False.
    * Methods 'to_pandas' and 'to_series' convert into pandas dataframe or 
      series.
    * The only attribute of the class is the underlying pandas dataframe. This 
      cannot be accessed by the user. Please use 'to_pandas' method to obtain 
      a copy of the underlying pandas dataframe.
    
    Attributes
    ----------
    nrow: Number of rows
    ncol: Number of columns
    shape(alias dim): tuple of number of rows and number of columns
    colnames: list of column names
    
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
            Returns a dataframe by dropping rows which have mssing values in
            selected columns
        relapce_na
            Returns a dataframe by replacing missing values in selected columns
        fill_na
            Returns a dataframe by filling missing values from up, down or 
            both directions for selected columns
    'string' methods:
        separate
            Returns a dataframe by splitting a string column into multiple 
            columns
        unite
            Returns a dataframe by combining multiple string columns 
        separate_rows
            Returns a dataframe by exploding a string column
    'completion' methods:
        expand
            Returns a dataframe with combinations of columns
        complete
            Returns a dataframe by creating additional rows with some 
            comninations of columns 
        slice extensions:
    
    'slice' extensions:
        slice_head
            Returns a dataframe with top few rows of the input
        slice_tail
            Returns a dataframe with last few rows of the input
        slice_max
            Returns a dataframe with few rows corrsponding to maximum values of
            some columns
        slice_min
            Returns a dataframe with few rows corrsponding to maximum values of
            some columns
        slice_sample
            Returns a dataframe with a sample of rows
    
    'join' methods:
        join, inner_join, outer_join, left_join, right_join, anti_join:
            Returns a joined dataframe of a pair of dataframes
    'set operations' methods:
        union, intersection, setdiff:
            Returns a dataframe after set like operations over a pair of 
            dataframes
    'bind' methods:
        rbind, cbind:
            Returns a dataframe by rowwise or column wise binding of a pair of 
            dataframes
    
    'misc' methods:
        add_rowid:
            Returns a dataframe with rowids added
    
    '''
    
    ##########################################################################
    # init
    ##########################################################################
    
    def __init__(self, x, check = True, copy = True):
        '''
        init
        Create tidy dataframe from a 'simple' ungrouped pandas dataframe

        Parameters
        ----------
        x : 'simple' pandas dataframe
        check : bool, Default is True
            Whether to check if the input pandas dataframe is 'simple'
        copy: bool, Default is True
            Whether the TidyPandasDataFrame object to be created should refer 
            to a copy of the input pandas dataframe or the input itself

        Notes
        -----
        1. A pandas dataframe is said to be 'simple' if:
            a. Column names (x.columns) are an unnamed pd.Index object of 
               unique strings.
            b. Row names (x.index) are an unnamed pd.RangeIndex object with 
               start = 0 and step = 1.
        2. Unless you are sure, in general usage:
            a. set arg 'check' to `True`.
            b. set arg `copy` to `True`.
        
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = TidyPandasDataFrame(penguins)
        penguins_tidy
        '''
        assert isinstance(check, bool),\
            "arg 'check' should be a bool"
        if check:
            flag_simple = is_simple(x, verbose = True)
            if not flag_simple:    
            # raise the error After informative warnings
                raise Exception(("Input pandas dataframe is not 'simple'."
                                 " See to above warnings."
                                 " Try the 'simplify' function."
                                 " ex: simplify(not simple pandas dataframe) "
                                 " --> simple pandas dataframe."
                                ))
        
        if copy:                       
            self.__data = (x.copy()
                            .convert_dtypes()
                            .fillna(pd.NA)
                            )
        else:
            self.__data = x
        return None
    
    ##########################################################################
    # repr
    ##########################################################################
    def __repr__(self):
        
        nr = self.get_nrow()
        nc = self.get_ncol()
        
        header_line   = f"# A tidy dataframe: {nr} X {nc}"
        head_10 = self.__data.head(10)
        # dtype_dict = _get_dtype_dict(self.__data)
        # for akey in dtype_dict:
        #     dtype_dict[akey] = akey +  " (" + dtype_dict[akey] + ")"
        # 
        # head_10 = self.__data.head(10).rename(columns = dtype_dict)
        pandas_str = head_10.__str__()

        left_over = nr - 10
        if left_over > 0:
            leftover_str = f"# ... with {left_over} more rows"

            tidy_string = (header_line +
                           '\n' +
                           pandas_str +
                           '\n' +
                           leftover_str
                           )
        else:
            tidy_string = (header_line +
                           '\n' +
                           pandas_str
                           )

        return tidy_string
    
    ##########################################################################
    # to_pandas methods
    ##########################################################################
    
    def to_pandas(self, copy = True):
        '''
        to_pandas
        Return copy of underlying pandas dataframe
        
        Parameters
        ----------
        copy: bool, default is True
            Whether to return a copy of pandas dataframe held by
            TidyPandasDataFrame object or to return the underlying
            pandas dataframe itself
        
        Returns
        -------
        pandas dataframe
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy.to_pandas()
        
        penguins.equals(penguins_tidy.to_pandas())
        '''
        assert isinstance(copy, bool),\
            "arg 'copy' should be a bool"
        if copy:
            res = self.__data.copy()
        else:
            res = self.__data
            
        return res
    
    ##########################################################################
    # to_series or pull
    ##########################################################################
    def pull(self, column_name, copy = True):
        '''
        pull (aka to_series)
        Returns a copy of column as pandas series
        
        Parameters
        ----------
        column_name : str
            Name of the column to be returned as pandas series
        copy: bool, default is True
            Whether to return a copy of the pandas series object

        Returns
        -------
        pandas series
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        penguins_tidy.pull("species")
        '''
        
        assert isinstance(column_name, str),\
            "arg 'column_name'should be a string"
        assert column_name in list(self.__data.columns), \
            "arg 'column_name' should be an existing column name"
        assert isinstance(copy, bool),\
            "arg 'copy' should be a bool"
        
        if copy:
            res = self.__data[column_name].copy()
        else:
            res = self.__data[column_name]
        return res
    
    to_series = pull
    
    ##########################################################################
    # pipe methods
    ##########################################################################
    
    # pipe
    def pipe(self, func, *args, **kwargs):
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
        assert callable(func),\
            "arg 'func' should be callable"
        
        return func(self, *args, **kwargs)
    
    # pipe_tee
    def pipe_tee(self, func, *args, **kwargs):
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
        assert callable(func),\
            "arg 'func' should be callable"
            
        func(self, *args, **kwargs) # side-effect
        
        return self
    
    
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
    
    @property
    def nrow(self):
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
    
    @property
    def ncol(self):
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
    
    @property
    def shape(self):
        return self.__data.shape
      
    @property
    def dim(self):
        return self.__data.shape
        
    def get_colnames(self):
        '''
        get_colnames
        Get the column names of the dataframe

        Returns
        -------
        list
            List of unique strings that form the column index of the underlying 
            pandas dataframe

        '''
        return list(self.__data.columns)
    
    @property
    def colnames(self):
        return list(self.__data.columns)
    
    ##########################################################################
    # summarizers -- skim, glimpse
    ##########################################################################
    
    def skim(self, return_self = False):
        '''
        Skim the tidy dataframe
        Provides a meaningful summary of the dataframe
        yo
        Parameters
        ----------
        return_self: bool (default is False)
            Whether to return the input
            This is helpful while piping the input to next method while skim is
            a side-effect
            
        Returns
        -------
        None when return_self is False, input otherwise
        
        Notes
        -----
        1. skim is done via 'skimpy' package is expected to be installed.
        '''
        # TODO: Implement package check
        
        skim(self.__data)
        
        if return_self:
            return self
        else:
            return None
        
        
    def glimpse(self, return_self = False):
        '''
        Glimpse the tidy dataframe
        Provides a transposed view of the dataframe helpful when there are 
        large number of columns
        
        Parameters
        ----------
        return_self: bool (default is False)
            Whether to return the input
            This is helpful while piping the input to next method while glimpse
            is a side-effect
            
        Returns
        -------
        None when return_self is False, input otherwise
        
        '''
        nr = self.get_nrow()
        nc = self.get_ncol()
        
        dtype_dict = _get_dtype_dict(self.to_pandas(copy = False))
        for akey in dtype_dict:
            dtype_dict[akey] = akey + " (" + dtype_dict[akey] + ")"
        
        head_nr = int(np.minimum(nr, 6))
        
        tr = self.head(head_nr).to_pandas().rename(columns = dtype_dict).T
        tr.columns = [""] * len(tr.columns)
        tr = tr.__str__()
        
        print(f"Rows: {nr}")  
        print(f"Columns: {nc}")
        print(tr)
        
        if return_self:
            return self
        else:
            return None
    
    ##########################################################################
    # validators
    ##########################################################################
    
    def _validate_by(self, by):
        
        assert _is_string_or_string_list(by),\
            "arg 'by' should be a string or a list of strings"
            
        by = _enlist(by)
        
        assert len(set(by)) == len(by),\
            "arg 'by' should have unique strings"
        
        assert set(self.get_colnames()).issuperset(by),\
            "arg 'by' should contain valid column names"
            
        return None
    
    def _validate_column_names(self, column_names):
        
        assert _is_string_or_string_list(column_names),\
            "arg 'column_names' should be a string or a list of strings"
            
        column_names = _enlist(column_names)
        
        assert len(set(column_names)) == len(column_names),\
            "arg 'column_names' should have unique strings"
        
        assert set(self.get_colnames()).issuperset(column_names),\
            "arg 'column_names' should contain valid column names"
        
        return None
      
    def _validate_order_by(self, order_by):
        
        assert _is_string_or_string_list(order_by),\
            "arg 'order_by' should be a string or a list of strings"
            
        order_by = _enlist(order_by)
        
        assert len(set(order_by)) == len(order_by),\
            "arg 'order_by' should have unique strings"
        
        assert set(self.get_colnames()).issuperset(order_by),\
            "arg 'order_by' should contain valid column names"
            
        return None
    
    ##########################################################################
    # add_row_number
    ##########################################################################
    def add_row_number(self, name = 'row_number', by = None):
        '''
        add_row_number (aka rowid_to_column)
        Add a row number column to TidyPandasDataFrame
        
        Parameters
        ----------
        name : str
            Name for the row number column
        by : str or list of strings
            Columns to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Row order is preserved.
        2. Column indicating row number is added as the first column 
           (to the left).
        3. Alias: rowid_to_column
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)

        penguins_tidy.add_row_number() # equivalently penguins_tidy.add_rowid()
        
        # add row number per group in the order of appearance
        penguins_tidy.add_row_number(by = 'sex')
        '''
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        assert isinstance(name, str),\
            "arg 'name' should be a string"
        if name[0] == "_":
            raise Exception("'name' should not start with an underscore")
        
        if name in cn:
            raise Expection("'name' should not be an existing column name.")
            
        if by is None:
            res = self.__data.assign(**{name : np.arange(nr)})
        else:
            self._validate_by(by)
            by = _enlist(by)
            res = (self
                   .__data
                   .assign(**{"_rn" : np.arange(nr)})
                   .groupby(by, sort = False, dropna = False)
                   .apply(lambda x: x.assign(**{name : np.arange(x.shape[0])}))
                   .reset_index(drop = True)
                   .sort_values("_rn", ignore_index = True)
                   .drop(columns = "_rn")
                   )
        
        col_order = [name] + cn
        
        return TidyPandasDataFrame(res.loc[:, col_order]
                                   , check = False
                                   , copy = False
                                   )
    
    rowid_to_column = add_row_number
    
    ##########################################################################
    # add_group_number
    ##########################################################################
    def add_group_number(self, name = 'group_number', by = None):
        '''
        add_group_number
        Add a group number column to TidyPandasDataFrame
        
        Parameters
        ----------
        name : str
            Name for the group number column
        by : str or list of strings
            Columns to group by
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Row order is preserved.
        2. Column indicating group number is added as the first column 
           (to the left).
        3. Number for the group is based on the first appearance of a
           group combination.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)

        penguins_tidy.add_group_number(by = 'sex')
        '''
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        assert isinstance(name, str),\
            "arg 'name' should be a string"
        if name[0] == "_":
            raise Exception("'name' should not start with an underscore")
        
        if name in cn:
            raise Expection("'name' should not be an existing column name.")
          
        self._validate_by(by)
        by = _enlist(by)
            
        group_order_frame = (
               self
               .__data
               .loc[:, by]
               .drop_duplicates()
               .assign(**{name: lambda x: np.arange(x.shape[0])})
               )
        
        row_name = _generate_new_string(cn + [name])
        
        res = (self.__data
                   .assign(**{row_name: lambda x: np.arange(x.shape[0])})
                   .merge(group_order_frame
                          , on = by
                          , how = 'left'
                          )
                   .sort_values(by = row_name)
                   .loc[:, [name] + cn]
                   .reset_index(drop = True)
                   )
        
        return TidyPandasDataFrame(res, check = False, copy = False)

    ##########################################################################
    # group_modify
    ##########################################################################
    def group_modify(self
                     , func
                     , by
                     , preserve_row_order = False
                     , row_order_column_name = "rowid_temp"
                     , is_pandas_udf = False
                     , **kwargs
                    ):
        '''
        group_modify
        Split by some columns, apply a function per chunk which returns a 
        dataframe and combine it back into a single dataframe
        
        Parameters
        ----------
        func: callable
            Type 1. A function: TidyDataFrame --> TidyDataFrame
            Type 2. A function: simple pandas df --> simple pandas df
            In latter case, set 'is_pandas_udf' to True
        
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
        
        Notes
        -----
        1. Chunks will always include the grouping columns.
        2. If grouping columns are found in output of 'func', then they are
           removed and replaced with value in input chunk.
        3. When 'preserve_row_order' is True, a temporary column is added
           to each chunk. udf should pass it through without modification.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins_tidy = tidy(load_penguins().convert_dtypes())
        
        # pick a sample of rows per chunk defined by 'species'
        penguins_tidy.group_modify(lambda x: x.sample(n = 3)
                                    , by = 'species'
                                    )
        
        # apply a pandas udf per chunk defined by 'species'
        # groupby columns are always added to the left
        penguins_tidy.group_modify(lambda x: x.select('year')
                                    , by = ['species', 'sex']
                                    )
                                        
        # preserve row order
        # a temporary column (default: 'rowid_temp') is added to each chunk
        # udf should not meddle with the temporary column
        (penguins_tidy
            .select('sex')
            # add 'row_number' column to illustrate row preservation
            .add_row_number()
            # print the output of each chunk
            # sample 2 rows
            .group_modify(lambda x: x.pipe_tee(print).sample(2)
                          , by = 'sex'
                          , preserve_row_order = True
                          )
          )
          
        # use kwargs
        penguins_tidy.group_modify(lambda x, **kwargs: x.sample(n = kwargs['size'])
                                    , by = 'species'
                                    , size = 3
                                    )
        '''
        self._validate_by(by)
        by = _enlist(by)
        cn = self.get_colnames()
        nr = self.get_nrow()
        
        assert callable(func),\
            "'func' should be a function: tidy df --> tidy df"
        assert isinstance(preserve_row_order, bool),\
            "arg 'preserve_row_order' should be a bool"
        assert isinstance(row_order_column_name, str),\
            "arg 'row_order_column_name' should be a string"
        assert row_order_column_name not in self.get_colnames(),\
            "arg 'row_order_column_name' should not be a existing column name"
        
        group_cn = _generate_new_string(cn)
        
        group_id_frame = self.__data.loc[:, by].drop_duplicates()
        group_id_frame[group_cn] = np.arange(group_id_frame.shape[0])
      
        # create wrapper function
        if is_pandas_udf:
            def wrapper_func(chunk, **kwargs):
                res = func(chunk.drop(columns = group_cn).reset_index(drop = True)
                           , **kwargs
                           )
                by_left = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res = res.drop(columns = by_left)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                
                cols_in_order = util_copy.copy(by)
                cols_in_order.extend(set(res.columns).difference(by))
                return res.loc[:, cols_in_order]
        else:
            def wrapper_func(chunk, **kwargs):
                # i/o are pdfs
                chunk_tidy = (TidyPandasDataFrame(chunk.reset_index(drop = True)
                                                 , check = False
                                                 , copy = False
                                                 )
                              .select(group_cn, include = False)
                              )
                res        = func(chunk_tidy, **kwargs).to_pandas(copy = False)
                by_left    = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res = res.drop(columns = by_left)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                
                cols_in_order = util_copy.copy(by)
                cols_in_order.extend(set(res.columns).difference(by))
                return res.loc[:, cols_in_order] 
        
        if preserve_row_order:
            
            assert row_order_column_name not in cn,\
                ("arg 'row_order_column_name' should not be an exisiting column "
                 "name"
                 )
            
            res = (self.__data
                        .assign(**{row_order_column_name: np.arange(nr)})
                        .merge(group_id_frame, on = by)
                        .groupby(group_cn, sort = False, dropna = False)
                        .apply(wrapper_func, **kwargs)
                        )
            try:
                res = (res.pipe(simplify)
                          .drop(columns = group_cn)
                          )
            except:
                raise Exception(("Result of apply is too complex to be simplified "
                                 "by `simplify` function"
                                 )
                                )
            
            if row_order_column_name in list(res.columns):
                res = (res.sort_values(row_order_column_name
                                       , ignore_index = True
                                       )
                          .drop(columns = row_order_column_name)
                          )
            else:
                raise Exception(
                    ("'row_order_column_name' in each chunk should "
                     "be retained, when 'preserve_row_order' is True"
                     ))
        else: # do not preserve row order
            res = (self.__data
                       .merge(group_id_frame, on = by)
                       .groupby(group_cn, sort = False, dropna = False)
                       .apply(wrapper_func, **kwargs)
                       )
            try:
                res = (res.pipe(simplify)
                          .drop(columns = group_cn)
                          )
            except:
                raise Exception(("Result of apply is too complex to be simplified "
                                 "by `simplify` function"))
        
        return TidyPandasDataFrame(res
                                   , check = False
                                   , copy = False
                                   )
    
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
        column_names : str or a list of strings
            list of column names to be selected
        predicate : callable, series --> bool
            function which returns a bool
        include : bool, default is True
            When True, column_names are selected, else dropped

        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Select works by either specifying column names or a predicate,
           not both.
        2. When predicate is used, predicate should accept a pandas series and
           return a bool. Each column is passed to the predicate and the result
           indicates whether the column should be selected or not.
        
        Examples
        --------
        import re
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        # select with names
        penguins_tidy.select(['sex', 'species'])
        
        # select using a predicate: only non-numeric columns
        penguins_tidy.select(predicate = lambda x: x.dtype != "string")
        
        # select columns ending with 'mm'
        penguins_tidy.select(
            predicate = lambda x: bool(re.match(".*mm$", x.name))
            )
        
        # invert the selection
        penguins_tidy.select(['sex', 'species'], include = False)
        '''
        
        cn = self.get_colnames()
        
        if (column_names is None) and (predicate is None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should not be None")
                            )
        if (column_names is not None) and (predicate is not None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should not be None"))
        
        if column_names is None:
            assert callable(predicate),\
                "arg 'predicate'(series --> bool) should be a function"
            col_bool_list = np.array(list(self.__data.apply(predicate
                                                            , axis = "index"
                                                            )
                                         )
                                    )
            column_names = list(np.array(cn)[col_bool_list])
        else:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        
        if not include:
            column_names = list(setlist(cn).difference(column_names))
        
        if len(column_names) == 0:
            raise Exception("At least one column should be selected")
            
        res = self.__data.loc[:, column_names]
        # return a copy due to loc
        return TidyPandasDataFrame(res, check = False, copy = True)
    
    ##########################################################################
    # relocate
    ##########################################################################
    
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
        Only one among 'before' and 'after' can be not None. When both are None,
        the columns are added to the begining of the dataframe (leftmost)
        
        Returns
        -------
        TidyPandasDataFrame
        
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
        
        self._validate_column_names(column_names) 
        column_names = _enlist(column_names)
        cn = self.get_colnames()
         
        assert not ((before is not None) and (after is not None)),\
            "Atleast one arg among 'before' and 'after' should be None"
            
        if after is not None:
            assert isinstance(after, str),\
                "arg 'after' should be a string"
            assert after in cn,\
                "arg 'after' should be a exisiting column name"
            assert not (after in column_names),\
                "arg 'after' should be an element of 'column_names'"
        
        if before is not None:
            assert isinstance(before, str),\
                "arg 'before' should be a string"
            assert before in cn,\
                "arg 'before' should be a exisiting column name"
            assert not (before in column_names),\
                "arg 'before' should be an element of 'column_names'"
        
        cc_setlist       = setlist(cn)
        cc_trunc_setlist = cc_setlist.difference(column_names)
            
        # case 1: relocate to start when both before and after are None
        if (before is None) and (after is None):
            new_colnames = column_names + list(cc_trunc_setlist)
        elif (before is not None):
            # case 2: before is not None
            pos_before   = np.where(np.array(cc_trunc_setlist) == before)
            pos_before   = int(pos_before[0])
            cc_left      = list(cc_trunc_setlist[:pos_before])
            cc_right     = list(cc_trunc_setlist[pos_before:])
            new_colnames = cc_left + column_names + cc_right
        else:
            # case 3: after is not None
            pos_after   = np.where(np.array(cc_trunc_setlist) == after)
            pos_after   = int(pos_after[0])    
            cc_left      = list(cc_trunc_setlist[ :(pos_after + 1)])
            cc_right     = list(cc_trunc_setlist[(pos_after + 1): ])
            new_colnames = cc_left + column_names + cc_right
      
        res = self.__data.loc[:, new_colnames]
        # return a copy due to loc
        return TidyPandasDataFrame(res, check = False, copy = True)
    
    ##########################################################################
    # rename
    ##########################################################################
    
    def rename(self, old_new_dict):
        '''
        rename
        Rename columns of the tidy pandas dataframe
        
        Parameters
        ----------
        old_new_dict: A dict with old names as keys and new names as values
        
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins_tidy = tidy(load_penguins().convert_dtypes())
        penguins_tidy

        flights_tidy.rename({'species': 'species_2'})
        '''
        cn = self.get_colnames()
        
        assert isinstance(old_new_dict, dict),\
            "arg 'old_new_dict' should be a dict"
        assert set(cn).issuperset(old_new_dict.keys()),\
            "keys of the input dict should be existing column names"
        assert _is_string_or_string_list(list(old_new_dict.values())),\
            "values of the dict should be strings"
        assert _is_unique_list(list(old_new_dict.values())),\
            "values of the dict should be unique"
        
        # new names should not intersect with 'remaining' names
        remaining = set(cn).difference(old_new_dict.keys())
        assert len(remaining.intersection(old_new_dict.values())) == 0,\
            ("New intended column names (values of the dict) lead to duplicate "
             "column names"
             )
        
        res = self.__data.rename(columns = old_new_dict)
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    ##########################################################################
    # slice
    ##########################################################################
    
    def slice(self, row_numbers, by = None):
        '''
        slice
        Subset rows of a TidyPandasDataFrame
        
        Parameters
        ----------
        row_numbers : int or list or 1-D numpy array of positive integers
            list/array of row numbers.
        by : list of strings, optional
            Column names to groupby. The default is None.

        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Grouped slice preserves the row order.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins_tidy = tidy(load_penguins().convert_dtypes())

        # pick first three rows of the dataframe
        penguins_tidy.slice(np.arange(3))
        
        # pick these row numbers: [0, 3, 8]
        penguins_tidy.slice([0, 3, 8])
        
        # pick first three rows per specie
        penguins_tidy.slice([0,1,2], by = "species")
        
        # pick first three rows for each species and sex combination
        penguins_tidy.slice(np.arange(3), by = ["species", "sex"])
        '''
        
        if isinstance(row_numbers, int):
            row_numbers = _enlist(row_numbers)
        
        if by is None:
            minval = np.min(row_numbers)
            maxval = np.max(row_numbers)
            assert minval >= 0 and maxval <= self.get_nrow(),\
                ("row numbers to slice should be in a range the max and min "
                 "rows of the dataframe"
                 )
            
            res = (self.__data
                       .iloc[row_numbers, :]
                       .reset_index(drop = True)
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            min_group_size = (self.__data
                                  .groupby(by, sort = False, dropna = False)
                                  .size()
                                  .min()
                                  )
                                  
            if np.max(row_numbers) > min_group_size:
                print("Minimum group size is: ", min_group_size)
                raise Exception(("Maximum row number to slice per group should" 
                                 "not exceed the number of rows of the group"
                                 ))
            
            res = (self.__data
                       .assign(**{"_rn": lambda x: np.arange(x.shape[0])})
                       .groupby(by, sort = False, dropna = False)
                       .apply(lambda chunk: chunk.iloc[row_numbers,:])
                       .reset_index(drop = True)
                       .sort_values("_rn", ignore_index = True)
                       .drop(columns = "_rn")
                       )
        
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    ##########################################################################
    # arrange
    ##########################################################################
        
    def arrange(self
                , column_names
                , ascending = False
                , na_position = 'last'
                , by = None
                ):
        '''
        arrange
        Orders the rows by the values of selected columns
        
        Parameters
        ----------
        column_names : list of strings
            column names to order by.
        by: str or a list of strings
            column names to group by
        ascending : bool or a list of booleans, optional
            The default is False.
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
        
        3. If the column provided in arrange are not sufficient to order 
        the rows of the dataframe, then row number is implicitly used to
        deterministicly break ties.
        
        '''
        
        cn = self.get_colnames()
        nr = self.get_nrow()
        
        self._validate_column_names(column_names)
        column_names = _enlist(column_names)
        
        if not isinstance(ascending, list):
            assert isinstance(ascending, bool),\
                "when arg 'ascending' is not a list, it should be a bool"
        else:
            assert all([isinstance(x, bool) for x in ascending]),\
                ("when arg 'ascending' is a list, "
                 "it should be a list of booleans"
                 )
            assert len(ascending) == len(column_names),\
                ("when arg 'ascending' is a list, its length should be same "
                 "as the length of 'column_names'"
                 )
        
        if by is not None:
            self._validate_by(by)
            by = _enlist(by)
            assert len(set(by).intersection(column_names)) == 0,\
                "'by' and 'column_names' should not have common names"
        
        def pdf_sortit(x):
            res = x.sort_values(by             = column_names
                                , axis         = 0
                                , ascending    = ascending
                                , inplace      = False
                                , kind         = 'quicksort'
                                , na_position  = na_position
                                , ignore_index = True
                                )
            return res
        
        if by is None:
            res = pdf_sortit(self.__data)
        else:
            rn_org = _generate_new_string(cn)
            g_id   = _generate_new_string(cn + [rn_org])
            wgn    = _generate_new_string(cn + [rn_org, g_id])
            rn_arr = _generate_new_string(cn + [rn_org, g_id, wgn])
            
            lam = lambda x, name: x.assign(**{name: np.arange(x.shape[0])})
            
            # org: rn_org <-> g_id <-> wgn
            org_order = (self.add_row_number(name = rn_org)
                             .add_group_number(name = g_id, by = by)
                             .select([rn_org, g_id])
                             .add_row_number(name = wgn, by = g_id)
                             .to_pandas(copy = False)
                             )
                               
            # arranged: rn_arr <-> g_id <-> wgn
            arranged_order = (self.add_group_number(name = g_id, by = by)
                                  .add_row_number(name = rn_arr)
                                  .to_pandas(copy = False)
                                  .groupby(g_id, sort = False)
                                  .apply(pdf_sortit)
                                  .reset_index(drop = True)
                                  .loc[:, [g_id, rn_arr]]
                                  .groupby(g_id, sort = False)
                                  .apply(lambda x: lam(x, wgn))
                                  .reset_index(drop = True)
                                  .loc[:, [g_id, wgn, rn_arr]]
                                  )
            
            # rn_org <-> rn_arr                   
            order_df = (org_order.merge(arranged_order, on = [g_id, wgn])
                                 .loc[:, [rn_org, rn_arr]]
                                 .sort_values(rn_org)
                                 )
            
            res = (self.__data
                       .iloc[list(order_df[rn_arr]), :]
                       .reset_index(drop = True)
                       )
            
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    ##########################################################################
    # filter
    ##########################################################################
        
    def filter(self, query = None, mask = None, by = None):
        '''
        filter
        subset some rows
        
        Parameters
        ----------
        query: str or a function
            when str, query string parsable by pandas eval
            when function, should output an array of booleans
            of length equal to numbers of rows of the input dataframe
        mask: list of booleans or
              numpy array or pandas Series of booleans
              of length equal to numbers of rows of the input dataframe
        by: Optional, str or list of strings
            column names to group by 
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Exactly one arg among 'query' and 'mask' should be provided
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins_tidy = tidy(load_penguins().convert_dtypes())
        
        # query with pandas eval. pandas eval does not support complicated expressions.
        penguins_tidy.filter("body_mass_g > 4000")
        
        # subset with a mask -- list or array or pandas Series of precomputed booleans
        penguins_tidy.filter(mask = (penguins_tidy.pull("year") == 2007))
        # equivalently:
        # penguins_tidy.filter(lambda x: x.year == 2007)
        
        # use complex expressions as a lambda function and filter
        penguins_tidy.filter(lambda x: x['bill_length_mm'] > np.mean(x['bill_length_mm']))
        
        # per group filter retains the row order
        penguins_tidy.filter(lambda x: x['bill_length_mm'] > np.mean(x['bill_length_mm']), by = 'sex')
        '''
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
            by = _enlist(by)
            self._validate_by(by)
        
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
                           .assign(**{"rn__": lambda x: np.array(x.shape[0])})
                           .groupby(by, sort = False, dropna = False)
                           .apply(lambda chunk: chunk.query(query))
                           .reset_index(drop = True)
                           .sort_values("rn__", ignore_index = True)
                           .drop(columns = "rn__")
                           )
                else:
                    res = (self.__data
                           .assign(**{"rn__": lambda x: np.arange(x.shape[0])})
                           .groupby(by, sort = False, dropna = False)
                           .apply(lambda chunk: (
                               chunk.assign(**{"mask__": query})
                                    .query("mask__")
                                    .drop(columns = "mask__")
                                    )
                                 )
                           .reset_index(drop = True)
                           .sort_values("rn__", ignore_index = True)
                           .drop(columns = "rn__")
                           ) 
        
        if query is None and mask is not None:
            res = self.__data.loc[mask, :]
        
        res = res.reset_index(drop = True)     
        return TidyPandasDataFrame(res, check = False)
    
    ##########################################################################
    # distinct
    ##########################################################################
        
    def distinct(self, column_names = None, keep_all = False):
        '''
        distinct
        subset unique rows from the dataframe
        
        Parameters
        ----------
        column_names: string or a list of strings
            Names of the column for distinct
        keep_all: bool
            Whether to keep all the columns or only the 'column_names'
        by: Optional, string or a list of strings
            Column names to groupby
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. distinct preserves the order of the rows of the input dataframe.
        2. 'column_names' and 'by' should not have common column names.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        tidy_penguins = tidy(load_penguins())
        
        tidy_penguins.distinct() # distinct over all columns
        
        tidy_penguins.distinct('species') # keep only 'distinct' columns
        tidy_penguins.distinct('species', keep_all = True) # keep all columns
        
        tidy_penguins.distinct(['bill_length_mm', 'bill_depth_mm'])
        '''
        
        cn = self.get_colnames()
        if column_names is not None:
            assert _is_string_or_string_list(column_names),\
                "arg 'column_names' should be a string or a list of strings"
            column_names = list(setlist(_enlist(column_names)))
            assert set(column_names).issubset(cn),\
                "arg 'column_names' should contain valid column names"
        else:
            column_names = cn
        assert isinstance(keep_all, bool)
        
        res = (self.__data
                   .drop_duplicates(subset = column_names
                                    , keep = "first"
                                    , ignore_index = True
                                    )
                   )
        
        if not keep_all:
            res = res.loc[:, column_names]
                       
        return TidyPandasDataFrame(res, check = False)
    
    ##########################################################################
    # mutate
    ##########################################################################
    
    def _mutate(self, dictionary, order_by = None, **kwargs):
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        # strategy
        # created 'mutated' copy and keep adding/modifying columns
        
        # handle 'order_by':
        # Orders the dataframe for 'mutate' and keeps a row order column
        # at the end of mutate operation, the dataframe is sorted in original 
        # order and row order column is deleted
        if order_by is not None:
            # ex: {'col_a': True, 'col_b': False} # ascending or not
            if isinstance(order_by, dict):
                self._validate_order_by(list(order_by.keys()))
                assert all([isinstance(x, bool) for x in list(order_by.values())]),\
                    "When 'order_by' is a dict, values should be booleans"
                mutated = (self.__data
                               .copy()
                               .assign(**{"_rn": lambda x: np.arange(x.shape[0])})
                               .sort_values(by = list(order_by.keys()),
                                            ascending = list(order_by.values())
                                            )
                               )
            else:
                self._validate_order_by(order_by)
                order_by = _enlist(order_by)
                mutated = (self.__data
                               .copy()
                               .assign(**{"_rn": lambda x: np.arange(x.shape[0])})
                               .sort_values(order_by)
                               )
        else:
            mutated = self.__data.copy()
        
        keys = dictionary.keys()
        # ensure new column names to be created are valid (do not start with '_')
        keys_flattened = list(
          np.concatenate([[x] if np.isscalar(x) else list(x) for x in keys])
          )
        assert np.all([_is_valid_colname(x) for x in keys_flattened]),\
          (f"column names to be created/modified should be valid column names. "
           "A valid column name should be a string not starting from '_'"
           )
        
        for akey in dictionary:
            
            assert isinstance(akey, (str, tuple)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            assert _is_string_or_string_list(list(akey)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            
            rhs = dictionary[akey]
            
            # akey is a string or a list of strings
            # no longer a tuple
            if isinstance(akey, tuple):
                if len(akey) == 1:
                    akey = akey[0]
                else:
                    assert len(akey) == len(set(akey)),\
                        "LHS should have unique strings"
                    akey = list(akey)
         
            if isinstance(akey, str):
                # three cases:
                #   1. direct assign
                #   2. assign via function
                #   3. assign via simple function
                #       3a. assign a preexisting column
                #       3b. assign with column args
                
                # 1. direct assign
                if isinstance(rhs, ((pd.Series, np.ndarray)) 
                                    or np.isscalar(rhs)):
                    mutated[akey] = rhs
                
                # 2. assign via function -- pandas style UDF
                elif callable(rhs) or isinstance(rhs, str):
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            mutated[akey] = rhs(mutated, **kwargs)
                        else:
                            mutated[akey] = rhs(mutated)
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        mutated[akey] = rhs(mutated)
                                 
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more than "
                         "two elements"
                         )
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    # 3a. assign a preexisting column
                    if len(rhs) == 1:
                        if callable(rhs[0]):
                            assert akey in list(mutated.columns),\
                                (f"When RHS is a tuple with function alone, "
                                 "{akey} should be an exisiting column"
                                 )
                            if callable(rhs[0]):
                                if _is_kwargable(rhs[0]):
                                    mutated[akey] = rhs[0](mutated[akey], **kwargs)
                                else:
                                    mutated[akey] = rhs[0](mutated[akey])
                        else:
                            # string case
                            fun = eval("lambda x, **kwargs: " + rhs[0])
                            mutated[akey] = fun(mutated[akey], **kwargs)
                    # 3b. assign with column args
                    else:
                        assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = _enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                mutated[akey] = rhs[0](*[mutated[acol] for acol in cols], **kwargs)
                            else:
                                mutated[akey] = rhs[0](*[mutated[acol] for acol in cols])
                        else:
                            # string case
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(cols)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )
                            mutated[akey] = (
                                fun(*[mutated[acol] for acol in cols], **kwargs)
                                )
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
            
            # multiple assignments
            else:
                # 1. direct assign is not supported for multi assignment
                
                # 2. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            rhs_res = rhs(mutated, **kwargs)
                        else:
                            rhs_res = rhs(mutated)
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        rhs_res = rhs(mutated, **kwargs)
                        
                    assert (isinstance(rhs_res, list) 
                            and len(rhs_res) == len(akey)
                            ),\
                        ("RHS should output a list of length equal to "
                         "length of LHS for key: {akey}"
                         )
                         
                    for apair in zip(akey, rhs_res):
                        mutated[apair[0]] = apair[1]
                        
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more "
                         "than two elements"
                         )
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    # 3a. assign a preexisting columns
                    if len(rhs) == 1:
                        assert set(akey).issubset(list(mutated.columns)),\
                            (f"When RHS is a tuple with function alone, {akey}"
                             "should be an exisiting columns"
                             )
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                rhs_res = rhs[0](*[mutated[acol] for acol in akey]
                                                 , **kwargs
                                                 )
                            else:
                                rhs_res = rhs[0](*[mutated[acol] for acol in akey])
                        else:
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(akey)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )  
                            rhs_res = fun(*[mutated[acol] for acol in akey], **kwargs)
                            
                        assert (isinstance(rhs_res, list)
                                and len(rhs_res) == len(akey)
                                ),\
                            (f"RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            mutated[apair[0]] = apair[1]
                    # 3b. assign with column args
                    else:
                        assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = _enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                rhs_res = rhs[0](*[mutated[x] for x in cols], **kwargs)
                            else:
                                rhs_res = rhs[0](*[mutated[x] for x in cols])
                        else:
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(cols)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )  
                            rhs_res = fun(*[mutated[x] for x in cols], **kwargs)
                        
                        assert (isinstance(rhs_res, list) 
                                and len(rhs_res) == len(akey)
                                ),\
                            ("RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            mutated[apair[0]] = apair[1]
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
        
        if order_by is not None:
            mutated = mutated.sort_values("_rn").drop(columns = ["_rn"])
              
        return TidyPandasDataFrame(mutated
                                   , check = False
                                   , copy = False
                                   )

    # mutate_across    
    def _mutate_across(self
                       , func
                       , column_names = None
                       , predicate = None
                       , prefix = ""
                       , order_by = None
                       , **kwargs
                       ):

        assert callable(func) or isinstance(func, str),\
            ("arg 'func' should be a function or a string which is convertible "
             "to a lambda function"
             )
        assert isinstance(prefix, str)

        if order_by is not None:
            ro_name = _generate_new_string(cn)
            if isinstance(order_by, dict):
                self._validate_by(list(order_by.keys()))
                assert all([isinstance(x, bool) for x in list(order_by.values())]),\
                    "When 'order_by' is a dict, values should be booleans"
                mutated = (self.__data
                               .copy()
                               .assign(**{ro_name: lambda x: np.arange(x.shape[0])})
                               .sort_values(by = list(order_by.keys()),
                                            ascending = list(order_by.values())
                                            )
                               )
            else:
                self._validate_by(order_by)
                order_by = _enlist(order_by)
                mutated = (self.__data
                               .copy()
                               .assign(**{ro_name: lambda x: np.arange(x.shape[0])})
                               .sort_values(order_by)
                               )
        else:
            mutated = self.__data.copy()
        
        if (column_names is not None) and (predicate is not None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should be None")
                             )
        
        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' "
                            "should be None"
                            )
        
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
            if callable(func):
                if _is_kwargable(func):
                    mutated[prefix + acol] = func(mutated[acol], **kwargs)
                else:
                    mutated[prefix + acol] = func(mutated[acol])
            else:    
                func = eval("lambda x, **kwargs: " + func)
                mutated[prefix + acol] = func(mutated[acol], **kwargs)
        
        if order_by is not None:
            mutated = mutated.sort_values(ro_name).drop(columns = [ro_name])
            
        return TidyPandasDataFrame(mutated, check = False, copy = False)
    
    def mutate(self
               , dictionary = None
               , func = None
               , column_names = None
               , predicate = None
               , prefix = ""
               , by = None
               , order_by = None
               , **kwargs
               ):
        '''
        mutate
        Add or modify some columns
        
        Parameters
        ----------
        dictionary: dict
            A dictionary of mutating operations. This supports multiple 
            styles, see the examples
        func: callable
            A function to apply on the 'column_names' or the columns chosen by
            the 'predicate'.
        column_names: string or a list of strings
            Names of the columns to apply 'func' on
        predicate: callable
            A function to choose columns. Input: pandas series, output: bool
        prefix: string
            Prefix the resulting summarization column after applying 'func'
        by: string or a list of strings
            column names to group by
        order_by: string or a list of strings
            Names of columns to order by, before applying the mutating function
        **kwargs
            for lambda functions in dictionary or func
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. mutate works in two modes:
            a. direct assign: A series or 1-D array is assignged as a column
            b. dictionary mode: A key should be a resulting column name. Value
                should be a specification of the mutating operation. See the 
                examples for various sytles.
            c. across mode: A function is applied across a set of columns or 
                columns selected by a predicate function.
        2. mutate preserves row order.
        3. 'by'(groupby) and 'order_by' columns should not be mutated.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins_tidy = tidy(load_penguins())
        
        # mutate with dict
        penguins_tidy.mutate({
          # 1. direct assign
          'ind' : np.arange(344), 
          
          # 2. using pandas style lambda function
          "yp1": lambda x: x['year'] + 1,
          
          # 2a. pass the content of lambda function as a string
          "yp1_string": "x['year'] + 1",
          
          # 3. pass a tuple of function and column list
          "yp1_abstract": (lambda x: x + 1, "year"),
          
          # 3a. pass the tuple of function content as string and the column list
          "yp2_abstract": ("x + 1", "year"),
          
          # 4. pass multiple columns
          "x_plus_y": ("x + y", ["bill_length_mm", "bill_depth_mm"]),
          # the above is equivalent to:
          # "x_plus_y": (lambda x, y: x + y, ["bill_length_mm", "bill_depth_mm"]),
          
          # 5. output multiple columns as a list
          ('x', 'y'): lambda x: [x['year'] - 1, x['year'] + 1],
          # the above is equivalent to:
          # ('x2', 'y2'): "[x['year'] - 1, x['year'] + 1]",
          
          # change an existing column: add one to 'bill_length_mm'
          'bill_length_mm': ("x + 1", ),
          # the above is equivalent to these:
          # 'bill_length_mm': ("x + 1", 'bill_length_mm'),
          # 'bill_length_mm': (lambda x: x + 1, 'bill_length_mm'),
          
          # use a mutated column for the subsequent key
          'ind_minus_1': ("x - 1", 'ind')
          })
            
        # mutate with dict and groupby    
        penguins_tidy.mutate({'x' : "x['year'] + np.mean(x['year']) - 4015"}
                             , by = 'sex'
                             )
        
        # mutate can use columns created in the dict before
        (penguins_tidy.select('year')
                      .mutate({'yp1': ("x + 1", 'year'),
                               'yp1m1': ("x - 1", 'yp1')
                              })
                      )
                                
        # use kwargs
        (penguins_tidy
         .select('year')
         .mutate({'yp1': ("x + kwargs['akwarg']", 'year')}, akwarg = 10)
         )
        
        # 'order_by' some column before the mutate opeation
        # order_by column 'bill_length_mm' before computing cumsum over 'year' columns
        # row order is preserved
        cumsum_df = (penguins_tidy.select(['year', 'species', 'bill_length_mm'])
                                  .mutate({'year_cumsum': (np.cumsum, 'year')},
                                          order_by = 'bill_length_mm'
                                          )
                                  )
        cumsum_df
        # confirm the computation:
        # pick five rows ordered by 'bill_length_mm' (asc order)
        cumsum_df.slice_min(n = 5,
                            order_by = 'bill_length_mm',
                            with_ties = False
                            )
        
        # across mode with column names
        (penguins_tidy.select(['bill_length_mm', 'body_mass_g'])
                      .mutate(column_names = ['bill_length_mm', 'body_mass_g']
                              , func = lambda x: x - np.mean(x)
                              , prefix = "demean_"
                              )
                      )
                      
        # grouped across with column names
        (penguins_tidy.select(['bill_length_mm', 'body_mass_g', 'species'])
                      .mutate(column_names = ['bill_length_mm', 'body_mass_g'],
                              func = lambda x: x - np.mean(x),
                              prefix = "demean_",
                              by = 'species'
                              )
                      )
          
        # across mode with predicate
        penguins_tidy.mutate(func = lambda x: x - np.mean(x),
                             predicate = dtypes.is_numeric_dtype,
                             prefix = "demean_"
                             )
          
        # grouped across with predicate without prefix
        # this will return a copy with columns changed without changing names
        penguins_tidy.mutate(func = lambda x: x - np.mean(x),
                             predicate = dtypes.is_numeric_dtype,
                             by = 'species'
                             )
        '''
        if dictionary is None and func is None:
            raise Exception(("Either dictionary or func with "
                             "predicate/column_names should be provided."
                            ))
        if by is None:
            if dictionary is not None:
                res = self._mutate(dictionary, order_by = order_by, **kwargs)
            else:
                res = self._mutate_across(func
                                          , column_names = column_names
                                          , predicate = predicate
                                          , prefix = prefix
                                          , order_by = order_by
                                          , **kwargs
                                          )
        else:
            self._validate_by(by)
            by = _enlist(by)
            cn = self.get_colnames()
            
            if dictionary is not None:
                
                keys = dictionary.keys()
                keys_flattened = list(
                  np.concatenate(
                    [[x] if np.isscalar(x) else list(x) for x in keys]
                    )
                  )
                # keys should not intersect with 'by' columns
                assert len(set(by).intersection(keys_flattened)) == 0,\
                    ("column names to be created by 'mutate' should not "
                     "intersect with 'by' columns"
                     )
                res = self.group_modify(
                    func = lambda chunk: chunk._mutate(dictionary, order_by = order_by)
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = _generate_new_string(cn)
                    , is_pandas_udf = False
                    , **kwargs
                    )
                # set the column order
                # original columns in old order
                # new column come next in dict order
                col_order = cn + list(setlist(keys_flattened).difference(cn))
                res = res.select(col_order)
            else:
                res = self.group_modify(
                    func = lambda chunk: (
                        chunk._mutate_across(func
                                             , column_names = column_names
                                             , predicate = predicate
                                             , prefix = prefix
                                             , order_by = order_by
                                             )
                        )
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = _generate_new_string(cn)
                    , is_pandas_udf = False
                    , **kwargs
                    )
                # set the column order
                col_order = res.get_colnames()
                col_order = cn + list(setlist(col_order).difference(cn))
                res = res.select(col_order)

        return res
    
    
    ##########################################################################
    # summarise
    ##########################################################################
 
    def _summarise(self, dictionary, **kwargs):
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        summary_dict = {}
        
        def _validate_rhs_val(akey, rhs_val):
            if not pd.isna(rhs_val):
                if not np.isscalar(rhs_val):
                    if not (len(rhs_val) == 1 and np.isscalar(rhs_val[0])):
                        raise Exception((f"Summarised value for key {akey} does not"
                                         " turn out to be a scalar or cannot be "
                                         "converted to a scalar")
                                        )
            return None
        
        for akey in dictionary:
            
            assert isinstance(akey, (str, tuple)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            assert _is_string_or_string_list(list(akey)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            
            rhs = dictionary[akey]
            
            # akey is a string or a list of strings
            # no longer a tuple
            if isinstance(akey, tuple):
                if len(akey) == 1:
                    akey = akey[0]
                else:
                    assert len(akey) == len(set(akey)),\
                        "LHS should have unique strings"
                    akey = list(akey)
         
            if isinstance(akey, str):
                # two cases:
                #   1. assign via function
                #   2. assign with column args
                
                # 1. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            rhs_val = rhs(self.__data, **kwargs)
                        else:
                            rhs_val = rhs(self.__data)
                            
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        rhs_val = rhs(self.__data, **kwargs)
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                                 
                # 2. assign via simple function
                elif isinstance(rhs, tuple):
                    if len(rhs) == 1:
                        rhs = (rhs[0], akey)
                    assert len(rhs) == 2,\
                        f"When RHS is a tuple, RHS should have two elements"
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS is a tuple, the second "
                             "element should be a string or a list of strings"
                             )
                    cols = _enlist(rhs[1])
                    assert set(cols).issubset(cn),\
                        f"RHS should contain valid columns for LHS '{akey}'"
                    
                    if callable(rhs[0]):
                        if _is_kwargable(rhs[0]):
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols]
                                             , **kwargs
                                             )
                        else:
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols])
                            
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                    else:
                        # string case
                        chars = list(string.ascii_lowercase[-3:] 
                                     + string.ascii_lowercase[:-3]
                                     )[:len(cols)]
                        fun = eval(("lambda " 
                                    + ", ".join(chars) 
                                    + ", **kwargs: " 
                                    + rhs[0]
                                    )
                                   )
                      
                        rhs_val = fun(*[self.__data[acol] for acol in cols], **kwargs)
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
            
            # multiple assignments
            else:
                # 1. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    if not callable(rhs):
                        rhs = eval("lambda x, **kwargs: " + rhs)
                    
                    if _is_kwargable(rhs):
                        rhs_val = rhs(self.__data, **kwargs)
                    else:
                        rhs_val = rhs(self.__data)
                        
                    [_validate_rhs_val(akey, x) for x in rhs_val]
                        
                    assert (isinstance(rhs_val, list) 
                            and len(rhs_val) == len(akey)
                            ),\
                        (f"RHS should output a list of length equal to "
                         "length of LHS for key: {akey}"
                         )
                         
                    for i in range(len(akey)):
                      summary_dict[akey[i]] = rhs_val[i]
                        
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) == 2,\
                        f"When RHS is a tuple, RHS should have two elements"
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS is a tuple, the second "
                             "element should be a string or a list of strings"
                             )
                    cols = _enlist(rhs[1])
                    assert set(cols).issubset(cn),\
                        f"RHS should contain valid columns for LHS '{akey}'"
                    
                    if callable(rhs[0]):
                        if _is_kwargable(rhs[0]):
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols]
                                             , **kwargs
                                             )
                        else:
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols])
                    else:
                        chars = list(string.ascii_lowercase[-3:] 
                                     + string.ascii_lowercase[:-3]
                                     )[:len(cols)]
                        fun = eval(("lambda " 
                                    + ", ".join(chars) 
                                    + ", **kwargs: " 
                                    + rhs[0]
                                    )
                                   )  
                        rhs_val = fun(*[self.__data[acol] for acol in cols], **kwargs)
                        
                    [_validate_rhs_val(akey, x) for x in rhs_val]
                    
                    assert (isinstance(rhs_val, list) 
                            and len(rhs_val) == len(akey)
                            ),\
                        (f"RHS should output a list of length equal "
                         "to length of LHS for key: {akey}"
                         )
                         
                    for i in range(len(akey)):
                      summary_dict[akey[i]] = rhs_val[i]
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
                                     
        res = pd.DataFrame(summary_dict, index = [0])
        return TidyPandasDataFrame(res, copy = False, check = False)
  
    def summarise(self
                  , dictionary = None
                  , func = None
                  , column_names = None
                  , predicate = None
                  , prefix = ""
                  , by = None
                  , **kwargs
                  ):
        '''
        summarise
        Creates one row per group summarising the input dataframe
        
        Parameters
        ----------
        dictionary: dict
            A dictionary of summarising operations. This supports multiple 
            styles, see the examples
        func: callable
            A function to apply on the 'column_names' or the columns chosen by
            the 'predicate'. func should return a scalar.
        column_names: string or a list of strings
            Names of the columns to apply 'func' on
        predicate: callable
            A function to choose columns. Input: pandas series, output: bool
        prefix: string
            Prefix the resulting summarization column after applying 'func'
        by: string or a list of strings
            column names to group by
        **kwargs
            for lambda functions in dictionary or func
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. When by is None, summarise outputs a single row dataframe.
        2. summarise function works in two modes:
            a. dictionary mode: A key should be a resulting column name. Value
                should be a specification of the summarising operation. See the 
                examples for various sytles.
            b. across mode: A function is applied across a set of columns or 
                columns selected by a predicate function.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        # summarise in dict mode
        penguins_tidy.summarise({
            # using pandas style lambda function
            "a_mean": lambda x: x['year'].mean(),
            
            # pass the content of lambda function as a string
            "b_mean": "x['year'].mean()",
            
            # pass a tuple of function and column list
            "a_median": (np.median, "year"),
            
            # pass a tuple of function to retain same column name
            "year": (np.median, ),
            
            # pass multiple columns as a string
            "x_plus_y_mean": ("np.mean(x + y)", ["bill_length_mm", "bill_depth_mm"]),
            
            # output multiple columns as a list
            ('x', 'y'): lambda x: list(np.quantile(x['year'], [0.25, 0.75])),
            
            # same as above in string style
            ('x2', 'y2'): "list(np.quantile(x['year'], [0.25, 0.75]))",
            
            # tuple style with multiple output
            ('A', 'B'): ("[np.mean(x + y), np.mean(x - y)]"
                         , ["bill_length_mm", "bill_depth_mm"]
                         )
            })
            
        # grouped summarise in dict mode
        penguins_tidy.summarise({"a_mean": lambda x: (np.mean, 'year')},
                                by = ['species', 'sex']
                                )
                                
        # use kwargs
        penguins_tidy.summarise(
          {"a_mean": lambda x, **kwargs: x['year'].mean() + kwargs['leap']},
          by = ['species', 'sex'],
          leap = 4
          )
        
        # equivalently:
        penguins_tidy.summarise(
          {"a_mean": "x['year'].mean() + kwargs['leap']"},
          by = ['species', 'sex'],
          leap = 4
          )
        
        # across mode with column names
        penguins_tidy.summarise(
          func = np.mean,
          column_names = ['bill_length_mm', 'bill_depth_mm']
          )
          
        # across mode with predicate
        penguins_tidy.summarise(
          func = np.mean,
          predicate = dtypes.is_numeric_dtype,
          prefix = "avg_"
          )
          
        # grouped across with predicate
        penguins_tidy.summarise(
          func = np.mean,
          predicate = dtypes.is_numeric_dtype,
          prefix = "avg_",
          by = ['species', 'sex']
          )
        '''
        
        if dictionary is None and func is None:
            raise Exception(("Either dictionary or func with "
                             "predicate/column_names should be provided"
                            ))
        
        # assertions for across case                    
        if dictionary is None:
            assert callable(func) or isinstance(func, str),\
                ("arg 'func' should be a function or a string which is convertible "
                 "to a lambda function"
                 )
            assert isinstance(prefix, str)
            if (column_names is not None) and (predicate is not None):
                raise Exception(("Exactly one among 'column_names' and 'predicate' "
                                 "should be None")
                                 )
            if (column_names is None) and (predicate is None):
                raise Exception("Exactly one among 'column_names' and 'predicate' "
                                "should be None"
                                )
            
            # use column_names
            if column_names is not None:
                self._validate_column_names(column_names)
                column_names = _enlist(column_names)
            # use predicate to assign appropriate column_names
            else:
                mask = list(self.__data.apply(predicate, axis = 0))
                assert all([isinstance(x, bool) for x in mask]),\
                    "predicate should return a boolean per column"
                column_names = self.__data.columns[mask]
            
            if isinstance(func, str):
                func = eval("lambda x, **kwargs: " + func)
            prefixed_names = [prefix + x for x in column_names]
            
        # simple summarise
        if by is None:
            if dictionary is not None:
                res = self._summarise(dictionary, **kwargs)
            else:
                res = (self.__data[column_names]
                           .agg(func, **kwargs)
                           .to_frame()
                           .T
                           .rename(columns = dict(zip(column_names, prefixed_names)))
                           )
                res = TidyPandasDataFrame(res, copy = False)
        else:
            self._validate_by(by)
            by = _enlist(by)
            cn = self.get_colnames()
            
            if dictionary is not None:
                # agg column names should not intersect with by columns
                keys = dictionary.keys()
                keys_flattened = list(
                  np.concatenate(
                    [[x] if np.isscalar(x) else list(x) for x in keys]
                    )
                  )
                assert len(set(by).intersection(keys_flattened)) == 0,\
                    ("column names to be created by 'summarise' should not "
                     "intersect with 'by' columns"
                     )
                # special case to bypass _summarise when the agg is over a 
                # single column
                dv = dictionary.values() 
                special_case = False
                if all([isinstance(x, tuple) for x in dv]):
                    # cover these: 
                    # {'new_col': (np.mean, 'col')}
                    # {'col': (np.mean, )}
                    if all([len(x) == 1 or len(_enlist(x[1])) == 1 for x in dv]):
                        if all([callable(x[0]) for x in dv]):
                            special_case = True
                
                if special_case:
                    def create_summary(anitem):
                        # case: {'col': (np.mean, )}
                        if len(anitem[1]) == 1:
                            res = (self.__data
                                       .groupby(by, dropna = False, sort = False)[anitem[0]]
                                       .agg(anitem[1][0])
                                       .reset_index()
                                       )
                        else:
                        # case: {'new_col': (np.mean, 'col')}
                            res = (self.__data
                                       .groupby(by, dropna = False, sort = False)[anitem[1][1]]
                                       .agg(anitem[1][0])
                                       .reset_index()
                                       .rename(columns = {anitem[1][1]: anitem[0]})
                                       )
                        return res
                      
                    aggs = [create_summary(x) for x in dictionary.items()]
                    if len(aggs) == 1:
                        res = aggs[0]
                    else:
                        res = functools.reduce(lambda df1, df2: df1.merge(df2, on = by)
                                               , aggs
                                               )
                    res = TidyPandasDataFrame(res, check = False, copy = False)
                else: # regular case
                    lam = lambda chunk: chunk._summarise(dictionary, **kwargs)
                    res = self.group_modify(func = lam, by = by)
                    
                # set the column order
                # by columns come first
                # aggreagated columns come next
                col_order = by + keys_flattened
                res = res.select(col_order)
            else:
                rowid_name = _generate_new_string(self.get_colnames())
                group_mapper = (self.distinct(by)
                                    .add_row_number(rowid_name)
                                    .to_pandas()
                                    )
                res = (self.__data
                           .merge(group_mapper, on = by)
                           .drop(columns = by)
                           .groupby(rowid_name, dropna = False, sort = False)[column_names]
                           .agg(func, **kwargs)
                           .reset_index()
                           .merge(group_mapper, on = rowid_name)
                           .drop(columns = rowid_name)
                           .rename(columns = dict(zip(column_names, prefixed_names)))
                           .loc[:, by + prefixed_names]
                           )
                res = TidyPandasDataFrame(res, check = False, copy = False)          
                # column order is set with 'by' column in the begining

        return res
    
    summarize = summarise
    
    ##########################################################################
    # Joins
    ##########################################################################
    
    # validate join args
    def _validate_join(self
                       , y
                       , how
                       , on
                       , on_x
                       , on_y
                       , sort
                       , suffix_y
                       ):
                           
        assert isinstance(y, TidyPandasDataFrame),\
            "arg 'y' should be a tidy pandas dataframe"
        
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        
        assert isinstance(how, str),\
            "arg 'how' should be a string"
        assert how in ['inner', 'outer', 'left', 'right'],\
            "arg 'how' should be one among: 'inner', 'outer', 'left', 'right'"
        
        if on is None:
            assert on_x is not None and on_y is not None,\
                ("When arg 'on' is None, " 
                 "both args'on_x' and 'on_y' should not be None"
                 )
            assert _is_string_or_string_list(on_x),\
                "arg 'on_x' should be a string or a list of strings"
            assert _is_string_or_string_list(on_y),\
                "arg 'on_y' should be a string or a list of strings"
            
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            
            assert _is_unique_list(on_x),\
                "arg 'on_x' should not have duplicates"
            assert _is_unique_list(on_y),\
                "arg 'on_y' should not have duplicates"    
            assert set(on_x).issubset(cn_x),\
                "arg 'on_x' should be a subset of column names of x"
            assert set(on_y).issubset(cn_y),\
                "arg 'on_y' should be a subset of column names of y"
            assert len(on_x) == len(on_y),\
                "Lengths of arg 'on_x' and arg 'on_y' should match"
        else: # on is provided
            assert on_x is None and on_y is None,\
                ("When arg 'on' is not None, " 
                 "both args'on_x' and 'on_y' should be None"
                 )
            assert _is_string_or_string_list(on),\
                "arg 'on' should be a string or a list of strings"
            on = _enlist(on)
            assert _is_unique_list(on),\
                "arg 'on' should not have duplicates"
            assert set(on).issubset(cn_x),\
                "arg 'on' should be a subset of column names of x"
            assert set(on).issubset(cn_y),\
                "arg 'on' should be a subset of column names of y"
        
        assert isinstance(suffix_y, str),\
            "arg 'suffix_y' should be a string"
            
        assert suffix_y != "",\
            "arg 'suffix_y' should be a non-empty string"
            
        assert isinstance(sort, bool),\
            "arg 'sort' should be a boolean"
        
        return None
        
    # join methods
    def _join(self
              , y
              , how
              , on = None
              , on_x = None
              , on_y = None
              , sort = True
              , suffix_y = "_y"
              , validate = True
              ):
        
        if validate:               
            self._validate_join(y = y
                                , how = how
                                , on = on
                                , on_x = on_x
                                , on_y = on_y
                                , sort = sort
                                , suffix_y = suffix_y
                                )
        
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        nr_x = self.get_nrow()
        nr_y = y.get_nrow()
        
        # degenerate case
        if on is None:
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            if on_x == on_y:
                on = util_copy.copy(on_x)
                on_x = None
                on_y = None
        
        if on is None:
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            new_colnames = cn_x + list(setlist(cn_y).difference(on_y))
        else:
            on = _enlist(on)
            new_colnames = cn_x + list(setlist(cn_y).difference(on))
        
        if sort:
            if on is not None:
                res = (self
                       .__data
                       .assign(**{"__rn_x": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"__rn_y": nr_y})
                                        )
                              , how = how
                              , on = on
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       .sort_values(by = ["__rn_x", "__rn_y"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["__rn_x", "__rn_y"])
                       )
            else:
                res = (self
                       .__data
                       .assign(**{"__rn_x": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"__rn_y": nr_y})
                                        )
                              , how = how
                              , left_on = on_x
                              , right_on = on_y
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       .sort_values(by = ["__rn_x", "__rn_y"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["__rn_x", "__rn_y"])
                       )
                # remove the right keys
                on_y_with_suffix = [x + suffix_y for x in on_y]
                right_keys = set(res.columns).intersection(on_y_with_suffix)
                res = res.drop(columns = right_keys)
        else:
            if on is not None:
                res = (self
                       .__data
                       .merge(right = y.to_pandas(copy = False)
                              , how = how
                              , on = on
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       )
            else:
                res = (self
                       .__data
                       .merge(right = y.to_pandas(copy = False)
                              , how = how
                              , left_on = on_x
                              , right_on = on_y
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       )
                # remove the right keys
                on_y_with_suffix = [x + suffix_y for x in on_y]
                right_keys = set(res.columns).intersection(on_y_with_suffix)
                res = res.drop(columns = right_keys)
        
        res = (TidyPandasDataFrame(res
                                   , check = False
                                   , copy = False
                                   )
                .relocate(cn_x)
                )
        
        return res
        
    def inner_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        inner_join
        Joins columns of y to self by matching rows
        Includes only matching keys
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s1.inner_join(penguins_tidy_s2, on = 'island')
        '''
                       
        res = self._join(y = y
                        , how = "inner"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    
    def outer_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        outer_join
        Joins columns of y to self by matching rows
        Includes all keys
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s1.outer_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "outer"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    full_join = outer_join
        
    def left_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        left_join
        Joins columns of y to self by matching rows
        Includes all keys in self
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s1.left_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "left"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
        
    def right_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        right_join
        Joins columns of y to self by matching rows
        Includes all keys in y
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s1.right_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "right"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    # row of A which match B
    def semi_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        semi_join
        Joins columns of y to self by matching rows
        Includes keys in self if present in y
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s2.semi_join(penguins_tidy_s1, on = 'island')
        '''
        self._validate_join(y = y
                            , how = "inner" # this has no significance
                            , on = on
                            , on_x = on_x
                            , on_y = on_y
                            , sort = sort
                            , suffix_y = suffix_y
                            )
        
        if on is None:
            y2 = y.distinct(on_y)
        else:
            y2 = y.distinct(on)
        
        res = self._join(y = y2
                         , how = "inner"
                         , on = on
                         , on_x = on_x
                         , on_y = on_y
                         , sort = sort
                         , suffix_y = suffix_y
                         , validate = False
                         )
        return res
    
    # rows of A that do not match B
    def anti_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        anti_join
        Joins columns of y to self by matching rows
        Includes keys in self if not present in y
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s2.anti_join(penguins_tidy_s1, on = 'island')
        '''
        self._validate_join(y = y
                            , how = "inner" # not significant
                            , on = on
                            , on_x = on_x
                            , on_y = on_y
                            , sort = sort
                            , suffix_y = suffix_y
                            )
        
        string = _generate_new_string(y.get_colnames())
        
        if on is None:
            on_value = on_y
        else:
            on_value = on
            
        res =  (self._join(y.distinct(on_value).add_row_number(name = string)
                           , how = "left"
                           , on = on
                           , on_x = on_x
                           , on_y = on_y
                           , sort = sort
                           , suffix_y = suffix_y
                           , validate = False
                           )
                    .filter(lambda x: x[string].isna())
                    .select(string, include = False)
                    )
        return res
    
    # cross join
    def cross_join(self, y, sort = True, suffix_y = "_y"):
        '''
        cross_join
        Joins columns of y to self by matching rows
        Includes all cartersian product
        
        Parameters
        ----------
        y: TidyPandasDataFrame
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                         .select(['species', 'bill_length_mm', 'island'])
                                         )
        penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                         .select(['species', 'island', 'bill_depth_mm'])
                                         )
                                         
        penguins_tidy_s2.cross_join(penguins_tidy_s1)
        '''
        
        assert isinstance(y, TidyPandasDataFrame),\
          "arg 'y' should be a TidyPandasDataFrame"
          
        assert isinstance(sort, bool),\
          "arg 'sort' should be a bool"
        
        assert isinstance(suffix_y, str),\
          "arg 'suffix_y' should be a string"
        
        assert suffix_y != "",\
          "arg 'suffix_y' should not be an empty string"
        
        if sort:
            res = (pd.merge(self.__data.assign({"__rn_x": lambda x: np.arange(x.shape[0])})
                            , y.to_pandas(copy = False).assign({"__rn_y": lambda x: np.arange(x.shape[0])})
                            , how = "cross"
                            , left_index = False
                            , right_index = False
                            , suffixes = ["", suffix_y]
                            )
                     .sort_values(by = ["__rn_x", "__rn_y"]
                                      , ignore_index = True
                                      )
                     .drop(columns = ["__rn_x", "__rn_y"])
                     )
        else:  
            res = pd.merge(self.__data
                           , y.to_pandas(copy = False)
                           , how = "cross"
                           , left_index = False
                           , right_index = False
                           , suffixes = ["", suffix_y]
                           )
        
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    ##########################################################################
    # bind methods
    ##########################################################################  
      
    def cbind(self, y):
        '''
        cbind
        bind columns of y to self
        
        Parameters
        ----------
        y: a TidyPandasDataFrame with same number of rows
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. The TidyPandasDataFrame to be binded should same same number of rows.
        2. Column names of the TidyPandasDataFrame should be different from self.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        (penguins_tidy.select(['species', 'island'])
                      .cbind(penguins_tidy.select(['bill_length_mm', 'bill_depth_mm']))
                      )
        '''
        assert isinstance(y, TidyPandasDataFrame)
        # column names should differ
        assert len(set(self.get_colnames()).intersection(y.get_colnames())) == 0,\
            "Column names among the dataframes should not be common. Did you intend to `cross_join` instead of `cbind`?"
                # number of rows should match
        assert self.get_nrow() == y.get_nrow(),\
            "Both dataframes should have same number of rows"
            
        res = (pd.concat([self.__data, y.to_pandas()]
                        , axis = 1
                        , ignore_index = False # not to loose column names
                        )
                 .reset_index(drop = True)
                 .convert_dtypes()
                 .fillna(pd.NA)
                 )
        return TidyPandasDataFrame(res, check = False)
    
    def rbind(self, y):
        '''
        rbind
        bind rows of y to self
        
        Parameters
        ----------
        y: a TidyPandasDataFrame
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. Result will have union of column names of self and y.
        2. Missing values are created when a column name is present in one
           dataframe and not in the other.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        (penguins_tidy.select(['species', 'island'])
                      .rbind(penguins_tidy.select(['island', 'bill_length_mm', 'bill_depth_mm']))
                      )
        '''
        res = (pd.concat([self.__data, y.to_pandas()]
                         , axis = 0
                         , ignore_index = True
                         )
                 .convert_dtypes()
                 .fillna(pd.NA)
                 )
        return TidyPandasDataFrame(res, check = False)
    
    ##########################################################################
    # count and add_count
    ##########################################################################
    
    def count(self
              , column_names = None
              , name = 'n'
              , ascending = False
              , wt = None
              ):
        '''
        count
        count rows by groups
        
        Parameters
        ----------
        column_names: None or string or a list of strings
            Column names to groupby before counting the rows. 
            When None, counts all rows.
        name: string (default: 'n')
            Column name of the resulting count column
        ascending: bool (default is False)
            Whether to sort the result in ascending order of the count column
        wt: None or string
            When a string, should be a column name with numeric dtype. 
            When wt is None, rows are counted.
            When wt is present, then wt column is summed up.
            
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        # count rows
        penguins_tidy.count()
        
        # count number of rows of 'sex' column
        penguins_tidy.count('sex', name = "cnt")
        
        # sum up a column (weighted sum of rows)
        penguins_tidy.count(['sex', 'species'], wt = 'year')
        '''
        
        assert (column_names is None
                or _is_string_or_string_list(column_names)
                ),\
            "arg 'column_names' is either None or a list of strings"
        if column_names is not None:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        assert isinstance(name, str),\
            "arg 'name' should a string"
        assert isinstance(ascending, bool),\
            "arg 'ascending' should be a boolean"
        if wt is not None:
            assert isinstance(wt, str),\
                "arg 'wt' should be a string"
            assert wt in self.get_colnames(),\
                f"'wt' column '{wt}' should be a valid column name"
            assert pd.api.types.is_numeric_dtype(self.pull(wt)),\
                f"'wt' column '{wt}' should be of numeric dtype"
        
        if column_names is not None:
            assert isinstance(name, str),\
                "arg 'name' should be a string"
            assert name not in column_names,\
                ("arg 'name' should not be an element of arg "
                 "'column_names'"
                 )
            
            if wt is None:
                res = (self.__data
                           .groupby(column_names
                                    , sort = False
                                    , dropna = False
                                    )
                           .size()
                           .reset_index(drop = False)
                           .rename(columns = {0: name})
                           .sort_values(by = name
                                        , axis         = 0
                                        , ascending    = ascending
                                        , inplace      = False
                                        , kind         = 'quicksort'
                                        , na_position  = 'first'
                                        , ignore_index = True
                                        )
                           )
                res = TidyPandasDataFrame(res, check = False, copy = False)
            else:
                res = (self.summarise(
                              {name: (np.sum, wt)}
                               , by = column_names
                               , wt = wt
                               )
                           .arrange(name, ascending = ascending)
                           )
        else:
            if wt is None:
                res = pd.DataFrame({name : self.get_nrow()}
                                   , index = [0]
                                   )
                res = TidyPandasDataFrame(res, check = False, copy = False)
            else:
                res = self.summarise({name: "np.sum(x[kwargs['wt']])"}
                                     , wt = wt
                                     )
              
        return res

    def add_count(self
                  , column_names = None
                  , name = 'n'
                  , ascending = False
                  , wt = None
                  ):
        
        '''
        add_count
        adds counts of rows by groups as a column
        
        Parameters
        ----------
        column_names: None or string or a list of strings
            Column names to groupby before counting the rows. 
            When None, counts all rows.
        name: string (default: 'n')
            Column name of the resulting count column
        ascending: bool (default is False)
            Whether to sort the result in ascending order of the count column
        wt: None or string
            When a string, should be a column name with numeric dtype. 
            When wt is None, rows are counted.
            When wt is present, then wt column is summed up.
            
        Examples
        --------
        from palmerpenguins import load_penguins
        import pandas.api.types as dtypes
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        # count rows
        penguins_tidy.add_count()
        
        # count number of rows of 'sex' column
        penguins_tidy.add_count('sex', name = "cnt")
        
        # sum up a column (weighted sum of rows)
        penguins_tidy.add_count(['sex', 'species'], wt = 'year')
        '''
        
        if column_names is not None:
            assert isinstance(name, str),\
                "arg 'name' should be a string"
            assert name not in self.colnames,\
                "arg 'name' should not be an existing column name"
        
        count_frame = self.count(column_names = column_names
                                 , name = name
                                 , ascending = ascending
                                 , wt = wt
                                 )
                                 
        if column_names is None:
            count_value = int(count_frame.to_pandas().iloc[0, 0])
            res = self.mutate({name: np.array(count_value)})
        else:
            res = self.left_join(count_frame, on = column_names, sort = True)
        
        return res
    
    # pivot methods
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = None
                    , id_cols = None
                    , sep = "__"
                    ):
        '''
        Pivot data from long to wide
        
        Parameters
        ----------
        names_from: string or list of strings
            column names whose unique combinations are expected to become column
            new names in the result
        
        values_from: string or list of strings
            column names to fill the new columns with
        
        values_fill: scalar (default is None)
            A value to fill the missing values with
            When None, missing values are left as-is
        
        values_fn: function or a dict of functions (default is None)
            A function to handle multiple values per row in the result.
            When a dict, keys should be same as arg 'values_from'
            When None, multiple values are in kept a list and a single value is
            not kept in a list (will be a scalar)
        
        id_cols: string or list of strings, default is None
            Names of the columns that should uniquely identify an observation 
            (row) after widening (columns of the original dataframe that are
            supposed to stay put)
            
        sep: string
            seperator to use while creating resulting column names
            
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes().fillna(pd.NA)
        penguins_tidy = tidy(penguins)
        
        import numpy as np
        
        # generic widening leads to list-columns
        penguins_tidy.pivot_wider(id_cols       = "island"
                                  , names_from  = "sex"
                                  , values_from = "bill_length_mm"
                                  )
        
        # aggregate with a function
        penguins_tidy.pivot_wider(id_cols       = "island"
                                  , names_from  = "sex"
                                  , values_from = "bill_length_mm"
                                  , values_fn   = np.mean
                                  )
                                  
        # choose different aggregation logic for value_from columns
        penguins_tidy.pivot_wider(
          id_cols       = "island"
          , names_from  = "species"
          , values_from = ["bill_length_mm", "bill_depth_mm"]
          , values_fn   = {"bill_length_mm" : np.mean, "bill_depth_mm" : list}
          )
                                  
        # aggregate with almost any function
        penguins_tidy.pivot_wider(
          id_cols       = "island"
          , names_from  = "species"
          , values_from = "sex"
          , values_fn   = lambda x: dict(pd.Series(x).value_counts())
          )
        
        # All three inputs: 'id_cols', 'names_from', 'values_from' can be lists
        penguins_tidy.pivot_wider(
            id_cols       = ["island", "sex"]
            , names_from  = "species"
            , values_from = "bill_length_mm"
            )
                                  
        penguins_tidy.pivot_wider(
            id_cols       = ["island", "sex"]
            , names_from  = "species"
            , values_from = ["bill_length_mm", "bill_depth_mm"]
            )
        
        penguins_tidy.pivot_wider(id_cols       = ["island", "sex"]
                                  , names_from  = ["species", "year"]
                                  , values_from = "bill_length_mm"
                                  )
                                  
        penguins_tidy.pivot_wider(
            id_cols       = ["island", "sex"]
            , names_from  = ["species", "year"]
            , values_from = ["bill_length_mm", "bill_depth_mm"]
            )
        
        # when id_cols is empty, all columns except the columns from
        # `names_from` and `values_from` are considered as id_cols
        (penguins_tidy
         .select(['flipper_length_mm', 'body_mass_g'], include = False)
         .pivot_wider(names_from    = ["species", "year"]
                      , values_from = ["bill_length_mm", "bill_depth_mm"]
                      )
         )
                                  
        # fill the missing values with something
        penguins_tidy.pivot_wider(id_cols       = "island"
                                  , names_from  = "species"
                                  , values_from = "bill_length_mm"
                                  , values_fn   = np.mean
                                  , values_fill = 0
                                  )
        '''
        
        cn = self.get_colnames()
        
        assert _is_string_or_string_list(names_from),\
            "arg 'names_from' should be string or a list of strings"
        names_from = _enlist(names_from)
        assert _is_unique_list(names_from),\
            "arg 'names_from' should have unique strings"
        assert set(names_from).issubset(cn),\
            "arg 'names_from' should be a subset of existing column names"
        
        assert _is_string_or_string_list(values_from),\
            "arg 'values_from' should be string or a list of strings"
        values_from = _enlist(values_from)
        assert set(values_from).issubset(cn),\
            "arg 'names_from' should have unique strings"
        assert len(set(values_from).intersection(names_from)) == 0,\
            ("arg 'names_from' and 'values_from' should not "
             "have common column names"
             )
        names_values_from = set(values_from).union(names_from)
        
        if id_cols is None:
            id_cols = list(set(cn).difference(names_values_from))
            if len(id_cols) == 0:
                raise Exception(
                    ("'id_cols' is turning out to be empty. Choose the "
                     "'names_from' and 'values_from' appropriately or specify "
                     "'id_cols' explicitly."
                     ))
            else:
                print("'id_cols' chosen: " + str(id_cols))
        else:
            assert _is_string_or_string_list(id_cols),\
                "arg 'id_cols' should be string or a list of strings"
            id_cols = _enlist(id_cols)
            assert _is_unique_list(id_cols),\
                "arg 'id_cols' should have unique strings"
            assert len(set(id_cols).intersection(names_values_from)) == 0,\
                ("arg 'id_cols' should not have common names with either "
                 "'names_from' or 'values_from'"
                 )
        if values_fill is not None:
            assert np.isscalar(values_fill),\
                "arg 'values_fill' should be a scalar"
        
        assert (values_fn is None 
                or callable(values_fn) 
                or isinstance(values_fn, dict)),\
            "When not None, arg 'values_fn' should be a callable or dict"
        
        if values_fn is None:
            
            def unlist_scalar(x):
                if len(x) == 1:
                    x = x[0]
                return x
            
            values_fn = unlist_scalar
            
        elif isinstance(values_fn, dict):
            assert isinstance(values_fn, dict),\
                ("When arg 'values_fn' is not a scalar, it should either be "
                 "a dict"
                 )
            keys_fn = set(values_fn.keys())
            assert set(values_from) == keys_fn,\
                ("When arg 'values_fn' is a dict, keys should match "
                 "arg 'values_from'"
                 )
            assert all([callable(x) for x in values_fn.values()]),\
                "When arg 'values_fn' is a dict, values should be functions"
        else:
            pass
        
        assert isinstance(sep, str),\
            "arg 'sep' should be a string"
        
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
                             , dropna     = False
                             , observed   = True
                             )
        res = simplify(res)
                             
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    def pivot_longer(self
                     , cols
                     , names_to = "name"
                     , values_to = "value"
                     , include = True
                     , values_drop_na = False
                     ):
        '''
        Pivot from wide to long
        aka melt
        
        Parameters
        ----------
        cols: list of strings
          Column names to be melted.
          The dtypes of the columns should match.
          Leftover columns are considered as 'id' columns.
          When include is False, 'cols' refers to leftover columns.
        names_to: string (default: 'name')
          Name of the resulting column which will hold the names of the columns
          to be melted
        values_to: string (default: 'value')
          Name of the resulting column which will hold the values of the columns
          to be melted
        include: bool
          If True, cols are used to melt. Else, cols are considered as 'id'
          columns
        values_drop_na: bool
          Whether to drop the rows corresponding to missing value in the result
          
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes().fillna(pd.NA)
        penguins_tidy = tidy(penguins)
        
        # pivot to bring values from columns ending with 'mm'
        cns = ['species'
               , 'bill_length_mm'
               , 'bill_depth_mm'
               , 'flipper_length_mm'
               ]
        (penguins_tidy.select(cns)
                      .pivot_longer(cols = ['bill_length_mm',
                                            'bill_depth_mm',
                                            'flipper_length_mm']
                                    )
                      )
                      
        # pivot by specifying 'id' columns to obtain the same result as above
        # this is helpful when there are many columns to melt
        (penguins_tidy.select(cns)
                      .pivot_longer(cols = 'species',
                                    include = False
                                    )
                      )
        '''
        # assertions
        cn = self.get_colnames()
        assert _is_string_or_string_list(cols),\
            "arg 'cols' should be a string or a list of strings"
        cols = _enlist(cols)
        assert set(cols).issubset(cn),\
            "arg 'cols' should be a subset of existing column names"
        assert isinstance(include, bool),\
            "arg 'include' should be a bool"
        if not include:
           cols = list(setlist(cn).difference(cols))
           assert len(cols) > 0,\
               "At least one column should be selected for melt"
        
        id_vars = set(cn).difference(cols)
        assert isinstance(names_to, str),\
            "arg 'names_to' should be a string"
        assert isinstance(values_to, str),\
            "arg 'values_to' should be a string"
        assert names_to not in id_vars,\
            "arg 'names_to' should not match a id column"
        assert values_to not in id_vars,\
            "arg 'values_to' should not match a id column"
        
        # core operation
        res = (self.__data
                   .melt(id_vars         = id_vars
                         , value_vars    = cols
                         , var_name      = names_to
                         , value_name    = values_to
                         , ignore_index  = True
                         )
                   .convert_dtypes()
                   .fillna(pd.NA)
                   )
                   
        res = TidyPandasDataFrame(res, check = False, copy = False)
        
        if values_drop_na:
            res = res.drop_na(values_to)
        
        return res
    
    ##########################################################################
    # slice extensions
    ##########################################################################
    
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
            proportion of rows to subset, should be between 0(exclusive) and 1 
            (inclusive)
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
            res = TidyPandasDataFrame(self.__data.head(n)
                                      , check = False
                                      , copy = False
                                      )
        else:
            self._validate_by(by)
            by = _enlist(by)
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
                 
                res = (self.group_modify(lambda x: x.slice(np.arange(n))
                                         , by = by
                                         , preserve_row_order = True
                                         )
                           )
            else:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'rounding_type' should be one among: 'round' (default)"
                     " , 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                    
                res = self.group_modify(
                          lambda x: x.slice(range(int(roundf(x.shape[0] * prop))))
                          , by = by
                          , preserve_row_order = True
                          )
            
        return res
    
    head = slice_head
    
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
            by = _enlist(by)
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
                 
                res = (self.group_modify(lambda x: x.tail(n).reset_index(drop = True)
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
                    
                res = (self.group_modify(
                    lambda x: (x.tail(int(roundf(x.shape[0] * prop)))
                                .reset_index(drop = True)
                                )
                    , by = by
                    , is_pandas_udf      = True
                    , preserve_row_order = True
                    ))
            
        return res
    
    tail = slice_tail
    
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
            Proportion of rows to subset, should be non-negative
        replace: bool, default is False
            Whether to sampling should be done with replacement
        weights: string, pandas series, numpy array (Default is None)
            When a string, it should be an column of numeric dtype
            When None, no weights are used
        random_state: positive integer (Default is None)
            Seed to keep the sampling reproducible
            When None, a pseudorandom seed is chosen
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
        
        # swor: sample without replacement
        # swr: sample with replacement
        
        # sample by specifiying count
        penguins_tidy.slice_sample(n = 5)                    # swor
        penguins_tidy.slice_sample(n = 5, replace = True)    # swr, smaller than input
        penguins_tidy.slice_sample(n = 1000, replace = True) # swr, larger than input
        
        # sample by specifiying proportion of number of rows of the input
        penguins_tidy.slice_sample(prop = 0.3)                 # swor
        penguins_tidy.slice_sample(prop = 0.3, replace = True) # swr, smaller than input
        penguins_tidy.slice_sample(prop = 1.1, replace = True) # swr, larger than input
        
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
            by = _enlist(by)
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
                    rs = x['__seed'].iloc[0]
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
                           .drop(columns = "__seed")
                           )
            else:
                def sample_chunk_prop(x):
                    rs = x['__seed'].iloc[0]
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
                           .drop(columns = "__seed")
                           )
            
            res = (res.sample(frac = 1, random_state = random_state)
                      .reset_index(drop = True)
                      )
            
        return TidyPandasDataFrame(res, check = False)
    
    sample = slice_sample
    
    ##########################################################################
    # slice min/max
    ##########################################################################
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        '''
        slice_min
        Subset some rows ordered by some columns
        
        # WIP subset 3 rows corresponding to 'bill_length_mm'
        '''
        nr = self.nrow
        cn = self.colnames

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
        self._validate_order_by(order_by)
          
        assert isinstance(with_ties, bool),\
            "arg 'with_ties' should be a bool"
            
        if with_ties:
            keep_value = "all"
        else:
            keep_value = "first"
        
        if by is None:  
            res = (self.__data
                       .nsmallest(n, columns = order_by, keep = keep_value)
                       .reset_index(drop = True)
                       .loc[:, cn]
                       .pipe(lambda x: TidyPandasDataFrame(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            if case_prop:
                
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'ties_method' should be one among: 'round' (default),"
                     " 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nsmallest(int(roundf(x.shape[0] * prop))
                                                    , columns = order_by
                                                    , keep = keep_value
                                                    )
                                          .pipe(TidyPandasDataFrame
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              )
                           .select(cn)
                           .arrange(order_by, by = by, ascending = True)
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
                    ("arg 'n' should not exceed the size of any chunk after "
                    "grouping")
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nsmallest(n
                                                    , columns = order_by
                                                    , keep = keep_value
                                                    )
                                          .pipe(TidyPandasDataFrame
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              )
                           .select(cn)
                           .arrange(order_by, by = by, ascending = True)
                      )
        return res
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        
        nr = self.get_nrow()
        cn = self.get_colnames()

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
          
        assert isinstance(with_ties, bool),\
            "arg 'with_ties' should be a bool"
            
        if with_ties:
            keep_value = "all"
        else:
            keep_value = "first"
        
        if by is None:  
            res = (self.__data
                       .nlargest(n, columns = order_by, keep = keep_value)
                       .reset_index(drop = True)
                       .convert_dtypes()
                       .loc[:, cn]
                       .pipe(lambda x: TidyPandasDataFrame(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            if case_prop:
                
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'ties_method' should be one among: 'round' (default),"
                     " 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nlargest(int(roundf(x.shape[0] * prop))
                                                    , columns = order_by
                                                    , keep = keep_value
                                                    )
                                          .pipe(TidyPandasDataFrame
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              )
                           .select(cn)
                           .arrange(order_by, by = by, ascending = False)
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
                    ("arg 'n' should not exceed the size of any chunk after "
                    "grouping")
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nlargest(n
                                                    , columns = order_by
                                                    , keep = keep_value
                                                    )
                                          .pipe(TidyPandasDataFrame
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              )
                           .select(cn)
                           .arrange(order_by, by = by, ascending = False)
                      )
        return res
    
    # expand and complete utilities
    
    def expand(self, column_names):
        assert isinstance(column_names, [list, tuple])
        flattened = flatten(list(column_names)) # get correct syntax
        assert _is_unique_list(flattened)
        
        
        
        return None
            
    def complete(self, l):
        expanded = df.expand(l)
        return expanded.left_join(df, on = expanded.get_colnames())
    
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
    
  ##############################################################################
  # any_na
  ##############################################################################
    def any_na(self):
        '''
        Is there a missing value in the dataframe?
        
        Returns
        -------
        bool
            True if there is a missing value, False otherwise
        '''
        res = (self.__data
                   .isna() # same dim as input
                   .any()  # series of booleans per column
                   .any()  # single value
                   )
                   
        return bool(res)
    
  ##############################################################################
  # replace_na
  ##############################################################################
    
    def replace_na(self, value):
        '''
        Replace missing values with a specified value
        
        Parameters
        ----------
        value: dict or a scalar
            When a dict, key should be a column name and value should be the
            value to replace by missing values of the column
            When a scalar, missing values of all columns will be replaved with
            value
            
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy.replace_na({'sex': 'unknown'})
        penguins_tidy.select(predicate = dtypes.is_numeric_dtype).replace_na(1)
        '''
        return TidyPandasDataFrame(self.__data.fillna(value).fillna(pd.NA)
                                   , copy = False
                                   )
    
    def drop_na(self, column_names = None):
        
        cn = self.get_colnames()
        if column_names is not None:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        else:
            column_names = cn
        
        res = (self.__data
                   .dropna(axis = "index"
                           , how = "any"
                           , subset = column_names
                           , inplace = False
                           )
                   .reset_index(drop = True)
                   )
        
        return TidyPandasDataFrame(res, check = False, copy = False)
    
  ##############################################################################
  # fill_na (fill)
  ##############################################################################
        
    def fill_na(self, column_direction_dict, by = None):
        '''
        fill_na
        Fill missing values from neighboring values per column
        
        Paramaters
        ----------
        column_direction_dict: dict
            where key is a columnname and value is the direction to fill.
            Direction should be one among: 'up', 'down', 'updown' and 'downup'
        by: string or list of strings
            column names to group by 
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. The output might still have missing values after applying fill_na.
        
        Examples
        --------
        df = tidy(pd.DataFrame({'A': [pd.NA, 1, 1, 2, 2, 3, pd.NA],
                                'B': [pd.NA, pd.NA, 1, 2, pd.NA, pd.NA, 3]
                               }
                              )
                 )
        df
        
        df.fill_na({'B': 'up'})
        df.fill_na({'B': 'down'})
        df.fill_na({'B': 'downup'})
        df.fill_na({'B': 'updown'})
        
        df.fill_na({'B': 'up'}, by = 'A')
        df.fill_na({'B': 'down'}, by = 'A')
        df.fill_na({'B': 'updown'}, by = 'A')
        df.fill_na({'B': 'downup'}, by = 'A')
        '''
        cn = self.get_colnames()
        assert isinstance(column_direction_dict, dict),\
            "arg 'column_direction_dict' shoulgithubd be a dict"
        assert set(column_direction_dict.keys()).issubset(cn),\
            ("keys of 'column_direction_dict' should be a subset of existing "
             "column names"
             )
        valid_methods = ["up", "down", "updown", "downup"]
        assert set(column_direction_dict.values()).issubset(valid_methods),\
            ("values of 'column_direction_dict' should be one among:  "
             "'up', 'down', 'updown', 'downup'"
             )
        
        if by is not None:
            self._validate_by(by)
            by = _enlist(by)
            
            # by columns should not be altered
            keys = column_direction_dict.keys()
            assert len(set(by).intersection(keys)) == 0,\
                ("'by' columns cannot be altered. keys of 'column_direction_dict'"
                 " should not intersect with 'by'"
                 )

        def fill_chunk(x, cdd):
            
            chunk = x.copy()
            
            for akey in cdd:
                method = cdd[akey]
                if method == "up":
                    chunk[akey] = chunk[akey].bfill()
                elif method == "down":
                    chunk[akey] = chunk[akey].ffill()
                elif method == "updown":
                    chunk[akey] = chunk[akey].bfill()
                    chunk[akey] = chunk[akey].ffill()
                else:
                    chunk[akey] = chunk[akey].ffill()
                    chunk[akey] = chunk[akey].bfill()
                
            return chunk
        
        if by is None:
            res = fill_chunk(self.__data, column_direction_dict)
        else:
            res = (self.__data
                       .assign(**{'_rn': lambda x: np.arange(x.shape[0])})
                       .groupby(by, sort = False, dropna = False)
                       .apply(fill_chunk, column_direction_dict)
                       .reset_index(drop = True)
                       .sort_values('_rn', ignore_index = True)
                       .drop(columns = '_rn')
                       .fillna(pd.NA)
                       )

        return TidyPandasDataFrame(res, copy = False, check = False)
    
    fill = fill_na
    
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
    
    def separate_rows(self, column_name, sep = ';'):
        '''
        split a string column using a seperator and create a row for each
        
        Parameters
        ----------
        column_name: string
            A column name to split
        
        sep: string
            regex to split
            
        Returns
        -------
        TidyPandasDataFrame
        
        Examples
        --------
        temp = tidy(pd.DataFrame({"A": ["hello;world", "hey,mister;o/mister"]}))
        temp.separate_rows('A', sep = ",|;")
        '''
        
        assert isinstance(column_name, str),\
            "arg 'column_name' should be a string"
        assert column_name in self.get_colnames(),\
            "arg 'column_name' should be an exisiting column name"
        assert isinstance(sep, str),\
            "arg 'sep' should be a string"
        
        def splitter(str_col):
            return [re.split(sep, x) for x in str_col]
            
        res = (self.__data
                   .assign(**{column_name: lambda x: splitter(x[column_name])})
                   .explode(column_name, ignore_index = True)
                   )
        
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    ##########################################################################
    # nest and unnest
    ##########################################################################
    
    
    ##########################################################################
    # nest_by
    ##########################################################################
    def nest_by(self
                , by = None
                , nest_column_name = 'data'
                , drop_by = True
                ):
        '''
        nest_by
        Nest all columns of tidy dataframe with respect to 'by' columns
        
        Parameters
        ----------
        by: str or list of strings
            Columns to stay, rest of them are nested
        nest_column_name: str
            Name of the resulting nested column (pandas Series)
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. 'by' should not contain all column names (some columns should be left
           for nesting)
           
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)

        penguins_tidy.nest_by(by = 'species')
        penguins_tidy.nest_by(by = ['species', 'sex'])
        '''
        cn = self.get_colnames()
        self._validate_by(by)
        by = _enlist(by)
        assert len(by) < len(cn),\
            "arg 'by' should not contain all exisiting column names"
        
        assert nest_column_name not in cn,\
            "arg 'nest_column_name' should not be an exisiting column name"
            
        assert isinstance(drop_by, bool),\
            "arg 'drop_by' should be a bool"
        
        extra = False
        if len(by) == 1:
            extra = True
            new_colname = _generate_new_string(cn)
            by = [new_colname] + by # note that by has changed
    
        if extra:
            go = (pdf.assign(**{new_colname: 1})
                     .groupby(by, sort = False, dropna = False)
                     )
        else:
            go = pdf.groupby(by, sort = False, dropna = False)
                      
        res = pd.DataFrame(list(go.groups.keys()), columns = by)
          
        
        # add data into nest_column_name column
        if drop_by:
            res[nest_column_name] = (
                pd.Series(map(lambda x: x.drop(columns = by)
                              , dict(tuple(go)).values()
                              )
                          )
                )
        else:
            # drop the extra column in the nested column
            if extra:
                res[nest_column_name] = (
                    pd.Series(map(lambda x: x.drop(columns = new_colname)
                                  , dict(tuple(go)).values()
                                  )
                              )
                    )
            else:
                res[nest_column_name] = pd.Series(dict(tuple(go)).values())
        
        if extra:
            res = res.drop(columns = new_colname)
        
        res = res.convert_dtypes().fillna(pd.NA)
        
        res[nest_column_name] = ( 
            res[nest_column_name].apply(lambda x: TidyPandasDataFrame(
                                                    x
                                                    , copy = False
                                                    , check = False
                                                    )
                                        )
                                )
        
        return TidyPandasDataFrame(res, copy = False, check = False)
                                   
    ##########################################################################
    # nest
    ##########################################################################                               
    def nest(self
             , column_names = None
             , nest_column_name = 'data'
             , include = True
             ):
        '''
        nest
        Nest columns of tidy dataframe
        
        Parameters
        ----------
        column_names: str or list of strings
            Columns to be nested
        nest_column_name: str
            Name of the resulting nested column (pandas Series)
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. 'column_names' should not contain all column names.
           
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        
        penguins_tidy.nest(['bill_length_mm', 'bill_depth_mm',
                            'flipper_length_mm', 'body_mass_g']
                           )
        penguins_tidy.nest(['species', 'sex'], include = False)
        '''
        
        cn = self.get_colnames()
        self._validate_column_names(column_names)
        column_names = _enlist(column_names)
        if not include:
            column_names = list(setlist(cn).difference(column_names))
        by = list(setlist(cn).difference(column_names))
        assert len(by) > 0,\
            "arg 'column_names' should not have all existing column names"
            
        return self.nest_by(by = by, nest_column_name = nest_column_name)
        
    ##########################################################################
    # unnest
    ##########################################################################    
    def unnest(self, nest_column_name = 'data'):
        '''
        unnest
        Unnest a nested column of a tidy dataframe
        
        Parameters
        ----------
        nest_column_name: str
            Name of the column to be unnested
        
        Returns
        -------
        TidyPandasDataFrame
        
        Notes
        -----
        1. unnest does not support a mix of nested lists and dataframe in the
           same column
        
        Examples
        --------
        import pandas as pd
        nested_pdf = pd.DataFrame({"A": [1,2,3, 4],
                                   "B": pd.Series([[10, 20, 30],
                                                   [40, 50],
                                                   [60],
                                                   70
                                                  ]
                                                 )
                          })
        nested_pdf
        
        # unnest nested lists
        tidy(nested_pdf).unnest('B')
        
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes()
        penguins_tidy = tidy(penguins)
        pen_nested_by_species = penguins_tidy.nest_by('species')
        pen_nested_by_species
        
        # unnest nested dataframes
        pen_nested_by_species.unnest('data')
        '''
        
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        assert nest_column_name in cn,\
            "arg 'nest_column_name' should be a exisiting column name"
            
        all_are_tidy = all(map(lambda x: isinstance(x, TidyPandasDataFrame)
                               , list(self.__data[nest_column_name])
                               )
                           )
        
        def is_list_scalar_na(x):
            if isinstance(x, (list, pd.core.arrays.floating.FloatingArray)):
                res = True
            elif np.isscalar(x):
                res = True
            elif isinstance(pd.isna(x), bool) and pd.isna(x):
                res = True
            else:
                res = False
            return res
        
        all_are_list = all(map(is_list_scalar_na
                               , list(self.__data[nest_column_name])
                               )
                           )
         
        assert all_are_tidy or all_are_list,\
            ("arg 'nest_column_name' column is neither a column of "
             "TidyPandasDataFrames nor a column or lists or scalars"
             )
        
        if all_are_tidy:
            cols = list(setlist(cn).difference([nest_column_name]))
            # create a filled up dataframe per row via cross join and rbind them
            res = pd.concat(
                map(lambda x: pd.merge(
                      self.__data.loc[[x], cols]
                      , (self.__data[nest_column_name][x]).to_pandas(copy = False)
                      , how = 'cross'
                      )
                    , range(nr)
                    )
                , axis = 'index'
                , ignore_index = True
                )
        else:
            res = (self.__data
                       .explode(column = nest_column_name)
                       .reset_index(drop = True)
                       )
                
        return TidyPandasDataFrame(res.convert_dtypes().fillna(pd.NA)
                                   , check = False
                                   , copy = False
                                   )
    
    ##########################################################################
    # split (group_split)
    ##########################################################################
    def split(self, by, as_dict = False):
        
        self._validate_by(by)
        by = _enlist(by)
        assert isinstance(as_dict, bool),\
            "arg 'as_dict' should be a bool"
        
        if as_dict:
            nt = namedtuple("group", by)
            
            # unpacking required when first element is a tuple
            if len(by) == 1:
                create_kv = lambda x: (nt(x[0])
                                       , (x[1].reset_index(drop = True)
                                              .pipe(TidyPandasDataFrame)
                                              )
                                       )
            else:
                create_kv = lambda x: (nt(*x[0])
                                       , (x[1].reset_index(drop = True)
                                              .pipe(TidyPandasDataFrame)
                                              )
                                       )
            
            res = dict((create_kv(x) for x in 
                tuple(self.__data.groupby(by, dropna = False, sort = False))
                ))
        else:
            res = [x[1] for x in tuple((self.__data
                                            .groupby(by
                                                     , dropna = False
                                                     , sort = False
                                                     )
                                            ))]
            
        return res
    
    group_split = split
    
    ##########################################################################
    # getitem and setitem based on pandas loc
    ##########################################################################
    
    def __getitem__(self, key):
      '''
      Subset some and columns of the dataframe
      Always returns a copy and not a view
      
      Parameters
      ----------
      key: tuple of x and y subset operations
      
      Returns
      -------
      TidyPandasDataFrame
      
      Notes
      -----
      1. Rows can be subset by specifying integer positions 
      (as int, list, slice and range objects) or by providing a boolean mask.
      
      2. Columns can be subset by specifiying integer positions 
      (as int, list, slice and range objects) or by specifying a list of unique 
      column names or by providing a boolean mask.
      
      3. Any combination of row and column specifications work together.
      
      Examples
      --------
      from palmerpenguins import load_penguins
      penguins = load_penguins().convert_dtypes()
      penguins_tidy = tidy(penguins)
      
      # Rows can be subset with integer indexes with slice objects
      # right end is not included in slice objects
      # first four rows
      penguins_tidy[0:4,:]
      
      # a row can be subset with a single integer
      # moreover subsetting always returns a dataframe
      penguins_tidy[10, :]
      
      # Rows can be subset with a boolean mask 
      penguins_tidy[penguins_tidy.pull('bill_length_mm') > 40, :]
      
      # Columns can be subset using column names
      penguins_tidy[0:5, ["species", "year"]]
      
      # A single column can be subset by specifying column name as a string
      # moreover subsetting always returns a dataframe 
      penguins_tidy[0:5, "species"] # same as: penguins_tidy[0:5, ["species"]]
      
      # columns can be subset by integer position
      penguins_tidy[[7, 6, 5], 0:3]
      
      # row and columns can be subset with different types of specifications
      penguins_tidy[0:2, 1] # same as: penguins_tidy[[0, 1], 'island']
      '''
      cn = self.colnames
      key = list(key)
      
      # handle a single integer for row
      if isinstance(key[0], int):
        key[0] = [key[0]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[0], slice):
        key[0] = range(self.nrow)[key[0]]
        
      # handle integer indexing for columns
      is_int = isinstance(key[1], int)
      is_int_list = (isinstance(key[1], list) 
                     and 
                     all([isinstance(x, int) for x in key[1]])
                     )
      if (is_int or is_int_list):
        key[1] = _enlist(key[1])
        assert _is_unique_list(key[1]),\
            "Integer index for columns should be unique"
        assert np.min(key[1]) >= 0,\
            "Integer index for columns should be non-negative"
        assert np.max(key[1]) < self.ncol,\
            "Integer index for columns is beyond the range"
        
        key[1] = [cn[x] for x in key[1]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[1], slice):
        key[1] = cn[key[1]]
      # handle single string column index
      elif isinstance(key[1], str):
        key[1] = _enlist(key[1])
      
      res = (self.to_pandas(copy = False)
                 .loc[key[0], key[1]]
                 .reset_index(drop = True)
                 )
      return TidyPandasDataFrame(res, check = False, copy = True)
    
    def __setitem__(self, key, value):
      '''
      Change a subset of the dataframe in-place
      
      Parameters
      ----------
      key: tuple of x and y subset operations
      value: value to be assigned
      
      Returns
      -------
      TidyPandasDataFrame
      
      Notes
      -----
      1. Rows can be subset by specifying integer positions 
      (as int, list, slice and range objects) or by providing a boolean mask.
      
      2. Columns can be subset by specifiying integer positions 
      (as int, list, slice and range objects) or by specifying a list of unique 
      column names or by providing a boolean mask.
      
      3. Any combination of row and column specifications work together.
      
      4. Assignment is done by "pdf.loc[exp1, exp2] = value". Incompatible value
      assignment exceptions are handled by this method and they will cascade.
      
      Examples
      --------
      from palmerpenguins import load_penguins
      penguins = load_penguins().convert_dtypes()
      penguins_tidy = tidy(penguins)
      
      # assign a single value with correct type
      penguins_tidy[0,0] = "a"
      penguins_tidy[0:5, :]
      
      # assign a multiple values with correct type
      penguins_tidy[0:3,0] = "b"
      penguins_tidy[0:5, :]
      
      # assign a row partially by a list of appropriate types
      penguins_tidy[0, ['species', 'bill_length_mm']] = ['c', 1]
      penguins_tidy[0:5, :]
      
      # assign a subset with another TidyPandasDataFrame
      penguins_tidy[0:2, 0:2] = pd.DataFrame({'species': ['d', 'e']
                                              , 'island': ['f', 'g']
                                              }).pipe(tidy)
      penguins_tidy[0:5, :]
      '''
      cn = self.colnames
      key = list(key)
      
      # handle a single integer for row
      if isinstance(key[0], int):
        key[0] = [key[0]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[0], slice):
        key[0] = range(self.nrow)[key[0]]
        
      # handle integer indexing for columns
      is_int = isinstance(key[1], int)
      is_int_list = (isinstance(key[1], list) 
               and 
               all([isinstance(x, int) for x in key[1]])
               )
      if (is_int or is_int_list):
        key[1] = _enlist(key[1])
        assert _is_unique_list(key[1]),\
            "Integer index for columns should be unique"
        assert np.min(key[1]) >= 0,\
            "Integer index for columns should be non-negative"
        assert np.max(key[1]) < self.ncol,\
            "Integer index for columns is beyond the range"
        
        key[1] = [cn[x] for x in key[1]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[1], slice):
        key[1] = cn[key[1]]
      # handle single string column index
      elif isinstance(key[1], str):
        key[1] = _enlist(key[1])
      
      # handle the case when value is a pandas dataframe
      if isinstance(value, (pd.DataFrame
                            , pd.core.groupby.generic.DataFrameGroupBy
                            )):
        msg = ("When arg 'value' is a dataframe, then it should be a "
               "TidyPandasDataFrame and not a pandas dataframe")
        raise Exception(msg)
      
      # handle when value is a tidy pdf
      if isinstance(value, TidyPandasDataFrame):
        self.to_pandas(copy = False).loc[key[0], key[1]] = value.to_pandas()
      # nothing to return as this is an inplace operation
    
    