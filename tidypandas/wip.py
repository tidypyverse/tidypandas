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

def generate_new_string(strings):
    
    assert isinstance(strings, list)
    assert all([isinstance(x, str) for x in strings])
    
    while True:
        random_string = "".join(np.random.choice(list(string.ascii_letters), 20))
        if random_string not in strings:
            break
    
    return random_string

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
         , copy = True
         , **kwargs
         ):
    
    res = TidyPandasDataFrame(simplify(pdf
                                       , drop_range_index = drop_range_index
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
from functools import reduce
from collections_extended import setlist
from skimpy import skim
import string as string

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
            Whether the TidyPandasDataFrame object to be created should refer to a copy of the input pandas dataframe or the input itself

        Notes
        -----
        1. A pandas dataframe is said to be 'simple' if:
            a. Column names (x.columns) are an unnamed pd.Index object of unique strings.
            b. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
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
        penguins_tidy = tidy(penguins)
        penguins_tidy
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
        
        if copy:                       
            self.__data = util_copy.copy(x)
        else:
            self.__data = x
        return None
    
    # repr method
    def __repr__(self):
        shape = self.__data.shape
        header_line   = '# A tidy dataframe: {nrow} X {ncol}'\
              .format(nrow = shape[0], ncol = shape[1])
        pandas_str    = self.__data.head(10).__str__()
        # pandas_str    = '\n'.join(pandas_str.split('\n')[:-1])
        
        left_over = shape[0] - 10
        if left_over > 0:
            leftover_str = "# ... with {left_over} more rows".format(left_over = left_over)
        
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
    # to pandas methods
    ##########################################################################
    
    # pandas copy method
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
        if copy:
            res = util_copy.copy(self.__data)
        else:
            res = self.__data
            
        return res
    
    # series copy method
    def pull(self, column_name, copy = True):
        '''
        pull
        Returns a copy of column as pandas series
        
        Parameters
        ----------
        column_name : str
            Name of the column to be returned as pandas series

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
            "Input column names should be a string"
        assert column_name in list(self.__data.columns), \
            "column_name is not an existing column name"
        
        if copy:
            res = util_copy.copy(self.__data[column_name])
        else:
            res = self.__data[column_name]
        return res
    
    ##########################################################################
    # pipe methods
    ##########################################################################
    
    # pipe method
    def pipe(self, func, **kwargs):
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
        return func(self, **kwargs)
    
    # pipe_tee
    def pipe_tee(self, func, **kwargs):
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
        func(self, **kwargs)
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
    
    def add_row_number(self, name = 'row_number', by = None):
        '''
        add_row_number
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
        2. Column indicating row number is added as the first column (to the left).
        3. Alias: rowid_to_column
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins = load_penguins().convert_dtypes().pipe(tidy)
        penguins_tidy = tidy(penguins)

        penguins_tidy.add_row_number() # equivalently penguins_tidy.add_rowid()
        
        # add row number per group in the order of appearance
        penguins_tidy.add_row_number(by = 'sex')
        '''
        nr = self.get_nrow()
        cn = self.get_colnames()
        
        assert isinstance(name, str),\
            "arg 'name' should be a string"
        if name[-2:] == "__":
            raise Exception("'name' should not be ending with '__'.")
        
        if name in cn:
            raise Expection("'name' should not be an existing column name.")
            
        if by is None:
            res = self.__data.assign(**{name : np.arange(nr)})
        else:
            self._validate_by(by)
            by = enlist(by)
            res = (self
                   .__data
                   .assign(**{"rn__" : np.arange(nr)})
                   .groupby(by, sort = False, dropna = False)
                   .apply(lambda x: x.assign(**{name : np.arange(x.shape[0])}))
                   .reset_index(drop = True)
                   .sort_values("rn__", ignore_index = True)
                   .drop(columns = "rn__")
                   )
        
        col_order = [name]
        col_order.extend(cn)
        
        return TidyPandasDataFrame(res.loc[:, col_order]
                                   , check = False
                                   , copy = False
                                   )
    
    rowid_to_column = add_row_number
    
    def _validate_by(self, by):
        
        assert is_string_or_string_list(by),\
            "arg 'by' should be a string or a list of strings"
            
        by = enlist(by)
        
        assert len(set(by)) == len(by),\
            "arg 'by' should have unique strings"
        
        assert set(self.get_colnames()).issuperset(by),\
            "arg 'by' should contain valid column names"
            
        return None
    
    def _validate_column_names(self, column_names):
        
        assert is_string_or_string_list(column_names),\
            "arg 'column_names' should be a string or a list of strings"
            
        column_names = enlist(column_names)
        
        assert len(set(column_names)) == len(column_names),\
            "arg 'column_names' should have unique strings"
        
        assert set(self.get_colnames()).issuperset(column_names),\
            "arg 'column_names' should contain valid column names"
        
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
        Split by some columns, apply a function per chunk
        which returns a dataframe and
        then combine it back into a single dataframe
        
        Parameters
        ----------
        func: callable
            Type 1. A function: TidyDataFrame --> TidyDataFrame
            Type 2. A function: simple pandas df --> simple pandas df
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
        
        Notes
        -----
        1. Chunks will always include the grouping columns.
        2. If grouping columns are found in output of 'func', then they are
           removed and replaced with value in input chunk.
        3. When 'preserve_row_order' is True, a temporary column is added
           to each chunk. Its should be retained and not tampered with.
        
        Examples
        --------
        from palmerpenguins import load_penguins
        penguins_tidy = tidy(load_penguins().convert_dtypes())
        
        # pick a sample of rows per chunk defined by 'species'
        penguins_tidy.apply_over_groups(lambda x: x.sample(n = 3)
                                        , by = 'species'
                                        )
        
        # apply a pandas udf per chunk defined by 'species'
        # groupby columns are always added
        penguins_tidy.apply_over_groups(lambda x: x.loc[:, ['year']]
                                        , by = ['species', 'sex']
                                        , is_pandas_udf = True
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
          .apply_over_groups(lambda x: x.pipe_tee(print).sample(2)
                             , by = 'sex'
                             , preserve_row_order = True
                             )
          )
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
                
                cols_in_order = util_copy.copy(by)
                cols_in_order.extend(set(res.columns).difference(by))
                return res.loc[:, cols_in_order]
        else:
            def wrapper_func(chunk, **kwargs):
                # i/o are pdfs
                chunk_tidy = TidyPandasDataFrame(chunk.reset_index(drop = True)
                                                 , check = False
                                                 , copy = False
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
        
        nr = self.get_nrow()
        if preserve_row_order:
            res = (self.__data
                        .assign(**{row_order_column_name: np.arange(nr)})
                        .groupby(by, sort = False, dropna = False)
                        .apply(wrapper_func)
                        .reset_index(drop = True)
                        )

            if row_order_column_name in list(res.columns):
                res = (res.sort_values(row_order_column_name
                                       , ignore_index = True
                                       )
                          .drop(columns = row_order_column_name)
                          )
            else:
                raise Exception(("'row_order_column_name' in each chunk should "
                                 "be retained, when preserve_row_order is True"
                                 ))
        else:
            res = (self.__data
                        .groupby(by, sort = False, dropna = False)
                        .apply(wrapper_func)
                        .reset_index(drop = True)
                        )
        
        if (not is_simple(res)) and is_pandas_udf:
            raise Exception(("Resulting dataframe after apply should be "
                             "'simple', examine the pandas UDF"))
            
        return TidyPandasDataFrame(res.convert_dtypes()
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
        flights_tidy.select(predicate = lambda x: x.dtype != "object")
        
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
        # return a copy due to loc
        return TidyPandasDataFrame(res, check = False, copy = True)
    
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
            new_colnames = util_copy.copy(column_names)
            new_colnames.extend(list(cc_trunc_setlist))
        elif (before is not None):
            # case 2: before is not None
            pos_before   = int(np.where([x == before for x in cc_trunc_setlist])[0])
            cc_left      = list(cc_trunc_setlist[ :pos_before])
            cc_right     = list(cc_trunc_setlist[pos_before: ])
            new_colnames = util_copy.copy(cc_left)
            new_colnames.extend(column_names)
            new_colnames.extend(cc_right)
        else:
            # case 3: after is not None
            pos_after    = int(np.where([x == after for x in cc_trunc_setlist])[0])      
            cc_left      = list(cc_trunc_setlist[ :(pos_after + 1)])
            cc_right     = list(cc_trunc_setlist[(pos_after + 1): ])
            new_colnames = util_copy.copy(cc_left)
            new_colnames.extend(column_names)
            new_colnames.extend(cc_right)
      
        res = self.__data.loc[:, new_colnames]
        # return a copy due to loc
        return TidyPandasDataFrame(res, check = False, copy = True)
    
    def rename(self, old_new_dict):
        '''
        rename
        Rename columns of the tidy pandas dataframe
        
        Parameters
        ----------
        old_new_dict: A dictionary with old names as keys and new names as values
        
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
        col_names = self.get_colnames()
        assert isinstance(old_new_dict, dict)
        assert set(col_names).issuperset(old_new_dict.keys()) # old names should be there
        assert is_unique_list(list(old_new_dict.values())) # new names should be unique
        # TODO some more checks on column names are required
        
        # new names should not intersect with 'remaining' names
        remaining = set(col_names).difference(old_new_dict.keys())
        assert len(remaining.intersection(old_new_dict.values())) == 0
        
        res = self.__data.rename(columns = old_new_dict)
        return TidyPandasDataFrame(res, check = False, copy = False)
    
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
            row_numbers = enlist(row_numbers)
        
        if by is None:
            minval = np.min(row_numbers)
            maxval = np.max(row_numbers)
            assert minval >= 0 and maxval <= self.get_nrow()
            
            res = self.__data.iloc[row_numbers,:].reset_index(drop = True)
        else:
            by = enlist(by)
            self._validate_by(by)
            
            min_group_size = (self.__data
                                  .groupby(by, sort = False, dropna = False)
                                  .size()
                                  .min()
                                  )
            if np.max(row_numbers) > min_group_size:
                print("Minimum group size is: ", min_group_size)
                raise Exception("Maximum row number to slice per group should not exceed the number of rows of the group")
            
            res = (self.__data
                       .assign(**{"rn__": lambda x: np.arange(x.shape[0])})
                       .groupby(by, sort = False, dropna = False)
                       .apply(lambda chunk: chunk.iloc[row_numbers,:])
                       .reset_index(drop = True)
                       .sort_values("rn__", ignore_index = True)
                       .drop(columns = "rn__")
                       )
        
        return TidyPandasDataFrame(res, check = False, copy = False)
        
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
        
        self._validate_column_names(column_names)
        column_names = enlist(column_names)
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
        
        res = self.__data.sort_values(by             = column_names
                                      , axis         = 0
                                      , ascending    = ascending
                                      , inplace      = False
                                      , kind         = 'quicksort'
                                      , na_position  = na_position
                                      , ignore_index = True
                                      )
            
        return TidyPandasDataFrame(res, check = False, copy = False)
        
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
        penguins_tidy.filter(mask = penguins_tidy.pull("year") == 2007)
        
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
            by = enlist(by)
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
            assert is_string_or_string_list(column_names),\
                "arg 'column_names' should be a string or a list of strings"
            column_names = list(setlist(enlist(column_names)))
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

    def mutate(self
               , dictionary = None
               , func = None
               , column_names = None
               , predicate = None
               , prefix = ""
               , by = None
               ):
        if dictionary is None and func is None:
            raise Exception(("Either dictionary or func with "
                             "predicate/column_names should be provided."
                            ))
        if by is None:
            if dictionary is not None:
                res = self._mutate(dictionary)
            else:
                res = self._mutate_across(func
                                          , column_names = column_names
                                          , predicate = predicate
                                          , prefix = prefix
                                          )
        else:
            self._validate_by(by)
            by = enlist(by)
            cn = self.get_colnames()
            
            if dictionary is not None:
                res = self.apply_over_groups(
                    func = lambda chunk: chunk._mutate(dictionary)
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = generate_new_string(cn)
                    , is_pandas_udf = False
                    )
            else:
                res = self.apply_over_groups(
                    func = lambda chunk: (
                        chunk._mutate_across(func
                                             , column_names = column_names
                                             , predicate = predicate
                                             , prefix = prefix
                                             )
                        )
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = generate_new_string(cn)
                    , is_pandas_udf = False
                    )

        return res
        
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
        nr = self.get_nrow()
        cn = self.get_colnames()
        mutated = util_copy.copy(self.__data)
        
        # akey is a column name to assign to
        # res should be a pandas series, 1D numpy array or a scalar
        def assign_column(df, akey, res):
            if isinstance(res, (pd.Series, np.ndarray)):
                if isinstance(res, np.ndarray):
                    assert res.ndim == 1,\
                        ("When result of RHS for key '{akey}' is a numpy" 
                         "it should be a 1D array"
                         )
                assert len(res) == 1 or len(res) == df.shape[0],\
                    (f"When a pandas series or numpy array, result of RHS "
                     "for key '{akey}' should be 1 or equal to number of rows"
                     " of the dataframe"
                     )
                df[akey] = res
            elif np.isscalar(res):
                df[akey] = res
            else:
                raise Exception((f"Result of RHS  for key '{akey}' should be "
                                 "a pandas series, numpy array or a scalar"
                                ))
            return None
            
            
        for akey in dictionary:
            
            assert isinstance(akey, (str, tuple)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            assert is_string_or_string_list(list(akey)),\
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
                    assign_column(mutated, akey, rhs)
                
                # 2. assign via function
                elif callable(rhs):
                    assign_column(mutated, akey, rhs(mutated))
                
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more than "
                         "two elements"
                         )
                    assert callable(rhs[0]),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function"
                         )
                    # 3a. assign a preexisting column
                    if len(rhs) == 1:
                        assert akey in list(mutated.columns),\
                            (f"When RHS is a tuple with function alone, {akey} "
                             "should be an exisiting column"
                             )
                        assign_column(mutated, akey, rhs[0](mutated[akey]))
                    # 3b. assign with column args
                    else:
                        assert is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        assign_column(mutated
                                      , akey
                                      , rhs[0](*[mutated[x] for x in cols])
                                      )
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
            
            # multiple assignments
            else:
                # 1. direct assign is not supported for multi assignment
                
                # 2. assign via function
                if callable(rhs):
                    rhs_res = rhs(mutated)
                    assert (isinstance(rhs_res, list) 
                            and len(rhs_res) == len(akey)
                            ),\
                        ("RHS should output a list of length equal to "
                         "length of LHS for key: {akey}"
                         )
                    for apair in zip(akey, rhs_res):
                        assign_column(mutated, apair[0], apair[1])
                        
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more "
                         "than two elements"
                         )
                    assert callable(rhs[0]),\
                        (f"When RHS is a tuple, first element of RHS should "
                         "be a function"
                         )
                    # 3a. assign a preexisting columns
                    if len(rhs) == 1:
                        assert set(akey).issubset(list(mutated.columns)),\
                            (f"When RHS is a tuple with function alone, {akey}"
                             "should be an exisiting columns"
                             )
                        rhs_res = rhs[0](*[mutated[acol] for acol in akey])
                        assert (isinstance(rhs_res, list)
                                and len(rhs_res) == len(akey)
                                ),\
                            (f"RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            assign_column(mutated, apair[0], apair[1])
                    # 3b. assign with column args
                    else:
                        assert is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        rhs_res = rhs[0](*[mutated[x] for x in cols])
                        assert (isinstance(rhs_res, list) 
                                and len(rhs_res) == len(akey)
                                ),\
                            ("RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            assign_column(mutated, apair[0], apair[1])
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
        
        col_order = util_copy.copy(cn)
        col_order.extend(list(set(list(mutated.columns)).difference(cn)))
                
        return TidyPandasDataFrame(mutated.loc[:, col_order]
                                   , check = False
                                   , copy = False
                                   )

    # basic extensions    
    def _mutate_across(self
                       , func
                       , column_names = None
                       , predicate = None
                       , prefix = ""
                       ):

        assert callable(func),\
            "arg 'func' should be a function"
        assert isinstance(prefix, str)

        mutated = util_copy.copy(self.__data)
        
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
            assert is_string_or_string_list(on_x),\
                "arg 'on_x' should be a string or a list of strings"
            assert is_string_or_string_list(on_y),\
                "arg 'on_y' should be a string or a list of strings"
            
            on_x = enlist(on_x)
            on_y = enlist(on_y)
            
            assert is_unique_list(on_x),\
                "arg 'on_x' should not have duplicates"
            assert is_unique_list(on_y),\
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
            assert is_string_or_string_list(on),\
                "arg 'on' should be a string or a list of strings"
            on = enlist(on)
            assert is_unique_list(on),\
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
              , sort = False
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
            on_x = enlist(on_x)
            on_y = enlist(on_y)
            if on_x == on_y:
                on = util_copy.copy(on_x)
                on_x = None
                on_y = None
        
        if on is None:
            on_x = enlist(on_x)
            on_y = enlist(on_y)
            new_colnames = cn_x + list(setlist(cn_y).difference(on_y))
        else:
            on = enlist(on)
            new_colnames = cn_x + list(setlist(cn_y).difference(on))
        
        if sort:
            if on is not None:
                res = (self
                       .__data
                       .assign(**{"rn_x__": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"rn_y__": nr_y})
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
                       .sort_values(by = ["rn_x__", "rn_y__"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["rn_x__", "rn_y__"])
                       )
            else:
                res = (self
                       .__data
                       .assign(**{"rn_x__": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"rn_y__": nr_y})
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
                       .sort_values(by = ["rn_x__", "rn_y__"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["rn_x__", "rn_y__"])
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
                       
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
                       
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
                       
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
                       
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
        
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
                   , sort = False
                   , suffix_y = "_y"
                   ):
        
        self._validate_join(y = y
                            , how = "inner" # not significant
                            , on = on
                            , on_x = on_x
                            , on_y = on_y
                            , sort = sort
                            , suffix_y = suffix_y
                            )
        
        string = generate_new_string(y.get_colnames())
        
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

    # binding methods
    def cbind(self, y):
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
                 )
        return TidyPandasDataFrame(res, check = False)
    
    def rbind(self, y):
        res = (pd.concat([self.__data, y.to_pandas()]
                         , axis = 0
                         , ignore_index = True
                         )
                 .convert_dtypes()
                 )
        return TidyPandasDataFrame(res, check = False)
    
    # count
    def count(self
              , column_names = None
              , count_column_name = 'n'
              , ascending = False
              ):
        
        assert (column_names is None
                or is_string_or_string_list(column_names)
                ),\
            "arg 'column_names' is either None or a list of strings"
        if column_names is not None:
            self._validate_column_names(column_names)
            column_names = enlist(column_names)
        assert isinstance(count_column_name, str),\
            "arg 'count_column_name' should a string"
        assert isinstance(ascending, bool),\
            "arg 'ascending' should be a boolean"
        
        if column_names is not None:
            assert count_column_name not in column_names,\
                ("arg 'count_column_name' should not be an element of arg "
                 "'column_names'"
                 )
            
            res = (self.__data
                       .groupby(column_names
                                , sort = False
                                , dropna = False
                                )
                       .size()
                       .reset_index(drop = False)
                       .rename(columns = {0: count_column_name})
                       )
            
            res = res.sort_values(by = count_column_name
                                  , axis         = 0
                                  , ascending    = ascending
                                  , inplace      = False
                                  , kind         = 'quicksort'
                                  , na_position  = 'first'
                                  , ignore_index = True
                                  )
        else:
            res = pd.DataFrame({count_column_name : self.get_nrow()}
                               , index = [0]
                               )
            
        return TidyPandasDataFrame(res, check = False)

    def add_count(self
                  , column_names = None
                  , count_column_name = 'n'
                  , ascending = False
                  ):
        
        count_frame = self.count(column_names = column_names
                                 , count_column_name = count_column_name
                                 , ascending = ascending
                                 )
                                 
        if column_names is None:
            res = self.mutate(
                {count_column_name : (count_frame.to_pandas(copy = False)
                                                 .iloc[0,0]
                                                 )}
                )
        else:
            res = self.inner_join(count_frame, on = column_names)
        
        return res
    
    # pivot methods
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = None
                    , id_cols = None
                    , drop_na = True
                    , retain_levels = False
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
        
        values_fn: function or a dict of functions, default is None
            A function to handle multiple values per row in the result.
            When a dict, keys should be same as arg 'values_from'
            When None, multiple values are in kept a list and a single value is
            not kept in a list
        
        id_cols: string or list of strings, default is None
            
        '''
        
        cn = self.get_colnames()
        
        assert is_string_or_string_list(names_from),\
            "arg 'names_from' should be string or a list of strings"
        names_from = enlist(names_from)
        assert is_unique_list(names_from),\
            "arg 'names_from' should have unique strings"
        assert set(names_from).issubset(cn),\
            "arg 'names_from' should be a subset of existing column names"
        
        assert is_string_or_string_list(values_from),\
            "arg 'values_from' should be string or a list of strings"
        values_from = enlist(values_from)
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
            assert is_string_or_string_list(id_cols),\
                "arg 'id_cols' should be string or a list of strings"
            id_cols = enlist(id_cols)
            assert is_unique_list(id_cols),\
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
        
        assert isinstance(drop_na, bool),\
            "arg 'drop_na' should be a bool"
        assert isinstance(retain_levels, bool),\
            "arg 'retain_levels' should be a bool"
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
                             , dropna     = drop_na
                             , observed   = retain_levels
                             )
                             
        return tidy(res, sep = sep, verbose = False, copy = False)
    
    def pivot_longer(self
                     , cols
                     , names_to = "name"
                     , values_to = "value"
                     , include = True
                     ):
        
        # assertions
        cn = self.get_colnames()
        assert is_string_or_string_list(cols),\
            "arg 'cols' should be a string or a list of strings"
        assert set(cols).issubset(cn),\
            "arg 'cols' should be a subset of exisiting column names"
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
                   )
        
        # convert missing value to pd.NA
        res[values_to] = pd.Series(np.where(pd.isna(res[values_to])
                                             , pd.NA
                                             , res[values_to]
                                             )
                                    )
                                    
        return TidyPandasDataFrame(res, check = False, copy = False)
    
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
    
    sample = slice_sample
    
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
            ("arg 'ties_method' should be one among: "
             "'all' (default), 'first', 'last'"
            )
        
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
                    ("arg 'ties_method' should be one among: "
                     " 'round' (default), 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                res = self.apply_over_groups(
                    lambda x: x.nsmallest(int(roundf(x.shape[0] * prop))
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
                    ("arg 'n' should not exceed the "
                     "size of any chunk after grouping"
                    )
                
                res = self.apply_over_groups(
                    lambda x: x.nsmallest(n
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
    def any_na(self):
        res = (self.__data
                   .isna() # same dim as input
                   .any()  # series of booleans per column
                   .any()  # single value
                   )
        return bool(res)
    
    def replace_na(self, column_replace_dict):
        
        assert isinstance(column_replace_dict, dict)
        cn = self.get_colnames()
        self_copy = util_copy.copy(self.__data)
        for akey in column_replace_dict:
            self_copy[akey] = self_copy[akey].fillna(column_replace_dict[akey])
        
        return TidyPandasDataFrame(self_copy)
    
    def drop_na(self, column_names = None):
        
        cn = self.get_colnames()
        if column_names is not None:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            assert set(column_names).issubset(cn)
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
    
    # nest and unnest ----
    
    def nest(self
             , column_names = None
             , by = None
             , nest_column_name = 'data'
             ):
        
        cn = self.get_colnames()
        
        if (((column_names is None) and (by is None))
            or ((column_names is not None) and (by is not None))):
            raise Exception(("Exactly one arg among 'column_names' and 'by' "
                             "should be None"
                             ))
        if by is not None:
            self._validate_by(by)
            by = enlist(by)
        else:
            self._validate_column_names(column_names)
            column_names = enlist(column_names)
            by = list(setlist(cn).difference(column_names))
        
        # work with by from here on 
        
        assert nest_column_name not in cn,\
            "arg 'nest_column_name' should not be an exisiting column name"
        
        # make a df from the distinct values
        go = self.__data.groupby(by, sort = False, dropna = False)
        res = pd.DataFrame(list(go.groups.keys()), columns = by)
        # add data into nest_column_name column
        res[nest_column_name] = (
            pd.Series(map(lambda x: TidyPandasDataFrame(x.drop(columns = by)
                                                        , check = False
                                                        )
                                         , dict(tuple(go)).values()
                                         )
                                    )
            )
        
        # non-exisiting groups result in NaN, make them empty dataframe                            
        cols   = setlist(cn).difference(by)
        na_pos = np.where(res[nest_column_name].isna().to_numpy())[0]
        for x in na_pos:
            res[nest_column_name][x] = (
                TidyPandasDataFrame(pd.DataFrame(columns = cols)
                                                      , check = False
                                                      ))
        
        return TidyPandasDataFrame(res, check = False, copy = False)
    
    def unnest(self, nest_column_name = 'data'):
        
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
            cols = setlist(cn).difference([nest_column_name])
            # create a filled up dataframe per row via cross join and rbind them
            res = pd.concat(
                map(lambda x: pd.merge(
                        self.__data.loc[[x], cols]
                        , (self.__data[nest_column_name][x]).to_pandas()
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
                
        return TidyPandasDataFrame(res, check = False, copy = False)
