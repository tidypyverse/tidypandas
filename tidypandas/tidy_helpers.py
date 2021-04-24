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
    elif isinstance(x, list) and len(x) > 1:
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

def simplify(pdf, sep = "__", verbose = False):
    '''
    simplify(pdf)
    
    Returns a pandas dataframe with simplified index structure. This might be helpful before creating a TidyDataFrame or a TidyGroupedDataFrame object.
    
    Parameters
    ----------
    pdf : pd.DataFrame or pd.core.groupby.DataFrameGroupBy
        A pandas dataframe or grouped pandas dataframe

    Returns
    -------
    A pandas dataframe with simplified index structure. If the input dataframe is grouped, then output is grouped too.
    
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
    
    ex4 = ex2.groupby('origin')
    ex4
    ex4.head()
    simplify(ex4)
    simplify(ex4).head()
    '''
    
    assert is_pdf(pdf)
    assert isinstance(sep, str)
    
    is_grouped = False
    if is_grouped_pdf(pdf):
        is_grouped = True
        gvs = pdf.grouper.names
        pdf = pdf.obj
    
    try:
        # handle column multiindex
        if isinstance(pdf.columns, pd.MultiIndex):
            # paste vertically with 'sep' and get unique names
            # a   d 
            # b c e
            # becomes
            # a__b,a__c, d__e
            lol = list(map(list, list(pdf.columns)))
            cns = list(map(lambda x: sep.join(map(str, x)).rstrip(sep), lol))
            pdf.columns = get_unique_names(cns)
        else:
            # avoid column index from having some name
            pdf.columns.name = None
            pdf.columns = get_unique_names(list(pdf.columns))
    except:
        if verbose:
            raise Exception("Unable to simplify: column index or multiindex")
    try:    
        # handle row multiindex 
        flag_row_multiindex = isinstance(pdf.index, pd.MultiIndex)
        flag_complex_index = True
        if isinstance(pdf.index, pd.RangeIndex):
            if pdf.index.start == 0 and pdf.index.step == 1:
                flag_complex_index = False
                  
        if flag_row_multiindex or flag_complex_index:
            # first, attempt to not drop the index, then drop when former fails
            try:
                pdf = pdf.reset_index(drop = False)
            except:
                pdf = pdf.reset_index(drop = True)
                if verbose:
                    warnings.warn("Dropped the row index")
    except:
        if verbose:
            raise Exception("Unable to simplify: row index or multiindex")
        
    if is_grouped:
        pdf = pdf.groupby(gvs)
        
    if verbose:
        print("Successfully simplified!")
    return pdf

def is_simple(pdf, verbose = False):
    '''
    is_simple
    Whether the input pandas dataframe is 'simple' or not

    Parameters
    ----------
    pdf : pandas ungrouped or grouped dataframe
        DESCRIPTION.
    verbose : bool, optional
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
    assert isinstance(pdf, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy))
    if isinstance(pdf, pd.core.groupby.DataFrameGroupBy):
        pdf = pdf.obj
    
    row_flag = not isinstance(pdf.index, pd.MultiIndex)
    columns  = list(pdf.columns)
    col_flag = not isinstance(columns, pd.MultiIndex)
    # check if row index is rangeIndex
    flag_no_index = False
    if isinstance(pdf.index, pd.RangeIndex):
        if pdf.index.start == 0 and pdf.index.step == 1:
            flag_no_index = True           
    # check if all column names are strings
    str_flag = all([isinstance(y, str) for y in columns])
    # check if column names are unique
    if len(set(columns)) == len(columns):
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