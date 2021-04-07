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
    elif isinstance(x, list):
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

def is_pdf(x):
    return isinstance(x, (pandas.DataFrame, pandas.core.groupby.DataFrameGroupBy))

def is_ungrouped_pdf(x):
    return isinstance(x, pandas.DataFrame)

def is_grouped_pdf(x):
    return isinstance(x, pandas.core.groupby.DataFrameGroupBy)

def is_unique_list(x):
    assert isinstance(x, list)
    return len(set(x)) == len(x)

def tidy(pdf, sep = "__", verbose = False):
    '''
    tidy(pdf)
    
    Returns a pandas dataframe with simplified index structure. This might be helpful before creating a TidyDataFrame object.
    
    Parameters
    ----------
    strings : pandas.DataFrame or pandas.core.groupby.DataFrameGroupBy
        A pandas dataframe or grouped pandas dataframe

    Returns
    -------
    A pandas dataframe with simplified index structure.If the input dataframe is grouped, then output is grouped too.
    
    Notes
    -----
    Returns a `tidy` pandas dataframe. A pandas dataframe is 'tidy' if:
    1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
    
    This is done by collapsing the column MultiIndex by concatenating names using separator 'sep' and ensuring that the resulting names are unique. The row Index or MultiIndex are added to the dataframe as columns if their names do not clash with the existing column names of the dataframe.
    
    Examples
    --------
    from nycflights13 import *

    ex1 = flights.groupby('dest').apply(lambda x: x.head(2))
    ex1
    tidy(ex1)
    
    ex2 = pd.crosstab(flights['origin'], flights['dest'])
    ex2
    tidy(ex2)
    
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
    tidy(ex3)
    
    ex4 = ex2.groupby('origin')
    ex4
    ex4.head()
    tidy(ex4)
    tidy(ex4).head()
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
            raise Exception("Unable to tidy: column index or multiindex")
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
            raise Exception("Unable to tidy: row index or multiindex")
        
    if is_grouped:
        pdf = pdf.groupby(gvs)
        
    if verbose:
        print("Successfully tidied!")
    return pdf
