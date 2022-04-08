# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------
import string

import numpy as np
import pandas as pd
import inspect
import pandas.api.types as dtypes
import warnings

def _is_kwargable(func):
    res = False
    spec = inspect.getfullargspec(func)
    if spec.varkw is not None:
        res = True
    return res

def _is_valid_colname(string):
    res = (isinstance(string, str)) and (len(string) != 0) and (string[0] != "_")
    return res
  
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
    >>> _is_string_or_string_list("bar")      # True
    >>> _is_string_or_string_list(["bar"])    # True
    >>> _is_string_or_string_list(("bar",))   # False
    >>> _is_string_or_string_list(["bar", 1]) # False
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
    >>> _enlist(["a"]) # ["a"]
    >>> _enlist("a")   # ["a"]
    >>> _enlist((1, )) # [(1, )]
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
    >>> _get_unique_names(['a', 'b'])               # ['a', 'b']
    >>> _get_unique_names(['a', 'a'])               # ['a', 'a_1']
    >>> _get_unique_names(['a', 'a', 'a_1'])        # ['a', 'a_1_1', 'a_1']
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

def _coerce_series(aseries):
    '''
    _coerce_series
    Convert the series type to its nullable type
    
    Parameters
    ----------
    aseries: A pandas series
    
    Returns
    -------
    A pandas series
    
    Notes
    -----
    If series cannot infer the type, it will return the series asis.
    '''
    # first try
    dt = str(aseries.convert_dtypes().dtype)
    
    # second try by dropping NA
    if dt == "object":
        dt = str(aseries.dropna().convert_dtypes().dtype)
    
    # warn if type cannot be detected        
    if dt == "object":
        warnings.warn(f"Could not infer the dtype, left asis")
    
    # return asis if not detected
    if dt == "object":
        ser = aseries
    else:
        ser = aseries.astype(dt)
    
    return ser


def _coerce_pdf(pdf):
    for acol in list(pdf.columns):
        pdf[acol] = _coerce_series(pdf[acol])
    return pdf
