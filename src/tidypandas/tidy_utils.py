# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from tidypandas._unexported_utils import (_is_unique_list,
                                          _get_unique_names, 
                                          _coerce_pdf
                                         )

# -----------------------------------------------------------------------------
# simplify
# -----------------------------------------------------------------------------

def simplify(pdf
             , sep = "__"
             , verbose = False
             ):
    '''
    simplify(pdf)
    
    Returns a pandas dataframe with simplified index structure.
    This might be helpful before creating a tidyframe object.
    
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
    >>> from nycflights13 import flights
    >>> ex1 = flights.groupby('dest').apply(lambda x: x.head(2))
    >>> ex1
    >>> simplify(ex1)
    >>> 
    >>> ex2 = pd.crosstab(flights['origin'], flights['dest'])
    >>> ex2
    >>> simplify(ex2)
    >>> 
    >>> ex3 = (flights.value_counts(['origin', 'dest', 'month', 'hour'])
    >>>               .reset_index()
    >>>               .rename(columns = {0 : 'n'})
    >>>               .pipe(lambda  x: pd.pivot_table(x
    >>>                                               , index = ['origin', 'month']
    >>>                                               , columns = ['dest', 'hour']
    >>>                                               , values = 'n'
    >>>                                               , fill_value = 0
    >>>                                               )
    >>>                     )
    >>>               )
    >>> ex3
    >>> simplify(ex3)
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
        raise Exception(f"Unable to simplify as column {acol} starts with " 
                        "an underscore"
                        )
            
    if verbose:
        print("Successfully simplified!")
    
    # simplify dtypes and maintain standard NAs
    pdf = pdf.convert_dtypes().fillna(pd.NA)
    
    pdf = _coerce_pdf(pdf)
    return pdf

# -----------------------------------------------------------------------------
# is_simple
# -----------------------------------------------------------------------------

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
    A pandas dataframe is 'simple' if:
    
        1. Column names (x.columns) are an unnamed pd.Index object of unique 
           strings. Column names do not start from "_".
        2. Row index is absent (pandas rangeindex starting from 0).
        
    Returns
    -------
    bool
        True if the input is 'simple' and False if the input is not 'simple'.
    
    Examples
    --------
    >>> from palmerpenguins import load_penguins
    >>> penguins = load_penguins().convert_dtypes()
    >>> ex1 = penguins.groupby('species').apply(lambda x: x.head(2))
    >>> ex1
    >>> 
    >>> is_simple(ex1, verbose = True)
    '''
    assert isinstance(pdf, pd.DataFrame),\
        "input should be a pandas dataframe"
    
    row_flag = not isinstance(pdf.index, pd.MultiIndex)
    col_flag = not isinstance(pdf.columns, pd.MultiIndex)
    # check if row index is rangeIndex
    flag_no_index = False
    if isinstance(pdf.index, pd.RangeIndex):
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
