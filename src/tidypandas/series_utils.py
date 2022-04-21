# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import functools
from tidypandas._unexported_utils import (
                                            _is_kwargable,
                                            _is_valid_colname,
                                            _is_string_or_string_list,
                                            _enlist,
                                            _get_unique_names,
                                            _is_unique_list,
                                            _get_dtype_dict,
                                            _generate_new_string,
                                            _coerce_series,
                                            _coerce_pdf
                                        )

def _extend(aseries, length):
    '''
    _extend
    Extends a series to a given length by repeating the values of the series
    
    Parameters
    ----------
    aseries: A pandas series
    length: int
        length to repeat to
    
    Returns
    -------
    A pandas series
    
    Examples
    --------
    >>> ser = pd.Series([1,2,3])
    >>> _extend(ser, 5)
    '''
    assert isinstance(aseries, pd.Series)
    assert isinstance(length, int)
    assert length > 0
    
    x = aseries.reset_index(drop = True)
    
    if len(x) >= length:
        res = x[0:length]
    else:
        ell = len(x)
        times = int(np.ceil(length / ell))
        y = x.copy()
        for i in range(times - 1):
            y = y.append(x, ignore_index = True)
        y = y[0:length]
        res = y
    return res

def ifelse(condition, yes, no):
    '''
    ifelse
    Vectorized if and else
    
    Parameters
    ----------
    condition: expression or list/array/Series
        Should evaluate to a boolean list/array/Series
    yes: expression or list/array/Series
        Should evaluate to a list/array/Series
    no: expression or list/array/Series
        Should evaluate to a list/array/Series
    
    Returns
    -------
    Pandas Series
        
    Notes
    -----
    1. Thin wrapper over np.where
    2. Does not preserve index.
    
    Examples
    --------
    >>> x = pd.Series([1,pd.NA,3]).astype('Int64')
    >>> y = pd.Series([4,5,6]).astype('Int64')
    >>> z = pd.Series([7,8,9]).astype('Int64')
    >>> 
    >>> ifelse(x > 2, y + 1, z - 1)
    '''
    cond = _coerce_series(pd.Series(condition))
    if not isinstance(yes, pd.Series):
        yes_series = pd.Series([yes] * len(cond))
    else:
        yes_series = yes
    if not isinstance(no, pd.Series):
        no_series = pd.Series([no] * len(cond))
    else:
        no_series = no
    
    res = [pd.NA] * len(cond)
    for i in range(len(cond)):
        if pd.isna(cond[i]):
            pass
        elif cond[i] == 1:
            res[i] = yes_series[i]
        else:
            res[i] = no_series[i]
    return _coerce_series(pd.Series(res))

if_else = ifelse
   
def coalesce(list_of_series):
    '''
    coalesce
    Given a list of pandas Series, coalesce finds the first non-missing value
    at each position for the first Series in the list
    
    Parameters
    ----------
    list_of_series: list
        List of pandas Series
    
    Returns
    -------
    Pandas Series of length of the first series in the input list
        
    Notes
    -----
    1. If on the inputs has length 1, then it is extended. 
       Else, inputs are expected to have same length.
    3. Does not preserve index.
    
    Examples
    --------
    >>> x = pd.Series([1, pd.NA, pd.NA])
    >>> y = pd.Series([4, 5    , pd.NA])
    >>> z = pd.Series([7, 8    , pd.NA])
    >>> a = 10
    >>> 
    >>> coalesce([x, y, z, a])
    '''
    assert isinstance(list_of_series, list)
    assert all([isinstance(x, pd.Series) or np.isscalar(x) for x in list_of_series])
    list_of_series = [pd.Series(x) for x in list_of_series]
    
    max_len = max(map(len, list_of_series))
    for index, aseries in enumerate(list_of_series):
        assert len(aseries) == 1 or len(aseries) == max_len,\
            ("A series should have length 1 or length equal to maximum length "
             "of all elements in the list"
             )
        if len(aseries) == 1:
            list_of_series[index] = pd.Series([aseries[0]] * max_len)
    
    
    list_of_series = list(map(lambda x: x.fillna(pd.NA), list_of_series))
    res = (functools.reduce(lambda x, y: x.combine_first(_extend(y, len(x)))
                            , list_of_series[1:]
                            , list_of_series[0]
                            )
                    .fillna(pd.NA)
                    .convert_dtypes()
                    .reset_index(drop = True) 
                    )
    return res

def case_when(list_of_tuples, default = pd.NA):
    '''
    case_when
    Vectorized version of multiple ifelse statements
    
    Parameters
    ----------
    list_of_tuples: list
        First element of the tuple should be an expression when evaluated should
        result in a boolean array/Series.
        Second element of the tuple should be the value to be assigned to the 
        postitions corresponding the boolean array/Series
    default: scalar value to assign when none of the conditions are met. 
        Default is pd.NA.
    
    Returns
    -------
    Pandas Series
        
    Notes
    -----
    1. Does not preserve index.
    
    Examples
    --------
    >>> x = pd.Series([3, 5 , pd.NA, 4]).astype('Int64')
    >>> 
    >>> case_when([(x >= 5, 500), (x >= 4, 400), (pd.isna(x), 1)])
    >>> case_when([(x >= 5, 500), (x >= 4, 400)], default = 100)
    '''
    assert isinstance(list_of_tuples, list)
    assert all([isinstance(atuple, tuple) for atuple in list_of_tuples])
    assert all([len(atuple) == 2 for atuple in list_of_tuples])
    
    res = coalesce(list(map(lambda acase: ifelse(_coerce_series(acase[0]), acase[1], pd.NA)
                            , list_of_tuples
                            )
                        )
                  )

    res = res.fillna(default).reset_index(drop = True)
    return res
    
def n_distinct(x, na_rm = False):
    '''
    n_distinct
    Number of distinct values in a series
    
    Parameters
    ----------
    x: A pandas series
    na_rm (default is False): bool, Should missing value be counted
    
    Returns
    -------
    int, Number of distinct values
    '''
    assert isinstance(x, pd.Series)
    return x.nunique(dropna = False)

def _order_series(x, na_position = "last"):
    '''
    _order_series
    Permutation order of the series
    
    Parameters
    ----------
    x: A pandas series
    na_position: str, One among "first", "last"
    
    Returns
    -------
    A pandas series indicating the permutation order
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, pd.NA]).astype('Int64')
    >>> _order_series(ser)
    '''
    assert isinstance(x, pd.Series)
    assert isinstance(na_position, str)
    assert na_position in ["first", "last"]
    
    res = pd.Series(x.sort_values(na_position = na_position).index)
    res = res.sort_values(na_position = na_position).index
    res = pd.Series(res).astype('Int64')
    return res

def order(x, ascending = True, na_position = "last"):
    '''
    order
    Permutation order of the series or a list of series
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    na_position: str
        One among: "first", "last"
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    A pandas series indicating the permutation order
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, pd.NA]).astype('Int64')
    >>> order(ser)
    >>> 
    >>> sers = [pd.Series([1, 1]).astype('Int64'), pd.Series([2, 1]).astype('Int64')]
    >>> order(sers)
    '''
    assert isinstance(x, (pd.Series, list)),\
        "x should be a pandas series or a list of series"
    if isinstance(x, list):
        assert all([isinstance(y, pd.Series) for y in x]),\
            "When a list, each element should be pandas series"
        assert len({ len(y) for y in x }) == 1,\
            "When a list, each element should be pandas series of same length"
            
    if isinstance(ascending, list):
        assert len(ascending) == len(x),\
            "'ascending' should be a list of bools with same length as 'x'"
        assert all([isinstance(y, bool) for y in ascending]),\
            "When 'ascending' is a list, each element should be a bool"
    else:
        assert isinstance(ascending, bool),\
            "'ascending' should be a bool"
        
    assert isinstance(na_position, str),\
        "'na_position' should be a string"
    assert na_position in ["first", "last"],\
        "'na_position' should be one among: 'first', 'last'"
    
    if isinstance(x, pd.Series):
        res = x.sort_values(na_position = na_position
                            , ascending = ascending
                            ).index
        res = pd.Series(res).astype('Int64')
    else:
        df = pd.concat(x, axis = "columns")
        res = df.sort_values(by = list(df.columns)
                             , na_position = na_position
                             , ascending = ascending
                             ).index
        res = pd.Series(res).astype('Int64')
    
    return res

def _rank(x, type, ascending = True, percent = False):
    '''
    _rank
    ranking order of the series or a list of series
    
    Parameters
    ----------
    x: A pandas series or a list of series
    type: One among "min", "first", "dense", "max"
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    A pandas series
    
    Notes
    -----
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, pd.NA]).astype('Int64')
    >>> _rank(ser, "first")
    >>> 
    >>> sers = [pd.Series([1, 1]).astype('Int64'), pd.Series([2, 1]).astype('Int64')]
    >>> _rank(sers, "min")
    '''
    assert isinstance(x, (pd.Series, list)),\
        "x should be a series or a list of pandas series"
    if isinstance(x, list):
        assert all([isinstance(y, pd.Series) for y in x]),\
            "When x is a list, each element should be pandas series"
        assert len({ len(y) for y in x }) == 1,\
            "When x is a list, all series should have equal length"
    
    if isinstance(ascending, list):
        assert len(ascending) == len(x),\
            "'ascending' should be a list of bools with same length as 'x'"
        if isinstance(ascending, list):
            assert all([isinstance(y, bool) for y in ascending]),\
                "When ascending is a list, it should be list of bools"
    else:
        assert isinstance(ascending, bool),\
            "ascending should be a bool"
            
    assert type in ["min", "first", "dense", "max"],\
        "type should be one among: min, first, dense, max"
    
    if isinstance(x, pd.Series):
        res = x.rank(method = type
                     , na_option = "keep"
                     , ascending = ascending
                     , pct = percent
                     )
    else:
        df = pd.DataFrame(x).T
        if isinstance(ascending, bool):
            df = df.apply(lambda x: x.rank(method = "dense"
                                           , na_option = "keep"
                                           , ascending = ascending
                                           )
                          )
        else:
            for acol, aorder in zip(list(df.columns), ascending):
                df[acol] = df[acol].rank(method = "dense"
                                         , na_option = "keep"
                                         , ascending = aorder
                                         )
        
        res = (df.apply(tuple, axis = 1)
                 .rank(method = type
                       , na_option = "keep"
                       , ascending = True
                       , pct = percent
                       )
                 )
    
    return res.convert_dtypes()

def min_rank(x, ascending = True):
    '''
    min_rank
    ranking order with ties set to minimum rank
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, 1, 3, pd.NA, 2]).astype('Int64')
    >>> min_rank(ser)
    '''
    return _rank(x, "min", ascending)
    
def row_number(x, ascending = True):
    '''
    row_number
    ranking order with ties set to first rank
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, 1, 3, pd.NA, 2]).astype('Int64')
    >>> row_number(ser)
    '''
    return _rank(x, "first", ascending)

def dense_rank(x, ascending = True):
    '''
    dense_rank
    ranking order with ties so that there is no gaps in ranks
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([4, 1, 2, 1, 3, pd.NA, 2]).astype('Int64')
    >>> dense_rank(ser)
    '''
    return _rank(x, "dense", ascending)

def percent_rank(x, ascending = True):
    '''
    percent_rank
    ranking order witbh ties set to minimum rank rescaled between 0 and 1
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, 1, 3, pd.NA, 2]).astype('Int64')
    >>> percent_rank(ser)
    '''
    return _rank(x, "min", ascending, True)

def cume_dist(x, ascending = True):
    '''
    cume_dist
    ranking order with ties set to maximum rank rescaled between 0 and 1
    
    Parameters
    ----------
    x: A pandas series or a list of series
    ascending: str or a list of bools
        When a list, should have match the length of x
    percent: bool
        Should the ranking order be converted to percentages
        
    Notes
    -----
    1. When x is a list of series, 2nd series is used to beak the ties in the 1st
    series and so on.
    
    Returns
    -------
    Missing values are left as is. If you want to treat them as the smallest
    or largest values, replace with Inf or -Inf before ranking.
    
    Examples
    --------
    >>> ser = pd.Series([3, 1, 2, 1, 3, pd.NA, 2]).astype('Int64')
    >>> cume_dist(ser)
    '''
    return _rank(x, "max", ascending, True)

def as_bool(x):
    '''
    as_bool
    Convert boolean or object dtype series to bool series with NAs as False
    Helpful in combining multiple series in 'filter'
    
    Parameters
    ----------
    x: pandas series
        Should have one of these dtypes: bool, boolean, object
    
    Returns
    -------
    A pandas series of dtype bool
    
    Examples
    --------
    ser = pd.Series([True, False, pd.NA])
    print(str(ser.dtype))
    
    print(as_bool(ser))
    print(str(as_bool(ser).dtype))
    
    ser2 = ser.convert_dtypes()
    print(str(ser2.dtype))
    
    print(as_bool(ser2))
    print(str(as_bool(ser2).dtype))
    '''
    assert isinstance(x, pd.Series),\
        "input should be a pandas series"
    assert isinstance(str(x.dtype), ('bool', 'boolean', 'object')),\
        "input should have one of these dypes: 'bool, boolean, object"
    if str(x.dtype) == "bool":
        res = x
    else:
        res = pd.Series([False if pd.isna(y) else bool(y) for y in x])
    return res
