
def bind_rows(x, rowid_column_name = "rowid"):
    '''
    bind_rows(x)
    
    Concatenates tidy dataframes along rows (2nd one below the 1st and so on)
    
    Parameters
    ----------
    x : list or dict of tidy dataframes
        Input dataframes may be mix of grouped and ungrouped dataframes. 
    rowid_column_name : string, optional
        Name of the identitifer column when the input is a dict. The default is "rowid".

    Notes
    ------
    1. Input might contain grouped dataframes too. But grouping is ignored in the row binding process and the output is always a tidy ungrouped dataframe.
    2. The column names of the result is the union of all column names of input. The missing values are replaced by NaN or appropriate missing type.

    Returns
    -------
    tidy ungrouped dataframe
    
    Examples
    --------
    from nycflights13 import flights
    flights_tidy = TidyDataFrame(flights).arrange('dep_time')
    flights_tidy
    
    flights_1 = flights_tidy.slice(range(3))
    flights_2 = flights_tidy.slice(range(3, 6))
    
    bind_rows([flights_1, flights_2])
    # rows binding happens by aligning columns correctly irrespective of their placement
    bind_rows([flights_1, flights_2.relocate('month')])
    
    # missing columns are filled by NaN
    bind_rows([flights_1.select(['year', 'month', 'day']), flights_2])
    
    # output is never grouped irrespective of the input
    bind_rows([flights_1.select(['year', 'month', 'day']), flights_2.group_by('hour')])
    '''
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
        pdfs = map(lambda y: y.ungroup().to_pandas(), x)
        
    res = pd.concat(pdfs, axis = 'index', ignore_index = True)
    if isinstance(x, dict):
        res[rowid_column_name] = pd.Categorical(res[rowid_column_name])        
    res = TidyDataFrame(res, check = False)
    if isinstance(x, dict):
        res = res.relocate(rowid_column_name)
    return res

def bind_cols(x):
    '''
    bind_cols(x)
    
    Concatenates tidy dataframes along columns (2nd one to the right of 1st and so on)
    
    Parameters
    ----------
    x : list of tidy dataframes
        Input dataframes may be mix of grouped and ungrouped dataframes. 

    Notes
    ------
    1. Input might contain grouped dataframes too. But grouping is ignored in the column binding process and the output is always a tidy ungrouped dataframe.
    2. Each input tidy dataframe is expected to have same number of rows.
    3. If some column names are duplicated, they will be renamed to make the resulting column names unique.

    Returns
    -------
    tidy ungrouped dataframe
    
    Examples
    --------
    from nycflights13 import flights
    flights_tidy = TidyDataFrame(flights).arrange('dep_time')
    flights_tidy
    
    flights_1 = flights_tidy.slice(range(3))
    flights_2 = flights_tidy.slice(range(3, 6))
    
    bind_rows([flights_1, flights_2])
    # rows binding happens by aligning columns correctly irrespective of their placement
    bind_rows([flights_1, flights_2.relocate('month')])
    
    # missing columns are filled by NaN
    bind_rows([flights_1.select(['year', 'month', 'day']), flights_2])
    
    # output is never grouped irrespective of the input
    bind_rows([flights_1.select(['year', 'month', 'day']), flights_2.group_by('hour')])

    '''
    pdfs = list(map(lambda y: y.ungroup().to_pandas(), x))
    rls  = set(map(lambda y: y.shape[0], pdfs))
    if len(rls) > 1:
        raise Exception("Cannot bind columns as input tidy dataframes do not have same number of rows")
    res = pd.concat(pdfs, axis = "columns", ignore_index = False)
    res.columns = get_unique_names(list(res.columns))
    return TidyDataFrame(res, check = False)
  
bind_columns = bind_cols