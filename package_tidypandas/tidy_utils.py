
def bind_rows(x, rowid_column_name = "rowid"):
    '''
    bind_rows(x)
    
    Concatenates tidy dataframes along rows (2nd one below the 1st and so on)
    
    Parameters
    ----------
    x : list or dict of tidy dataframes
        Input dataframes may be grouped too, but the output is not grouped. 
    rowid_column_name : string, optional
        Name of the identitifer column when the input is a dict. The default is "rowid".

    Notes
    ------
    1. Input might contain grouped dataframes too. But grouping is ignored in the row binding process and the output is always a tidy ungrouped dataframe.
    2. The column names of the result is the union of all column names of input. The missing values are replaced by NaN or appropriate missing type.

    Returns
    -------
    tidy ungrouped dataframe

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
        pdfs = map(lambda y: y.to_pandas(), x)
        
    res = pd.concat(pdfs, axis = 'index', ignore_index = True)
    if isinstance(x, dict):
        res[rowid_column_name] = pd.Categorical(res[rowid_column_name])        
    res = TidyDataFrame(res, check = False)
    if isinstance(x, dict):
        res = res.relocate(rowid_column_name)
    return res

def bind_cols(x):
    
    pdfs = list(map(lambda y: y.ungroup().to_pandas(), x))
    rls  = set(map(lambda y: y.shape[0], pdfs))
    if len(rls) > 1:
        raise Exception("Cannot bind columns as input tidy dataframes have different number of rows")
    res = pd.concat(pdfs, axis = "columns", ignore_index = False)
    res.columns = get_unique_names(list(res.columns))
    return TidyDataFrame(res, check = False)
  
