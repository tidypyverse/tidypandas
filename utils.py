#---------------------------------------------
def is_pandas_frame(df):
    
    res = df.__class__.__name__ in ["DataFrame", "DataFrameGroupBy"]
    
    return(res)

#---------------------------------------------
def is_tidy_frame(df):
    
    if is_grouped_frame(df):
        res = True
    else:
        # check if row and column indexes are not multiindexes
        row_flag = not isinstance(df.index, pd.MultiIndex)
        columns  = list(df.columns)
        col_flag = not isinstance(columns, pd.MultiIndex)
        
        # check if row index is rangeIndex
        flag_no_index = isinstance(df.index, (pd.RangeIndex, pd.Int64Index))
        
        # check if all column names are strings
        str_or_not = list(set(map(lambda x:isinstance(x, str), columns)))
        if len(str_or_not) == 1 and str_or_not[0] == True:
            str_flag = True
        else:
            str_flag = False
        
        # check if column names are unique
        if len(set(columns)) == len(columns):
            unique_flag = True
        else:
            unique_flag = False
        
        res = row_flag and col_flag and flag_no_index and str_flag and unique_flag
    
    return(res)

#---------------------------------------------  
def is_grouped_frame(df):
  res = df.__class__.__name__ == "DataFrameGroupBy"
  return(res)

#---------------------------------------------
def ungroup(df, check = True):
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df):
        res = df.obj
    else:
        res = df
    return res

#---------------------------------------------
def count(df, column_names_to_group_by = [], count_column_name = "n", check = True):
  
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
      
    if isinstance(column_names_to_group_by, str):
        column_names_to_group_by = [column_names_to_group_by]
        
    grouped_flag = is_grouped_frame(df)
  
    if grouped_flag:
        group_vars   = df.grouper.names
        df_ungrouped = df.obj
    else:
        group_vars   = []
        df_ungrouped = df
         
    new_group_vars = list(set(group_vars + column_names_to_group_by))
  
    if count_column_name in new_group_vars:
        raise Exception("count column name should be different from groupby variable names")
  
    # count by looking at grouped sizes
    # TODO: this should be rewritten with tidy verbs
    res = (df_ungrouped.groupby(new_group_vars)
                       .size()
                       .reset_index()
                       .rename(columns = {0: count_column_name}))
  
    if grouped_flag:
        res = res.groupby(group_vars)
  
    return res

#---------------------------------------------
def group_vars(df, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df):
        res = df.grouper.names
    else:
        res = []
    
    return res

#---------------------------------------------
def group_by(df, column_names, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    assert not is_grouped_frame(df)
    
    res = df.groupby(by = column_names
                     , sort = False
                     , observed = True
                     )
    
    return res

#---------------------------------------------
def select(df, column_names, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if isinstance(column_names, str):
        column_names = [column_names]
    
    if is_grouped_frame(df):
        groupvars    = group_vars(df)
        column_names = column_names + groupvars
        column_names = list(set(column_names))
        
    
        res = (ungroup(df, check = False).loc[:, column_names]
                          .pipe(group_by, groupvars, check = False))
    else:
        res = df.loc[:, column_names]
        
    return(res)

#---------------------------------------------
def summarize(df, name_func_dict, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    summarized_output = {key: None for key in name_func_dict.keys()}
    if is_grouped_frame(df):
        for key in name_func_dict:
            summarized_output[key] = df.apply(name_func_dict[key])
        res = pd.concat(summarized_output, axis = 1, join = "outer")
        res.columns = name_func_dict.keys()
        res = res.reset_index()
    else:
        for key in name_func_dict:
            summarized_output[key] = df.pipe(name_func_dict[key])
        res = pd.DataFrame(summarized_output, index = [0])
    
    return(res)

#---------------------------------------------
def mutate(df, func_list, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df):
        groupvars = group_vars(df)
        for func in func_list:
            df = df.apply(func)
        df = df.reset_index(drop = True).pipe(group_by, groupvars)
    else:
        for func in func_list:
            df = df.pipe(func)
    
    return(df)

#---------------------------------------------
def arrange(df, check = True, **kwargs):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df):
        groupvars = group_vars(df, check = False)
        res = (df.apply(lambda x: x.sort_values(**kwargs))
                 .reset_index(drop = True)
                 .groupby(groupvars)
                 )
        
    else:
        res = df.sort_values(**kwargs)
    
    return res

#---------------------------------------------
def filter(df, query = None, mask = None, check = True):
   
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if query is None and mask is None:
        raise Exception("Both 'query' and 'mask' cannot be None")
    if query is not None and mask is not None:
        raise Exception("One among 'query' and 'mask' should be None")
    if query is not None and mask is None:
        if is_grouped_frame(df):
            groupvars = group_vars(df, check = False)
            res = (df.obj
                     .query(query)
                     .groupby(groupvars)
                     )
        else:
            res = df.query(query)
    if query is None and mask is not None:
        if is_grouped_frame(df):
            raise Exception("'mask' does not work with grouped dataframe")
        else:
            res = df.query(query)
        res = df.iloc[mask, :]
        
    return res
