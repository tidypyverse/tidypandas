#---------------------------------------------
def is_pandas_frame(df):
    
    res = df.__class__.__name__ in ["DataFrame", "DataFrameGroupBy"]
    
    return(res)

#---------------------------------------------
def is_tidy_frame(df):
    
    if is_grouped_frame(df):
        res = True
    else:
        row_flag = not isinstance(df.index, pd.MultiIndex)
        col_flag = not isinstance(df.columns, pd.MultiIndex)
  
        res = row_flag and col_flag
    
    return(res)

#---------------------------------------------  
def is_grouped_frame(df):
  res = df.__class__.__name__ == "DataFrameGroupBy"
  return(res)

#---------------------------------------------
def ungroup(df):
    assert is_pandas_frame(df)
    assert is_tidy_frame(df)
    if is_grouped_frame(df):
        res = df.obj
    else:
        res = df
    return res

#---------------------------------------------
def count(df, column_names_to_group_by = [], count_column_name = "n"):
  
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
def select(df, column_names, check = False):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    if isinstance(column_names, str):
        column_names = [column_names]
    
    if is_grouped_frame(df):
        groupvars    = group_vars(df)
        column_names = column_names + groupvars
        column_names = list(set(column_names))
        
    
        res = (ungroup(df).loc[:, column_names]
                          .pipe(group_by, groupvars))
    else:
        res = df.loc[:, column_names]
        
    return(res)
