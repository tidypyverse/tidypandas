#---------------------------------------------
def is_pandas_frame(df):
    
    res = df.__class__.__name__ in ["DataFrame", "DataFrameGroupBy"]
    
    return(res)

#---------------------------------------------
def is_tidy_frame(df):
    
    row_flag = not isinstance(training.index, pd.MultiIndex)
    col_flag = not isinstance(training.columns, pd.MultiIndex)
  
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
        temp_list = []
        temp_list.append(column_names_to_group_by)
        column_names_to_group_by = temp_list
        
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
    res = (df_ungrouped.groupby(new_group_vars)
                       .size()
                       .reset_index()
                       .rename(columns = {0: count_column_name}))
  
    if grouped_flag:
        res = res.groupby(group_vars)
  
    return res
