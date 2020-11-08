# function to convert to the appropriate tidyframe type
def tidy(x):
  
  assert is_tidy_frame(x)
  
  df_type = x.__class__.__name__ # ["DataFrame", "DataFrameGroupBy"]
  if df_type == "DataFrame":
    res = tidyDataFrame(x)
    print("Successfully created a tidyDataFrame object")
  else:
    res = tidyGroupedDataFrame(x)
    print("Successfully created a tidyGroupedDataFrame object:")
  
  return res

#--------------------------------------------- 
def is_pandas_frame(df):
    
    res = df.__class__.__name__ in ["DataFrame", "DataFrameGroupBy"]
    
    return(res)

#---------------------------------------------
def is_tidy_frame(df):
    
    assert is_pandas_frame(df)
    
    # groupby object specific checks
    res = True
    if is_grouped_frame(df, check = False):
        groupvars = group_vars(df, check = False)
        if len(groupvars) == 0:
            res = False
        if not isinstance(df.grouper.axis, (pd.RangeIndex, pd.Int64Index)):
            res = False
        df = df.obj
    # all groupby object checks are done
    # proceed with DataFrame checks if necessary
    if res:
        # check if row and column indexes are not multiindexes
        row_flag = not isinstance(df.index, pd.MultiIndex)
        columns  = list(df.columns)
        col_flag = not isinstance(columns, pd.MultiIndex)
        # check if row index is rangeIndex
        flag_no_index = isinstance(df.index, (pd.RangeIndex, pd.Int64Index))
        # check if all column names are strings
        str_flag = all(map(lambda x:isinstance(x, str), columns))
        # check if column names are unique
        if len(set(columns)) == len(columns):
            unique_flag = True
        else:
            unique_flag = False
        
        res = (row_flag
              and col_flag 
              and flag_no_index 
              and str_flag 
              and unique_flag)
    
    return(res)

#---------------------------------------------  
def is_grouped_frame(df, check = True):
    
    if check:
        assert is_pandas_frame(df)
    
    res = df.__class__.__name__ == "DataFrameGroupBy"
    return(res)

#---------------------------------------------
def group_vars(df, check = True):
    
    if check:
        assert is_pandas_frame(df)
    
    if is_grouped_frame(df):
        res = df.grouper.names
    else:
        res = []
    
    return res
