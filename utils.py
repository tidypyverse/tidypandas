# dplyr count like utility: returns an ungrouped dataframe
def dplyr_count(df, count_column_name = "n"):
  
  assert (df.__class__.__name__ == "DataFrameGroupBy"), "input should be a grouped dataframe"
  
  res = df.size().reset_index()
  groupby_vars = df.grouper.names
  if count_column_name in groupby_vars:
    raise Exception("count column name should be different from groupby variable names")
  res.rename(columns = {0: count_column_name}, inplace = True)
    
  return res

# dplyr add_count like utility: returns an ungrouped dataframe   
def dplyr_add_count(df, count_column_name = "n"):
  
  assert (df.__class__.__name__ == "DataFrameGroupBy"), "input should be a grouped dataframe"
  
  count_frame  = dplyr_count(df)
  groupby_vars = df.grouper.names
  column_vars  = df.obj.columns.to_list()
  if count_column_name in column_vars:
    raise Exception("count column name should be different from column names of grouped input dataframe")
  
  res = pd.merge(df.obj, count_frame, on = df.grouper.names)
  return res
  
# dplyr ungroup like utility: returns an ungrouped dataframe
def dplyr_ungroup(df):
  
  df_class = df.__class__.__name__
  if df_class == 'DataFrame':
    res = df
  else:
    assert (df.__class__.__name__ == "DataFrameGroupBy"), "input should be a grouped dataframe"
    res = df.obj
  
  return res
